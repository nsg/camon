use std::collections::VecDeque;
use std::path::PathBuf;

use tokio::sync::mpsc;

use super::GopSegment;
use crate::buffer::EvictedSegment;
use crate::storage::{DetectionStore, EventType, MotionStore, WarmEventEntry, WarmEventIndex};

const NANOS_PER_SEC: u64 = 1_000_000_000;
const NANOS_PER_MS: u64 = 1_000_000;

struct WarmEvent {
    segments: Vec<GopSegment>,
    first_pts: u64,
    last_motion_pts: u64,
    total_bytes: usize,
    has_objects: bool,
}

impl WarmEvent {
    fn duration_ns(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ns).sum()
    }
}

pub struct WarmWriter {
    receiver: mpsc::UnboundedReceiver<EvictedSegment>,
    motion_store: MotionStore,
    detection_store: DetectionStore,
    data_dir: PathBuf,
    camera_id: String,
    pre_padding_ns: u64,
    post_padding_ns: u64,
    pre_buffer: VecDeque<GopSegment>,
    pre_buffer_duration_ns: u64,
    current_event: Option<WarmEvent>,
    warm_index: Option<WarmEventIndex>,
}

impl WarmWriter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        receiver: mpsc::UnboundedReceiver<EvictedSegment>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        data_dir: PathBuf,
        camera_id: String,
        pre_padding_secs: u64,
        post_padding_secs: u64,
        warm_index: Option<WarmEventIndex>,
    ) -> Self {
        Self {
            receiver,
            motion_store,
            detection_store,
            data_dir,
            camera_id,
            pre_padding_ns: pre_padding_secs * NANOS_PER_SEC,
            post_padding_ns: post_padding_secs * NANOS_PER_SEC,
            pre_buffer: VecDeque::new(),
            pre_buffer_duration_ns: 0,
            current_event: None,
            warm_index,
        }
    }

    pub async fn run(mut self) {
        while let Some(evicted) = self.receiver.recv().await {
            self.process_segment(evicted);
        }
        // Channel closed — finalize any pending event
        if self.current_event.is_some() {
            self.finalize_event().await;
        }
        tracing::debug!(camera = %self.camera_id, "warm writer shutting down");
    }

    fn process_segment(&mut self, evicted: EvictedSegment) {
        let has_motion = self
            .motion_store
            .has_motion(&evicted.camera_id, evicted.sequence);
        let segment = evicted.segment;

        let has_objects = has_motion
            && self
                .detection_store
                .has_detections(&evicted.camera_id, evicted.sequence);

        if has_motion {
            if let Some(ref mut event) = self.current_event {
                event.last_motion_pts = segment.start_pts;
                event.total_bytes += segment.data.len();
                if has_objects {
                    event.has_objects = true;
                }
                event.segments.push(segment);
            } else {
                // Start new event — prepend pre-buffer
                let mut segments: Vec<GopSegment> = self.pre_buffer.drain(..).collect();
                self.pre_buffer_duration_ns = 0;
                let first_pts = segments
                    .first()
                    .map(|s| s.start_pts)
                    .unwrap_or(segment.start_pts);
                let total_bytes: usize =
                    segments.iter().map(|s| s.data.len()).sum::<usize>() + segment.data.len();
                let motion_pts = segment.start_pts;
                segments.push(segment);
                self.current_event = Some(WarmEvent {
                    segments,
                    first_pts,
                    last_motion_pts: motion_pts,
                    total_bytes,
                    has_objects,
                });
            }
        } else if let Some(ref mut event) = self.current_event {
            let elapsed_since_motion = segment.start_pts.saturating_sub(event.last_motion_pts);
            if elapsed_since_motion <= self.post_padding_ns {
                event.total_bytes += segment.data.len();
                event.segments.push(segment);
            } else {
                // Post-padding expired — finalize via spawn
                let mut event = self.current_event.take().unwrap();
                let data_dir = self.data_dir.clone();
                let camera_id = self.camera_id.clone();
                let has_objects = event.has_objects;
                let warm_index = self.warm_index.clone();
                tokio::spawn(async move {
                    write_event(
                        &data_dir,
                        &camera_id,
                        &mut event,
                        has_objects,
                        warm_index.as_ref(),
                    )
                    .await;
                });
                // This non-motion segment goes into pre-buffer for next event
                self.push_pre_buffer(segment);
            }
        } else {
            self.push_pre_buffer(segment);
        }
    }

    fn push_pre_buffer(&mut self, segment: GopSegment) {
        self.pre_buffer_duration_ns += segment.duration_ns;
        self.pre_buffer.push_back(segment);
        while self.pre_buffer_duration_ns > self.pre_padding_ns {
            if let Some(old) = self.pre_buffer.pop_front() {
                self.pre_buffer_duration_ns =
                    self.pre_buffer_duration_ns.saturating_sub(old.duration_ns);
            } else {
                break;
            }
        }
    }

    async fn finalize_event(&mut self) {
        if let Some(ref mut event) = self.current_event.take() {
            let has_objects = event.has_objects;
            write_event(
                &self.data_dir,
                &self.camera_id,
                event,
                has_objects,
                self.warm_index.as_ref(),
            )
            .await;
        }
    }
}

async fn write_event(
    data_dir: &std::path::Path,
    camera_id: &str,
    event: &mut WarmEvent,
    has_objects: bool,
    warm_index: Option<&WarmEventIndex>,
) {
    let duration_ns = event.duration_ns();
    let duration_ms = duration_ns / NANOS_PER_MS;
    let segment_count = event.segments.len();
    let total_bytes = event.total_bytes;

    let subdir = if has_objects { "objects" } else { "movements" };
    let camera_dir = data_dir.join(camera_id).join(subdir);
    if let Err(e) = tokio::fs::create_dir_all(&camera_dir).await {
        tracing::error!(
            camera = %camera_id,
            error = %e,
            "failed to create warm storage directory"
        );
        return;
    }

    let filename = format!("{}_{}.ts", event.first_pts, duration_ms);
    let file_path = camera_dir.join(&filename);

    let mut data = Vec::with_capacity(total_bytes);
    for seg in &event.segments {
        data.extend_from_slice(&seg.data);
    }

    let file_size = data.len() as u64;
    match tokio::fs::write(&file_path, &data).await {
        Ok(()) => {
            tracing::info!(
                camera = %camera_id,
                path = %file_path.display(),
                segments = segment_count,
                bytes = total_bytes,
                duration_ms = duration_ms,
                "wrote warm event file"
            );
            if let Some(index) = warm_index {
                index.insert(
                    camera_id,
                    WarmEventEntry {
                        start_pts_ns: event.first_pts,
                        duration_ms: duration_ms as u32,
                        event_type: if has_objects {
                            EventType::Object
                        } else {
                            EventType::Movement
                        },
                        file_size,
                    },
                );
            }
        }
        Err(e) => {
            tracing::error!(
                camera = %camera_id,
                path = %file_path.display(),
                error = %e,
                "failed to write warm event file"
            );
        }
    }
}
