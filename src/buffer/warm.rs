use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::GopSegment;
use crate::buffer::EvictedSegment;
use crate::storage::warm_index::DetectionDetail;
use crate::storage::{DetectionStore, EventType, MotionStore, WarmEventEntry, WarmEventIndex};

const NANOS_PER_SEC: u64 = 1_000_000_000;
const NANOS_PER_MS: u64 = 1_000_000;

struct WarmEvent {
    segments: Vec<GopSegment>,
    first_pts: u64,
    last_motion_pts: u64,
    total_bytes: usize,
    has_objects: bool,
    object_classes: Vec<String>,
    filmstrip_frames: Option<Arc<Vec<Vec<u8>>>>,
    backend: Option<String>,
    model: Option<String>,
    detection_details: Vec<DetectionDetail>,
}

impl WarmEvent {
    fn duration_ns(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ns).sum()
    }
}

pub struct WarmWriter {
    receiver: mpsc::Receiver<EvictedSegment>,
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
    movement_retention_ns: u64,
    object_retention_ns: u64,
}

const PRUNE_INTERVAL_SECS: u64 = 3600;

impl WarmWriter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        receiver: mpsc::Receiver<EvictedSegment>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        data_dir: PathBuf,
        camera_id: String,
        pre_padding_secs: u64,
        post_padding_secs: u64,
        warm_index: Option<WarmEventIndex>,
        movement_retention_days: u64,
        object_retention_days: u64,
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
            movement_retention_ns: movement_retention_days * 86400 * NANOS_PER_SEC,
            object_retention_ns: object_retention_days * 86400 * NANOS_PER_SEC,
        }
    }

    pub async fn run(mut self) {
        let mut prune_interval =
            tokio::time::interval(std::time::Duration::from_secs(PRUNE_INTERVAL_SECS));
        prune_interval.tick().await;

        loop {
            tokio::select! {
                evicted = self.receiver.recv() => {
                    match evicted {
                        Some(seg) => self.process_segment(seg),
                        None => break,
                    }
                }
                _ = prune_interval.tick() => {
                    self.run_prune().await;
                }
            }
        }

        if self.current_event.is_some() {
            self.finalize_event().await;
        }
        tracing::debug!(camera = %self.camera_id, "warm writer shutting down");
    }

    async fn run_prune(&self) {
        if let Some(ref index) = self.warm_index {
            index
                .prune(self.movement_retention_ns, self.object_retention_ns)
                .await;
        }
    }

    fn process_segment(&mut self, evicted: EvictedSegment) {
        let has_motion = self
            .motion_store
            .has_motion(&evicted.camera_id, evicted.sequence);
        let segment = evicted.segment;

        let det_info = self
            .detection_store
            .get_detection_info(&evicted.camera_id, evicted.sequence);
        let has_objects = has_motion && !det_info.is_empty();

        if has_motion {
            if let Some(ref mut event) = self.current_event {
                event.last_motion_pts = segment.start_pts;
                event.total_bytes += segment.data.len();
                if has_objects {
                    event.has_objects = true;
                    for info in &det_info {
                        if !event.object_classes.contains(&info.object_class) {
                            event.object_classes.push(info.object_class.clone());
                        }
                        event.detection_details.push(DetectionDetail {
                            class: info.object_class.clone(),
                            confidence: info.confidence,
                        });
                        if event.backend.is_none() {
                            event.backend = Some(info.backend.clone());
                            event.model = Some(info.model.clone());
                        }
                    }
                    if event.filmstrip_frames.is_none() {
                        event.filmstrip_frames = self
                            .detection_store
                            .get_filmstrip(&evicted.camera_id, evicted.sequence);
                    }
                }
                event.segments.push(segment);
            } else {
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

                let filmstrip = if has_objects {
                    self.detection_store
                        .get_filmstrip(&evicted.camera_id, evicted.sequence)
                } else {
                    None
                };

                let (backend, model) = det_info
                    .first()
                    .map(|i| (Some(i.backend.clone()), Some(i.model.clone())))
                    .unwrap_or((None, None));

                let detection_details = det_info
                    .iter()
                    .map(|i| DetectionDetail {
                        class: i.object_class.clone(),
                        confidence: i.confidence,
                    })
                    .collect();

                let object_classes = det_info
                    .iter()
                    .map(|i| i.object_class.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                self.current_event = Some(WarmEvent {
                    segments,
                    first_pts,
                    last_motion_pts: motion_pts,
                    total_bytes,
                    has_objects,
                    object_classes,
                    filmstrip_frames: filmstrip,
                    backend,
                    model,
                    detection_details,
                });
            }
        } else if let Some(ref mut event) = self.current_event {
            let elapsed_since_motion = segment.start_pts.saturating_sub(event.last_motion_pts);
            if elapsed_since_motion <= self.post_padding_ns {
                event.total_bytes += segment.data.len();
                event.segments.push(segment);
            } else {
                let event = self.current_event.take().unwrap();
                let data_dir = self.data_dir.clone();
                let camera_id = self.camera_id.clone();
                let warm_index = self.warm_index.clone();
                tokio::spawn(async move {
                    write_event(&data_dir, &camera_id, event, warm_index.as_ref()).await;
                });
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
        if let Some(event) = self.current_event.take() {
            write_event(
                &self.data_dir,
                &self.camera_id,
                event,
                self.warm_index.as_ref(),
            )
            .await;
        }
    }
}

async fn write_event(
    data_dir: &std::path::Path,
    camera_id: &str,
    event: WarmEvent,
    warm_index: Option<&WarmEventIndex>,
) {
    let duration_ns = event.duration_ns();
    let duration_ms = duration_ns / NANOS_PER_MS;
    let segment_count = event.segments.len();
    let total_bytes = event.total_bytes;

    let subdir = if event.has_objects {
        "objects"
    } else {
        "movements"
    };
    let camera_dir = data_dir.join(camera_id).join(subdir);
    if let Err(e) = tokio::fs::create_dir_all(&camera_dir).await {
        tracing::error!(
            camera = %camera_id,
            error = %e,
            "failed to create warm storage directory"
        );
        return;
    }

    let stem = format!("{}_{}", event.first_pts, duration_ms);
    let file_path = camera_dir.join(format!("{}.ts", stem));

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

            // Write sidecar JSON with detection metadata
            if event.has_objects {
                let mut meta = serde_json::Map::new();
                if let Some(ref backend) = event.backend {
                    meta.insert("backend".to_string(), serde_json::json!(backend));
                }
                if let Some(ref model) = event.model {
                    meta.insert("model".to_string(), serde_json::json!(model));
                }

                // Deduplicate detections by class, keeping highest confidence
                let mut best: std::collections::HashMap<String, f32> =
                    std::collections::HashMap::new();
                for d in &event.detection_details {
                    let entry = best.entry(d.class.clone()).or_insert(0.0);
                    if d.confidence > *entry {
                        *entry = d.confidence;
                    }
                }
                let detections: Vec<serde_json::Value> = best
                    .iter()
                    .map(|(class, confidence)| {
                        serde_json::json!({"class": class, "confidence": confidence})
                    })
                    .collect();
                meta.insert("detections".to_string(), serde_json::json!(detections));

                let meta_path = file_path.with_extension("json");
                if let Err(e) =
                    tokio::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).await
                {
                    tracing::warn!(error = %e, "failed to write event metadata");
                }
            }

            // Write filmstrip thumbnails
            let has_filmstrip = if let Some(ref frames) = event.filmstrip_frames {
                let mut wrote = false;
                for (i, jpeg) in frames.iter().enumerate() {
                    let thumb_path = camera_dir.join(format!("{}_thumb_{}.jpg", stem, i));
                    if let Err(e) = tokio::fs::write(&thumb_path, jpeg).await {
                        tracing::warn!(error = %e, "failed to write filmstrip thumbnail");
                    } else {
                        wrote = true;
                    }
                }
                wrote
            } else {
                false
            };

            if let Some(index) = warm_index {
                index.insert(
                    camera_id,
                    WarmEventEntry {
                        start_pts_ns: event.first_pts,
                        duration_ms: duration_ms as u32,
                        event_type: if event.has_objects {
                            EventType::Object
                        } else {
                            EventType::Movement
                        },
                        file_size,
                        object_classes: event.object_classes,
                        backend: event.backend,
                        model: event.model,
                        detections: event.detection_details,
                        has_filmstrip,
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
