use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::GopSegment;
use crate::buffer::HotBuffer;
use crate::config::WarmConfig;
use crate::storage::warm_index::DetectionDetail;
use crate::storage::{DetectionStore, EventType, WarmEventEntry, WarmEventIndex};

const NANOS_PER_MS: u64 = 1_000_000;

/// A complete motion event, assembled from the hot buffer the moment its
/// post-padding elapsed. Segment data is `Arc`-shared with the hot buffer, so
/// holding a finished event does not duplicate video bytes.
pub struct FinishedEvent {
    pub(crate) segments: Vec<GopSegment>,
    pub(crate) first_pts: u64,
    pub(crate) total_bytes: usize,
    pub(crate) has_objects: bool,
    pub(crate) object_classes: Vec<String>,
    pub(crate) filmstrip_frames: Option<Arc<Vec<Vec<u8>>>>,
    pub(crate) backend: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) detection_details: Vec<DetectionDetail>,
}

impl FinishedEvent {
    fn duration_ns(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ns).sum()
    }
}

/// Assemble a finished event from the hot buffer and detection store.
///
/// Called by the analyzer the moment a motion run closes, while every segment
/// in `[first_motion_seq - pre-padding .. last_seq]` is still resident in RAM
/// and the detection metadata for those sequences has not been cleaned up yet.
/// Pre-padding walks backwards from the first motion segment, staying within
/// `pre_padding_ns` and never reaching before `min_start_seq` (the end of the
/// previous event) or the start of the buffer.
///
/// Returns `None` if none of the requested segments are in the buffer any
/// more (only possible for runs longer than the hot buffer itself).
pub fn assemble_event(
    buffer: &HotBuffer,
    detection_store: Option<&DetectionStore>,
    camera_id: &str,
    first_motion_seq: u64,
    last_seq: u64,
    min_start_seq: u64,
    pre_padding_ns: u64,
) -> Option<FinishedEvent> {
    // Walk backwards from the first motion segment to find the pre-padding
    // start, matching the old rolling-window semantics (total pre-padding
    // duration stays <= pre_padding_ns).
    let earliest = min_start_seq.max(buffer.first_sequence());
    let mut start_seq = first_motion_seq.max(earliest);
    let mut pre_duration_ns = 0u64;
    while start_seq > earliest {
        let duration_ns = match buffer.get_segment_by_sequence(start_seq - 1) {
            Some(seg) => seg.duration_ns,
            None => break,
        };
        if pre_duration_ns + duration_ns > pre_padding_ns {
            break;
        }
        pre_duration_ns += duration_ns;
        start_seq -= 1;
    }

    let mut segments = Vec::new();
    for seq in start_seq..=last_seq {
        match buffer.get_segment_by_sequence(seq) {
            Some(seg) => segments.push(seg.clone()),
            None => tracing::warn!(
                camera = %camera_id,
                sequence = seq,
                "event segment already evicted, event will have a gap"
            ),
        }
    }
    let first_pts = segments.first().map(|s| s.start_pts)?;
    let total_bytes = segments.iter().map(|s| s.data.len()).sum();

    // Metadata is read fresh, while the analyzer's store cleanup cannot have
    // pruned these sequences yet (they are still in the hot buffer).
    let mut object_classes: Vec<String> = Vec::new();
    let mut detection_details = Vec::new();
    let mut backend = None;
    let mut model = None;
    let mut filmstrip_frames = None;
    if let Some(store) = detection_store {
        for seq in first_motion_seq..=last_seq {
            for info in store.get_detection_info(camera_id, seq) {
                if !object_classes.contains(&info.object_class) {
                    object_classes.push(info.object_class.clone());
                }
                detection_details.push(DetectionDetail {
                    class: info.object_class,
                    confidence: info.confidence,
                });
                if backend.is_none() {
                    backend = Some(info.backend);
                    model = Some(info.model);
                }
            }
            if filmstrip_frames.is_none() {
                filmstrip_frames = store.get_filmstrip(camera_id, seq);
            }
        }
    }

    Some(FinishedEvent {
        segments,
        first_pts,
        total_bytes,
        has_objects: !detection_details.is_empty(),
        object_classes,
        filmstrip_frames,
        backend,
        model,
        detection_details,
    })
}

/// Persists finished events to warm storage and prunes expired ones.
///
/// Receives complete events from the analyzer over a bounded channel and
/// writes each one inline — no detached spawns — so awaiting the writer task
/// at shutdown guarantees every accepted event reached disk.
pub struct WarmWriter {
    receiver: mpsc::Receiver<FinishedEvent>,
    data_dir: PathBuf,
    camera_id: String,
    warm_index: Option<WarmEventIndex>,
    movement_retention_ns: u64,
    object_retention_ns: u64,
}

const PRUNE_INTERVAL_SECS: u64 = 3600;
const NANOS_PER_SEC: u64 = 1_000_000_000;

impl WarmWriter {
    pub fn new(
        receiver: mpsc::Receiver<FinishedEvent>,
        camera_id: String,
        warm_config: &WarmConfig,
        warm_index: Option<WarmEventIndex>,
    ) -> Self {
        Self {
            receiver,
            data_dir: PathBuf::from(&warm_config.data_dir),
            camera_id,
            warm_index,
            movement_retention_ns: warm_config.movement_retention_days * 86400 * NANOS_PER_SEC,
            object_retention_ns: warm_config.object_retention_days * 86400 * NANOS_PER_SEC,
        }
    }

    pub async fn run(mut self) {
        let mut prune_interval =
            tokio::time::interval(std::time::Duration::from_secs(PRUNE_INTERVAL_SECS));
        prune_interval.tick().await;

        // recv() drains buffered events after all senders drop, so the queue
        // is fully written out before the task exits at shutdown.
        loop {
            tokio::select! {
                event = self.receiver.recv() => {
                    match event {
                        Some(event) => {
                            write_event(&self.data_dir, &self.camera_id, event, self.warm_index.as_ref())
                                .await;
                        }
                        None => break,
                    }
                }
                _ = prune_interval.tick() => {
                    self.run_prune().await;
                }
            }
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
}

fn concatenate_segments(segments: &[GopSegment], capacity: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(capacity);
    for seg in segments {
        data.extend_from_slice(&seg.data);
    }
    data
}

fn build_sidecar_json(event: &FinishedEvent) -> String {
    let mut meta = serde_json::Map::new();
    if let Some(ref backend) = event.backend {
        meta.insert("backend".to_string(), serde_json::json!(backend));
    }
    if let Some(ref model) = event.model {
        meta.insert("model".to_string(), serde_json::json!(model));
    }

    let deduped = deduplicate_detections(&event.detection_details);
    let detections: Vec<serde_json::Value> = deduped
        .iter()
        .map(|(class, confidence)| serde_json::json!({"class": class, "confidence": confidence}))
        .collect();
    meta.insert("detections".to_string(), serde_json::json!(detections));

    serde_json::to_string(&meta).unwrap()
}

fn deduplicate_detections(details: &[DetectionDetail]) -> Vec<(String, f32)> {
    let mut best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for d in details {
        let entry = best.entry(d.class.clone()).or_insert(0.0);
        if d.confidence > *entry {
            *entry = d.confidence;
        }
    }
    best.into_iter().collect()
}

async fn write_filmstrip(camera_dir: &std::path::Path, stem: &str, frames: &[Vec<u8>]) -> bool {
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
}

fn build_index_entry(
    event: &FinishedEvent,
    duration_ms: u64,
    file_size: u64,
    has_filmstrip: bool,
) -> WarmEventEntry {
    WarmEventEntry {
        start_pts_ns: event.first_pts,
        duration_ms: duration_ms as u32,
        event_type: if event.has_objects {
            EventType::Object
        } else {
            EventType::Movement
        },
        file_size,
        object_classes: event.object_classes.clone(),
        backend: event.backend.clone(),
        model: event.model.clone(),
        detections: event.detection_details.clone(),
        has_filmstrip,
    }
}

async fn write_event(
    data_dir: &std::path::Path,
    camera_id: &str,
    event: FinishedEvent,
    warm_index: Option<&WarmEventIndex>,
) {
    let duration_ms = event.duration_ns() / NANOS_PER_MS;
    let segment_count = event.segments.len();

    let subdir = if event.has_objects {
        "objects"
    } else {
        "movements"
    };
    let camera_dir = data_dir.join(camera_id).join(subdir);
    if let Err(e) = tokio::fs::create_dir_all(&camera_dir).await {
        tracing::error!(camera = %camera_id, error = %e, "failed to create warm storage directory");
        return;
    }

    let stem = format!("{}_{}", event.first_pts, duration_ms);
    let file_path = camera_dir.join(format!("{}.ts", stem));
    let data = concatenate_segments(&event.segments, event.total_bytes);
    let file_size = data.len() as u64;

    if let Err(e) = tokio::fs::write(&file_path, &data).await {
        tracing::error!(camera = %camera_id, path = %file_path.display(), error = %e, "failed to write warm event file");
        return;
    }

    tracing::info!(
        camera = %camera_id,
        path = %file_path.display(),
        segments = segment_count,
        bytes = event.total_bytes,
        duration_ms = duration_ms,
        "wrote warm event file"
    );

    if event.has_objects {
        let meta_path = file_path.with_extension("json");
        if let Err(e) = tokio::fs::write(&meta_path, build_sidecar_json(&event)).await {
            tracing::warn!(error = %e, "failed to write event metadata");
        }
    }

    let has_filmstrip = match event.filmstrip_frames {
        Some(ref frames) => write_filmstrip(&camera_dir, &stem, frames).await,
        None => false,
    };

    if let Some(index) = warm_index {
        index.insert(
            camera_id,
            build_index_entry(&event, duration_ms, file_size, has_filmstrip),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::DetectionEntry;

    const SEC: u64 = 1_000_000_000;

    fn segment(start_pts: u64, duration_ns: u64, byte: u8) -> GopSegment {
        GopSegment {
            start_pts,
            duration_ns,
            data: Arc::new(vec![byte; 4]),
            frame_count: 1,
        }
    }

    /// A hot buffer with `count` one-second segments (seq 0..count), where
    /// segment N starts at N seconds and holds bytes [N; 4].
    fn populated_buffer(count: u64) -> std::sync::Arc<std::sync::RwLock<HotBuffer>> {
        use crate::locks::LockExt;
        let buffer = HotBuffer::new("cam".to_string(), 3600);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..count {
                buf.push(segment(seq * SEC, SEC, seq as u8));
            }
        }
        buffer
    }

    #[test]
    fn assembly_includes_pre_padding_within_window() {
        use crate::locks::LockExt;
        let buffer = populated_buffer(10);
        let buf = buffer.read_recover();
        // Motion at seq 5, padding through seq 7, 2s of pre-padding.
        let event = assemble_event(&buf, None, "cam", 5, 7, 0, 2 * SEC).unwrap();
        // Pre-padding reaches back to seq 3 (segments 3 and 4 fill 2s).
        assert_eq!(event.segments.len(), 5);
        assert_eq!(event.first_pts, 3 * SEC);
        assert_eq!(event.segments[0].data[0], 3);
        assert_eq!(event.segments[4].data[0], 7);
        assert_eq!(event.total_bytes, 20);
        assert!(!event.has_objects);
    }

    #[test]
    fn assembly_clamps_pre_padding_to_min_start_seq() {
        use crate::locks::LockExt;
        let buffer = populated_buffer(10);
        let buf = buffer.read_recover();
        // Previous event ended at seq 4 — pre-padding must not reach past 5.
        let event = assemble_event(&buf, None, "cam", 6, 8, 5, 30 * SEC).unwrap();
        assert_eq!(event.first_pts, 5 * SEC);
        assert_eq!(event.segments.len(), 4);
    }

    #[test]
    fn assembly_clamps_pre_padding_to_buffer_start() {
        use crate::locks::LockExt;
        // 5s buffer, 10 segments pushed: seq 0..=4 evicted.
        let buffer = HotBuffer::new("cam".to_string(), 5);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..10u64 {
                buf.push(segment(seq * SEC, SEC, seq as u8));
            }
        }
        let buf = buffer.read_recover();
        assert_eq!(buf.first_sequence(), 5);
        let event = assemble_event(&buf, None, "cam", 7, 9, 0, 30 * SEC).unwrap();
        assert_eq!(event.first_pts, 5 * SEC);
        assert_eq!(event.segments.len(), 5);
    }

    #[test]
    fn assembly_returns_none_when_all_segments_evicted() {
        use crate::locks::LockExt;
        let buffer = HotBuffer::new("cam".to_string(), 5);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..10u64 {
                buf.push(segment(seq * SEC, SEC, seq as u8));
            }
        }
        let buf = buffer.read_recover();
        assert!(assemble_event(&buf, None, "cam", 1, 3, 0, 0).is_none());
    }

    #[test]
    fn assembly_gathers_fresh_detection_metadata() {
        use crate::locks::LockExt;
        let buffer = populated_buffer(10);
        let store = DetectionStore::new(&["cam".to_string()]);
        store.insert(
            "cam",
            DetectionEntry {
                id: store.next_id(),
                segment_sequence: 5,
                object_class: "person".to_string(),
                confidence: 0.9,
                frame_jpeg: Arc::new(vec![1]),
                backend: "ollama".to_string(),
                model: "test-model".to_string(),
            },
        );
        store.insert(
            "cam",
            DetectionEntry {
                id: store.next_id(),
                segment_sequence: 6,
                object_class: "person".to_string(),
                confidence: 0.7,
                frame_jpeg: Arc::new(vec![1]),
                backend: "ollama".to_string(),
                model: "test-model".to_string(),
            },
        );
        store.insert_filmstrip("cam", 5, Arc::new(vec![vec![0xff]]));

        let buf = buffer.read_recover();
        let event = assemble_event(&buf, Some(&store), "cam", 5, 7, 0, 0).unwrap();
        assert!(event.has_objects);
        assert_eq!(event.object_classes, vec!["person".to_string()]);
        assert_eq!(event.backend.as_deref(), Some("ollama"));
        assert_eq!(event.model.as_deref(), Some("test-model"));
        assert_eq!(event.detection_details.len(), 2);
        assert!(event.filmstrip_frames.is_some());
        // Sidecar dedupes to the best confidence per class.
        let deduped = deduplicate_detections(&event.detection_details);
        assert_eq!(deduped, vec![("person".to_string(), 0.9)]);
    }

    #[tokio::test]
    async fn write_event_persists_files_and_indexes_with_stem_key() {
        use crate::locks::LockExt;
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let mut event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC).unwrap()
        };
        event.filmstrip_frames = Some(Arc::new(vec![vec![0xff], vec![0xfe]]));

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let first_pts = event.first_pts;
        write_event(dir.path(), "cam", event, Some(&index)).await;

        // 4 one-second segments (seq 4..=7) => stem "{first_pts}_{4000}".
        let stem = format!("{}_4000", first_pts);
        let movements = dir.path().join("cam").join("movements");
        assert!(movements.join(format!("{}.ts", stem)).exists());
        assert!(movements.join(format!("{}_thumb_0.jpg", stem)).exists());
        assert!(movements.join(format!("{}_thumb_1.jpg", stem)).exists());
        // Movement-only events have no sidecar.
        assert!(!movements.join(format!("{}.json", stem)).exists());

        let entry = index.find_event("cam", first_pts).unwrap();
        assert_eq!(entry.duration_ms, 4000);
        assert_eq!(entry.event_type, EventType::Movement);
        assert_eq!(entry.file_size, 16);
        assert!(entry.has_filmstrip);
        assert_eq!(
            index.resolve_file_path("cam", &entry),
            movements.join(format!("{}.ts", stem))
        );
    }
}
