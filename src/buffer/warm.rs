use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::mpsc;

use super::GopSegment;
use crate::buffer::HotBuffer;
use crate::config::WarmConfig;
use crate::locks::LockExt;
use crate::storage::backend::{WarmStorageBackend, WriteOutcome};
use crate::storage::warm_index::DetectionDetail;
use crate::storage::{DetectionStore, EventType};

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
    /// True when this event is a follow-on chunk of a longer motion run that
    /// was split at the duration cap. Written to the sidecar as
    /// `"continues": true` so a UI can stitch the chain back together.
    pub(crate) continues: bool,
    /// True when this is a continuous-recording chunk (analytics disabled), as
    /// opposed to a motion event. Routes the file to the `continuous/` dir.
    pub(crate) is_continuous: bool,
}

impl FinishedEvent {
    pub(crate) fn duration_ns(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ns).sum()
    }

    /// Event duration in milliseconds — one half of the on-disk file stem
    /// (`{start_pts}_{duration_ms}.ts`), so the analyzer can record the
    /// identity of a written event in the event registry.
    pub(crate) fn duration_ms(&self) -> u64 {
        self.duration_ns() / NANOS_PER_MS
    }

    /// Storage classification: object detections win, then continuous
    /// recording, otherwise a plain movement event.
    pub(crate) fn event_type(&self) -> EventType {
        if self.has_objects {
            EventType::Object
        } else if self.is_continuous {
            EventType::Continuous
        } else {
            EventType::Movement
        }
    }
}

/// Everything the warm writer accepts over its channel. The writer owns ALL
/// mutations of warm-storage files — nothing else ever touches them — so both
/// fresh writes and post-hoc upgrades funnel through here, in FIFO order.
pub enum WriterMessage {
    /// Persist a newly finished event.
    Event(FinishedEvent),
    /// Upgrade an already-written movement event to an object event: rewrite
    /// the sidecar with detections, move the files from `movements/` to
    /// `objects/` (which switches the retention class from
    /// `movement_retention_days` to `object_retention_days`), and update the
    /// warm index entry. Sent by the detection worker when an Ollama verdict
    /// lands after its covering event already reached disk.
    Upgrade(EventUpgrade),
}

/// A post-hoc movement→object upgrade for one on-disk event.
pub struct EventUpgrade {
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    pub object_classes: Vec<String>,
    pub detections: Vec<DetectionDetail>,
    pub backend: String,
    pub model: String,
    /// Preserved from the original event so the chain-stitching flag
    /// survives the sidecar rewrite.
    pub continues: bool,
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
/// `filmstrip_frames` are the thumbnails the analyzer extracted for this run;
/// they belong to the run, not to any single sequence, so they are handed in
/// rather than looked up.
///
/// Returns `None` if none of the requested segments are in the buffer any
/// more (only possible for runs longer than the hot buffer itself).
#[allow(clippy::too_many_arguments)]
pub fn assemble_event(
    buffer: &HotBuffer,
    detection_store: Option<&DetectionStore>,
    camera_id: &str,
    first_motion_seq: u64,
    last_seq: u64,
    min_start_seq: u64,
    pre_padding_ns: u64,
    continues: bool,
    filmstrip_frames: Option<Arc<Vec<Vec<u8>>>>,
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
        continues,
        is_continuous: false,
    })
}

/// Assemble a continuous-recording chunk from the hot buffer.
///
/// Reuses [`assemble_event`] with no detection store and no pre-padding: a
/// continuous chunk is simply the raw segment range `[start_seq..=last_seq]`,
/// GOP-aligned so each `.ts` decodes on its own. `continues` chains successive
/// chunks (false only for the first chunk after startup).
pub fn assemble_continuous_chunk(
    buffer: &HotBuffer,
    camera_id: &str,
    start_seq: u64,
    last_seq: u64,
    continues: bool,
) -> Option<FinishedEvent> {
    // min_start_seq == start_seq and pre_padding_ns == 0 suppress any reach-back.
    let mut event = assemble_event(
        buffer, None, camera_id, start_seq, last_seq, start_seq, 0, continues, None,
    )?;
    event.is_continuous = true;
    Some(event)
}

/// How often the continuous recorder wakes to check whether a chunk is due.
/// Finer than any sane cap, so chunks land within a second of the cap; the
/// wakeup itself is cheap (a lock, a subtraction).
const CONTINUOUS_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// Decide whether to roll a continuous chunk now, and over which inclusive
/// sequence range.
///
/// Pure so the boundary logic is unit-testable. `next_seq` is the first
/// not-yet-written sequence; `last_sequence_exclusive` is the hot buffer's
/// `last_sequence()` (one past the newest resident segment);
/// `pending_duration_ns` is the summed duration of `[next_seq, last_sequence)`.
/// A chunk rolls once the pending footage reaches `cap_ns`, or immediately when
/// `force` is set (shutdown flush of whatever remains). A zero cap only rolls on
/// `force`. Returns the inclusive `(start, last)` range, or `None`.
fn plan_continuous_roll(
    next_seq: u64,
    last_sequence_exclusive: u64,
    pending_duration_ns: u64,
    cap_ns: u64,
    force: bool,
) -> Option<(u64, u64)> {
    if last_sequence_exclusive <= next_seq {
        return None; // nothing pending
    }
    let should_roll = force || (cap_ns > 0 && pending_duration_ns >= cap_ns);
    should_roll.then_some((next_seq, last_sequence_exclusive - 1))
}

/// Per-camera continuous-recording driver (analytics-disabled "dumb NVR" mode).
///
/// With no analyzer to close motion runs, this task owns the roll loop: it
/// tracks the first not-yet-written sequence and, on each tick, rolls a chunk
/// once `max_event_duration` of footage has accumulated in the hot buffer (or
/// flushes whatever remains at shutdown). Chunks are assembled as `Arc` clones
/// and handed to the same per-camera [`WarmWriter`] over the existing channel;
/// `send().await` never drops a chunk. Successive chunks are flagged
/// `continues` (all but the first after startup) so a UI can stitch the chain.
///
/// Chunk-boundary timing is lifecycle, so tick-based monotonic timing is used;
/// segment content and PTS are untouched.
pub async fn run_continuous_recorder(
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    tx: mpsc::Sender<WriterMessage>,
    max_event_duration: Duration,
    shutdown: Arc<AtomicBool>,
) {
    let cap_ns = max_event_duration.as_nanos() as u64;
    let mut next_seq: u64 = 0;
    let mut first_chunk = true;
    let mut interval = tokio::time::interval(CONTINUOUS_CHECK_INTERVAL);
    tracing::info!(camera = %camera_id, "continuous recorder started");

    loop {
        interval.tick().await;
        let force = shutdown.load(Ordering::Relaxed);

        // Plan + assemble under the read lock; release it before awaiting send.
        let planned = {
            let buf = buffer.read_recover();
            // The cap is always < hot_duration, so pending segments are still
            // resident. If eviction ever outran us, warn and skip the gap.
            if buf.first_sequence() > next_seq {
                tracing::warn!(
                    camera = %camera_id,
                    first_sequence = buf.first_sequence(),
                    next_seq,
                    "continuous recorder fell behind eviction, chunk will have a gap"
                );
                next_seq = buf.first_sequence();
            }
            let pending_ns = buf
                .total_duration_ns()
                .saturating_sub(buf.sequence_to_offset_ns(next_seq).unwrap_or(0));
            plan_continuous_roll(next_seq, buf.last_sequence(), pending_ns, cap_ns, force).and_then(
                |(start, last)| {
                    assemble_continuous_chunk(&buf, &camera_id, start, last, !first_chunk)
                        .map(|ev| (ev, last))
                },
            )
        };

        if let Some((event, last)) = planned {
            if tx.send(WriterMessage::Event(event)).await.is_err() {
                tracing::error!(camera = %camera_id, "warm writer gone, continuous chunk lost");
                return;
            }
            next_seq = last + 1;
            first_chunk = false;
        }

        if force {
            break;
        }
    }

    tracing::info!(camera = %camera_id, "continuous recorder stopped");
}

/// Persists finished events to warm storage and prunes expired ones.
///
/// Receives complete events (and post-hoc upgrade requests from the
/// detection worker) over a bounded channel and handles each one inline — no
/// detached spawns — so awaiting the writer task at shutdown guarantees every
/// accepted event reached disk. The writer owns ALL file mutations under its
/// camera's warm-storage directory.
pub struct WarmWriter {
    receiver: mpsc::Receiver<WriterMessage>,
    camera_id: String,
    backend: Arc<dyn WarmStorageBackend>,
    movement_retention_ns: u64,
    object_retention_ns: u64,
    continuous_retention_ns: u64,
    /// Low-space guard threshold: before each event write, if the storage
    /// filesystem has less free space than this, the oldest events are
    /// emergency-pruned first. 0 disables the guard.
    min_free_bytes: u64,
}

const PRUNE_INTERVAL_SECS: u64 = 3600;
const NANOS_PER_SEC: u64 = 1_000_000_000;

impl WarmWriter {
    pub fn new(
        receiver: mpsc::Receiver<WriterMessage>,
        camera_id: String,
        warm_config: &WarmConfig,
        backend: Arc<dyn WarmStorageBackend>,
    ) -> Self {
        Self {
            receiver,
            camera_id,
            backend,
            movement_retention_ns: warm_config.movement_retention_days * 86400 * NANOS_PER_SEC,
            object_retention_ns: warm_config.object_retention_days * 86400 * NANOS_PER_SEC,
            continuous_retention_ns: warm_config.continuous_retention_days * 86400 * NANOS_PER_SEC,
            min_free_bytes: warm_config.min_free_bytes,
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
                message = self.receiver.recv() => {
                    match message {
                        Some(WriterMessage::Event(event)) => self.handle_event(event).await,
                        Some(WriterMessage::Upgrade(upgrade)) => self.handle_upgrade(upgrade).await,
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

    /// Write one event with the full durability ladder: low-space guard,
    /// atomic write, and — should the disk fill up despite the guard — one
    /// emergency-prune-and-retry. A still-failing write drops the event with
    /// an error log; the writer task itself never crashes or wedges.
    async fn handle_event(&self, event: FinishedEvent) {
        self.backend
            .guard_free_space(&self.camera_id, self.min_free_bytes)
            .await;
        match self.backend.write_event(&self.camera_id, &event).await {
            WriteOutcome::NoSpace => {
                tracing::warn!(
                    camera = %self.camera_id,
                    "disk full while writing event despite guard, emergency pruning and retrying once"
                );
                self.backend
                    .emergency_prune(&self.camera_id, self.min_free_bytes)
                    .await;
                let retry = self.backend.write_event(&self.camera_id, &event).await;
                if retry != WriteOutcome::Written {
                    tracing::error!(
                        camera = %self.camera_id,
                        first_pts = event.first_pts,
                        bytes = event.total_bytes,
                        "dropping event: write failed again after emergency prune"
                    );
                }
            }
            WriteOutcome::Written | WriteOutcome::Failed => {}
        }
    }

    async fn handle_upgrade(&self, upgrade: EventUpgrade) {
        self.backend.upgrade_event(&self.camera_id, &upgrade).await;
    }

    async fn run_prune(&self) {
        self.backend
            .prune(
                self.movement_retention_ns,
                self.object_retention_ns,
                self.continuous_retention_ns,
            )
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::backend::deduplicate_detections;
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
        let event = assemble_event(&buf, None, "cam", 5, 7, 0, 2 * SEC, false, None).unwrap();
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
        let event = assemble_event(&buf, None, "cam", 6, 8, 5, 30 * SEC, false, None).unwrap();
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
        let event = assemble_event(&buf, None, "cam", 7, 9, 0, 30 * SEC, false, None).unwrap();
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
        assert!(assemble_event(&buf, None, "cam", 1, 3, 0, 0, false, None).is_none());
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
        let buf = buffer.read_recover();
        let filmstrip = Arc::new(vec![vec![0xff]]);
        let event = assemble_event(
            &buf,
            Some(&store),
            "cam",
            5,
            7,
            0,
            0,
            false,
            Some(filmstrip),
        )
        .unwrap();
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

    // ---- Continuous recording (analytics disabled) ----

    #[test]
    fn plan_roll_waits_until_cap_reached() {
        // 3s pending, cap 5s: not yet.
        assert_eq!(plan_continuous_roll(0, 3, 3 * SEC, 5 * SEC, false), None);
        // 5s pending, cap 5s: rolls [0..=4].
        assert_eq!(
            plan_continuous_roll(0, 5, 5 * SEC, 5 * SEC, false),
            Some((0, 4))
        );
        // Over the cap rolls everything pending.
        assert_eq!(
            plan_continuous_roll(0, 7, 7 * SEC, 5 * SEC, false),
            Some((0, 6))
        );
    }

    #[test]
    fn plan_roll_resumes_from_next_seq() {
        // Already wrote through seq 4; pending [5..=9] is 5s at cap 5s.
        assert_eq!(
            plan_continuous_roll(5, 10, 5 * SEC, 5 * SEC, false),
            Some((5, 9))
        );
    }

    #[test]
    fn plan_roll_nothing_pending_is_none() {
        assert_eq!(plan_continuous_roll(5, 5, 0, 5 * SEC, false), None);
        // Even forced, an empty range yields nothing to flush.
        assert_eq!(plan_continuous_roll(5, 5, 0, 5 * SEC, true), None);
    }

    #[test]
    fn plan_roll_force_flushes_partial_chunk() {
        // Below the cap, but shutdown forces the remaining 2s out.
        assert_eq!(
            plan_continuous_roll(0, 2, 2 * SEC, 5 * SEC, true),
            Some((0, 1))
        );
    }

    #[test]
    fn plan_roll_zero_cap_only_rolls_on_force() {
        assert_eq!(plan_continuous_roll(0, 4, 4 * SEC, 0, false), None);
        assert_eq!(plan_continuous_roll(0, 4, 4 * SEC, 0, true), Some((0, 3)));
    }

    #[test]
    fn continuous_chunk_has_no_detections_and_no_pre_padding() {
        use crate::locks::LockExt;
        let buffer = populated_buffer(10);
        let buf = buffer.read_recover();
        // Roll [2..=6] with no detection store at all.
        let event = assemble_continuous_chunk(&buf, "cam", 2, 6, false).unwrap();
        assert!(event.is_continuous);
        assert!(!event.has_objects);
        assert!(!event.continues);
        assert!(event.detection_details.is_empty());
        assert!(event.filmstrip_frames.is_none());
        // No pre-padding: starts exactly at the requested seq.
        assert_eq!(event.first_pts, 2 * SEC);
        assert_eq!(event.segments.len(), 5);
        assert_eq!(event.event_type(), EventType::Continuous);
    }
}
