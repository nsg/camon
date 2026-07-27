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

/// Persists finished events to warm storage.
///
/// Receives complete events (and post-hoc upgrade requests from the
/// detection worker) over a bounded channel and handles each one inline — no
/// detached spawns — so awaiting the writer task at shutdown guarantees every
/// accepted event reached disk. The writer owns ALL file mutations under its
/// camera's warm-storage directory.
///
/// Retention is NOT its business: pruning sweeps the whole store, not one
/// camera, so it belongs to the single [`RetentionTask`]. What stays here is
/// the write-triggered part of space management — a write that finds the disk
/// full has to be able to make room for itself.
pub struct WarmWriter {
    receiver: mpsc::Receiver<WriterMessage>,
    camera_id: String,
    backend: Arc<dyn WarmStorageBackend>,
    /// Low-space guard threshold: before each event write, if the storage
    /// filesystem has less free space than this, the oldest events are
    /// emergency-pruned first. 0 disables the guard.
    min_free_bytes: u64,
}

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
            min_free_bytes: warm_config.min_free_bytes,
        }
    }

    pub async fn run(mut self) {
        // recv() drains buffered events after all senders drop, so the queue
        // is fully written out before the task exits at shutdown.
        while let Some(message) = self.receiver.recv().await {
            match message {
                WriterMessage::Event(event) => self.handle_event(event).await,
                WriterMessage::Upgrade(upgrade) => self.handle_upgrade(upgrade).await,
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
}

/// How much footage ages out between two scheduled sweeps.
const PRUNE_INTERVAL: Duration = Duration::from_secs(3600);

/// How often the retention task wakes to compare the clock against its next
/// deadline. Cheap enough not to matter, fine enough that a shutdown between
/// sweeps is not noticeable.
const RETENTION_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// The single owner of scheduled retention for the whole warm store.
///
/// A prune sweeps every camera, so exactly one of these runs per process. One
/// per writer — which is what used to happen — meant N concurrent whole-store
/// sweeps racing on the same snapshot/delete/unindex sequence.
///
/// The shutdown flag is handed to the backend rather than only checked around
/// the sweep: a sweep is long (sequential deletes, each able to sit on a remote
/// request timeout) and shutdown joins this task, so the sweep itself stops
/// early, between events. Cancelling it from out here instead — dropping the
/// future mid-delete — could strip an event's `.ts` and leave its sidecar and
/// thumbnails behind, invisible to a scan that only looks at `.ts` files.
pub struct RetentionTask {
    backend: Arc<dyn WarmStorageBackend>,
    movement_retention_ns: u64,
    object_retention_ns: u64,
    continuous_retention_ns: u64,
    shutdown: Arc<AtomicBool>,
}

impl RetentionTask {
    pub fn new(
        backend: Arc<dyn WarmStorageBackend>,
        warm_config: &WarmConfig,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        // Config validation bounds the days well below the wrap, but a wrapped
        // retention is a *short* one — it deletes footage — so saturate rather
        // than depend on that bound staying correct.
        let retention_ns = |days: u64| days.saturating_mul(86400).saturating_mul(NANOS_PER_SEC);
        Self {
            backend,
            movement_retention_ns: retention_ns(warm_config.movement_retention_days),
            object_retention_ns: retention_ns(warm_config.object_retention_days),
            continuous_retention_ns: retention_ns(warm_config.continuous_retention_days),
            shutdown,
        }
    }

    /// The first sweep is one full interval in, as it was when the writers
    /// owned the tick: startup already scanned the store, and an operator
    /// restarting camon is rarely asking for deletions.
    ///
    /// Deadlines advance by whole intervals from the previous deadline, so the
    /// cadence is fixed-rate: a sweep that takes 20 minutes does not push the
    /// next one out to 80 minutes, and a backend timing out repeatedly cannot
    /// drift retention arbitrarily late. Intervals that elapsed during a long
    /// sweep are skipped rather than fired back to back — the deletions they
    /// would have made are still due, and the next sweep makes them.
    pub async fn run(self) {
        let mut interval = tokio::time::interval(RETENTION_POLL_INTERVAL);
        // The poll exists to notice a deadline, not to catch up on missed
        // wakeups: replaying them after a long sweep would achieve nothing.
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut next_prune = tokio::time::Instant::now() + PRUNE_INTERVAL;
        tracing::debug!("retention task started");

        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;
            if tokio::time::Instant::now() >= next_prune {
                self.backend
                    .prune(
                        self.movement_retention_ns,
                        self.object_retention_ns,
                        self.continuous_retention_ns,
                        &self.shutdown,
                    )
                    .await;
                let now = tokio::time::Instant::now();
                while next_prune <= now {
                    next_prune += PRUNE_INTERVAL;
                }
            }
        }

        tracing::debug!("retention task stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::backend::deduplicate_detections;
    use crate::storage::{DetectionEntry, WarmEventEntry};

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

    // ---- Who prunes: the retention task, never a writer ----

    use std::sync::atomic::AtomicUsize;

    /// Records what the writer and the retention task ask of storage. Only the
    /// writer path is implemented; the read path is never reached from here.
    #[derive(Default)]
    struct RecordingBackend {
        writes: AtomicUsize,
        /// Sweeps started, counted on entry to `prune`.
        prunes: AtomicUsize,
        /// Sweeps that ran to completion.
        prunes_finished: AtomicUsize,
        /// Sweeps that saw the shutdown flag and returned early.
        prunes_cancelled: AtomicUsize,
        guards: AtomicUsize,
        emergency_prunes: AtomicUsize,
        /// When set, the next write reports a full disk and clears the flag.
        no_space_next_write: AtomicBool,
        /// How long a sweep takes, for cadence tests.
        prune_duration: Duration,
        /// When set, a sweep parks here until notified — a sweep that is in
        /// flight and cannot be hurried, which is what a remote backend
        /// sitting on request timeouts looks like from out here.
        prune_gate: Option<Arc<tokio::sync::Notify>>,
    }

    #[async_trait::async_trait]
    impl WarmStorageBackend for RecordingBackend {
        async fn write_event(&self, _camera_id: &str, _event: &FinishedEvent) -> WriteOutcome {
            self.writes.fetch_add(1, Ordering::Relaxed);
            if self.no_space_next_write.swap(false, Ordering::Relaxed) {
                WriteOutcome::NoSpace
            } else {
                WriteOutcome::Written
            }
        }

        async fn upgrade_event(&self, _camera_id: &str, _upgrade: &EventUpgrade) {}

        async fn prune(
            &self,
            _movement_ns: u64,
            _object_ns: u64,
            _continuous_ns: u64,
            cancel: &AtomicBool,
        ) {
            self.prunes.fetch_add(1, Ordering::Relaxed);
            if let Some(gate) = &self.prune_gate {
                gate.notified().await;
            }
            // A real sweep is a loop of per-event deletes that checks `cancel`
            // between them; one delete is the granularity, so model it as one
            // sleep and one check.
            tokio::time::sleep(self.prune_duration).await;
            if cancel.load(Ordering::Relaxed) {
                self.prunes_cancelled.fetch_add(1, Ordering::Relaxed);
                return;
            }
            self.prunes_finished.fetch_add(1, Ordering::Relaxed);
        }

        async fn guard_free_space(&self, _camera_id: &str, _min_free_bytes: u64) {
            self.guards.fetch_add(1, Ordering::Relaxed);
        }

        async fn emergency_prune(&self, _camera_id: &str, _min_free_bytes: u64) {
            self.emergency_prunes.fetch_add(1, Ordering::Relaxed);
        }

        fn free_space(&self) -> std::io::Result<u64> {
            Ok(u64::MAX)
        }

        async fn scan(&self) {}

        fn recover_orphans(&self) {}

        fn query(&self, _camera_id: &str, _from_ns: u64, _to_ns: u64) -> Vec<WarmEventEntry> {
            unimplemented!("read path unused in writer tests")
        }

        fn find_event(&self, _camera_id: &str, _start_pts_ns: u64) -> Option<WarmEventEntry> {
            unimplemented!("read path unused in writer tests")
        }

        async fn read_video(
            &self,
            _camera_id: &str,
            _entry: &WarmEventEntry,
            _range: Option<crate::storage::backend::RangeRequest>,
        ) -> std::io::Result<crate::storage::backend::VideoStream> {
            unimplemented!("read path unused in writer tests")
        }

        async fn read_thumbnail(
            &self,
            _camera_id: &str,
            _entry: &WarmEventEntry,
        ) -> Result<Vec<u8>, crate::storage::backend::ThumbnailError> {
            unimplemented!("read path unused in writer tests")
        }

        async fn read_filmstrip(
            &self,
            _camera_id: &str,
            _entry: &WarmEventEntry,
            _index: u8,
        ) -> std::io::Result<Vec<u8>> {
            unimplemented!("read path unused in writer tests")
        }
    }

    fn test_event() -> FinishedEvent {
        FinishedEvent {
            segments: vec![segment(0, SEC, 1)],
            first_pts: 0,
            total_bytes: 4,
            has_objects: false,
            object_classes: Vec::new(),
            filmstrip_frames: None,
            backend: None,
            model: None,
            detection_details: Vec::new(),
            continues: false,
            is_continuous: false,
        }
    }

    /// That the writer no longer runs the scheduled prune is pinned by the
    /// type system, not by a test: it holds no retention config at all any
    /// more, so there is nothing to prune *with*. A timing test could only
    /// wait out the one-hour tick, and one that closes the channel first
    /// passes against the old code too. What is worth pinning is the half of
    /// space management that stayed: the pre-write guard.
    #[tokio::test]
    async fn writer_guards_free_space_before_writing() {
        let backend = Arc::new(RecordingBackend::default());
        let (tx, rx) = mpsc::channel(4);
        let writer = WarmWriter::new(
            rx,
            "cam".to_string(),
            &WarmConfig::default(),
            backend.clone(),
        );
        let handle = tokio::spawn(writer.run());

        tx.send(WriterMessage::Event(test_event())).await.unwrap();
        drop(tx);
        handle.await.unwrap();

        assert_eq!(backend.writes.load(Ordering::Relaxed), 1);
        assert_eq!(backend.guards.load(Ordering::Relaxed), 1);
    }

    /// The other half of space management stays with the writer: a write that
    /// finds the disk full makes room itself instead of waiting for the sweep.
    #[tokio::test]
    async fn write_that_hits_a_full_disk_emergency_prunes_and_retries() {
        let backend = Arc::new(RecordingBackend::default());
        backend.no_space_next_write.store(true, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel(4);
        let writer = WarmWriter::new(
            rx,
            "cam".to_string(),
            &WarmConfig::default(),
            backend.clone(),
        );
        let handle = tokio::spawn(writer.run());

        tx.send(WriterMessage::Event(test_event())).await.unwrap();
        drop(tx);
        handle.await.unwrap();

        assert_eq!(backend.emergency_prunes.load(Ordering::Relaxed), 1);
        assert_eq!(backend.writes.load(Ordering::Relaxed), 2, "no retry");
        assert_eq!(backend.prunes.load(Ordering::Relaxed), 0);
    }

    fn spawn_retention(
        backend: Arc<RecordingBackend>,
        shutdown: &Arc<AtomicBool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(
            RetentionTask::new(backend, &WarmConfig::default(), Arc::clone(shutdown)).run(),
        )
    }

    /// The clock is paused, so these run on the real hourly constant in
    /// virtual time: exact, instant, and no sleeps to be flaky about.
    #[tokio::test(start_paused = true)]
    async fn retention_sweeps_at_a_fixed_rate_whatever_a_sweep_costs() {
        // A sweep taking half the interval must not push the next one out.
        let backend = Arc::new(RecordingBackend {
            prune_duration: PRUNE_INTERVAL / 2,
            ..Default::default()
        });
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = spawn_retention(backend.clone(), &shutdown);

        tokio::time::sleep(PRUNE_INTERVAL / 2).await;
        assert_eq!(
            backend.prunes.load(Ordering::Relaxed),
            0,
            "swept before the first interval elapsed"
        );

        // Now at 3.25 intervals. Fixed-rate starts sweeps at 1x, 2x and 3x
        // (the third still running). Fixed-delay — deadline measured from the
        // end of the previous sweep — would have managed two, at 1x and 2.5x.
        tokio::time::sleep(PRUNE_INTERVAL * 11 / 4).await;
        assert_eq!(
            backend.prunes.load(Ordering::Relaxed),
            3,
            "cadence drifted with the sweep duration"
        );
        assert_eq!(backend.prunes_finished.load(Ordering::Relaxed), 2);

        shutdown.store(true, Ordering::Relaxed);
        tokio::time::timeout(PRUNE_INTERVAL, handle)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn retention_task_stops_between_sweeps_without_waiting_out_the_interval() {
        let backend = Arc::new(RecordingBackend::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = spawn_retention(backend.clone(), &shutdown);

        tokio::time::sleep(PRUNE_INTERVAL / 4).await;
        shutdown.store(true, Ordering::Relaxed);
        tokio::time::timeout(RETENTION_POLL_INTERVAL * 3, handle)
            .await
            .expect("retention task sat on the hourly deadline instead of leaving")
            .unwrap();
        assert_eq!(backend.prunes.load(Ordering::Relaxed), 0);
    }

    /// Shutdown during a sweep is the case that decides join-vs-abort. The
    /// sweep gets the flag and stops itself between events; the task then ends
    /// normally, which is what lets `graceful_shutdown` join it. Cancelling it
    /// from outside would land mid-delete and orphan sidecars and thumbnails
    /// where the `.ts`-only startup scan can never see them again.
    #[tokio::test(start_paused = true)]
    async fn retention_task_stops_during_a_sweep_without_being_cancelled() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let backend = Arc::new(RecordingBackend {
            prune_gate: Some(Arc::clone(&gate)),
            ..Default::default()
        });
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = spawn_retention(backend.clone(), &shutdown);

        // Park the task inside a sweep, the way a slow backend would.
        tokio::time::sleep(PRUNE_INTERVAL + RETENTION_POLL_INTERVAL).await;
        assert_eq!(backend.prunes.load(Ordering::Relaxed), 1);
        assert_eq!(backend.prunes_finished.load(Ordering::Relaxed), 0);

        shutdown.store(true, Ordering::Relaxed);
        gate.notify_one(); // the delete in flight completes

        tokio::time::timeout(RETENTION_POLL_INTERVAL * 3, handle)
            .await
            .expect("a sweep in flight held shutdown up")
            .expect("the task was cancelled instead of stopping itself");
        assert_eq!(
            backend.prunes_cancelled.load(Ordering::Relaxed),
            1,
            "the sweep ignored the shutdown flag and ran to completion"
        );
        assert_eq!(backend.prunes_finished.load(Ordering::Relaxed), 0);
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
