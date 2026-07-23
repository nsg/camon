use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::mpsc;

use super::GopSegment;
use crate::buffer::HotBuffer;
use crate::config::WarmConfig;
use crate::locks::LockExt;
use crate::storage::warm_index::{free_space_bytes, should_emergency_prune, DetectionDetail};
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
    /// True when this event is a follow-on chunk of a longer motion run that
    /// was split at the duration cap. Written to the sidecar as
    /// `"continues": true` so a UI can stitch the chain back together.
    pub(crate) continues: bool,
    /// True when this is a continuous-recording chunk (analytics disabled), as
    /// opposed to a motion event. Routes the file to the `continuous/` dir.
    pub(crate) is_continuous: bool,
}

impl FinishedEvent {
    fn duration_ns(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ns).sum()
    }

    /// Storage classification: object detections win, then continuous
    /// recording, otherwise a plain movement event.
    fn event_type(&self) -> EventType {
        if self.has_objects {
            EventType::Object
        } else if self.is_continuous {
            EventType::Continuous
        } else {
            EventType::Movement
        }
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
        buffer, None, camera_id, start_seq, last_seq, start_seq, 0, continues,
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
    tx: mpsc::Sender<FinishedEvent>,
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
            if tx.send(event).await.is_err() {
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
                event = self.receiver.recv() => {
                    match event {
                        Some(event) => self.handle_event(event).await,
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
        self.guard_free_space().await;
        match write_event(
            &self.data_dir,
            &self.camera_id,
            &event,
            self.warm_index.as_ref(),
        )
        .await
        {
            WriteOutcome::NoSpace => {
                tracing::warn!(
                    camera = %self.camera_id,
                    "disk full while writing event despite guard, emergency pruning and retrying once"
                );
                self.emergency_prune().await;
                let retry = write_event(
                    &self.data_dir,
                    &self.camera_id,
                    &event,
                    self.warm_index.as_ref(),
                )
                .await;
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

    /// Low-space guard: before an event write, emergency-prune the oldest
    /// events while free space is below `min_free_bytes`.
    async fn guard_free_space(&self) {
        if self.min_free_bytes == 0 || self.warm_index.is_none() {
            return;
        }
        // data_dir may not exist before the first write; statvfs needs it.
        let _ = tokio::fs::create_dir_all(&self.data_dir).await;
        match free_space_bytes(&self.data_dir) {
            Ok(free) if should_emergency_prune(free, self.min_free_bytes) => {
                tracing::warn!(
                    camera = %self.camera_id,
                    free_bytes = free,
                    min_free_bytes = self.min_free_bytes,
                    "storage low on space, emergency-pruning oldest events"
                );
                self.emergency_prune().await;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(camera = %self.camera_id, error = %e, "free-space check failed");
            }
        }
    }

    /// Delete oldest events (continuous → movements → objects) until free
    /// space is back above the threshold or nothing is left to delete.
    async fn emergency_prune(&self) {
        let Some(ref index) = self.warm_index else {
            return;
        };
        let data_dir = self.data_dir.clone();
        let min_free = self.min_free_bytes;
        let deleted = index
            .emergency_prune(move || {
                // Stop as soon as space recovers; a failing statvfs also stops
                // the prune rather than deleting everything blindly.
                free_space_bytes(&data_dir)
                    .map(|free| !should_emergency_prune(free, min_free))
                    .unwrap_or(true)
            })
            .await;
        if deleted == 0 {
            tracing::warn!(
                camera = %self.camera_id,
                "emergency prune freed nothing (no events left to delete)"
            );
        } else {
            tracing::warn!(camera = %self.camera_id, deleted, "emergency prune complete");
        }
    }

    async fn run_prune(&self) {
        if let Some(ref index) = self.warm_index {
            index
                .prune(
                    self.movement_retention_ns,
                    self.object_retention_ns,
                    self.continuous_retention_ns,
                )
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

    // Only follow-on chunks carry `continues`; omit it otherwise so ordinary
    // sidecars stay unchanged.
    if event.continues {
        meta.insert("continues".to_string(), serde_json::json!(true));
    }

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

/// Path of the staging file for an atomic write: `{file_name}.tmp` next to
/// the final path. Startup orphan recovery keys off this exact convention.
fn tmp_path(final_path: &std::path::Path) -> PathBuf {
    let mut name = final_path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    final_path.with_file_name(name)
}

fn is_no_space(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::ENOSPC)
}

/// Write `data` to `path`, fsyncing before returning so the bytes are durable
/// (not just in the page cache) even across a power cut.
async fn write_file_synced(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::File::create(path).await?;
    file.write_all(data).await?;
    file.sync_all().await?;
    Ok(())
}

/// Atomically write a small metadata file (sidecar/thumbnail): stage as
/// `.tmp`, then rename. No fsync — the one fsync per event is spent on the
/// video; a metadata file lost to a power cut is acceptable, a torn one is
/// not (and recovery deletes any leftover `.tmp`).
async fn write_metadata_atomic(final_path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = tmp_path(final_path);
    tokio::fs::write(&tmp, data).await?;
    if let Err(e) = tokio::fs::rename(&tmp, final_path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}

async fn write_filmstrip(camera_dir: &std::path::Path, stem: &str, frames: &[Vec<u8>]) -> bool {
    let mut wrote = false;
    for (i, jpeg) in frames.iter().enumerate() {
        let thumb_path = camera_dir.join(format!("{}_thumb_{}.jpg", stem, i));
        if let Err(e) = write_metadata_atomic(&thumb_path, jpeg).await {
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
        event_type: event.event_type(),
        file_size,
        object_classes: event.object_classes.clone(),
        backend: event.backend.clone(),
        model: event.model.clone(),
        detections: event.detection_details.clone(),
        has_filmstrip,
        continues: event.continues,
        // Live writes are never recovered files; the flag only enters the
        // index via startup orphan recovery + sidecar scan.
        recovered: false,
    }
}

/// Result of a single event write attempt.
#[derive(Debug, PartialEq, Eq)]
enum WriteOutcome {
    Written,
    /// The write failed with ENOSPC — worth an emergency prune and one retry.
    NoSpace,
    /// The write failed for any other reason (already logged).
    Failed,
}

/// Persist one event durably. Write order is deliberate:
///
/// 1. video bytes → `{stem}.ts.tmp`, then fsync — the footage is durable and
///    recoverable (via startup orphan recovery) from this point on, before
///    anything else is risked;
/// 2. sidecar and thumbnails, each atomically under their final names;
/// 3. rename `{stem}.ts.tmp` → `{stem}.ts` — the commit point. The index scan
///    only ever looks at `.ts` files, so a crash at any earlier step leaves a
///    recoverable `.tmp` (plus adoptable metadata), never a half-indexed
///    event; a crash after the rename leaves a complete event.
async fn write_event(
    data_dir: &std::path::Path,
    camera_id: &str,
    event: &FinishedEvent,
    warm_index: Option<&WarmEventIndex>,
) -> WriteOutcome {
    let duration_ms = event.duration_ns() / NANOS_PER_MS;
    let segment_count = event.segments.len();

    let camera_dir = data_dir.join(camera_id).join(event.event_type().dir_name());
    if let Err(e) = tokio::fs::create_dir_all(&camera_dir).await {
        tracing::error!(camera = %camera_id, error = %e, "failed to create warm storage directory");
        return if is_no_space(&e) {
            WriteOutcome::NoSpace
        } else {
            WriteOutcome::Failed
        };
    }

    let stem = format!("{}_{}", event.first_pts, duration_ms);
    let file_path = camera_dir.join(format!("{}.ts", stem));
    let staging_path = tmp_path(&file_path);
    let data = concatenate_segments(&event.segments, event.total_bytes);
    let file_size = data.len() as u64;

    // Step 1: footage first. Once this returns, the video survives a crash.
    if let Err(e) = write_file_synced(&staging_path, &data).await {
        // A partial staging file from a failed write is deleted rather than
        // left for recovery: the disk is under pressure and the writer is
        // about to either retry from scratch or drop the event knowingly.
        let _ = tokio::fs::remove_file(&staging_path).await;
        tracing::error!(camera = %camera_id, path = %staging_path.display(), error = %e,
            "failed to write warm event file");
        return if is_no_space(&e) {
            WriteOutcome::NoSpace
        } else {
            WriteOutcome::Failed
        };
    }

    // Step 2: metadata under final names, so a crash before the commit rename
    // lets recovery adopt them. Failures here are non-fatal — the video wins.
    // Object events always get a sidecar (detections); follow-on chunks get one
    // too — even movement-only chunks — so `continues` survives a restart scan.
    if event.has_objects || event.continues {
        let meta_path = file_path.with_extension("json");
        if let Err(e) =
            write_metadata_atomic(&meta_path, build_sidecar_json(event).as_bytes()).await
        {
            tracing::warn!(error = %e, "failed to write event metadata");
        }
    }
    let has_filmstrip = match event.filmstrip_frames {
        Some(ref frames) => write_filmstrip(&camera_dir, &stem, frames).await,
        None => false,
    };

    // Step 3: commit.
    if let Err(e) = tokio::fs::rename(&staging_path, &file_path).await {
        tracing::error!(camera = %camera_id, path = %file_path.display(), error = %e,
            "failed to finalize warm event file");
        let _ = tokio::fs::remove_file(&staging_path).await;
        return if is_no_space(&e) {
            WriteOutcome::NoSpace
        } else {
            WriteOutcome::Failed
        };
    }

    tracing::info!(
        camera = %camera_id,
        path = %file_path.display(),
        segments = segment_count,
        bytes = event.total_bytes,
        duration_ms = duration_ms,
        "wrote warm event file"
    );

    if let Some(index) = warm_index {
        index.insert(
            camera_id,
            build_index_entry(event, duration_ms, file_size, has_filmstrip),
        );
    }
    WriteOutcome::Written
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
        let event = assemble_event(&buf, None, "cam", 5, 7, 0, 2 * SEC, false).unwrap();
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
        let event = assemble_event(&buf, None, "cam", 6, 8, 5, 30 * SEC, false).unwrap();
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
        let event = assemble_event(&buf, None, "cam", 7, 9, 0, 30 * SEC, false).unwrap();
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
        assert!(assemble_event(&buf, None, "cam", 1, 3, 0, 0, false).is_none());
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
        let event = assemble_event(&buf, Some(&store), "cam", 5, 7, 0, 0, false).unwrap();
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
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false).unwrap()
        };
        event.filmstrip_frames = Some(Arc::new(vec![vec![0xff], vec![0xfe]]));

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let first_pts = event.first_pts;
        let outcome = write_event(dir.path(), "cam", &event, Some(&index)).await;
        assert_eq!(outcome, WriteOutcome::Written);

        // 4 one-second segments (seq 4..=7) => stem "{first_pts}_{4000}".
        let stem = format!("{}_4000", first_pts);
        let movements = dir.path().join("cam").join("movements");
        assert!(movements.join(format!("{}.ts", stem)).exists());
        assert!(movements.join(format!("{}_thumb_0.jpg", stem)).exists());
        assert!(movements.join(format!("{}_thumb_1.jpg", stem)).exists());
        // Movement-only events have no sidecar.
        assert!(!movements.join(format!("{}.json", stem)).exists());
        // Atomic pattern leaves no .tmp staging residue behind.
        let leftovers: Vec<_> = std::fs::read_dir(&movements)
            .unwrap()
            .flatten()
            .filter(|e| e.path().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "staging residue: {leftovers:?}");

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

    #[tokio::test]
    async fn movement_follow_on_chunk_writes_continues_sidecar() {
        use crate::locks::LockExt;
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        // Follow-on chunk: no pre-padding (min_start_seq == first_motion_seq),
        // movement-only, continues == true.
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 5, 0, true).unwrap()
        };
        assert!(!event.has_objects);
        assert!(event.continues);
        let first_pts = event.first_pts;

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, Some(&index)).await;

        let duration_ms = (7 - 5 + 1) * 1000;
        let stem = format!("{}_{}", first_pts, duration_ms);
        let movements = dir.path().join("cam").join("movements");
        // A movement chunk that continues DOES get a sidecar, carrying the flag.
        let sidecar = movements.join(format!("{}.json", stem));
        assert!(sidecar.exists());
        let json = std::fs::read_to_string(&sidecar).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["continues"], serde_json::json!(true));

        let entry = index.find_event("cam", first_pts).unwrap();
        assert_eq!(entry.event_type, EventType::Movement);
        assert!(entry.continues);
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

    #[tokio::test]
    async fn continuous_first_chunk_no_continues_then_follow_on_continues() {
        use crate::locks::LockExt;
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(20);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());

        // First chunk after startup: continues == false.
        let first = {
            let buf = buffer.read_recover();
            assemble_continuous_chunk(&buf, "cam", 0, 4, false).unwrap()
        };
        let first_pts = first.first_pts;
        write_event(dir.path(), "cam", &first, Some(&index)).await;

        // Follow-on chunk: continues == true.
        let second = {
            let buf = buffer.read_recover();
            assemble_continuous_chunk(&buf, "cam", 5, 9, true).unwrap()
        };
        let second_pts = second.first_pts;
        write_event(dir.path(), "cam", &second, Some(&index)).await;

        let continuous = dir.path().join("cam").join("continuous");
        // Both chunks routed to continuous/.
        assert!(continuous.join(format!("{}_5000.ts", first_pts)).exists());
        assert!(continuous.join(format!("{}_5000.ts", second_pts)).exists());
        // First chunk: no sidecar (nothing to persist). Follow-on: continues sidecar.
        assert!(!continuous.join(format!("{}_5000.json", first_pts)).exists());
        assert!(continuous
            .join(format!("{}_5000.json", second_pts))
            .exists());

        let e1 = index.find_event("cam", first_pts).unwrap();
        assert_eq!(e1.event_type, EventType::Continuous);
        assert!(!e1.continues);
        let e2 = index.find_event("cam", second_pts).unwrap();
        assert_eq!(e2.event_type, EventType::Continuous);
        assert!(e2.continues);
        assert_eq!(
            index.resolve_file_path("cam", &e2),
            continuous.join(format!("{}_5000.ts", second_pts))
        );
    }

    #[tokio::test]
    async fn continuous_chunks_round_trip_through_scan() {
        use crate::locks::LockExt;
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(20);

        // Write a first + follow-on continuous chunk with the real writer.
        let writer_index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        for (start, last, continues) in [(0u64, 4u64, false), (5, 9, true)] {
            let event = {
                let buf = buffer.read_recover();
                assemble_continuous_chunk(&buf, "cam", start, last, continues).unwrap()
            };
            write_event(dir.path(), "cam", &event, Some(&writer_index)).await;
        }

        // A fresh index scanning the same dir must recover type + continues.
        let scanned = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        scanned.scan();
        let events = scanned.query("cam", 0, u64::MAX);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.event_type == EventType::Continuous));
        assert!(!events[0].continues);
        assert!(events[1].continues);
    }
}
