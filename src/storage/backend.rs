//! Storage-backend abstraction over warm event storage.
//!
//! The warm writer and the playback API do not touch the filesystem directly;
//! they go through a [`WarmStorageBackend`]. Today the only implementation is
//! [`LocalDiskBackend`], which owns the on-disk layout, the atomic write ladder,
//! crash recovery, retention, and lazy ffmpeg thumbnailing that used to live in
//! `buffer/warm.rs` and `api/server.rs`.
//!
//! The trait is shaped so a remote HTTP backend can slot in later without
//! changing a single caller:
//!
//! * an upgrade is expressed as an *intent* ([`upgrade_event`](WarmStorageBackend::upgrade_event))
//!   rather than "rename these paths" — LocalDisk moves files, a remote backend
//!   would rewrite a sidecar in place (it has no rename);
//! * thumbnails are *acquired through the backend*
//!   ([`read_thumbnail`](WarmStorageBackend::read_thumbnail)) — LocalDisk keeps
//!   today's lazy ffmpeg generation + on-disk caching, a remote backend fetches
//!   a pre-rendered image;
//! * video is returned as a *stream*
//!   ([`read_video`](WarmStorageBackend::read_video)) — callers never see a
//!   `PathBuf` or a fully-buffered `Vec<u8>`; the body is an async byte stream
//!   with HTTP Range support, so a 10-60 MB event never lands whole in RAM.

use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use tokio_util::io::ReaderStream;

use crate::buffer::warm::{EventUpgrade, FinishedEvent};
use crate::buffer::GopSegment;
use crate::storage::warm_index::{
    free_space_bytes, should_emergency_prune, DetectionDetail, EmergencyOutcome,
};
use crate::storage::{EventType, WarmEventEntry, WarmEventIndex};

const NANOS_PER_MS: u64 = 1_000_000;

/// Result of a single event write attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Written,
    /// The write failed with ENOSPC — worth an emergency prune and one retry.
    NoSpace,
    /// The write failed for any other reason (already logged).
    Failed,
}

/// Why a thumbnail could not be produced. Every variant is an internal error;
/// a missing *event* is a not-found decided by the caller, not a thumbnail
/// error. The `&'static str` messages match the pre-refactor API responses
/// byte-for-byte.
#[derive(Debug)]
pub enum ThumbnailError {
    SpawnFailed,
    ProcessError,
    GenerationFailed,
    ReadFailed,
}

impl ThumbnailError {
    pub fn message(&self) -> &'static str {
        match self {
            ThumbnailError::SpawnFailed => "failed to spawn ffmpeg",
            ThumbnailError::ProcessError => "ffmpeg process error",
            ThumbnailError::GenerationFailed => "thumbnail generation failed",
            ThumbnailError::ReadFailed => "failed to read thumbnail",
        }
    }
}

/// A boxed async byte stream of a (possibly partial) video body. Boxed so both
/// backends can return their own concrete stream (a local file reader, a remote
/// HTTP body) behind one type.
pub type VideoByteStream = Pin<Box<dyn Stream<Item = std::io::Result<Bytes>> + Send>>;

/// A single-range request parsed from an HTTP `Range` header. Only single
/// ranges are modeled; multi-range requests are declined upstream and served in
/// full. Mirrors the three RFC 7233 byte-range-spec forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeRequest {
    /// `bytes=a-b` (both bounds) or `bytes=a-` (open-ended); `end` is inclusive.
    FromTo { start: u64, end: Option<u64> },
    /// `bytes=-n`: the final `n` bytes of the object.
    Suffix(u64),
}

impl RangeRequest {
    /// Render back to a request `Range` header value, for forwarding to a
    /// Range-capable remote backend.
    pub fn header_value(&self) -> String {
        match self {
            RangeRequest::FromTo {
                start,
                end: Some(end),
            } => format!("bytes={start}-{end}"),
            RangeRequest::FromTo { start, end: None } => format!("bytes={start}-"),
            RangeRequest::Suffix(n) => format!("bytes=-{n}"),
        }
    }
}

/// How a [`read_video`](WarmStorageBackend::read_video) call resolved the
/// (optional) requested range — the handler maps this straight onto a status
/// line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedRange {
    /// The whole object is being streamed — respond `200` + `Accept-Ranges`.
    Full,
    /// A satisfied partial range `[start, end]` inclusive — respond `206`.
    Partial { start: u64, end: u64 },
    /// The requested range fell outside the object — respond `416`
    /// (`Content-Range: bytes */total`). The stream is empty.
    Unsatisfiable,
}

/// A streamed video read: the async body, the total object size, and how the
/// requested range was resolved. The body is never fully buffered in RAM.
pub struct VideoStream {
    pub stream: VideoByteStream,
    pub total_size: u64,
    pub range: ServedRange,
}

/// Resolve a requested range against an object's total size, returning the
/// satisfied inclusive `[start, end]` or `None` when unsatisfiable (RFC 7233):
/// a `bytes=a-` / `bytes=a-b` whose `a >= total`, or an empty `bytes=-0`
/// suffix. An open or over-long upper bound is clamped to the last byte.
fn resolve_range(req: RangeRequest, total: u64) -> Option<(u64, u64)> {
    if total == 0 {
        return None;
    }
    match req {
        RangeRequest::FromTo { start, end } => {
            if start >= total {
                return None;
            }
            let end = end.unwrap_or(total - 1).min(total - 1);
            if end < start {
                return None;
            }
            Some((start, end))
        }
        RangeRequest::Suffix(n) => {
            if n == 0 {
                return None;
            }
            let n = n.min(total);
            Some((total - n, total - 1))
        }
    }
}

/// Everything the warm writer and the playback API need from storage.
///
/// The in-RAM index ([`WarmEventIndex`]) that answers `query`/`find_event` and
/// backs `prune`/`emergency_prune` is an implementation detail of the concrete
/// backend, not part of this contract.
#[async_trait]
pub trait WarmStorageBackend: Send + Sync {
    // ---- writer path ----

    /// Durably persist a finished event (video bytes + optional sidecar +
    /// filmstrip frames) and index it. Atomicity, fsync, and commit ordering
    /// are the backend's concern.
    async fn write_event(&self, camera_id: &str, event: &FinishedEvent) -> WriteOutcome;

    /// Apply a movement→object upgrade as an intent: attach the new detections
    /// and reclassify the event. LocalDisk renames the files into `objects/`
    /// and rewrites the sidecar; a remote backend rewrites a sidecar in place.
    async fn upgrade_event(&self, camera_id: &str, upgrade: &EventUpgrade);

    /// Delete every event older than its per-class retention.
    ///
    /// `cancel` is the shutdown flag. A sweep is long (a remote backend deletes
    /// one event at a time, each able to sit on a request timeout) and the
    /// drain waits for it, so implementations must poll this between events and
    /// stop early — but never part-way through one event, which would strip a
    /// `.ts` and orphan its sidecar and thumbnails where no scan can find them.
    async fn prune(
        &self,
        movement_max_age_ns: u64,
        object_max_age_ns: u64,
        continuous_max_age_ns: u64,
        cancel: &std::sync::atomic::AtomicBool,
    );

    /// Low-space guard run before a write: if free space is below
    /// `min_free_bytes`, emergency-prune the oldest events. `min_free_bytes == 0`
    /// disables the guard.
    async fn guard_free_space(&self, camera_id: &str, min_free_bytes: u64);

    /// Delete oldest events (cheapest tier first) until free space recovers
    /// above `min_free_bytes` or nothing is left. Used after a write hits
    /// ENOSPC despite the guard.
    async fn emergency_prune(&self, camera_id: &str, min_free_bytes: u64);

    /// Free bytes available on the backing store, for the low-space guard.
    fn free_space(&self) -> std::io::Result<u64>;

    // ---- startup ----

    /// Rebuild the in-RAM index from durable storage. Async because a remote
    /// backend rebuilds its index over HTTP (list + sidecar fetches);
    /// LocalDisk's body is synchronous filesystem work.
    async fn scan(&self);

    /// Salvage writes interrupted by a crash or power cut, before the scan.
    fn recover_orphans(&self);

    // ---- API read path ----

    /// Events overlapping `[from_ns, to_ns]`, oldest first.
    fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry>;

    /// The event with exactly this start PTS, if indexed.
    fn find_event(&self, camera_id: &str, start_pts_ns: u64) -> Option<WarmEventEntry>;

    /// End of this camera's newest stored event, in wall-clock nanoseconds, or
    /// `None` when it has nothing stored. Seeds the recording watchdog: silence
    /// has to be measured from the last footage that exists, not from process
    /// start, or a nightly restart resets it before it can ever be reported.
    fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64>;

    /// Stream a stored event's video (callers never see a path, and the body is
    /// never fully buffered). `range` carries an optional single HTTP range; the
    /// returned [`VideoStream`] reports the total size and how the range was
    /// resolved (full / partial / unsatisfiable).
    async fn read_video(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        range: Option<RangeRequest>,
    ) -> std::io::Result<VideoStream>;

    /// Acquire the event's poster thumbnail. LocalDisk lazily generates it from
    /// the stored video via ffmpeg on first request and caches the result.
    async fn read_thumbnail(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
    ) -> Result<Vec<u8>, ThumbnailError>;

    /// Read one filmstrip frame (`index` in `0..filmstrip_frames`).
    async fn read_filmstrip(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        index: u8,
    ) -> std::io::Result<Vec<u8>>;
}

/// The local-filesystem backend: the on-disk warm store.
///
/// Owns the data directory and the in-RAM [`WarmEventIndex`]. This is the home
/// of everything filesystem-specific — the fsync/rename atomic ladder, statvfs
/// free-space checks, ffmpeg thumbnailing — that used to live inline in the
/// writer and the API handlers.
pub struct LocalDiskBackend {
    data_dir: PathBuf,
    index: WarmEventIndex,
    camera_ids: Vec<String>,
}

impl LocalDiskBackend {
    pub fn new(data_dir: PathBuf, camera_ids: &[String]) -> Self {
        let index = WarmEventIndex::new(camera_ids, data_dir.clone());
        Self {
            data_dir,
            index,
            camera_ids: camera_ids.to_vec(),
        }
    }
}

#[async_trait]
impl WarmStorageBackend for LocalDiskBackend {
    async fn write_event(&self, camera_id: &str, event: &FinishedEvent) -> WriteOutcome {
        write_event(&self.data_dir, camera_id, event, Some(&self.index)).await
    }

    async fn upgrade_event(&self, camera_id: &str, upgrade: &EventUpgrade) {
        upgrade_event(&self.data_dir, camera_id, upgrade, Some(&self.index)).await
    }

    async fn prune(
        &self,
        movement_max_age_ns: u64,
        object_max_age_ns: u64,
        continuous_max_age_ns: u64,
        cancel: &std::sync::atomic::AtomicBool,
    ) {
        self.index
            .prune(
                movement_max_age_ns,
                object_max_age_ns,
                continuous_max_age_ns,
                || cancel.load(std::sync::atomic::Ordering::Relaxed),
            )
            .await;
    }

    async fn guard_free_space(&self, camera_id: &str, min_free_bytes: u64) {
        if min_free_bytes == 0 {
            return;
        }
        // data_dir may not exist before the first write; statvfs needs it.
        let _ = tokio::fs::create_dir_all(&self.data_dir).await;
        match self.free_space() {
            Ok(free) if should_emergency_prune(free, min_free_bytes) => {
                tracing::warn!(
                    camera = %camera_id,
                    free_bytes = free,
                    min_free_bytes = min_free_bytes,
                    "storage low on space, emergency-pruning oldest events"
                );
                self.emergency_prune(camera_id, min_free_bytes).await;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(camera = %camera_id, error = %e, "free-space check failed");
            }
        }
    }

    async fn emergency_prune(&self, camera_id: &str, min_free_bytes: u64) {
        let data_dir = self.data_dir.clone();
        let outcome = self
            .index
            .emergency_prune(move || {
                // Stop as soon as space recovers; a failing statvfs also stops
                // the prune rather than deleting everything blindly.
                free_space_bytes(&data_dir)
                    .map(|free| !should_emergency_prune(free, min_free_bytes))
                    .unwrap_or(true)
            })
            .await;

        // Three ways to reclaim nothing, three different things to go and do
        // about it — and this is read during a disk emergency.
        match outcome {
            EmergencyOutcome {
                deleted: 0,
                failed,
                missing: _,
            } if failed > 0 => {
                tracing::error!(
                    camera = %camera_id,
                    failed,
                    "emergency prune could not delete ANY event: there are events to delete \
                     and the filesystem is refusing to — check for a read-only mount or \
                     failing disk (per-file errors at debug level)"
                );
            }
            EmergencyOutcome {
                deleted: 0,
                missing,
                ..
            } if missing > 0 => {
                tracing::warn!(
                    camera = %camera_id,
                    missing,
                    "emergency prune found its candidates already gone from disk; \
                     dropped the stale index entries"
                );
            }
            EmergencyOutcome { deleted: 0, .. } => {
                tracing::warn!(
                    camera = %camera_id,
                    "emergency prune had nothing left to delete"
                );
            }
            EmergencyOutcome {
                deleted,
                failed,
                missing,
            } => {
                tracing::warn!(
                    camera = %camera_id,
                    deleted,
                    failed,
                    missing,
                    "emergency prune complete"
                );
            }
        }
    }

    fn free_space(&self) -> std::io::Result<u64> {
        free_space_bytes(&self.data_dir)
    }

    async fn scan(&self) {
        self.index.scan();
    }

    fn recover_orphans(&self) {
        crate::storage::recover_orphans(&self.data_dir, &self.camera_ids);
    }

    fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        self.index.query(camera_id, from_ns, to_ns)
    }

    fn find_event(&self, camera_id: &str, start_pts_ns: u64) -> Option<WarmEventEntry> {
        self.index.find_event(camera_id, start_pts_ns)
    }

    fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        self.index.newest_event_end_ns(camera_id)
    }

    async fn read_video(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        range: Option<RangeRequest>,
    ) -> std::io::Result<VideoStream> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let path = self.index.resolve_file_path(camera_id, entry);
        let mut file = tokio::fs::File::open(&path).await?;
        let total_size = file.metadata().await?.len();

        let Some(req) = range else {
            return Ok(VideoStream {
                stream: Box::pin(ReaderStream::new(file)),
                total_size,
                range: ServedRange::Full,
            });
        };

        match resolve_range(req, total_size) {
            Some((start, end)) => {
                file.seek(std::io::SeekFrom::Start(start)).await?;
                // `+ 1` because the range is inclusive on both ends.
                let limited = file.take(end - start + 1);
                Ok(VideoStream {
                    stream: Box::pin(ReaderStream::new(limited)),
                    total_size,
                    range: ServedRange::Partial { start, end },
                })
            }
            None => Ok(VideoStream {
                // Empty body: the handler answers 416 with a short text message.
                stream: Box::pin(ReaderStream::new(tokio::io::empty())),
                total_size,
                range: ServedRange::Unsatisfiable,
            }),
        }
    }

    async fn read_thumbnail(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
    ) -> Result<Vec<u8>, ThumbnailError> {
        let ts_path = self.index.resolve_file_path(camera_id, entry);
        let thumb_path = ts_path.with_extension("jpg");

        // Cache hit: the poster frame was already rendered.
        if let Ok(data) = tokio::fs::read(&thumb_path).await {
            return Ok(data);
        }

        // Lazily render + cache the poster frame from the stored video.
        generate_thumbnail(&ts_path, &thumb_path).await?;
        tokio::fs::read(&thumb_path)
            .await
            .map_err(|_| ThumbnailError::ReadFailed)
    }

    async fn read_filmstrip(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        index: u8,
    ) -> std::io::Result<Vec<u8>> {
        let ts_path = self.index.resolve_file_path(camera_id, entry);
        let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
        let thumb_path = ts_path
            .parent()
            .unwrap_or(&self.data_dir)
            .join(format!("{}_thumb_{}.jpg", stem, index));
        tokio::fs::read(&thumb_path).await
    }
}

// ---------------------------------------------------------------------------
// Filesystem write mechanics (moved verbatim out of buffer/warm.rs). Kept as
// free functions with their original signatures so the writer's behavior tests
// exercise them exactly as before.
// ---------------------------------------------------------------------------

fn concatenate_segments(segments: &[GopSegment], capacity: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(capacity);
    for seg in segments {
        data.extend_from_slice(&seg.data);
    }
    data
}

fn build_sidecar_json(event: &FinishedEvent) -> String {
    sidecar_json(
        event.backend.as_deref(),
        event.model.as_deref(),
        &event.detection_details,
        event.continues,
    )
}

/// Sidecar JSON shared by fresh writes and post-hoc upgrades.
fn sidecar_json(
    backend: Option<&str>,
    model: Option<&str>,
    detection_details: &[DetectionDetail],
    continues: bool,
) -> String {
    let mut meta = serde_json::Map::new();
    if let Some(backend) = backend {
        meta.insert("backend".to_string(), serde_json::json!(backend));
    }
    if let Some(model) = model {
        meta.insert("model".to_string(), serde_json::json!(model));
    }

    let deduped = deduplicate_detections(detection_details);
    let detections: Vec<serde_json::Value> = deduped
        .iter()
        .map(|(class, confidence)| serde_json::json!({"class": class, "confidence": confidence}))
        .collect();
    meta.insert("detections".to_string(), serde_json::json!(detections));

    // Only follow-on chunks carry `continues`; omit it otherwise so ordinary
    // sidecars stay unchanged.
    if continues {
        meta.insert("continues".to_string(), serde_json::json!(true));
    }

    serde_json::to_string(&meta).unwrap()
}

pub(crate) fn deduplicate_detections(details: &[DetectionDetail]) -> Vec<(String, f32)> {
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
fn tmp_path(final_path: &Path) -> PathBuf {
    let mut name = final_path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    final_path.with_file_name(name)
}

fn is_no_space(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::ENOSPC)
}

/// Write `data` to `path`, fsyncing before returning so the bytes are durable
/// (not just in the page cache) even across a power cut.
async fn write_file_synced(path: &Path, data: &[u8]) -> std::io::Result<()> {
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
async fn write_metadata_atomic(final_path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = tmp_path(final_path);
    tokio::fs::write(&tmp, data).await?;
    if let Err(e) = tokio::fs::rename(&tmp, final_path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}

/// Write the filmstrip thumbnails and return how many landed on disk. Frames
/// are numbered contiguously from 0, so a mid-run failure would truncate the
/// visible strip; in practice these small writes rarely fail.
async fn write_filmstrip(camera_dir: &Path, stem: &str, frames: &[Vec<u8>]) -> usize {
    let mut wrote = 0;
    for (i, jpeg) in frames.iter().enumerate() {
        let thumb_path = camera_dir.join(format!("{}_thumb_{}.jpg", stem, i));
        if let Err(e) = write_metadata_atomic(&thumb_path, jpeg).await {
            tracing::warn!(error = %e, "failed to write filmstrip thumbnail");
        } else {
            wrote += 1;
        }
    }
    wrote
}

fn build_index_entry(
    event: &FinishedEvent,
    duration_ms: u64,
    file_size: u64,
    filmstrip_frames: usize,
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
        filmstrip_frames,
        continues: event.continues,
        // Live writes are never recovered files; the flag only enters the
        // index via startup orphan recovery + sidecar scan.
        recovered: false,
        delete_failed: false,
    }
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
    data_dir: &Path,
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
    let filmstrip_frames = match event.filmstrip_frames {
        Some(ref frames) => write_filmstrip(&camera_dir, &stem, frames).await,
        None => 0,
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
            build_index_entry(event, duration_ms, file_size, filmstrip_frames),
        );
    }
    WriteOutcome::Written
}

/// Apply a post-hoc movement→object upgrade. Runs only on the writer task,
/// so it serializes behind any pending write of the same event (FIFO
/// channel) and never races another file mutation.
///
/// Step order is chosen for crash safety — the index scan only ever looks at
/// `.ts` files, so the `.ts` rename is the commit point:
///
/// 1. write the new sidecar (with detections) atomically into `objects/`;
/// 2. rename the `.ts` from `movements/` to `objects/` — the commit;
/// 3. move the filmstrip thumbnails;
/// 4. delete the old `movements/` sidecar, if any.
///
/// A crash before step 2 leaves a stray sidecar in `objects/` that the scan
/// ignores; after step 2 the event is object-classified with its detections.
/// If the movement file is missing entirely (write failed, already pruned,
/// or a duplicate upgrade), the upgrade is skipped with a warning — the
/// detections remain visible in the detection store/API.
async fn upgrade_event(
    data_dir: &Path,
    camera_id: &str,
    upgrade: &EventUpgrade,
    warm_index: Option<&WarmEventIndex>,
) {
    let stem = format!("{}_{}", upgrade.start_pts_ns, upgrade.duration_ms);
    let camera_dir = data_dir.join(camera_id);
    let movements = camera_dir.join(EventType::Movement.dir_name());
    let objects = camera_dir.join(EventType::Object.dir_name());
    let src_ts = movements.join(format!("{stem}.ts"));
    let dst_ts = objects.join(format!("{stem}.ts"));

    if tokio::fs::metadata(&src_ts).await.is_err() {
        tracing::warn!(
            camera = %camera_id,
            path = %src_ts.display(),
            "movement event missing on disk, skipping object upgrade \
             (detections remain available in the detection store)"
        );
        return;
    }
    if let Err(e) = tokio::fs::create_dir_all(&objects).await {
        tracing::error!(camera = %camera_id, error = %e,
            "failed to create objects directory for upgrade");
        return;
    }

    // Step 1: the new sidecar, under its final name in objects/.
    let sidecar = sidecar_json(
        Some(&upgrade.backend),
        Some(&upgrade.model),
        &upgrade.detections,
        upgrade.continues,
    );
    let dst_sidecar = objects.join(format!("{stem}.json"));
    if let Err(e) = write_metadata_atomic(&dst_sidecar, sidecar.as_bytes()).await {
        tracing::error!(camera = %camera_id, error = %e,
            "failed to write upgraded sidecar, aborting upgrade");
        return;
    }

    // Step 2: commit — move the footage.
    if let Err(e) = tokio::fs::rename(&src_ts, &dst_ts).await {
        tracing::error!(camera = %camera_id, error = %e,
            "failed to move event to objects/, aborting upgrade");
        let _ = tokio::fs::remove_file(&dst_sidecar).await;
        return;
    }

    // Steps 3 + 4: thumbnails follow, the stale movement sidecar goes.
    for i in 0..4 {
        let name = format!("{stem}_thumb_{i}.jpg");
        let _ = tokio::fs::rename(movements.join(&name), objects.join(&name)).await;
    }
    let _ = tokio::fs::remove_file(movements.join(format!("{stem}.json"))).await;

    if let Some(index) = warm_index {
        let updated = index.update_event(camera_id, upgrade.start_pts_ns, |entry| {
            entry.event_type = EventType::Object;
            entry.object_classes = upgrade.object_classes.clone();
            entry.detections = upgrade.detections.clone();
            entry.backend = Some(upgrade.backend.clone());
            entry.model = Some(upgrade.model.clone());
        });
        if !updated {
            // A retention sweep that snapshotted this event as a movement,
            // then found `movements/{stem}.ts` already renamed away, drops the
            // entry as vanished. The footage is fine and now lives under
            // objects/, so re-index it here rather than leaving it invisible
            // until the next startup scan re-reads the directory.
            tracing::warn!(camera = %camera_id, start_pts_ns = upgrade.start_pts_ns,
                "upgraded event was not in the warm index (pruned mid-upgrade?), re-indexing it");
            index.insert(
                camera_id,
                reindexed_upgrade(&dst_ts, &objects, &stem, upgrade).await,
            );
        }
    }

    tracing::info!(
        camera = %camera_id,
        path = %dst_ts.display(),
        classes = ?upgrade.object_classes,
        "upgraded movement event to object event"
    );
}

/// Rebuild an index entry for an event that was upgraded while nothing in the
/// index described it any more. Everything comes from the upgrade itself
/// except the file size and the filmstrip count, which are read back off disk
/// exactly as the startup scan would read them.
async fn reindexed_upgrade(
    dst_ts: &Path,
    objects: &Path,
    stem: &str,
    upgrade: &EventUpgrade,
) -> WarmEventEntry {
    let file_size = tokio::fs::metadata(dst_ts)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let mut filmstrip_frames = 0;
    while objects
        .join(format!("{stem}_thumb_{filmstrip_frames}.jpg"))
        .exists()
    {
        filmstrip_frames += 1;
    }
    WarmEventEntry {
        start_pts_ns: upgrade.start_pts_ns,
        duration_ms: upgrade.duration_ms,
        event_type: EventType::Object,
        file_size,
        object_classes: upgrade.object_classes.clone(),
        backend: Some(upgrade.backend.clone()),
        model: Some(upgrade.model.clone()),
        detections: upgrade.detections.clone(),
        filmstrip_frames,
        continues: upgrade.continues,
        // The sidecar this upgrade just wrote carries no `recovered` flag, so a
        // rescan would report false here too.
        recovered: false,
        delete_failed: false,
    }
}

/// Render a single poster frame from `ts_path` into `thumb_path` with ffmpeg.
/// Local-only: a remote backend serves pre-rendered thumbnails instead.
async fn generate_thumbnail(ts_path: &Path, thumb_path: &Path) -> Result<(), ThumbnailError> {
    let mut child = tokio::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(ts_path)
        .args(["-frames:v", "1", "-vf", "scale=320:-1", "-q:v", "5", "-y"])
        .arg(thumb_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| ThumbnailError::SpawnFailed)?;

    let status = child
        .wait()
        .await
        .map_err(|_| ThumbnailError::ProcessError)?;

    if !status.success() {
        return Err(ThumbnailError::GenerationFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::warm::{assemble_continuous_chunk, assemble_event};
    use crate::buffer::HotBuffer;
    use crate::locks::LockExt;

    const SEC: u64 = 1_000_000_000;

    /// Drain a [`VideoStream`] body to bytes (test-only; real callers stream).
    async fn drain(vs: VideoStream) -> Vec<u8> {
        use futures_util::StreamExt;
        let mut buf = Vec::new();
        let mut stream = vs.stream;
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.unwrap());
        }
        buf
    }

    fn segment(start_pts: u64, duration_ns: u64, byte: u8) -> GopSegment {
        GopSegment {
            start_pts,
            duration_ns,
            data: std::sync::Arc::new(vec![byte; 4]),
            frame_count: 1,
        }
    }

    /// A hot buffer with `count` one-second segments (seq 0..count), where
    /// segment N starts at N seconds and holds bytes [N; 4].
    fn populated_buffer(count: u64) -> std::sync::Arc<std::sync::RwLock<HotBuffer>> {
        let buffer = HotBuffer::new("cam".to_string(), 3600);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..count {
                buf.push(segment(seq * SEC, SEC, seq as u8));
            }
        }
        buffer
    }

    fn upgrade_for(event: &FinishedEvent) -> EventUpgrade {
        EventUpgrade {
            start_pts_ns: event.first_pts,
            duration_ms: event.duration_ms() as u32,
            object_classes: vec!["person".to_string()],
            detections: vec![
                DetectionDetail {
                    class: "person".to_string(),
                    confidence: 0.7,
                },
                DetectionDetail {
                    class: "person".to_string(),
                    confidence: 0.9,
                },
            ],
            backend: "ollama".to_string(),
            model: "test-model".to_string(),
            continues: false,
        }
    }

    #[tokio::test]
    async fn write_event_persists_files_and_indexes_with_stem_key() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let mut event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };
        event.filmstrip_frames = Some(std::sync::Arc::new(vec![vec![0xff], vec![0xfe]]));

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
        assert_eq!(entry.filmstrip_frames, 2);
        assert_eq!(
            index.resolve_file_path("cam", &entry),
            movements.join(format!("{}.ts", stem))
        );
    }

    #[tokio::test]
    async fn movement_follow_on_chunk_writes_continues_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        // Follow-on chunk: no pre-padding (min_start_seq == first_motion_seq),
        // movement-only, continues == true.
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 5, 0, true, None).unwrap()
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

    #[tokio::test]
    async fn upgrade_moves_movement_event_to_objects() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let mut event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };
        event.filmstrip_frames = Some(std::sync::Arc::new(vec![vec![0xff], vec![0xfe]]));
        let first_pts = event.first_pts;

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, Some(&index)).await;
        assert_eq!(
            index.find_event("cam", first_pts).unwrap().event_type,
            EventType::Movement
        );

        upgrade_event(dir.path(), "cam", &upgrade_for(&event), Some(&index)).await;

        let stem = format!("{}_4000", first_pts);
        let movements = dir.path().join("cam").join("movements");
        let objects = dir.path().join("cam").join("objects");
        // Files moved: .ts, sidecar, thumbnails all under objects/ now.
        assert!(objects.join(format!("{stem}.ts")).exists());
        assert!(objects.join(format!("{stem}.json")).exists());
        assert!(objects.join(format!("{stem}_thumb_0.jpg")).exists());
        assert!(objects.join(format!("{stem}_thumb_1.jpg")).exists());
        assert!(!movements.join(format!("{stem}.ts")).exists());
        assert!(!movements.join(format!("{stem}_thumb_0.jpg")).exists());

        // Sidecar carries the detections (deduped to best per class).
        let json = std::fs::read_to_string(objects.join(format!("{stem}.json"))).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["backend"], serde_json::json!("ollama"));
        assert_eq!(parsed["model"], serde_json::json!("test-model"));
        assert_eq!(
            parsed["detections"][0]["class"],
            serde_json::json!("person")
        );
        assert!((parsed["detections"][0]["confidence"].as_f64().unwrap() - 0.9).abs() < 0.01);
        assert!(parsed.get("continues").is_none());

        // Index entry updated in place: retention class is now Object.
        let entry = index.find_event("cam", first_pts).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        assert_eq!(entry.backend.as_deref(), Some("ollama"));
        assert_eq!(entry.detections.len(), 2);
        // resolve_file_path follows the new event type.
        assert_eq!(
            index.resolve_file_path("cam", &entry),
            objects.join(format!("{stem}.ts"))
        );
    }

    #[tokio::test]
    async fn upgraded_event_round_trips_through_scan() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 5, 0, true, None).unwrap()
        };
        let first_pts = event.first_pts;
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, Some(&index)).await;

        let mut upgrade = upgrade_for(&event);
        upgrade.continues = true;
        upgrade_event(dir.path(), "cam", &upgrade, Some(&index)).await;

        // A fresh scan of the directory sees an object event with the
        // continues flag preserved; no stale movement sidecar remains.
        let scanned = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        scanned.scan();
        let entry = scanned.find_event("cam", first_pts).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert!(entry.continues);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path().join("cam").join("movements"))
            .unwrap()
            .flatten()
            .collect();
        assert!(leftovers.is_empty(), "movement residue: {leftovers:?}");
    }

    /// The interleaving the index-level key test cannot reach: a sweep
    /// snapshots the movement, this upgrade renames the file into objects/,
    /// and only then does the sweep look — finds `movements/{stem}.ts` gone,
    /// and unindexes the entry. The footage is fine, so the upgrade puts it
    /// back rather than leaving it invisible until the next restart.
    #[tokio::test]
    async fn upgrade_reindexes_an_event_a_racing_prune_unindexed() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let mut event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };
        event.filmstrip_frames = Some(std::sync::Arc::new(vec![vec![0xff], vec![0xfe]]));
        let first_pts = event.first_pts;

        // Files on disk, deliberately absent from the index: exactly the state
        // a racing sweep leaves behind when it unindexes as vanished.
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, None).await;
        assert!(index.find_event("cam", first_pts).is_none());

        upgrade_event(dir.path(), "cam", &upgrade_for(&event), Some(&index)).await;

        let entry = index
            .find_event("cam", first_pts)
            .expect("upgraded event stayed invisible until the next restart");
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        assert_eq!(entry.detections.len(), 2);
        // Read back off disk, as the startup scan would have.
        assert_eq!(entry.file_size, 16);
        assert_eq!(entry.filmstrip_frames, 2);
        assert_eq!(
            index.resolve_file_path("cam", &entry),
            dir.path()
                .join("cam")
                .join("objects")
                .join(format!("{first_pts}_4000.ts"))
        );
    }

    /// The trait seam itself: the flag the retention task holds has to reach
    /// the sweep, not just exist.
    #[tokio::test]
    async fn local_disk_prune_honors_the_cancel_flag() {
        let (backend, entry, dir) = backend_with_event().await;
        let start_pts = entry.start_pts_ns;

        backend
            .prune(1, 1, 1, &std::sync::atomic::AtomicBool::new(true))
            .await;
        assert!(backend.find_event("cam", start_pts).is_some());

        backend
            .prune(1, 1, 1, &std::sync::atomic::AtomicBool::new(false))
            .await;
        assert!(backend.find_event("cam", start_pts).is_none());
        drop(dir);
    }

    #[tokio::test]
    async fn upgrade_of_missing_event_is_a_safe_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let upgrade = EventUpgrade {
            start_pts_ns: 12345,
            duration_ms: 4000,
            object_classes: vec!["person".to_string()],
            detections: vec![],
            backend: "ollama".to_string(),
            model: "m".to_string(),
            continues: false,
        };
        // Never written (or already pruned): nothing happens, nothing panics.
        upgrade_event(dir.path(), "cam", &upgrade, Some(&index)).await;
        assert!(!dir.path().join("cam").join("objects").exists());
        assert!(index.find_event("cam", 12345).is_none());
    }

    #[tokio::test]
    async fn continuous_first_chunk_no_continues_then_follow_on_continues() {
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

    #[tokio::test]
    async fn local_disk_backend_writes_reads_and_upgrades_through_the_trait() {
        // Trait-level smoke test: exercise the public seam end-to-end.
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let mut event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };
        event.filmstrip_frames = Some(std::sync::Arc::new(vec![vec![0xff], vec![0xfe]]));
        let first_pts = event.first_pts;

        let backend: std::sync::Arc<dyn WarmStorageBackend> = std::sync::Arc::new(
            LocalDiskBackend::new(dir.path().to_path_buf(), &["cam".to_string()]),
        );

        assert_eq!(
            backend.write_event("cam", &event).await,
            WriteOutcome::Written
        );

        // Indexed and queryable through the trait.
        let entry = backend.find_event("cam", first_pts).unwrap();
        assert_eq!(entry.event_type, EventType::Movement);
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 1);

        // Video comes back as a stream, not a path or a Vec.
        let video = drain(backend.read_video("cam", &entry, None).await.unwrap()).await;
        assert_eq!(video.len(), 16);

        // Filmstrip frames are readable through the backend.
        assert!(backend.read_filmstrip("cam", &entry, 0).await.is_ok());
        assert!(backend.read_filmstrip("cam", &entry, 1).await.is_ok());

        // Upgrade intent reclassifies the event to Object.
        backend.upgrade_event("cam", &upgrade_for(&event)).await;
        let upgraded = backend.find_event("cam", first_pts).unwrap();
        assert_eq!(upgraded.event_type, EventType::Object);
        assert_eq!(upgraded.object_classes, vec!["person".to_string()]);
    }

    // ---- range resolution --------------------------------------------------

    #[test]
    fn resolve_range_covers_every_form() {
        let total = 100;
        // bytes=10-19  → the middle window.
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 10,
                    end: Some(19)
                },
                total
            ),
            Some((10, 19))
        );
        // bytes=50-    → from an offset to EOF.
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 50,
                    end: None
                },
                total
            ),
            Some((50, 99))
        );
        // bytes=0-     → the whole object, as a partial.
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 0,
                    end: None
                },
                total
            ),
            Some((0, 99))
        );
        // bytes=-20    → the suffix.
        assert_eq!(
            resolve_range(RangeRequest::Suffix(20), total),
            Some((80, 99))
        );
        // Upper bound past EOF is clamped to the last byte.
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 90,
                    end: Some(500)
                },
                total
            ),
            Some((90, 99))
        );
        // A suffix longer than the object clamps to the whole object.
        assert_eq!(
            resolve_range(RangeRequest::Suffix(500), total),
            Some((0, 99))
        );
    }

    #[test]
    fn resolve_range_rejects_unsatisfiable() {
        let total = 100;
        // start at or past EOF.
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 100,
                    end: None
                },
                total
            ),
            None
        );
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 200,
                    end: Some(300)
                },
                total
            ),
            None
        );
        // bytes=-0 is an empty, unsatisfiable suffix.
        assert_eq!(resolve_range(RangeRequest::Suffix(0), total), None);
        // Any range against a zero-length object is unsatisfiable.
        assert_eq!(
            resolve_range(
                RangeRequest::FromTo {
                    start: 0,
                    end: None
                },
                0
            ),
            None
        );
    }

    #[test]
    fn range_request_renders_header_value() {
        assert_eq!(
            RangeRequest::FromTo {
                start: 10,
                end: Some(19)
            }
            .header_value(),
            "bytes=10-19"
        );
        assert_eq!(
            RangeRequest::FromTo {
                start: 50,
                end: None
            }
            .header_value(),
            "bytes=50-"
        );
        assert_eq!(RangeRequest::Suffix(20).header_value(), "bytes=-20");
    }

    // ---- LocalDisk streamed range reads ------------------------------------

    /// A backend holding one 16-byte movement event; returns (backend, entry).
    async fn backend_with_event() -> (LocalDiskBackend, WarmEventEntry, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };
        let first_pts = event.first_pts;
        let backend = LocalDiskBackend::new(dir.path().to_path_buf(), &["cam".to_string()]);
        assert_eq!(
            backend.write_event("cam", &event).await,
            WriteOutcome::Written
        );
        let entry = backend.find_event("cam", first_pts).unwrap();
        (backend, entry, dir)
    }

    #[tokio::test]
    async fn local_disk_full_read_reports_size() {
        let (backend, entry, _dir) = backend_with_event().await;
        let vs = backend.read_video("cam", &entry, None).await.unwrap();
        assert_eq!(vs.total_size, 16);
        assert!(matches!(vs.range, ServedRange::Full));
        assert_eq!(drain(vs).await.len(), 16);
    }

    #[tokio::test]
    async fn local_disk_range_reads_start_middle_suffix() {
        // The 16-byte body is four 4-byte segments: [4;4][5;4][6;4][7;4].
        let (backend, entry, _dir) = backend_with_event().await;

        // Start window: bytes 0-3 → the first segment (all 4s).
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 0,
                    end: Some(3),
                }),
            )
            .await
            .unwrap();
        assert!(matches!(
            vs.range,
            ServedRange::Partial { start: 0, end: 3 }
        ));
        assert_eq!(vs.total_size, 16);
        assert_eq!(drain(vs).await, vec![4u8; 4]);

        // Middle window: bytes 4-11 spans the 5s and 6s segments.
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 4,
                    end: Some(11),
                }),
            )
            .await
            .unwrap();
        assert!(matches!(
            vs.range,
            ServedRange::Partial { start: 4, end: 11 }
        ));
        let mut expected = vec![5u8; 4];
        expected.extend_from_slice(&[6u8; 4]);
        assert_eq!(drain(vs).await, expected);

        // Suffix: last 4 bytes → the final segment (all 7s).
        let vs = backend
            .read_video("cam", &entry, Some(RangeRequest::Suffix(4)))
            .await
            .unwrap();
        assert!(matches!(
            vs.range,
            ServedRange::Partial { start: 12, end: 15 }
        ));
        assert_eq!(drain(vs).await, vec![7u8; 4]);
    }

    #[tokio::test]
    async fn local_disk_open_ended_range_clamps_to_eof() {
        let (backend, entry, _dir) = backend_with_event().await;
        // bytes=12- → from offset 12 to EOF (clamped end = 15).
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 12,
                    end: None,
                }),
            )
            .await
            .unwrap();
        assert!(matches!(
            vs.range,
            ServedRange::Partial { start: 12, end: 15 }
        ));
        assert_eq!(drain(vs).await, vec![7u8; 4]);

        // An upper bound past EOF is clamped rather than rejected.
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 8,
                    end: Some(999),
                }),
            )
            .await
            .unwrap();
        assert!(matches!(
            vs.range,
            ServedRange::Partial { start: 8, end: 15 }
        ));
        assert_eq!(drain(vs).await.len(), 8);
    }

    #[tokio::test]
    async fn local_disk_unsatisfiable_range_is_reported() {
        let (backend, entry, _dir) = backend_with_event().await;
        // start == total → unsatisfiable, empty body, size still reported.
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 16,
                    end: None,
                }),
            )
            .await
            .unwrap();
        assert!(matches!(vs.range, ServedRange::Unsatisfiable));
        assert_eq!(vs.total_size, 16);
        assert!(drain(vs).await.is_empty());
    }
}
