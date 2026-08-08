//! Storage-backend abstraction over warm event storage.

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use tokio_util::io::ReaderStream;

use crate::buffer::warm::{EventUpgrade, FinishedEvent};
use crate::locks::SingleFlight;
use crate::storage::event_index::{
    filmstrip_frame_count, EmergencyOutcome, EventPage, MAX_FILMSTRIP_FRAMES,
};
use crate::storage::warm_index::{free_space_bytes, should_emergency_prune, sidecar_json};
use crate::storage::{EventRef, EventType, StorageAnchor, WarmEventEntry, WarmEventIndex};

const NANOS_PER_MS: u64 = 1_000_000;

/// Result of a single event write attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Written,
    /// The write failed with ENOSPC — worth an emergency prune and one retry.
    NoSpace,
    /// The write failed for any other reason (already logged) — including a commit that
    /// landed but could not be made durable, where the event is on disk and indexed yet not
    /// guaranteed to survive a power cut. Never retried by the writer.
    Failed,
}

/// Why a thumbnail could not be produced. Every variant is an internal error; a missing *event*
/// is a not-found decided by the caller, not a thumbnail error. The `&'static str` messages
/// match the pre-refactor API responses byte-for-byte.
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

impl ServedRange {
    /// A partial range, but only if `[start, end]` really is one of an object of `total` bytes:
    /// bounds in order and inside the object.
    pub fn partial(start: u64, end: u64, total: u64) -> Option<Self> {
        (start <= end && end < total).then_some(Self::Partial { start, end })
    }

    /// How many bytes the body of this range is, given the object's total size,
    /// or `None` for a range that is not one of that object. Never subtracts
    /// without having established the order first.
    pub fn body_len(&self, total: u64) -> Option<u64> {
        match *self {
            ServedRange::Full => Some(total),
            ServedRange::Unsatisfiable => Some(0),
            ServedRange::Partial { start, end } => {
                (start <= end && end < total).then(|| end - start + 1)
            }
        }
    }
}

/// A streamed video read: the async body, the total object size, and how the
/// requested range was resolved. The body is never fully buffered in RAM.
pub struct VideoStream {
    pub stream: VideoByteStream,
    pub total_size: u64,
    pub range: ServedRange,
}

/// Resolve a requested range against an object's total size, returning the satisfied inclusive
/// `[start, end]` or `None` when unsatisfiable (RFC 7233): a `bytes=a-` / `bytes=a-b` whose `a
/// >= total`, or an empty `bytes=-0` suffix.
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
#[async_trait]
pub trait WarmStorageBackend: Send + Sync {
    /// Durably persist a finished event (video bytes + optional sidecar +
    /// filmstrip frames) and index it. Atomicity, fsync, and commit ordering
    /// are the backend's concern.
    async fn write_event(&self, camera_id: &str, event: &FinishedEvent) -> WriteOutcome;

    /// Apply a movement→object upgrade as an intent: attach the new detections
    /// and reclassify the event. LocalDisk renames the files into `objects/`
    /// and rewrites the sidecar; a remote backend rewrites a sidecar in place.
    async fn upgrade_event(&self, camera_id: &str, upgrade: &EventUpgrade);

    /// Delete events older than their per-class retention, bounded by the per-camera share one
    /// sweep may take (`cap_sweep_deletions`), so a forward clock jump cannot empty an archive
    /// in a single pass.
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

    /// Rebuild the in-RAM index from durable storage. Async because a remote backend rebuilds
    /// its index over HTTP (list + sidecar fetches); LocalDisk's body is synchronous filesystem
    /// work.
    async fn scan(&self) -> std::io::Result<()>;

    /// Salvage writes interrupted by a crash or power cut, before the scan.
    fn recover_orphans(&self);

    /// The volume watch, for a backend that has a local filesystem to lose.
    fn volume_anchor(&self) -> Option<&Arc<StorageAnchor>> {
        None
    }

    /// One page of the events overlapping the request's window, oldest first
    /// and never more than the page's limit (see [`EventPage`]).
    fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry>;

    /// The one indexed event this key names, if it is still there.
    fn find_event(&self, camera_id: &str, event: EventRef) -> Option<WarmEventEntry>;

    /// End of this camera's newest stored event, in wall-clock nanoseconds, or `None` when it
    /// has nothing stored.
    fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64>;

    /// Stream a stored event's video (callers never see a path, and the body is never fully
    /// buffered).
    async fn read_video(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        range: Option<RangeRequest>,
    ) -> std::io::Result<VideoStream>;

    /// Acquire the event's poster thumbnail. LocalDisk lazily generates it from
    /// the stored video via ffmpeg on first request and caches the result;
    /// concurrent requests for the same missing thumbnail share that one render.
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
pub struct LocalDiskBackend {
    data_dir: PathBuf,
    index: WarmEventIndex,
    camera_ids: Vec<String>,
    anchor: Arc<StorageAnchor>,
    /// One ffmpeg render per missing poster frame, however many requests ask
    /// for it at once — the event list loads a whole page of posters, so the
    /// simultaneous miss is the normal case rather than the rare one.
    thumbnails: SingleFlight<PathBuf>,
}

impl LocalDiskBackend {
    /// Marks `data_dir` on the way through — the only I/O in here — so the
    /// device the archive is on is recorded before anything is written to it.
    pub fn new(data_dir: PathBuf, camera_ids: &[String]) -> Self {
        let index = WarmEventIndex::new(camera_ids, data_dir.clone());
        let anchor = Arc::new(StorageAnchor::new(data_dir.clone()));
        Self {
            data_dir,
            index,
            camera_ids: camera_ids.to_vec(),
            anchor,
            thumbnails: SingleFlight::new(),
        }
    }
}

#[cfg(test)]
impl LocalDiskBackend {
    /// The index the listing routes read, and read exclusively — so a test can
    /// stock an archive of any depth without writing a byte, which is what a
    /// test about the *cost* of a listing needs.
    pub(crate) fn index_for_tests(&self) -> &WarmEventIndex {
        &self.index
    }
}

#[async_trait]
impl WarmStorageBackend for LocalDiskBackend {
    async fn write_event(&self, camera_id: &str, event: &FinishedEvent) -> WriteOutcome {
        write_event(&self.data_dir, camera_id, event, &self.index).await
    }

    async fn upgrade_event(&self, camera_id: &str, upgrade: &EventUpgrade) {
        upgrade_event(&self.data_dir, camera_id, upgrade, &self.index).await
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
        // With the volume gone this whole guard is about the wrong disk: the `create_dir_all`
        // below would rebuild the storage tree on whatever is behind the mountpoint, and
        // statvfs would report that filesystem's free space.
        if !self.anchor.is_intact() {
            return;
        }
        // data_dir may not exist before the first write; statvfs needs it.
        let _ = crate::durable::create_dir_all_synced_async(&self.data_dir).await;
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
        // Free space read through a path that no longer leads to the storage volume is a figure
        // about some other disk, and the response to a low figure here is to delete footage.
        if !self.anchor.is_intact() {
            tracing::warn!(
                camera = %camera_id,
                "skipping emergency prune: the storage volume is not the one camon started \
                 with, so free space here says nothing about the archive"
            );
            return;
        }
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

    async fn scan(&self) -> std::io::Result<()> {
        // Always `Ok`: this walk reports the directories and files it could not read one by one
        // and indexes the rest, and a data dir that is not there at all is an empty store
        // rather than an unknown one — nothing else could have written to it either.
        self.index.scan_off_thread().await;
        Ok(())
    }

    fn recover_orphans(&self) {
        crate::storage::recover_orphans(&self.data_dir, &self.camera_ids);
    }

    fn volume_anchor(&self) -> Option<&Arc<StorageAnchor>> {
        Some(&self.anchor)
    }

    fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry> {
        self.index.query(camera_id, page)
    }

    /// Resolved by the whole key: here the event type is a directory and the
    /// duration is in the filename, so all three fields together name exactly
    /// one stored file.
    fn find_event(&self, camera_id: &str, event: EventRef) -> Option<WarmEventEntry> {
        self.index.find_event(camera_id, event)
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

        // `resolve_range` already keeps the bounds in order and inside the
        // file, so the constructor never refuses here; it is what makes that a
        // property of the type rather than of this one call site.
        match resolve_range(req, total_size)
            .and_then(|(start, end)| ServedRange::partial(start, end, total_size))
        {
            Some(served @ ServedRange::Partial { start, end }) => {
                file.seek(std::io::SeekFrom::Start(start)).await?;
                // `+ 1` because the range is inclusive on both ends.
                let limited = file.take(end - start + 1);
                Ok(VideoStream {
                    stream: Box::pin(ReaderStream::new(limited)),
                    total_size,
                    range: served,
                })
            }
            // `partial` builds nothing else, so this is the unsatisfiable case.
            _ => Ok(VideoStream {
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

        read_or_generate_thumbnail(&self.thumbnails, &thumb_path, || {
            generate_thumbnail(&ts_path, &thumb_path)
        })
        .await
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

/// Local sidecars carry no `event_type`: the directory an event lives in is
/// what says what it is, and the scan reads it from there.
fn build_sidecar_json(event: &FinishedEvent) -> String {
    sidecar_json(
        None,
        event.backend.as_deref(),
        event.model.as_deref(),
        &event.detection_details,
        event.continues,
    )
}

fn is_no_space(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::ENOSPC)
}

/// Atomically write a small metadata file (sidecar/thumbnail): stage as `.tmp`, then rename.
async fn write_metadata_atomic(final_path: &Path, data: &[u8]) -> std::io::Result<()> {
    crate::durable::replace_atomic_async(final_path, data).await
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
        // The filesystem counts the sidecar and the thumbnails without being told to, and
        // `statvfs` is this backend's accounting authority — see
        // [`crate::storage::contract`].
        sidecar_bytes: 0,
        thumbnail_bytes: 0,
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
async fn write_event(
    data_dir: &Path,
    camera_id: &str,
    event: &FinishedEvent,
    warm_index: &WarmEventIndex,
) -> WriteOutcome {
    let duration_ms = event.duration_ns() / NANOS_PER_MS;
    let segment_count = event.segments.len();

    let camera_dir = data_dir.join(camera_id).join(event.event_type().dir_name());
    // Synced, so that on a first ever write the directory the event is about to
    // be committed into cannot itself be lost to a power cut. A no-op once the
    // tree exists, which is every write but the first.
    if let Err(e) = crate::durable::create_dir_all_synced_async(&camera_dir).await {
        tracing::error!(camera = %camera_id, error = %e, "failed to create warm storage directory");
        return if is_no_space(&e) {
            WriteOutcome::NoSpace
        } else {
            WriteOutcome::Failed
        };
    }

    let stem = format!("{}_{}", event.first_pts, duration_ms);
    let file_path = camera_dir.join(format!("{}.ts", stem));
    let staging_path = crate::durable::tmp_path(&file_path);
    // The segments are written as they are held: the event's bytes reach the
    // file back to back either way, and nothing here needs them contiguous in
    // memory first.
    let chunks: Vec<&[u8]> = event.segments.iter().map(|s| s.data.as_slice()).collect();
    let file_size = chunks.iter().map(|c| c.len() as u64).sum();

    // Step 1: footage first. Once this returns, the video survives a crash.
    if let Err(e) = crate::durable::write_all_synced_async(&staging_path, &chunks).await {
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

    // Step 2: metadata under final names, so a crash before the commit rename lets recovery
    // adopt them.
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

    // Step 4: make the commit durable. One fsync of the directory covers the
    // `.ts` entry and the sidecar/thumbnail entries renamed alongside it.
    let synced = crate::durable::sync_dir_async(&camera_dir).await;
    if let Err(e) = &synced {
        tracing::error!(camera = %camera_id, path = %camera_dir.display(), error = %e,
            "failed to fsync warm storage directory: the event's bytes are on disk and it is \
             indexed and served, but the directory entry naming them is not durable, so a power \
             cut could still lose it");
    }

    tracing::info!(
        camera = %camera_id,
        path = %file_path.display(),
        segments = segment_count,
        bytes = event.total_bytes,
        duration_ms = duration_ms,
        "wrote warm event file"
    );

    // Indexed either way: the bytes are on disk under their final name, so the
    // event is served, pruned and counted like any other, and a restart's scan
    // would find it regardless. Only the *guarantee* is missing.
    warm_index.insert(
        camera_id,
        build_index_entry(event, duration_ms, file_size, filmstrip_frames),
    );
    commit_outcome(synced)
}

/// What a commit whose directory fsync failed is worth reporting as.
fn commit_outcome(synced: std::io::Result<()>) -> WriteOutcome {
    match synced {
        Ok(()) => WriteOutcome::Written,
        Err(_) => WriteOutcome::Failed,
    }
}

/// Apply a post-hoc movement→object upgrade. Runs only on the writer task, so it serializes
/// behind any pending write of the same event (FIFO channel) and never races another file
/// mutation.
async fn upgrade_event(
    data_dir: &Path,
    camera_id: &str,
    upgrade: &EventUpgrade,
    warm_index: &WarmEventIndex,
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
    if let Err(e) = crate::durable::create_dir_all_synced_async(&objects).await {
        tracing::error!(camera = %camera_id, error = %e,
            "failed to create objects directory for upgrade");
        return;
    }

    // Step 1: the new sidecar, under its final name in objects/.
    let sidecar = sidecar_json(
        None,
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
    for i in 0..MAX_FILMSTRIP_FRAMES {
        let name = format!("{stem}_thumb_{i}.jpg");
        let _ = tokio::fs::rename(movements.join(&name), objects.join(&name)).await;
    }
    let _ = tokio::fs::remove_file(movements.join(format!("{stem}.json"))).await;

    // Step 5: both sides of the move, destination first and *only* then the
    // source (see `sync_move`).
    if let Err(e) = sync_move(&objects, &movements).await {
        tracing::error!(camera = %camera_id, error = %e,
            "failed to fsync a directory after upgrade: the upgrade is applied, but a power \
             cut could still undo it");
    }

    let updated = warm_index.update_event(
        camera_id,
        upgrade.start_pts_ns,
        upgrade.duration_ms,
        |entry| {
            entry.event_type = EventType::Object;
            entry.object_classes = upgrade.object_classes.clone();
            entry.detections = upgrade.detections.clone();
            entry.backend = Some(upgrade.backend.clone());
            entry.model = Some(upgrade.model.clone());
        },
    );
    if !updated {
        // A retention sweep that snapshotted this event as a movement, then found
        // `movements/{stem}.ts` already renamed away, drops the entry as vanished.
        tracing::warn!(camera = %camera_id, start_pts_ns = upgrade.start_pts_ns,
            "upgraded event was not in the warm index (pruned mid-upgrade?), re-indexing it");
        warm_index.insert(
            camera_id,
            reindexed_upgrade(&dst_ts, &objects, &stem, upgrade).await,
        );
    }

    tracing::info!(
        camera = %camera_id,
        path = %dst_ts.display(),
        classes = ?upgrade.object_classes,
        "upgraded movement event to object event"
    );
}

/// Make a rename *between* two directories durable: fsync the destination, then the source. The
/// order is not a preference and the `?` is not a shortcut — each step is only safe once the
/// one before it succeeded:
async fn sync_move(dest: &Path, src: &Path) -> std::io::Result<()> {
    crate::durable::sync_dir_async(dest).await?;
    crate::durable::sync_dir_async(src).await
}

/// Rebuild an index entry for an event that was upgraded while nothing in the index described
/// it any more.
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
    let filmstrip_frames =
        filmstrip_frame_count(|i| objects.join(format!("{stem}_thumb_{i}.jpg")).exists());
    WarmEventEntry {
        start_pts_ns: upgrade.start_pts_ns,
        duration_ms: upgrade.duration_ms,
        event_type: EventType::Object,
        file_size,
        sidecar_bytes: 0,
        thumbnail_bytes: 0,
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

/// Read the cached poster frame at `thumb_path`, calling `generate` to render it if it is
/// missing — once, however many callers miss it together.
async fn read_or_generate_thumbnail<F, Fut>(
    flight: &SingleFlight<PathBuf>,
    thumb_path: &Path,
    generate: F,
) -> Result<Vec<u8>, ThumbnailError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<(), ThumbnailError>>,
{
    // Cache hit: the poster frame was already rendered.
    if let Ok(data) = tokio::fs::read(thumb_path).await {
        return Ok(data);
    }

    let _flight = flight.acquire(thumb_path.to_path_buf()).await;

    // Whoever we queued behind has just rendered it.
    if let Ok(data) = tokio::fs::read(thumb_path).await {
        return Ok(data);
    }

    generate().await?;
    tokio::fs::read(thumb_path)
        .await
        .map_err(|_| ThumbnailError::ReadFailed)
}

/// How long one poster render may take before its ffmpeg is killed.
const THUMBNAIL_RENDER_TIMEOUT: Duration = Duration::from_secs(15);

/// Render a single poster frame from `ts_path` into `thumb_path` with ffmpeg.
/// Local-only: a remote backend serves pre-rendered thumbnails instead.
async fn generate_thumbnail(ts_path: &Path, thumb_path: &Path) -> Result<(), ThumbnailError> {
    publish_rendered(thumb_path, |staged| render_poster_frame(ts_path, staged)).await
}

/// Run `render` against a staging path and publish its output to `thumb_path` by rename,
/// bounded by [`THUMBNAIL_RENDER_TIMEOUT`].
async fn publish_rendered<F, Fut>(thumb_path: &Path, render: F) -> Result<(), ThumbnailError>
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: std::future::Future<Output = Result<(), ThumbnailError>>,
{
    let staged = crate::durable::tmp_path(thumb_path);
    let rendered = tokio::time::timeout(THUMBNAIL_RENDER_TIMEOUT, render(staged.clone())).await;

    let result = match rendered {
        Ok(Ok(())) => tokio::fs::rename(&staged, thumb_path)
            .await
            .map_err(|_| ThumbnailError::GenerationFailed),
        Ok(Err(e)) => Err(e),
        Err(_) => {
            // Dropping the render future is what kills the child: the timeout
            // has already done that by the time we get here.
            tracing::warn!(
                timeout = ?THUMBNAIL_RENDER_TIMEOUT,
                thumbnail = %thumb_path.display(),
                "thumbnail render timed out, killed ffmpeg"
            );
            Err(ThumbnailError::GenerationFailed)
        }
    };

    if result.is_err() {
        let _ = tokio::fs::remove_file(&staged).await;
    }
    result
}

/// The ffmpeg half of a poster render, writing wherever it is pointed.
async fn render_poster_frame(ts_path: &Path, out_path: PathBuf) -> Result<(), ThumbnailError> {
    let mut child = tokio::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(ts_path)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "scale=320:-1",
            "-q:v",
            "5",
            "-f",
            "image2",
            "-y",
        ])
        .arg(&out_path)
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
    use crate::buffer::{GopSegment, HotBuffer};
    use crate::locks::LockExt;
    use crate::storage::event_index::DetectionDetail;

    const SEC: u64 = 1_000_000_000;

    fn contract_backend() -> (tempfile::TempDir, LocalDiskBackend) {
        let dir = tempfile::tempdir().unwrap();
        let backend = LocalDiskBackend::new(dir.path().to_path_buf(), &["cam".to_string()]);
        (dir, backend)
    }

    fn bytes_under(dir: &Path) -> u64 {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return 0;
        };
        entries
            .filter_map(Result::ok)
            .map(|e| match e.metadata() {
                Ok(m) if m.is_dir() => bytes_under(&e.path()),
                Ok(m) => m.len(),
                Err(_) => 0,
            })
            .sum()
    }

    #[tokio::test]
    async fn contract_a_written_event_reads_back_whole() {
        let (_dir, backend) = contract_backend();
        crate::storage::contract::contract_tests::a_written_event_reads_back_whole(&backend).await;
    }

    #[tokio::test]
    async fn contract_an_event_costs_nothing_once_it_is_deleted() {
        let (dir, backend) = contract_backend();
        let root = dir.path().to_path_buf();
        crate::storage::contract::contract_tests::an_event_costs_nothing_once_it_is_deleted(
            &backend,
            || bytes_under(&root),
        )
        .await;
    }

    #[tokio::test]
    async fn contract_a_prune_that_starts_stopped_deletes_nothing() {
        let (_dir, backend) = contract_backend();
        crate::storage::contract::contract_tests::a_prune_that_starts_stopped_deletes_nothing(
            &backend,
        )
        .await;
    }

    #[tokio::test]
    async fn contract_a_rewritten_event_replaces_its_entry() {
        let (_dir, backend) = contract_backend();
        crate::storage::contract::contract_tests::a_rewritten_event_replaces_its_entry(&backend)
            .await;
    }

    #[tokio::test]
    async fn contract_an_upgrade_reclassifies_the_one_indexed_event() {
        let (_dir, backend) = contract_backend();
        crate::storage::contract::contract_tests::an_upgrade_reclassifies_the_one_indexed_event(
            &backend,
        )
        .await;
    }

    #[tokio::test]
    async fn contract_an_upgrade_of_a_deleted_event_indexes_nothing() {
        let (_dir, backend) = contract_backend();
        crate::storage::contract::contract_tests::an_upgrade_of_a_deleted_event_indexes_nothing(
            &backend,
        )
        .await;
    }

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
        let outcome = write_event(dir.path(), "cam", &event, &index).await;
        assert_eq!(outcome, WriteOutcome::Written);

        let stem = format!("{}_4000", first_pts);
        let movements = dir.path().join("cam").join("movements");
        assert!(movements.join(format!("{}.ts", stem)).exists());
        assert!(movements.join(format!("{}_thumb_0.jpg", stem)).exists());
        assert!(movements.join(format!("{}_thumb_1.jpg", stem)).exists());
        assert!(!movements.join(format!("{}.json", stem)).exists());
        let leftovers: Vec<_> = std::fs::read_dir(&movements)
            .unwrap()
            .flatten()
            .filter(|e| e.path().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "staging residue: {leftovers:?}");

        let entry = index
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Movement))
            .unwrap();
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
    async fn written_event_bytes_are_the_segments_back_to_back() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let outcome = write_event(dir.path(), "cam", &event, &index).await;
        assert_eq!(outcome, WriteOutcome::Written);

        let expected: Vec<u8> = event
            .segments
            .iter()
            .flat_map(|s| s.data.iter().copied())
            .collect();
        let path = dir.path().join("cam").join("movements").join(format!(
            "{}_{}.ts",
            event.first_pts,
            event.duration_ms()
        ));
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert_eq!(expected.len(), event.total_bytes);
    }

    #[tokio::test]
    async fn first_write_commits_into_a_tree_that_did_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("does").join("not").join("exist");
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };

        let backend = LocalDiskBackend::new(data_dir.clone(), &["cam".to_string()]);
        backend.guard_free_space("cam", 1).await;
        assert!(data_dir.is_dir(), "the guard creates the storage root");
        let outcome = backend.write_event("cam", &event).await;

        assert_eq!(outcome, WriteOutcome::Written);
        assert!(data_dir
            .join("cam")
            .join("movements")
            .join(format!("{}_4000.ts", event.first_pts))
            .exists());
    }

    #[test]
    fn a_commit_that_cannot_be_synced_is_not_reported_as_written() {
        assert_eq!(commit_outcome(Ok(())), WriteOutcome::Written);
        assert_eq!(
            commit_outcome(Err(std::io::Error::other("fsync failed"))),
            WriteOutcome::Failed
        );
    }

    #[tokio::test]
    async fn sync_move_stops_at_a_destination_it_cannot_sync() {
        let dir = tempfile::tempdir().unwrap();
        let objects = dir.path().join("objects");
        let movements = dir.path().join("movements");
        std::fs::create_dir_all(&objects).unwrap();
        std::fs::create_dir_all(&movements).unwrap();
        sync_move(&objects, &movements).await.unwrap();

        assert!(sync_move(&dir.path().join("gone"), &movements)
            .await
            .is_err());
        assert!(sync_move(&objects, &dir.path().join("gone")).await.is_err());
    }

    #[tokio::test]
    async fn write_fails_when_the_event_directory_cannot_be_created() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("cam"), b"not a directory").unwrap();
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 0, SEC, false, None).unwrap()
        };

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let outcome = write_event(dir.path(), "cam", &event, &index).await;
        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(index
            .find_event(
                "cam",
                EventRef::new(event.first_pts, 4000, EventType::Movement)
            )
            .is_none());
    }

    #[tokio::test]
    async fn movement_follow_on_chunk_writes_continues_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 5, 0, true, None).unwrap()
        };
        assert!(!event.has_objects);
        assert!(event.continues);
        let first_pts = event.first_pts;

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, &index).await;

        let duration_ms = (7 - 5 + 1) * 1000;
        let stem = format!("{}_{}", first_pts, duration_ms);
        let movements = dir.path().join("cam").join("movements");
        let sidecar = movements.join(format!("{}.json", stem));
        assert!(sidecar.exists());
        let json = std::fs::read_to_string(&sidecar).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["continues"], serde_json::json!(true));

        let entry = index
            .find_event(
                "cam",
                EventRef::new(first_pts, duration_ms as u32, EventType::Movement),
            )
            .unwrap();
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
        write_event(dir.path(), "cam", &event, &index).await;
        assert_eq!(
            index
                .find_event("cam", EventRef::new(first_pts, 4000, EventType::Movement))
                .unwrap()
                .event_type,
            EventType::Movement
        );

        upgrade_event(dir.path(), "cam", &upgrade_for(&event), &index).await;

        let stem = format!("{}_4000", first_pts);
        let movements = dir.path().join("cam").join("movements");
        let objects = dir.path().join("cam").join("objects");
        assert!(objects.join(format!("{stem}.ts")).exists());
        assert!(objects.join(format!("{stem}.json")).exists());
        assert!(objects.join(format!("{stem}_thumb_0.jpg")).exists());
        assert!(objects.join(format!("{stem}_thumb_1.jpg")).exists());
        assert!(!movements.join(format!("{stem}.ts")).exists());
        assert!(!movements.join(format!("{stem}_thumb_0.jpg")).exists());

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

        let entry = index
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Object))
            .unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        assert_eq!(entry.backend.as_deref(), Some("ollama"));
        assert_eq!(entry.detections.len(), 2);
        assert_eq!(
            index.resolve_file_path("cam", &entry),
            objects.join(format!("{stem}.ts"))
        );
    }

    fn sibling(index: &WarmEventIndex, duration_ms: u32) -> Option<WarmEventEntry> {
        index
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .into_iter()
            .find(|e| e.duration_ms == duration_ms)
    }

    #[tokio::test]
    async fn siblings_sharing_a_start_pts_are_upgraded_and_pruned_by_full_key() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let (short, long) = {
            let buf = buffer.read_recover();
            (
                assemble_event(&buf, None, "cam", 5, 5, 0, 0, false, None).unwrap(),
                assemble_event(&buf, None, "cam", 5, 7, 0, 0, false, None).unwrap(),
            )
        };
        let first_pts = long.first_pts;
        assert_eq!(short.first_pts, first_pts);
        let (short_ms, long_ms) = (short.duration_ms() as u32, long.duration_ms() as u32);
        assert_ne!(short_ms, long_ms);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &short, &index).await;
        write_event(dir.path(), "cam", &long, &index).await;
        assert_eq!(
            index.query("cam", EventPage::unbounded(0, u64::MAX)).len(),
            2
        );

        let (target_ms, spared_ms) = (short_ms, long_ms);
        upgrade_event(dir.path(), "cam", &upgrade_for(&short), &index).await;

        let movements = dir.path().join("cam").join("movements");
        let objects = dir.path().join("cam").join("objects");
        let upgraded = sibling(&index, target_ms).expect("the upgraded event left the index");
        assert_eq!(upgraded.event_type, EventType::Object);
        assert_eq!(upgraded.object_classes, vec!["person".to_string()]);
        assert_eq!(
            index.resolve_file_path("cam", &upgraded),
            objects.join(format!("{first_pts}_{target_ms}.ts"))
        );

        let untouched = sibling(&index, spared_ms).expect("the sibling left the index");
        assert_eq!(
            untouched.event_type,
            EventType::Movement,
            "the upgrade reclassified its sibling"
        );
        assert!(untouched.object_classes.is_empty());
        assert!(untouched.backend.is_none());
        assert_eq!(
            index.resolve_file_path("cam", &untouched),
            movements.join(format!("{first_pts}_{spared_ms}.ts"))
        );
        assert!(movements
            .join(format!("{first_pts}_{spared_ms}.ts"))
            .exists());
        assert!(!movements
            .join(format!("{first_pts}_{target_ms}.ts"))
            .exists());

        index.prune(1, u64::MAX, u64::MAX, || false).await;
        assert!(sibling(&index, spared_ms).is_none());
        assert!(sibling(&index, target_ms).is_some());
        assert!(!movements
            .join(format!("{first_pts}_{spared_ms}.ts"))
            .exists());
        assert!(objects.join(format!("{first_pts}_{target_ms}.ts")).exists());
    }

    #[tokio::test]
    async fn same_start_siblings_are_each_served_by_their_own_key() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let (movement, chunk) = {
            let buf = buffer.read_recover();
            (
                assemble_event(&buf, None, "cam", 5, 7, 0, 0, false, None).unwrap(),
                assemble_continuous_chunk(&buf, "cam", 5, 7, false).unwrap(),
            )
        };
        let first_pts = movement.first_pts;
        assert_eq!(chunk.first_pts, first_pts);
        let duration_ms = movement.duration_ms() as u32;
        assert_eq!(chunk.duration_ms() as u32, duration_ms);

        let backend = LocalDiskBackend::new(dir.path().to_path_buf(), &["cam".to_string()]);
        backend.write_event("cam", &movement).await;
        backend.write_event("cam", &chunk).await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            2
        );

        let mut posters = Vec::new();
        for event_type in [EventType::Movement, EventType::Continuous] {
            let key = EventRef::new(first_pts, duration_ms, event_type);
            let entry = backend
                .find_event("cam", key)
                .unwrap_or_else(|| panic!("{key} is not indexed"));
            assert_eq!(EventRef::of(&entry), key, "{key} served another event");
            let ts = dir
                .path()
                .join("cam")
                .join(event_type.dir_name())
                .join(format!("{first_pts}_{duration_ms}.ts"));
            let resolved = backend.index.resolve_file_path("cam", &entry);
            assert_eq!(resolved, ts);
            assert!(ts.exists());
            posters.push(resolved.with_extension("jpg"));
        }
        assert_ne!(posters[0], posters[1]);

        assert!(backend
            .find_event(
                "cam",
                EventRef::new(first_pts, duration_ms + 1000, EventType::Movement)
            )
            .is_none());
        assert!(backend
            .find_event(
                "cam",
                EventRef::new(first_pts, duration_ms, EventType::Object)
            )
            .is_none());
    }

    #[tokio::test]
    async fn local_sidecars_name_no_event_type() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        let event = {
            let buf = buffer.read_recover();
            assemble_event(&buf, None, "cam", 5, 7, 5, 0, true, None).unwrap()
        };
        let first_pts = event.first_pts;

        let fresh: serde_json::Value = serde_json::from_str(&build_sidecar_json(&event)).unwrap();
        assert!(
            fresh.get("event_type").is_none(),
            "a local sidecar named its own type: {fresh}"
        );

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, &index).await;
        upgrade_event(dir.path(), "cam", &upgrade_for(&event), &index).await;

        let json = std::fs::read_to_string(
            dir.path()
                .join("cam")
                .join("objects")
                .join(format!("{first_pts}_3000.json")),
        )
        .unwrap();
        let upgraded: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            upgraded.get("event_type").is_none(),
            "the upgrade's sidecar named its type: {upgraded}"
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
        write_event(dir.path(), "cam", &event, &index).await;

        let mut upgrade = upgrade_for(&event);
        upgrade.continues = true;
        upgrade_event(dir.path(), "cam", &upgrade, &index).await;

        let scanned = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        scanned.scan();
        let entry = scanned
            .find_event("cam", EventRef::new(first_pts, 3000, EventType::Object))
            .unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert!(entry.continues);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        let leftovers: Vec<_> = std::fs::read_dir(dir.path().join("cam").join("movements"))
            .unwrap()
            .flatten()
            .collect();
        assert!(leftovers.is_empty(), "movement residue: {leftovers:?}");
    }

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

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let writer_index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        write_event(dir.path(), "cam", &event, &writer_index).await;
        assert!(index
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Movement))
            .is_none());

        upgrade_event(dir.path(), "cam", &upgrade_for(&event), &index).await;

        let entry = index
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Object))
            .expect("upgraded event stayed invisible until the next restart");
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        assert_eq!(entry.detections.len(), 2);
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

    #[tokio::test]
    async fn local_disk_prune_honors_the_cancel_flag() {
        let (backend, entry, dir) = backend_with_event().await;
        let key = EventRef::of(&entry);

        backend
            .prune(1, 1, 1, &std::sync::atomic::AtomicBool::new(true))
            .await;
        assert!(backend.find_event("cam", key).is_some());

        backend
            .prune(1, 1, 1, &std::sync::atomic::AtomicBool::new(false))
            .await;
        assert!(backend.find_event("cam", key).is_none());
        drop(dir);
    }

    #[tokio::test]
    async fn a_storage_volume_that_moved_stops_the_emergency_prune() {
        let (backend, entry, dir) = backend_with_event().await;
        let key = EventRef::of(&entry);

        std::fs::remove_file(dir.path().join(".camon-volume")).unwrap();
        backend
            .volume_anchor()
            .unwrap()
            .check(std::time::Instant::now());
        backend.guard_free_space("cam", u64::MAX).await;
        backend.emergency_prune("cam", u64::MAX).await;
        assert!(
            backend.find_event("cam", key).is_some(),
            "pruned the archive to free space on a filesystem it is not on"
        );

        std::fs::write(dir.path().join(".camon-volume"), b"back").unwrap();
        backend
            .volume_anchor()
            .unwrap()
            .check(std::time::Instant::now());
        backend.emergency_prune("cam", u64::MAX).await;
        assert!(backend.find_event("cam", key).is_none());
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
        upgrade_event(dir.path(), "cam", &upgrade, &index).await;
        assert!(!dir.path().join("cam").join("objects").exists());
        assert!(index
            .find_event("cam", EventRef::new(12345, 4000, EventType::Movement))
            .is_none());
    }

    #[tokio::test]
    async fn continuous_first_chunk_no_continues_then_follow_on_continues() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(20);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());

        let first = {
            let buf = buffer.read_recover();
            assemble_continuous_chunk(&buf, "cam", 0, 4, false).unwrap()
        };
        let first_pts = first.first_pts;
        write_event(dir.path(), "cam", &first, &index).await;

        let second = {
            let buf = buffer.read_recover();
            assemble_continuous_chunk(&buf, "cam", 5, 9, true).unwrap()
        };
        let second_pts = second.first_pts;
        write_event(dir.path(), "cam", &second, &index).await;

        let continuous = dir.path().join("cam").join("continuous");
        assert!(continuous.join(format!("{}_5000.ts", first_pts)).exists());
        assert!(continuous.join(format!("{}_5000.ts", second_pts)).exists());
        assert!(!continuous.join(format!("{}_5000.json", first_pts)).exists());
        assert!(continuous
            .join(format!("{}_5000.json", second_pts))
            .exists());

        let e1 = index
            .find_event("cam", EventRef::new(first_pts, 5000, EventType::Continuous))
            .unwrap();
        assert_eq!(e1.event_type, EventType::Continuous);
        assert!(!e1.continues);
        let e2 = index
            .find_event(
                "cam",
                EventRef::new(second_pts, 5000, EventType::Continuous),
            )
            .unwrap();
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

        let writer_index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        for (start, last, continues) in [(0u64, 4u64, false), (5, 9, true)] {
            let event = {
                let buf = buffer.read_recover();
                assemble_continuous_chunk(&buf, "cam", start, last, continues).unwrap()
            };
            write_event(dir.path(), "cam", &event, &writer_index).await;
        }

        let scanned = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        scanned.scan();
        let events = scanned.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.event_type == EventType::Continuous));
        assert!(!events[0].continues);
        assert!(events[1].continues);
    }

    #[tokio::test]
    async fn local_disk_backend_writes_reads_and_upgrades_through_the_trait() {
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

        let entry = backend
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Movement))
            .unwrap();
        assert_eq!(entry.event_type, EventType::Movement);
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            1
        );

        let video = drain(backend.read_video("cam", &entry, None).await.unwrap()).await;
        assert_eq!(video.len(), 16);

        assert!(backend.read_filmstrip("cam", &entry, 0).await.is_ok());
        assert!(backend.read_filmstrip("cam", &entry, 1).await.is_ok());

        backend.upgrade_event("cam", &upgrade_for(&event)).await;
        let upgraded = backend
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Object))
            .unwrap();
        assert_eq!(upgraded.event_type, EventType::Object);
        assert_eq!(upgraded.object_classes, vec!["person".to_string()]);
    }

    #[test]
    fn resolve_range_covers_every_form() {
        let total = 100;
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
        assert_eq!(
            resolve_range(RangeRequest::Suffix(20), total),
            Some((80, 99))
        );
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
        assert_eq!(
            resolve_range(RangeRequest::Suffix(500), total),
            Some((0, 99))
        );
    }

    #[test]
    fn resolve_range_rejects_unsatisfiable() {
        let total = 100;
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
        assert_eq!(resolve_range(RangeRequest::Suffix(0), total), None);
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
        let entry = backend
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Movement))
            .unwrap();
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
        let (backend, entry, _dir) = backend_with_event().await;

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

    #[derive(Default)]
    struct Renders {
        total: std::sync::atomic::AtomicUsize,
        live: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
    }

    impl Renders {
        async fn run(&self, path: &Path, gate: &tokio::sync::Semaphore) {
            use std::sync::atomic::Ordering;
            self.total.fetch_add(1, Ordering::SeqCst);
            let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            gate.acquire().await.unwrap().forget();
            let staged = crate::durable::tmp_path(path);
            tokio::fs::write(&staged, b"jpeg").await.unwrap();
            tokio::fs::rename(&staged, path).await.unwrap();
            self.live.fetch_sub(1, Ordering::SeqCst);
        }

        fn total(&self) -> usize {
            self.total.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn peak(&self) -> usize {
            self.peak.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_thumbnail_requests_render_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poster.jpg");
        let flight = std::sync::Arc::new(SingleFlight::<PathBuf>::new());
        let renders = std::sync::Arc::new(Renders::default());
        let gate = std::sync::Arc::new(tokio::sync::Semaphore::new(0));
        let start = std::sync::Arc::new(tokio::sync::Barrier::new(9));

        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let (flight, renders, gate, start, path) = (
                    std::sync::Arc::clone(&flight),
                    std::sync::Arc::clone(&renders),
                    std::sync::Arc::clone(&gate),
                    std::sync::Arc::clone(&start),
                    path.clone(),
                );
                tokio::spawn(async move {
                    start.wait().await;
                    read_or_generate_thumbnail(&flight, &path, || async {
                        renders.run(&path, &gate).await;
                        Ok(())
                    })
                    .await
                })
            })
            .collect();

        start.wait().await;
        gate.add_permits(8);
        for t in tasks {
            assert_eq!(t.await.unwrap().unwrap(), b"jpeg");
        }

        assert_eq!(
            renders.total(),
            1,
            "one ffmpeg per thumbnail, not per request"
        );
        assert_eq!(renders.peak(), 1, "renders of one thumbnail overlapped");
        assert_eq!(flight.live_keys(), 0, "in-flight entry outlived the render");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn different_thumbnails_render_concurrently() {
        let dir = tempfile::tempdir().unwrap();
        let flight = std::sync::Arc::new(SingleFlight::<PathBuf>::new());
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));

        let tasks: Vec<_> = (0..2)
            .map(|i| {
                let (flight, barrier) = (
                    std::sync::Arc::clone(&flight),
                    std::sync::Arc::clone(&barrier),
                );
                let path = dir.path().join(format!("poster{i}.jpg"));
                tokio::spawn(async move {
                    read_or_generate_thumbnail(&flight, &path, || async {
                        barrier.wait().await;
                        tokio::fs::write(&path, b"jpeg").await.unwrap();
                        Ok(())
                    })
                    .await
                })
            })
            .collect();

        for t in tasks {
            let served = tokio::time::timeout(std::time::Duration::from_secs(5), t)
                .await
                .expect("distinct thumbnails serialized against each other")
                .unwrap()
                .unwrap();
            assert_eq!(served, b"jpeg");
        }
        assert_eq!(flight.live_keys(), 0);
    }

    #[tokio::test]
    async fn failed_thumbnail_generation_stays_retryable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poster.jpg");
        let flight = SingleFlight::<PathBuf>::new();

        let failed = read_or_generate_thumbnail(&flight, &path, || async {
            Err(ThumbnailError::SpawnFailed)
        })
        .await;
        assert!(matches!(failed, Err(ThumbnailError::SpawnFailed)));
        assert_eq!(flight.live_keys(), 0, "failure left the key held");

        let served = read_or_generate_thumbnail(&flight, &path, || async {
            tokio::fs::write(&path, b"jpeg").await.unwrap();
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(served, b"jpeg");
    }

    #[tokio::test]
    async fn partially_written_thumbnail_is_never_served() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poster.jpg");
        let flight = SingleFlight::<PathBuf>::new();

        let failed = read_or_generate_thumbnail(&flight, &path, || async {
            publish_rendered(&path, |staged| async move {
                tokio::fs::write(&staged, b"jp").await.unwrap();
                Err(ThumbnailError::GenerationFailed)
            })
            .await
        })
        .await;
        assert!(matches!(failed, Err(ThumbnailError::GenerationFailed)));
        assert!(!path.exists(), "a failed render published its half-image");
        assert!(
            !crate::durable::tmp_path(&path).exists(),
            "a failed render left its staging file behind"
        );

        let served = read_or_generate_thumbnail(&flight, &path, || async {
            publish_rendered(&path, |staged| async move {
                tokio::fs::write(&staged, b"jpeg").await.unwrap();
                Ok(())
            })
            .await
        })
        .await
        .unwrap();
        assert_eq!(served, b"jpeg");
    }

    #[tokio::test(start_paused = true)]
    async fn wedged_render_times_out_and_leaves_no_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poster.jpg");

        let err = publish_rendered(&path, |staged| async move {
            std::fs::write(&staged, b"jp").unwrap();
            std::future::pending::<()>().await;
            unreachable!("the timeout must cancel a render that never finishes")
        })
        .await
        .unwrap_err();

        assert!(matches!(err, ThumbnailError::GenerationFailed));
        assert!(
            !path.exists(),
            "a timed-out render published its half-image"
        );
        assert!(
            !crate::durable::tmp_path(&path).exists(),
            "a timed-out render left its staging file behind"
        );
    }

    #[tokio::test]
    async fn cached_thumbnail_skips_generation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poster.jpg");
        tokio::fs::write(&path, b"cached").await.unwrap();
        let flight = SingleFlight::<PathBuf>::new();

        let served = read_or_generate_thumbnail(&flight, &path, || async {
            panic!("cache hit must not render");
        })
        .await
        .unwrap();
        assert_eq!(served, b"cached");
    }

    #[tokio::test]
    async fn thumbnail_read_failure_after_generation_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("poster.jpg");
        let flight = SingleFlight::<PathBuf>::new();

        let err = read_or_generate_thumbnail(&flight, &path, || async { Ok(()) })
            .await
            .unwrap_err();
        assert!(matches!(err, ThumbnailError::ReadFailed));
    }
}
