//! Storage-backend abstraction over warm event storage.
//!
//! The warm writer and the playback API do not touch storage directly; they go
//! through a [`WarmStorageBackend`]. [`LocalDiskBackend`] lives here and owns
//! the on-disk layout, the atomic write ladder, crash recovery and lazy ffmpeg
//! thumbnailing that used to live in `buffer/warm.rs` and `api/server.rs`;
//! [`StathostBackend`](crate::storage::StathostBackend) is the remote one. What
//! they have in common — the in-RAM index and the retention skeletons over it —
//! is in [`event_index`](crate::storage::event_index), so a backend here is
//! object I/O and the policy that goes with it, nothing more.
//!
//! The trait is shaped so the two can differ where they must, without a caller
//! ever knowing which one it has:
//!
//! * an upgrade is expressed as an *intent* ([`upgrade_event`](WarmStorageBackend::upgrade_event))
//!   rather than "rename these paths" — LocalDisk moves files, the remote
//!   backend rewrites a sidecar in place (it has no rename);
//! * thumbnails are *acquired through the backend*
//!   ([`read_thumbnail`](WarmStorageBackend::read_thumbnail)) — LocalDisk keeps
//!   today's lazy ffmpeg generation + on-disk caching, single-flighted per
//!   thumbnail so a page of posters is one render each rather than one per
//!   request, the remote backend fetches a pre-rendered image;
//! * video is returned as a *stream*
//!   ([`read_video`](WarmStorageBackend::read_video)) — callers never see a
//!   `PathBuf` or a fully-buffered `Vec<u8>`; the body is an async byte stream
//!   with HTTP Range support, so a 10-60 MB event never lands whole in RAM.

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
    /// The write failed for any other reason (already logged) — including a
    /// commit that landed but could not be made durable, where the event is on
    /// disk and indexed yet not guaranteed to survive a power cut. Never
    /// retried by the writer.
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

impl ServedRange {
    /// A partial range, but only if `[start, end]` really is one of an object
    /// of `total` bytes: bounds in order and inside the object.
    ///
    /// The check belongs here rather than at each caller because what is on the
    /// other side of it is subtraction. A `Partial` is served with a
    /// `Content-Length` of `end - start + 1`, so a reversed pair underflows —
    /// a panic in a debug build, and camon ships debug — and an `end` past the
    /// object promises a body longer than the stream will ever produce, which
    /// hangs the player instead. Local disk resolves its own ranges and cannot
    /// produce either; the remote backend is repeating a `Content-Range` a
    /// server it does not control wrote, and that is not the same thing.
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
/// The in-RAM index that answers `query`/`find_event` and backs
/// `prune`/`emergency_prune` is not part of this contract: both backends own an
/// [`EventIndex`](crate::storage::event_index::EventIndex) and neither exposes
/// it.
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

    /// Delete events older than their per-class retention, bounded by the
    /// per-camera share one sweep may take (`cap_sweep_deletions`), so a
    /// forward clock jump cannot empty an archive in a single pass.
    ///
    /// `cancel` is the shutdown flag, and how finely it must be polled is part
    /// of this contract rather than a matter of taste. A sweep is long, and on
    /// a remote backend *one event* is several requests, each able to sit on a
    /// request timeout: polling only between events therefore leaves a whole
    /// event's worth of them inside the one that was in progress when the flag
    /// went up, which is more than the drain's phase 3 has to give (see
    /// [`crate::storage::contract`]'s third guarantee). So an implementation
    /// whose deletions are remote requests **must poll `cancel` between them**
    /// — this `cancel`, the one passed here, and not merely some flag of its
    /// own that production happens to alias to it — and may report
    /// [`Removal::Abandoned`](crate::storage::event_index::Removal) to say that
    /// it did: the entry stays indexed, unflagged and uncounted, and the pass
    /// ends. A backend that read only its own stop would keep this promise by
    /// coincidence and break it for any caller whose `cancel` is its own.
    ///
    /// What that costs is bounded by the *order* a backend deletes in, and
    /// choosing that order so a stop is survivable anywhere is the real
    /// obligation here. Stopping part-way must never be able to leave a video
    /// whose type record is gone — an event the next rebuild reads back as the
    /// wrong class, and expires on the wrong retention. Both backends arrange
    /// that, in opposite directions and for reasons of their own: local disk
    /// unlinks the metadata first and the `.ts` last, so the survivor of an
    /// interrupted delete is a bare `.ts` whose type is still its directory and
    /// which the next sweep expires again; the remote store deletes the video
    /// before the sidecar, so the survivor is metadata with no video, which
    /// indexes nothing and the next startup collects. Neither can produce a
    /// video that has lost the record of what it is.
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
    ///
    /// `Err` means the index does not describe the store — not that some events
    /// were skipped, which both backends report as they go, but that the
    /// backend never found out what is there. It is a `Result` rather than a
    /// log line because an index that was never built and an index of an empty
    /// store are the same object in RAM and the opposite instruction to
    /// retention, and only a caller holding the error can tell them apart. It
    /// is not fatal: startup continues, because a camera that cannot list its
    /// archive can still record into it.
    async fn scan(&self) -> std::io::Result<()>;

    /// Salvage writes interrupted by a crash or power cut, before the scan.
    fn recover_orphans(&self);

    /// The volume watch, for a backend that has a local filesystem to lose.
    ///
    /// `None` by default, which is the honest answer for a backend that owns no
    /// filesystem: nothing can be unmounted out from under a remote store, and
    /// a host that has gone away fails its uploads outright rather than letting
    /// them land somewhere else and report success. That silent redirection is
    /// the entire fault [`StorageAnchor`] exists to notice.
    fn volume_anchor(&self) -> Option<&Arc<StorageAnchor>> {
        None
    }

    // ---- API read path ----

    /// One page of the events overlapping the request's window, oldest first
    /// and never more than the page's limit (see [`EventPage`]).
    fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry>;

    /// The one indexed event this key names, if it is still there.
    ///
    /// The key is whole ([`EventRef`]) because a start PTS is not an identity:
    /// two events can begin on the same keyframe — a movement event and the
    /// continuous chunk covering it, or a run and the shorter chunk it was split
    /// from — and a lookup by start alone offered an arbitrary one of them for
    /// playback. What each backend does with the parts differs; see the impls.
    fn find_event(&self, camera_id: &str, event: EventRef) -> Option<WarmEventEntry>;

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
///
/// Owns the data directory and the in-RAM [`WarmEventIndex`]. This is the home
/// of everything filesystem-specific — the fsync/rename atomic ladder, statvfs
/// free-space checks, ffmpeg thumbnailing — that used to live inline in the
/// writer and the API handlers.
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
        // With the volume gone this whole guard is about the wrong disk: the
        // `create_dir_all` below would rebuild the storage tree on whatever is
        // behind the mountpoint, and statvfs would report that filesystem's
        // free space. See `emergency_prune` for why nothing is deleted either.
        if !self.anchor.is_intact() {
            return;
        }
        // data_dir may not exist before the first write; statvfs needs it.
        // Synced, because this runs before every write and would otherwise be
        // what creates the storage root: the event write's own synced
        // `create_dir_all` walks up only as far as the first directory that
        // already exists, so a root left durable-less here is a root the first
        // power cut can take away with everything under it.
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
        // Free space read through a path that no longer leads to the storage
        // volume is a figure about some other disk, and the response to a low
        // figure here is to delete footage. Deleting the archive to make room
        // on a filesystem the archive is not on is the one thing camon does in
        // this situation that cannot be undone afterwards, so it stops until
        // the volume is back. Writing carries on — footage in the wrong place
        // beats no footage, and it can be moved. The verdict is a cached flag,
        // so this costs no syscall on the write path, and it reads intact until
        // a check has actually proved otherwise.
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
        // Always `Ok`: this walk reports the directories and files it could not
        // read one by one and indexes the rest, and a data dir that is not there
        // at all is an empty store rather than an unknown one — nothing else
        // could have written to it either.
        //
        // Off the runtime, because the body of it is blocking filesystem work
        // that a failing disk can hold for a long time — see
        // [`WarmEventIndex::scan_off_thread`].
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

// ---------------------------------------------------------------------------
// Filesystem write mechanics (moved verbatim out of buffer/warm.rs). Kept as
// free functions with their original signatures so the writer's behavior tests
// exercise them exactly as before.
// ---------------------------------------------------------------------------

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

/// Atomically write a small metadata file (sidecar/thumbnail): stage as
/// `.tmp`, then rename.
///
/// The *contents* are deliberately not fsynced. The one contents fsync per
/// event is spent on the video, which is the thing that cannot be
/// reconstructed; a sidecar that a power cut leaves empty or torn costs the
/// event's detections and its `continues` flag, and the index scan already
/// falls back to a plain movement when a sidecar will not parse. Staging still
/// earns its keep: it keeps a *reader* from ever seeing a half-written file,
/// and any leftover `.tmp` is swept at startup.
///
/// The *rename* is made durable, but not here — these files live in the same
/// directory as the `.ts` they belong to, so the single directory fsync after
/// the commit rename covers their entries too.
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
        // The filesystem counts the sidecar and the thumbnails without being
        // told to, and `statvfs` is this backend's accounting authority — see
        // [`crate::storage::contract`]. A second figure maintained here could
        // only drift from the one that decides anything.
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
///
/// 1. video bytes → `{stem}.ts.tmp`, then fsync — the footage is durable and
///    recoverable (via startup orphan recovery) from this point on, before
///    anything else is risked;
/// 2. sidecar and thumbnails, each atomically under their final names;
/// 3. rename `{stem}.ts.tmp` → `{stem}.ts` — the commit point. The index scan
///    only ever looks at `.ts` files, so a crash at any earlier step leaves a
///    recoverable `.tmp` (plus adoptable metadata), never a half-indexed
///    event; a crash after the rename leaves a complete event;
/// 4. fsync the event directory, which is what makes step 3 survive a power
///    cut — the rename is atomic, but the directory entry naming the file is
///    not durable until the directory is synced. A failure here still leaves
///    the event on disk and indexed, but it is reported as `Failed`, not
///    `Written` (see [`commit_outcome`]).
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
///
/// Not `Written`: this trait promises to *durably* persist, `Written` is what
/// resets the camera's recording-silence watchdog, and a storage stack failing
/// every fsync would otherwise look perfectly healthy while nothing it wrote
/// was guaranteed to survive. Not `NoSpace` either, whatever errno says — that
/// outcome asks the writer for an emergency prune and a retry, and there is
/// nothing to retry when the event is already committed. `Failed` is not
/// retried by the writer, so reporting it honestly costs no rewrite and drops
/// no footage; it only stops a durability failure from being called a success.
fn commit_outcome(synced: std::io::Result<()>) -> WriteOutcome {
    match synced {
        Ok(()) => WriteOutcome::Written,
        Err(_) => WriteOutcome::Failed,
    }
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
/// 4. delete the old `movements/` sidecar, if any;
/// 5. fsync `objects/` and then `movements/`, so the entries the rename moved
///    between them are durable and a power cut cannot resurrect the event as a
///    movement — in that order and no further than the first failure, for the
///    reason [`sync_move`] gives.
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
        // A retention sweep that snapshotted this event as a movement,
        // then found `movements/{stem}.ts` already renamed away, drops the
        // entry as vanished. The footage is fine and now lives under
        // objects/, so re-index it here rather than leaving it invisible
        // until the next startup scan re-reads the directory.
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

/// Make a rename *between* two directories durable: fsync the destination,
/// then the source. The order is not a preference and the `?` is not a
/// shortcut — each step is only safe once the one before it succeeded:
///
/// * destination first, because it holds the only remaining name for the file.
///   Making the source's deletion durable while the destination entry is still
///   only in the page cache is the one combination that loses the event
///   outright: a power cut then takes away both names at once. So a failed
///   destination sync must not be followed by the source sync — hence `?`.
/// * source second, and its failure is survivable rather than fixable: the old
///   name can come back, leaving the same footage listed twice (the scan does
///   not deduplicate) — once as the upgraded object event and once as the
///   movement it was, which then expires on the shorter movement retention.
///   That is a visible duplicate and some wasted disk, which is the cheaper
///   half of the trade against losing the recording.
async fn sync_move(dest: &Path, src: &Path) -> std::io::Result<()> {
    crate::durable::sync_dir_async(dest).await?;
    crate::durable::sync_dir_async(src).await
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

/// Read the cached poster frame at `thumb_path`, calling `generate` to render
/// it if it is missing — once, however many callers miss it together.
///
/// The page-load pattern is N requests for the same not-yet-rendered thumbnail
/// arriving at once; without the guard each one spawns its own ffmpeg for the
/// identical output file. Waiters re-read the cache after the guard, so they
/// find the finished file instead of rendering it again, and the guard is keyed
/// on the thumbnail path so unrelated thumbnails never queue behind each other.
/// A failed render is not remembered: the guard is released, the key is dropped
/// and the next request is free to try again.
///
/// Split out from [`LocalDiskBackend::read_thumbnail`] so tests can count
/// renders without an ffmpeg on the machine.
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

/// How long one poster render may take before its ffmpeg is killed. The work
/// is decoding the *first* frame of a local file and scaling it to 320px — no
/// seek, no encode of any length — which is a fraction of a second even on an
/// SBC, so this is not a budget but a ceiling on a wedged process. It has to be
/// one, because the render holds the thumbnail's single-flight key: without it
/// one stuck ffmpeg parks every other request for that poster forever, where
/// before it only cost its own request.
const THUMBNAIL_RENDER_TIMEOUT: Duration = Duration::from_secs(15);

/// Render a single poster frame from `ts_path` into `thumb_path` with ffmpeg.
/// Local-only: a remote backend serves pre-rendered thumbnails instead.
async fn generate_thumbnail(ts_path: &Path, thumb_path: &Path) -> Result<(), ThumbnailError> {
    publish_rendered(thumb_path, |staged| render_poster_frame(ts_path, staged)).await
}

/// Run `render` against a staging path and publish its output to `thumb_path`
/// by rename, bounded by [`THUMBNAIL_RENDER_TIMEOUT`].
///
/// ffmpeg fills its output file in place, so pointing it at the live name
/// publishes the image progressively: the unguarded cache read in
/// [`read_or_generate_thumbnail`] can pick up a half-written frame and serve it
/// as the poster, and a render that is killed part-way — the timeout here, or a
/// disconnecting client dropping this future onto `kill_on_drop` — leaves a
/// truncated `.jpg` that every later request treats as a permanent cache hit.
/// Staging is what the rest of the storage layer already does for exactly this
/// reason ([`crate::durable`]); a reader here therefore sees either no file or
/// the whole image.
///
/// The staging file is removed on every failure this function observes. A
/// cancellation it never returns from can still leave one behind, which is why
/// the name is `{stem}.jpg.tmp` — one per thumbnail, overwritten by the next
/// attempt rather than accumulating, and swept by startup orphan recovery.
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
///
/// `-f image2` is what the staging name costs: ffmpeg picks its output format
/// from the extension, and `.jpg.tmp` is not one it knows. Naming the muxer
/// explicitly produces the same bytes the `.jpg` name used to.
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

    // ---- the shared storage contract --------------------------------------
    //
    // One assertion body, two backends: see `storage::contract::contract_tests`
    // for why these are written there and called here. Local disk is the
    // backend whose guarantees the remote one was supposed to reproduce, so it
    // is also the one whose passing them proves the assertions are about the
    // contract rather than about stathost.

    fn contract_backend() -> (tempfile::TempDir, LocalDiskBackend) {
        let dir = tempfile::tempdir().unwrap();
        let backend = LocalDiskBackend::new(dir.path().to_path_buf(), &["cam".to_string()]);
        (dir, backend)
    }

    /// Every byte under `dir`, which is what this backend's store *is*.
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
        let outcome = write_event(dir.path(), "cam", &event, &index).await;
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

    /// The video is written straight from the event's shared segments rather
    /// than from one buffer holding a copy of them all, so the file has to be
    /// pinned as their bytes in order: a chunk written twice, skipped or
    /// reordered would still produce a plausible `.ts` of the right size.
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

    /// Whether an fsync reached the platter is not observable from a test, and
    /// neither is a lost directory entry without cutting real power. What is
    /// observable is that the durable variants used on the commit path behave
    /// like the plain ones did: a first-ever write, where the whole tree has to
    /// be created and synced from `data_dir` down, still commits.
    ///
    /// Driven through the backend in the order the writer really uses — the
    /// low-space guard first, which is what creates `data_dir` on a first run,
    /// then the write. Calling `write_event` alone would skip the guard and so
    /// skip the directory that matters most.
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

    /// The policy a real fsync failure would exercise, which no test can
    /// provoke: the directory is openable whenever the rename into it just
    /// succeeded, and a permission trick would be a no-op for a root test
    /// runner. What is pinned is the decision itself — a commit that could not
    /// be made durable is not a successful write, because `Written` is what
    /// resets the recording-silence watchdog (`buffer::warm`), and `Failed`
    /// rather than `NoSpace` because only `NoSpace` is retried.
    #[test]
    fn a_commit_that_cannot_be_synced_is_not_reported_as_written() {
        assert_eq!(commit_outcome(Ok(())), WriteOutcome::Written);
        assert_eq!(
            commit_outcome(Err(std::io::Error::other("fsync failed"))),
            WriteOutcome::Failed
        );
    }

    /// The `?` in `sync_move` is load-bearing: syncing the source directory
    /// after the destination sync failed makes the *removal* of the old name
    /// durable while the new name is not, which loses the event entirely. That
    /// the source is skipped is enforced by the type, not observable here; what
    /// a test can pin is that a destination that cannot be synced ends the
    /// sequence with an error instead of being logged past.
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
        // A source that cannot be synced is reported too, but only after the
        // destination is durable — a duplicate movement entry, not a lost event.
        assert!(sync_move(&objects, &dir.path().join("gone")).await.is_err());
    }

    /// A regular file where the camera directory belongs: the directory can
    /// never be created, whatever user the test runs as. The error has to reach
    /// the caller as a failed write rather than being swallowed by the sync.
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
        write_event(dir.path(), "cam", &event, &index).await;

        let duration_ms = (7 - 5 + 1) * 1000;
        let stem = format!("{}_{}", first_pts, duration_ms);
        let movements = dir.path().join("cam").join("movements");
        // A movement chunk that continues DOES get a sidecar, carrying the flag.
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
        let entry = index
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Object))
            .unwrap();
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

    /// One camera's events, newest information first, reached by duration alone
    /// — which is how a test names a member of a run of equal starts whose
    /// *type* is the thing under test and so cannot be part of the lookup.
    fn sibling(index: &WarmEventIndex, duration_ms: u32) -> Option<WarmEventEntry> {
        index
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .into_iter()
            .find(|e| e.duration_ms == duration_ms)
    }

    /// Two events sharing a start PTS are two events. The upgrade rewrites the
    /// one it named — not whichever of the pair a binary search on the start
    /// returns — and the sweep afterwards deletes the one *it* names. The
    /// remote backend has had this test; the local index reached in by start
    /// alone, so an upgrade could flip a sibling to `Object` while the event
    /// actually upgraded stayed a movement pointing at a path the rename had
    /// moved away from.
    #[tokio::test]
    async fn siblings_sharing_a_start_pts_are_upgraded_and_pruned_by_full_key() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        // Same first segment, so the same start PTS; different lengths, so two
        // distinct files and two distinct keys.
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

        // Upgrade one named sibling. Which of the two it is does not matter now
        // that every path into the index carries a whole key; the start-only
        // lookup this replaced returned an unspecified member of the run, so
        // whichever one was named, the other could be the one rewritten.
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

        // The movement sibling expires; the object one does not.
        index.prune(1, u64::MAX, u64::MAX, || false).await;
        assert!(sibling(&index, spared_ms).is_none());
        assert!(sibling(&index, target_ms).is_some());
        assert!(!movements
            .join(format!("{first_pts}_{spared_ms}.ts"))
            .exists());
        assert!(objects.join(format!("{first_pts}_{target_ms}.ts")).exists());
    }

    /// The read path, on the same pair: each sibling is served as itself.
    ///
    /// Two stored events share this start — one movement, one continuous chunk
    /// of the same length, which is what a camera recording continuously while
    /// something moves in front of it produces — so the URLs asking for them
    /// differ only in the type. The lookup this replaced binary-searched the
    /// start and could only ever answer with one of the two, whichever std
    /// landed on: a request for the movement served the continuous chunk's
    /// video, duration and poster frame, or the other way round. Nothing about
    /// it looked like an error.
    ///
    /// It also pins the paths those bytes come from apart. The poster frame is
    /// rendered lazily and single-flighted on its output path
    /// ([`LocalDiskBackend::thumbnails`]), so two events resolving to one
    /// thumbnail path would share a render and serve one picture for both.
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
            // And the objects behind it are that event's own: the same stem
            // under its own tier directory.
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
        // The single-flight key for a lazily rendered poster frame is its output
        // path, so the pair must not share one.
        assert_ne!(posters[0], posters[1]);

        // A key nothing is stored under is absent rather than nearly-matching:
        // the same start and type with another duration, and the third type.
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

    /// Local sidecars carry no `event_type`: the directory an event lives in is
    /// what says what it is, and a second answer in the JSON could only drift
    /// from it. Both local writers pass `None` — the fresh write and the
    /// upgrade — and every other assertion here parses the sidecar for the
    /// fields it wants, so a `Some` slipping into either would go unseen.
    #[tokio::test]
    async fn local_sidecars_name_no_event_type() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = populated_buffer(10);
        // A continuing movement, so the fresh write produces a sidecar at all.
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

        // A fresh scan of the directory sees an object event with the
        // continues flag preserved; no stale movement sidecar remains.
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

        // Files on disk, deliberately absent from the index under test:
        // written through a throwaway index, leaving exactly the state a
        // racing sweep leaves behind when it unindexes as vanished.
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

    /// A `min_free_bytes` of `u64::MAX` makes every filesystem look full, which
    /// is exactly what an unmounted `data_dir` does to the guard: it reads a
    /// device that has nothing to do with the archive. Deleting real footage on
    /// the strength of that reading is the one move here with no way back, so
    /// while the anchor says the volume moved, nothing is deleted — and the
    /// same call with the volume in place still deletes, so the veto is what
    /// stopped it and not the setup.
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
        // Never written (or already pruned): nothing happens, nothing panics.
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

        // First chunk after startup: continues == false.
        let first = {
            let buf = buffer.read_recover();
            assemble_continuous_chunk(&buf, "cam", 0, 4, false).unwrap()
        };
        let first_pts = first.first_pts;
        write_event(dir.path(), "cam", &first, &index).await;

        // Follow-on chunk: continues == true.
        let second = {
            let buf = buffer.read_recover();
            assemble_continuous_chunk(&buf, "cam", 5, 9, true).unwrap()
        };
        let second_pts = second.first_pts;
        write_event(dir.path(), "cam", &second, &index).await;

        let continuous = dir.path().join("cam").join("continuous");
        // Both chunks routed to continuous/.
        assert!(continuous.join(format!("{}_5000.ts", first_pts)).exists());
        assert!(continuous.join(format!("{}_5000.ts", second_pts)).exists());
        // First chunk: no sidecar (nothing to persist). Follow-on: continues sidecar.
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

        // Write a first + follow-on continuous chunk with the real writer.
        let writer_index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        for (start, last, continues) in [(0u64, 4u64, false), (5, 9, true)] {
            let event = {
                let buf = buffer.read_recover();
                assemble_continuous_chunk(&buf, "cam", start, last, continues).unwrap()
            };
            write_event(dir.path(), "cam", &event, &writer_index).await;
        }

        // A fresh index scanning the same dir must recover type + continues.
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

        // Video comes back as a stream, not a path or a Vec.
        let video = drain(backend.read_video("cam", &entry, None).await.unwrap()).await;
        assert_eq!(video.len(), 16);

        // Filmstrip frames are readable through the backend.
        assert!(backend.read_filmstrip("cam", &entry, 0).await.is_ok());
        assert!(backend.read_filmstrip("cam", &entry, 1).await.is_ok());

        // Upgrade intent reclassifies the event to Object.
        backend.upgrade_event("cam", &upgrade_for(&event)).await;
        let upgraded = backend
            .find_event("cam", EventRef::new(first_pts, 4000, EventType::Object))
            .unwrap();
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

    // --- lazy thumbnail generation -----------------------------------------
    //
    // Against `read_or_generate_thumbnail` rather than `read_thumbnail`, so a
    // render can be counted and made slow without an ffmpeg on the machine.

    /// Counts renders and the renders running at once, and refuses to finish
    /// until the gate opens — so a missing guard shows up as overlap, not as a
    /// race the scheduler happens to serialize.
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
            // Publish the way production does — staged, then renamed — so a
            // request whose unguarded cache read lands mid-render sees no file
            // rather than an empty or half-written one.
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
        // All eight requests leave the line together; the file cannot exist
        // before the gate opens, so every one of them takes the miss path.
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
        // Both renders only clear the barrier if they run at the same time; a
        // guard shared across thumbnails would deadlock into the timeout.
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

        // Nothing was remembered: the next request renders and is served.
        let served = read_or_generate_thumbnail(&flight, &path, || async {
            tokio::fs::write(&path, b"jpeg").await.unwrap();
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(served, b"jpeg");
    }

    /// A render that dies part-way through its output must not poison the
    /// cache: the half-written bytes stay on the staging path, never the live
    /// one, so no request ever reads them and the next request renders afresh.
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

        // The truncated attempt was not remembered as a cache hit.
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

    /// A wedged render is killed at the timeout instead of holding its
    /// single-flight key forever, and whatever it had written is cleaned up.
    ///
    /// Paused time: the runtime jumps the clock when nothing is runnable, so
    /// the 15-second ceiling elapses instantly. The partial write is done with
    /// the blocking API to keep the pause deterministic — an async write could
    /// still be in flight on the blocking pool when the clock jumps, and land
    /// after the cleanup it is supposed to precede.
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

        // Generation "succeeds" without leaving a file behind — unchanged
        // behaviour: the read afterwards fails and that is what is reported.
        let err = read_or_generate_thumbnail(&flight, &path, || async { Ok(()) })
            .await
            .unwrap_err();
        assert!(matches!(err, ThumbnailError::ReadFailed));
    }
}
