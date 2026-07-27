//! Remote "stathost" warm-storage backend.
//!
//! [`stathost`](https://github.com/nsg/stathost) is a small static file host:
//!
//! * `PUT /{bucket}/{path}` and `DELETE /{bucket}/{path}` — authenticated with
//!   `Authorization: Bearer <token>`;
//! * `GET /{bucket}/{path}` — public;
//! * `GET /{bucket}/_meta/list?detail=true` — authenticated, returns
//!   `[{"path","size","mtime"}]`.
//!
//! Requires stathost **0.2.0 or later** (atomic uploads, detailed listing,
//! Range requests). There is deliberately no fallback for older servers.
//!
//! This backend slots in behind [`WarmStorageBackend`] with no on-disk layout:
//! every event is three sibling objects under `{camera_id}/` —
//! `{start_pts_ns}_{duration_ms}.ts`, a `.json` sidecar, and eager
//! `{stem}_thumb_{i}.jpg` filmstrip frames. **The event type lives in the
//! sidecar** (`"event_type"`), not in a directory, so a sidecar is *always*
//! uploaded (unlike local mode, where it is conditional).
//!
//! Notable divergences from [`LocalDiskBackend`], all deliberate:
//!
//! * **Retention-by-space is a client-side budget.** The client can't see the
//!   server's disk, so `max_stored_bytes` caps tracked usage; when it is
//!   exceeded the oldest events are pruned (continuous → movements → objects),
//!   mirroring the local emergency prune order. The disk-shaped
//!   `min_free_bytes` guard argument is ignored here.
//! * **Thumbnails are eager.** Filmstrip frame 0 doubles as the poster; there
//!   is no ffmpeg on remote bytes. [`read_thumbnail`] fetches `thumb_0` or
//!   returns a clear error when the event has no frames.
//! * **Interrupted-upload hygiene.** The `.ts` is uploaded first, sidecar and
//!   thumbs after, so a crash mid-sequence leaves at worst a `.ts` without a
//!   sidecar — indexed as a plain movement event on the next scan. stathost's
//!   uploads are atomic server-side, so a truncated object can't be served; a
//!   zero-byte `.ts` in the listing still gets a warning at scan time.
//! * **`read_video` streams the object with Range support.** The body is never
//!   fully buffered; a forwarded `Range` header yields a `206`. A `200` to a
//!   range request is legal HTTP and degrades to streaming the full body (the
//!   client replays from the start).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::TryStreamExt;
use serde::Deserialize;

use crate::buffer::warm::{EventUpgrade, FinishedEvent};
use crate::buffer::GopSegment;
use crate::config::StathostConfig;
use crate::locks::LockExt;
use crate::storage::backend::{
    deduplicate_detections, RangeRequest, ServedRange, ThumbnailError, VideoStream,
    WarmStorageBackend, WriteOutcome,
};
use crate::storage::warm_index::{parse_event_filename, parse_sidecar_json, DetectionDetail};
use crate::storage::{EventType, WarmEventEntry};

const NANOS_PER_MS: u64 = 1_000_000;

/// Writes, deletes and the scan are awaited inline by the serial per-camera
/// warm writer, so an unbounded request stalls that camera's recording and its
/// shutdown. Both clients bound the connect phase; the rest differs by call
/// shape.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Total per-request ceiling for the non-streaming calls: delete, sidecar and
/// thumbnail GETs, and the whole-bucket listing — which is not small, a busy
/// bucket lists thousands of entries at startup.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Total per-request ceiling for uploads: event videos reach tens of MB, which
/// [`REQUEST_TIMEOUT`] would drop on a slow uplink. This bounds one request,
/// not one event — `write_event` retries the video once, then uploads a sidecar
/// and each thumbnail, so a half-broken link can hold the warm writer for a
/// multiple of this.
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(300);
/// Idle budget for the streaming client only. reqwest arms it flat until the
/// response headers arrive, then per response frame with a reset on each — the
/// right shape for a body a player drains at its own pace. The flat phase is
/// harmless here because a ranged GET carries no request body.
const STREAM_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// The remote warm store: an HTTP client plus a self-maintained in-RAM index.
pub struct StathostBackend {
    http: Http,
    /// Client-side storage budget in bytes; 0 means unlimited.
    max_stored_bytes: u64,
    /// Per-camera event lists, each sorted by `start_pts_ns` (the query/find
    /// key). Kept coherent on write/upgrade/prune.
    cameras: HashMap<String, RwLock<Vec<WarmEventEntry>>>,
    /// Sum of indexed `file_size` — the figure the budget is measured against.
    used_bytes: AtomicU64,
}

impl StathostBackend {
    pub fn new(config: &StathostConfig, camera_ids: &[String]) -> Self {
        let base = format!(
            "{}/{}",
            config.url.trim_end_matches('/'),
            config.bucket.trim_matches('/')
        );
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(Vec::new()));
        }
        Self {
            http: Http {
                client: reqwest::Client::builder()
                    .connect_timeout(CONNECT_TIMEOUT)
                    .build()
                    .expect("stathost http client"),
                stream_client: reqwest::Client::builder()
                    .connect_timeout(CONNECT_TIMEOUT)
                    .read_timeout(STREAM_READ_TIMEOUT)
                    .build()
                    .expect("stathost streaming http client"),
                base,
                token: config.token.clone(),
            },
            max_stored_bytes: config.max_stored_bytes,
            cameras,
            used_bytes: AtomicU64::new(0),
        }
    }

    fn used(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    // ---- in-RAM index (self-contained; mirrors WarmEventIndex's RAM half) ----

    fn insert_entry(&self, camera_id: &str, entry: WarmEventEntry) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write_recover();
            let pos = entries
                .binary_search_by_key(&entry.start_pts_ns, |e| e.start_pts_ns)
                .unwrap_or_else(|p| p);
            entries.insert(pos, entry);
        }
    }

    /// Remove one event from the index by its start PTS, returning the removed
    /// entry so the caller can reconcile `used_bytes`.
    fn remove_entry(&self, camera_id: &str, start_pts_ns: u64) -> Option<WarmEventEntry> {
        let lock = self.cameras.get(camera_id)?;
        let mut entries = lock.write_recover();
        let idx = entries
            .binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns)
            .ok()?;
        Some(entries.remove(idx))
    }

    // ---- object-store helpers ----

    fn ts_key(camera_id: &str, entry: &WarmEventEntry) -> String {
        format!(
            "{camera_id}/{}_{}.ts",
            entry.start_pts_ns, entry.duration_ms
        )
    }

    /// Delete every object belonging to one event. Returns whether the video
    /// object is gone (deleted or already absent) — the signal for whether the
    /// index entry may be dropped. A genuine transport failure returns `false`
    /// so the entry survives for the next prune tick.
    async fn delete_event_objects(&self, camera_id: &str, entry: &WarmEventEntry) -> bool {
        let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
        let ts_gone = match self.http.delete(&format!("{camera_id}/{stem}.ts")).await {
            DeleteOutcome::Deleted | DeleteOutcome::Missing => true,
            DeleteOutcome::Failed => {
                tracing::warn!(camera = %camera_id, stem = %stem,
                    "failed to delete event video from stathost, will retry next prune tick");
                false
            }
        };
        if !ts_gone {
            return false;
        }
        // Best-effort cleanup of the siblings; a lingering sidecar/thumb is
        // harmless once the video is gone.
        let _ = self.http.delete(&format!("{camera_id}/{stem}.json")).await;
        for i in 0..entry.filmstrip_frames {
            let _ = self
                .http
                .delete(&format!("{camera_id}/{stem}_thumb_{i}.jpg"))
                .await;
        }
        true
    }

    /// Enforce the client-side storage budget: while tracked usage exceeds
    /// `max_stored_bytes`, delete the oldest events cheapest-tier-first
    /// (continuous → movements → objects). A transport failure stops the pass
    /// (retried on the next tick). No-op when the budget is unlimited.
    async fn enforce_budget(&self, camera_id: &str) {
        if self.max_stored_bytes == 0 || self.used() <= self.max_stored_bytes {
            return;
        }
        let mut deleted = 0u64;
        for tier in [
            EventType::Continuous,
            EventType::Movement,
            EventType::Object,
        ] {
            let mut candidates: Vec<(String, WarmEventEntry)> = Vec::new();
            for (cam, lock) in self.cameras.iter() {
                let entries = lock.read_recover();
                candidates.extend(
                    entries
                        .iter()
                        .filter(|e| e.event_type == tier)
                        .cloned()
                        .map(|e| (cam.clone(), e)),
                );
            }
            candidates.sort_by_key(|(_, e)| e.start_pts_ns);

            for (cam, entry) in candidates {
                if self.used() <= self.max_stored_bytes {
                    if deleted > 0 {
                        tracing::warn!(camera = %camera_id, deleted, "budget prune complete");
                    }
                    return;
                }
                if !self.delete_event_objects(&cam, &entry).await {
                    // Can't reclaim right now; leave the rest for the next tick.
                    return;
                }
                if let Some(removed) = self.remove_entry(&cam, entry.start_pts_ns) {
                    self.used_bytes
                        .fetch_sub(removed.file_size, Ordering::Relaxed);
                }
                deleted += 1;
                tracing::warn!(
                    camera = %cam,
                    start_pts_ns = entry.start_pts_ns,
                    event_type = ?entry.event_type,
                    "budget prune: deleted event to stay under max_stored_bytes"
                );
            }
        }
        if deleted > 0 {
            tracing::warn!(camera = %camera_id, deleted, "budget prune complete");
        }
    }
}

#[async_trait]
impl WarmStorageBackend for StathostBackend {
    async fn write_event(&self, camera_id: &str, event: &FinishedEvent) -> WriteOutcome {
        let duration_ms = event.duration_ns() / NANOS_PER_MS;
        let stem = format!("{}_{}", event.first_pts, duration_ms);
        let data = concatenate_segments(&event.segments, event.total_bytes);
        let file_size = data.len() as u64;
        let event_type = event.event_type();

        // Step 1: the video, first — one retry, then drop (logged) so a failed
        // write is never lost silently.
        let ts_key = format!("{camera_id}/{stem}.ts");
        if self.http.put(&ts_key, data.clone()).await.is_err() {
            tracing::warn!(camera = %camera_id, stem = %stem,
                "stathost video upload failed, retrying once");
            if self.http.put(&ts_key, data).await.is_err() {
                tracing::error!(
                    camera = %camera_id,
                    first_pts = event.first_pts,
                    bytes = event.total_bytes,
                    "dropping event: stathost video upload failed after retry"
                );
                return WriteOutcome::Failed;
            }
        }

        // Step 2: the sidecar — ALWAYS, since it is the sole carrier of the
        // event type and detections. Non-fatal on failure (video wins; the
        // scan indexes a sidecar-less .ts as a plain movement event).
        let sidecar = sidecar_json(
            event_type,
            event.backend.as_deref(),
            event.model.as_deref(),
            &event.detection_details,
            event.continues,
        );
        if self
            .http
            .put(&format!("{camera_id}/{stem}.json"), sidecar.into_bytes())
            .await
            .is_err()
        {
            tracing::warn!(camera = %camera_id, stem = %stem,
                "failed to upload event sidecar to stathost");
        }

        // Step 3: eager filmstrip thumbnails; frame 0 doubles as the poster.
        let filmstrip_frames = match &event.filmstrip_frames {
            Some(frames) => {
                let mut wrote = 0;
                for (i, jpeg) in frames.iter().enumerate() {
                    let key = format!("{camera_id}/{stem}_thumb_{i}.jpg");
                    if self.http.put(&key, jpeg.clone()).await.is_err() {
                        tracing::warn!(camera = %camera_id, stem = %stem,
                            "failed to upload filmstrip thumbnail to stathost");
                    } else {
                        wrote += 1;
                    }
                }
                wrote
            }
            None => 0,
        };

        self.insert_entry(
            camera_id,
            WarmEventEntry {
                start_pts_ns: event.first_pts,
                duration_ms: duration_ms as u32,
                event_type,
                file_size,
                object_classes: event.object_classes.clone(),
                backend: event.backend.clone(),
                model: event.model.clone(),
                detections: event.detection_details.clone(),
                filmstrip_frames,
                continues: event.continues,
                recovered: false,
            },
        );
        self.used_bytes.fetch_add(file_size, Ordering::Relaxed);

        tracing::info!(
            camera = %camera_id,
            stem = %stem,
            bytes = event.total_bytes,
            duration_ms = duration_ms,
            "wrote warm event to stathost"
        );
        WriteOutcome::Written
    }

    async fn upgrade_event(&self, camera_id: &str, upgrade: &EventUpgrade) {
        // The upgrade rewrites the sidecar in place — no video is moved. If the
        // event was never indexed (write failed, or already pruned) there is
        // nothing to rewrite; the detections remain in the detection store.
        if self.find_event(camera_id, upgrade.start_pts_ns).is_none() {
            tracing::warn!(
                camera = %camera_id,
                start_pts_ns = upgrade.start_pts_ns,
                "event not indexed, skipping object upgrade \
                 (detections remain available in the detection store)"
            );
            return;
        }
        let stem = format!("{}_{}", upgrade.start_pts_ns, upgrade.duration_ms);
        let sidecar = sidecar_json(
            EventType::Object,
            Some(&upgrade.backend),
            Some(&upgrade.model),
            &upgrade.detections,
            upgrade.continues,
        );
        if self
            .http
            .put(&format!("{camera_id}/{stem}.json"), sidecar.into_bytes())
            .await
            .is_err()
        {
            tracing::error!(camera = %camera_id, stem = %stem,
                "failed to upload upgraded sidecar to stathost, aborting upgrade");
            return;
        }

        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write_recover();
            if let Ok(i) = entries.binary_search_by_key(&upgrade.start_pts_ns, |e| e.start_pts_ns) {
                let entry = &mut entries[i];
                entry.event_type = EventType::Object;
                entry.object_classes = upgrade.object_classes.clone();
                entry.detections = upgrade.detections.clone();
                entry.backend = Some(upgrade.backend.clone());
                entry.model = Some(upgrade.model.clone());
            }
        }

        tracing::info!(
            camera = %camera_id,
            stem = %stem,
            classes = ?upgrade.object_classes,
            "upgraded movement event to object event on stathost"
        );
    }

    async fn prune(
        &self,
        movement_max_age_ns: u64,
        object_max_age_ns: u64,
        continuous_max_age_ns: u64,
    ) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let max_age = |t: EventType| match t {
            EventType::Movement => movement_max_age_ns,
            EventType::Object => object_max_age_ns,
            EventType::Continuous => continuous_max_age_ns,
        };

        for (camera_id, lock) in self.cameras.iter() {
            let expired: Vec<WarmEventEntry> = {
                let entries = lock.read_recover();
                entries
                    .iter()
                    .filter(|e| now_ns.saturating_sub(e.start_pts_ns) > max_age(e.event_type))
                    .cloned()
                    .collect()
            };
            if expired.is_empty() {
                continue;
            }

            let mut deleted = 0u64;
            for entry in &expired {
                if self.delete_event_objects(camera_id, entry).await {
                    if let Some(removed) = self.remove_entry(camera_id, entry.start_pts_ns) {
                        self.used_bytes
                            .fetch_sub(removed.file_size, Ordering::Relaxed);
                    }
                    deleted += 1;
                }
            }
            if deleted > 0 {
                tracing::info!(camera = %camera_id, deleted, "pruned expired warm events");
            }
        }
    }

    async fn guard_free_space(&self, camera_id: &str, _min_free_bytes: u64) {
        // The disk-shaped `min_free_bytes` guard does not apply to a remote
        // host; the client-side budget is the authority.
        self.enforce_budget(camera_id).await;
    }

    async fn emergency_prune(&self, camera_id: &str, _min_free_bytes: u64) {
        self.enforce_budget(camera_id).await;
    }

    fn free_space(&self) -> std::io::Result<u64> {
        // "Free space" is the remaining client-side budget; unlimited budgets
        // report the max so the guard never fires.
        if self.max_stored_bytes == 0 {
            Ok(u64::MAX)
        } else {
            Ok(self.max_stored_bytes.saturating_sub(self.used()))
        }
    }

    async fn scan(&self) {
        let start = std::time::Instant::now();
        let items = match self.http.list().await {
            Ok(items) => items,
            Err(e) => {
                tracing::error!(error = %e, "stathost list failed; warm index left empty");
                return;
            }
        };

        // Full path set, so filmstrip frames can be counted without extra GETs.
        let all_paths: HashSet<&str> = items.iter().map(|i| i.path.as_str()).collect();
        let mut total = 0usize;

        for item in &items {
            let Some((camera_id, stem)) = split_ts_key(&item.path) else {
                continue;
            };
            if !self.cameras.contains_key(camera_id) {
                continue;
            }
            let Some((start_pts_ns, duration_ms)) = parse_event_filename(stem) else {
                tracing::warn!(path = %item.path, "skipping stathost object with unparsable name");
                continue;
            };
            if item.size == 0 {
                tracing::warn!(path = %item.path,
                    "zero-byte .ts on stathost (interrupted upload?)");
            }

            // Fetch the sibling sidecar; a missing/garbled one → movement event.
            let sidecar = match self.http.get(&format!("{camera_id}/{stem}.json")).await {
                Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
                    .ok()
                    .map(|v| parse_sidecar_json(&v)),
                Err(_) => None,
            };
            let event_type = sidecar
                .as_ref()
                .and_then(|s| s.event_type)
                .unwrap_or(EventType::Movement);

            let mut filmstrip_frames = 0usize;
            while all_paths
                .contains(format!("{camera_id}/{stem}_thumb_{filmstrip_frames}.jpg").as_str())
            {
                filmstrip_frames += 1;
            }

            let entry = WarmEventEntry {
                start_pts_ns,
                duration_ms,
                event_type,
                file_size: item.size,
                object_classes: sidecar
                    .as_ref()
                    .map(|s| s.classes.clone())
                    .unwrap_or_default(),
                backend: sidecar.as_ref().and_then(|s| s.backend.clone()),
                model: sidecar.as_ref().and_then(|s| s.model.clone()),
                detections: sidecar
                    .as_ref()
                    .map(|s| s.detections.clone())
                    .unwrap_or_default(),
                filmstrip_frames,
                continues: sidecar.as_ref().map(|s| s.continues).unwrap_or(false),
                recovered: sidecar.as_ref().map(|s| s.recovered).unwrap_or(false),
            };
            self.used_bytes.fetch_add(item.size, Ordering::Relaxed);
            self.insert_entry(camera_id, entry);
            total += 1;
        }

        tracing::info!(
            total_events = total,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "stathost warm index scan complete"
        );
    }

    fn recover_orphans(&self) {
        // Interrupted uploads are a server-side concern; nothing to salvage
        // client-side.
    }

    /// Every event overlapping `[from_ns, to_ns]`. An inverted range is empty.
    ///
    /// Entries are ordered by start PTS only, so the upper bound binary-searches
    /// but the lower one cannot: a long event (a continuous chunk) can start
    /// far before the window and still reach into it, and "ends after `from_ns`"
    /// is not monotone in start order. The candidate prefix is filtered instead.
    fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        if from_ns > to_ns {
            return Vec::new();
        }
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
                let end = entries.partition_point(|e| e.start_pts_ns <= to_ns);
                entries[..end]
                    .iter()
                    .filter(|e| {
                        e.start_pts_ns
                            .saturating_add((e.duration_ms as u64) * NANOS_PER_MS)
                            >= from_ns
                    })
                    .cloned()
                    .collect()
            }
            None => Vec::new(),
        }
    }

    fn find_event(&self, camera_id: &str, start_pts_ns: u64) -> Option<WarmEventEntry> {
        let lock = self.cameras.get(camera_id)?;
        let entries = lock.read_recover();
        entries
            .binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns)
            .ok()
            .map(|i| entries[i].clone())
    }

    async fn read_video(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        range: Option<RangeRequest>,
    ) -> std::io::Result<VideoStream> {
        let key = Self::ts_key(camera_id, entry);
        let resp = self
            .http
            .get_ranged(&key, range)
            .await
            .map_err(reqwest_io)?;
        let status = resp.status();

        // The server understood the range and declined it.
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let total = resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range_total)
                .unwrap_or(entry.file_size);
            return Ok(VideoStream {
                stream: Box::pin(futures_util::stream::empty()),
                total_size: total,
                range: ServedRange::Unsatisfiable,
            });
        }

        if let Err(e) = resp.error_for_status_ref() {
            return Err(reqwest_io(e));
        }

        // 206 → a satisfied partial range; anything else (200, or a server that
        // ignored the header) degrades to streaming the full body.
        let (served, total_size) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            match resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range)
            {
                Some((start, end, total)) => (ServedRange::Partial { start, end }, total),
                // 206 without a parseable Content-Range: treat the body as full.
                None => (
                    ServedRange::Full,
                    resp.content_length().unwrap_or(entry.file_size),
                ),
            }
        } else {
            (
                ServedRange::Full,
                resp.content_length().unwrap_or(entry.file_size),
            )
        };

        let stream = resp.bytes_stream().map_err(reqwest_io);
        Ok(VideoStream {
            stream: Box::pin(stream),
            total_size,
            range: served,
        })
    }

    async fn read_thumbnail(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
    ) -> Result<Vec<u8>, ThumbnailError> {
        // Thumbnails are eager: frame 0 is the poster. Never run ffmpeg on
        // remote bytes — if there is no filmstrip, there is no thumbnail.
        if entry.filmstrip_frames == 0 {
            return Err(ThumbnailError::GenerationFailed);
        }
        let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
        self.http
            .get(&format!("{camera_id}/{stem}_thumb_0.jpg"))
            .await
            .map_err(|_| ThumbnailError::ReadFailed)
    }

    async fn read_filmstrip(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        index: u8,
    ) -> std::io::Result<Vec<u8>> {
        let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
        self.http
            .get(&format!("{camera_id}/{stem}_thumb_{index}.jpg"))
            .await
            .map_err(reqwest_io)
    }
}

// ---------------------------------------------------------------------------
// HTTP object-store client
// ---------------------------------------------------------------------------

/// Thin reqwest wrapper over the stathost object API. `base` is
/// `{url}/{bucket}` with no trailing slash.
struct Http {
    /// Every call that completes within one request/response, bounded by a
    /// per-request total timeout.
    client: reqwest::Client,
    /// Playback only. A total timeout would cut long streams short, so this one
    /// carries [`STREAM_READ_TIMEOUT`] instead — which in turn would be wrong
    /// for the uploads on `client`, where reqwest counts the whole request body
    /// write against it.
    stream_client: reqwest::Client,
    base: String,
    token: String,
}

enum DeleteOutcome {
    Deleted,
    /// The object was already absent (404) — treated as success for pruning.
    Missing,
    /// Transport or server error — the caller should retry later.
    Failed,
}

impl Http {
    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base, path)
    }

    async fn put(&self, path: &str, body: Vec<u8>) -> Result<(), reqwest::Error> {
        self.client
            .put(self.url(path))
            .bearer_auth(&self.token)
            .timeout(UPLOAD_TIMEOUT)
            .body(body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    async fn delete(&self, path: &str) -> DeleteOutcome {
        match self
            .client
            .delete(self.url(path))
            .bearer_auth(&self.token)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
        {
            Ok(resp) => {
                if resp.status() == reqwest::StatusCode::NOT_FOUND {
                    DeleteOutcome::Missing
                } else if resp.error_for_status().is_ok() {
                    DeleteOutcome::Deleted
                } else {
                    DeleteOutcome::Failed
                }
            }
            Err(_) => DeleteOutcome::Failed,
        }
    }

    /// Fetch an object's bytes. Public route, but the bearer is harmless.
    async fn get(&self, path: &str) -> Result<Vec<u8>, reqwest::Error> {
        let resp = self
            .client
            .get(self.url(path))
            .bearer_auth(&self.token)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.bytes().await?.to_vec())
    }

    /// Start a streamed GET, optionally forwarding a single `Range`. The raw
    /// response is returned unvalidated so the caller can distinguish `206`
    /// (partial), `200` (full / range-ignored) and `416` (unsatisfiable).
    ///
    /// Uses `stream_client`, deliberately without a total timeout: the body is
    /// handed to a player that drains it at its own pace, so only the connect
    /// and per-frame idle budgets apply.
    async fn get_ranged(
        &self,
        path: &str,
        range: Option<RangeRequest>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let mut req = self
            .stream_client
            .get(self.url(path))
            .bearer_auth(&self.token);
        if let Some(range) = range {
            req = req.header(reqwest::header::RANGE, range.header_value());
        }
        req.send().await
    }

    /// List every object in the bucket via the detailed listing
    /// (stathost >= 0.2.0). No fallback: an unexpected response shape is an
    /// error, surfaced by the caller.
    async fn list(&self) -> Result<Vec<ListEntry>, reqwest::Error> {
        self.client
            .get(format!("{}/_meta/list?detail=true", self.base))
            .bearer_auth(&self.token)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
    }
}

/// A `_meta/list?detail=true` entry. Extra fields (e.g. `mtime`) are ignored;
/// event time comes from the filename.
#[derive(Deserialize)]
struct ListEntry {
    path: String,
    size: u64,
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn concatenate_segments(segments: &[GopSegment], capacity: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(capacity);
    for seg in segments {
        data.extend_from_slice(&seg.data);
    }
    data
}

/// Split `{camera_id}/{stem}.ts` into `(camera_id, stem)`; `None` for anything
/// that is not a `.ts` object with a camera prefix.
fn split_ts_key(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_suffix(".ts")?;
    rest.rsplit_once('/')
}

/// Sidecar JSON for the remote backend — identical to the local sidecar plus a
/// leading `"event_type"`, the only carrier of the type without a directory.
fn sidecar_json(
    event_type: EventType,
    backend: Option<&str>,
    model: Option<&str>,
    detection_details: &[DetectionDetail],
    continues: bool,
) -> String {
    let mut meta = serde_json::Map::new();
    meta.insert(
        "event_type".to_string(),
        serde_json::json!(event_type.as_str()),
    );
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
    if continues {
        meta.insert("continues".to_string(), serde_json::json!(true));
    }
    serde_json::to_string(&meta).unwrap()
}

fn reqwest_io(e: reqwest::Error) -> std::io::Error {
    std::io::Error::other(e)
}

/// Parse a `206` `Content-Range: bytes start-end/total` into its three numbers.
fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let total: u64 = total.trim().parse().ok()?;
    let (start, end) = range.split_once('-')?;
    let start: u64 = start.trim().parse().ok()?;
    let end: u64 = end.trim().parse().ok()?;
    Some((start, end, total))
}

/// Parse the total size out of a `416` `Content-Range: bytes */total`.
fn parse_content_range_total(value: &str) -> Option<u64> {
    let (_, total) = value.trim().strip_prefix("bytes ")?.split_once('/')?;
    total.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    use axum::{
        extract::{Path, RawQuery, State},
        http::{HeaderMap, StatusCode},
        routing::any,
        Json, Router,
    };

    // ---- in-process stathost stub -----------------------------------------

    #[derive(Clone)]
    struct Stub {
        files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        token: String,
        fail_writes: Arc<AtomicBool>,
        /// When set, GET ignores an incoming `Range` and answers a full `200` —
        /// a legal HTTP response the client must handle by replaying in full.
        ignore_range: Arc<AtomicBool>,
    }

    /// Drain a [`VideoStream`] body to bytes (test-only).
    async fn drain(vs: VideoStream) -> Vec<u8> {
        use futures_util::StreamExt;
        let mut buf = Vec::new();
        let mut stream = vs.stream;
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.unwrap());
        }
        buf
    }

    /// Resolve a single-range `Range` header against `total`, mirroring real
    /// stathost semantics: `Some((start, end))` inclusive, or `None` for a
    /// `416`-worthy unsatisfiable range.
    fn parse_stub_range(header: &str, total: u64) -> Option<(u64, u64)> {
        let spec = header.trim().strip_prefix("bytes=")?;
        if spec.contains(',') {
            return None;
        }
        let (s, e) = spec.split_once('-')?;
        if s.is_empty() {
            let n: u64 = e.trim().parse().ok()?;
            if n == 0 || total == 0 {
                return None;
            }
            let n = n.min(total);
            return Some((total - n, total - 1));
        }
        let start: u64 = s.trim().parse().ok()?;
        if start >= total {
            return None;
        }
        let end = if e.trim().is_empty() {
            total - 1
        } else {
            e.trim().parse::<u64>().ok()?.min(total - 1)
        };
        if end < start {
            return None;
        }
        Some((start, end))
    }

    fn authorized(headers: &HeaderMap, token: &str) -> bool {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {token}"))
            .unwrap_or(false)
    }

    async fn handler(
        State(stub): State<Stub>,
        Path((_bucket, path)): Path<(String, String)>,
        RawQuery(query): RawQuery,
        headers: HeaderMap,
        method: axum::http::Method,
        body: axum::body::Bytes,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;

        if path == "_meta/list" {
            if !authorized(&headers, &stub.token) {
                return StatusCode::UNAUTHORIZED.into_response();
            }
            // Mirrors stathost >= 0.2.0: plain array without ?detail=true,
            // [{"path","size","mtime"}] with it.
            let detail = query.as_deref() == Some("detail=true");
            let files = stub.files.lock().unwrap();
            let mut paths: Vec<String> = files.keys().cloned().collect();
            paths.sort();
            return if detail {
                let arr: Vec<serde_json::Value> = paths
                    .iter()
                    .map(|p| serde_json::json!({"path": p, "size": files[p].len(), "mtime": 0}))
                    .collect();
                Json(arr).into_response()
            } else {
                Json(paths).into_response()
            };
        }

        match method {
            axum::http::Method::PUT => {
                if !authorized(&headers, &stub.token) {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                if stub.fail_writes.load(Ordering::Relaxed) {
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
                stub.files.lock().unwrap().insert(path, body.to_vec());
                StatusCode::OK.into_response()
            }
            axum::http::Method::DELETE => {
                if !authorized(&headers, &stub.token) {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                if stub.files.lock().unwrap().remove(&path).is_some() {
                    StatusCode::OK.into_response()
                } else {
                    StatusCode::NOT_FOUND.into_response()
                }
            }
            // GET is public.
            _ => {
                let bytes = match stub.files.lock().unwrap().get(&path) {
                    Some(bytes) => bytes.clone(),
                    None => return StatusCode::NOT_FOUND.into_response(),
                };
                let total = bytes.len() as u64;
                let range = headers.get("range").and_then(|v| v.to_str().ok());
                match range {
                    // A server may legally answer a range request with a full 200.
                    Some(_) if stub.ignore_range.load(Ordering::Relaxed) => full_200(bytes),
                    Some(r) => match parse_stub_range(r, total) {
                        Some((start, end)) => {
                            let slice = bytes[start as usize..=end as usize].to_vec();
                            let mut resp = (StatusCode::PARTIAL_CONTENT, slice).into_response();
                            resp.headers_mut().insert(
                                "content-range",
                                format!("bytes {start}-{end}/{total}").parse().unwrap(),
                            );
                            resp.headers_mut()
                                .insert("accept-ranges", "bytes".parse().unwrap());
                            resp
                        }
                        None => {
                            let mut resp = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
                            resp.headers_mut().insert(
                                "content-range",
                                format!("bytes */{total}").parse().unwrap(),
                            );
                            resp
                        }
                    },
                    None => full_200(bytes),
                }
            }
        }
    }

    /// A `200 OK` full body advertising `Accept-Ranges: bytes`.
    fn full_200(bytes: Vec<u8>) -> axum::response::Response {
        use axum::response::IntoResponse;
        let mut resp = bytes.into_response();
        resp.headers_mut()
            .insert("accept-ranges", "bytes".parse().unwrap());
        resp
    }

    async fn spawn_stub(token: &str) -> (String, Stub) {
        let stub = Stub {
            files: Arc::new(Mutex::new(HashMap::new())),
            token: token.to_string(),
            fail_writes: Arc::new(AtomicBool::new(false)),
            ignore_range: Arc::new(AtomicBool::new(false)),
        };
        let app = Router::new()
            .route("/{bucket}/{*path}", any(handler))
            .with_state(stub.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), stub)
    }

    fn backend_for(url: &str, token: &str, max_stored_bytes: u64) -> StathostBackend {
        let config = StathostConfig {
            url: url.to_string(),
            bucket: "cams".to_string(),
            token: token.to_string(),
            max_stored_bytes,
            enabled: true,
        };
        StathostBackend::new(&config, &["cam".to_string()])
    }

    // ---- event fixtures ---------------------------------------------------

    fn segment(start_pts: u64, byte: u8, len: usize) -> GopSegment {
        GopSegment {
            start_pts,
            duration_ns: 1_000_000_000,
            data: Arc::new(vec![byte; len]),
            frame_count: 1,
        }
    }

    /// A movement event at `first_pts` (1s long), `size` bytes of video, with
    /// two filmstrip frames.
    fn movement_event(first_pts: u64, size: usize) -> FinishedEvent {
        FinishedEvent {
            segments: vec![segment(first_pts, 0xab, size)],
            first_pts,
            total_bytes: size,
            has_objects: false,
            object_classes: Vec::new(),
            filmstrip_frames: Some(Arc::new(vec![vec![0x01, 0x02], vec![0x03, 0x04]])),
            backend: None,
            model: None,
            detection_details: Vec::new(),
            continues: false,
            is_continuous: false,
        }
    }

    fn continuous_event(first_pts: u64, size: usize) -> FinishedEvent {
        let mut e = movement_event(first_pts, size);
        e.is_continuous = true;
        e.filmstrip_frames = None;
        e
    }

    fn upgrade_for(first_pts: u64) -> EventUpgrade {
        EventUpgrade {
            start_pts_ns: first_pts,
            duration_ms: 1000,
            object_classes: vec!["person".to_string()],
            detections: vec![DetectionDetail {
                class: "person".to_string(),
                confidence: 0.9,
            }],
            backend: "ollama".to_string(),
            model: "m".to_string(),
            continues: false,
        }
    }

    // ---- tests ------------------------------------------------------------

    #[tokio::test]
    async fn write_then_scan_round_trip_detailed_list() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);

        let event = movement_event(1_000, 40);
        assert_eq!(
            backend.write_event("cam", &event).await,
            WriteOutcome::Written
        );

        // Indexed and queryable in the writer's own index.
        let entry = backend.find_event("cam", 1_000).unwrap();
        assert_eq!(entry.event_type, EventType::Movement);
        assert_eq!(entry.file_size, 40);
        assert_eq!(entry.filmstrip_frames, 2);

        // Video and thumbnails come back through the trait (streamed).
        let vs = backend.read_video("cam", &entry, None).await.unwrap();
        assert!(matches!(vs.range, ServedRange::Full));
        assert_eq!(vs.total_size, 40);
        assert_eq!(drain(vs).await.len(), 40);
        assert_eq!(
            backend.read_thumbnail("cam", &entry).await.unwrap(),
            vec![0x01, 0x02]
        );
        assert_eq!(
            backend.read_filmstrip("cam", &entry, 1).await.unwrap(),
            vec![0x03, 0x04]
        );

        // A fresh backend rebuilding from the same host recovers the event,
        // its type, size (detailed list), and filmstrip count.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", 1_000).unwrap();
        assert_eq!(e.event_type, EventType::Movement);
        assert_eq!(e.file_size, 40);
        assert_eq!(e.filmstrip_frames, 2);
        assert_eq!(scanned.free_space().unwrap(), u64::MAX); // unlimited budget
    }

    #[tokio::test]
    async fn object_event_sidecar_carries_type_and_scans_back() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);

        let mut event = movement_event(4_000, 20);
        event.has_objects = true;
        event.object_classes = vec!["car".to_string()];
        event.detection_details = vec![DetectionDetail {
            class: "car".to_string(),
            confidence: 0.8,
        }];
        backend.write_event("cam", &event).await;

        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", 4_000).unwrap();
        assert_eq!(e.event_type, EventType::Object);
        assert_eq!(e.object_classes, vec!["car".to_string()]);
        assert_eq!(e.detections.len(), 1);
    }

    #[tokio::test]
    async fn upgrade_rewrites_sidecar_without_reuploading_video() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend.write_event("cam", &movement_event(5_000, 25)).await;

        // Capture the video bytes and count writes before the upgrade.
        let ts_key = "cam/5000_1000.ts";
        let before = stub.files.lock().unwrap().get(ts_key).cloned().unwrap();

        backend.upgrade_event("cam", &upgrade_for(5_000)).await;

        // The index flipped to Object...
        let e = backend.find_event("cam", 5_000).unwrap();
        assert_eq!(e.event_type, EventType::Object);
        assert_eq!(e.object_classes, vec!["person".to_string()]);
        // ...the video object is byte-for-byte unchanged (no re-upload)...
        assert_eq!(
            stub.files.lock().unwrap().get(ts_key).cloned().unwrap(),
            before
        );
        // ...and the sidecar now declares the object type.
        let sidecar = stub
            .files
            .lock()
            .unwrap()
            .get("cam/5000_1000.json")
            .cloned()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&sidecar).unwrap();
        assert_eq!(json["event_type"], serde_json::json!("object"));
        assert_eq!(json["detections"][0]["class"], serde_json::json!("person"));
    }

    #[tokio::test]
    async fn delete_via_prune_removes_all_objects() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        // A movement event far enough in the past to expire.
        let old_pts = 1_000_000_000; // 1s after epoch
        backend
            .write_event("cam", &movement_event(old_pts, 30))
            .await;
        assert_eq!(stub.files.lock().unwrap().len(), 4); // ts + json + 2 thumbs

        // Prune with a tiny movement retention → the old event goes.
        backend.prune(1, u64::MAX, u64::MAX).await;

        assert!(backend.find_event("cam", old_pts).is_none());
        assert!(stub.files.lock().unwrap().is_empty());
        assert_eq!(backend.used_bytes.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn budget_prune_evicts_cheapest_and_oldest_first() {
        let (url, _stub) = spawn_stub("secret").await;
        // Budget of 60 bytes; three 40-byte events (120 total) overflow it.
        let backend = backend_for(&url, "secret", 60);

        // Oldest first: a continuous chunk, then a movement, then an object.
        backend
            .write_event("cam", &continuous_event(1_000, 40))
            .await;

        let mut obj = movement_event(3_000, 40);
        obj.has_objects = true;
        obj.object_classes = vec!["person".to_string()];
        backend.write_event("cam", &obj).await;

        backend.write_event("cam", &movement_event(2_000, 40)).await;

        // Enforce the budget (as the pre-write guard would).
        backend.guard_free_space("cam", 0).await;

        // Cheapest tier first: the continuous chunk is evicted; the movement
        // and object survive (usage 80 > 60 would need another eviction, but
        // the movement is next-cheapest — check exact survivors).
        // 120 - 40 (continuous) = 80, still > 60, so the movement goes too.
        assert!(backend.find_event("cam", 1_000).is_none()); // continuous evicted
        assert!(backend.find_event("cam", 2_000).is_none()); // movement evicted
        assert!(backend.find_event("cam", 3_000).is_some()); // object survives
        assert!(backend.used_bytes.load(Ordering::Relaxed) <= 60);
    }

    #[tokio::test]
    async fn scan_tolerates_ts_without_sidecar() {
        let (url, stub) = spawn_stub("secret").await;
        // A lone .ts with no sidecar (as an interrupted upload would leave).
        stub.files
            .lock()
            .unwrap()
            .insert("cam/9000_1000.ts".to_string(), vec![0u8; 10]);

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;
        let e = backend.find_event("cam", 9_000).unwrap();
        // No sidecar → indexed as a plain movement event.
        assert_eq!(e.event_type, EventType::Movement);
        assert_eq!(e.filmstrip_frames, 0);
        assert!(e.object_classes.is_empty());
    }

    #[tokio::test]
    async fn write_retries_then_drops_on_persistent_failure() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_writes.store(true, Ordering::Relaxed);

        let outcome = backend.write_event("cam", &movement_event(6_000, 30)).await;
        assert_eq!(outcome, WriteOutcome::Failed);
        // The event was not indexed and nothing landed on the host.
        assert!(backend.find_event("cam", 6_000).is_none());
        assert!(stub.files.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn read_thumbnail_errors_when_no_filmstrip() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let event = continuous_event(7_000, 30); // no filmstrip frames
        backend.write_event("cam", &event).await;
        let entry = backend.find_event("cam", 7_000).unwrap();
        assert!(backend.read_thumbnail("cam", &entry).await.is_err());
    }

    // ---- streamed Range playback ------------------------------------------

    #[tokio::test]
    async fn read_video_serves_partial_and_suffix_ranges() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        // A 40-byte movement event (body is 40 × 0xab).
        backend.write_event("cam", &movement_event(8_000, 40)).await;
        let entry = backend.find_event("cam", 8_000).unwrap();

        // bytes=10-19 → a 206 with a 10-byte body and the right Content-Range.
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 10,
                    end: Some(19),
                }),
            )
            .await
            .unwrap();
        assert_eq!(vs.range, ServedRange::Partial { start: 10, end: 19 });
        assert_eq!(vs.total_size, 40);
        assert_eq!(drain(vs).await, vec![0xab; 10]);

        // bytes=-5 → the last five bytes.
        let vs = backend
            .read_video("cam", &entry, Some(RangeRequest::Suffix(5)))
            .await
            .unwrap();
        assert_eq!(vs.range, ServedRange::Partial { start: 35, end: 39 });
        assert_eq!(drain(vs).await, vec![0xab; 5]);
    }

    #[tokio::test]
    async fn read_video_reports_unsatisfiable_range() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend.write_event("cam", &movement_event(9_000, 40)).await;
        let entry = backend.find_event("cam", 9_000).unwrap();

        // start past EOF → the stub answers 416; we surface Unsatisfiable + size.
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 100,
                    end: None,
                }),
            )
            .await
            .unwrap();
        assert_eq!(vs.range, ServedRange::Unsatisfiable);
        assert_eq!(vs.total_size, 40);
        assert!(drain(vs).await.is_empty());
    }

    #[tokio::test]
    async fn read_video_degrades_to_full_when_server_ignores_range() {
        let (url, stub) = spawn_stub("secret").await;
        // A 200 with the full body is a legal answer to a range request.
        stub.ignore_range.store(true, Ordering::Relaxed);
        let backend = backend_for(&url, "secret", 0);
        backend
            .write_event("cam", &movement_event(10_000, 40))
            .await;
        let entry = backend.find_event("cam", 10_000).unwrap();

        // A range was requested, but the full body comes back as a 200.
        let vs = backend
            .read_video(
                "cam",
                &entry,
                Some(RangeRequest::FromTo {
                    start: 10,
                    end: Some(19),
                }),
            )
            .await
            .unwrap();
        assert_eq!(vs.range, ServedRange::Full);
        assert_eq!(vs.total_size, 40);
        assert_eq!(drain(vs).await.len(), 40);
    }

    const SEC: u64 = 1_000_000_000;

    /// A backend whose index holds `spans` and which never talks to a host.
    fn indexed(spans: &[(u64, u32)]) -> StathostBackend {
        let backend = backend_for("http://127.0.0.1:1", "secret", 0);
        for &(start_pts_ns, duration_ms) in spans {
            backend.insert_entry(
                "cam",
                WarmEventEntry {
                    start_pts_ns,
                    duration_ms,
                    event_type: EventType::Continuous,
                    file_size: 0,
                    object_classes: Vec::new(),
                    backend: None,
                    model: None,
                    detections: Vec::new(),
                    filmstrip_frames: 0,
                    continues: false,
                    recovered: false,
                },
            );
        }
        backend
    }

    #[test]
    fn query_returns_long_events_that_started_before_the_window() {
        // A 100s chunk starting at 0, then two 1s events that end long before
        // the window: sorted by start, "ends before from" is false-then-true,
        // so a binary search on it skips right past the chunk that does overlap.
        let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let hits = backend.query("cam", 50 * SEC, 60 * SEC);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_pts_ns, 0);
    }

    #[test]
    fn query_returns_every_overlapping_event_in_start_order() {
        let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let starts: Vec<u64> = backend
            .query("cam", 0, u64::MAX)
            .iter()
            .map(|e| e.start_pts_ns)
            .collect();
        assert_eq!(starts, vec![0, 10 * SEC, 20 * SEC]);
        assert!(backend.query("unknown", 0, u64::MAX).is_empty());
    }

    #[test]
    fn zero_duration_events_are_found_at_their_start() {
        let backend = indexed(&[(10 * SEC, 0)]);
        assert_eq!(backend.query("cam", 10 * SEC, 10 * SEC).len(), 1);
        assert!(backend.query("cam", 10 * SEC + 1, 20 * SEC).is_empty());
    }

    #[test]
    fn query_bounds_include_events_that_only_touch_them() {
        let backend = indexed(&[(10 * SEC, 5_000)]);
        // Ends exactly at from_ns.
        assert_eq!(backend.query("cam", 15 * SEC, 20 * SEC).len(), 1);
        assert!(backend.query("cam", 15 * SEC + 1, 20 * SEC).is_empty());
        // Starts exactly at to_ns.
        assert_eq!(backend.query("cam", 0, 10 * SEC).len(), 1);
        assert!(backend.query("cam", 0, 10 * SEC - 1).is_empty());
    }

    #[test]
    fn query_with_an_inverted_range_is_empty() {
        // These bounds used to be computed independently and sliced, which
        // panicked here with start > end.
        let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000)]);
        assert!(backend.query("cam", u64::MAX, 0).is_empty());
        assert!(backend.query("cam", 20 * SEC, 5 * SEC).is_empty());
    }
}
