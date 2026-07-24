//! Remote "stathost" warm-storage backend.
//!
//! [`stathost`](https://github.com/nsg/stathost) is a small static file host:
//!
//! * `PUT /{bucket}/{path}` and `DELETE /{bucket}/{path}` — authenticated with
//!   `Authorization: Bearer <token>`;
//! * `GET /{bucket}/{path}` — public;
//! * `GET /{bucket}/_meta/list` — authenticated, returns a JSON array of path
//!   strings (a detailed variant, `?detail=true`, may return
//!   `[{"path","size","mtime"}]`).
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
//!   sidecar — indexed as a plain movement event on the next scan. (Until
//!   stathost ships atomic PUT, a `.ts` can also land truncated; a zero-byte
//!   `.ts` is warned about at scan time.)
//! * **`read_video` buffers the whole object in RAM.** Acceptable for 10–60 MB
//!   events today; a future pass should stream (stathost has no Range support
//!   yet either).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use async_trait::async_trait;
use serde::Deserialize;

use crate::buffer::warm::{EventUpgrade, FinishedEvent};
use crate::buffer::GopSegment;
use crate::config::StathostConfig;
use crate::locks::LockExt;
use crate::storage::backend::{
    deduplicate_detections, ThumbnailError, WarmStorageBackend, WriteOutcome,
};
use crate::storage::warm_index::{parse_event_filename, parse_sidecar_json, DetectionDetail};
use crate::storage::{EventType, WarmEventEntry};

const NANOS_PER_MS: u64 = 1_000_000;

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
                client: reqwest::Client::new(),
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
            // Only warn on a real, known-zero size (interrupted upload); a plain
            // list carries no sizes, so stay quiet there.
            if item.has_size && item.size == 0 {
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

    fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
                let start = entries.partition_point(|e| {
                    e.start_pts_ns + (e.duration_ms as u64) * NANOS_PER_MS < from_ns
                });
                let end = entries.partition_point(|e| e.start_pts_ns <= to_ns);
                entries[start..end].to_vec()
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
    ) -> std::io::Result<Vec<u8>> {
        // NOTE: buffers the whole object in RAM. Fine for 10–60 MB events;
        // revisit with a streaming/Range pass once stathost supports Range.
        self.http
            .get(&Self::ts_key(camera_id, entry))
            .await
            .map_err(reqwest_io)
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
    client: reqwest::Client,
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
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.bytes().await?.to_vec())
    }

    /// List every object in the bucket, preferring the detailed variant and
    /// falling back cleanly to the plain shape (whether the server returns
    /// plain strings or errors on `?detail=true`).
    async fn list(&self) -> Result<Vec<ListEntry>, reqwest::Error> {
        match self.list_raw(true).await {
            Ok(items) => Ok(items),
            Err(_) => self.list_raw(false).await,
        }
    }

    async fn list_raw(&self, detail: bool) -> Result<Vec<ListEntry>, reqwest::Error> {
        let mut url = format!("{}/_meta/list", self.base);
        if detail {
            url.push_str("?detail=true");
        }
        let items: Vec<RawListItem> = self
            .client
            .get(url)
            .bearer_auth(&self.token)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(items.into_iter().map(ListEntry::from).collect())
    }
}

/// A `_meta/list` entry, normalized across the plain and detailed shapes.
struct ListEntry {
    path: String,
    size: u64,
    /// Whether `size` came from the server (detailed shape) or is a placeholder
    /// (plain shape). Gates the zero-byte-upload warning.
    has_size: bool,
}

/// Tolerant of both list shapes: a bare `"path"` string, or a
/// `{"path","size","mtime"}` object (extra fields, e.g. `mtime`, are ignored).
#[derive(Deserialize)]
#[serde(untagged)]
enum RawListItem {
    Detailed { path: String, size: Option<u64> },
    Bare(String),
}

impl From<RawListItem> for ListEntry {
    fn from(raw: RawListItem) -> Self {
        match raw {
            RawListItem::Detailed { path, size } => ListEntry {
                path,
                size: size.unwrap_or(0),
                has_size: size.is_some(),
            },
            RawListItem::Bare(path) => ListEntry {
                path,
                size: 0,
                has_size: false,
            },
        }
    }
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

    #[derive(Clone, Copy, PartialEq)]
    enum ListMode {
        /// `?detail=true` returns `[{"path","size"}]`.
        Detail,
        /// Always returns plain `["path"]`, even for `?detail=true`.
        Plain,
        /// `?detail=true` errors (400); the plain call returns `["path"]`.
        DetailUnsupported,
    }

    #[derive(Clone)]
    struct Stub {
        files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        token: String,
        mode: ListMode,
        fail_writes: Arc<AtomicBool>,
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
            let detail = query.as_deref() == Some("detail=true");
            let files = stub.files.lock().unwrap();
            let mut paths: Vec<String> = files.keys().cloned().collect();
            paths.sort();
            return match stub.mode {
                ListMode::Detail if detail => {
                    let arr: Vec<serde_json::Value> = paths
                        .iter()
                        .map(|p| serde_json::json!({"path": p, "size": files[p].len(), "mtime": 0}))
                        .collect();
                    Json(arr).into_response()
                }
                ListMode::DetailUnsupported if detail => StatusCode::BAD_REQUEST.into_response(),
                _ => Json(paths).into_response(),
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
            _ => match stub.files.lock().unwrap().get(&path) {
                Some(bytes) => bytes.clone().into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
        }
    }

    async fn spawn_stub(mode: ListMode, token: &str) -> (String, Stub) {
        let stub = Stub {
            files: Arc::new(Mutex::new(HashMap::new())),
            token: token.to_string(),
            mode,
            fail_writes: Arc::new(AtomicBool::new(false)),
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
        let (url, _stub) = spawn_stub(ListMode::Detail, "secret").await;
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

        // Video and thumbnails come back through the trait.
        assert_eq!(backend.read_video("cam", &entry).await.unwrap().len(), 40);
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
    async fn scan_tolerates_plain_list_shape() {
        let (url, _stub) = spawn_stub(ListMode::Plain, "secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend.write_event("cam", &movement_event(2_000, 30)).await;

        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", 2_000).unwrap();
        assert_eq!(e.event_type, EventType::Movement);
        // Plain list carries no sizes → size unknown (0).
        assert_eq!(e.file_size, 0);
        assert_eq!(e.filmstrip_frames, 2);
    }

    #[tokio::test]
    async fn scan_falls_back_when_detail_unsupported() {
        let (url, _stub) = spawn_stub(ListMode::DetailUnsupported, "secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend.write_event("cam", &movement_event(3_000, 30)).await;

        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        // Fallback to the plain call still recovers the event.
        assert!(scanned.find_event("cam", 3_000).is_some());
    }

    #[tokio::test]
    async fn object_event_sidecar_carries_type_and_scans_back() {
        let (url, _stub) = spawn_stub(ListMode::Detail, "secret").await;
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
        let (url, stub) = spawn_stub(ListMode::Detail, "secret").await;
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
        let (url, stub) = spawn_stub(ListMode::Detail, "secret").await;
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
        let (url, _stub) = spawn_stub(ListMode::Detail, "secret").await;
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
        let (url, stub) = spawn_stub(ListMode::Detail, "secret").await;
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
        let (url, stub) = spawn_stub(ListMode::Detail, "secret").await;
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
        let (url, _stub) = spawn_stub(ListMode::Detail, "secret").await;
        let backend = backend_for(&url, "secret", 0);
        let event = continuous_event(7_000, 30); // no filmstrip frames
        backend.write_event("cam", &event).await;
        let entry = backend.find_event("cam", 7_000).unwrap();
        assert!(backend.read_thumbnail("cam", &entry).await.is_err());
    }
}
