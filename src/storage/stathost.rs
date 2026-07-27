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
//! uploaded (unlike local mode, where it is conditional) and — for every event
//! except the plain movement one that a sidecar-less `.ts` already scans back
//! as — *required*: a write that cannot store it is failed rather than
//! reported as written.
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
//! * **The `.ts` upload is the commit point**, the way the staging rename is in
//!   local mode: the sidecar goes up first, the video second, thumbs last. A
//!   `.ts` that outlived its sidecar would scan back as a plain movement and
//!   expire on the wrong retention (2 days instead of 14), so no video is
//!   uploaded until its metadata is durable. Nothing is rolled back afterwards:
//!   a failed upload can still have landed (a timeout or a proxy error says
//!   nothing about what the origin committed), and both leftovers are benign —
//!   an orphan `.json` is invisible to the scan, which walks `.ts` objects
//!   only, and a phantom `.ts` still has the sidecar that types it correctly.
//!   stathost's uploads are atomic server-side, so a truncated object can't be
//!   served; a zero-byte `.ts` in the listing still gets a warning at scan time.
//! * **An unreadable sidecar is not a movement event.** The scan applies the
//!   movement default only to a *confirmed* 404; anything else — a transport
//!   failure, unparsable bytes, valid JSON naming no type — leaves the type
//!   unknown. Such an event is still indexed and served, but every decision
//!   that would need its type errs toward keeping it: age-based pruning
//!   measures it against the longest configured retention, and budget eviction
//!   tiers it with the objects. The prune tick re-reads its sidecar, which is
//!   the only in-process retry there is — [`scan`](StathostBackend::scan) runs
//!   once at startup.
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
use crate::storage::warm_index::{
    cap_sweep_deletions, parse_event_filename, parse_sidecar_json, wall_clock_ns, DetectionDetail,
    SidecarData,
};
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
/// not one event — `write_event` uploads a sidecar and a video, each retried
/// once, then thumbnails up to the first failure, so a half-broken link can
/// hold the warm writer for a multiple of this.
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
    /// Start PTSs whose sidecar the scan could not read, per camera. Their
    /// [`WarmEventEntry::event_type`] is a placeholder, not a fact — see
    /// [`Self::mark_unknown_type`].
    unknown_type: HashMap<String, RwLock<HashSet<u64>>>,
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
        let mut unknown_type = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(Vec::new()));
            unknown_type.insert(id.clone(), RwLock::new(HashSet::new()));
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
            unknown_type,
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
        self.clear_unknown_type(camera_id, start_pts_ns);
        let lock = self.cameras.get(camera_id)?;
        let mut entries = lock.write_recover();
        let idx = entries
            .binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns)
            .ok()?;
        Some(entries.remove(idx))
    }

    /// Remember that this event resisted deletion, so the sweep's per-sweep
    /// deletion cap stops being spent on it: a store that refuses one event for
    /// good would otherwise block every deletion behind it, sweep after sweep.
    /// The flag is in-RAM and a restart clears it, which is the retry.
    fn mark_delete_failed(&self, camera_id: &str, start_pts_ns: u64) {
        let Some(lock) = self.cameras.get(camera_id) else {
            return;
        };
        let mut entries = lock.write_recover();
        if let Ok(idx) = entries.binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns) {
            entries[idx].delete_failed = true;
        }
    }

    /// Record that this event's type could not be established. `event_type` on
    /// the entry is then a display placeholder, not a retention class: pruning
    /// it as a movement would delete an object event twelve days early. Instead
    /// [`WarmStorageBackend::prune`] gives it the longest *configured*
    /// retention, which cannot expire before its own whatever its true type is,
    /// and still expires — a permanently unreadable sidecar would otherwise pin
    /// its footage forever on a store whose budget is unlimited by default.
    fn mark_unknown_type(&self, camera_id: &str, start_pts_ns: u64) {
        if let Some(lock) = self.unknown_type.get(camera_id) {
            lock.write_recover().insert(start_pts_ns);
        }
    }

    fn has_unknown_type(&self, camera_id: &str, start_pts_ns: u64) -> bool {
        self.unknown_type
            .get(camera_id)
            .is_some_and(|lock| lock.read_recover().contains(&start_pts_ns))
    }

    /// Drop the marker once the type is settled or the event is gone. Only
    /// [`Self::scan`] ever sets one, and it runs once per process, so in
    /// practice this fires when an entry leaves the index; it also keeps
    /// `upgrade_event` from leaving a "type unknown" marker on an event it just
    /// proved to be an object.
    fn clear_unknown_type(&self, camera_id: &str, start_pts_ns: u64) {
        if let Some(lock) = self.unknown_type.get(camera_id) {
            lock.write_recover().remove(&start_pts_ns);
        }
    }

    // ---- object-store helpers ----

    /// Read and parse one event's sidecar, retried once — the same allowance
    /// the write path gives it, and worth it here because the alternative to a
    /// readable sidecar is a guessed retention class.
    async fn read_sidecar(&self, camera_id: &str, stem: &str) -> SidecarRead {
        let key = format!("{camera_id}/{stem}.json");
        for attempt in 0..2 {
            match self.http.get_optional(&key).await {
                Ok(None) => return SidecarRead::Absent,
                Ok(Some(bytes)) => {
                    return match serde_json::from_slice::<serde_json::Value>(&bytes) {
                        Ok(value) => {
                            // Bytes that name no type are not bytes that say
                            // "movement" either.
                            let data = parse_sidecar_json(&value);
                            match data.event_type {
                                Some(_) => SidecarRead::Parsed(data),
                                None => SidecarRead::Typeless(data),
                            }
                        }
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "unparsable stathost sidecar");
                            SidecarRead::Unreadable
                        }
                    };
                }
                // Debug, not warn: the prune tick retries every held event
                // hourly, and a store that is down would log one line per
                // event per sweep. The scan's aggregate warn carries the news.
                Err(e) if attempt == 1 => {
                    tracing::debug!(key = %key, error = %e, "could not read stathost sidecar");
                    return SidecarRead::Unreadable;
                }
                Err(_) => {}
            }
        }
        SidecarRead::Unreadable
    }

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

    /// Re-read the sidecars of events whose type an earlier scan could not
    /// establish, and index what they say.
    ///
    /// [`WarmStorageBackend::scan`] runs exactly once per process, so without
    /// this a hold would last until a restart however quickly the store
    /// recovered. The scheduled sweep is the only place in-process where a
    /// retry can happen at all; it costs one GET per held event, and a store
    /// with nothing held issues none.
    async fn resolve_unknown_types(&self, camera_id: &str, cancel: &std::sync::atomic::AtomicBool) {
        let held: Vec<u64> = match self.unknown_type.get(camera_id) {
            Some(lock) => lock.read_recover().iter().copied().collect(),
            None => return,
        };
        let mut resolved = 0u64;
        for start_pts_ns in held {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let Some(entry) = self.find_event(camera_id, start_pts_ns) else {
                self.clear_unknown_type(camera_id, start_pts_ns);
                continue;
            };
            let stem = format!("{start_pts_ns}_{}", entry.duration_ms);
            let sidecar = match self.read_sidecar(camera_id, &stem).await {
                SidecarRead::Parsed(s) => Some(s),
                SidecarRead::Absent => None,
                // Still unreadable, or still naming no type: keep the hold.
                SidecarRead::Unreadable | SidecarRead::Typeless(_) => continue,
            };
            if let Some(lock) = self.cameras.get(camera_id) {
                let mut entries = lock.write_recover();
                if let Ok(i) = entries.binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns) {
                    apply_sidecar(&mut entries[i], sidecar.as_ref());
                }
            }
            self.clear_unknown_type(camera_id, start_pts_ns);
            resolved += 1;
        }
        if resolved > 0 {
            tracing::info!(camera = %camera_id, resolved,
                "read event types that an earlier scan could not; normal retention resumes");
        }
    }

    /// Enforce the client-side storage budget: while tracked usage exceeds
    /// `max_stored_bytes`, delete the oldest events cheapest-tier-first
    /// (continuous → movements → objects). A transport failure stops the pass
    /// (retried on the next tick). No-op when the budget is unlimited.
    ///
    /// Like the local low-space guard, this is deliberately outside the sweep's
    /// [`cap_sweep_deletions`] cap — the budget is not clock-derived — so a
    /// full budget during a held-back drain can delete the footage the sweep is
    /// holding.
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
                        // An event of unknown type is evicted with the objects,
                        // the tier kept longest: its placeholder says movement,
                        // and evicting on that guess would throw away footage
                        // this whole path exists to keep.
                        .filter(|e| {
                            if self.has_unknown_type(cam, e.start_pts_ns) {
                                tier == EventType::Object
                            } else {
                                e.event_type == tier
                            }
                        })
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

        // Step 1: the sidecar, before the video. It is the sole carrier of the
        // event type, so an event whose sidecar is missing is not a slightly
        // poorer event — it is the wrong kind of event, expiring on the wrong
        // retention after the next scan. One retry, then fail the write before
        // the video is uploaded at all.
        let sidecar_key = format!("{camera_id}/{stem}.json");
        let sidecar = sidecar_json(
            event_type,
            event.backend.as_deref(),
            event.model.as_deref(),
            &event.detection_details,
            event.continues,
        )
        .into_bytes();
        if self.http.put(&sidecar_key, sidecar.clone()).await.is_err() {
            tracing::warn!(camera = %camera_id, stem = %stem,
                "stathost sidecar upload failed, retrying once");
            if self.http.put(&sidecar_key, sidecar).await.is_err() {
                if sidecar_required(event) {
                    tracing::error!(
                        camera = %camera_id,
                        first_pts = event.first_pts,
                        bytes = event.total_bytes,
                        "dropping event: stathost sidecar upload failed after retry"
                    );
                    return WriteOutcome::Failed;
                }
                tracing::warn!(camera = %camera_id, stem = %stem,
                    "stathost sidecar upload failed after retry; \
                     a scan rebuilds this movement event unchanged without it");
            }
        }

        // Step 2: the video — one retry, then drop (logged) so a failed write
        // is never lost silently. Nothing is rolled back: a PUT that reports
        // failure may still have committed server-side, and deleting the
        // sidecar of such a phantom .ts would leave precisely the bare video
        // this order exists to prevent. The residue is harmless either way.
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

        // Step 3: eager filmstrip thumbnails; frame 0 doubles as the poster.
        // Non-fatal — the UI hides frames that fail to load. The scan counts
        // frames contiguously from 0, so a gap stops the upload: what is
        // indexed now is what a scan would rebuild later.
        let filmstrip_frames = match &event.filmstrip_frames {
            Some(frames) => {
                let mut wrote = 0;
                for (i, jpeg) in frames.iter().enumerate() {
                    let key = format!("{camera_id}/{stem}_thumb_{i}.jpg");
                    if self.http.put(&key, jpeg.clone()).await.is_err() {
                        tracing::warn!(camera = %camera_id, stem = %stem, frame = i,
                            "failed to upload filmstrip thumbnail to stathost");
                        break;
                    }
                    wrote += 1;
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
                delete_failed: false,
            },
        );
        self.used_bytes.fetch_add(file_size, Ordering::Relaxed);
        self.clear_unknown_type(camera_id, event.first_pts);

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
                // The sidecar just written carries the upgrade's `continues`;
                // the index has to say the same thing (LocalDisk rebuilds the
                // whole entry here, which is where this was being lost).
                entry.continues = upgrade.continues;
            }
        }
        // The type is now established. An upgrade only ever targets an event
        // written by this process, so it cannot reach one the scan held — but
        // this and `write_event` are the two places a type becomes a fact, and
        // neither may leave a "type unknown" marker behind it.
        self.clear_unknown_type(camera_id, upgrade.start_pts_ns);

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
        cancel: &std::sync::atomic::AtomicBool,
    ) {
        let now_ns = wall_clock_ns();
        let max_age = |t: EventType| match t {
            EventType::Movement => movement_max_age_ns,
            EventType::Object => object_max_age_ns,
            EventType::Continuous => continuous_max_age_ns,
        };
        // An event whose type the scan could not read is measured against the
        // longest configured retention instead of its placeholder's: no true
        // type can expire later than that, so nothing is ever deleted early,
        // and unlike an indefinite hold it does terminate.
        let unknown_max_age = movement_max_age_ns
            .max(object_max_age_ns)
            .max(continuous_max_age_ns);

        // Deleting one event here is several sequential HTTP requests, each
        // able to sit on a request timeout, so shutdown gets checked between
        // events and between cameras — never part-way through an event.
        let stop = || cancel.load(Ordering::Relaxed);
        for (camera_id, lock) in self.cameras.iter() {
            if stop() {
                break;
            }
            // First give held events a chance to be typed, so one that resolves
            // is pruned on its real retention in this same sweep.
            self.resolve_unknown_types(camera_id, cancel).await;
            let (indexed, expired) = {
                let entries = lock.read_recover();
                let expired: Vec<WarmEventEntry> = entries
                    .iter()
                    .filter(|e| {
                        let limit = if self.has_unknown_type(camera_id, e.start_pts_ns) {
                            unknown_max_age
                        } else {
                            max_age(e.event_type)
                        };
                        now_ns.saturating_sub(e.start_pts_ns) > limit
                    })
                    .cloned()
                    .collect();
                (entries.len(), expired)
            };
            if expired.is_empty() {
                continue;
            }
            let expired = cap_sweep_deletions(camera_id, indexed, expired);

            let mut deleted = 0u64;
            for entry in &expired {
                if stop() {
                    break;
                }
                if self.delete_event_objects(camera_id, entry).await {
                    if let Some(removed) = self.remove_entry(camera_id, entry.start_pts_ns) {
                        self.used_bytes
                            .fetch_sub(removed.file_size, Ordering::Relaxed);
                    }
                    deleted += 1;
                } else {
                    self.mark_delete_failed(camera_id, entry.start_pts_ns);
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
        let mut unknown_type = 0usize;
        let mut typeless = 0usize;

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

            // Fetch the sibling sidecar. Only a confirmed absence means
            // "movement" — that is the one event written without a sidecar.
            let (sidecar, type_known) = match self.read_sidecar(camera_id, stem).await {
                SidecarRead::Parsed(s) => (Some(s), true),
                SidecarRead::Absent => (None, true),
                SidecarRead::Unreadable => (None, false),
                SidecarRead::Typeless(s) => {
                    // Deterministic, so no later scan will clear this by
                    // itself: name the object so it can actually be fixed.
                    tracing::warn!(path = %item.path,
                        "stathost sidecar names no event type; retention falls back to the \
                         longest configured age until the sidecar is repaired");
                    typeless += 1;
                    (Some(s), false)
                }
            };
            let mut filmstrip_frames = 0usize;
            while all_paths
                .contains(format!("{camera_id}/{stem}_thumb_{filmstrip_frames}.jpg").as_str())
            {
                filmstrip_frames += 1;
            }

            let mut entry = WarmEventEntry {
                start_pts_ns,
                duration_ms,
                event_type: EventType::Movement,
                file_size: item.size,
                object_classes: Vec::new(),
                backend: None,
                model: None,
                detections: Vec::new(),
                filmstrip_frames,
                continues: false,
                recovered: false,
                delete_failed: false,
            };
            apply_sidecar(&mut entry, sidecar.as_ref());
            self.used_bytes.fetch_add(item.size, Ordering::Relaxed);
            self.insert_entry(camera_id, entry);
            if !type_known {
                self.mark_unknown_type(camera_id, start_pts_ns);
                unknown_type += 1;
            }
            total += 1;
        }

        if unknown_type > 0 {
            tracing::warn!(
                unknown_type,
                typeless,
                "events indexed without a known type: shown as movement but pruned on the \
                 longest configured retention. `typeless` of them need their sidecar fixed; \
                 the rest may resolve on a later start"
            );
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

/// What the scan learned about one event's sidecar. The two failure variants
/// both mean "type unknown" but differ in whether looking again can help.
enum SidecarRead {
    Parsed(SidecarData),
    /// A confirmed 404: the event was written without a sidecar, which only a
    /// plain movement event ever is — see [`sidecar_required`].
    Absent,
    /// Unreachable or unparsable — a transient condition, so a later scan may
    /// well resolve it.
    Unreadable,
    /// Valid JSON naming no recognized `event_type` (`{}`, `null`, a typo).
    /// Deterministic content: re-reading it will never say anything different,
    /// so this one needs an operator, not a retry. The rest of the sidecar is
    /// still used — it is only the *type* that is missing.
    Typeless(SidecarData),
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

    /// Fetch an object, separating a confirmed absence (`Ok(None)`) from a
    /// failure to find out (`Err`). The scan needs that difference: a missing
    /// sidecar is information, an unreachable one is not.
    async fn get_optional(&self, path: &str) -> Result<Option<Vec<u8>>, reqwest::Error> {
        let resp = self
            .client
            .get(self.url(path))
            .bearer_auth(&self.token)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(resp.error_for_status()?.bytes().await?.to_vec()))
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

/// Write the sidecar-derived half of an index entry, `None` meaning "no sidecar
/// exists" — the plain movement event. Shared by the scan and the prune tick's
/// re-read of held events so the two can never disagree about what a sidecar
/// says; the fields not set here (size, filmstrip count) come from the listing.
fn apply_sidecar(entry: &mut WarmEventEntry, sidecar: Option<&SidecarData>) {
    entry.event_type = sidecar
        .and_then(|s| s.event_type)
        .unwrap_or(EventType::Movement);
    entry.object_classes = sidecar.map(|s| s.classes.clone()).unwrap_or_default();
    entry.backend = sidecar.and_then(|s| s.backend.clone());
    entry.model = sidecar.and_then(|s| s.model.clone());
    entry.detections = sidecar.map(|s| s.detections.clone()).unwrap_or_default();
    entry.continues = sidecar.is_some_and(|s| s.continues);
    entry.recovered = sidecar.is_some_and(|s| s.recovered);
}

/// Whether this event's sidecar carries anything a sidecar-less scan would not
/// already assume. It does not for a plain movement event: `has_objects` is set
/// from a non-empty detection list, and `backend`/`model` are `Some` only
/// alongside detections, so the sidecar of a first-chunk movement event says
/// only "movement, no detections" — exactly the scan's default for a bare
/// `.ts`. Losing it is therefore invisible, while losing any other event's
/// sidecar rewrites its retention class. Local mode draws the same line one
/// field earlier (it writes a sidecar iff `has_objects || continues`); the
/// delta is `continuous`, which has no directory to fall back on here.
fn sidecar_required(event: &FinishedEvent) -> bool {
    event.event_type() != EventType::Movement || event.continues
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
        /// Breaks one object of an event while the others go through.
        put_fault: Arc<Mutex<Option<PutFault>>>,
        /// When set, GETs whose path ends with this suffix answer `500` — an
        /// unreadable object, as distinct from an absent one (`404`).
        fail_get_suffix: Arc<Mutex<Option<String>>>,
        /// DELETEs of exactly these paths answer `500`: an object the store
        /// refuses to drop, as distinct from one already gone (`404`).
        fail_delete_paths: Arc<Mutex<HashSet<String>>>,
        /// When set, GET ignores an incoming `Range` and answers a full `200` —
        /// a legal HTTP response the client must handle by replaying in full.
        ignore_range: Arc<AtomicBool>,
    }

    /// A PUT failure injected by path suffix. `stored` decides whether the
    /// object lands anyway before the error is returned — the shape of an
    /// upload timeout or a proxy 5xx over a body the origin already committed,
    /// which a client cannot tell from an upload that never happened.
    #[derive(Clone)]
    struct PutFault {
        suffix: String,
        stored: bool,
    }

    impl Stub {
        fn fail_puts(&self, suffix: &str, stored: bool) {
            *self.put_fault.lock().unwrap() = Some(PutFault {
                suffix: suffix.to_string(),
                stored,
            });
        }

        fn fail_gets(&self, suffix: &str) {
            *self.fail_get_suffix.lock().unwrap() = Some(suffix.to_string());
        }

        fn clear_faults(&self) {
            *self.put_fault.lock().unwrap() = None;
            *self.fail_get_suffix.lock().unwrap() = None;
        }

        fn has(&self, path: &str) -> bool {
            self.files.lock().unwrap().contains_key(path)
        }
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
                let fault = stub.put_fault.lock().unwrap().clone();
                if let Some(fault) = fault.filter(|f| path.ends_with(&f.suffix)) {
                    if fault.stored {
                        stub.files.lock().unwrap().insert(path, body.to_vec());
                    }
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
                stub.files.lock().unwrap().insert(path, body.to_vec());
                StatusCode::OK.into_response()
            }
            axum::http::Method::DELETE => {
                if !authorized(&headers, &stub.token) {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                if stub.fail_delete_paths.lock().unwrap().contains(&path) {
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
                if stub.files.lock().unwrap().remove(&path).is_some() {
                    StatusCode::OK.into_response()
                } else {
                    StatusCode::NOT_FOUND.into_response()
                }
            }
            // GET is public.
            _ => {
                let fail_get = stub.fail_get_suffix.lock().unwrap().clone();
                if fail_get.is_some_and(|s| path.ends_with(&s)) {
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
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
            put_fault: Arc::new(Mutex::new(None)),
            fail_get_suffix: Arc::new(Mutex::new(None)),
            fail_delete_paths: Arc::new(Mutex::new(HashSet::new())),
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

    /// A movement event carrying a detection — the type only the sidecar can
    /// record on a store without directories.
    fn object_event(first_pts: u64, size: usize) -> FinishedEvent {
        let mut e = movement_event(first_pts, size);
        e.has_objects = true;
        e.object_classes = vec!["car".to_string()];
        e.detection_details = vec![DetectionDetail {
            class: "car".to_string(),
            confidence: 0.8,
        }];
        e
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

        backend.write_event("cam", &object_event(4_000, 20)).await;

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

        // The upgrade carries the original event's chain flag into the sidecar
        // it rewrites, so the index has to take it too — LocalDisk rebuilds the
        // whole entry here and cannot drift, this one mutates in place.
        let mut upgrade = upgrade_for(5_000);
        upgrade.continues = true;
        backend.upgrade_event("cam", &upgrade).await;

        // The index flipped to Object...
        let e = backend.find_event("cam", 5_000).unwrap();
        assert!(e.continues);
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
        assert_eq!(json["continues"], serde_json::json!(true));
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
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        assert!(backend.find_event("cam", old_pts).is_none());
        assert!(stub.files.lock().unwrap().is_empty());
        assert_eq!(backend.used_bytes.load(Ordering::Relaxed), 0);
    }

    /// The per-sweep deletion cap is the remote store's protection against a
    /// forward clock jump too: the whole index expiring at once must not empty
    /// the bucket in one sweep.
    #[tokio::test]
    async fn prune_caps_how_much_one_sweep_deletes() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        for i in 0..40u64 {
            backend
                .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
                .await;
        }

        // Every event is expired; a quarter of the 40 indexed may go.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 30);

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 22);
    }

    /// An event the store refuses to delete sits at the head of the sweep, so
    /// without the cap exempting known failures it would spend the whole budget
    /// on the same objects every hour and never reach the ones behind them.
    #[tokio::test]
    async fn an_undeletable_event_does_not_block_the_sweep_forever() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        for i in 0..12u64 {
            backend
                .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
                .await;
        }
        // The four oldest videos — the whole cap for a 12-event index.
        {
            let mut refused = stub.fail_delete_paths.lock().unwrap();
            for i in 0..4u64 {
                refused.insert(format!("cam/{}_1000.ts", 1_000_000_000 + i * 1_000_000));
            }
        }

        // First sweep: the entire budget goes on the four that refuse.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 12);

        // Second: retrying those is free, so it reaches four behind them.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(
            backend.query("cam", 0, u64::MAX).len(),
            8,
            "a stuck head of the queue blocked the whole sweep"
        );
    }

    /// Shutdown reaches this backend as a raised flag, and one event here is
    /// several sequential HTTP deletes: a cancelled sweep must issue none.
    #[tokio::test]
    async fn a_cancelled_prune_deletes_nothing() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let old_pts = 1_000_000_000;
        backend
            .write_event("cam", &movement_event(old_pts, 30))
            .await;
        let before = stub.files.lock().unwrap().len();

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(true))
            .await;

        assert!(backend.find_event("cam", old_pts).is_some());
        assert_eq!(stub.files.lock().unwrap().len(), before);
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
        // A lone .ts, as a plain movement event whose sidecar upload failed
        // leaves (an interruption cannot: the sidecar precedes the video).
        stub.files
            .lock()
            .unwrap()
            .insert("cam/9000_1000.ts".to_string(), vec![0u8; 10]);

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;
        let e = backend.find_event("cam", 9_000).unwrap();
        // A confirmed-absent sidecar is what a plain movement event is written
        // with, so the default is a fact about the write path, not a fallback.
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

    /// The sidecar is the only record of an event's type here, so a write that
    /// could not store one must not report success: the in-RAM index would keep
    /// serving the object event until a restart, after which the scan would
    /// call the leftover `.ts` a movement and expire it 12 days early.
    #[tokio::test]
    async fn a_failed_sidecar_fails_the_write_before_the_video() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".json", false);

        let outcome = backend.write_event("cam", &object_event(11_000, 30)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(backend.find_event("cam", 11_000).is_none());
        assert_eq!(backend.used_bytes.load(Ordering::Relaxed), 0);
        // The video was never attempted, so there is no bare .ts for a later
        // scan to call a movement event.
        assert!(!stub.has("cam/11000_1000.ts"));
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert!(scanned.find_event("cam", 11_000).is_none());
    }

    /// A plain movement event is the one event a sidecar-less `.ts` already
    /// scans back as unchanged, so its sidecar is not worth the footage.
    #[tokio::test]
    async fn a_failed_sidecar_does_not_cost_a_movement_event() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".json", false);

        let outcome = backend
            .write_event("cam", &movement_event(15_000, 30))
            .await;

        assert_eq!(outcome, WriteOutcome::Written);
        assert!(stub.has("cam/15000_1000.ts"));
        assert!(!stub.has("cam/15000_1000.json"));
    }

    /// The exemption above is only sound while a sidecar-less scan rebuilds a
    /// movement event *exactly*. This fails the day a field is added to
    /// `sidecar_json` or a scan default changes — which is the whole reason a
    /// blanket "always require the sidecar" rule was tempting.
    #[tokio::test]
    async fn a_movement_event_scans_back_identically_without_its_sidecar() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend
            .write_event("cam", &movement_event(16_000, 30))
            .await;
        let written = backend.find_event("cam", 16_000).unwrap();

        stub.files.lock().unwrap().remove("cam/16000_1000.json");
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;

        assert_eq!(scanned.find_event("cam", 16_000).unwrap(), written);
    }

    /// A PUT that reports failure may still have committed — an upload timeout
    /// or a proxy 5xx says nothing about the origin. The video is therefore
    /// never rolled back: deleting the sidecar of a phantom `.ts` would leave
    /// exactly the bare video this write order exists to prevent.
    #[tokio::test]
    async fn a_video_that_lands_despite_a_failed_put_keeps_its_type() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".ts", true); // committed server-side, 500 to the client

        let outcome = backend.write_event("cam", &object_event(12_000, 30)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(backend.find_event("cam", 12_000).is_none());
        // Both objects are still there, so the next scan adopts the phantom
        // video as the object event it is — not as a movement event on a
        // two-day retention.
        stub.clear_faults();
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", 12_000).unwrap();
        assert_eq!(e.event_type, EventType::Object);
        assert_eq!(e.object_classes, vec!["car".to_string()]);
    }

    /// The mirror case: the video genuinely did not land. The orphan sidecar
    /// left behind is invisible to the scan, which walks `.ts` objects only.
    #[tokio::test]
    async fn an_orphan_sidecar_indexes_nothing() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".ts", false);

        let outcome = backend.write_event("cam", &object_event(17_000, 30)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(stub.has("cam/17000_1000.json"));
        stub.clear_faults();
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert!(scanned.find_event("cam", 17_000).is_none());
    }

    /// Thumbnails are decoration — the UI hides frames that fail to load — so
    /// losing them must not cost the footage.
    #[tokio::test]
    async fn a_failed_thumbnail_is_not_fatal() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".jpg", false);

        let outcome = backend.write_event("cam", &object_event(13_000, 30)).await;

        assert_eq!(outcome, WriteOutcome::Written);
        let entry = backend.find_event("cam", 13_000).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.filmstrip_frames, 0);

        // ...and the index a restart rebuilds agrees with the one in RAM.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", 13_000).unwrap();
        assert_eq!(e.event_type, EventType::Object);
        assert_eq!(e.filmstrip_frames, 0);
        assert!(backend.read_thumbnail("cam", &entry).await.is_err());
    }

    /// 1s after the epoch: older than any retention a test configures.
    const OLD_PTS: u64 = 1_000_000_000;

    /// One flaky GET during startup must not decide a retention class. An
    /// unreadable sidecar leaves the type *unknown*, and an unknown type is
    /// not a movement type — the old scan collapsed both onto Movement, which
    /// deletes an object event twelve days early from the read side alone.
    #[tokio::test]
    async fn an_unreadable_sidecar_is_not_pruned_as_a_movement_event() {
        let (url, stub) = spawn_stub("secret").await;
        backend_for(&url, "secret", 0)
            .write_event("cam", &object_event(OLD_PTS, 30))
            .await;

        // A restart that cannot read the sidecar: 500, not 404.
        stub.fail_gets(".json");
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;

        // Indexed and visible (losing the footage from the UI would be its own
        // bug), but not deleted by the sweep its placeholder type invites.
        assert!(scanned.find_event("cam", OLD_PTS).is_some());
        scanned
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(scanned.find_event("cam", OLD_PTS).is_some());
        assert!(stub.has("cam/1000000000_1000.ts"));

        // The hold is a longer retention, not an immortal one: once every
        // configured age has passed, an event nobody can type still goes.
        scanned.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(scanned.find_event("cam", OLD_PTS).is_none());
        assert!(!stub.has("cam/1000000000_1000.ts"));
    }

    /// `scan` runs once per process, so the sweep is the only place a held
    /// event can ever be typed without a restart.
    #[tokio::test]
    async fn a_prune_tick_resolves_a_held_event() {
        let (url, stub) = spawn_stub("secret").await;
        backend_for(&url, "secret", 0)
            .write_event("cam", &object_event(OLD_PTS, 30))
            .await;
        stub.fail_gets(".json");
        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;
        assert!(backend.has_unknown_type("cam", OLD_PTS));

        // The store recovers; the next sweep reads the sidecar it could not.
        stub.clear_faults();
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        let entry = backend.find_event("cam", OLD_PTS).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.object_classes, vec!["car".to_string()]);
        assert!(!backend.has_unknown_type("cam", OLD_PTS));

        // Typed again, it prunes on its own retention: kept as an object...
        backend.prune(1, u64::MAX, 1, &AtomicBool::new(false)).await;
        assert!(backend.find_event("cam", OLD_PTS).is_some());
        // ...and gone once the object retention itself expires.
        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(backend.find_event("cam", OLD_PTS).is_none());
    }

    /// Valid JSON that names no type is not a movement event either — and
    /// unlike a failed read it will never resolve itself, so it must be held
    /// rather than quietly given a two-day retention.
    #[tokio::test]
    async fn a_sidecar_naming_no_type_is_held_not_assumed() {
        let (url, stub) = spawn_stub("secret").await;
        stub.files
            .lock()
            .unwrap()
            .insert("cam/1000000000_1000.ts".to_string(), vec![0u8; 10]);
        stub.files
            .lock()
            .unwrap()
            .insert("cam/1000000000_1000.json".to_string(), b"{}".to_vec());

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;
        assert!(backend.has_unknown_type("cam", OLD_PTS));

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        // A re-read says the same thing, so the hold survives the sweep.
        assert!(backend.find_event("cam", OLD_PTS).is_some());
        assert!(backend.has_unknown_type("cam", OLD_PTS));
    }

    /// Bytes that are not JSON are a failed read, not an absent sidecar: the
    /// distinction is the difference between holding the event and pruning it
    /// as a movement.
    #[tokio::test]
    async fn an_unparsable_sidecar_is_held_not_treated_as_absent() {
        let (url, stub) = spawn_stub("secret").await;
        stub.files
            .lock()
            .unwrap()
            .insert("cam/1000000000_1000.ts".to_string(), vec![0u8; 10]);
        stub.files
            .lock()
            .unwrap()
            .insert("cam/1000000000_1000.json".to_string(), b"not json".to_vec());

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;
        assert!(backend.has_unknown_type("cam", OLD_PTS));

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(backend.find_event("cam", OLD_PTS).is_some());
    }

    /// Eviction order is the other decision that reads the event type. A held
    /// event's placeholder says movement; evicting on that would spend the
    /// footage the hold exists to protect.
    #[tokio::test]
    async fn budget_eviction_tiers_an_unknown_type_with_the_objects() {
        let (url, stub) = spawn_stub("secret").await;
        let writer = backend_for(&url, "secret", 0);
        writer.write_event("cam", &object_event(1_000, 40)).await;
        writer.write_event("cam", &movement_event(2_000, 40)).await;

        // The object event's sidecar is unreadable on the next start...
        stub.fail_gets("1000_1000.json");
        let backend = backend_for(&url, "secret", 60);
        backend.scan().await;
        assert!(backend.has_unknown_type("cam", 1_000));

        // ...so the budget must still evict the genuine movement event first,
        // even though the held one is older and labelled movement too.
        backend.guard_free_space("cam", 0).await;
        assert!(backend.find_event("cam", 2_000).is_none());
        assert!(backend.find_event("cam", 1_000).is_some());
    }

    /// A thumbnail gap stops the uploads: the scan counts frames contiguously
    /// from 0, so continuing past a failure would index frames it will never
    /// see again and leave the extra objects stranded on the host.
    #[tokio::test]
    async fn a_thumbnail_gap_stops_the_filmstrip() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let mut event = movement_event(18_000, 30);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        stub.fail_puts("_thumb_1.jpg", false);

        backend.write_event("cam", &event).await;

        assert_eq!(
            backend.find_event("cam", 18_000).unwrap().filmstrip_frames,
            1
        );
        assert!(stub.has("cam/18000_1000_thumb_0.jpg"));
        assert!(!stub.has("cam/18000_1000_thumb_2.jpg"));
    }

    /// The movement exemption rests on the sidecar of a plain movement event
    /// saying nothing the scan does not already assume. Pinning the literal
    /// bytes catches a field added to `sidecar_json` — which the entry-equality
    /// test above cannot, since a new field would be default in both halves.
    #[test]
    fn a_plain_movement_sidecar_carries_nothing_but_its_type() {
        assert_eq!(
            sidecar_json(EventType::Movement, None, None, &[], false),
            r#"{"detections":[],"event_type":"movement"}"#
        );
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
                    delete_failed: false,
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
