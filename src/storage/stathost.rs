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
//! The in-RAM index is not this backend's own: it is an
//! [`EventIndex`](crate::storage::event_index::EventIndex) like local disk's,
//! keyed by the stem ([`EventKey`]) instead of by a path, and the retention
//! sweep and space-pressure eviction run through the shared skeletons in
//! [`event_index`](crate::storage::event_index). What is genuinely this
//! backend's is below.
//!
//! Notable divergences from [`LocalDiskBackend`], all deliberate:
//!
//! * **Retention-by-space is a client-side budget.** The client can't see the
//!   server's disk, so `max_stored_bytes` caps tracked usage; when it is
//!   exceeded the oldest events are evicted cheapest tier first, the same order
//!   and the same skeleton as the local emergency prune — but with the opposite
//!   failure policy, for the reason
//!   [`EvictionPolicy`](crate::storage::event_index::EvictionPolicy) gives. The
//!   disk-shaped `min_free_bytes` guard argument is ignored here.
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
//!   an orphan `.json` is invisible to the scan's index, which walks `.ts`
//!   objects only (the scan collects it separately, see below), and a phantom
//!   `.ts` still has the sidecar that types it correctly.
//!   stathost's uploads are atomic server-side, so a truncated object can't be
//!   served; a zero-byte `.ts` in the listing still gets a warning at scan time.
//!   There is nothing here matching local mode's fsync of the video and of the
//!   directory it is committed into: this backend owns no filesystem, and a
//!   `PUT` that has been acknowledged is durable or not by the *server's*
//!   rules. Nothing camon can do from the client side changes that.
//! * **A write is an overwrite.** `PUT` of a key that already exists replaces
//!   it, so re-writing a stem re-writes one event rather than creating a
//!   second: the index replaces the entry under that stem, the budget is
//!   charged the difference, and thumbnails the shorter event has no frame for
//!   are deleted. See [`event_key`] for what identifies an event here.
//! * **A read resolves by stem, type and all.** An API request names an event
//!   by its whole key ([`EventRef`]), and
//!   [`find_event`](StathostBackend::find_event) matches only the stem part of
//!   it: the same objects answer to both types across a movement→object
//!   upgrade, so a link made before one still plays after it. Local disk, where
//!   the type is a directory, has to match all three.
//! * **Metadata whose video never landed is collected at startup.** Nothing
//!   else can: the index walks `.ts` objects, and an event's siblings are only
//!   deleted with it. See [`StathostBackend::sweep_orphaned_metadata`].
//! * **An unreadable sidecar is not a movement event.** The scan applies the
//!   movement default only to a *confirmed* 404; anything else — a transport
//!   failure, unparsable bytes, valid JSON naming no type — leaves the type
//!   unknown. Such an event is still indexed and served, but every decision
//!   that would need its type errs toward keeping it: age-based pruning
//!   measures it against the longest configured retention, and budget eviction
//!   tiers it with the objects. The prune tick re-reads its sidecar, which is
//!   the only in-process retry there is — [`scan`](StathostBackend::scan) runs
//!   once at startup.
//! * **The startup scan fans out.** It blocks startup by design — the index,
//!   the byte budget and the orphan sweep's safety all depend on it finishing
//!   before the first camera writes — and it needs one sidecar GET per stored
//!   event, so awaiting them one at a time made "no camera is recording yet" a
//!   function of the archive's size. Reads and orphan probes run
//!   [`SCAN_CONCURRENCY`] at a time instead. What that changes is when requests
//!   are *issued*; results are still consumed in listing order on one task, and
//!   each request still decides its own 404-versus-failure.
//! * **`read_video` streams the object with Range support.** The body is never
//!   fully buffered; a forwarded `Range` header yields a `206`. A `200` to a
//!   range request is legal HTTP and degrades to streaming the full body (the
//!   client replays from the start).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::RwLock;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use serde::Deserialize;

use crate::buffer::warm::{EventUpgrade, FinishedEvent};
use crate::buffer::GopSegment;
use crate::config::StathostConfig;
use crate::locks::LockExt;
use crate::storage::backend::{
    RangeRequest, ServedRange, ThumbnailError, VideoStream, WarmStorageBackend, WriteOutcome,
};
use crate::storage::event_index::{
    evict_tiers, sweep_expired, EmergencyOutcome, EventIdentity, EventIndex, EvictionPolicy,
    Removal,
};
use crate::storage::warm_index::{
    parse_event_filename, parse_sidecar_json, sidecar_json, wall_clock_ns, SidecarData,
};
use crate::storage::{EventRef, EventType, WarmEventEntry};

const NANOS_PER_MS: u64 = 1_000_000;

/// This backend's event identity: the stem every one of an event's keys is
/// built from, `{camera_id}/{start_pts_ns}_{duration_ms}.*`. See
/// [`EventIdentity`] for why local disk spells this differently and why neither
/// spelling can be imposed on the other.
type EventKey = (u64, u32);

fn event_key(entry: &WarmEventEntry) -> EventKey {
    EventIdentity::of(entry)
}

fn key_stem(key: EventKey) -> String {
    format!("{}_{}", key.0, key.1)
}

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

/// How many requests [`StathostBackend::scan`] keeps in flight — its sidecar
/// reads, and the orphan sweep's probes and deletes.
///
/// The scan is awaited by `init_storage` *before* the first camera is spawned,
/// so every round trip it makes is time nothing is recording. Its work is one
/// small GET per stored event, and those are latency-bound rather than
/// bandwidth-bound — a sidecar is a few hundred bytes — so awaiting them one at
/// a time makes startup a function of the archive's size: a bucket holding a
/// few thousand events costs that many round trips, minutes of them on any link
/// that is not a LAN. The per-request timeouts bound each one, not the total.
///
/// 16 is sized against what the far end is. stathost is a small single-process
/// static file host, typically the same box camon runs on or next to, and each
/// in-flight request is one HTTP/1.1 connection — so this is also the number of
/// sockets camon opens at once. Deep enough that the round trip is hidden
/// (16 × a 40 ms RTT ≈ 400 events/s, so the same few thousand events finish in
/// seconds), shallow enough not to be a burst worth calling a load, and it
/// leaves the host able to answer whatever else is asking. Doubling it would
/// save a few seconds of startup for twice the connections, against a server
/// that is about to start absorbing this process's uploads.
///
/// Note that camon's other remote dependency, Ollama, is deliberately held to
/// exactly *one* in-flight request. That limit is about a GPU that degrades
/// under parallel load; serving static files has no such property.
const SCAN_CONCURRENCY: usize = 16;

/// The remote warm store: an HTTP client over the shared in-RAM index.
pub struct StathostBackend {
    http: Http,
    /// Client-side storage budget in bytes; 0 means unlimited. Measured against
    /// [`EventIndex::used_bytes`], which the index maintains as the sum of what
    /// it holds — the two cannot drift.
    max_stored_bytes: u64,
    events: EventIndex<EventKey>,
    /// Events whose sidecar the scan could not read, per camera. Their
    /// [`WarmEventEntry::event_type`] is a placeholder, not a fact — see
    /// [`Self::mark_unknown_type`]. This has no local-disk counterpart: there
    /// the type is the directory, so it cannot be unreadable.
    unknown_type: HashMap<String, RwLock<HashSet<EventKey>>>,
}

impl StathostBackend {
    pub fn new(config: &StathostConfig, camera_ids: &[String]) -> Self {
        let base = format!(
            "{}/{}",
            config.url.trim_end_matches('/'),
            config.bucket.trim_matches('/')
        );
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
            events: EventIndex::new(camera_ids),
            unknown_type: camera_ids
                .iter()
                .map(|id| (id.clone(), RwLock::new(HashSet::new())))
                .collect(),
        }
    }

    fn used(&self) -> u64 {
        self.events.used_bytes()
    }

    /// Record that this event's type could not be established. `event_type` on
    /// the entry is then a display placeholder, not a retention class: pruning
    /// it as a movement would delete an object event twelve days early. Instead
    /// [`WarmStorageBackend::prune`] gives it the longest *configured*
    /// retention, which cannot expire before its own whatever its true type is,
    /// and still expires — a permanently unreadable sidecar would otherwise pin
    /// its footage forever on a store whose budget is unlimited by default.
    fn mark_unknown_type(&self, camera_id: &str, key: EventKey) {
        if let Some(lock) = self.unknown_type.get(camera_id) {
            lock.write_recover().insert(key);
        }
    }

    fn has_unknown_type(&self, camera_id: &str, key: EventKey) -> bool {
        self.unknown_type
            .get(camera_id)
            .is_some_and(|lock| lock.read_recover().contains(&key))
    }

    /// Drop the marker once the type is settled. Only [`Self::scan`] ever sets
    /// one, and it runs once per process, so in practice this fires where a
    /// type becomes a fact — `write_event` and `upgrade_event`, neither of which
    /// may leave a "type unknown" marker on an event it just proved.
    ///
    /// A marker for an event that has *left* the index is collected by
    /// [`Self::resolve_unknown_types`] on the next prune tick instead: removal
    /// runs inside the shared index, which knows nothing about this backend's
    /// side table, and nothing between the two reads a marker whose entry is
    /// gone (both the age filter and eviction only ever walk indexed entries).
    fn clear_unknown_type(&self, camera_id: &str, key: EventKey) {
        if let Some(lock) = self.unknown_type.get(camera_id) {
            lock.write_recover().remove(&key);
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

    /// [`Self::read_sidecar`] with the event carried through, so the scan's
    /// fan-out can pair each result with what it belongs to.
    async fn read_sidecar_for(&self, event: ScannedEvent) -> (ScannedEvent, SidecarRead) {
        let read = self.read_sidecar(&event.camera_id, &event.stem).await;
        (event, read)
    }

    fn ts_key(camera_id: &str, entry: &WarmEventEntry) -> String {
        format!(
            "{camera_id}/{}_{}.ts",
            entry.start_pts_ns, entry.duration_ms
        )
    }

    /// Delete every object belonging to one event: **thumbnails, then the
    /// video, then the sidecar**. The video's outcome is the event's outcome —
    /// only once it is gone (or was already) may the index entry go, and only
    /// then is the sidecar deleted at all.
    ///
    /// Local disk unlinks metadata first and the `.ts` last, and this used to
    /// copy that. The reason it does not is that the two backends fail
    /// differently on either side of the video:
    ///
    /// * *Before* the video, only thumbnails go. They are decoration and carry
    ///   nothing; the local order's whole point — that an interruption must not
    ///   strand metadata nothing looks for — is preserved for them by the
    ///   startup sweep, which local disk cannot have (see
    ///   [`Self::sweep_orphaned_metadata`]).
    /// * *After* the video, the sidecar. Deleting it first would mean a refused
    ///   video delete — the common failure, and the one that persists — leaves
    ///   a `.ts` with no record of its type, which the next scan reads back as
    ///   a plain **movement**. That is not automatically the shortest
    ///   retention: `continuous_retention_days` defaults to 1 against
    ///   movement's 2, and all three are freely configurable in any order, so
    ///   the reclassification can just as easily extend an expired event's life
    ///   as shorten it. Local disk has no such exposure — its type is the
    ///   directory, which a delete never touches. Keeping the sidecar until the
    ///   video is actually gone means a failed delete is retried against the
    ///   event's true retention, and a crash between the two leaves an orphan
    ///   sidecar the next startup collects.
    ///
    /// Thumbnail failures are not a reason to keep the footage: holding an
    /// expired recording because a thumbnail resisted breaks a larger promise
    /// than the kilobytes it saves, and what is left is collected once the
    /// video is gone.
    async fn delete_event_objects(&self, camera_id: &str, entry: &WarmEventEntry) -> Removal {
        let stem = key_stem(event_key(entry));
        for i in 0..entry.filmstrip_frames {
            let _ = self
                .http
                .delete(&format!("{camera_id}/{stem}_thumb_{i}.jpg"))
                .await;
        }
        let removal = match self.http.delete(&format!("{camera_id}/{stem}.ts")).await {
            DeleteOutcome::Deleted => Removal::Deleted,
            DeleteOutcome::Missing => Removal::Missing,
            DeleteOutcome::Failed => {
                // Per-event detail is debug: a store that is down fails every
                // delete of every sweep, and both callers report one aggregate
                // warning per pass.
                tracing::debug!(camera = %camera_id, stem = %stem,
                    "failed to delete event video from stathost, will retry on a later tick");
                return Removal::Failed;
            }
        };
        let _ = self.http.delete(&format!("{camera_id}/{stem}.json")).await;
        removal
    }

    /// Delete the filmstrip frames a rewrite of this stem no longer has, and
    /// leave the index describing what is actually on the host.
    ///
    /// Every other object of a rewritten event is overwritten in place by its
    /// `PUT`; thumbnails past the new frame count are the one thing that has to
    /// be removed, since the scan counts frames contiguously from 0 and would
    /// otherwise hand this event the previous write's tail.
    ///
    /// Deleting **top down** is what makes a failure survivable. A delete that
    /// fails stops the trim, and everything above the frame that refused is
    /// already gone, so what remains is exactly `0..=refused` — still
    /// contiguous, so the entry can simply claim that many frames and agree
    /// with both the host and the next scan. Nothing is stranded: those frames
    /// are inside the range [`Self::delete_event_objects`] deletes, so they go
    /// with the event. Bottom-up would leave a hole, and everything above it
    /// invisible to the scan and to the event's own delete — a permanent leak
    /// out of one transient failure.
    ///
    /// The write is not failed over this. It is decoration, the footage is
    /// already stored, and the index is honest either way.
    async fn trim_thumbnails(&self, camera_id: &str, key: EventKey, keep: usize, had: usize) {
        let stem = key_stem(key);
        for i in (keep..had).rev() {
            if matches!(
                self.http
                    .delete(&format!("{camera_id}/{stem}_thumb_{i}.jpg"))
                    .await,
                DeleteOutcome::Failed
            ) {
                tracing::warn!(camera = %camera_id, stem = %stem, frame = i,
                    "could not delete a filmstrip frame this event no longer has; \
                     it stays part of the event and is deleted with it");
                self.events
                    .update(camera_id, key, |entry| entry.filmstrip_frames = i + 1);
                return;
            }
        }
    }

    /// Delete metadata whose video is not on the host: the sidecar (and any
    /// thumbnails) of an event whose `.ts` upload never landed.
    ///
    /// Nothing else collects them. The index is built from `.ts` objects, so
    /// such an object is never indexed, never counted against the budget, and
    /// never deleted alongside an event — on a flaky uplink they accumulate for
    /// the life of the bucket, and the budget drifts further below true remote
    /// usage with every one. The listing the scan already fetched is all it
    /// takes to find them.
    ///
    /// The sidecar goes up *before* the video, so "no video" is also what a
    /// write still in progress looks like. Four things stand between this and
    /// one, three of them absolute:
    ///
    /// * [`WarmStorageBackend::scan`] is awaited once, from `init_storage`,
    ///   before the backend is handed to any warm writer — so no upload from
    ///   *this* process has started, and none is pending either: a write that
    ///   fails is dropped with an error, never queued (the writer retries
    ///   `NoSpace` alone, an outcome this backend never returns).
    /// * Only camera prefixes this process owns are touched — the same ones the
    ///   index is built from — and only names this backend writes, parsing as
    ///   an event stem.
    /// * Every candidate is re-checked against the host immediately before it
    ///   is deleted, and deleted only on a *confirmed* absence. The listing is
    ///   a snapshot taken before the (possibly long) indexing pass, and it is
    ///   not the only writer's snapshot: another camon on the same camera id,
    ///   or a `PUT` that outlived the process which issued it (a client-side
    ///   timeout says nothing about what the origin commits — the same fact
    ///   that stops this backend rolling anything back), can land a video after
    ///   it was taken. A failure to find out is not an absence and never
    ///   deletes.
    ///
    /// **The residual race, stated rather than papered over:** between that
    /// re-check answering "absent" and the `DELETE` arriving, a `PUT` already
    /// in flight can commit. stathost has no conditional delete, so the window
    /// cannot be closed from the client side — only narrowed to one request
    /// round-trip, which is what the re-check does. Its cost if it is ever lost
    /// is one video with no type record, which the scan reads as a plain
    /// movement: the same state this backend already tolerates whenever a
    /// sidecar cannot be stored, not a lost recording. Reaching it needs a
    /// second writer on a camera id this process owns, which is unsupported for
    /// several other reasons already — two such instances prune and evict each
    /// other's events.
    ///
    /// Local disk deliberately has no equivalent sweep: there, metadata is
    /// written under its final name *before* the `.ts.tmp` is committed, so
    /// "metadata with no video" is also what an event awaiting startup recovery
    /// looks like, and collecting it would destroy the sidecar of footage about
    /// to be salvaged. This backend stages nothing and recovers nothing.
    async fn sweep_orphaned_metadata(&self, items: &[ListEntry], all_paths: &HashSet<&str>) {
        // Grouped by stem before anything is asked of the host: a failed upload
        // orphans a sidecar and every filmstrip frame together, and all of them
        // turn on the one question the probe answers — is the video there? One
        // probe settles the whole stem, where probing per object asked it again
        // for each frame.
        let mut orphans: Vec<(String, Vec<String>)> = Vec::new();
        let mut by_stem: HashMap<String, usize> = HashMap::new();
        for item in items {
            let Some((camera_id, stem)) = split_metadata_key(&item.path) else {
                continue;
            };
            if !self.events.owns_camera(camera_id) || parse_event_filename(stem).is_none() {
                continue;
            }
            let ts_key = format!("{camera_id}/{stem}.ts");
            if all_paths.contains(ts_key.as_str()) {
                continue;
            }
            match by_stem.get(&ts_key) {
                Some(&at) => orphans[at].1.push(item.path.clone()),
                None => {
                    by_stem.insert(ts_key.clone(), orphans.len());
                    orphans.push((ts_key, vec![item.path.clone()]));
                }
            }
        }

        // Unordered: these tallies are sums, and one stem's outcome never
        // depends on another's.
        let (deleted, landed, failed) = futures_util::stream::iter(orphans)
            .map(|(ts_key, paths)| self.sweep_one_stem(ts_key, paths))
            .buffer_unordered(SCAN_CONCURRENCY)
            .fold((0usize, 0usize, 0usize), |acc, one| async move {
                (acc.0 + one.0, acc.1 + one.1, acc.2 + one.2)
            })
            .await;
        if deleted > 0 {
            tracing::info!(
                deleted,
                "deleted stathost metadata whose event video never landed"
            );
        }
        if landed > 0 {
            tracing::info!(
                landed,
                "stathost events whose video appeared after the listing was taken; metadata kept"
            );
        }
        if failed > 0 {
            tracing::warn!(
                failed,
                "could not collect orphaned stathost metadata; the next startup retries"
            );
        }
    }

    /// One stem's share of [`Self::sweep_orphaned_metadata`]: probe the video
    /// once, and delete this stem's metadata objects only if the probe came
    /// back with a confirmed absence. Returns `(deleted, landed, failed)`.
    ///
    /// The probe and the deletes it authorises stay in one future, so running
    /// stems concurrently does not widen the window the module header calls the
    /// residual race: it is still the one round trip between the two requests.
    async fn sweep_one_stem(&self, ts_key: String, paths: Vec<String>) -> (usize, usize, usize) {
        match self.http.probe_exists(&ts_key).await {
            // Confirmed absent: an orphan, as of one request ago.
            Ok(false) => {
                let (mut deleted, mut failed) = (0usize, 0usize);
                for path in &paths {
                    match self.http.delete(path).await {
                        DeleteOutcome::Deleted => deleted += 1,
                        DeleteOutcome::Missing => {}
                        DeleteOutcome::Failed => failed += 1,
                    }
                }
                (deleted, 0, failed)
            }
            // It landed after the listing was taken — not an orphan at all.
            Ok(true) => (0, 1, 0),
            // Could not find out. Nothing is deleted on a maybe.
            Err(_) => (0, 0, 1),
        }
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
        let held: Vec<EventKey> = match self.unknown_type.get(camera_id) {
            Some(lock) => lock.read_recover().iter().copied().collect(),
            None => return,
        };
        let mut resolved = 0u64;
        for key in held {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if !self.events.contains(camera_id, key) {
                self.clear_unknown_type(camera_id, key);
                continue;
            }
            let sidecar = match self.read_sidecar(camera_id, &key_stem(key)).await {
                SidecarRead::Parsed(s) => Some(s),
                SidecarRead::Absent => None,
                // Still unreadable, or still naming no type: keep the hold.
                SidecarRead::Unreadable | SidecarRead::Typeless(_) => continue,
            };
            self.events.update(camera_id, key, |entry| {
                apply_sidecar(entry, sidecar.as_ref())
            });
            self.clear_unknown_type(camera_id, key);
            resolved += 1;
        }
        if resolved > 0 {
            tracing::info!(camera = %camera_id, resolved,
                "read event types that an earlier scan could not; normal retention resumes");
        }
    }

    /// Enforce the client-side storage budget: while tracked usage exceeds
    /// `max_stored_bytes`, delete the oldest events cheapest-tier-first. No-op
    /// when the budget is unlimited.
    ///
    /// This is the local low-space guard's counterpart, sharing its skeleton and
    /// differing only in [`EvictionPolicy`] — which is where the argument for
    /// stopping on a failure and demoting rather than excluding is written down.
    async fn enforce_budget(&self, camera_id: &str) {
        if self.max_stored_bytes == 0 || self.used() <= self.max_stored_bytes {
            return;
        }
        let outcome = evict_tiers(
            &self.events,
            EvictionPolicy {
                skip_failed: false,
                stop_on_failure: true,
                reason: "budget prune: deleted event to stay under max_stored_bytes",
            },
            // An event of unknown type is evicted with the objects, the tier
            // kept longest: its placeholder says movement, and evicting on that
            // guess would throw away footage this whole path exists to keep.
            |cam, entry| {
                if self.has_unknown_type(cam, event_key(entry)) {
                    EventType::Object
                } else {
                    entry.event_type
                }
            },
            || self.used() <= self.max_stored_bytes,
            |cam, entry| async move { self.delete_event_objects(&cam, &entry).await },
        )
        .await;
        if outcome != EmergencyOutcome::default() {
            tracing::warn!(
                camera = %camera_id,
                deleted = outcome.deleted,
                missing = outcome.missing,
                failed = outcome.failed,
                "budget prune complete"
            );
        }
    }
}

#[async_trait]
impl WarmStorageBackend for StathostBackend {
    async fn write_event(&self, camera_id: &str, event: &FinishedEvent) -> WriteOutcome {
        // Checked, not cast: the index entry — and so [`event_key`], the
        // identity every object of this event is keyed by — holds a `u32`,
        // while the duration is computed as `u64`. A silent truncation would
        // put the video under a stem no index entry names. It takes an event of
        // over 49 days for that, which `max_event_duration_secs` makes
        // unreachable; the conversion is here so the invariant is stated rather
        // than assumed.
        let Ok(duration_ms) = u32::try_from(event.duration_ns() / NANOS_PER_MS) else {
            tracing::error!(
                camera = %camera_id,
                first_pts = event.first_pts,
                duration_ns = event.duration_ns(),
                "dropping event: duration does not fit the storage key"
            );
            return WriteOutcome::Failed;
        };
        let key = (event.first_pts, duration_ms);
        let stem = key_stem(key);
        // Contiguous, unlike the local backend's write, by choice rather than
        // by necessity: reqwest can stream a body from the segments as they
        // are, but only as a second request shape for the server to accept —
        // and a wire change that cannot be tested against the real stathost is
        // not worth the megabytes. Held as `Bytes` so the retry below shares
        // the buffer instead of doubling the event's footprint.
        let data = Bytes::from(concatenate_segments(&event.segments, event.total_bytes));
        let file_size = data.len() as u64;
        let event_type = event.event_type();

        // Step 1: the sidecar, before the video. It is the sole carrier of the
        // event type, so an event whose sidecar is missing is not a slightly
        // poorer event — it is the wrong kind of event, expiring on the wrong
        // retention after the next scan. One retry, then fail the write before
        // the video is uploaded at all.
        let sidecar_key = format!("{camera_id}/{stem}.json");
        let sidecar = Bytes::from(
            sidecar_json(
                Some(event_type),
                event.backend.as_deref(),
                event.model.as_deref(),
                &event.detection_details,
                event.continues,
            )
            .into_bytes(),
        );
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
                    // Copied, not shared, because the filmstrip is typed
                    // `Arc<Vec<Vec<u8>>>` where the event assembles it. Making
                    // it shareable all the way here would mean retyping it at
                    // the source for four JPEGs an event.
                    let body = Bytes::from(jpeg.clone());
                    if self.http.put(&key, body).await.is_err() {
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

        let replaced = self.events.insert(
            camera_id,
            WarmEventEntry {
                start_pts_ns: event.first_pts,
                duration_ms,
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
        self.clear_unknown_type(camera_id, key);

        if let Some(previous) = replaced {
            self.trim_thumbnails(camera_id, key, filmstrip_frames, previous.filmstrip_frames)
                .await;
        }

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
        let key = (upgrade.start_pts_ns, upgrade.duration_ms);
        if !self.events.contains(camera_id, key) {
            tracing::warn!(
                camera = %camera_id,
                start_pts_ns = upgrade.start_pts_ns,
                "event not indexed, skipping object upgrade \
                 (detections remain available in the detection store)"
            );
            return;
        }
        let stem = key_stem(key);
        let sidecar = sidecar_json(
            Some(EventType::Object),
            Some(&upgrade.backend),
            Some(&upgrade.model),
            &upgrade.detections,
            upgrade.continues,
        );
        if self
            .http
            .put(
                &format!("{camera_id}/{stem}.json"),
                Bytes::from(sidecar.into_bytes()),
            )
            .await
            .is_err()
        {
            tracing::error!(camera = %camera_id, stem = %stem,
                "failed to upload upgraded sidecar to stathost, aborting upgrade");
            return;
        }

        self.events.update(camera_id, key, |entry| {
            entry.event_type = EventType::Object;
            entry.object_classes = upgrade.object_classes.clone();
            entry.detections = upgrade.detections.clone();
            entry.backend = Some(upgrade.backend.clone());
            entry.model = Some(upgrade.model.clone());
            // The sidecar just written carries the upgrade's `continues`;
            // the index has to say the same thing (LocalDisk rebuilds the
            // whole entry here, which is where this was being lost).
            entry.continues = upgrade.continues;
        });
        // The type is now established. An upgrade only ever targets an event
        // written by this process, so it cannot reach one the scan held — but
        // this and `write_event` are the two places a type becomes a fact, and
        // neither may leave a "type unknown" marker behind it.
        self.clear_unknown_type(camera_id, key);

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
        for camera_id in self.events.camera_ids() {
            if stop() {
                break;
            }
            // First give held events a chance to be typed, so one that resolves
            // is pruned on its real retention in this same sweep. This is also
            // what drops holds on events a previous sweep deleted: unindexing
            // does not clear them, and nothing between here and there reads one.
            self.resolve_unknown_types(camera_id, cancel).await;

            let expired = self.events.expired_for_sweep(camera_id, now_ns, |e| {
                if self.has_unknown_type(camera_id, event_key(e)) {
                    unknown_max_age
                } else {
                    max_age(e.event_type)
                }
            });
            if expired.is_empty() {
                continue;
            }

            let outcome =
                sweep_expired(&self.events, camera_id, expired, stop, |entry| async move {
                    self.delete_event_objects(camera_id, &entry).await
                })
                .await;

            if outcome.deleted > 0 {
                tracing::info!(
                    camera = %camera_id,
                    deleted = outcome.deleted,
                    "pruned expired warm events"
                );
            }
            if outcome.failed > 0 {
                tracing::warn!(
                    camera = %camera_id,
                    failed = outcome.failed,
                    "expired warm events are still on stathost after a failed delete, \
                     kept indexed for the next prune tick (stems at debug level)"
                );
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

        // Everything the listing alone settles, in listing order. Only the
        // sidecar reads below need the network.
        let mut pending: Vec<ScannedEvent> = Vec::new();
        for item in &items {
            let Some((camera_id, stem)) = split_ts_key(&item.path) else {
                continue;
            };
            if !self.events.owns_camera(camera_id) {
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
            let mut filmstrip_frames = 0usize;
            while all_paths
                .contains(format!("{camera_id}/{stem}_thumb_{filmstrip_frames}.jpg").as_str())
            {
                filmstrip_frames += 1;
            }
            pending.push(ScannedEvent {
                camera_id: camera_id.to_string(),
                stem: stem.to_string(),
                start_pts_ns,
                duration_ms,
                file_size: item.size,
                filmstrip_frames,
            });
        }

        let total = pending.len();
        let mut unknown_type = 0usize;
        let mut typeless = 0usize;

        // `buffered`, not `buffer_unordered`: the reads overlap, but their
        // results are handed back in listing order, so the index is built by
        // exactly the sequence of insertions a serial scan made and the
        // warnings below keep a stable order. Every insertion still happens
        // here, on one task — the fan-out covers the request, not the index.
        let mut reads = futures_util::stream::iter(pending)
            .map(|event| self.read_sidecar_for(event))
            .buffered(SCAN_CONCURRENCY);

        while let Some((event, read)) = reads.next().await {
            // Only a confirmed absence means "movement" — that is the one event
            // written without a sidecar. Concurrency does not touch this: the
            // 404 that says "absent" and the error that says "could not find
            // out" are both decided inside the one request they belong to.
            let (sidecar, type_known) = match read {
                SidecarRead::Parsed(s) => (Some(s), true),
                SidecarRead::Absent => (None, true),
                SidecarRead::Unreadable => (None, false),
                SidecarRead::Typeless(s) => {
                    // Deterministic, so no later scan will clear this by
                    // itself: name the object so it can actually be fixed.
                    tracing::warn!(path = %format_args!("{}/{}.ts", event.camera_id, event.stem),
                        "stathost sidecar names no event type; retention falls back to the \
                         longest configured age until the sidecar is repaired");
                    typeless += 1;
                    (Some(s), false)
                }
            };

            let mut entry = WarmEventEntry {
                start_pts_ns: event.start_pts_ns,
                duration_ms: event.duration_ms,
                event_type: EventType::Movement,
                file_size: event.file_size,
                object_classes: Vec::new(),
                backend: None,
                model: None,
                detections: Vec::new(),
                filmstrip_frames: event.filmstrip_frames,
                continues: false,
                recovered: false,
                delete_failed: false,
            };
            apply_sidecar(&mut entry, sidecar.as_ref());
            self.events.insert(&event.camera_id, entry);
            if !type_known {
                self.mark_unknown_type(&event.camera_id, (event.start_pts_ns, event.duration_ms));
                unknown_type += 1;
            }
        }

        self.sweep_orphaned_metadata(&items, &all_paths).await;

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
        // client-side. What an interrupted upload can leave behind is metadata
        // without a video, which [`StathostBackend::sweep_orphaned_metadata`]
        // collects during the scan — it needs the listing the scan already has.
    }

    fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        self.events.query(camera_id, from_ns, to_ns)
    }

    fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        self.events.newest_event_end_ns(camera_id)
    }

    /// Resolved by stem — start and duration — with the event type in the key
    /// deliberately ignored.
    ///
    /// This is the one place the two backends read a request differently, and it
    /// follows from their layouts. Here the type lives *inside* the sidecar:
    /// a movement→object upgrade rewrites that one object in place and moves
    /// nothing, so the same bytes answer to both types over their lifetime and
    /// the stem is the whole identity ([`EventIdentity`]). Honoring the type
    /// would 404 every URL a client already holds the moment an event is
    /// upgraded — a link taken from the event list, or the playlist a player is
    /// part way through — while pointing at footage that is still right there.
    /// Local disk cannot do the same: the type is a directory there, so two
    /// types under one stem are two different files.
    fn find_event(&self, camera_id: &str, event: EventRef) -> Option<WarmEventEntry> {
        self.events
            .find(camera_id, (event.start_pts_ns, event.duration_ms))
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

    /// `Bytes` rather than `Vec<u8>`: an upload that has to be retried needs a
    /// second body, and a whole event is tens of megabytes to hold twice.
    async fn put(&self, path: &str, body: Bytes) -> Result<(), reqwest::Error> {
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

    /// Whether an object is on the host, without fetching it: a one-byte ranged
    /// GET, so probing a video worth tens of megabytes costs one byte (the body
    /// is dropped unread in any case). Range support is required of the server
    /// anyway — playback depends on it — where `HEAD` is not.
    ///
    /// `Err` means "could not find out", which a caller that deletes on absence
    /// must not treat as one. A zero-byte object answers `416` and so reads as
    /// unknown rather than absent, which errs the safe way for the one caller.
    async fn probe_exists(&self, path: &str) -> Result<bool, reqwest::Error> {
        let resp = self
            .client
            .get(self.url(path))
            .bearer_auth(&self.token)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        resp.error_for_status()?;
        Ok(true)
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

/// One `.ts` object [`StathostBackend::scan`] is going to index, carrying
/// everything the listing alone settles about it. Splitting that off from the
/// sidecar read is what lets the reads overlap while the index is still built
/// in listing order.
///
/// The names are owned rather than borrowed from the listing: a borrowed
/// `ScannedEvent<'a>` makes the scan's fan-out closure higher-ranked over `'a`,
/// which rustc cannot infer for a closure returning a future. Two short strings
/// per stored event, once at startup, buys a readable fan-out.
struct ScannedEvent {
    camera_id: String,
    stem: String,
    start_pts_ns: u64,
    duration_ms: u32,
    file_size: u64,
    filmstrip_frames: usize,
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

/// Split a metadata object key — `{camera_id}/{stem}.json` or
/// `{camera_id}/{stem}_thumb_{i}.jpg` — into `(camera_id, stem)`, where the
/// stem is the one its `.ts` sibling carries. `None` for anything else,
/// including a `.jpg` that is not a numbered filmstrip frame: what this matches
/// is what [`StathostBackend::sweep_orphaned_metadata`] deletes, so it matches
/// only names this backend writes.
fn split_metadata_key(path: &str) -> Option<(&str, &str)> {
    let rest = match path.strip_suffix(".json") {
        Some(rest) => rest,
        None => {
            let (rest, index) = path.strip_suffix(".jpg")?.rsplit_once("_thumb_")?;
            if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            rest
        }
    };
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
    use crate::storage::event_index::DetectionDetail;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
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
        /// Paths that appear the instant after a listing is served: an upload
        /// committing while a scan walks the snapshot it took.
        commit_after_list: Arc<Mutex<Vec<String>>>,
        /// Every GET path served, in arrival order — what a caller asked for,
        /// and how many times.
        gets: Arc<Mutex<Vec<String>>>,
        /// GETs currently being served, and the high-water mark: the client's
        /// fan-out width as the server actually saw it.
        in_flight: Arc<AtomicUsize>,
        peak_gets: Arc<AtomicUsize>,
        /// Latency added to every GET, so a serial caller is distinguishable
        /// from a concurrent one by the clock.
        get_delay_ms: Arc<AtomicU64>,
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

        /// How many times this exact path was fetched.
        fn get_count(&self, path: &str) -> usize {
            self.gets
                .lock()
                .unwrap()
                .iter()
                .filter(|p| *p == path)
                .count()
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
            let response = if detail {
                let arr: Vec<serde_json::Value> = paths
                    .iter()
                    .map(|p| serde_json::json!({"path": p, "size": files[p].len(), "mtime": 0}))
                    .collect();
                Json(arr).into_response()
            } else {
                Json(paths).into_response()
            };
            drop(files);
            // Whatever was landing while the snapshot was taken lands now.
            for path in stub.commit_after_list.lock().unwrap().drain(..) {
                stub.files.lock().unwrap().insert(path, vec![0u8; 10]);
            }
            return response;
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
                stub.gets.lock().unwrap().push(path.clone());
                let now = stub.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                stub.peak_gets.fetch_max(now, Ordering::SeqCst);
                let delay = stub.get_delay_ms.load(Ordering::Relaxed);
                if delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                let resp = get_response(&stub, &path, &headers);
                stub.in_flight.fetch_sub(1, Ordering::SeqCst);
                resp
            }
        }
    }

    fn get_response(stub: &Stub, path: &str, headers: &HeaderMap) -> axum::response::Response {
        use axum::response::IntoResponse;

        let fail_get = stub.fail_get_suffix.lock().unwrap().clone();
        if fail_get.is_some_and(|s| path.ends_with(&s)) {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
        let bytes = match stub.files.lock().unwrap().get(path) {
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
                    resp.headers_mut()
                        .insert("content-range", format!("bytes */{total}").parse().unwrap());
                    resp
                }
            },
            None => full_200(bytes),
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
            commit_after_list: Arc::new(Mutex::new(Vec::new())),
            gets: Arc::new(Mutex::new(Vec::new())),
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak_gets: Arc::new(AtomicUsize::new(0)),
            get_delay_ms: Arc::new(AtomicU64::new(0)),
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

    /// A second event at the same start PTS, twice as long: same start,
    /// different stem, and so a different set of objects on the host. Nothing
    /// enforces the uniqueness of a start PTS, which is why an event is
    /// identified by its stem and not by where a binary search on the start
    /// happens to land.
    fn longer_movement_event(first_pts: u64, size: usize) -> FinishedEvent {
        let mut e = movement_event(first_pts, size);
        e.segments.push(segment(first_pts + SEC, 0xcd, size));
        e.total_bytes = size * 2;
        e
    }

    /// The key an API request carries for one stem. This backend resolves by
    /// stem alone and ignores the type in the key (see its `find_event`), so
    /// these lookups name `Movement` whatever the event turns out to be — a
    /// deliberate choice, pinned by
    /// `find_event_resolves_by_stem_across_an_upgrade`.
    fn url_key(start_pts_ns: u64, duration_ms: u32) -> EventRef {
        EventRef::new(start_pts_ns, duration_ms, EventType::Movement)
    }

    fn sibling(backend: &StathostBackend, duration_ms: u32) -> Option<WarmEventEntry> {
        backend
            .query("cam", 0, u64::MAX)
            .into_iter()
            .find(|e| e.duration_ms == duration_ms)
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
        let entry = backend.find_event("cam", url_key(1_000, 1000)).unwrap();
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
        let e = scanned.find_event("cam", url_key(1_000, 1000)).unwrap();
        assert_eq!(e.event_type, EventType::Movement);
        assert_eq!(e.file_size, 40);
        assert_eq!(e.filmstrip_frames, 2);
        assert_eq!(scanned.free_space().unwrap(), u64::MAX); // unlimited budget
    }

    /// A `PUT` of a key that exists is an update, so writing a stem twice
    /// rewrites one event. The index used to gain a second entry for it and the
    /// budget was charged twice — an in-RAM store of two events where the host
    /// holds one, drifting the client-side budget away from real usage.
    #[tokio::test]
    async fn a_rewritten_stem_replaces_its_entry_rather_than_adding_one() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);

        backend.write_event("cam", &movement_event(1_000, 40)).await;
        // Same start and duration — the same stem, and so the same objects.
        backend.write_event("cam", &movement_event(1_000, 25)).await;

        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 1);
        assert_eq!(
            backend
                .find_event("cam", url_key(1_000, 1000))
                .unwrap()
                .file_size,
            25
        );
        assert_eq!(backend.used(), 25);
        // ts + json + 2 thumbs, overwritten in place.
        assert_eq!(stub.files.lock().unwrap().len(), 4);
    }

    /// The scan counts filmstrip frames contiguously from 0, so a thumbnail the
    /// rewrite has no frame for would be served as part of this event and would
    /// outlive it — the delete only removes the frames the entry knows about.
    #[tokio::test]
    async fn a_shorter_rewrite_deletes_the_thumbnails_it_no_longer_has() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);

        let mut event = movement_event(2_000, 30);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        backend.write_event("cam", &event).await;
        assert!(stub.has("cam/2000_1000_thumb_2.jpg"));

        let mut shorter = movement_event(2_000, 30);
        shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
        backend.write_event("cam", &shorter).await;

        assert_eq!(
            backend
                .find_event("cam", url_key(2_000, 1000))
                .unwrap()
                .filmstrip_frames,
            1
        );
        assert!(stub.has("cam/2000_1000_thumb_0.jpg"));
        assert!(!stub.has("cam/2000_1000_thumb_1.jpg"));
        assert!(!stub.has("cam/2000_1000_thumb_2.jpg"));

        // What a restart rebuilds agrees with the index in RAM.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert_eq!(
            scanned
                .find_event("cam", url_key(2_000, 1000))
                .unwrap()
                .filmstrip_frames,
            1
        );
    }

    /// Two events sharing a start PTS are two events. Everything that reaches
    /// into the index by key — the upgrade's in-place rewrite and the sweep's
    /// removal — has to find the one it named, not whichever of the pair a
    /// binary search on the start returns.
    #[tokio::test]
    async fn siblings_sharing_a_start_pts_are_upgraded_and_removed_by_stem() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        backend
            .write_event("cam", &longer_movement_event(OLD_PTS, 40))
            .await;
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 2);
        assert_eq!(backend.used(), 40 + 80);

        // Upgrade only the longer one.
        let mut upgrade = upgrade_for(OLD_PTS);
        upgrade.duration_ms = 2000;
        backend.upgrade_event("cam", &upgrade).await;
        assert_eq!(
            sibling(&backend, 2000).unwrap().event_type,
            EventType::Object
        );
        assert_eq!(
            sibling(&backend, 1000).unwrap().event_type,
            EventType::Movement
        );
        let sidecar = stub
            .files
            .lock()
            .unwrap()
            .get(&format!("cam/{OLD_PTS}_1000.json"))
            .cloned()
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&sidecar).unwrap()["event_type"],
            serde_json::json!("movement"),
            "the upgrade rewrote its sibling's sidecar"
        );

        // The movement sibling expires; the object one does not.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(sibling(&backend, 1000).is_none());
        assert!(sibling(&backend, 2000).is_some());
        assert!(!stub.has(&format!("cam/{OLD_PTS}_1000.ts")));
        assert!(stub.has(&format!("cam/{OLD_PTS}_2000.ts")));
        assert_eq!(
            backend.used(),
            80,
            "the wrong sibling's bytes were refunded"
        );
    }

    /// The read path, on the same pair: each sibling is served as itself.
    ///
    /// An API request names a stem, and the two events under this start hold
    /// different objects on the host — different videos, different lengths. The
    /// lookup this replaced binary-searched the start alone, so one of the two
    /// URLs was always answered with the other event's recording: the wrong
    /// video streamed under the right link, at the wrong duration.
    #[tokio::test]
    async fn same_start_siblings_are_each_served_by_their_own_key() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend
            .write_event("cam", &movement_event(30_000, 40))
            .await;
        backend
            .write_event("cam", &longer_movement_event(30_000, 40))
            .await;
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 2);

        for (duration_ms, file_size) in [(1000u32, 40u64), (2000, 80)] {
            let entry = backend
                .find_event("cam", url_key(30_000, duration_ms))
                .unwrap_or_else(|| panic!("{duration_ms}ms sibling is not indexed"));
            assert_eq!(entry.duration_ms, duration_ms);
            assert_eq!(entry.file_size, file_size);
            // And the bytes behind it are that event's own.
            let vs = backend.read_video("cam", &entry, None).await.unwrap();
            assert_eq!(vs.total_size, file_size);
            assert_eq!(drain(vs).await.len(), file_size as usize);
        }

        // A stem nothing is stored under: this start, no such duration.
        assert!(backend.find_event("cam", url_key(30_000, 3000)).is_none());
    }

    /// The one asymmetry between the backends: the event type in a request's key
    /// is ignored here, because the objects it names do not depend on it. An
    /// upgrade rewrites the sidecar in place and moves nothing, so honoring the
    /// type would 404 a link a client already holds — the event list it came
    /// from, or the playlist a player is part way through — while the footage
    /// sits right where it was. Local disk cannot do this: the type is a
    /// directory there, so it is part of the path.
    #[tokio::test]
    async fn find_event_resolves_by_stem_across_an_upgrade() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend
            .write_event("cam", &movement_event(31_000, 40))
            .await;
        backend.upgrade_event("cam", &upgrade_for(31_000)).await;

        // The key a client took from the listing before the upgrade still
        // resolves, and to the event as it is now.
        for event_type in [
            EventType::Movement,
            EventType::Object,
            EventType::Continuous,
        ] {
            let entry = backend
                .find_event("cam", EventRef::new(31_000, 1000, event_type))
                .unwrap_or_else(|| panic!("{event_type:?} key stopped resolving"));
            assert_eq!(entry.event_type, EventType::Object);
        }
        // The stem is still the whole identity: the duration has to be right.
        assert!(backend
            .find_event("cam", EventRef::new(31_000, 2000, EventType::Object))
            .is_none());
    }

    /// The same for the two flags an entry carries: a failed delete and a type
    /// the scan could not read both belong to one stem, not to a start PTS.
    #[tokio::test]
    async fn flags_and_type_holds_follow_the_stem_not_the_start_pts() {
        let (url, stub) = spawn_stub("secret").await;
        let writer = backend_for(&url, "secret", 0);
        writer
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        writer
            .write_event("cam", &longer_movement_event(OLD_PTS, 40))
            .await;

        // Only the longer sibling's sidecar is unreadable on the next start.
        stub.fail_gets("_2000.json");
        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;
        assert!(backend.has_unknown_type("cam", (OLD_PTS, 2000)));
        assert!(!backend.has_unknown_type("cam", (OLD_PTS, 1000)));

        // The typed sibling expires as a movement; the held one is measured
        // against the longest configured retention and stays.
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert(format!("cam/{OLD_PTS}_2000.ts"));
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(sibling(&backend, 1000).is_none());
        assert!(sibling(&backend, 2000).is_some());

        // Now expire everything: the held sibling is tried, refuses, and is the
        // one flagged.
        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
        let held = sibling(&backend, 2000).unwrap();
        assert!(held.delete_failed);
        assert!(backend.has_unknown_type("cam", (OLD_PTS, 2000)));
    }

    #[tokio::test]
    async fn object_event_sidecar_carries_type_and_scans_back() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);

        backend.write_event("cam", &object_event(4_000, 20)).await;

        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", url_key(4_000, 1000)).unwrap();
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
        let e = backend.find_event("cam", url_key(5_000, 1000)).unwrap();
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

        assert!(backend.find_event("cam", url_key(old_pts, 1000)).is_none());
        assert!(stub.files.lock().unwrap().is_empty());
        assert_eq!(backend.used(), 0);
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

        assert!(backend.find_event("cam", url_key(old_pts, 1000)).is_some());
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
        assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none()); // continuous evicted
        assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none()); // movement evicted
        assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some()); // object survives
        assert!(backend.used() <= 60);
    }

    /// Budget eviction runs ahead of every write, so an object the store
    /// refuses must not be re-attempted by every pass: it would spend each one
    /// on the same doomed delete and never reach the events that would free
    /// space. Local disk's emergency prune skips its own failures for exactly
    /// this reason.
    #[tokio::test]
    async fn budget_eviction_skips_an_event_it_already_failed_to_delete() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 60);
        for pts in [1_000u64, 2_000, 3_000] {
            backend.write_event("cam", &movement_event(pts, 40)).await;
        }
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert("cam/1000_1000.ts".to_string());

        // First pass: the oldest refuses, and the pass stops there.
        backend.guard_free_space("cam", 0).await;
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 3);
        assert!(
            backend
                .find_event("cam", url_key(1_000, 1000))
                .unwrap()
                .delete_failed
        );

        // Second: it is skipped, and the budget is enforced around it.
        backend.guard_free_space("cam", 0).await;
        assert!(backend.find_event("cam", url_key(1_000, 1000)).is_some());
        assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
        assert!(backend.find_event("cam", url_key(3_000, 1000)).is_none());
        assert!(stub.has("cam/1000_1000.ts"));
    }

    /// An object that was already gone reclaimed nothing on the host — it only
    /// corrected an index entry describing nothing. Its entry still has to go,
    /// and the pass still has to go on to something that does free bytes.
    #[tokio::test]
    async fn budget_eviction_unindexes_an_object_that_is_already_gone() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 60);
        backend.write_event("cam", &movement_event(1_000, 40)).await;
        backend.write_event("cam", &movement_event(2_000, 40)).await;
        backend.write_event("cam", &movement_event(3_000, 40)).await;
        // Someone else removed the oldest video behind camon's back.
        stub.files.lock().unwrap().remove("cam/1000_1000.ts");

        backend.guard_free_space("cam", 0).await;

        assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none());
        assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
        assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some());
        assert_eq!(backend.used(), 40);
    }

    /// An outage flags one candidate per pass (the pass stops at the first
    /// failure). If flagging *excluded* an event from eviction, the store
    /// coming back would leave the budget permanently over its limit: nothing
    /// already written would ever be reconsidered, and the hourly sweep only
    /// retries events that are age-expired. Flagging demotes instead.
    #[tokio::test]
    async fn budget_eviction_recovers_after_an_outage_flagged_every_candidate() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 60);
        for pts in [1_000u64, 2_000, 3_000, 4_000] {
            backend.write_event("cam", &movement_event(pts, 40)).await;
        }
        {
            let mut refused = stub.fail_delete_paths.lock().unwrap();
            for pts in [1_000u64, 2_000, 3_000, 4_000] {
                refused.insert(format!("cam/{pts}_1000.ts"));
            }
        }

        // The store is unreachable: every pass flags one more candidate.
        for _ in 0..4 {
            backend.guard_free_space("cam", 0).await;
        }
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 4);
        assert!(backend
            .query("cam", 0, u64::MAX)
            .iter()
            .all(|e| e.delete_failed));

        // It comes back. Eviction has to reconsider what it flagged, or the
        // budget stays at 160 of 60 for the life of the process.
        stub.fail_delete_paths.lock().unwrap().clear();
        backend.guard_free_space("cam", 0).await;
        assert!(backend.used() <= 60);
        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 1);
    }

    /// The video is deleted before the sidecar, so a refused video delete keeps
    /// the event's type. The old order lost it, and a type-less survivor is not
    /// simply "expired sooner": `continuous_retention_days` defaults to 1 day
    /// against movement's 2, and all three retentions are freely configurable,
    /// so reading a continuous chunk back as a movement can keep it a day
    /// longer than its own class allows.
    #[tokio::test]
    async fn a_video_that_refuses_to_delete_keeps_its_sidecar_and_its_type() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let mut event = continuous_event(OLD_PTS, 30);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01]]));
        backend.write_event("cam", &event).await;
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert(format!("cam/{OLD_PTS}_1000.ts"));

        // Expire it as a continuous chunk (1 day) while movements keep 2.
        backend
            .prune(u64::MAX, u64::MAX, 1, &AtomicBool::new(false))
            .await;

        let entry = backend.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
        assert!(entry.delete_failed);
        assert!(stub.has(&format!("cam/{OLD_PTS}_1000.ts")));
        assert!(stub.has(&format!("cam/{OLD_PTS}_1000.json")), "type lost");
        // Thumbnails are decoration and carry no type; they go first.
        assert!(!stub.has(&format!("cam/{OLD_PTS}_1000_thumb_0.jpg")));

        // A restart still knows what it is, so the retry measures it against
        // the retention it actually belongs to.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert_eq!(
            scanned
                .find_event("cam", url_key(OLD_PTS, 1000))
                .unwrap()
                .event_type,
            EventType::Continuous
        );
        // ...and once the store lets go, the whole event goes.
        stub.fail_delete_paths.lock().unwrap().clear();
        scanned
            .prune(u64::MAX, u64::MAX, 1, &AtomicBool::new(false))
            .await;
        assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
        assert!(stub.files.lock().unwrap().is_empty());
    }

    /// A thumbnail the store refuses to delete stays part of the event rather
    /// than leaking: frames are trimmed top-down, so what survives is still
    /// contiguous from 0 and the entry can say so — which is what the next scan
    /// counts, and what the event's own delete removes.
    #[tokio::test]
    async fn a_thumbnail_that_refuses_to_delete_stays_part_of_the_event() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let mut event = movement_event(24_000, 30);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        backend.write_event("cam", &event).await;
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert("cam/24000_1000_thumb_1.jpg".to_string());

        let mut shorter = movement_event(24_000, 30);
        shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
        backend.write_event("cam", &shorter).await;

        // Frame 2 went; frame 1 refused, so the event still has 0 and 1.
        assert!(!stub.has("cam/24000_1000_thumb_2.jpg"));
        assert!(stub.has("cam/24000_1000_thumb_1.jpg"));
        assert_eq!(
            backend
                .find_event("cam", url_key(24_000, 1000))
                .unwrap()
                .filmstrip_frames,
            2,
            "index disagrees with the host about what exists"
        );

        // The scan counts the same thing, and the event's delete takes it all.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert_eq!(
            scanned
                .find_event("cam", url_key(24_000, 1000))
                .unwrap()
                .filmstrip_frames,
            2
        );
        stub.fail_delete_paths.lock().unwrap().clear();
        scanned
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(stub.files.lock().unwrap().is_empty(), "leaked a thumbnail");
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
        let e = backend.find_event("cam", url_key(9_000, 1000)).unwrap();
        // A confirmed-absent sidecar is what a plain movement event is written
        // with, so the default is a fact about the write path, not a fallback.
        assert_eq!(e.event_type, EventType::Movement);
        assert_eq!(e.filmstrip_frames, 0);
        assert!(e.object_classes.is_empty());
    }

    /// Seed `count` stored events, one `.ts` plus one sidecar each, at 1s
    /// intervals from `first_pts`. `event_type` is written into every sidecar.
    fn seed_events(stub: &Stub, first_pts: u64, count: u64, duration_ms: u32, event_type: &str) {
        let mut files = stub.files.lock().unwrap();
        for i in 0..count {
            let stem = format!("{}_{duration_ms}", first_pts + i * SEC);
            files.insert(format!("cam/{stem}.ts"), vec![0u8; 10]);
            files.insert(
                format!("cam/{stem}.json"),
                format!(r#"{{"event_type":"{event_type}"}}"#).into_bytes(),
            );
        }
    }

    /// The scan is awaited before the first camera is spawned, so its cost is
    /// startup latency with nothing recording. One awaited round trip per
    /// stored event made that a function of the archive's size; the reads now
    /// overlap [`SCAN_CONCURRENCY`]-wide.
    #[tokio::test]
    async fn the_scan_reads_sidecars_concurrently() {
        let (url, stub) = spawn_stub("secret").await;
        seed_events(&stub, 1_000, 64, 1000, "object");
        // Enough per-request latency that a serial scan cannot hide in the
        // noise: 64 × 50ms = 3.2s of it.
        stub.get_delay_ms.store(50, Ordering::Relaxed);

        let backend = backend_for(&url, "secret", 0);
        let started = std::time::Instant::now();
        backend.scan().await;
        let elapsed = started.elapsed();

        assert_eq!(backend.query("cam", 0, u64::MAX).len(), 64);
        // Serial is 3.2s of injected latency alone and measures at ~3.4s;
        // sixteen at a time is four waves, ~0.24s. The bound sits between them
        // with room for a loaded machine on either count.
        assert!(
            elapsed < Duration::from_millis(2_000),
            "sidecar reads were serial: {elapsed:?}"
        );
        let peak = stub.peak_gets.load(Ordering::SeqCst);
        assert!(peak > 1, "no reads overlapped");
        assert!(peak <= SCAN_CONCURRENCY, "fan-out ran unbounded: {peak}");
    }

    /// Overlapping the reads must not reach the index: entries stay sorted, and
    /// each stored event is indexed exactly once and charged once.
    #[tokio::test]
    async fn a_concurrent_scan_indexes_every_event_once_and_in_order() {
        let (url, stub) = spawn_stub("secret").await;
        {
            let mut files = stub.files.lock().unwrap();
            for i in 0..40u64 {
                let stem = format!("{}_1000", 1_000 + i * SEC);
                files.insert(format!("cam/{stem}.ts"), vec![0u8; 10]);
                // Alternating types, so a result delivered against the wrong
                // event would show up as a type on the wrong entry.
                let event_type = if i % 2 == 0 { "object" } else { "continuous" };
                files.insert(
                    format!("cam/{stem}.json"),
                    format!(r#"{{"event_type":"{event_type}"}}"#).into_bytes(),
                );
            }
        }
        stub.get_delay_ms.store(5, Ordering::Relaxed);

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;

        let entries = backend.query("cam", 0, u64::MAX);
        assert_eq!(entries.len(), 40, "an event was dropped or duplicated");
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.start_pts_ns, 1_000 + i as u64 * SEC);
            assert_eq!(
                entry.event_type,
                if i % 2 == 0 {
                    EventType::Object
                } else {
                    EventType::Continuous
                }
            );
        }
        // The budget is the sum of the index, so a double insertion shows here.
        assert_eq!(backend.used(), 40 * 10);
    }

    /// The distinction the whole sidecar path rests on survives the fan-out: a
    /// read that failed is "unknown", never the confirmed absence that means
    /// "movement". Half of these sidecars answer `500` while the other half
    /// answer normally, in the same pass.
    #[tokio::test]
    async fn a_concurrent_scan_never_reads_a_failure_as_an_absence() {
        let (url, stub) = spawn_stub("secret").await;
        seed_events(&stub, 1_000, 20, 1000, "object");
        seed_events(&stub, 1_000, 20, 2000, "object");
        // Only the 2000ms-duration events' sidecars are unreadable.
        stub.fail_gets("_2000.json");

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;

        let entries = backend.query("cam", 0, u64::MAX);
        assert_eq!(entries.len(), 40);
        for entry in &entries {
            let key = event_key(entry);
            if entry.duration_ms == 1000 {
                assert_eq!(entry.event_type, EventType::Object);
                assert!(!backend.has_unknown_type("cam", key), "held a read event");
            } else {
                // Not indexed as a movement event: the type is on hold, so
                // pruning measures it against the longest retention.
                assert!(
                    backend.has_unknown_type("cam", key),
                    "a failed read was taken for a confirmed absence"
                );
            }
        }
    }

    #[tokio::test]
    async fn write_retries_then_drops_on_persistent_failure() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_writes.store(true, Ordering::Relaxed);

        let outcome = backend.write_event("cam", &movement_event(6_000, 30)).await;
        assert_eq!(outcome, WriteOutcome::Failed);
        // The event was not indexed and nothing landed on the host.
        assert!(backend.find_event("cam", url_key(6_000, 1000)).is_none());
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
        assert!(backend.find_event("cam", url_key(11_000, 1000)).is_none());
        assert_eq!(backend.used(), 0);
        // The video was never attempted, so there is no bare .ts for a later
        // scan to call a movement event.
        assert!(!stub.has("cam/11000_1000.ts"));
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert!(scanned.find_event("cam", url_key(11_000, 1000)).is_none());
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
        let written = backend.find_event("cam", url_key(16_000, 1000)).unwrap();

        stub.files.lock().unwrap().remove("cam/16000_1000.json");
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;

        assert_eq!(
            scanned.find_event("cam", url_key(16_000, 1000)).unwrap(),
            written
        );
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
        assert!(backend.find_event("cam", url_key(12_000, 1000)).is_none());
        // Both objects are still there, so the next scan adopts the phantom
        // video as the object event it is — not as a movement event on a
        // two-day retention.
        stub.clear_faults();
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", url_key(12_000, 1000)).unwrap();
        assert_eq!(e.event_type, EventType::Object);
        assert_eq!(e.object_classes, vec!["car".to_string()]);
    }

    /// The mirror case: the video genuinely did not land. The orphan sidecar
    /// left behind indexes nothing — the scan walks `.ts` objects only — and so
    /// nothing else would ever delete it either: it is never indexed, never
    /// counted against the budget, and never a sibling of an event. The scan
    /// collects it instead of leaving it to accumulate for the life of the
    /// bucket, one flaky upload at a time.
    #[tokio::test]
    async fn an_orphan_sidecar_indexes_nothing_and_is_collected() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".ts", false);

        let outcome = backend.write_event("cam", &object_event(17_000, 30)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(stub.has("cam/17000_1000.json"));
        stub.clear_faults();
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert!(scanned.find_event("cam", url_key(17_000, 1000)).is_none());
        assert!(!stub.has("cam/17000_1000.json"), "orphan sidecar kept");
    }

    /// Thumbnails orphan the same way: an upload that got as far as the
    /// filmstrip before the video failed leaves them behind too.
    #[tokio::test]
    async fn orphaned_thumbnails_are_collected_and_live_ones_are_not() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        backend
            .write_event("cam", &movement_event(19_000, 30))
            .await;
        // Orphans of an event whose video is not on the host.
        stub.files
            .lock()
            .unwrap()
            .insert("cam/20000_1000_thumb_0.jpg".to_string(), vec![0x01]);
        stub.files
            .lock()
            .unwrap()
            .insert("cam/20000_1000.json".to_string(), b"{}".to_vec());

        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;

        assert!(!stub.has("cam/20000_1000_thumb_0.jpg"));
        assert!(!stub.has("cam/20000_1000.json"));
        // The live event keeps every one of its objects.
        assert_eq!(
            scanned
                .find_event("cam", url_key(19_000, 1000))
                .unwrap()
                .filmstrip_frames,
            2
        );
        assert!(stub.has("cam/19000_1000.json"));
        assert!(stub.has("cam/19000_1000_thumb_1.jpg"));
    }

    /// One failed upload orphans a sidecar and every filmstrip frame at once,
    /// and all of them turn on the same question. The sweep asks the host once
    /// per stem, not once per object it is about to delete.
    #[tokio::test]
    async fn the_sweep_probes_an_orphaned_stem_once() {
        let (url, stub) = spawn_stub("secret").await;
        {
            let mut files = stub.files.lock().unwrap();
            files.insert("cam/24000_1000.json".to_string(), b"{}".to_vec());
            for i in 0..6 {
                files.insert(format!("cam/24000_1000_thumb_{i}.jpg"), vec![0x01]);
            }
        }

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;

        assert_eq!(
            stub.get_count("cam/24000_1000.ts"),
            1,
            "the same video was probed once per orphaned object"
        );
        assert!(
            stub.files.lock().unwrap().is_empty(),
            "an orphaned object survived the sweep"
        );
    }

    /// The listing is a snapshot, and it is not necessarily *this* process's
    /// snapshot: another camon on the same camera id, or a `PUT` that outlived
    /// the process which issued it, can commit a video after the bucket was
    /// listed. The sweep re-checks every candidate against the host immediately
    /// before deleting it, so a sidecar whose video landed in that window keeps
    /// the event's only record of its type.
    #[tokio::test]
    async fn the_sweep_keeps_a_sidecar_whose_video_landed_after_the_listing() {
        let (url, stub) = spawn_stub("secret").await;
        stub.files.lock().unwrap().insert(
            "cam/22000_1000.json".to_string(),
            br#"{"event_type":"object"}"#.to_vec(),
        );
        // The video commits the moment the scan has its listing — the shape of
        // an upload in flight under a camera id this process also owns.
        stub.commit_after_list
            .lock()
            .unwrap()
            .push("cam/22000_1000.ts".to_string());

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;

        assert!(stub.has("cam/22000_1000.json"), "live sidecar collected");
        assert!(stub.has("cam/22000_1000.ts"));
        // Not indexed — it was not in the listing — but the next start reads it
        // back as the object event its surviving sidecar says it is.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        assert_eq!(
            scanned
                .find_event("cam", url_key(22_000, 1000))
                .unwrap()
                .event_type,
            EventType::Object
        );
    }

    /// A failure to find out whether the video is there is not an absence.
    #[tokio::test]
    async fn the_sweep_keeps_metadata_it_could_not_check() {
        let (url, stub) = spawn_stub("secret").await;
        stub.files
            .lock()
            .unwrap()
            .insert("cam/23000_1000.json".to_string(), b"{}".to_vec());
        // The re-check itself fails: a 500 on the video probe, not a 404.
        stub.fail_gets(".ts");

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;

        assert!(stub.has("cam/23000_1000.json"));
    }

    /// The sweep deletes, so it may only touch what this process is the
    /// authority for: the cameras it owns, under names this backend writes. A
    /// second camon sharing the bucket has uploads in flight that look exactly
    /// like orphans — sidecar first, video second.
    #[tokio::test]
    async fn the_scan_only_collects_orphans_of_cameras_it_owns() {
        let (url, stub) = spawn_stub("secret").await;
        {
            let mut files = stub.files.lock().unwrap();
            // Another camon's in-flight write: sidecar up, video still going.
            files.insert("other/21000_1000.json".to_string(), b"{}".to_vec());
            // Ours, but not something this backend writes.
            files.insert("cam/notes.txt".to_string(), b"hi".to_vec());
            files.insert("cam/settings.json".to_string(), b"{}".to_vec());
            files.insert("cam/logo.jpg".to_string(), vec![0x01]);
        }

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await;

        assert!(stub.has("other/21000_1000.json"));
        assert!(stub.has("cam/notes.txt"));
        assert!(stub.has("cam/settings.json"));
        assert!(stub.has("cam/logo.jpg"));
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
        let entry = backend.find_event("cam", url_key(13_000, 1000)).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.filmstrip_frames, 0);

        // ...and the index a restart rebuilds agrees with the one in RAM.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await;
        let e = scanned.find_event("cam", url_key(13_000, 1000)).unwrap();
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
        assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
        scanned
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
        assert!(stub.has("cam/1000000000_1000.ts"));

        // The hold is a longer retention, not an immortal one: once every
        // configured age has passed, an event nobody can type still goes.
        scanned.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
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
        assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));

        // The store recovers; the next sweep reads the sidecar it could not.
        stub.clear_faults();
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        let entry = backend.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(entry.object_classes, vec!["car".to_string()]);
        assert!(!backend.has_unknown_type("cam", (OLD_PTS, 1000)));

        // Typed again, it prunes on its own retention: kept as an object...
        backend.prune(1, u64::MAX, 1, &AtomicBool::new(false)).await;
        assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
        // ...and gone once the object retention itself expires.
        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
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
        assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        // A re-read says the same thing, so the hold survives the sweep.
        assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
        assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));
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
        assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
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
        assert!(backend.has_unknown_type("cam", (1_000, 1000)));

        // ...so the budget must still evict the genuine movement event first,
        // even though the held one is older and labelled movement too.
        backend.guard_free_space("cam", 0).await;
        assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
        assert!(backend.find_event("cam", url_key(1_000, 1000)).is_some());
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
            backend
                .find_event("cam", url_key(18_000, 1000))
                .unwrap()
                .filmstrip_frames,
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
            sidecar_json(Some(EventType::Movement), None, None, &[], false),
            r#"{"detections":[],"event_type":"movement"}"#
        );
    }

    #[tokio::test]
    async fn read_thumbnail_errors_when_no_filmstrip() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let event = continuous_event(7_000, 30); // no filmstrip frames
        backend.write_event("cam", &event).await;
        let entry = backend.find_event("cam", url_key(7_000, 1000)).unwrap();
        assert!(backend.read_thumbnail("cam", &entry).await.is_err());
    }

    // ---- streamed Range playback ------------------------------------------

    #[tokio::test]
    async fn read_video_serves_partial_and_suffix_ranges() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        // A 40-byte movement event (body is 40 × 0xab).
        backend.write_event("cam", &movement_event(8_000, 40)).await;
        let entry = backend.find_event("cam", url_key(8_000, 1000)).unwrap();

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
        let entry = backend.find_event("cam", url_key(9_000, 1000)).unwrap();

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
        let entry = backend.find_event("cam", url_key(10_000, 1000)).unwrap();

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
            backend.events.insert(
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
