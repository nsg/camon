//! Remote "stathost" warm-storage backend.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::RwLock;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use serde::Deserialize;

use crate::buffer::warm::{EventUpgrade, FinishedEvent};
use crate::buffer::{wall_clock_ns, GopSegment};
use crate::config::StathostConfig;
use crate::locks::{LockExt, MutexExt};
use crate::retry::{RetrySchedule, Streak};
use crate::storage::backend::{
    RangeRequest, ServedRange, ThumbnailError, VideoStream, WarmStorageBackend, WriteOutcome,
};
use crate::storage::contract::{
    sleep_unless, Attempted, ByteBudget, Recovery, Reservation, RetryPolicy, StopFlag,
};
use crate::storage::event_index::{
    evict_tiers, filmstrip_frame_count, sweep_expired, EmergencyOutcome, EventIdentity, EventIndex,
    EventPage, EvictionPolicy, Removal,
};
use crate::storage::warm_index::{
    parse_event_filename, parse_sidecar_json, sidecar_json, SidecarData,
};
use crate::storage::{EventRef, EventType, WarmEventEntry};

const NANOS_PER_MS: u64 = 1_000_000;

/// This backend's event identity: the stem every one of an event's keys is built from,
/// `{camera_id}/{start_pts_ns}_{duration_ms}.*`.
type EventKey = (u64, u32);

fn event_key(entry: &WarmEventEntry) -> EventKey {
    EventIdentity::of(entry)
}

fn key_stem(key: EventKey) -> String {
    format!("{}_{}", key.0, key.1)
}

// The three object keys of one event. [`split_ts_key`] and
// [`split_metadata_key`] are their inverses.
fn ts_key(camera_id: &str, stem: &str) -> String {
    format!("{camera_id}/{stem}.ts")
}

fn sidecar_key(camera_id: &str, stem: &str) -> String {
    format!("{camera_id}/{stem}.json")
}

fn thumb_key(camera_id: &str, stem: &str, frame: usize) -> String {
    format!("{camera_id}/{stem}_thumb_{frame}.jpg")
}

/// Writes, deletes and the scan are awaited inline by the serial per-camera warm writer, so an
/// unbounded request stalls that camera's recording and its shutdown. Both clients bound the
/// connect phase; the rest differs by call shape.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Total per-request ceiling for the non-streaming calls: delete, sidecar and
/// thumbnail GETs, and the whole-bucket listing — which is not small, a busy
/// bucket lists thousands of entries at startup.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Total per-request ceiling for uploads, which reach tens of MB.
pub(crate) const UPLOAD_TIMEOUT: Duration = Duration::from_secs(300);
/// Idle budget for the streaming client only: reqwest arms it flat until the
/// response headers arrive, then per response frame — the right shape for a
/// body a player drains at its own pace.
const STREAM_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// One of the backend's two HTTP clients: the plain one, or — with a `read_timeout` — the
/// streaming one.
fn build_client(read_timeout: Option<Duration>) -> std::io::Result<reqwest::Client> {
    #[cfg(test)]
    if fail_client_build() {
        return Err(client_build_error("forced by a test"));
    }
    let mut builder = reqwest::Client::builder().connect_timeout(CONNECT_TIMEOUT);
    if let Some(read_timeout) = read_timeout {
        builder = builder.read_timeout(read_timeout);
    }
    builder.build().map_err(client_build_error)
}

/// What a failed client build says to the operator. It names the three things
/// `reqwest` does here that can fail on a real box; the underlying error alone
/// names none of them.
fn client_build_error(cause: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(format!(
        "the stathost HTTP client could not be built — check the TLS backend, the system \
         root certificate store and any proxy settings in the environment: {cause}"
    ))
}

#[cfg(test)]
thread_local! {
    /// Makes the next [`build_client`] fail. Per thread rather than global: a
    /// process-wide switch would fail whichever unrelated backend happened to
    /// be under construction at the time.
    static FAIL_CLIENT_BUILD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn fail_client_build() -> bool {
    FAIL_CLIENT_BUILD.with(std::cell::Cell::get)
}

/// Make [`build_client`] fail on this thread, for the tests that pin what a
/// backend which cannot build one does about it.
#[cfg(test)]
pub(crate) fn force_client_build_failure(fail: bool) {
    FAIL_CLIENT_BUILD.with(|forced| forced.set(fail));
}

/// How long one object request waits before being sent again, and how far that wait grows.
#[cfg(not(test))]
const OBJECT_RETRY_WAIT: RetrySchedule = RetrySchedule {
    start: Duration::from_secs(2),
    max: Duration::from_secs(8),
};
/// Milliseconds under test, for the reason [`SCAN_RETRY`] gives.
#[cfg(test)]
const OBJECT_RETRY_WAIT: RetrySchedule = RetrySchedule {
    start: Duration::from_millis(1),
    max: Duration::from_millis(4),
};

/// What one object request — an upload, a sidecar read — is worth: the attempt and one
/// more, waited out and classified.
const OBJECT_RETRY: RetryPolicy = RetryPolicy {
    attempts: 2,
    schedule: OBJECT_RETRY_WAIT,
};

/// Wall clock one prune tick will spend, per camera, re-reading the sidecars of events whose
/// type an earlier scan could not establish.
#[cfg(not(test))]
const RESOLVE_BUDGET: Duration = Duration::from_secs(10);
/// Short enough under test that a hold list which would take seconds to read
/// through is visibly cut off, long enough that a healthy re-read of a handful
/// of sidecars finishes inside it.
#[cfg(test)]
const RESOLVE_BUDGET: Duration = Duration::from_millis(50);

/// How many requests a [`scan`](StathostBackend::scan) keeps in flight — its sidecar reads,
/// and the orphan sweep's probes and deletes.
const SCAN_CONCURRENCY: usize = 16;

/// How long a scan waits before asking the store again, and how far that wait grows: 2s, 4s,
/// 8s, 16s — the four waits [`SCAN_ATTEMPTS`] leaves room for, the last of them the cap.
#[cfg(not(test))]
const SCAN_RETRY: RetrySchedule = RetrySchedule {
    start: Duration::from_secs(2),
    max: Duration::from_secs(16),
};
/// Milliseconds under test: what is pinned is how many attempts are made and what the failures
/// leave behind, neither of which needs the wall clock.
#[cfg(test)]
const SCAN_RETRY: RetrySchedule = RetrySchedule {
    start: Duration::from_millis(1),
    max: Duration::from_millis(4),
};

/// Listings one scan is worth before it gives up and leaves the backend
/// un-scanned. Giving up is not the end of it: the retention tick starts the
/// series again until one succeeds.
const SCAN_ATTEMPTS: u32 = 5;

/// Wall clock a *startup* scan may spend failing: listings that never arrived and the waits
/// between them. Whichever runs out first — this or [`SCAN_ATTEMPTS`] — ends the series.
#[cfg(not(test))]
const SCAN_LISTING_BUDGET: Duration = Duration::from_secs(45);
/// Long enough under test that a series of fast refusals never runs into it
/// (five of those cost about 15ms of waits), short enough that the test which
/// pins the deadline against a listing that never answers finishes in it.
#[cfg(test)]
const SCAN_LISTING_BUDGET: Duration = Duration::from_millis(500);

/// Which scan this is, and so whether it may collect orphaned metadata.
#[derive(Clone, Copy, PartialEq)]
enum ScanKind {
    /// The scan `init_storage` awaits before the first camera is spawned.
    /// Nothing of this process's can be in flight, so the sweep runs.
    Startup,
    /// A later attempt, from the retention tick, healing an un-scanned index while cameras
    /// record.
    Heal,
}

/// Whether a scan pass got through everything its listing named.
#[derive(Clone, Copy, PartialEq)]
enum ScanPass {
    /// The index now describes the store.
    Complete,
    /// Shutdown arrived part-way through the sidecar reads. The entries already
    /// inserted are true — they came from the listing — but the archive was not
    /// walked to the end, so the pass says nothing about what else is there.
    Interrupted,
}

/// The remote warm store: an HTTP client over the shared in-RAM index.
pub struct StathostBackend {
    http: Http,
    /// Client-side storage budget in bytes; 0 means unlimited.
    budget: ByteBudget,
    /// Shutdown, as the drain raises it. Every request-issuing loop here reads
    /// it before sending, so a stop costs at most the one request already in
    /// flight — which is exactly what the drain's phase 3 is sized for.
    stop: StopFlag,
    events: EventIndex<EventKey>,
    /// Whether [`Self::events`] has ever been rebuilt from what the store actually holds.
    scanned: std::sync::atomic::AtomicBool,
    /// Rate-limits warnings for writes that proceed after eviction cannot meet the byte budget.
    budget_overshoots: std::sync::Mutex<Streak>,
    /// How many times budget enforcement has refused to run because of that, across every
    /// camera — the index it refuses on is one index, and so is the budget.
    budget_refusals: std::sync::Mutex<Streak>,
    /// Events whose sidecar the scan could not read, per camera.
    unknown_type: HashMap<String, RwLock<HashSet<EventKey>>>,
    /// Where the next [`Self::resolve_unknown_types`] pass starts in this camera's hold list.
    /// Only its movement matters, not its value.
    resolve_cursor: HashMap<String, std::sync::atomic::AtomicUsize>,
}

impl StathostBackend {
    /// Build the backend, or report why its HTTP clients could not be.
    pub fn new(
        config: &StathostConfig,
        camera_ids: &[String],
        stop: StopFlag,
    ) -> std::io::Result<Self> {
        let base = format!(
            "{}/{}",
            config.url.trim_end_matches('/'),
            config.bucket.trim_matches('/')
        );
        Ok(Self {
            http: Http {
                client: build_client(None)?,
                stream_client: build_client(Some(STREAM_READ_TIMEOUT))?,
                base,
                token: config.token.clone(),
            },
            budget: ByteBudget::new(config.max_stored_bytes),
            stop,
            events: EventIndex::new(camera_ids),
            scanned: std::sync::atomic::AtomicBool::new(false),
            budget_refusals: std::sync::Mutex::new(Streak::new()),
            budget_overshoots: std::sync::Mutex::new(Streak::new()),
            unknown_type: camera_ids
                .iter()
                .map(|id| (id.clone(), RwLock::new(HashSet::new())))
                .collect(),
            resolve_cursor: camera_ids
                .iter()
                .map(|id| (id.clone(), std::sync::atomic::AtomicUsize::new(0)))
                .collect(),
        })
    }

    fn used(&self) -> u64 {
        self.events.used_bytes()
    }

    /// The index as retention may see it: `None` until a scan has rebuilt it from the store.
    fn scanned_events(&self) -> Option<&EventIndex<EventKey>> {
        self.scanned.load(Ordering::Acquire).then_some(&self.events)
    }

    /// Record that the index now describes the store. Sticky by design: a transient failure of
    /// anything afterwards leaves this set, because the question it answers is "has the archive
    /// ever been read", and the answer to that cannot become no again.
    fn mark_scanned(&self) {
        self.scanned.store(true, Ordering::Release);
    }

    /// Record that this event's type could not be established.
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

    /// Drop the marker once the type is settled.
    fn clear_unknown_type(&self, camera_id: &str, key: EventKey) {
        if let Some(lock) = self.unknown_type.get(camera_id) {
            lock.write_recover().remove(&key);
        }
    }

    /// Read and parse one event's sidecar, through the same [`OBJECT_RETRY`] every object
    /// request here goes through — worth the allowance because the alternative to a readable
    /// sidecar is a guessed retention class.
    async fn read_sidecar(&self, camera_id: &str, stem: &str) -> SidecarRead {
        let key = sidecar_key(camera_id, stem);
        let read = OBJECT_RETRY
            .run(&self.stop, recoverable, || self.http.get_optional(&key))
            .await;
        match read {
            Attempted::Done(None) => SidecarRead::Absent,
            Attempted::Done(Some(bytes)) => {
                match serde_json::from_slice::<serde_json::Value>(&bytes) {
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
                }
            }
            // Debug, not warn: the prune tick retries every held event
            // hourly, and a store that is down would log one line per
            // event per sweep. The scan's aggregate warn carries the news.
            Attempted::Failed(e) => {
                tracing::debug!(key = %key, error = %e, "could not read stathost sidecar");
                SidecarRead::Unreadable
            }
            // Shutdown. The hold stays, and the next start reads it again.
            Attempted::Abandoned => SidecarRead::Unreadable,
        }
    }

    /// Upload one object, retried and classified by [`OBJECT_RETRY`] and abandoned rather than
    /// re-sent once shutdown has been asked for.
    async fn upload(&self, key: &str, body: Bytes) -> bool {
        let sent = OBJECT_RETRY
            .run(&self.stop, recoverable, || self.http.put(key, body.clone()))
            .await;
        match sent {
            Attempted::Done(()) => true,
            Attempted::Failed(e) => {
                tracing::warn!(key = %key, error = %e, "stathost upload failed");
                false
            }
            Attempted::Abandoned => {
                tracing::warn!(key = %key,
                    "abandoned a stathost upload for shutdown rather than starting a request \
                     the drain would have to wait out");
                false
            }
        }
    }

    /// [`Self::read_sidecar`] with the event carried through, so the scan's
    /// fan-out can pair each result with what it belongs to.
    async fn read_sidecar_for(&self, event: ScannedEvent) -> (ScannedEvent, SidecarRead) {
        let read = self.read_sidecar(&event.camera_id, &event.stem).await;
        (event, read)
    }

    /// Delete every object belonging to one event: **thumbnails, then the video, then the
    /// sidecar**.
    async fn delete_event_objects(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        stopping: &impl Fn() -> bool,
    ) -> Removal {
        let stem = key_stem(event_key(entry));
        for i in (0..entry.filmstrip_frames).rev() {
            if stopping() {
                return Removal::Abandoned;
            }
            if matches!(
                self.http.delete(&thumb_key(camera_id, &stem, i)).await,
                DeleteOutcome::Failed
            ) {
                // Stop descending, but do not stop deleting the event.
                tracing::debug!(camera = %camera_id, stem = %stem, frame = i,
                    "failed to delete a filmstrip frame; keeping the frames below it so what \
                     is left is still contiguous, and going on to the video");
                break;
            }
        }
        if stopping() {
            return Removal::Abandoned;
        }
        let removal = match self.http.delete(&ts_key(camera_id, &stem)).await {
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
        // The video is gone, so the entry must go with it whatever happens
        // next: this is the one request whose omission leaves an orphan rather
        // than an inconsistency, and the startup sweep is what collects it.
        if !stopping() {
            let _ = self.http.delete(&sidecar_key(camera_id, &stem)).await;
        }
        removal
    }

    /// Delete the filmstrip frames a rewrite of this stem no longer has, and leave the index
    /// describing what is actually on the host.
    async fn trim_thumbnails(&self, camera_id: &str, key: EventKey, keep: usize, had: usize) {
        let stem = key_stem(key);
        for i in (keep..had).rev() {
            if self.stop.stopped() {
                self.events
                    .update(camera_id, key, |entry| entry.filmstrip_frames = i + 1);
                return;
            }
            if matches!(
                self.http.delete(&thumb_key(camera_id, &stem, i)).await,
                DeleteOutcome::Failed
            ) {
                tracing::warn!(camera = %camera_id, stem = %stem, frame = i,
                    "could not delete a filmstrip frame this event no longer has; \
                     it stays part of the event and is deleted with it");
                // The frames this event keeps past the new count are the *previous* write's,
                // and nothing here knows what they weigh — they were never priced, because
                // they are not what was just uploaded.
                self.events
                    .update(camera_id, key, |entry| entry.filmstrip_frames = i + 1);
                return;
            }
        }
    }

    /// Delete metadata whose video is not on the host: the sidecar (and any thumbnails) of an
    /// event whose `.ts` upload never landed.
    async fn sweep_orphaned_metadata(&self, items: &[ListEntry], sizes: &HashMap<&str, u64>) {
        // Grouped by stem before anything is asked of the host: a failed upload orphans a
        // sidecar and every filmstrip frame together, and all of them turn on the one question
        // the probe answers — is the video there?
        let mut orphans: Vec<(String, Vec<String>)> = Vec::new();
        let mut by_stem: HashMap<String, usize> = HashMap::new();
        for item in items {
            let Some((camera_id, stem)) = split_metadata_key(&item.path) else {
                continue;
            };
            if !self.events.owns_camera(camera_id) || parse_event_filename(stem).is_none() {
                continue;
            }
            let ts_key = ts_key(camera_id, stem);
            if sizes.contains_key(ts_key.as_str()) {
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

    /// One stem's share of [`Self::sweep_orphaned_metadata`]: probe the video once, and delete
    /// this stem's metadata objects only if the probe came back with a confirmed absence.
    /// Returns `(deleted, landed, failed)`.
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

    /// The other half of the upgrade/sweep race, and the half the index cannot see.
    async fn abandon_upgrade_of_a_vanished_video(
        &self,
        camera_id: &str,
        stem: &str,
        key: EventKey,
    ) -> bool {
        if self.stop.stopped() {
            return false;
        }
        if !matches!(
            self.http.probe_exists(&ts_key(camera_id, stem)).await,
            Ok(false)
        ) {
            return false;
        }
        tracing::warn!(
            camera = %camera_id,
            stem = %stem,
            "the video of an event being upgraded is no longer on stathost — retention deleted \
             it while the upgrade was in flight. Dropping the index entry and the sidecar the \
             upgrade wrote (detections remain available in the detection store)"
        );
        self.events.remove(camera_id, key);
        if !self.stop.stopped() {
            let _ = self.http.delete(&sidecar_key(camera_id, stem)).await;
        }
        true
    }

    /// Collect the metadata of an event whose video has just failed to upload, at the moment
    /// the orphan is created rather than at the next boot.
    async fn collect_orphaned_metadata(&self, camera_id: &str, stem: &str) {
        // Once shutdown is up, no further request is issued: the drain is
        // waiting, and the next startup's sweep collects this anyway.
        if self.stop.stopped() {
            return;
        }
        match self.http.probe_exists(&ts_key(camera_id, stem)).await {
            Ok(false) => {
                if self.stop.stopped() {
                    return;
                }
                match self.http.delete(&sidecar_key(camera_id, stem)).await {
                    DeleteOutcome::Failed => tracing::warn!(camera = %camera_id, stem = %stem,
                        "could not delete the sidecar of an event whose video never landed; \
                         the next startup collects it"),
                    _ => tracing::info!(camera = %camera_id, stem = %stem,
                        "deleted the sidecar of an event whose video never landed"),
                }
            }
            // The video is there after all: the failure was a timeout or a
            // proxy over a body the origin committed. The sidecar types it, and
            // must stay.
            Ok(true) => tracing::warn!(camera = %camera_id, stem = %stem,
                "a video upload reported failure but the object is on the host; \
                 keeping its sidecar, and the next scan indexes the event"),
            // Could not find out. Nothing is deleted on a maybe.
            Err(_) => {}
        }
    }

    /// Re-read the sidecars of events whose type an earlier scan could not establish, and index
    /// what they say.
    async fn resolve_unknown_types(&self, camera_id: &str, cancel: &std::sync::atomic::AtomicBool) {
        let held: Vec<EventKey> = match self.unknown_type.get(camera_id) {
            Some(lock) => lock.read_recover().iter().copied().collect(),
            None => return,
        };
        // Holds on events that have left the index are dropped without asking the store
        // anything.
        let held: Vec<EventKey> = held
            .into_iter()
            .filter(|&key| {
                let indexed = self.events.contains(camera_id, key);
                if !indexed {
                    self.clear_unknown_type(camera_id, key);
                }
                indexed
            })
            .collect();
        if held.is_empty() {
            return;
        }

        // Bounded, and fanned out inside the bound, for the reason
        // [`RESOLVE_BUDGET`] gives: the deletions this tick exists to make are
        // queued behind these reads.
        let deadline = tokio::time::Instant::now() + RESOLVE_BUDGET;
        let held_count = held.len();
        // Ordered, then rotated to where *this camera's* last pass stopped.
        let cursor = self.resolve_cursor.get(camera_id);
        let mut held = held;
        held.sort_unstable();
        let start = cursor.map_or(0, |c| c.load(Ordering::Relaxed) % held_count);
        held.rotate_left(start);
        let mut reads = futures_util::stream::iter(held)
            .map(|key| async move { (key, self.read_sidecar(camera_id, &key_stem(key)).await) })
            .buffered(SCAN_CONCURRENCY);

        let mut resolved = 0u64;
        let mut read_back = 0usize;
        // The budget bounds the *wait*, not just the gap between results.
        while let Ok(Some((key, read))) = tokio::time::timeout_at(deadline, reads.next()).await {
            read_back += 1;
            // `Some(sidecar)` — the outer one — is a type that is now settled; the inner
            // `None` is the settled answer "there is no sidecar", the plain movement event.
            // Unreadable or typeless settles nothing and keeps the hold.
            let settled = match read {
                SidecarRead::Parsed(s) => Some(Some(s)),
                SidecarRead::Absent => Some(None),
                SidecarRead::Unreadable | SidecarRead::Typeless(_) => None,
            };
            if let Some(sidecar) = settled {
                self.events.update(camera_id, key, |entry| {
                    apply_sidecar(entry, sidecar.as_ref())
                });
                self.clear_unknown_type(camera_id, key);
                resolved += 1;
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
        }
        // Where this camera's next tick starts. At least one, so a pass that
        // read nothing back at all still moves the window instead of re-issuing
        // the same doomed reads for ever.
        if let Some(cursor) = cursor {
            cursor.fetch_add(read_back.max(1), Ordering::Relaxed);
        }
        if resolved > 0 {
            tracing::info!(camera = %camera_id, resolved,
                "read event types that an earlier scan could not; normal retention resumes");
        }
        if read_back < held_count {
            tracing::info!(
                camera = %camera_id,
                read_back,
                held = held_count,
                "stopped re-reading held event types to get on with the sweep; the rest are \
                 re-read on the next tick and keep the longest retention until then"
            );
        }
    }

    /// Rebuild the index from the store, retrying a host that is not answering yet.
    async fn scan_with_retries(
        &self,
        kind: ScanKind,
        stop: impl Fn() -> bool,
    ) -> std::io::Result<()> {
        // Only startup is spending footage on this; a heal is a background task
        // and takes the ordinary per-request ceiling instead.
        let deadline = match kind {
            ScanKind::Startup => Some(tokio::time::Instant::now() + SCAN_LISTING_BUDGET),
            ScanKind::Heal => None,
        };
        let left = || {
            deadline.map(|d: tokio::time::Instant| {
                d.saturating_duration_since(tokio::time::Instant::now())
            })
        };
        let mut delay = SCAN_RETRY.start;
        let mut attempt = 1u32;
        loop {
            if stop() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "shutdown",
                ));
            }
            // What is left of the budget is what this attempt's listing gets, so the series
            // cannot outlast it by a whole request timeout.
            let listing_timeout = left().unwrap_or(REQUEST_TIMEOUT).min(REQUEST_TIMEOUT);
            match self.scan_once(kind, &stop, listing_timeout).await {
                Ok(ScanPass::Complete) => return Ok(()),
                // Keep partial entries but leave the index unready, preventing retention until a
                // complete startup scan.
                Ok(ScanPass::Interrupted) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "shutdown",
                    ))
                }
                Err(e) => {
                    // Decided before the wait rather than after it: a budget with no room left
                    // for both the wait and the request it leads to is spent, and issuing a
                    // listing on the remainder only replaces this error with a timeout.
                    let wait = crate::retry::jittered(delay);
                    let spent = left().is_some_and(|left| wait >= left);
                    if attempt >= SCAN_ATTEMPTS || spent || stop() {
                        // No promise of a later scan on the way out at
                        // shutdown: there is no later tick to make it.
                        if stop() {
                            tracing::info!(error = %e, attempts = attempt,
                                "abandoning the stathost warm index scan for shutdown");
                        } else {
                            tracing::warn!(
                                error = %e,
                                attempts = attempt,
                                gave_up_on_the_clock = spent,
                                "could not list stathost: the warm index does not describe \
                                 the store, so retention, the byte budget and the orphan \
                                 sweep stay paused until a later scan succeeds"
                            );
                        }
                        return Err(reqwest_io(e));
                    }
                    tracing::info!(
                        error = %e,
                        attempt,
                        delay_ms = delay.as_millis() as u64,
                        "stathost listing failed; retrying the warm index scan"
                    );
                    sleep_unless(wait, &stop).await;
                    delay = SCAN_RETRY.next(delay);
                    attempt += 1;
                }
            }
        }
    }

    /// Raise an entry the heal yielded to from movement to object, when the sidecar on the
    /// store says object and the index still says movement.
    fn join_object_type(
        &self,
        camera_id: &str,
        key: EventKey,
        sidecar: &SidecarData,
        sidecar_bytes: u64,
    ) -> bool {
        let joined = self.events.reidentify_if(camera_id, key, |entry| {
            if entry.event_type != EventType::Movement {
                return false;
            }
            // Mirrors `upgrade_event`'s index half, from the sidecar that
            // upgrade wrote rather than from the upgrade it no longer has.
            entry.event_type = EventType::Object;
            entry.object_classes = sidecar.classes.clone();
            entry.detections = sidecar.detections.clone();
            entry.backend = sidecar.backend.clone();
            entry.model = sidecar.model.clone();
            entry.continues = sidecar.continues;
            // The one size the store *is* newer about: this sidecar is the
            // one the failed upgrade wrote, and the listing weighed it.
            entry.sidecar_bytes = sidecar_bytes;
            true
        });
        if joined {
            // The type is a fact again, so no hold may outlive it — an earlier
            // interrupted pass can have indexed this event with an unreadable
            // sidecar and marked it.
            self.clear_unknown_type(camera_id, key);
        }
        joined
    }

    /// Hand a startup pass's collected entries to the index, one camera at a time and each list
    /// in one step — sorted once rather than n times.
    fn take_collected(&self, collected: HashMap<String, Vec<WarmEventEntry>>) {
        for (camera_id, entries) in collected {
            self.events.replace_camera(&camera_id, entries);
        }
    }

    /// One pass: list the bucket, index every `.ts` belonging to a camera this process owns,
    /// and — at startup only, see [`ScanKind`] — collect metadata whose video never landed.
    async fn scan_once(
        &self,
        kind: ScanKind,
        stop: &impl Fn() -> bool,
        listing_timeout: Duration,
    ) -> Result<ScanPass, reqwest::Error> {
        let start = std::time::Instant::now();
        let items = self.http.list(listing_timeout).await?;

        // Every object's name and size, so filmstrip frames can be counted — and every
        // event's full cost priced — without a single extra request.
        let sizes: HashMap<&str, u64> = items.iter().map(|i| (i.path.as_str(), i.size)).collect();

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
            let sidecar_bytes = sizes
                .get(sidecar_key(camera_id, stem).as_str())
                .copied()
                .unwrap_or(0);
            // The frame count is a high-water mark, not a tally of what is there: a filmstrip
            // missing its first frame still has the rest, and the deletes walk
            // `0..filmstrip_frames` — see [`filmstrip_frame_count`].
            let frame_size = |i: usize| sizes.get(thumb_key(camera_id, stem, i).as_str()).copied();
            let filmstrip_frames = filmstrip_frame_count(|i| frame_size(i).is_some());
            let thumbnail_bytes: u64 = (0..filmstrip_frames).filter_map(frame_size).sum();
            pending.push(ScannedEvent {
                camera_id: camera_id.to_string(),
                stem: stem.to_string(),
                start_pts_ns,
                duration_ms,
                file_size: item.size,
                sidecar_bytes,
                thumbnail_bytes,
                filmstrip_frames,
            });
        }

        let total = pending.len();
        let mut unknown_type = 0usize;
        let mut typeless = 0usize;

        // `buffered`, not `buffer_unordered`: the reads overlap, but their results are handed
        // back in listing order, so the index is built by exactly the sequence of insertions a
        // serial scan made and the warnings below keep a stable order.
        let mut reads = futures_util::stream::iter(pending)
            .map(|event| self.read_sidecar_for(event))
            .buffered(SCAN_CONCURRENCY);

        // Events this pass put in the index, events it found the live write
        // path had already indexed better than a listing can, and — of those —
        // the ones the store could still tell something about (heal only).
        let mut indexed = 0usize;
        let mut yielded = 0usize;
        let mut joined = 0usize;
        // A startup pass builds each camera's list here and hands it over whole at the end
        // (`take_collected`).
        let mut collected: HashMap<String, Vec<WarmEventEntry>> = HashMap::new();

        while let Some((event, read)) = reads.next().await {
            // One archive's worth of round trips is a long time to hold a
            // shutdown drain that is measured in one event's deletes, and an
            // index nobody is going to use is not worth finishing.
            if stop() {
                // What was collected still goes in: it came from the store and is true, exactly
                // as it was when each entry was inserted singly. What does not happen is
                // `mark_scanned` below, so nothing prunes on a half-built index.
                self.take_collected(collected);
                tracing::info!(
                    indexed,
                    of = total,
                    "stathost warm index scan stopped by shutdown; the index is not marked \
                     as describing the store and the next start scans again"
                );
                return Ok(ScanPass::Interrupted);
            }
            // Only a confirmed absence means "movement" — that is the one event written
            // without a sidecar.
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
                sidecar_bytes: event.sidecar_bytes,
                thumbnail_bytes: event.thumbnail_bytes,
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
            // A live heal yields to newer write/upgrade state already indexed; startup has no
            // concurrent writers and may overwrite. Remaining store/index disagreements wait
            // for the next startup scan.
            let key = (event.start_pts_ns, event.duration_ms);
            let landed = match kind {
                ScanKind::Startup => {
                    collected
                        .entry(event.camera_id.clone())
                        .or_default()
                        .push(entry);
                    true
                }
                ScanKind::Heal => self.events.insert_absent(&event.camera_id, entry),
            };
            if !landed {
                yielded += 1;
                // The one thing a yielded entry can be behind on: see
                // [`Self::join_object_type`].
                if let Some(s) = sidecar
                    .as_ref()
                    .filter(|s| s.event_type == Some(EventType::Object))
                {
                    if self.join_object_type(&event.camera_id, key, s, event.sidecar_bytes) {
                        joined += 1;
                    }
                }
                continue;
            }
            indexed += 1;
            if !type_known {
                self.mark_unknown_type(&event.camera_id, (event.start_pts_ns, event.duration_ms));
                unknown_type += 1;
            }
        }

        self.take_collected(collected);

        // Every event the listing named has been accounted for, so the index now describes the
        // archive rather than merely this session's writes — the whole of what retention was
        // waiting for.
        self.mark_scanned();

        if kind == ScanKind::Startup {
            self.sweep_orphaned_metadata(&items, &sizes).await;
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
        if joined > 0 {
            tracing::warn!(
                joined,
                "stathost sidecars typed events this process still had as movements: an \
                 upgrade's sidecar upload reported failure and had landed after all. Their \
                 retention class is corrected; the failed upload was reported when it happened"
            );
        }
        tracing::info!(
            total_events = total,
            // Both zero outside a heal: only there is anything else writing.
            already_indexed = yielded,
            retyped_from_store = joined,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "stathost warm index scan complete"
        );
        if kind == ScanKind::Heal {
            // At warn, like the refusals it ends: an operator who was shown the
            // pause in a production log must be shown the recovery in the same
            // one, or a fault that fixed itself reads as a fault that lasted.
            tracing::warn!(
                "stathost warm index rebuilt after an earlier scan failed; retention and the \
                 byte budget resume. Metadata whose video never landed is left for the next \
                 startup, when no upload of this process's can be mistaken for an orphan"
            );
        }
        Ok(ScanPass::Complete)
    }

    /// Claim room for an event about to be written, and hold it until the write is over.
    async fn make_room(&self, camera_id: &str, cost: u64) -> Option<Reservation<'_>> {
        // Nothing to reserve against, three ways.
        if self.budget.unlimited() || self.stop.stopped() || self.scanned_events().is_none() {
            return None;
        }
        let held = self.budget.reserve(cost);
        self.enforce_budget(camera_id).await;
        // Verified rather than assumed: eviction stops on the first refused delete, and it can
        // run out of candidates long before it runs out of overshoot.
        self.report_overshoot(
            camera_id,
            "eviction could not free enough for this event; an event larger than the whole \
             budget, or a store refusing deletes, keeps it there",
            cost,
        );
        Some(held)
    }

    /// Say that the store is above its cap, on [`Streak`]'s widening schedule.
    fn report_overshoot(&self, camera_id: &str, reason: &str, incoming_bytes: u64) {
        if !self.budget.over(self.used()) {
            return;
        }
        if let Some(over) = self.budget_overshoots.lock_recover().record() {
            tracing::warn!(
                camera = %camera_id,
                occurrences = over,
                used_bytes = self.used(),
                incoming_bytes,
                max_stored_bytes = self.budget.limit(),
                "over the stathost byte budget: {reason}. Recording continues — the cap is a \
                 number, the footage is not — but the store is above it until a later write can \
                 evict more"
            );
        }
    }

    /// Enforce the client-side storage budget: while tracked usage exceeds `max_stored_bytes`,
    /// delete the oldest events cheapest-tier-first. No-op when the budget is unlimited.
    async fn enforce_budget(&self, camera_id: &str) {
        // Nothing to enforce, and so nothing to refuse: an unlimited budget
        // never evicts on any index.
        if self.budget.unlimited() {
            return;
        }
        // Nothing to enforce *for*, either, once shutdown has been asked for.
        if self.stop.stopped() {
            return;
        }
        let Some(events) = self.scanned_events() else {
            // Counted and reported for the store, not for the camera that
            // happened to ask: one index, one budget, one fault.
            if let Some(refusals) = self.budget_refusals.lock_recover().record() {
                tracing::warn!(
                    refusals,
                    max_stored_bytes = self.budget.limit(),
                    "not enforcing the stathost byte budget: the warm index has never been \
                     rebuilt from the store, so what is stored there is unknown and uploads \
                     are running unbounded. Retention resumes when a scan succeeds"
                );
            }
            return;
        };
        if !self.budget.over(self.used()) {
            return;
        }
        let outcome = evict_tiers(
            events,
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
            || !self.budget.over(self.used()),
            || self.stop.stopped(),
            // Eviction runs on a writer task, which has no `cancel` of its own:
            // the shutdown flag is the only stop there is here, and it is the
            // one the pass above polls too.
            |cam, entry| async move {
                self.delete_event_objects(&cam, &entry, &|| self.stop.stopped())
                    .await
            },
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
        // Checked, not cast: the index entry — and so [`event_key`], the identity every
        // object of this event is keyed by — holds a `u32`, while the duration is computed as
        // `u64`.
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
        // Keep a contiguous request body for server compatibility; `Bytes` lets retries share it.
        let data = Bytes::from(concatenate_segments(&event.segments, event.total_bytes));
        let file_size = data.len() as u64;
        let event_type = event.event_type();

        let sidecar_key = sidecar_key(camera_id, &stem);
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
        let frames: &[Vec<u8>] = match &event.filmstrip_frames {
            Some(frames) => frames,
            None => &[],
        };

        // Everything this event will cost the store, priced before a byte of it is sent: the
        // video, the sidecar that types it, and the filmstrip frames.
        let sidecar_bytes = sidecar.len() as u64;
        let frame_bytes: u64 = frames.iter().map(|f| f.len() as u64).sum();
        let room = self
            .make_room(camera_id, file_size + sidecar_bytes + frame_bytes)
            .await;

        // Step 1: the sidecar, before the video.
        let sidecar_stored = self.upload(&sidecar_key, sidecar).await;
        if !sidecar_stored {
            if sidecar_required(event) {
                tracing::error!(
                    camera = %camera_id,
                    first_pts = event.first_pts,
                    bytes = event.total_bytes,
                    "dropping event: stathost sidecar upload failed"
                );
                return WriteOutcome::Failed;
            }
            tracing::warn!(camera = %camera_id, stem = %stem,
                "stathost sidecar upload failed; \
                 a scan rebuilds this movement event unchanged without it");
        }

        // Step 2: the video — retried, then dropped (logged) so a failed write is never lost
        // silently.
        let ts_key = ts_key(camera_id, &stem);
        if !self.upload(&ts_key, data).await {
            tracing::error!(
                camera = %camera_id,
                first_pts = event.first_pts,
                bytes = event.total_bytes,
                "dropping event: stathost video upload failed"
            );
            if sidecar_stored {
                self.collect_orphaned_metadata(camera_id, &stem).await;
            }
            return WriteOutcome::Failed;
        }

        // Step 3: eager filmstrip thumbnails; frame 0 doubles as the poster.
        let mut filmstrip_frames = 0usize;
        let mut stored_frame_bytes = 0u64;
        for (i, jpeg) in frames.iter().enumerate() {
            let key = thumb_key(camera_id, &stem, i);
            // Copied, not shared, because the filmstrip is typed `Arc<Vec<Vec<u8>>>` where the
            // event assembles it. Making it shareable all the way here would mean retyping it
            // at the source for four JPEGs an event.
            if !self.upload(&key, Bytes::from(jpeg.clone())).await {
                tracing::warn!(camera = %camera_id, stem = %stem, frame = i,
                    "failed to upload filmstrip thumbnail to stathost");
                break;
            }
            filmstrip_frames += 1;
            stored_frame_bytes += jpeg.len() as u64;
        }

        // Handing the bytes over from the reservation to the index, in that order and with
        // nothing between: the index takes them first and the claim is released immediately
        // after, so a concurrent `make_room` sees them once.
        let replaced = self.events.insert(
            camera_id,
            WarmEventEntry {
                start_pts_ns: event.first_pts,
                duration_ms,
                event_type,
                file_size,
                // What actually landed, not what was priced: a sidecar or a
                // frame that never arrived costs the store nothing, and the
                // budget has to agree with the host rather than with the plan.
                sidecar_bytes: if sidecar_stored { sidecar_bytes } else { 0 },
                thumbnail_bytes: stored_frame_bytes,
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
        drop(room);
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
        let sidecar_key = sidecar_key(camera_id, &stem);
        let sidecar = sidecar_json(
            Some(EventType::Object),
            Some(&upgrade.backend),
            Some(&upgrade.model),
            &upgrade.detections,
            upgrade.continues,
        );
        let sidecar_bytes = sidecar.len() as u64;
        // An upgraded sidecar carries the detections the movement event had none of, so it is
        // bigger than the one it replaces — and it is stored before anything accounts for it.
        let growth = self
            .events
            .find(camera_id, key)
            .map_or(0, |entry| sidecar_bytes.saturating_sub(entry.sidecar_bytes));
        let room = self.budget.reserve(growth);
        if !self
            .upload(&sidecar_key, Bytes::from(sidecar.into_bytes()))
            .await
        {
            tracing::error!(camera = %camera_id, stem = %stem,
                "failed to upload upgraded sidecar to stathost, aborting upgrade");
            return;
        }

        // The retention sweep runs on its own task and deletes an event's objects one request
        // at a time, so it can overtake an upgrade between the check above and the `PUT` that
        // has just landed.
        let reclassified = self.events.reidentify(camera_id, key, |entry| {
            entry.event_type = EventType::Object;
            entry.object_classes = upgrade.object_classes.clone();
            entry.detections = upgrade.detections.clone();
            entry.backend = Some(upgrade.backend.clone());
            entry.model = Some(upgrade.model.clone());
            // The sidecar just written carries the upgrade's `continues`;
            // the index has to say the same thing (LocalDisk rebuilds the
            // whole entry here, which is where this was being lost).
            entry.continues = upgrade.continues;
            entry.sidecar_bytes = sidecar_bytes;
        });
        drop(room);
        if !reclassified {
            tracing::warn!(
                camera = %camera_id,
                stem = %stem,
                "retention deleted this event while its object upgrade was in flight; \
                 removing the sidecar the upgrade wrote back \
                 (detections remain available in the detection store)"
            );
            // Not once shutdown is up: the next startup's orphan sweep collects
            // it, and issuing a request here is the drain waiting for one.
            if !self.stop.stopped() {
                let _ = self.http.delete(&sidecar_key).await;
            }
            self.clear_unknown_type(camera_id, key);
            return;
        }
        // The type is now established.
        self.clear_unknown_type(camera_id, key);
        if self
            .abandon_upgrade_of_a_vanished_video(camera_id, &stem, key)
            .await
        {
            return;
        }
        // The growth is in the index now, and nothing evicted for it. Say so if
        // it took the store over: this upgrade may be the last thing this
        // camera does for a while, and there is no write behind it to notice.
        self.report_overshoot(
            camera_id,
            "an object upgrade's sidecar grew past it, and an upgrade does not evict — a \
             sidecar's growth is not worth an event's footage",
            growth,
        );

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
        // Unknown types use the longest retention so they terminate without early deletion.
        let unknown_max_age = movement_max_age_ns
            .max(object_max_age_ns)
            .max(continuous_max_age_ns);

        // Check shutdown between every potentially timeout-bound HTTP deletion.
        let stop = || cancel.load(Ordering::Relaxed);
        // What the per-request gates poll.
        let stopping = || stop() || self.stop.stopped();

        // This tick is also what heals an index no scan has ever filled: the startup attempts
        // are bounded, so without a retry here a store that came back a minute after boot would
        // stay unpruned until the next restart.
        if self.scanned_events().is_none() && !stop() {
            // Its own failure is already reported; here it only decides whether
            // this tick has an index to sweep.
            let _ = self.scan_with_retries(ScanKind::Heal, stop).await;
        }
        let Some(events) = self.scanned_events() else {
            // Silent when it is shutdown that cut the scan short: there is no
            // next tick to promise, and the next start says all of this again.
            if !stop() {
                tracing::warn!(
                    "not pruning stathost: the warm index has never been rebuilt from the \
                     store, so an empty index would be read as an empty archive and expired \
                     footage is accumulating unseen. The next tick retries the scan"
                );
            }
            return;
        };

        for camera_id in events.camera_ids() {
            if stop() {
                break;
            }
            // First give held events a chance to be typed, so one that resolves is pruned on
            // its real retention in this same sweep.
            self.resolve_unknown_types(camera_id, cancel).await;

            let expired = events.expired_for_sweep(camera_id, now_ns, |e| {
                if self.has_unknown_type(camera_id, event_key(e)) {
                    unknown_max_age
                } else {
                    max_age(e.event_type)
                }
            });
            if expired.is_empty() {
                continue;
            }

            let outcome = sweep_expired(events, camera_id, expired, stop, |entry| async move {
                self.delete_event_objects(camera_id, &entry, &stopping)
                    .await
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
        // "Free space" is what is left of the client-side budget once both what
        // is stored and what is being written are counted; unlimited budgets
        // report the max so the guard never fires.
        Ok(self.budget.remaining(self.used()))
    }

    async fn scan(&self) -> std::io::Result<()> {
        // Nothing of this process's is recording yet, so this is the one scan that may also
        // collect orphans — and the one whose failures are paid for in footage nobody
        // records, which is what bounds the series by the clock.
        self.scan_with_retries(ScanKind::Startup, || false).await
    }

    fn recover_orphans(&self) {
        // Interrupted uploads are a server-side concern; nothing to salvage client-side.
    }

    fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry> {
        self.events.query(camera_id, page)
    }

    fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        self.events.newest_event_end_ns(camera_id)
    }

    /// Resolved by stem — start and duration — with the event type in the key deliberately
    /// ignored.
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
        let key = ts_key(camera_id, &key_stem(event_key(entry)));
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
            // A 206 says the body is a *slice*, and only its `Content-Range` says which one.
            let content_range = resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range)
                .and_then(|(start, end, total)| {
                    ServedRange::partial(start, end, total).map(|served| (served, total))
                });
            match content_range {
                Some((served, total)) => (served, total),
                None => {
                    let header = resp
                        .headers()
                        .get(reqwest::header::CONTENT_RANGE)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("(absent)")
                        .to_string();
                    tracing::warn!(
                        key = %key,
                        content_range = %header,
                        "stathost answered a range request with a 206 whose Content-Range \
                         is missing or does not describe a range of the object; refusing to \
                         serve the partial body as if it were the whole event"
                    );
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "206 without a usable Content-Range",
                    ));
                }
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
        let stem = key_stem(event_key(entry));
        self.http
            .get(&thumb_key(camera_id, &stem, 0))
            .await
            .map_err(|_| ThumbnailError::ReadFailed)
    }

    async fn read_filmstrip(
        &self,
        camera_id: &str,
        entry: &WarmEventEntry,
        index: u8,
    ) -> std::io::Result<Vec<u8>> {
        let stem = key_stem(event_key(entry));
        self.http
            .get(&thumb_key(camera_id, &stem, usize::from(index)))
            .await
            .map_err(reqwest_io)
    }
}

/// Thin reqwest wrapper over the stathost object API. `base` is
/// `{url}/{bucket}` with no trailing slash.
struct Http {
    /// Every call that completes within one request/response, bounded by a
    /// per-request total timeout.
    client: reqwest::Client,
    /// Playback only. A total timeout would cut long streams short, so this one carries
    /// [`STREAM_READ_TIMEOUT`] instead — which in turn would be wrong for the uploads on
    /// `client`, where reqwest counts the whole request body write against it.
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

    /// Whether an object is on the host, without fetching it: a one-byte ranged GET, so probing
    /// a video worth tens of megabytes costs one byte (the body is dropped unread in any case).
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

    /// Start a streamed GET, optionally forwarding a single `Range`. The raw response is
    /// returned unvalidated so the caller can distinguish `206` (partial), `200` (full /
    /// range-ignored) and `416` (unsatisfiable).
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

    /// List every object in the bucket via the detailed listing (stathost >= 0.2.0). No
    /// fallback: an unexpected response shape is an error, surfaced by the caller.
    async fn list(&self, timeout: Duration) -> Result<Vec<ListEntry>, reqwest::Error> {
        self.client
            .get(format!("{}/_meta/list?detail=true", self.base))
            .bearer_auth(&self.token)
            .timeout(timeout)
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

/// One `.ts` object [`StathostBackend::scan`] is going to index, carrying everything the
/// listing alone settles about it. Splitting that off from the sidecar read is what lets the
/// reads overlap while the index is still built in listing order.
struct ScannedEvent {
    camera_id: String,
    stem: String,
    start_pts_ns: u64,
    duration_ms: u32,
    file_size: u64,
    /// What this event's sidecar and its filmstrip frames weigh, off the
    /// listing — the only account there is of what an event's metadata costs
    /// the store.
    sidecar_bytes: u64,
    thumbnail_bytes: u64,
    filmstrip_frames: usize,
}

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
/// `{camera_id}/{stem}_thumb_{i}.jpg` — into `(camera_id, stem)`, where the stem is the one
/// its `.ts` sibling carries.
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

/// Write the sidecar-derived half of an index entry, `None` meaning "no sidecar exists" — the
/// plain movement event.
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

/// Whether this event's sidecar carries anything a sidecar-less scan would not already assume.
fn sidecar_required(event: &FinishedEvent) -> bool {
    event.event_type() != EventType::Movement || event.continues
}

/// Whether a failed stathost request is worth sending again.
fn recoverable(e: &reqwest::Error) -> Recovery {
    match e.status() {
        Some(status) => {
            if status.is_server_error()
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status == reqwest::StatusCode::REQUEST_TIMEOUT
            {
                Recovery::Transient
            } else {
                Recovery::Permanent
            }
        }
        None => Recovery::Transient,
    }
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
mod tests;
