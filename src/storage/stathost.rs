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
//! [`event_index`](crate::storage::event_index). Neither is the *policy* those
//! skeletons are run under: what a backend must guarantee about durability,
//! accounting, cancellation and index acceptance is written down once in
//! [`contract`](crate::storage::contract), along with the mechanics — the
//! shutdown flag, the retry schedule and its classification, the byte
//! reservation — that keep the answer the same here as it is on local disk.
//! What is genuinely this backend's is below.
//!
//! Notable divergences from [`LocalDiskBackend`], all deliberate:
//!
//! * **Retention-by-space is a client-side budget.** The client can't see the
//!   server's disk, so `max_stored_bytes` caps tracked usage; when it is
//!   exceeded the oldest events are evicted cheapest tier first, the same order
//!   and the same skeleton as the local emergency prune — but with the opposite
//!   failure policy, for the reason
//!   [`EvictionPolicy`](crate::storage::event_index::EvictionPolicy) gives. The
//!   disk-shaped `min_free_bytes` guard argument is ignored here. Where local
//!   disk has `statvfs` and `ENOSPC` — an arbiter that counts every byte and
//!   settles every race for the last of them — this has an in-RAM sum, so it
//!   has to do both of those jobs by hand: an event is charged its *whole* cost
//!   (video, sidecar, filmstrip frames) and a write *reserves* that cost before
//!   it uploads a byte of it, so concurrent writers see each other. See
//!   [`StathostBackend::make_room`].
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
//! * **Metadata whose video never landed is collected twice over.** The write
//!   that orphaned it collects it — one probe, on a write that has already
//!   failed, at the one moment there is no ambiguity about whose upload this is
//!   ([`StathostBackend::collect_orphaned_metadata`]) — and the startup scan
//!   sweeps whatever a crash or a lost process left behind
//!   ([`StathostBackend::sweep_orphaned_metadata`]). Only the *startup* scan
//!   may do the second one, for the reason [`ScanKind`] gives, which is exactly
//!   why the first one exists: a box that stays up for months would otherwise
//!   accumulate one orphan per failed upload until it was rebooted.
//! * **An unreadable sidecar is not a movement event.** The scan applies the
//!   movement default only to a *confirmed* 404; anything else — a transport
//!   failure, unparsable bytes, valid JSON naming no type — leaves the type
//!   unknown. Such an event is still indexed and served, but every decision
//!   that would need its type errs toward keeping it: age-based pruning
//!   measures it against the longest configured retention, and budget eviction
//!   tiers it with the objects. The prune tick re-reads its sidecar, one of the
//!   two things it retries; the other is the scan itself.
//! * **An upgrade checks that the sweep did not overtake it.** Retention runs
//!   on its own task and deletes an event one request at a time, so it can
//!   remove an event between the upgrade's "is this indexed?" and the sidecar
//!   `PUT` that follows. A `PUT` always succeeds, so that combination used to
//!   write the sidecar of a deleted event back onto the store and report an
//!   upgrade. Local disk cannot reach it: there the commit is a *rename* of the
//!   video, which fails once the video is gone. Here the check has to come
//!   afterwards, and deleting the object it just created is what it costs when
//!   it fires.
//! * **An index that has never been rebuilt is not an empty archive.** In RAM
//!   the two are the same object and to retention they are opposite
//!   instructions: on an empty archive there is nothing to prune, no bytes
//!   against the budget and no orphan to collect, while on an unknown one every
//!   such conclusion is a guess made against footage that is still there and
//!   still growing. A listing that fails at startup — a boot-time network race
//!   is enough, the unit only orders after `network.target` — must therefore not
//!   read as "the store is empty". So the scan is retried ([`SCAN_RETRY`]) on a
//!   deadline ([`SCAN_LISTING_BUDGET`], because startup is time nothing is
//!   recording), and until one of those attempts walks a listing to the end the
//!   backend is *un-scanned*: retention, the byte budget and the orphan sweep
//!   refuse to run and say so, writes and reads carry on, and the retention tick
//!   keeps retrying the scan until it heals. Success is sticky — the state means
//!   "never rebuilt", not "the last request failed" — see
//!   [`StathostBackend::scanned_events`]. A healing scan is the one scan that is
//!   not the only writer of the index, so it yields to what the live write path
//!   has already put there ([`ScanKind::Heal`]).
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
    evict_tiers, sweep_expired, EmergencyOutcome, EventIdentity, EventIndex, EventPage,
    EvictionPolicy, Removal,
};
use crate::storage::warm_index::{
    parse_event_filename, parse_sidecar_json, sidecar_json, SidecarData,
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
///
/// Visible to the crate because the shutdown drain's budget is divided around
/// it: phase 3 is sized so that one upload can always use its whole timeout,
/// and [`crate::shutdown`] pins that against this constant rather than against
/// a copy of the number.
pub(crate) const UPLOAD_TIMEOUT: Duration = Duration::from_secs(300);
/// Idle budget for the streaming client only. reqwest arms it flat until the
/// response headers arrive, then per response frame with a reset on each — the
/// right shape for a body a player drains at its own pace. The flat phase is
/// harmless here because a ranged GET carries no request body.
const STREAM_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long one object request waits before being sent again, and how far that
/// wait grows. Only the first step is reachable at [`OBJECT_RETRY`]'s two
/// attempts; the doubling is here so that raising the allowance does not also
/// have to invent a schedule.
///
/// The wait is what was missing rather than the retry: an upload used to be
/// re-sent the instant it failed, which on the failure that actually happens —
/// a store that is briefly not there — spends the second attempt inside the same
/// outage as the first. Jittered on the way out, because a rack of cameras
/// coming back from one power cut is a fleet of camons re-sending at the same
/// second.
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

/// What one object request — an upload, a sidecar read — is worth: the attempt
/// and one more, waited out and classified.
///
/// Two is the allowance this backend has always given, and it stays two: a
/// third attempt at an event's video is another [`UPLOAD_TIMEOUT`] held by a
/// camera's writer, and the failures that outlast two attempts are outages that
/// outlast any number of them. What is new is that a refusal the store will
/// repeat — a bad token, a path it will not take — costs one attempt instead of
/// two, and that neither attempt is issued once shutdown has been asked for.
const OBJECT_RETRY: RetryPolicy = RetryPolicy {
    attempts: 2,
    schedule: OBJECT_RETRY_WAIT,
};

/// Wall clock one prune tick will spend, per camera, re-reading the sidecars of
/// events whose type an earlier scan could not establish.
///
/// The re-reads used to be unbounded and one at a time, ahead of every deletion
/// the sweep was there to make: a store holding a thousand such events, on a
/// link where a sidecar GET takes a second, put retention an archive's worth of
/// round trips behind schedule every hour — and the sweep is what reclaims the
/// space, so the pause lands exactly where space is already the problem. So the
/// pass fans out like the scan does and is cut off against the clock — the
/// *wait* for the next result is bounded by what is left of this, not merely
/// the gap between results, or one read at the head of the fan-out could spend
/// two request timeouts inside a ten-second budget.
///
/// The holds it did not reach are re-read on the next tick, starting where this
/// camera's last pass stopped, so a prefix that never resolves cannot starve the
/// tail behind it. *This camera's* — a window shared between cameras advances by
/// the sum of what all of them read and can therefore land on the same place
/// every tick, which is starvation arrived at from the other direction; see
/// [`StathostBackend::resolve_cursor`]. Nothing is lost by waiting either: a held event is measured against the
/// longest configured retention meanwhile, which cannot expire before its true
/// one — so the worst a hold that waits several ticks costs is that its footage
/// is kept a little longer than it had to be.
#[cfg(not(test))]
const RESOLVE_BUDGET: Duration = Duration::from_secs(10);
/// Short enough under test that a hold list which would take seconds to read
/// through is visibly cut off, long enough that a healthy re-read of a handful
/// of sidecars finishes inside it.
#[cfg(test)]
const RESOLVE_BUDGET: Duration = Duration::from_millis(50);

/// How many requests a [`scan`](StathostBackend::scan) keeps in flight — its
/// sidecar reads, and the orphan sweep's probes and deletes.
///
/// The startup scan is awaited by `init_storage` *before* the first camera is
/// spawned, so every round trip it makes is time nothing is recording. Its work
/// is one small GET per stored event, and those are latency-bound rather than
/// bandwidth-bound — a sidecar is a few hundred bytes — so awaiting them one at
/// a time makes startup a function of the archive's size: a bucket holding a
/// few thousand events costs that many round trips, minutes of them on any link
/// that is not a LAN. The per-request timeouts bound each one, not the total.
/// (A heal costs no recording time — it runs on the retention task while the
/// cameras are up — but it does compete with their uploads for the one host,
/// which the same width suits.)
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

/// How long a scan waits before asking the store again, and how far that wait
/// grows: 2s, 4s, 8s, 16s — the four waits [`SCAN_ATTEMPTS`] leaves room for,
/// the last of them the cap.
///
/// Jittered like every other retry here. Only one scan runs at a time in one
/// process, so nothing of camon's own retries in lockstep with it; a rack of
/// cameras coming back from one power cut is a fleet of camons listing one host
/// at the same second, which is the same pile-up seen from the server's side.
#[cfg(not(test))]
const SCAN_RETRY: RetrySchedule = RetrySchedule {
    start: Duration::from_secs(2),
    max: Duration::from_secs(16),
};
/// Milliseconds under test: the tests here pin *how many* attempts are made and
/// what state the failures leave behind, and neither needs the wall clock. The
/// clock cannot be paused instead — these tests drive a real HTTP stub, and a
/// paused runtime advances time while a request is in flight, which fires the
/// request timeouts.
#[cfg(test)]
const SCAN_RETRY: RetrySchedule = RetrySchedule {
    start: Duration::from_millis(1),
    max: Duration::from_millis(4),
};

/// Listings one scan is worth before it gives up and leaves the backend
/// un-scanned. Giving up is not the end of it: the retention tick starts the
/// series again until one succeeds.
const SCAN_ATTEMPTS: u32 = 5;

/// Wall clock a *startup* scan may spend failing: listings that never arrived
/// and the waits between them. Whichever runs out first — this or
/// [`SCAN_ATTEMPTS`] — ends the series.
///
/// The attempt count alone bounds nothing worth bounding. A refused connection
/// fails in a millisecond, so five of those cost the waits and nothing else; a
/// host that accepts the connection and then says nothing — half-open link,
/// wedged server, a firewall dropping instead of rejecting — costs a full
/// [`REQUEST_TIMEOUT`] every time, and five of those is five minutes. Startup
/// awaits this series *before any camera is spawned*, so five minutes of it is
/// five minutes of a camera system recording nothing. That is worse than
/// starting un-scanned, which costs retention until a tick heals it.
///
/// So it is a deadline: each attempt's listing gets what is left of it, and the
/// series stops once it is spent. Startup is therefore dark for this long at
/// worst, whatever the host does.
///
/// A heal has no deadline at all. Nothing is off the air while it runs — it is
/// a background task, and shutdown stops it between requests — so its listings
/// get the ordinary [`REQUEST_TIMEOUT`] each. That is also what keeps this
/// deadline from becoming a trap: a bucket whose listing genuinely needs longer
/// than startup was willing to wait is scanned by the first heal instead, a
/// minute later. Truncating that one too would turn "slow store" into an
/// un-scanned backend no restart could clear — the shape of fault this whole
/// state machine exists to end.
///
/// The band between the two is a real gap, stated rather than papered over: an
/// installation whose listing reliably takes between this and
/// [`REQUEST_TIMEOUT`] never completes a *startup* scan, on any boot. Its index
/// and its retention are healed a minute in and are no worse for it, but the
/// orphan sweep is startup's alone — for the reason [`ScanKind`] gives — so
/// that one installation never collects the metadata of uploads whose video
/// failed, and accumulates it. Widening this constant to cover such a store
/// would buy that back for the price of the same delay before every camera on
/// every boot, which is the trade this number is.
///
/// Neither ceiling touches the indexing pass. Once a listing arrives it is
/// walked to the end however long that takes: a half-built index is not a
/// rebuilt one, so cutting it short would leave the backend un-scanned every
/// time and never heal either.
#[cfg(not(test))]
const SCAN_LISTING_BUDGET: Duration = Duration::from_secs(45);
/// Long enough under test that a series of fast refusals never runs into it
/// (five of those cost about 15ms of waits), short enough that the test which
/// pins the deadline against a listing that never answers finishes in it.
#[cfg(test)]
const SCAN_LISTING_BUDGET: Duration = Duration::from_millis(500);

/// Which scan this is, and so whether it may collect orphaned metadata.
///
/// The orphan sweep asks "is there a `.ts` for this sidecar?" and deletes the
/// metadata when the answer is no. That question only has an honest answer
/// while nothing of this process's is mid-upload: the sidecar goes up *before*
/// the video, so a camera uploading a large event looks exactly like an orphan
/// for as long as the upload takes — up to [`UPLOAD_TIMEOUT`], not the one
/// round trip the module header calls the residual race.
#[derive(Clone, Copy, PartialEq)]
enum ScanKind {
    /// The scan `init_storage` awaits before the first camera is spawned.
    /// Nothing of this process's can be in flight, so the sweep runs.
    Startup,
    /// A later attempt, from the retention tick, healing an un-scanned index
    /// while cameras record. It rebuilds the index — which is what retention
    /// was waiting for — and leaves orphaned metadata for the next startup,
    /// which can tell an orphan from an upload in progress.
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
    /// Client-side storage budget in bytes; 0 means unlimited. Measured against
    /// [`EventIndex::used_bytes`], which the index maintains as the sum of what
    /// every entry costs the store ([`WarmEventEntry::stored_bytes`]) — the two
    /// cannot drift. The budget also carries the reservations that keep two
    /// cameras writing at once from both reading a total that predates the
    /// other; see [`crate::storage::contract`].
    budget: ByteBudget,
    /// Shutdown, as the drain raises it. Every request-issuing loop here reads
    /// it before sending, so a stop costs at most the one request already in
    /// flight — which is exactly what the drain's phase 3 is sized for.
    stop: StopFlag,
    events: EventIndex<EventKey>,
    /// Whether [`Self::events`] has ever been rebuilt from what the store
    /// actually holds. Set by the first scan that walks a listing to the end
    /// and never cleared; read only through [`Self::scanned_events`], which
    /// hands out the index itself so that a caller who must not act on an
    /// un-scanned one cannot reach it without answering the question.
    scanned: std::sync::atomic::AtomicBool,
    /// How many times a write has gone ahead over the byte budget because
    /// eviction could not free enough for it — reported on [`Streak`]'s
    /// widening schedule for the same reason the refusals are: this is asked
    /// before every write, and a store that is permanently over its cap would
    /// otherwise be one warning per event for ever.
    budget_overshoots: std::sync::Mutex<Streak>,
    /// How many times budget enforcement has refused to run because of that,
    /// across every camera — the index it refuses on is one index, and so is
    /// the budget. Enforcement is asked before every write, so the refusal is
    /// reported on the widening schedule [`Streak`] exists for rather than once
    /// per event.
    budget_refusals: std::sync::Mutex<Streak>,
    /// Events whose sidecar the scan could not read, per camera. Their
    /// [`WarmEventEntry::event_type`] is a placeholder, not a fact — see
    /// [`Self::mark_unknown_type`]. This has no local-disk counterpart: there
    /// the type is the directory, so it cannot be unreadable.
    unknown_type: HashMap<String, RwLock<HashSet<EventKey>>>,
    /// Where the next [`Self::resolve_unknown_types`] pass starts in this
    /// camera's hold list. Only its movement matters, not its value.
    ///
    /// Per camera, beside the hold list it indexes into, because one cursor
    /// shared across cameras does not merely advance untidily — it *resonates*.
    /// Each pass advances the shared value by its own camera's read count, so
    /// with C cameras every camera's window jumps by the sum of all of them per
    /// tick; a camera whose hold list divides that sum lands on the same window
    /// every hour and never reads the rest of its list again. Two cameras with
    /// 32 stable holds apiece, reading a fan-out's worth each, is enough: 32
    /// advance, 32 modulo 32 is zero, and both freeze. And the conditions are
    /// not exotic — they are the steady state this pass is for, where the reads
    /// succeed, take one attempt, and quantize to [`SCAN_CONCURRENCY`] under
    /// uniform latency.
    resolve_cursor: HashMap<String, std::sync::atomic::AtomicUsize>,
}

impl StathostBackend {
    pub fn new(config: &StathostConfig, camera_ids: &[String], stop: StopFlag) -> Self {
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
        }
    }

    fn used(&self) -> u64 {
        self.events.used_bytes()
    }

    /// The index as retention may see it: `None` until a scan has rebuilt it
    /// from the store.
    ///
    /// The gate hands back the index rather than answering a question about it,
    /// so that the two paths which must not act on an un-scanned index —
    /// [`WarmStorageBackend::prune`] and [`Self::enforce_budget`] — cannot
    /// reach the events they would delete without going through it. The third
    /// deleter, the orphan sweep, needs no gate here: it lives inside
    /// [`Self::scan_once`] and runs only on the listing that pass just got,
    /// which is a stronger guarantee than this one. Everything else (writes,
    /// reads,
    /// the API's queries) is correct either way and uses [`Self::events`]
    /// directly: an event this process wrote is in there whether or not a scan
    /// ever ran, and a query that can only offer this session's events is a
    /// thin answer, not a wrong one. Deleting on that same index *is* wrong.
    fn scanned_events(&self) -> Option<&EventIndex<EventKey>> {
        self.scanned.load(Ordering::Acquire).then_some(&self.events)
    }

    /// Record that the index now describes the store. Sticky by design: a
    /// transient failure of anything afterwards leaves this set, because the
    /// question it answers is "has the archive ever been read", and the answer
    /// to that cannot become no again.
    fn mark_scanned(&self) {
        self.scanned.store(true, Ordering::Release);
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

    /// Drop the marker once the type is settled. Only a scan ever sets one, and
    /// scans are rare — one at startup, and one per retention tick for as long
    /// as the startup one never got a listing — so in practice this fires where
    /// a type becomes a fact: `write_event` and `upgrade_event`, neither of
    /// which may leave a "type unknown" marker on an event it just proved.
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

    /// Read and parse one event's sidecar, through the same
    /// [`OBJECT_RETRY`] every object request here goes through — worth the
    /// allowance because the alternative to a readable sidecar is a guessed
    /// retention class.
    async fn read_sidecar(&self, camera_id: &str, stem: &str) -> SidecarRead {
        let key = format!("{camera_id}/{stem}.json");
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

    /// Upload one object, retried and classified by [`OBJECT_RETRY`] and
    /// abandoned rather than re-sent once shutdown has been asked for.
    /// `false` means the object is not known to be on the host — which is not
    /// the same as known not to be, and no caller here treats it as such.
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
    ///
    /// **Within the thumbnails, top down** — highest index first, the same
    /// direction and for the same reason as [`Self::trim_thumbnails`]. The scan
    /// counts an event's frames contiguously from 0 and stops at the first gap,
    /// so what an interrupted deletion leaves has to be a *prefix* to be seen
    /// at all. Deleting `0` first and stopping there leaves `1..` with no `0`
    /// in front of them: the next scan counts zero frames, the orphan sweep
    /// passes over them because the video they belong to is still on the store,
    /// and this method — which only ever deletes as many frames as the entry
    /// records — later takes the event away without them. Those objects are
    /// then unreachable by anything until some restart that happens after the
    /// video has gone, which on a box that stays up is never. Top down cannot
    /// produce that: whatever survives is `0..=n`, the scan counts it, the
    /// budget is charged for it, and the event's own deletion covers it.
    ///
    /// A frame the store *refuses* ends the descent for the same reason. It is
    /// still there afterwards, so deleting the ones below it would leave a gap
    /// around it — the identical stranding, reached without any interruption at
    /// all: one refused `DELETE` in the middle of a healthy pass. Stopping
    /// leaves `0..=i`, which is a prefix.
    ///
    /// It does *not* stop the event's deletion, and that asymmetry is the
    /// existing rule rather than a new one: **the video's outcome is the
    /// event's outcome**. A thumbnail that will not go is decoration that will
    /// not go; holding an expired recording for it breaks a larger promise than
    /// the kilobytes it saves, and it is not flagged as a refusal either —
    /// [`WarmEventEntry::delete_failed`] is what makes eviction demote an event
    /// and what the sweep counts, and neither should turn on a JPEG. So the
    /// pass goes on to the video, whose outcome decides everything, and the
    /// frames left behind become orphans the startup sweep collects once their
    /// video is gone — the same fate a refused frame has always had here.
    ///
    /// **Every one of these requests is gated on the shutdown flag**, not just
    /// the pass around them. One event is up to six sequential `DELETE`s, each
    /// able to sit on a [`REQUEST_TIMEOUT`]: checking only between *events*
    /// leaves six minutes of post-stop work inside one of them, which is more
    /// than the drain's whole phase-3 budget. That gate is what makes the
    /// orderings above load-bearing rather than decorative — before it, the
    /// only way to stop mid-event was a crash. Every step now has its reason:
    /// thumbnails top down so a stop leaves a countable prefix, the video next
    /// so nothing survives it that the scan would mistype, and the sidecar last
    /// so an abandoned delete leaves an orphan the startup sweep collects
    /// rather than a `.ts` whose retention class the next scan has to guess.
    ///
    /// `stopping` is the caller's signal, not this backend's own. That
    /// distinction is the whole point: the trait promises that
    /// [`prune`](WarmStorageBackend::prune)'s `cancel` is honoured *between
    /// requests*, and honouring it here is the only place that can be true. A
    /// backend that polled only its constructor flag would keep that promise by
    /// coincidence — production hands it the same `AtomicBool` — and break it
    /// for any caller that raised `cancel` on its own, which is precisely what a
    /// test, or a future caller with a narrower stop, would do. The write path
    /// has no `cancel` to offer and passes the shutdown flag instead.
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
                self.http
                    .delete(&format!("{camera_id}/{stem}_thumb_{i}.jpg"))
                    .await,
                DeleteOutcome::Failed
            ) {
                // Stop descending, but do not stop deleting the event. A frame
                // the store refused is a frame that is still there, so carrying
                // on to the ones below it would leave a gap around it — and the
                // whole point of going top down is that what survives is a
                // prefix. Everything below stays, and the set is `0..=i`.
                tracing::debug!(camera = %camera_id, stem = %stem, frame = i,
                    "failed to delete a filmstrip frame; keeping the frames below it so what \
                     is left is still contiguous, and going on to the video");
                break;
            }
        }
        if stopping() {
            return Removal::Abandoned;
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
        // The video is gone, so the entry must go with it whatever happens
        // next: this is the one request whose omission leaves an orphan rather
        // than an inconsistency, and the startup sweep is what collects it.
        if !stopping() {
            let _ = self.http.delete(&format!("{camera_id}/{stem}.json")).await;
        }
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
    /// Stopping for shutdown leaves exactly what a refused delete leaves —
    /// `0..=i` still on the host and the entry saying so — so the flag is
    /// checked in the same place, and for the same reason every other delete
    /// loop here checks it: these are `REQUEST_TIMEOUT`-sized requests on a
    /// task the drain is waiting for.
    async fn trim_thumbnails(&self, camera_id: &str, key: EventKey, keep: usize, had: usize) {
        let stem = key_stem(key);
        for i in (keep..had).rev() {
            if self.stop.stopped() {
                self.events
                    .update(camera_id, key, |entry| entry.filmstrip_frames = i + 1);
                return;
            }
            if matches!(
                self.http
                    .delete(&format!("{camera_id}/{stem}_thumb_{i}.jpg"))
                    .await,
                DeleteOutcome::Failed
            ) {
                tracing::warn!(camera = %camera_id, stem = %stem, frame = i,
                    "could not delete a filmstrip frame this event no longer has; \
                     it stays part of the event and is deleted with it");
                // The frames this event keeps past the new count are the
                // *previous* write's, and nothing here knows what they weigh —
                // they were never priced, because they are not what was just
                // uploaded. So the entry under-counts by their size until the
                // next scan prices the stem off a listing. Bounded by a few
                // JPEGs of an event that has already had one delete refused.
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
    async fn sweep_orphaned_metadata(&self, items: &[ListEntry], sizes: &HashMap<&str, u64>) {
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

    /// The other half of the upgrade/sweep race, and the half the index cannot
    /// see.
    ///
    /// [`WarmStorageBackend::upgrade_event`]'s index check catches the sweep
    /// that got as far as unindexing the event. It cannot catch the sweep that
    /// has deleted the objects and has *not* reached its `index.remove` yet:
    /// the entry is still there, the reclassification lands on it, and the
    /// sidecar this call just wrote is sitting on a store that no longer holds
    /// the video it describes. Inverting the sweep to unindex first is not
    /// available — the delete order is what makes an interrupted delete
    /// recoverable — so the question is asked of the store instead, exactly as
    /// the write path asks it ([`Self::collect_orphaned_metadata`]), and after
    /// the `PUT` rather than before, which is what makes the answer conclusive:
    ///
    /// * the sweep deleted the video before this probe — the probe sees that,
    ///   and this call takes back both the sidecar and the index entry;
    /// * the sweep deletes it after — then its own sidecar delete, which comes
    ///   after its video delete, removes what this call wrote. Nothing leaks
    ///   either way.
    ///
    /// Returns whether the upgrade was taken back. The residual is now confined
    /// to the probe itself: a probe that cannot be made — the store did not
    /// answer, or shutdown is up — leaves the upgrade standing. Nothing is taken
    /// back on a maybe, because taking back the entry of an event that is really
    /// there would hide live footage until the next restart.
    ///
    /// What that leaves costs three things, not two, and they only all appear
    /// together in one shape: the sweep's video `DELETE` committed but its
    /// response was lost, so the sweep classified it `Failed` and kept the entry
    /// (flagged), *and* this probe was unanswerable too. Then there is an orphan
    /// sidecar, a log line saying "upgraded" about footage that has gone, and —
    /// the third — a stale index entry still charged to the byte budget and
    /// still offered for playback, describing a video that is not there.
    ///
    /// When it heals depends on what the reclassification did to the entry's
    /// retention, and the honest bound has two halves:
    ///
    /// * If the event is *still* expired as an object — an old enough
    ///   recording, or an installation whose object retention is no longer than
    ///   its movement retention — the **next prune tick** reaches it. It is
    ///   selected by age like anything else, the `delete_failed` flag it
    ///   carries exempts it from the per-sweep cap so it cannot be held back,
    ///   the video delete comes back a confirmed `Missing`, and the entry is
    ///   unindexed — bytes refunded — with the sidecar deleted after it. An
    ///   hour.
    /// * If the reclassification lifted it *out* of expiry, which is the
    ///   ordinary case (a three-day-old movement past a two-day retention is a
    ///   three-day-old object well inside a fourteen-day one), the next tick
    ///   does not select it at all. `delete_failed` exempts an entry that has
    ///   been selected from the cap; it does not put one on the list. So the
    ///   entry stands — charged to the budget, offered for playback, describing
    ///   a video that is not there — until object retention catches up with it
    ///   or the next start rebuilds the index from a listing that no longer
    ///   names it, whichever comes first. Days, bounded by the retention
    ///   config, rather than an hour.
    ///
    /// Selecting flagged entries regardless of age would collapse the second
    /// case into the first, and it is not worth what it costs: `delete_failed`
    /// is set by any transient refusal, so that rule would delete correctly
    /// classified object footage the moment one delete failed against it. A
    /// stale entry that is visible and bounded beats retention that means
    /// something different for every event a network hiccup has touched.
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
            self.http
                .probe_exists(&format!("{camera_id}/{stem}.ts"))
                .await,
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
            let _ = self.http.delete(&format!("{camera_id}/{stem}.json")).await;
        }
        true
    }

    /// Collect the metadata of an event whose video has just failed to upload,
    /// at the moment the orphan is created rather than at the next boot.
    ///
    /// [`Self::sweep_orphaned_metadata`] is the backstop and cannot be more
    /// than that: it needs a whole-bucket listing, and it may only run at
    /// startup, because from a listing alone an upload in progress is
    /// indistinguishable from an orphan. Here there is no such ambiguity — this
    /// process's own upload has just failed, on this task, and nothing else of
    /// its own is writing this stem — so the same question the sweep asks can be
    /// asked now, of one object, and answered before the flaky uplink has
    /// stranded a hundred more.
    ///
    /// It asks rather than assumes, for the reason the write path never rolls
    /// anything back: a `PUT` that reported failure may have committed anyway,
    /// and deleting the sidecar of such a video would leave a bare `.ts` that
    /// the next scan reads as a plain movement — the wrong retention class,
    /// which is the fault the sidecar-first order exists to prevent. Only a
    /// *confirmed* absence deletes; a probe that cannot find out leaves
    /// everything for the next startup, which is where this used to be left in
    /// every case.
    ///
    /// The residual race is the sweep's, narrowed the same way and no further:
    /// between the probe answering "absent" and the `DELETE` arriving, a `PUT`
    /// still in flight — this call's own second attempt, abandoned by a client
    /// timeout — can commit. Its cost is the same one video with no type record.
    async fn collect_orphaned_metadata(&self, camera_id: &str, stem: &str) {
        // Once shutdown is up, no further request is issued: the drain is
        // waiting, and the next startup's sweep collects this anyway.
        if self.stop.stopped() {
            return;
        }
        match self
            .http
            .probe_exists(&format!("{camera_id}/{stem}.ts"))
            .await
        {
            Ok(false) => {
                if self.stop.stopped() {
                    return;
                }
                match self.http.delete(&format!("{camera_id}/{stem}.json")).await {
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

    /// Re-read the sidecars of events whose type an earlier scan could not
    /// establish, and index what they say.
    ///
    /// A scan re-reads every sidecar, but a scan only runs at startup and, for
    /// as long as none of those has succeeded, once per retention tick — so on
    /// a store whose index was built normally a hold would last until a restart
    /// however quickly the store recovered. This is the retry, and it costs one
    /// GET per held event: a store with nothing held issues none.
    async fn resolve_unknown_types(&self, camera_id: &str, cancel: &std::sync::atomic::AtomicBool) {
        let held: Vec<EventKey> = match self.unknown_type.get(camera_id) {
            Some(lock) => lock.read_recover().iter().copied().collect(),
            None => return,
        };
        // Holds on events that have left the index are dropped without asking
        // the store anything. This is free and it is most of what a long hold
        // list is after a few sweeps, so it happens before the budget rather
        // than inside it — otherwise a pass cut short would leave stale markers
        // standing and hand the *next* pass the same list to walk again.
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
        // Without the rotation the hold list has whatever order the `HashSet`
        // iterates in — stable for a given set — so a prefix that never
        // resolves is re-read every tick and the tail behind it is never
        // reached at all: the events furthest from being typed would be exactly
        // the ones nothing ever asks about again. Sorting first is what makes
        // "where the last pass stopped" mean something, and the cursor is this
        // camera's own for the reason [`Self::resolve_cursor`] gives.
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
        // Checking it after a result arrives bounds nothing: the read at the
        // head of the fan-out is two attempts at a request timeout each, which
        // on a store that has stopped answering is minutes against a ten-second
        // budget, spent before the check is ever reached.
        while let Ok(Some((key, read))) = tokio::time::timeout_at(deadline, reads.next()).await {
            read_back += 1;
            // `Some(sidecar)` — the outer one — is a type that is now settled;
            // the inner `None` is the settled answer "there is no sidecar", the
            // plain movement event. Unreadable or typeless settles nothing and
            // keeps the hold.
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

    /// Rebuild the index from the store, retrying a host that is not answering
    /// yet.
    ///
    /// One failed listing used to be the whole story: the index stayed empty,
    /// an empty index is exactly what an empty archive looks like, and so
    /// retention found nothing to prune, the budget saw no bytes used and the
    /// orphan sweep never ran — for the life of the process, cleared only by a
    /// restart. The listing is a single request made at the one moment camon is
    /// most likely to find the network still coming up, which is what these
    /// attempts are for.
    ///
    /// Failing all of them is not fatal and does not stop startup: cameras
    /// still record and still upload. What it costs is retention, and the
    /// caller says so.
    ///
    /// Two things end the series short of success: for a startup scan
    /// [`SCAN_LISTING_BUDGET`], which is what keeps a wedged host from holding
    /// the cameras off the air, and for either kind `stop` — the shutdown flag
    /// as the caller has it. Shutdown is checked before each attempt, during
    /// each wait, and inside the indexing pass, so the drain waits for at most
    /// the request already in flight rather than for the schedule or for an
    /// archive's worth of round trips. What it does not do is interrupt a
    /// request mid-flight: nothing here deletes, so an abandoned scan costs
    /// requests and no consistency.
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
            // What is left of the budget is what this attempt's listing gets,
            // so the series cannot outlast it by a whole request timeout. Never
            // more than one request is worth either way: a listing is a large
            // response and [`REQUEST_TIMEOUT`] is what one is allowed.
            let listing_timeout = left().unwrap_or(REQUEST_TIMEOUT).min(REQUEST_TIMEOUT);
            match self.scan_once(kind, &stop, listing_timeout).await {
                Ok(ScanPass::Complete) => return Ok(()),
                // Shutdown, part-way through indexing. Whatever was inserted
                // stays — it came from the store and is true — but the index is
                // not marked as describing it, so nothing prunes on a half of
                // one and the next start scans again from scratch.
                Ok(ScanPass::Interrupted) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "shutdown",
                    ))
                }
                Err(e) => {
                    // Decided before the wait rather than after it: a budget
                    // with no room left for both the wait and the request it
                    // leads to is spent, and issuing a listing on the remainder
                    // only replaces this error with a timeout.
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

    /// Raise an entry the heal yielded to from movement to object, when the
    /// sidecar on the store says object and the index still says movement.
    ///
    /// This repairs a state only the store can report: an upgrade whose sidecar
    /// `PUT` reported failure and committed anyway. `upgrade_event` gives up
    /// before its own index update when the `PUT` errs — it cannot know the
    /// origin took the bytes — so the store says object while RAM says
    /// movement, and nothing in the process ever revisits it. Left alone, that
    /// event expires on movement retention twelve days early, and budget
    /// eviction, which evicts movements before objects, reaches for it first.
    ///
    /// Safe in both directions of the race [`insert_absent`] exists for, and
    /// for the same reason: the type only ever moves one way. Object is
    /// terminal — no same-identity re-write exists (each `FinishedEvent` is
    /// written once, and the one write retry is on an outcome this backend
    /// never returns) and `resolve_unknown_types` can never hold an object
    /// entry (the marker is set only where the entry was just made a movement,
    /// and every path that types the entry clears it). Anything that ever
    /// re-writes an already-written stem as a movement breaks this and hands
    /// the guard a stale classification to re-apply. So a sidecar reading
    /// object is authoritative whichever process wrote it, while a sidecar
    /// reading movement says nothing about an entry that may have been upgraded
    /// since the read. The guard runs inside the index's write lock, so a live
    /// upgrade is wholly before this or wholly after it: an entry already
    /// upgraded in RAM keeps its own detections, which are the fresher ones.
    ///
    /// Only the type and what travels with it — the sidecar's own size
    /// included, since the sidecar being read *is* the one the failed upgrade
    /// wrote. The video's size and the filmstrip count are left exactly as
    /// [`insert_absent`] left them: a sidecar rewrite does not touch the `.ts`
    /// or its thumbnails, so the listing has nothing newer to say about
    /// either.
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

    /// One pass: list the bucket, index every `.ts` belonging to a camera this
    /// process owns, and — at startup only, see [`ScanKind`] — collect metadata
    /// whose video never landed.
    ///
    /// The listing failure is returned rather than logged and dropped. It is
    /// the one error here that changes what the index *means*: everything below
    /// it degrades one event (a sidecar that will not read leaves one type
    /// unknown), while a listing that does not arrive leaves every event in the
    /// archive unaccounted for, and only the caller can decide how long to keep
    /// asking.
    ///
    /// `listing_timeout` is what the caller's schedule leaves this attempt (see
    /// [`SCAN_LISTING_BUDGET`]); `stop` ends the pass between sidecar reads,
    /// which on a large archive is the difference between a drain of seconds
    /// and one of minutes.
    async fn scan_once(
        &self,
        kind: ScanKind,
        stop: &impl Fn() -> bool,
        listing_timeout: Duration,
    ) -> Result<ScanPass, reqwest::Error> {
        let start = std::time::Instant::now();
        let items = self.http.list(listing_timeout).await?;

        // Every object's name and size, so filmstrip frames can be counted —
        // and every event's full cost priced — without a single extra request.
        // The listing is the only account there is of what an event's metadata
        // weighs on the store, and a budget that skipped it would be measured
        // against a figure the store never agreed to.
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
            let mut filmstrip_frames = 0usize;
            let sidecar_bytes = sizes
                .get(format!("{camera_id}/{stem}.json").as_str())
                .copied()
                .unwrap_or(0);
            let mut thumbnail_bytes = 0u64;
            while let Some(frame) =
                sizes.get(format!("{camera_id}/{stem}_thumb_{filmstrip_frames}.jpg").as_str())
            {
                thumbnail_bytes += frame;
                filmstrip_frames += 1;
            }
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

        // `buffered`, not `buffer_unordered`: the reads overlap, but their
        // results are handed back in listing order, so the index is built by
        // exactly the sequence of insertions a serial scan made and the
        // warnings below keep a stable order. Every insertion this pass makes
        // happens here, on one task — the fan-out covers the request, not the
        // index. Under [`ScanKind::Heal`] it is not the only writer, though:
        // see the insert below.
        let mut reads = futures_util::stream::iter(pending)
            .map(|event| self.read_sidecar_for(event))
            .buffered(SCAN_CONCURRENCY);

        // Events this pass put in the index, events it found the live write
        // path had already indexed better than a listing can, and — of those —
        // the ones the store could still tell something about (heal only).
        let mut indexed = 0usize;
        let mut yielded = 0usize;
        let mut joined = 0usize;

        while let Some((event, read)) = reads.next().await {
            // One archive's worth of round trips is a long time to hold a
            // shutdown drain that is measured in one event's deletes, and an
            // index nobody is going to use is not worth finishing.
            if stop() {
                tracing::info!(
                    indexed,
                    of = total,
                    "stathost warm index scan stopped by shutdown; the index is not marked \
                     as describing the store and the next start scans again"
                );
                return Ok(ScanPass::Interrupted);
            }
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
            // A heal runs while the write path is live, and what the index
            // already holds under this identity came from that path: an event
            // this process uploaded, with a type an upgrade may have settled
            // since, a size a rewrite may have changed, and thumbnails that may
            // have landed after the listing was taken. All of that is fresher
            // than a listing taken seconds ago and a sidecar read after it, so
            // the rebuild yields to it — it is here for the events the index
            // does *not* have. Overwriting instead put a stale movement entry
            // over an upgraded one, which expires real object footage twelve
            // days early with no unknown-type marker to repair it by, and
            // reset filmstrip counts so the extra frames leaked on delete.
            // The startup scan has no such rival — it is awaited before the
            // first camera is spawned — and must overwrite, because the index
            // it inherits from `write_event`'s own bookkeeping is nothing.
            //
            // Yielding leaves the rest of what the listing said about an event
            // this process wrote unreconciled, deliberately: a size or a
            // filmstrip count the index and the store disagree on is a
            // disagreement the store cannot win, since the entry is newer than
            // the listing. So is an event in neither the listing nor the index
            // — a `.ts` whose upload reported failure and committed after the
            // snapshot — which no pass of this scan can see at all. Both are
            // the next startup's to settle, where the index starts empty and
            // the listing is the only account there is.
            let key = (event.start_pts_ns, event.duration_ms);
            let landed = match kind {
                ScanKind::Startup => {
                    self.events.insert(&event.camera_id, entry);
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

        // Every event the listing named has been accounted for, so the index
        // now describes the archive rather than merely this session's writes —
        // the whole of what retention was waiting for. Only reached by a pass
        // that ran to the end: an interrupted one returns above, before this.
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

    /// Claim room for an event about to be written, and hold it until the write
    /// is over.
    ///
    /// The guard used to be the whole of it: read the total, evict down to the
    /// cap, then write. That answers a question about the past — the event
    /// being guarded for is not in the total, so every write overshoots by its
    /// own size — and it answers it independently on every camera's writer
    /// task, so N cameras writing at once each read a total that predates the
    /// other N-1 and the store lands N events over the cap. Reserving first
    /// puts the incoming event *and* every other write in flight into the
    /// figure the eviction is measured against, which is the only version of
    /// the question that has a right answer.
    ///
    /// The reservation is released when the returned guard is dropped, which the
    /// caller does as soon as the bytes are indexed and counted for real — not
    /// at the end of the write, or the same bytes would be in both totals for
    /// as long as the trailing cleanup takes.
    ///
    /// **What it does is not undone if the write then fails.** Eviction issues
    /// real `DELETE`s against footage that is really there, and there is no
    /// putting it back if the upload this made room for turns out never to
    /// land. That is the same bargain local disk makes — its emergency prune
    /// deletes before the retry that may still fail — and it is why the pass is
    /// refused outright once shutdown is up, where the write is known in
    /// advance not to be going anywhere.
    ///
    /// **Room is made, not guaranteed.** If eviction cannot get under the cap —
    /// an event larger than the whole budget, a store refusing `DELETE`s,
    /// nothing left to evict — the write goes ahead anyway and says so. The
    /// alternative is refusing to record, and a client-side cap is not worth
    /// that: `max_stored_bytes` is a number an operator picked, while the
    /// footage is the thing camon exists to keep. Refusing would also fail in
    /// exactly the situations that need recording most — a store that has
    /// stopped accepting `DELETE`s would stop accepting footage too, for as
    /// long as it lasted, on a cap that is nowhere near the store's real limit.
    /// So the overshoot is bounded by what eviction could not reclaim, it is
    /// reported, and every later write tries again.
    async fn make_room(&self, camera_id: &str, cost: u64) -> Option<Reservation<'_>> {
        // Nothing to reserve against, three ways.
        //
        // An unlimited budget refuses no write. Shutdown means this write is
        // about to be abandoned before it sends a byte, so making room for it
        // would delete stored footage for nothing — the same early-out
        // [`Self::enforce_budget`] takes one call deeper, repeated here only so
        // that the reservation is skipped too. Neither is what makes the
        // guarantee hold; see that method for what does.
        //
        // And an un-scanned index is a total that describes this session's
        // writes rather than the store, which is C2's rule: the two paths that
        // delete on that total refuse to. A write before the first successful
        // scan therefore reserves nothing and evicts nothing — a deliberate
        // residual, and the honest one, because a write that cannot know what
        // is stored cannot know what reserving would mean either. It is the
        // same window in which uploads run unbounded, which the guard ahead of
        // this write counts and reports, once, on [`Streak`]'s schedule.
        if self.budget.unlimited() || self.stop.stopped() || self.scanned_events().is_none() {
            return None;
        }
        let held = self.budget.reserve(cost);
        self.enforce_budget(camera_id).await;
        // Verified rather than assumed: eviction stops on the first refused
        // delete, and it can run out of candidates long before it runs out of
        // overshoot.
        //
        // This is not the only way the store goes over — see
        // [`Self::report_overshoot`], which the other one reports through too.
        self.report_overshoot(
            camera_id,
            "eviction could not free enough for this event; an event larger than the whole \
             budget, or a store refusing deletes, keeps it there",
            cost,
        );
        Some(held)
    }

    /// Say that the store is above its cap, on [`Streak`]'s widening schedule.
    ///
    /// Shared by the two paths that can put it there, because they are one
    /// fault to an operator — the store is over, and camon carried on — and
    /// reporting them on separate schedules would let a store that alternates
    /// between them stay quiet. The `reason` is what differs.
    ///
    /// The second path is the reason this is a method rather than a block
    /// inside [`Self::make_room`]. An object upgrade rewrites the sidecar with
    /// the detections it adds, so it grows the store, and it deliberately does
    /// not evict for that growth — tens of bytes are not worth an event's
    /// footage. The argument that it therefore needs no report of its own was
    /// wrong: it assumed a write always follows to absorb it, and one need not.
    /// A camera that has stopped producing events leaves the serial detection
    /// worker draining a queue of jobs, and every upgrade those produce lands
    /// *after* that camera's last ordinary write. Several sidecars can grow
    /// with no [`Self::make_room`] ever running again, and a crossed budget
    /// would sit unreported until recording resumed or retention came round.
    /// "An upgrade only ever targets an event this process wrote" proves a
    /// write *before* it, which is not the one that would have noticed.
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

    /// Enforce the client-side storage budget: while tracked usage exceeds
    /// `max_stored_bytes`, delete the oldest events cheapest-tier-first. No-op
    /// when the budget is unlimited.
    ///
    /// This is the local low-space guard's counterpart, sharing its skeleton and
    /// differing only in [`EvictionPolicy`] — which is where the argument for
    /// stopping on a failure and demoting rather than excluding is written down.
    ///
    /// Both of its answers need an index that has seen the store. "Under
    /// budget" measured against this session's writes alone is a fiction that
    /// grows with the archive, and evicting on it would delete the newest
    /// footage — the only footage the index knows — while everything older sits
    /// there uncounted. So an un-scanned index refuses instead, and does not
    /// try to heal itself: this runs ahead of every write, on the camera's own
    /// writer task, and a retry schedule there would stall recording. The
    /// retention tick does the healing.
    ///
    /// Called two ways: as the pre-write guard the writer asks for
    /// ([`WarmStorageBackend::guard_free_space`]), and from [`Self::make_room`]
    /// with the incoming event already reserved.
    async fn enforce_budget(&self, camera_id: &str) {
        // Nothing to enforce, and so nothing to refuse: an unlimited budget
        // never evicts on any index.
        if self.budget.unlimited() {
            return;
        }
        // Nothing to enforce *for*, either, once shutdown has been asked for.
        // Eviction is real `DELETE`s against real stored footage, and the write
        // this guard runs ahead of is one the same shutdown is about to abandon
        // before it sends a byte. Deleting an archive's oldest events to make
        // room for an event that will never be written is the one thing here
        // that cannot be undone afterwards — and it would spend the drain's
        // budget doing it. Silent: the next start says everything there is to
        // say about the budget.
        //
        // An early-out, not the guarantee. What actually bounds the post-stop
        // cost is [`Self::delete_event_objects`], which reads the flag before
        // every single request and is the check no path can go round; this one
        // and [`Self::make_room`]'s save a candidate walk that would issue
        // nothing anyway. They are here because "we do not delete footage
        // during a shutdown" is a decision, and a decision belongs where it is
        // taken rather than only where it is enforced.
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
        // Checked, not cast: the index entry — and so [`event_key`], the
        // identity every object of this event is keyed by — holds a `u32`,
        // while the duration is computed as `u64`. A silent truncation would
        // put the video under a stem no index entry names.
        //
        // It takes an event of over 49 days for that. No single segment can
        // contribute more than the span bound in
        // `GopSegment::finalize_with_media_pts`, so no one bad measurement
        // reaches it alone — but the sum is bounded only while
        // `max_event_duration_secs` rolls the event, and in event mode 0 is a
        // permitted setting meaning "never chunk, let motion end close it". A
        // camera with 49 days of unbroken motion under that setting is exactly
        // what this conversion is for: the event is dropped, loudly, rather
        // than uploaded under a stem the index cannot name.
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
        let frames: &[Vec<u8>] = match &event.filmstrip_frames {
            Some(frames) => frames,
            None => &[],
        };

        // Everything this event will cost the store, priced before a byte of it
        // is sent: the video, the sidecar that types it, and the filmstrip
        // frames. Charging the video alone is what let the store sit
        // permanently over a cap it believed it was under, and asking after the
        // upload would be asking too late — see [`Self::make_room`].
        let sidecar_bytes = sidecar.len() as u64;
        let frame_bytes: u64 = frames.iter().map(|f| f.len() as u64).sum();
        let room = self
            .make_room(camera_id, file_size + sidecar_bytes + frame_bytes)
            .await;

        // Step 1: the sidecar, before the video. It is the sole carrier of the
        // event type, so an event whose sidecar is missing is not a slightly
        // poorer event — it is the wrong kind of event, expiring on the wrong
        // retention after the next scan. Retried by [`OBJECT_RETRY`], then fail
        // the write before the video is uploaded at all.
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

        // Step 2: the video — retried, then dropped (logged) so a failed write
        // is never lost silently. Nothing is rolled back on the strength of the
        // failure alone: a PUT that reports failure may still have committed
        // server-side, and deleting the sidecar of such a phantom .ts would
        // leave precisely the bare video this order exists to prevent. What the
        // failure does buy is the right to *ask* — see
        // [`Self::collect_orphaned_metadata`].
        let ts_key = format!("{camera_id}/{stem}.ts");
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
        // Non-fatal — the UI hides frames that fail to load. The scan counts
        // frames contiguously from 0, so a gap stops the upload: what is
        // indexed now is what a scan would rebuild later.
        let mut filmstrip_frames = 0usize;
        let mut stored_frame_bytes = 0u64;
        for (i, jpeg) in frames.iter().enumerate() {
            let key = format!("{camera_id}/{stem}_thumb_{i}.jpg");
            // Copied, not shared, because the filmstrip is typed
            // `Arc<Vec<Vec<u8>>>` where the event assembles it. Making
            // it shareable all the way here would mean retyping it at
            // the source for four JPEGs an event.
            if !self.upload(&key, Bytes::from(jpeg.clone())).await {
                tracing::warn!(camera = %camera_id, stem = %stem, frame = i,
                    "failed to upload filmstrip thumbnail to stathost");
                break;
            }
            filmstrip_frames += 1;
            stored_frame_bytes += jpeg.len() as u64;
        }

        // Handing the bytes over from the reservation to the index, in that
        // order and with nothing between: the index takes them first and the
        // claim is released immediately after, so a concurrent `make_room`
        // sees them once. Holding the claim through the trim below — which is
        // network deletes, up to four of them at a request timeout each — would
        // have every other camera's write count this event twice for minutes
        // and evict a victim it did not need to. Releasing *before* the insert
        // would leave the opposite gap, and that one lets a write through.
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
        let sidecar_key = format!("{camera_id}/{stem}.json");
        let sidecar = sidecar_json(
            Some(EventType::Object),
            Some(&upgrade.backend),
            Some(&upgrade.model),
            &upgrade.detections,
            upgrade.continues,
        );
        let sidecar_bytes = sidecar.len() as u64;
        // An upgraded sidecar carries the detections the movement event had
        // none of, so it is bigger than the one it replaces — and it is stored
        // before anything accounts for it. Only the *growth* is reserved, and
        // no eviction is triggered for it: that growth is tens of bytes, and
        // deleting an event's footage to make room for them would be a wildly
        // disproportionate answer. The claim is there so that a concurrent
        // write's eviction is measured against a total that includes it.
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

        // The retention sweep runs on its own task and deletes an event's
        // objects one request at a time, so it can overtake an upgrade between
        // the check above and the `PUT` that has just landed. Local disk cannot
        // reach this state: there the upgrade commits by *renaming* the video,
        // which simply fails once the video is gone. Here the commit is a `PUT`
        // of a sidecar, and a `PUT` always succeeds — so the event that has
        // left the index in the meantime has just had its sidecar written back
        // onto a store that no longer holds its video, and the old code
        // reported that as an upgrade.
        //
        // Doing the index update first and undoing it on a failed `PUT` would
        // not help: a `PUT` that reports failure may still have committed. So
        // the check is here, where the answer is known, and what it costs when
        // it fires is deleting the object this call created.
        //
        // `reidentify` rather than `update`, for the accounting: the sidecar
        // this upgrade wrote is a different length from the one it replaced —
        // it carries detections the movement event had none of — and only the
        // path that re-places an entry may move what it weighs.
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
        // The type is now established. An upgrade only ever targets an event
        // written by this process, so it cannot reach one the scan held — but
        // this and `write_event` are the two places a type becomes a fact, and
        // neither may leave a "type unknown" marker behind it.
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
        // An event whose type the scan could not read is measured against the
        // longest configured retention instead of its placeholder's: no true
        // type can expire later than that, so nothing is ever deleted early,
        // and unlike an indefinite hold it does terminate.
        let unknown_max_age = movement_max_age_ns
            .max(object_max_age_ns)
            .max(continuous_max_age_ns);

        // Deleting one event here is several sequential HTTP requests, each
        // able to sit on a request timeout, so the flag is checked between
        // cameras, between events, and — inside
        // [`Self::delete_event_objects`] — between the requests one event's
        // deletion is made of.
        let stop = || cancel.load(Ordering::Relaxed);
        // What the per-request gates poll. Either signal ends a deletion: the
        // sweep's own `cancel`, which is what the trait promises is honoured at
        // that granularity, and the process shutdown flag, which is the same
        // `AtomicBool` in production and a different one in any test that wants
        // to raise one without the other.
        let stopping = || stop() || self.stop.stopped();

        // This tick is also what heals an index no scan has ever filled: the
        // startup attempts are bounded, so without a retry here a store that
        // came back a minute after boot would stay unpruned until the next
        // restart. It runs before the sweep so a scan that succeeds is pruned
        // in the same tick, and only while un-scanned — once the index
        // describes the store, re-listing it every hour would be one large
        // request an hour for nothing.
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
            // First give held events a chance to be typed, so one that resolves
            // is pruned on its real retention in this same sweep. This is also
            // what drops holds on events a previous sweep deleted: unindexing
            // does not clear them, and nothing between here and there reads one.
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
        // Nothing of this process's is recording yet, so this is the one scan
        // that may also collect orphans — and the one whose failures are paid
        // for in footage nobody records, which is what bounds the series by the
        // clock. Nothing to consult about shutdown this early either: the
        // signal handlers are registered once startup is done, so until this
        // returns a SIGTERM is the default action and there is no drain.
        self.scan_with_retries(ScanKind::Startup, || false).await
    }

    fn recover_orphans(&self) {
        // Interrupted uploads are a server-side concern; nothing to salvage
        // client-side. What an interrupted upload can leave behind is metadata
        // without a video, which [`StathostBackend::sweep_orphaned_metadata`]
        // collects during the scan — it needs the listing the scan already has.
    }

    fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry> {
        self.events.query(camera_id, page)
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
    ///
    /// The one request here whose ceiling is the caller's rather than
    /// [`REQUEST_TIMEOUT`]: the scan retries this call, and what has to be
    /// bounded is the series rather than any one attempt in it. See
    /// [`SCAN_LISTING_BUDGET`].
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
    /// What this event's sidecar and its filmstrip frames weigh, off the
    /// listing — the only account there is of what an event's metadata costs
    /// the store.
    sidecar_bytes: u64,
    thumbnail_bytes: u64,
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

/// Whether a failed stathost request is worth sending again.
///
/// A status is an answer from the origin about *this request*, so anything but
/// the three that mean "not now" is an answer another identical request will
/// get again: a bad or expired token, a path the server will not accept, a
/// bucket that is not there. Retrying those spends a whole request timeout on a
/// task a camera's recording is queued behind, twice, to learn nothing — and on
/// an upload it spends it having sent the event's megabytes up the link a second
/// time.
///
/// No status at all means the response never arrived: a refused connection, a
/// reset mid-body, a client-side timeout. Those are about the moment, and the
/// moment is exactly what a retry is for. Some of them are not — a hostname
/// that does not resolve, a certificate that will never verify — and those are
/// deliberately left in this bucket rather than picked apart from reqwest's
/// error chain: telling them apart is guesswork against another crate's
/// internals, and what misclassifying them costs is one extra attempt at a
/// request that fails in milliseconds. A misconfigured URL is a startup-shaped
/// fault an operator sees in the logs either way.
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
        /// Listings still to be refused with a `500` before the store starts
        /// answering — a host that is not up yet.
        list_failures: Arc<AtomicUsize>,
        /// Latency added to every listing. A value larger than any timeout is a
        /// host that accepted the connection and then said nothing, which is
        /// the failure that costs a whole request timeout rather than a
        /// millisecond.
        list_delay_ms: Arc<AtomicU64>,
        /// Listings asked for, refusals included: how many times the client
        /// came back.
        lists: Arc<AtomicUsize>,
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
        /// Every PUT path served, in arrival order — what was uploaded, in what
        /// order, and how many times each was attempted.
        puts: Arc<Mutex<Vec<String>>>,
        /// Latency added to every PUT: a slow uplink, and the window a test
        /// needs to raise the shutdown flag *inside* an upload.
        put_delay_ms: Arc<AtomicU64>,
        /// Every DELETE path served, in arrival order. What a pass reclaimed —
        /// and, for a pass that should have reclaimed nothing, proof that it
        /// did not.
        deletes: Arc<Mutex<Vec<String>>>,
        /// Latency added to every DELETE, for the tests that need to look at
        /// the backend's accounting from inside a deletion.
        delete_delay_ms: Arc<AtomicU64>,
        /// Requests of this method are held open until the test lets them go.
        ///
        /// This is how a test gets *inside* a request — to raise the shutdown
        /// flag, run a sweep, or read the backend's accounting while an upload
        /// is in flight — without picking a millisecond figure and hoping the
        /// machine is quick enough to beat it. A gate the test opens is an
        /// instrument; a sleep it races is a coin toss on a loaded box.
        hold_puts: Arc<AtomicBool>,
        hold_deletes: Arc<AtomicBool>,
        hold_gets: Arc<AtomicBool>,
    }

    /// A PUT failure injected by path suffix. `stored` decides whether the
    /// object lands anyway before the error is returned — the shape of an
    /// upload timeout or a proxy 5xx over a body the origin already committed,
    /// which a client cannot tell from an upload that never happened.
    #[derive(Clone)]
    struct PutFault {
        suffix: String,
        stored: bool,
        /// What the client is told. A `5xx` is a store having a moment; a `4xx`
        /// is the store refusing this request and every identical one after it.
        status: StatusCode,
    }

    impl Stub {
        fn fail_puts(&self, suffix: &str, stored: bool) {
            self.refuse_puts(suffix, stored, StatusCode::INTERNAL_SERVER_ERROR);
        }

        /// Fail PUTs with a chosen status, so a test can tell a store that is
        /// having a moment from one that is refusing the request itself.
        fn refuse_puts(&self, suffix: &str, stored: bool, status: StatusCode) {
            *self.put_fault.lock().unwrap() = Some(PutFault {
                suffix: suffix.to_string(),
                stored,
                status,
            });
        }

        /// How many times this exact path was uploaded.
        fn put_count(&self, path: &str) -> usize {
            self.puts
                .lock()
                .unwrap()
                .iter()
                .filter(|p| *p == path)
                .count()
        }

        fn fail_gets(&self, suffix: &str) {
            *self.fail_get_suffix.lock().unwrap() = Some(suffix.to_string());
        }

        fn clear_faults(&self) {
            *self.put_fault.lock().unwrap() = None;
            *self.fail_get_suffix.lock().unwrap() = None;
        }

        /// Refuse the next `n` listings. A whole scan's worth of them is the
        /// boot-time race the un-scanned state exists for: the store is not
        /// answering yet, and everything else the client does still works.
        fn fail_next_lists(&self, n: usize) {
            self.list_failures.store(n, Ordering::SeqCst);
        }

        /// The store comes back.
        fn serve_lists_again(&self) {
            self.list_failures.store(0, Ordering::SeqCst);
        }

        /// Answer listings this slowly. With a delay longer than any timeout
        /// the client is willing to wait, this is the host that accepts a
        /// connection and then goes quiet — the failure that costs a whole
        /// request timeout instead of a millisecond.
        fn hang_lists(&self, delay: Duration) {
            self.list_delay_ms
                .store(delay.as_millis() as u64, Ordering::SeqCst);
        }

        fn lists(&self) -> usize {
            self.lists.load(Ordering::SeqCst)
        }

        fn has(&self, path: &str) -> bool {
            self.files.lock().unwrap().contains_key(path)
        }

        /// Every byte the host is holding. The client-side budget is a claim
        /// about exactly this number, so a test that has written its way to a
        /// known store checks the claim against the thing it is about rather
        /// than against arithmetic copied out of the code under test — which
        /// is how the budget came to count videos and nothing else.
        fn stored_bytes(&self) -> u64 {
            self.files
                .lock()
                .unwrap()
                .values()
                .map(|v| v.len() as u64)
                .sum()
        }

        /// Every GET served so far, clearing the record — so a test can count
        /// what one phase asked for without the setup's requests in the total.
        fn take_gets(&self) -> Vec<String> {
            std::mem::take(&mut self.gets.lock().unwrap())
        }

        /// Every PUT served so far, clearing the record.
        fn take_puts(&self) -> Vec<String> {
            std::mem::take(&mut self.puts.lock().unwrap())
        }

        /// Every DELETE served so far, clearing the record.
        fn take_deletes(&self) -> Vec<String> {
            std::mem::take(&mut self.deletes.lock().unwrap())
        }

        /// Hold every request of this kind open — they arrive, are recorded,
        /// and then wait — until [`Stub::release`] lets them through.
        fn hold(&self, gate: &Arc<AtomicBool>) {
            gate.store(true, Ordering::SeqCst);
        }

        fn release(&self, gate: &Arc<AtomicBool>) {
            gate.store(false, Ordering::SeqCst);
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
            stub.lists.fetch_add(1, Ordering::SeqCst);
            let delay = stub.list_delay_ms.load(Ordering::Relaxed);
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            if stub.list_failures.load(Ordering::SeqCst) > 0 {
                stub.list_failures.fetch_sub(1, Ordering::SeqCst);
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
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
                stub.puts.lock().unwrap().push(path.clone());
                wait_on_gate(&stub.hold_puts).await;
                let delay = stub.put_delay_ms.load(Ordering::Relaxed);
                if delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                let fault = stub.put_fault.lock().unwrap().clone();
                if let Some(fault) = fault.filter(|f| path.ends_with(&f.suffix)) {
                    if fault.stored {
                        stub.files.lock().unwrap().insert(path, body.to_vec());
                    }
                    return fault.status.into_response();
                }
                stub.files.lock().unwrap().insert(path, body.to_vec());
                StatusCode::OK.into_response()
            }
            axum::http::Method::DELETE => {
                if !authorized(&headers, &stub.token) {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                stub.deletes.lock().unwrap().push(path.clone());
                wait_on_gate(&stub.hold_deletes).await;
                let delay = stub.delete_delay_ms.load(Ordering::Relaxed);
                if delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
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
                // Read first, then take the latency: a slow response carries
                // what the object was when the request arrived. That is what a
                // real one does, and it is the only way a test can put a write
                // *inside* the window of a read that is already under way.
                let resp = get_response(&stub, &path, &headers);
                wait_on_gate(&stub.hold_gets).await;
                let delay = stub.get_delay_ms.load(Ordering::Relaxed);
                if delay > 0 {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
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
            list_failures: Arc::new(AtomicUsize::new(0)),
            list_delay_ms: Arc::new(AtomicU64::new(0)),
            lists: Arc::new(AtomicUsize::new(0)),
            gets: Arc::new(Mutex::new(Vec::new())),
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak_gets: Arc::new(AtomicUsize::new(0)),
            get_delay_ms: Arc::new(AtomicU64::new(0)),
            puts: Arc::new(Mutex::new(Vec::new())),
            put_delay_ms: Arc::new(AtomicU64::new(0)),
            deletes: Arc::new(Mutex::new(Vec::new())),
            delete_delay_ms: Arc::new(AtomicU64::new(0)),
            hold_puts: Arc::new(AtomicBool::new(false)),
            hold_deletes: Arc::new(AtomicBool::new(false)),
            hold_gets: Arc::new(AtomicBool::new(false)),
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
        backend_stopped_by(url, token, max_stored_bytes, StopFlag::never())
    }

    /// A backend that shares `stop` with whoever raises it — the drain, in
    /// production; a test, here.
    fn backend_stopped_by(
        url: &str,
        token: &str,
        max_stored_bytes: u64,
        stop: StopFlag,
    ) -> StathostBackend {
        let config = StathostConfig {
            url: url.to_string(),
            bucket: "cams".to_string(),
            token: token.to_string(),
            max_stored_bytes,
            enabled: true,
        };
        StathostBackend::new(&config, &["cam".to_string()], stop)
    }

    /// A scanned backend owning more than one camera — the ordinary
    /// installation, and the one a per-backend cursor gets wrong.
    async fn scanned_backend_with_cameras(url: &str, cameras: &[&str]) -> StathostBackend {
        let config = StathostConfig {
            url: url.to_string(),
            bucket: "cams".to_string(),
            token: "secret".to_string(),
            max_stored_bytes: 0,
            enabled: true,
        };
        let ids: Vec<String> = cameras.iter().map(|c| c.to_string()).collect();
        let backend = StathostBackend::new(&config, &ids, StopFlag::never());
        backend.scan().await.unwrap();
        backend
    }

    /// Block while `gate` is raised, so a test can act from inside a request
    /// that has arrived and not yet been answered. Bounded so a test that
    /// forgets to release fails rather than hanging the suite.
    async fn wait_on_gate(gate: &Arc<AtomicBool>) {
        for _ in 0..10_000 {
            if !gate.load(Ordering::SeqCst) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Wait for something the stub is about to be told, polling because what is
    /// being waited for is a real request arriving at a real socket. Panics
    /// rather than hanging the suite if it never happens.
    async fn wait_until(mut condition: impl FnMut() -> bool) {
        for _ in 0..5_000 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("the stub never reached the state this test needs");
    }

    /// A backend whose store already holds `events` and whose budget does not
    /// fit them.
    ///
    /// Seeded through a second, unlimited backend and then scanned in, rather
    /// than written through this one: a write now makes room for itself before
    /// it uploads (see [`StathostBackend::make_room`]), so writing an
    /// over-budget store into existence through the very guard under test would
    /// evict it on the way in. What this builds is the state an operator
    /// actually arrives at — a budget lowered, or a restart onto a store that
    /// has grown past one.
    async fn over_budget_backend(
        url: &str,
        events: &[FinishedEvent],
        max_stored_bytes: u64,
    ) -> StathostBackend {
        let seeder = backend_for(url, "secret", 0);
        for event in events {
            assert_eq!(
                seeder.write_event("cam", event).await,
                WriteOutcome::Written
            );
        }
        let backend = backend_for(url, "secret", max_stored_bytes);
        backend.scan().await.unwrap();
        backend
    }

    /// A backend that has already been through a startup scan, which is the
    /// only state `init_storage` ever hands on: retention and budget eviction
    /// refuse to act on an index no scan has filled, so a test that prunes or
    /// evicts has to start from one. The store is empty at this point, so the
    /// scan costs one listing and indexes nothing.
    async fn scanned_backend_for(url: &str, token: &str, max_stored_bytes: u64) -> StathostBackend {
        let backend = backend_for(url, token, max_stored_bytes);
        backend.scan().await.unwrap();
        backend
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
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .into_iter()
            .find(|e| e.duration_ms == duration_ms)
    }

    fn continuous_event(first_pts: u64, size: usize) -> FinishedEvent {
        let mut e = movement_event(first_pts, size);
        e.is_continuous = true;
        e.filmstrip_frames = None;
        e
    }

    /// What one event will cost the store: its video, the sidecar that types
    /// it, and its filmstrip frames. Derived from the event rather than
    /// hardcoded, because the accounting fix is precisely that the metadata
    /// counts — a budget expressed as a multiple of the video alone would pin
    /// the bug back in place.
    fn cost_of(event: &FinishedEvent) -> u64 {
        let sidecar = sidecar_json(
            Some(event.event_type()),
            event.backend.as_deref(),
            event.model.as_deref(),
            &event.detection_details,
            event.continues,
        )
        .len() as u64;
        let frames: u64 = event
            .filmstrip_frames
            .iter()
            .flat_map(|f| f.iter())
            .map(|f| f.len() as u64)
            .sum();
        event.total_bytes as u64 + sidecar + frames
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
        scanned.scan().await.unwrap();
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

        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            1
        );
        assert_eq!(
            backend
                .find_event("cam", url_key(1_000, 1000))
                .unwrap()
                .file_size,
            25
        );
        // The budget counts what the host actually holds — all four objects,
        // not the video alone.
        assert_eq!(backend.used(), stub.stored_bytes());
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
        scanned.scan().await.unwrap();
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
        let backend = scanned_backend_for(&url, "secret", 0).await;
        backend
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        backend
            .write_event("cam", &longer_movement_event(OLD_PTS, 40))
            .await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            2
        );
        assert_eq!(backend.used(), stub.stored_bytes());

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
            stub.stored_bytes(),
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
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            2
        );

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
        backend.scan().await.unwrap();
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
        scanned.scan().await.unwrap();
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
        let backend = scanned_backend_for(&url, "secret", 0).await;
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
        let backend = scanned_backend_for(&url, "secret", 0).await;
        for i in 0..40u64 {
            backend
                .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
                .await;
        }

        // Every event is expired; a quarter of the 40 indexed may go.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            30
        );

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            22
        );
    }

    /// An event the store refuses to delete sits at the head of the sweep, so
    /// without the cap exempting known failures it would spend the whole budget
    /// on the same objects every hour and never reach the ones behind them.
    #[tokio::test]
    async fn an_undeletable_event_does_not_block_the_sweep_forever() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
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
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            12
        );

        // Second: retrying those is free, so it reaches four behind them.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            8,
            "a stuck head of the queue blocked the whole sweep"
        );
    }

    /// Shutdown reaches this backend as a raised flag, and one event here is
    /// several sequential HTTP deletes: a cancelled sweep must issue none.
    #[tokio::test]
    async fn a_cancelled_prune_deletes_nothing() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
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

    // ---- the un-scanned state ---------------------------------------------

    /// The listing is one request, made at the moment the network is least
    /// likely to be up. When every attempt at it fails the index holds only
    /// what this process has written since, which is indistinguishable from a
    /// store that is nearly empty — and pruning on that reads a full archive as
    /// nothing to do, forever, until someone restarts camon.
    #[tokio::test]
    async fn a_scan_that_never_listed_the_store_refuses_to_prune() {
        let (url, stub) = spawn_stub("secret").await;
        stub.fail_next_lists(usize::MAX);
        let backend = backend_for(&url, "secret", 0);

        assert!(backend.scan().await.is_err(), "an unlisted store scanned");
        assert_eq!(
            stub.lists(),
            SCAN_ATTEMPTS as usize,
            "the startup scan did not make its attempts"
        );

        // Recording is unaffected by any of this — that is the point of not
        // failing startup — so events pile up in an index of this session only.
        backend
            .write_event("cam", &movement_event(1_000_000_000, 30))
            .await;
        let stored = stub.files.lock().unwrap().len();

        // Long expired, and pruned anyway if the empty index is believed.
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        assert!(
            backend
                .find_event("cam", url_key(1_000_000_000, 1000))
                .is_some(),
            "pruned against an index that has never seen the store"
        );
        assert_eq!(stub.files.lock().unwrap().len(), stored, "deleted objects");
        assert_eq!(
            stub.lists(),
            2 * SCAN_ATTEMPTS as usize,
            "the tick did not retry the scan it is the only retry for"
        );
    }

    /// The budget's two answers both need an index that has seen the store:
    /// "under budget" is measured against a sum of what is indexed, and the
    /// eviction that follows deletes what is indexed. Un-scanned, that is this
    /// session's writes — the newest footage there is — while everything older
    /// sits on the host uncounted and unevicted.
    #[tokio::test]
    async fn a_scan_that_never_listed_the_store_refuses_to_enforce_the_budget() {
        let (url, stub) = spawn_stub("secret").await;
        stub.fail_next_lists(usize::MAX);
        // 60 bytes of budget against 120 written: enough to evict twice.
        let backend = backend_for(&url, "secret", 60);
        assert!(backend.scan().await.is_err());

        for pts in [1_000u64, 2_000, 3_000] {
            backend.write_event("cam", &movement_event(pts, 40)).await;
        }
        let stored = stub.files.lock().unwrap().len();
        let listed = stub.lists();

        backend.guard_free_space("cam", 0).await;

        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            3,
            "evicted the only events it could see"
        );
        assert_eq!(stub.files.lock().unwrap().len(), stored);
        assert_eq!(
            stub.lists(),
            listed,
            "the write path waited on a scan; that stalls the camera it guards"
        );
    }

    /// The failure this exists for is a race with the network coming up, so the
    /// attempt that succeeds is usually the second or the third.
    #[tokio::test]
    async fn a_listing_that_comes_back_before_the_attempts_run_out_scans_normally() {
        let (url, stub) = spawn_stub("secret").await;
        seed_events(&stub, 1_000, 3, 1000, "object");
        stub.fail_next_lists(SCAN_ATTEMPTS as usize - 1);

        let backend = backend_for(&url, "secret", 0);
        backend.scan().await.unwrap();

        assert_eq!(stub.lists(), SCAN_ATTEMPTS as usize);
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            3
        );
        // Retention runs on it like any other scanned index.
        backend
            .prune(u64::MAX, 1, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty());
    }

    /// The startup attempts are bounded, so a store that comes back a minute
    /// after boot would otherwise leave retention off until the next restart.
    /// The retention tick retries the scan while — and only while — the index
    /// has never been built.
    #[tokio::test]
    async fn the_retention_tick_heals_an_index_the_startup_scan_never_built() {
        let (url, stub) = spawn_stub("secret").await;
        // Footage from before this process started: only a listing reveals it,
        // and it is long expired.
        seed_events(&stub, 1_000_000_000, 2, 1000, "movement");
        stub.fail_next_lists(usize::MAX);
        let backend = backend_for(&url, "secret", 0);
        assert!(backend.scan().await.is_err());
        assert!(backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty());

        stub.serve_lists_again();

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty()
                && stub.files.lock().unwrap().is_empty(),
            "the tick did not rebuild the index and prune what it found"
        );

        // And having healed, it stops asking: re-listing a whole bucket every
        // hour is the request the scan exists to make once.
        let listed = stub.lists();
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert_eq!(stub.lists(), listed, "kept re-scanning a scanned index");
    }

    /// The state means "never rebuilt", not "the last request failed". A store
    /// that goes away after a good scan must not throw the index back to
    /// refusing: what it learned is still the best account of the archive there
    /// is, and retention is exactly what an unreachable store needs to keep.
    #[tokio::test]
    async fn a_scan_that_succeeded_stays_scanned_when_a_later_one_fails() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        backend
            .write_event("cam", &movement_event(1_000_000_000, 30))
            .await;

        stub.fail_next_lists(usize::MAX);
        assert!(backend.scan().await.is_err());

        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(
            backend
                .find_event("cam", url_key(1_000_000_000, 1000))
                .is_none(),
            "a failed listing un-scanned an index that had been built"
        );
        assert!(stub.files.lock().unwrap().is_empty());
    }

    /// A healing scan runs while cameras record, and an upload in progress has
    /// its sidecar on the host before its video — for as long as the video
    /// takes. The sweep cannot tell that from an orphan, so it does not run
    /// here at all; the next startup, where nothing of this process's can be in
    /// flight, collects what accumulated. See [`ScanKind`].
    #[tokio::test]
    async fn a_healing_rescan_leaves_orphaned_metadata_for_the_next_startup() {
        let (url, stub) = spawn_stub("secret").await;
        // An expired event, which is how this test knows the heal happened at
        // all, and a sidecar whose video is not there.
        seed_events(&stub, 1_000_000_000, 1, 1000, "movement");
        stub.files.lock().unwrap().insert(
            "cam/5000_1000.json".to_string(),
            br#"{"event_type":"object"}"#.to_vec(),
        );
        stub.fail_next_lists(usize::MAX);
        let backend = backend_for(&url, "secret", 0);
        assert!(backend.scan().await.is_err());

        stub.serve_lists_again();
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        assert!(
            !stub.has("cam/1000000000_1000.ts")
                && backend
                    .query("cam", EventPage::unbounded(0, u64::MAX))
                    .is_empty(),
            "the heal did not rebuild the index and prune what it found"
        );
        assert!(
            stub.has("cam/5000_1000.json"),
            "a heal swept metadata a camera may have been uploading"
        );
        // The next startup is the one that may, and does.
        let restarted = backend_for(&url, "secret", 0);
        restarted.scan().await.unwrap();
        assert!(!stub.has("cam/5000_1000.json"), "orphan sidecar kept");
    }

    /// Five attempts is a bound on a host that refuses connections in a
    /// millisecond. A host that accepts and then says nothing costs a whole
    /// request timeout each time, and startup awaits this series before any
    /// camera is spawned: five of those is five minutes of a camera system
    /// recording nothing, which is worse than starting un-scanned.
    #[tokio::test]
    async fn a_listing_that_never_answers_gives_up_on_the_clock_not_the_attempt_count() {
        let (url, stub) = spawn_stub("secret").await;
        // Far longer than the budget, and longer than `REQUEST_TIMEOUT` would
        // allow too: without a deadline of its own the series waits for this.
        stub.hang_lists(Duration::from_secs(30));
        let backend = backend_for(&url, "secret", 0);

        let started = std::time::Instant::now();
        assert!(backend.scan().await.is_err());
        let elapsed = started.elapsed();

        assert!(
            elapsed < SCAN_LISTING_BUDGET * 4,
            "the startup scan held the cameras for {elapsed:?}"
        );
        assert!(
            stub.lists() < SCAN_ATTEMPTS as usize,
            "spent the attempt count on a host that answers nothing"
        );
    }

    /// The retention task is joined by the shutdown drain on a bound of one
    /// event's deletes. A heal on that task must respect the same flag: its
    /// waits between attempts are the drain's waits.
    #[tokio::test]
    async fn a_shutdown_stops_the_scan_from_retrying() {
        let (url, stub) = spawn_stub("secret").await;
        stub.fail_next_lists(usize::MAX);
        // Long enough to raise the flag while the first attempt is in flight.
        stub.hang_lists(Duration::from_millis(30));
        let backend = Arc::new(backend_for(&url, "secret", 0));

        let cancel = Arc::new(AtomicBool::new(false));
        let healing = tokio::spawn({
            let (backend, cancel) = (Arc::clone(&backend), Arc::clone(&cancel));
            async move { backend.prune(1, u64::MAX, u64::MAX, &cancel).await }
        });

        wait_until(|| stub.lists() >= 1).await;
        cancel.store(true, Ordering::SeqCst);
        healing.await.unwrap();

        assert_eq!(
            stub.lists(),
            1,
            "kept retrying the scan after shutdown was asked for"
        );
    }

    /// The wait between attempts is the drain's wait too, and the flag it must
    /// notice is raised while it is already sleeping — so the wait polls rather
    /// than sleeping through the whole delay it was given.
    #[tokio::test]
    async fn a_wait_between_scan_attempts_ends_when_shutdown_arrives_during_it() {
        let flag = Arc::new(AtomicBool::new(false));
        let raiser = {
            let flag = Arc::clone(&flag);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                flag.store(true, Ordering::SeqCst);
            })
        };

        let started = std::time::Instant::now();
        // A backoff far longer than any drain is allowed to take.
        sleep_unless(Duration::from_secs(30), &|| flag.load(Ordering::SeqCst)).await;
        let elapsed = started.elapsed();
        raiser.await.unwrap();

        assert!(
            elapsed < crate::storage::contract::SHUTDOWN_POLL * 4,
            "the wait sat through {elapsed:?} of a shutdown"
        );
        assert!(elapsed >= Duration::from_millis(20), "did not wait at all");
    }

    /// And the pass itself stops, rather than walking an archive's worth of
    /// sidecars inside a drain measured in one event's deletes. What it leaves
    /// is not a rebuilt index: it never reached the end of the listing, so it
    /// knows nothing about what it did not read.
    #[tokio::test]
    async fn a_shutdown_part_way_through_a_scan_leaves_it_unscanned() {
        let (url, stub) = spawn_stub("secret").await;
        seed_events(&stub, 1_000_000_000, 4, 1000, "movement");
        stub.fail_next_lists(usize::MAX);
        let backend = Arc::new(backend_for(&url, "secret", 0));
        assert!(backend.scan().await.is_err());

        stub.serve_lists_again();
        // Every sidecar read is slow enough to raise the flag inside one.
        stub.get_delay_ms.store(100, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        let healing = tokio::spawn({
            let (backend, cancel) = (Arc::clone(&backend), Arc::clone(&cancel));
            async move { backend.prune(1, u64::MAX, u64::MAX, &cancel).await }
        });

        wait_until(|| !stub.gets.lock().unwrap().is_empty()).await;
        cancel.store(true, Ordering::SeqCst);
        healing.await.unwrap();

        // Nothing was pruned on what it did manage to read...
        assert!(stub.has("cam/1000000000_1000.ts"));
        // ...and the next tick still finds an index that must be rebuilt,
        // which it would not if an interrupted pass counted as a rebuild.
        stub.get_delay_ms.store(0, Ordering::Relaxed);
        let listed = stub.lists();
        backend
            .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        assert!(
            stub.lists() > listed,
            "an interrupted pass was taken for a rebuilt index"
        );
        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "the completed heal did not prune what it found"
        );
    }

    /// The one scan that is not the only writer of the index. A heal reads a
    /// sidecar over the network and inserts what it says some round trips
    /// later, and in between the live write path can settle the same event's
    /// type, size or filmstrip — all of which the listing predates. Writing the
    /// listing's version over that would expire object footage on movement
    /// retention twelve days early, with no unknown-type marker to repair it
    /// by, so the heal yields to whatever the index already holds.
    #[tokio::test]
    async fn a_heal_yields_to_an_upgrade_that_landed_while_it_was_reading() {
        const LIVE_PTS: u64 = 5_000_000_000;
        let (url, stub) = spawn_stub("secret").await;
        // Footage from before this process: only a listing reveals it.
        seed_events(&stub, 1_000_000_000, 2, 1000, "movement");
        stub.fail_next_lists(usize::MAX);
        let backend = Arc::new(backend_for(&url, "secret", 0));
        assert!(backend.scan().await.is_err());

        // A live event of this session: uploaded and indexed as a movement,
        // which is what its sidecar on the host says too.
        backend
            .write_event("cam", &movement_event(LIVE_PTS, 40))
            .await;

        stub.serve_lists_again();
        // A sidecar read that takes long enough for a detection to come back
        // while it is in flight.
        stub.get_delay_ms.store(100, Ordering::Relaxed);
        let healing = tokio::spawn({
            let backend = Arc::clone(&backend);
            // Retention long enough that this sweep only heals.
            async move {
                backend
                    .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
                    .await
            }
        });

        let sidecar = format!("cam/{LIVE_PTS}_1000.json");
        wait_until(|| stub.get_count(&sidecar) >= 1).await;
        // The heal is holding a movement sidecar it has already read.
        backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;
        healing.await.unwrap();

        let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
        assert_eq!(
            entry.event_type,
            EventType::Object,
            "the heal wrote a stale movement over an upgraded event"
        );
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        // And it still did what it was for: the archive it could not see.
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            3
        );
    }

    /// The other half of that yield. An upgrade whose sidecar `PUT` reported
    /// failure may have committed anyway — the client cannot tell — and
    /// `upgrade_event` gives up before its index update when it does, leaving
    /// the store saying object and RAM saying movement with nothing in the
    /// process that ever looks again. The event then expires twelve days early,
    /// and budget eviction, which takes movements before objects, reaches it
    /// first. A heal reads that sidecar, so it can put the type right; it is
    /// the only pass that ever will before a restart.
    #[tokio::test]
    async fn a_heal_types_an_upgrade_whose_sidecar_landed_despite_reporting_failure() {
        const LIVE_PTS: u64 = 5_000_000_000;
        let (url, stub) = spawn_stub("secret").await;
        stub.fail_next_lists(usize::MAX);
        let backend = backend_for(&url, "secret", 0);
        assert!(backend.scan().await.is_err());

        backend
            .write_event("cam", &movement_event(LIVE_PTS, 40))
            .await;
        // The upgraded sidecar lands at the origin and the client is told it
        // did not, which is a timeout or a proxy error over a committed body.
        stub.fail_puts(".json", true);
        backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;
        stub.clear_faults();

        let stale = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
        assert_eq!(
            stale.event_type,
            EventType::Movement,
            "not the state to fix"
        );

        stub.serve_lists_again();
        backend
            .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
        assert_eq!(
            entry.event_type,
            EventType::Object,
            "the heal read an object sidecar and left the index on movement"
        );
        assert_eq!(entry.object_classes, vec!["person".to_string()]);
        assert_eq!(entry.detections.len(), 1);
        assert_eq!(entry.backend.as_deref(), Some("ollama"));
        // What the sidecar says nothing newer about is untouched: the video and
        // its thumbnails are the ones the index already had.
        assert_eq!(entry.file_size, 40);
        assert_eq!(entry.filmstrip_frames, 2);
    }

    /// The join only ever moves an entry forward. An event the index already
    /// holds as an object is the *later* account of it — the join exists for an
    /// index that is behind the store, not for one that is ahead of it — so a
    /// sidecar naming different detections leaves them alone. That is what
    /// makes the join safe against a live upgrade landing while the heal reads:
    /// whichever order they fall in, the detections that survive are the ones
    /// the write path put there.
    #[tokio::test]
    async fn a_heal_leaves_the_detections_of_an_entry_it_already_has_as_an_object() {
        const LIVE_PTS: u64 = 6_000_000_000;
        let (url, stub) = spawn_stub("secret").await;
        stub.fail_next_lists(usize::MAX);
        let backend = backend_for(&url, "secret", 0);
        assert!(backend.scan().await.is_err());

        backend
            .write_event("cam", &movement_event(LIVE_PTS, 40))
            .await;
        backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;

        // An object sidecar on the store that says something else — a second
        // writer, or an upgrade this one has already superseded.
        stub.files.lock().unwrap().insert(
            format!("cam/{LIVE_PTS}_1000.json"),
            br#"{"event_type":"object","detections":[{"class":"car","confidence":0.5}]}"#.to_vec(),
        );

        stub.serve_lists_again();
        backend
            .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;

        let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
        assert_eq!(entry.event_type, EventType::Object);
        assert_eq!(
            entry.object_classes,
            vec!["person".to_string()],
            "the heal wrote the store's detections over the write path's"
        );
    }

    #[tokio::test]
    async fn budget_prune_evicts_cheapest_and_oldest_first() {
        let (url, _stub) = spawn_stub("secret").await;
        let continuous = continuous_event(1_000, 40);
        let movement = movement_event(2_000, 40);
        let mut obj = movement_event(3_000, 40);
        obj.has_objects = true;
        obj.object_classes = vec!["person".to_string()];
        let budget = cost_of(&obj) + cost_of(&movement) / 2;

        // The budget above fits the object event and not the movement beside
        // it, priced off what the three events actually cost the store —
        // sidecars and filmstrips included, which is the whole of what the
        // budget is about.
        let backend = over_budget_backend(&url, &[continuous, obj, movement], budget).await;
        assert!(backend.used() > budget, "the store is not over its budget");

        // Enforce the budget (as the pre-write guard would).
        backend.guard_free_space("cam", 0).await;

        // Cheapest tier first: the continuous chunk goes, and the store is
        // still over, so the movement follows. The object is the tier kept
        // longest and survives both.
        assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none()); // continuous evicted
        assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none()); // movement evicted
        assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some()); // object survives
        assert!(backend.used() <= budget);
    }

    /// Budget eviction runs ahead of every write, so an object the store
    /// refuses must not be re-attempted by every pass: it would spend each one
    /// on the same doomed delete and never reach the events that would free
    /// space. Local disk's emergency prune skips its own failures for exactly
    /// this reason.
    #[tokio::test]
    async fn budget_eviction_skips_an_event_it_already_failed_to_delete() {
        let (url, stub) = spawn_stub("secret").await;
        let budget = cost_of(&movement_event(0, 40)) * 3 / 2;
        let backend = over_budget_backend(
            &url,
            &[
                movement_event(1_000, 40),
                movement_event(2_000, 40),
                movement_event(3_000, 40),
            ],
            budget,
        )
        .await;
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert("cam/1000_1000.ts".to_string());

        // First pass: the oldest refuses, and the pass stops there.
        backend.guard_free_space("cam", 0).await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            3
        );
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
        let cost = cost_of(&movement_event(0, 40));
        let backend = over_budget_backend(
            &url,
            &[
                movement_event(1_000, 40),
                movement_event(2_000, 40),
                movement_event(3_000, 40),
            ],
            cost * 3 / 2,
        )
        .await;
        // Someone else removed the oldest video behind camon's back.
        stub.files.lock().unwrap().remove("cam/1000_1000.ts");

        backend.guard_free_space("cam", 0).await;

        assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none());
        assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
        assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some());
        assert_eq!(backend.used(), cost);
    }

    /// An outage flags one candidate per pass (the pass stops at the first
    /// failure). If flagging *excluded* an event from eviction, the store
    /// coming back would leave the budget permanently over its limit: nothing
    /// already written would ever be reconsidered, and the hourly sweep only
    /// retries events that are age-expired. Flagging demotes instead.
    #[tokio::test]
    async fn budget_eviction_recovers_after_an_outage_flagged_every_candidate() {
        let (url, stub) = spawn_stub("secret").await;
        let budget = cost_of(&movement_event(0, 40)) * 3 / 2;
        let backend = over_budget_backend(
            &url,
            &[
                movement_event(1_000, 40),
                movement_event(2_000, 40),
                movement_event(3_000, 40),
                movement_event(4_000, 40),
            ],
            budget,
        )
        .await;
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
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            4
        );
        assert!(backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .iter()
            .all(|e| e.delete_failed));

        // It comes back. Eviction has to reconsider what it flagged, or the
        // budget stays four events over its limit for the life of the process.
        stub.fail_delete_paths.lock().unwrap().clear();
        backend.guard_free_space("cam", 0).await;
        assert!(backend.used() <= budget);
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            1
        );
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
        let backend = scanned_backend_for(&url, "secret", 0).await;
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
        scanned.scan().await.unwrap();
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
        scanned.scan().await.unwrap();
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
        backend.scan().await.unwrap();
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
        seed_events_for(stub, "cam", first_pts, count, duration_ms, event_type);
    }

    fn seed_events_for(
        stub: &Stub,
        camera_id: &str,
        first_pts: u64,
        count: u64,
        duration_ms: u32,
        event_type: &str,
    ) {
        let mut files = stub.files.lock().unwrap();
        for i in 0..count {
            let stem = format!("{}_{duration_ms}", first_pts + i * SEC);
            files.insert(format!("{camera_id}/{stem}.ts"), vec![0u8; 10]);
            files.insert(
                format!("{camera_id}/{stem}.json"),
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
        backend.scan().await.unwrap();
        let elapsed = started.elapsed();

        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            64
        );
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
        backend.scan().await.unwrap();

        let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
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
        // The budget is the sum of the index, so a double insertion shows
        // here — and it is the sum of everything each event costs the host,
        // sidecars included, so a rebuild that priced videos alone would too.
        assert_eq!(backend.used(), stub.stored_bytes());
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
        backend.scan().await.unwrap();

        let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
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
        scanned.scan().await.unwrap();
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
        // Written without its sidecar rather than stripped of one afterwards,
        // so the two entries are priced against the same store.
        stub.fail_puts(".json", false);
        backend
            .write_event("cam", &movement_event(16_000, 30))
            .await;
        let written = backend.find_event("cam", url_key(16_000, 1000)).unwrap();

        stub.clear_faults();
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await.unwrap();

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
        scanned.scan().await.unwrap();
        let e = scanned.find_event("cam", url_key(12_000, 1000)).unwrap();
        assert_eq!(e.event_type, EventType::Object);
        assert_eq!(e.object_classes, vec!["car".to_string()]);
    }

    /// The mirror case: the video genuinely did not land. The orphan sidecar
    /// left behind indexes nothing — the scan walks `.ts` objects only — and so
    /// nothing else would ever delete it either: it is never indexed, never
    /// counted against the budget, and never a sibling of an event.
    ///
    /// It used to wait for the next *startup* to be collected, which on a flaky
    /// uplink is one orphan per failed write for however many weeks the box
    /// stays up. The write that created it collects it instead, at the one
    /// moment there is no ambiguity about whose upload this is.
    #[tokio::test]
    async fn an_orphan_sidecar_is_collected_by_the_write_that_orphaned_it() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".ts", false);

        let outcome = backend.write_event("cam", &object_event(17_000, 30)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(
            !stub.has("cam/17000_1000.json"),
            "the sidecar of a video that never landed was left for a reboot"
        );
        // The probe is what authorises the delete, and it asks about the video.
        assert_eq!(stub.get_count("cam/17000_1000.ts"), 1);
        stub.clear_faults();
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await.unwrap();
        assert!(scanned.find_event("cam", url_key(17_000, 1000)).is_none());
    }

    /// And it asks rather than assumes. A `PUT` that reported failure may have
    /// committed anyway, and deleting *that* video's sidecar would leave the
    /// bare `.ts` the sidecar-first order exists to prevent — read back as a
    /// plain movement on the wrong retention.
    #[tokio::test]
    async fn a_sidecar_is_kept_when_the_video_landed_despite_reporting_failure() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".ts", true); // committed server-side, 500 to the client

        assert_eq!(
            backend.write_event("cam", &object_event(18_500, 30)).await,
            WriteOutcome::Failed
        );

        assert!(
            stub.has("cam/18500_1000.json"),
            "collected the sidecar of a video that is on the host"
        );
        assert!(stub.has("cam/18500_1000.ts"));
    }

    /// A probe that cannot find out is not an absence. Nothing is deleted on a
    /// maybe, and the startup sweep — which asks the same question later, when
    /// nothing of this process's is in flight — remains the backstop.
    #[tokio::test]
    async fn nothing_is_collected_when_the_probe_itself_fails() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.fail_puts(".ts", false);
        stub.fail_gets(".ts");

        assert_eq!(
            backend.write_event("cam", &object_event(18_700, 30)).await,
            WriteOutcome::Failed
        );

        assert!(stub.has("cam/18700_1000.json"));
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
        scanned.scan().await.unwrap();

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
        backend.scan().await.unwrap();

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
        backend.scan().await.unwrap();

        assert!(stub.has("cam/22000_1000.json"), "live sidecar collected");
        assert!(stub.has("cam/22000_1000.ts"));
        // Not indexed — it was not in the listing — but the next start reads it
        // back as the object event its surviving sidecar says it is.
        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await.unwrap();
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
        backend.scan().await.unwrap();

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
        backend.scan().await.unwrap();

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
        scanned.scan().await.unwrap();
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
        scanned.scan().await.unwrap();

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

    /// A scan that succeeded is not run again, so the sweep is the only place a
    /// held event can ever be typed without a restart.
    #[tokio::test]
    async fn a_prune_tick_resolves_a_held_event() {
        let (url, stub) = spawn_stub("secret").await;
        backend_for(&url, "secret", 0)
            .write_event("cam", &object_event(OLD_PTS, 30))
            .await;
        stub.fail_gets(".json");
        let backend = backend_for(&url, "secret", 0);
        backend.scan().await.unwrap();
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
        backend.scan().await.unwrap();
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
        backend.scan().await.unwrap();
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
        let backend = backend_for(&url, "secret", cost_of(&object_event(1_000, 40)));
        backend.scan().await.unwrap();
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

    // ---- the shared storage contract --------------------------------------
    //
    // One assertion body, two backends: see `storage::contract::contract_tests`
    // for why these are written there and called here.

    #[tokio::test]
    async fn contract_a_written_event_reads_back_whole() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        crate::storage::contract::contract_tests::a_written_event_reads_back_whole(&backend).await;
    }

    #[tokio::test]
    async fn contract_an_event_costs_nothing_once_it_is_deleted() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        crate::storage::contract::contract_tests::an_event_costs_nothing_once_it_is_deleted(
            &backend,
            || stub.stored_bytes(),
        )
        .await;
    }

    #[tokio::test]
    async fn contract_a_prune_that_starts_stopped_deletes_nothing() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        crate::storage::contract::contract_tests::a_prune_that_starts_stopped_deletes_nothing(
            &backend,
        )
        .await;
    }

    #[tokio::test]
    async fn contract_a_rewritten_event_replaces_its_entry() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        crate::storage::contract::contract_tests::a_rewritten_event_replaces_its_entry(&backend)
            .await;
    }

    #[tokio::test]
    async fn contract_an_upgrade_reclassifies_the_one_indexed_event() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        crate::storage::contract::contract_tests::an_upgrade_reclassifies_the_one_indexed_event(
            &backend,
        )
        .await;
    }

    #[tokio::test]
    async fn contract_an_upgrade_of_a_deleted_event_indexes_nothing() {
        let (url, _stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        crate::storage::contract::contract_tests::an_upgrade_of_a_deleted_event_indexes_nothing(
            &backend,
        )
        .await;
    }

    // ---- cancellation: a write inside the shutdown drain -------------------

    /// One event write is up to eight sequential uploads — a sidecar and a
    /// video, each with a retry, then four filmstrip frames — and each may sit
    /// on its whole [`UPLOAD_TIMEOUT`]. Nothing used to interrupt that, so a
    /// single event could hold a camera's writer for forty minutes: past the
    /// drain's phase-3 budget, which is sized for *one* upload timeout and
    /// nothing more (`crate::shutdown`).
    ///
    /// The rule that makes the arithmetic true again is that no further request
    /// is issued once the flag is up. Here the flag rises while the first
    /// upload is in flight, and that upload is the last one there is.
    ///
    /// What is asserted is the request *count*, not the clock. Post-stop time is
    /// (requests issued) x (the per-request timeout), so "one request" is a
    /// stricter statement about the drain's budget than any millisecond figure
    /// could be — and it is the same statement on a loaded box, where a
    /// wall-clock bound only measures the box.
    #[tokio::test]
    async fn a_write_issues_no_further_upload_once_shutdown_is_asked_for() {
        let (url, stub) = spawn_stub("secret").await;
        let flag = Arc::new(AtomicBool::new(false));
        let backend = backend_stopped_by(&url, "secret", 0, StopFlag::shared(Arc::clone(&flag)));
        // The first upload is held open until the flag is up, so the flag
        // really does rise *inside* it however busy the machine is.
        stub.hold(&stub.hold_puts);

        let event = movement_event(1_000, 40);
        let (outcome, ()) = tokio::join!(backend.write_event("cam", &event), async {
            wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
            flag.store(true, Ordering::SeqCst);
            stub.release(&stub.hold_puts);
        });

        assert_eq!(outcome, WriteOutcome::Failed);
        assert_eq!(
            stub.puts.lock().unwrap().len(),
            1,
            "requests were issued after shutdown was asked for: {:?}",
            stub.puts.lock().unwrap()
        );
    }

    /// And a write that starts stopped issues nothing at all — the drain is
    /// waiting on this task, and an upload begun now is one it has to sit out.
    #[tokio::test]
    async fn a_write_that_starts_stopped_uploads_nothing() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_stopped_by(
            &url,
            "secret",
            0,
            StopFlag::shared(Arc::new(AtomicBool::new(true))),
        );

        let outcome = backend.write_event("cam", &movement_event(1_000, 40)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert!(stub.puts.lock().unwrap().is_empty());
        assert!(backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty());
    }

    // ---- accounting: the whole cost, and room secured in advance -----------

    /// The budget used to count `.ts` bytes and nothing else, so a store with a
    /// sidecar and four filmstrip frames per event was permanently over a cap
    /// it believed it was under — and every extra byte was one no eviction
    /// would ever reclaim, because eviction is measured against the same
    /// figure.
    #[tokio::test]
    async fn an_events_whole_cost_is_charged_and_not_just_its_video() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        let event = movement_event(1_000, 40);

        backend.write_event("cam", &event).await;

        let entry = backend.find_event("cam", url_key(1_000, 1000)).unwrap();
        assert_eq!(entry.file_size, 40, "the video's own size still means that");
        assert!(entry.sidecar_bytes > 0 && entry.thumbnail_bytes > 0);
        assert_eq!(backend.used(), cost_of(&event));
        assert_eq!(
            backend.used(),
            stub.stored_bytes(),
            "the client-side budget and the host disagree about what is stored"
        );
    }

    /// A rebuild has to price events the same way, off the listing — otherwise
    /// a restart resets the budget to the sum of the videos and the store goes
    /// over again until the next write.
    #[tokio::test]
    async fn a_rebuild_prices_events_the_way_the_write_path_did() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        for pts in [1_000u64, 2_000] {
            backend.write_event("cam", &movement_event(pts, 40)).await;
        }

        let scanned = backend_for(&url, "secret", 0);
        scanned.scan().await.unwrap();

        assert_eq!(scanned.used(), backend.used());
        assert_eq!(scanned.used(), stub.stored_bytes());
    }

    /// Two cameras write at once, which is the normal case on any installation
    /// with more than one. The guard used to read the byte total on each
    /// writer's own task, before either event was in it, so both saw room and
    /// both wrote: the store landed a whole event over its cap for every write
    /// in flight. A reservation puts both of them into the figure the eviction
    /// is measured against.
    #[tokio::test]
    async fn two_writes_in_flight_cannot_both_walk_through_the_budget() {
        let (url, stub) = spawn_stub("secret").await;
        let cost = cost_of(&movement_event(0, 40));
        // Room for three events, two of them already stored.
        let backend = over_budget_backend(
            &url,
            &[movement_event(1_000, 40), movement_event(2_000, 40)],
            cost * 3,
        )
        .await;
        assert_eq!(backend.used(), cost * 2);
        // Enough of a window that the second write is polled while the first is
        // still uploading. No gate is needed: both writes claim their room
        // before either awaits anything, so the overlap this is about is
        // settled by the time the first byte is sent.
        stub.put_delay_ms.store(20, Ordering::SeqCst);

        let (third, fourth) = (movement_event(3_000, 40), movement_event(4_000, 40));
        tokio::join!(
            backend.write_event("cam", &third),
            backend.write_event("cam", &fourth),
        );

        assert!(
            backend.used() <= cost * 3,
            "the store is {} bytes over a budget of {}",
            backend.used() - cost * 3,
            cost * 3
        );
        assert_eq!(backend.used(), stub.stored_bytes());
    }

    // ---- retry classification ---------------------------------------------

    /// A store that refuses the request itself — a bad token, a path it will
    /// not take — answers the second attempt exactly as it answered the first.
    /// Sending it is an [`UPLOAD_TIMEOUT`]'s worth of a camera's writer, and on
    /// a video it is the event's megabytes up the link again, to learn nothing.
    #[tokio::test]
    async fn a_refused_upload_is_not_sent_a_second_time() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.refuse_puts(".json", false, StatusCode::FORBIDDEN);

        let outcome = backend.write_event("cam", &object_event(1_000, 40)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert_eq!(stub.put_count("cam/1000_1000.json"), 1);
    }

    /// A store having a moment gets its second attempt — that is what the
    /// allowance is for, and the wait between the two is [`OBJECT_RETRY`]'s
    /// (pinned in `storage::contract`).
    #[tokio::test]
    async fn a_store_having_a_moment_gets_its_second_attempt() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = backend_for(&url, "secret", 0);
        stub.refuse_puts(".json", false, StatusCode::SERVICE_UNAVAILABLE);

        let outcome = backend.write_event("cam", &object_event(1_000, 40)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert_eq!(stub.put_count("cam/1000_1000.json"), 2);
    }

    // ---- an upgrade overtaken by the sweep --------------------------------

    /// Retention runs on its own task, so it can overtake an upgrade between
    /// the check that the event is indexed and the sidecar `PUT` that lands.
    /// A `PUT` always succeeds, so what used to happen is that the sidecar of a
    /// deleted event was written back onto the store — an orphan nothing but a
    /// reboot collects — and the upgrade logged success for footage that was
    /// gone. Local disk cannot reach this: its upgrade commits by renaming the
    /// video, which simply fails once the video is not there.
    #[tokio::test]
    async fn an_upgrade_overtaken_by_a_sweep_leaves_no_sidecar_behind() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        backend
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        // The upgrade's sidecar upload is held open while the sweep runs inside
        // it — the interleaving, rather than a store that is simply missing the
        // event when the upgrade starts.
        stub.take_puts();
        stub.hold(&stub.hold_puts);

        let upgrade = upgrade_for(OLD_PTS);
        tokio::join!(backend.upgrade_event("cam", &upgrade), async {
            wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
            backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
            stub.release(&stub.hold_puts);
        });

        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "the upgrade re-indexed an event the sweep had deleted"
        );
        assert!(
            !stub.has(&format!("cam/{OLD_PTS}_1000.json")),
            "the upgrade left the sidecar of a deleted event on the store"
        );
        // And it worked this out without asking the store. That is the whole
        // value of the index check over the probe that backs it up: in the
        // ordering a sweep usually leaves behind, the answer is already in RAM,
        // and a request here is one more thing for the drain to wait on.
        assert_eq!(
            stub.get_count(&format!("cam/{OLD_PTS}_1000.ts")),
            0,
            "the upgrade probed the store for something the index already knew"
        );
    }

    /// The scenario the whole cancellation guarantee turns on. A capped store
    /// sits *at* its cap in the steady state — that is what reserve-then-evict
    /// makes it do — and the drain keeps handing the warm writer events after
    /// the flag goes up. So the first post-flag write finds itself over budget,
    /// and without a gate it would evict real stored footage, with real
    /// `DELETE`s, to make room for an event that the very next check is about
    /// to abandon unsent: footage destroyed for a recording that never
    /// happened, and up to five request timeouts of the drain's budget spent
    /// destroying it.
    #[tokio::test]
    async fn a_stopped_write_over_a_full_store_deletes_nothing() {
        let (url, stub) = spawn_stub("secret").await;
        let cost = cost_of(&movement_event(0, 40));
        let seeder = backend_for(&url, "secret", 0);
        for pts in [1_000u64, 2_000] {
            seeder.write_event("cam", &movement_event(pts, 40)).await;
        }
        let flag = Arc::new(AtomicBool::new(false));
        // Exactly at the cap, which is where a capped store lives.
        let backend = backend_stopped_by(
            &url,
            "secret",
            cost * 2,
            StopFlag::shared(Arc::clone(&flag)),
        );
        backend.scan().await.unwrap();
        assert_eq!(backend.used(), cost * 2);
        stub.take_deletes();
        let stored = stub.stored_bytes();

        flag.store(true, Ordering::SeqCst);
        // What the drain does: the writer keeps draining its queue, guard and
        // all, after the flag is up.
        backend.guard_free_space("cam", 0).await;
        let outcome = backend.write_event("cam", &movement_event(3_000, 40)).await;

        assert_eq!(outcome, WriteOutcome::Failed);
        assert_eq!(
            stub.take_deletes(),
            Vec::<String>::new(),
            "a stopped write evicted stored footage to make room for itself"
        );
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            2
        );
        assert_eq!(stub.stored_bytes(), stored);
    }

    /// The sweep's own deletes are gated request by request, not merely event by
    /// event: one event is up to six of them and each can sit on a request
    /// timeout, so a flag checked only between events leaves six minutes of
    /// post-stop work inside one.
    #[tokio::test]
    async fn a_sweep_stops_between_the_deletes_of_a_single_event() {
        let (url, stub) = spawn_stub("secret").await;
        let flag = Arc::new(AtomicBool::new(false));
        let backend = backend_stopped_by(&url, "secret", 0, StopFlag::shared(Arc::clone(&flag)));
        backend.scan().await.unwrap();
        backend
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        stub.take_deletes();
        // The first delete is held open until the flag is up.
        stub.hold(&stub.hold_deletes);

        let cancel = AtomicBool::new(false);
        tokio::join!(backend.prune(1, 1, 1, &cancel), async {
            wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
            flag.store(true, Ordering::SeqCst);
            stub.release(&stub.hold_deletes);
        });

        assert_eq!(
            stub.take_deletes().len(),
            1,
            "the sweep kept deleting an event's objects after shutdown was asked for"
        );
        // Nothing is flagged as having resisted deletion: the store was fine.
        assert!(
            !backend
                .find_event("cam", url_key(OLD_PTS, 1000))
                .unwrap()
                .delete_failed
        );
    }

    /// The eviction pass is the one the drain waits on with a camera's
    /// recording queued behind it, and the flag can go up while it is already
    /// running — after the guard has decided to evict and between two of the
    /// events it is deleting. Neither of the checks that refuse a pass *before*
    /// it starts can reach that; the skeleton's own `cancel` is what does.
    #[tokio::test]
    async fn an_eviction_already_under_way_stops_when_shutdown_arrives() {
        let (url, stub) = spawn_stub("secret").await;
        let cost = cost_of(&movement_event(0, 40));
        let seeder = backend_for(&url, "secret", 0);
        for pts in [1_000u64, 2_000, 3_000, 4_000] {
            seeder.write_event("cam", &movement_event(pts, 40)).await;
        }
        let flag = Arc::new(AtomicBool::new(false));
        // Room for one: the pass has three events to get through.
        let backend = backend_stopped_by(&url, "secret", cost, StopFlag::shared(Arc::clone(&flag)));
        backend.scan().await.unwrap();
        stub.take_deletes();
        // The first delete is held open until the flag is up.
        stub.hold(&stub.hold_deletes);

        tokio::join!(backend.guard_free_space("cam", 0), async {
            wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
            flag.store(true, Ordering::SeqCst);
            stub.release(&stub.hold_deletes);
        });

        assert_eq!(
            stub.take_deletes().len(),
            1,
            "the eviction kept deleting stored footage after shutdown was asked for"
        );
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            4,
            "an event was unindexed by a pass that never finished deleting it"
        );
    }

    /// The sweep's `cancel` is the flag the trait promises is honoured between
    /// the requests one event's deletion is made of — and it has to be *that*
    /// flag, not the backend's own. Production hands both the same
    /// `AtomicBool`, so a backend polling only its constructor flag would look
    /// correct here for ever and be wrong for any caller with a stop of its
    /// own. This raises `cancel` alone, with the shutdown flag left down, and
    /// the deletion stops all the same.
    #[tokio::test]
    async fn a_sweep_stops_between_deletes_on_the_cancel_it_was_given() {
        let (url, stub) = spawn_stub("secret").await;
        // No shutdown flag at all: the only stop in this test is `cancel`.
        let backend = backend_for(&url, "secret", 0);
        backend.scan().await.unwrap();
        backend
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        stub.take_deletes();
        stub.hold(&stub.hold_deletes);

        let cancel = AtomicBool::new(false);
        tokio::join!(backend.prune(1, 1, 1, &cancel), async {
            wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
            cancel.store(true, Ordering::SeqCst);
            stub.release(&stub.hold_deletes);
        });

        assert_eq!(
            stub.take_deletes().len(),
            1,
            "the sweep kept deleting an event's objects after its own cancel was raised"
        );
        assert!(
            !backend
                .find_event("cam", url_key(OLD_PTS, 1000))
                .unwrap()
                .delete_failed,
            "a cancelled deletion was recorded as one the store refused"
        );
    }

    /// A deletion cut short mid-thumbnail must leave a *prefix*, because a
    /// prefix is the only thing the next scan can see.
    ///
    /// The scan counts an event's frames contiguously from 0 and stops at the
    /// first gap. So deleting frame 0 first and stopping there strands the rest
    /// where nothing can reach them: the scan records zero frames, the orphan
    /// sweep passes over them because the video is still on the store, and the
    /// event's own later deletion only removes as many frames as the entry
    /// records — which is none of them. They would sit there until some restart
    /// that happened to come *after* the video had gone, which on a box that
    /// stays up for months is never.
    ///
    /// This walks the whole scenario rather than just the ordering: interrupt a
    /// sweep after one thumbnail delete, restart onto the store, and then let
    /// retention take the event away. Nothing may be left.
    #[tokio::test]
    async fn a_deletion_cut_short_leaves_thumbnails_a_rebuild_can_still_see() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        let mut event = movement_event(OLD_PTS, 40);
        // Three frames, so "a prefix" is a claim with content: one is deleted
        // and two have to survive together, in order, from index 0.
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        backend.write_event("cam", &event).await;
        let stem = format!("{OLD_PTS}_1000");
        stub.take_deletes();
        stub.hold(&stub.hold_deletes);

        let cancel = AtomicBool::new(false);
        tokio::join!(backend.prune(1, 1, 1, &cancel), async {
            wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
            cancel.store(true, Ordering::SeqCst);
            stub.release(&stub.hold_deletes);
        });

        // Exactly one frame went, and it was the *last* one.
        assert_eq!(
            stub.take_deletes(),
            vec![format!("cam/{stem}_thumb_2.jpg")],
            "the sweep deleted the wrong end of the filmstrip, or did not stop"
        );
        assert!(stub.has(&format!("cam/{stem}_thumb_0.jpg")));
        assert!(stub.has(&format!("cam/{stem}_thumb_1.jpg")));
        assert!(stub.has(&format!("cam/{stem}.ts")));

        // A restart sees the survivors, because they are a prefix.
        let restarted = backend_for(&url, "secret", 0);
        restarted.scan().await.unwrap();
        let entry = restarted.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
        assert_eq!(
            entry.filmstrip_frames, 2,
            "the rebuild cannot see the thumbnails the interrupted sweep left"
        );
        assert_eq!(
            restarted.used(),
            stub.stored_bytes(),
            "the rebuild is not charged for everything the store is holding"
        );

        // And retention finishes the job it was interrupted in the middle of,
        // taking every object with it.
        restarted.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(
            restarted
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "the second sweep did not delete the event"
        );
        assert!(
            stub.files.lock().unwrap().is_empty(),
            "objects survived the event they belong to: {:?}",
            stub.files.lock().unwrap().keys().collect::<Vec<_>>()
        );
    }

    /// The same stranding as the test above, reached without any interruption:
    /// one refused `DELETE` in the middle of an otherwise healthy pass.
    ///
    /// A frame the store refuses is still there afterwards, so carrying on down
    /// past it deletes *around* it and leaves a gap — and a gap is what the scan
    /// cannot see past. Stopping the descent leaves a prefix instead.
    ///
    /// Cancellation is deliberately not what this test uses, because it cannot
    /// tell the two implementations apart: whether the loop breaks on the
    /// refusal or carries on to the next frame, the very next thing either does
    /// is read the stop flag and abandon, so both leave every frame in place. A
    /// video that also refuses keeps the event alive with no timing in the test
    /// at all, and separates them completely.
    #[tokio::test]
    async fn a_refused_filmstrip_frame_keeps_the_ones_below_it_contiguous() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        let mut event = movement_event(OLD_PTS, 40);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        backend.write_event("cam", &event).await;
        let stem = format!("{OLD_PTS}_1000");
        stub.take_deletes();
        {
            let mut refused = stub.fail_delete_paths.lock().unwrap();
            // The top frame will not go...
            refused.insert(format!("cam/{stem}_thumb_2.jpg"));
            // ...and neither will the video, which is what keeps the event
            // indexed for a later pass to find — the state the leak needs.
            refused.insert(format!("cam/{stem}.ts"));
        }

        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

        // A refused frame does not stop the event's deletion: the video was
        // still attempted, and its outcome is the event's.
        let attempted = stub.take_deletes();
        assert!(
            attempted.contains(&format!("cam/{stem}.ts")),
            "a refused thumbnail stopped the event's own deletion: {attempted:?}"
        );
        assert!(
            backend
                .find_event("cam", url_key(OLD_PTS, 1000))
                .unwrap()
                .delete_failed,
            "the video refused and the event was not flagged for a retry"
        );
        for i in 0..3 {
            assert!(
                stub.has(&format!("cam/{stem}_thumb_{i}.jpg")),
                "frame {i} was deleted from under the frame that refused, leaving a gap"
            );
        }

        // So a restart sees all three, and the retry that follows takes them.
        let restarted = backend_for(&url, "secret", 0);
        restarted.scan().await.unwrap();
        assert_eq!(
            restarted
                .find_event("cam", url_key(OLD_PTS, 1000))
                .unwrap()
                .filmstrip_frames,
            3,
            "the rebuild cannot see the frames the refused delete left behind"
        );
        assert_eq!(restarted.used(), stub.stored_bytes());

        stub.fail_delete_paths.lock().unwrap().clear();
        restarted.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(
            stub.files.lock().unwrap().is_empty(),
            "objects survived the event they belong to: {:?}",
            stub.files.lock().unwrap().keys().collect::<Vec<_>>()
        );
    }

    /// And a refused frame is not the *event's* outcome. The video's is — that
    /// rule predates all of this — so an expired recording is not held back
    /// because a JPEG resisted, and nothing is flagged as a refusal on a
    /// thumbnail's account: [`WarmEventEntry::delete_failed`] is what demotes an
    /// event in eviction and what a sweep counts, and neither should turn on
    /// decoration. What is left over becomes an orphan, which is what the
    /// startup sweep is for.
    #[tokio::test]
    async fn a_refused_filmstrip_frame_is_not_the_events_outcome() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        let mut event = movement_event(OLD_PTS, 40);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        backend.write_event("cam", &event).await;
        let stem = format!("{OLD_PTS}_1000");
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert(format!("cam/{stem}_thumb_2.jpg"));

        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "an expired recording was held back because a thumbnail resisted"
        );
        assert!(!stub.has(&format!("cam/{stem}.ts")));
        assert!(!stub.has(&format!("cam/{stem}.json")));

        // The frames left behind are orphans now — no video, no entry — and the
        // startup sweep is the thing that collects those.
        stub.fail_delete_paths.lock().unwrap().clear();
        let restarted = backend_for(&url, "secret", 0);
        restarted.scan().await.unwrap();
        assert!(
            stub.files.lock().unwrap().is_empty(),
            "the startup sweep did not collect the frames a refused delete stranded: {:?}",
            stub.files.lock().unwrap().keys().collect::<Vec<_>>()
        );
    }

    /// The policy when room cannot be made: record anyway, and say so. An event
    /// bigger than the whole budget is the sharpest form of it — refusing would
    /// mean this camera never stores anything again, on a cap that is a number
    /// an operator typed rather than a disk that is actually full.
    #[tokio::test]
    async fn an_event_too_big_for_the_whole_budget_is_still_recorded() {
        let (url, _stub) = spawn_stub("secret").await;
        let budget = cost_of(&movement_event(0, 40)) / 2;
        let backend = scanned_backend_for(&url, "secret", budget).await;

        let outcome = backend.write_event("cam", &movement_event(1_000, 40)).await;

        assert_eq!(outcome, WriteOutcome::Written);
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            1
        );
        assert!(
            backend.used() > budget,
            "the store is under a budget it cannot fit one event into"
        );
        // And the overshoot is visible where the write path steers by it.
        assert_eq!(backend.free_space().unwrap(), 0);
    }

    /// The same when eviction is the thing that cannot free anything: a store
    /// refusing `DELETE`s is a store that will still take footage, and refusing
    /// to give it any would turn one outage into two.
    #[tokio::test]
    async fn a_store_that_refuses_deletes_still_gets_the_footage() {
        let (url, stub) = spawn_stub("secret").await;
        let cost = cost_of(&movement_event(0, 40));
        let backend = over_budget_backend(
            &url,
            &[movement_event(1_000, 40), movement_event(2_000, 40)],
            cost * 2,
        )
        .await;
        for pts in [1_000u64, 2_000] {
            stub.fail_delete_paths
                .lock()
                .unwrap()
                .insert(format!("cam/{pts}_1000.ts"));
        }

        let outcome = backend.write_event("cam", &movement_event(3_000, 40)).await;

        assert_eq!(outcome, WriteOutcome::Written);
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            3
        );
        assert!(backend.used() > cost * 2);
    }

    /// An upgraded sidecar is bigger than the one it replaces — it carries the
    /// detections the movement event had none of — and it is stored before
    /// anything accounts for it. The growth is claimed for the duration, so a
    /// write racing the upgrade evicts against a total that includes it.
    #[tokio::test]
    async fn an_upgrade_claims_the_growth_of_the_sidecar_it_is_writing() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 100_000).await;
        backend.write_event("cam", &movement_event(1_000, 40)).await;
        let before = backend.free_space().unwrap();
        stub.take_puts();
        // Held open so the budget can be read from inside the upload.
        stub.hold(&stub.hold_puts);

        let upgrade = upgrade_for(1_000);
        let (_, during) = tokio::join!(backend.upgrade_event("cam", &upgrade), async {
            wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
            let during = backend.free_space().unwrap();
            stub.release(&stub.hold_puts);
            during
        });

        assert!(
            during < before,
            "the sidecar's growth was uploaded against a budget that had not been told"
        );
        // And the figure does not move again: the claim is handed to the index,
        // never counted on top of it.
        assert_eq!(backend.free_space().unwrap(), during);
    }

    /// The handover from reservation to index has to be complete before the
    /// write's trailing cleanup, which is more network deletes. Holding the
    /// claim through those would have every other camera's write count this
    /// event twice for as long as they take, and evict a victim it did not need
    /// to.
    #[tokio::test]
    async fn a_rewrites_reservation_is_released_before_its_thumbnails_are_trimmed() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 100_000).await;
        let mut event = movement_event(2_000, 30);
        event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
        backend.write_event("cam", &event).await;

        let mut shorter = movement_event(2_000, 30);
        shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
        // Held open so the accounting can be read from inside the trim.
        stub.hold(&stub.hold_deletes);

        let (_, during) = tokio::join!(backend.write_event("cam", &shorter), async {
            wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
            let during = backend.free_space().unwrap();
            stub.release(&stub.hold_deletes);
            during
        });

        assert_eq!(
            during,
            backend.free_space().unwrap(),
            "the write's reservation was still counted while its thumbnails were being trimmed"
        );
    }

    /// The budget bounds the *wait*, not just the gap between results: a store
    /// that has stopped answering makes the read at the head of the fan-out two
    /// request timeouts long, and a deadline checked only after a result
    /// arrives never gets to say anything about it.
    #[tokio::test]
    async fn a_prune_does_not_wait_out_a_held_read_past_its_budget() {
        let (url, stub) = spawn_stub("secret").await;
        seed_events(&stub, OLD_PTS, 8, 1000, "object");
        stub.fail_gets(".json");
        let backend = scanned_backend_for(&url, "secret", 0).await;
        // The reads are held open and never answered — the host that accepted
        // the connection and then said nothing, which is the failure that costs
        // a whole request timeout. A pass whose budget bounds only the gap
        // *between* results never reaches the check at all and sits here for as
        // long as the reads take; a pass that bounds the wait returns while they
        // are still held. What is asserted is which of those happened, not how
        // many milliseconds it took.
        stub.hold(&stub.hold_gets);

        let finished = tokio::time::timeout(
            Duration::from_secs(5),
            backend.prune(1, 1, 1, &AtomicBool::new(false)),
        )
        .await;
        stub.release(&stub.hold_gets);

        assert!(
            finished.is_ok(),
            "the tick was still waiting on held sidecar reads, past a budget of {RESOLVE_BUDGET:?}"
        );
    }

    /// And the next tick starts where the last one stopped. The hold list has
    /// whatever order its `HashSet` iterates in, so a pass that always starts at
    /// the front re-reads the same unresolvable prefix every hour and never
    /// reaches the tail behind it at all — the events furthest from being typed
    /// would be exactly the ones nothing ever asks about again.
    ///
    /// Asserted by membership rather than by counting: how *many* holds a tick
    /// gets through depends on the machine, but which one it starts from does
    /// not. The oldest hold is the head of the sorted list, so a pass that
    /// always starts at the front reads it every single tick — and a pass that
    /// starts where the last one stopped cannot read it again until the window
    /// has been all the way round.
    #[tokio::test]
    async fn consecutive_prune_ticks_reach_different_held_events() {
        let (url, stub) = spawn_stub("secret").await;
        const HELD: u64 = 80;
        seed_events(&stub, OLD_PTS, HELD, 1000, "object");
        stub.fail_gets(".json");
        let backend = scanned_backend_for(&url, "secret", 0).await;
        stub.take_gets();
        // Slow enough that no single tick can get all the way round the hold
        // list and wrap back to its head.
        stub.get_delay_ms.store(25, Ordering::SeqCst);

        // Two ticks with no expiry, so only the re-reads happen.
        let oldest_hold = format!("cam/{OLD_PTS}_1000.json");
        let read_oldest = |stub: &Stub| stub.take_gets().contains(&oldest_hold);
        backend
            .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        let first = read_oldest(&stub);
        backend
            .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        let second = read_oldest(&stub);

        assert!(first, "the first tick did not start at the oldest hold");
        assert!(
            !second,
            "the second tick started at the oldest hold again; every tick re-reads the same \
             prefix and the holds behind it are never reached"
        );
    }

    /// Two cameras, and the reason the cursor cannot be shared between them.
    ///
    /// A shared cursor is advanced by every camera's pass, so each camera's
    /// window moves by the *sum* of what all of them read. Whenever that sum is
    /// a multiple of a camera's hold count, that camera lands on the same window
    /// every tick and the rest of its list is never read again — starvation
    /// reached from the opposite direction to the one the rotation was added
    /// for, and reached under exactly the conditions this pass is normally in.
    ///
    /// Constructed without a clock in it. The sidecar reads are held open and
    /// never answered, so every pass reads nothing back and advances its cursor
    /// by the floor of one; the fan-out issues exactly [`SCAN_CONCURRENCY`]
    /// requests and no more, because no slot is ever freed. With 32 holds per
    /// camera the window is a strict half of the list, so which holds it covers
    /// is an observable fact rather than a matter of timing:
    ///
    /// * per camera, this camera's second tick starts at its own hold 1;
    /// * shared, both cameras have moved it, so the second tick starts at 2 or 3
    ///   depending on which camera the index happened to walk first — and hold 1
    ///   is not read in either case.
    #[tokio::test]
    async fn a_second_cameras_holds_do_not_move_this_ones_window() {
        let (url, stub) = spawn_stub("secret").await;
        const HELD: u64 = 32;
        seed_events_for(&stub, "cam", OLD_PTS, HELD, 1000, "object");
        seed_events_for(&stub, "other", OLD_PTS, HELD, 1000, "object");
        stub.fail_gets(".json");
        let backend = scanned_backend_with_cameras(&url, &["cam", "other"]).await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            HELD as usize
        );
        assert_eq!(
            backend
                .query("other", EventPage::unbounded(0, u64::MAX))
                .len(),
            HELD as usize
        );
        stub.take_gets();
        // Held and never answered: every pass reads nothing back, so every
        // cursor advances by its floor of one and nothing depends on latency.
        stub.hold(&stub.hold_gets);

        let hold_at = |i: u64| format!("cam/{}_1000.json", OLD_PTS + i * SEC);
        let sweep = |stub: &Stub| -> HashSet<String> { stub.take_gets().into_iter().collect() };

        // Two ticks with nothing expired, so only the re-reads happen.
        backend
            .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        let first = sweep(&stub);
        backend
            .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
            .await;
        let second = sweep(&stub);
        stub.release(&stub.hold_gets);

        assert!(
            first.contains(&hold_at(0)),
            "the first tick did not start at this camera's oldest hold"
        );
        // Between them these four pin the second window to exactly `1..=16`,
        // which is the only placement a per-camera cursor that moved by one can
        // produce: hold 1 present rules out a window the other camera pushed
        // past it, hold 0 absent rules out one that never moved at all, hold 16
        // present rules out one narrower than the fan-out, and hold 17 absent
        // rules out one wider.
        assert!(
            second.contains(&hold_at(1)),
            "the second tick skipped this camera's hold 1: the other camera's reads moved \
             this camera's window, which is how a window comes to land on the same place \
             every tick and never read the rest of the list"
        );
        assert!(
            !second.contains(&hold_at(0)),
            "the second tick started where the first did: this camera's window is not moving"
        );
        assert!(
            second.contains(&hold_at(SCAN_CONCURRENCY as u64)),
            "the second tick's window is narrower than the fan-out that issues it, so what \
             it covers per tick is not what the budget and the fan-out say it is"
        );
        assert!(
            !second.contains(&hold_at(SCAN_CONCURRENCY as u64 + 1)),
            "the second tick's window is wider than the fan-out that issued it"
        );
    }

    /// An upgrade grows the sidecar and does not evict for it, so it can take
    /// the store over the cap on its own — and it can be the *last* thing a
    /// camera does, with the detection worker draining its queue after the last
    /// ordinary write, so there is no later `make_room` to notice. It therefore
    /// reports for itself, on the same streak the write path uses.
    #[tokio::test]
    async fn an_upgrade_that_crosses_the_budget_says_so() {
        let (url, stub) = spawn_stub("secret").await;
        let event = movement_event(1_000, 40);
        // Exactly enough for the event as written, and not a byte for the
        // detections the upgrade is about to add to its sidecar.
        let backend = scanned_backend_for(&url, "secret", cost_of(&event)).await;
        backend.write_event("cam", &event).await;
        assert_eq!(
            backend.free_space().unwrap(),
            0,
            "the store is not exactly full"
        );
        assert_eq!(
            backend.budget_overshoots.lock_recover().count(),
            0,
            "the write itself was already over"
        );

        backend.upgrade_event("cam", &upgrade_for(1_000)).await;

        assert!(
            backend.used() > cost_of(&event),
            "the upgrade's sidecar did not grow the store"
        );
        assert_eq!(
            backend.budget_overshoots.lock_recover().count(),
            1,
            "an upgrade took the store over its cap and recorded nothing"
        );

        // And on the *same* streak the write path uses, so a store that
        // alternates between the two sources cannot stay quiet by splitting its
        // occurrences across two schedules. The store refuses to give the one
        // stored event up, so the write cannot evict its way under either.
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert("cam/1000_1000.ts".to_string());
        backend.write_event("cam", &movement_event(2_000, 40)).await;
        assert_eq!(
            backend.budget_overshoots.lock_recover().count(),
            2,
            "the write path counts its overshoots somewhere else"
        );
    }

    /// The hard ordering of the upgrade/sweep race, which the index check alone
    /// cannot see: the sweep has deleted the objects and has *not* reached its
    /// `index.remove` yet, so the reclassification lands on an entry that is
    /// still there and the sidecar is written back onto a store with no video
    /// under it. Constructed by deleting the objects directly while the
    /// upgrade's `PUT` is in flight — which is exactly the state the sweep is
    /// in between its own two steps.
    #[tokio::test]
    async fn an_upgrade_whose_video_vanished_mid_put_takes_itself_back() {
        let (url, stub) = spawn_stub("secret").await;
        let backend = scanned_backend_for(&url, "secret", 0).await;
        backend
            .write_event("cam", &movement_event(OLD_PTS, 40))
            .await;
        stub.take_puts();
        stub.hold(&stub.hold_puts);

        let upgrade = upgrade_for(OLD_PTS);
        tokio::join!(backend.upgrade_event("cam", &upgrade), async {
            wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
            // The sweep's object deletes, with its index removal still to come.
            {
                let mut files = stub.files.lock().unwrap();
                files.remove(&format!("cam/{OLD_PTS}_1000.ts"));
                files.remove(&format!("cam/{OLD_PTS}_1000.json"));
            }
            stub.release(&stub.hold_puts);
        });

        assert!(
            !stub.has(&format!("cam/{OLD_PTS}_1000.json")),
            "the upgrade left a sidecar on a store that has no video for it"
        );
        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "the index still describes footage that is gone"
        );
    }

    // ---- a prune tick that has held types to re-read -----------------------

    /// The re-read of held sidecars runs ahead of every deletion the sweep
    /// exists to make, and it used to be serial and unbounded: an archive's
    /// worth of round trips, hourly, in front of the pass that reclaims the
    /// space. It is fanned out and cut off at [`RESOLVE_BUDGET`] now, and the
    /// sweep behind it runs either way.
    #[tokio::test]
    async fn a_prune_bounds_the_time_it_spends_re_reading_held_types() {
        let (url, stub) = spawn_stub("secret").await;
        const HELD: u64 = 100;
        seed_events(&stub, OLD_PTS, HELD, 1000, "object");
        // Every sidecar is unreadable, so the scan holds every event's type and
        // the prune tick has the whole archive to re-read.
        stub.fail_gets(".json");
        let backend = scanned_backend_for(&url, "secret", 0).await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            HELD as usize
        );
        // Only now: the scan's own reads are not what this is measuring.
        stub.take_gets();
        stub.get_delay_ms.store(25, Ordering::SeqCst);

        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

        let sidecar_reads = stub
            .take_gets()
            .iter()
            .filter(|p| p.ends_with(".json"))
            .count();
        assert!(
            sidecar_reads < HELD as usize,
            "the tick re-read {sidecar_reads} sidecars of {HELD} held events instead of \
             stopping at its budget"
        );
        assert!(
            stub.peak_gets.load(Ordering::SeqCst) > 1,
            "the re-reads were issued one at a time"
        );
        // And the sweep behind them still deleted its share (a quarter of the
        // archive, per `cap_sweep_deletions`).
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            HELD as usize * 3 / 4,
            "the sweep did not get to its deletions"
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
                    sidecar_bytes: 0,
                    thumbnail_bytes: 0,
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
        let hits = backend.query("cam", EventPage::unbounded(50 * SEC, 60 * SEC));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_pts_ns, 0);
    }

    #[test]
    fn query_returns_every_overlapping_event_in_start_order() {
        let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let starts: Vec<u64> = backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .iter()
            .map(|e| e.start_pts_ns)
            .collect();
        assert_eq!(starts, vec![0, 10 * SEC, 20 * SEC]);
        assert!(backend
            .query("unknown", EventPage::unbounded(0, u64::MAX))
            .is_empty());
    }

    #[test]
    fn zero_duration_events_are_found_at_their_start() {
        let backend = indexed(&[(10 * SEC, 0)]);
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(10 * SEC, 10 * SEC))
                .len(),
            1
        );
        assert!(backend
            .query("cam", EventPage::unbounded(10 * SEC + 1, 20 * SEC))
            .is_empty());
    }

    #[test]
    fn query_bounds_include_events_that_only_touch_them() {
        let backend = indexed(&[(10 * SEC, 5_000)]);
        // Ends exactly at from_ns.
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(15 * SEC, 20 * SEC))
                .len(),
            1
        );
        assert!(backend
            .query("cam", EventPage::unbounded(15 * SEC + 1, 20 * SEC))
            .is_empty());
        // Starts exactly at to_ns.
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, 10 * SEC))
                .len(),
            1
        );
        assert!(backend
            .query("cam", EventPage::unbounded(0, 10 * SEC - 1))
            .is_empty());
    }

    #[test]
    fn query_with_an_inverted_range_is_empty() {
        // These bounds used to be computed independently and sliced, which
        // panicked here with start > end.
        let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000)]);
        assert!(backend
            .query("cam", EventPage::unbounded(u64::MAX, 0))
            .is_empty());
        assert!(backend
            .query("cam", EventPage::unbounded(20 * SEC, 5 * SEC))
            .is_empty());
    }
}
