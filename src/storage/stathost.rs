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
    evict_tiers, filmstrip_frame_count, sweep_expired, EmergencyOutcome, EventIdentity, EventIndex,
    EventPage, EvictionPolicy, Removal,
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

/// One of the backend's two HTTP clients: the plain one, or — with a
/// `read_timeout` — the streaming one.
///
/// Building a client is where `reqwest` starts the TLS backend, loads the
/// system root certificates and parses the environment's proxy settings, so
/// this fails on a box that has been stripped of its CA bundle or handed a
/// malformed `HTTPS_PROXY`. Both are permanent and both are an operator's to
/// fix — and a panic from inside a constructor names neither. The failure is
/// returned so the caller can end startup with an error that does
/// ([`crate::app::RunError::WarmStorage`]), not so the process can limp on.
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

/// What a failed client build says to the operator, from wherever it came.
/// The three causes named are the three things `reqwest` does here that can
/// fail on a real box; the underlying error alone says none of them.
fn client_build_error(cause: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(format!(
        "the stathost HTTP client could not be built — check the TLS backend, the system \
         root certificate store and any proxy settings in the environment: {cause}"
    ))
}

#[cfg(test)]
thread_local! {
    /// Makes the next [`build_client`] fail, for the test that pins what a
    /// backend which cannot build one does about it. Per thread rather than
    /// global because the test harness runs every test on its own thread, and a
    /// process-wide switch would fail whichever unrelated backend happened to
    /// be under construction at the time.
    static FAIL_CLIENT_BUILD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn fail_client_build() -> bool {
    FAIL_CLIENT_BUILD.with(std::cell::Cell::get)
}

/// Make [`build_client`] fail on this thread, for the tests that pin what a
/// backend which cannot build one does about it — here, and in `app`, where
/// what matters is that the failure ends startup.
#[cfg(test)]
pub(crate) fn force_client_build_failure(fail: bool) {
    FAIL_CLIENT_BUILD.with(|forced| forced.set(fail));
}

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
    /// Build the backend, or report why its HTTP clients could not be.
    ///
    /// Fallible because `reqwest` initializes the TLS stack and reads the
    /// environment's proxy settings when a client is built: a box whose root
    /// certificate store will not load, or whose `HTTPS_PROXY` is malformed,
    /// fails here and fails every time it starts. That is a configuration
    /// fault an operator has to be told about, and a panic from inside a
    /// constructor tells them the least of any option: the error is returned
    /// so startup can end with a line that names the fault
    /// ([`crate::app::RunError::WarmStorage`] — fatal, because with stathost
    /// configured there is no other place footage persists).
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
                self.http.delete(&thumb_key(camera_id, &stem, i)).await,
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
    /// already gone, so what remains is exactly `0..=refused` — a prefix, which
    /// the entry can claim as a count and be right about against both the host
    /// and the next scan, with no frame above it left over to account for.
    /// Bottom-up would strand the frames above the hole instead: the entry
    /// would claim fewer than are there, and `delete_event_objects` walks
    /// `0..filmstrip_frames`, so the survivors would outlive the event. (A
    /// rebuilt index would find them — the scan counts to the highest frame
    /// there is, not to the first gap — but nothing rebuilds an index in the
    /// life of one process, and the delete that leaks them runs long before.)
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
                self.http.delete(&thumb_key(camera_id, &stem, i)).await,
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

    /// Hand a startup pass's collected entries to the index, one camera at a
    /// time and each list in one step — sorted once rather than n times.
    ///
    /// Replacing a camera's whole list is only right for a startup scan, and
    /// only because of what that scan is: awaited before the first camera is
    /// spawned, over an index no write path has touched yet, from a listing in
    /// which one stored object appears once. Nothing of this process's can be
    /// displaced because nothing of this process's is there.
    fn take_collected(&self, collected: HashMap<String, Vec<WarmEventEntry>>) {
        for (camera_id, entries) in collected {
            self.events.replace_camera(&camera_id, entries);
        }
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
            let sidecar_bytes = sizes
                .get(sidecar_key(camera_id, stem).as_str())
                .copied()
                .unwrap_or(0);
            // The frame count is a high-water mark, not a tally of what is
            // there: a filmstrip missing its first frame still has the rest,
            // and the deletes walk `0..filmstrip_frames` — see
            // [`filmstrip_frame_count`]. The bytes are only what the listing
            // really named, so a hole costs the budget nothing it does not owe.
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
        // A startup pass builds each camera's list here and hands it over whole
        // at the end (`take_collected`). Inserting event by event lands each
        // one in the middle of a list the store gave no useful order to — stems
        // are decimal and sort lexicographically, and a box with no RTC dates a
        // whole archive from one start PTS — so every insertion shifts the rest
        // and building the index costs O(n²) memory traffic in the one pass
        // that walks the entire archive. A heal cannot do this: it runs beside
        // the live write path and must yield to what that path has already
        // indexed, one identity at a time.
        let mut collected: HashMap<String, Vec<WarmEventEntry>> = HashMap::new();

        while let Some((event, read)) = reads.next().await {
            // One archive's worth of round trips is a long time to hold a
            // shutdown drain that is measured in one event's deletes, and an
            // index nobody is going to use is not worth finishing.
            if stop() {
                // What was collected still goes in: it came from the store and
                // is true, exactly as it was when each entry was inserted
                // singly. What does not happen is `mark_scanned` below, so
                // nothing prunes on a half-built index.
                self.take_collected(collected);
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
        // Non-fatal — the UI hides frames that fail to load. A failed frame
        // still stops the upload, and the entry claims the prefix that landed:
        // not because a later scan could not see past a gap (it counts to the
        // highest frame there is — see [`filmstrip_frame_count`]), but because
        // the frames are a strip. Uploading 3 after 2 failed spends requests,
        // on the link that just refused one, to produce a filmstrip with a hole
        // in the middle of it; stopping leaves a shorter one that is whole.
        let mut filmstrip_frames = 0usize;
        let mut stored_frame_bytes = 0u64;
        for (i, jpeg) in frames.iter().enumerate() {
            let key = thumb_key(camera_id, &stem, i);
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
        let sidecar_key = sidecar_key(camera_id, &stem);
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
            // A 206 says the body is a *slice*, and only its `Content-Range`
            // says which one. Without a usable header there is nothing to serve
            // it as: passing the slice off as the whole file hands the player
            // an object whose bytes are not at the offsets it will seek to, and
            // relaying a range the arithmetic cannot make sense of (start past
            // end, end past the total) is a length this process would have to
            // subtract its way to — an underflow in a debug build, and this
            // codebase ships debug. The read fails instead; the API turns that
            // into one clean error rather than a corrupt playback.
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
mod tests;
