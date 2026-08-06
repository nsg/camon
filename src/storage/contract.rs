//! What every warm-storage backend promises, and the mechanics that make two
//! very different backends promise the same thing.
//!
//! [`WarmStorageBackend`](crate::storage::WarmStorageBackend) says what a
//! backend can be *asked*. This module says what it must *guarantee* while
//! answering, which is the half that was never written down — and so the half
//! that drifted. [`LocalDiskBackend`](crate::storage::LocalDiskBackend) grew its
//! guarantees inside its own write ladder, where they read as filesystem
//! mechanics rather than as policy;
//! [`StathostBackend`](crate::storage::StathostBackend) then reimplemented the
//! ladder over HTTP and kept the mechanics it could translate, which left every
//! guarantee that had no translation quietly unkept. That divergence, not any
//! one of its symptoms, is what this module exists to end: the same argument
//! [`event_index`](crate::storage::event_index) makes about the in-RAM index —
//! only the objects differ, so only the object I/O is a backend's own.
//!
//! # The contract
//!
//! ## 1. Durability
//!
//! An event reported [`Written`](crate::storage::backend::WriteOutcome::Written)
//! is stored, under its final name, with the metadata that types it — and it is
//! still all of that after a power cut. Each backend names its own *commit
//! point* (local disk: the `.ts` rename, made durable by a directory fsync; the
//! remote store: the acknowledged `.ts` `PUT`, durable by the server's rules)
//! and nothing before that point may be visible to a rebuild. A backend that
//! cannot make the promise says so — `Written` is what resets the recording
//! watchdog, so a durability failure reported as success is a camera system
//! that looks healthy while it stores nothing.
//!
//! ## 2. Accounting
//!
//! [`free_space`](crate::storage::WarmStorageBackend::free_space) is what the
//! write path steers by, so it must answer for **every byte an event costs** —
//! the video and the sidecar and the filmstrip frames alike, since none of them
//! is ever reclaimed on its own. And the room for an incoming event must be
//! **secured in advance, or settled by an arbiter that cannot be wrong about
//! it** — never merely discovered afterwards.
//!
//! Those are two different discharges and both are legitimate:
//!
//! * **Local disk** delegates to the filesystem, which is the arbiter. `statvfs`
//!   counts every byte that exists without being told what any of them are for,
//!   and the write itself is the reservation: an event that does not fit fails
//!   with `ENOSPC` at the moment it stops fitting, whichever concurrent writer
//!   got there first. Nothing in RAM can be wrong about it, so nothing in RAM is
//!   kept.
//! * **The remote store** has no arbiter — it cannot see the server's disk — so
//!   it secures the room in advance instead. A client-side budget is measured
//!   against the sum of what the index holds, that sum is the *whole* cost of
//!   each event
//!   ([`WarmEventEntry::stored_bytes`](crate::storage::WarmEventEntry::stored_bytes)),
//!   and what is *about to be* written is claimed before it is sent
//!   ([`ByteBudget`] reservations) so that two cameras writing at once do not
//!   each read a total that predates the other.
//!
//! **What happens when the room cannot be made is a policy, and it is the same
//! one on both:** the write goes ahead and says so. Local disk gets that from
//! `ENOSPC` — the writer emergency-prunes and retries once, and an event that
//! still does not fit is dropped *because the filesystem refused it*, not
//! because a number said so. The remote store has no such refusal available:
//! nothing stops the `PUT` landing, so refusing would be camon's own choice, and
//! it would fall exactly where recording matters most — a store that has stopped
//! accepting `DELETE`s is a store whose eviction cannot free anything, and it
//! would still happily take footage. So an overshoot is bounded by what eviction
//! could not reclaim, it is reported at `warn`, and every later write tries
//! again. A cap an operator typed is not worth a recording.
//!
//! ## 3. Cancellation
//!
//! No single operation may outlast the shutdown budget it is drained inside.
//! camon's shutdown is a raised flag and nothing else ([`crate::shutdown`]), and
//! phase 3 of the drain is sized so that **one** in-flight remote upload may
//! spend its whole [`UPLOAD_TIMEOUT`](crate::storage::stathost::UPLOAD_TIMEOUT)
//! and everything queued behind it is abandoned. So the rule is not "check the
//! flag often" but something exact:
//!
//! > once the flag is up, an operation may finish the request it has already
//! > issued and must issue no further one.
//!
//! Every wait obeys it ([`sleep_unless`]), and so does every loop that issues
//! requests — the upload and read ladders ([`RetryPolicy::run`], which asks
//! before each attempt rather than after each failure), the retention sweep, the
//! eviction pass that runs *ahead of a write*, and the individual `DELETE`s one
//! event's removal is made of. That last one is not pedantry: one event is up to
//! six requests, and a flag checked only between events leaves six request
//! timeouts of post-stop work inside one of them — more than the whole phase-3
//! budget, spent deleting stored footage for a write the same shutdown is about
//! to abandon.
//!
//! A backend whose every step is local I/O — local disk's — discharges this
//! structurally: there is no unbounded wait to cancel, and abandoning a write or
//! an eviction that takes milliseconds would lose footage, or leave a disk full,
//! for nothing. Its passes are handed a predicate that never fires, and say so
//! where they are handed it.
//!
//! ## 4. Index acceptance
//!
//! What may enter the in-RAM index, and on whose authority. The rules are the
//! index's ([`event_index`](crate::storage::event_index)) and are stated here
//! because they bind both backends:
//!
//! * **One entry per identity.** A re-written event replaces its entry rather
//!   than adding one, and the byte total moves by the difference — the store
//!   holds one event either way ([`insert`](crate::storage::event_index::EventIndex::insert)).
//! * **The live write path outranks a rebuild.** A scan that runs while cameras
//!   are recording yields to what is already indexed
//!   ([`insert_absent`](crate::storage::event_index::EventIndex::insert_absent)),
//!   because an entry this process wrote is newer than any listing.
//! * **A classification only moves one way.** Movement → object is terminal, so
//!   a rebuild may raise a stale entry but never lower a fresh one.
//! * **An event whose objects are gone is not indexed.** Retention runs
//!   concurrently with the write path on both backends, so a write or an
//!   upgrade that is overtaken by a sweep must not re-index the event it lost,
//!   and must not leave metadata behind for a rebuild to find.
//!
//! # What lives here
//!
//! Only the mechanics that had diverged: cancellation-aware waiting, the retry
//! schedule and the classification that decides whether retrying is even the
//! right response, and the byte reservation. The object I/O stays with the
//! backend that owns the objects — that part is *supposed* to differ.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::retry::{jittered, RetrySchedule};

/// How often a wait looks up to see whether shutdown has been asked for.
///
/// camon's shutdown is a raised `AtomicBool` and nothing else, so a long wait
/// either polls it or ignores it; the tasks these waits run on are joined rather
/// than aborted, and every wait one sits through is the drain waiting too.
pub(crate) const SHUTDOWN_POLL: Duration = Duration::from_millis(250);

/// Sleep out `delay`, returning early if `stop` reports shutdown meanwhile.
pub(crate) async fn sleep_unless(delay: Duration, stop: &impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + delay;
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() || stop() {
            return;
        }
        tokio::time::sleep(left.min(SHUTDOWN_POLL)).await;
    }
}

/// The shutdown flag, as the storage layer holds it.
///
/// A newtype rather than a bare `Arc<AtomicBool>` so that the two questions a
/// backend actually asks — "may I issue another request?" and "wait this long
/// unless we are stopping" — are the only two it can ask, and so that the
/// backends built without one ([`StopFlag::never`], and every test that is not
/// about shutdown) are visibly that rather than accidentally that.
#[derive(Clone)]
pub struct StopFlag(Option<Arc<AtomicBool>>);

impl StopFlag {
    /// The process shutdown flag, shared with the drain that raises it.
    pub fn shared(flag: Arc<AtomicBool>) -> Self {
        Self(Some(flag))
    }

    /// A flag that is never raised. Tests only: production always builds the
    /// remote backend with the flag the drain raises, and the startup scan —
    /// which does run before the signal handlers are registered, so a SIGTERM
    /// during it is still the default action — passes its own never-firing
    /// predicate to [`scan_with_retries`](crate::storage::StathostBackend)
    /// rather than swapping the backend's flag out.
    pub fn never() -> Self {
        Self(None)
    }

    pub fn stopped(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    /// Sleep out `delay` unless shutdown arrives during it.
    pub async fn sleep_unless_stopped(&self, delay: Duration) {
        sleep_unless(delay, &|| self.stopped()).await;
    }
}

/// Whether a failed attempt is worth making again.
///
/// The distinction is not decoration. A retry of a request the far end has
/// already refused on its merits — a bad token, a path it will not accept —
/// buys nothing and costs a full request timeout on a task a camera's recording
/// is queued behind; a retry of a connection that was reset is the whole reason
/// the retry exists. Retrying both alike, immediately, is what this replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Recovery {
    /// Something about the moment: a reset connection, a timeout, a proxy that
    /// was restarting, a server that is briefly out of capacity.
    Transient,
    /// Something about the request: the same bytes sent again get the same
    /// answer, so the only thing another attempt spends is time.
    Permanent,
}

/// How one remote operation is retried: how many attempts it is worth and how
/// long to wait between them.
#[derive(Clone, Copy)]
pub(crate) struct RetryPolicy {
    /// Total attempts, the first one included. `1` is "no retry".
    pub(crate) attempts: u32,
    /// The backoff between attempts, jittered on the way out — a fleet of camons
    /// coming back from one power cut must not re-attempt in lockstep.
    pub(crate) schedule: RetrySchedule,
}

/// How a retried operation ended.
pub(crate) enum Attempted<T, E> {
    Done(T),
    /// Every attempt this policy allows was spent, or the failure was one that
    /// retrying cannot mend. Carries the last error for the caller to report.
    Failed(E),
    /// Shutdown arrived before an attempt could be issued. Nothing was sent, so
    /// nothing is in flight and the drain is waiting for nothing — which is the
    /// entire point of asking before the attempt rather than after the failure.
    Abandoned,
}

impl RetryPolicy {
    /// Run `op` until it succeeds, until retrying it stops being worth it, or
    /// until shutdown — whichever comes first.
    ///
    /// The stop flag is read *before* each attempt, including the first. That
    /// ordering is the cancellation guarantee in this module's contract: a
    /// request already in flight is finished (nothing here can cancel one
    /// mid-body, and abandoning a `PUT` says nothing about whether the origin
    /// committed it), and no further request is issued, so the drain waits for
    /// at most the one request rather than for a whole ladder of them.
    pub(crate) async fn run<T, E, F, Fut>(
        &self,
        stop: &StopFlag,
        classify: impl Fn(&E) -> Recovery,
        mut op: F,
    ) -> Attempted<T, E>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        let mut delay = self.schedule.start;
        for attempt in 1..=self.attempts {
            if stop.stopped() {
                return Attempted::Abandoned;
            }
            match op().await {
                Ok(value) => return Attempted::Done(value),
                Err(e) => {
                    if attempt == self.attempts || classify(&e) == Recovery::Permanent {
                        return Attempted::Failed(e);
                    }
                    stop.sleep_unless_stopped(jittered(delay)).await;
                    delay = self.schedule.next(delay);
                }
            }
        }
        // Unreachable for `attempts >= 1`: the loop returns on its last pass.
        // A policy configured with zero attempts issues no request, which is a
        // failure to do the work rather than a success at it.
        Attempted::Abandoned
    }
}

/// A client-side cap on stored bytes, and the reservations that keep concurrent
/// writers from walking through it together.
///
/// For a backend that cannot see the store's disk. The cap itself is only half
/// of it: a guard that reads "used" and then writes has already answered a
/// question about the past by the time it acts, and every camera's writer task
/// asks the same question independently. So a write reserves its cost first and
/// releases it once the bytes are indexed (or the write has failed), and the
/// figure everything steers by is used-plus-reserved.
///
/// `limit == 0` means unlimited — a documented sentinel, not an accident, and
/// the reason every method here is a no-op in that case rather than a division
/// of zero among writers.
pub(crate) struct ByteBudget {
    limit: u64,
    reserved: AtomicU64,
}

impl ByteBudget {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            reserved: AtomicU64::new(0),
        }
    }

    pub(crate) fn limit(&self) -> u64 {
        self.limit
    }

    pub(crate) fn unlimited(&self) -> bool {
        self.limit == 0
    }

    /// What the store holds plus what is already promised to writes in flight.
    /// Saturating, so a reservation released twice — which nothing here can do,
    /// [`Reservation`] releases on drop — could not wrap this into "empty".
    pub(crate) fn committed(&self, used: u64) -> u64 {
        used.saturating_add(self.reserved.load(Ordering::Relaxed))
    }

    /// Claim `bytes` of the budget for a write that is about to happen. The
    /// claim is released when the returned guard is dropped, whichever way the
    /// write ended — the bytes are either indexed by then (and so counted for
    /// real) or never arrived.
    pub(crate) fn reserve(&self, bytes: u64) -> Reservation<'_> {
        self.reserved.fetch_add(bytes, Ordering::Relaxed);
        Reservation {
            budget: self,
            bytes,
        }
    }

    /// Whether `used` plus everything reserved is over the cap. Always `false`
    /// on an unlimited budget.
    pub(crate) fn over(&self, used: u64) -> bool {
        !self.unlimited() && self.committed(used) > self.limit
    }

    /// What is left of the budget, as a free-space figure. `u64::MAX` when
    /// unlimited, so a caller comparing against a low-space threshold never
    /// fires.
    pub(crate) fn remaining(&self, used: u64) -> u64 {
        if self.unlimited() {
            u64::MAX
        } else {
            self.limit.saturating_sub(self.committed(used))
        }
    }
}

/// Bytes claimed from a [`ByteBudget`] for one write in progress, returned on
/// drop. Held across the write's `await`s, so a panic or an early return gives
/// them back rather than leaking a permanent phantom occupancy.
pub(crate) struct Reservation<'a> {
    budget: &'a ByteBudget,
    bytes: u64,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.budget
            .reserved
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// The contract's own assertions, written once and run against every backend.
///
/// The divergence is the disease, so the cure has to be provable convergence:
/// each of these is one assertion body with two call sites, one in each
/// backend's test module. A guarantee that only holds where it was written down
/// is how the two backends came apart in the first place.
#[cfg(test)]
pub(crate) mod contract_tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use crate::buffer::warm::{EventUpgrade, FinishedEvent};
    use crate::buffer::GopSegment;
    use crate::storage::backend::WriteOutcome;
    use crate::storage::event_index::DetectionDetail;
    use crate::storage::{EventPage, EventType, WarmStorageBackend};

    /// A one-second movement event at `first_pts`, `size` bytes of video, with
    /// two filmstrip frames — the same shape on either backend, because a
    /// `FinishedEvent` is what the analyzer hands both of them.
    pub(crate) fn event(first_pts: u64, size: usize) -> FinishedEvent {
        FinishedEvent {
            segments: vec![GopSegment {
                start_pts: first_pts,
                duration_ns: 1_000_000_000,
                data: Arc::new(vec![0xab; size]),
                frame_count: 1,
            }],
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

    fn upgrade(first_pts: u64) -> EventUpgrade {
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

    /// **Durability.** An event reported `Written` is stored — readable back
    /// through the trait, in full, at once. Not the whole guarantee (the rest of
    /// it is about a power cut, and neither backend's commit point can be
    /// interrupted from inside the process, so each pins its own ladder: the
    /// fsync order on one side, the sidecar-before-video order on the other),
    /// but the half a caller can actually check, and the half that fails first
    /// when a backend reports a write it did not make.
    pub(crate) async fn a_written_event_reads_back_whole(backend: &dyn WarmStorageBackend) {
        assert_eq!(
            backend.write_event("cam", &event(1_000, 40)).await,
            WriteOutcome::Written
        );

        let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1);
        let video = backend
            .read_video("cam", &entries[0], None)
            .await
            .expect("an event reported Written is not readable");
        assert_eq!(video.total_size, 40);
        let mut body = Vec::new();
        let mut stream = video.stream;
        while let Some(chunk) = futures_util::StreamExt::next(&mut stream).await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(body, vec![0xab; 40]);
    }

    /// **Accounting.** Every byte an event costs is accounted for as part of
    /// *that event*, so deleting the event gives all of them back — the sidecar
    /// and the filmstrip frames included. Nothing else ever collects them: they
    /// are not indexed on their own, not expired on their own, and not evicted
    /// on their own, so a byte an event's deletion leaves behind is a byte
    /// nothing will ever reclaim.
    ///
    /// `stored` is what the underlying store really holds, which is the only
    /// figure both backends can be asked for in the same terms — one of them
    /// measures its budget in RAM and the other delegates to `statvfs`, and that
    /// difference is guarantee 2's, not this assertion's. The *other* half of
    /// guarantee 2 — reserving the room in advance — has no shared form for the
    /// same reason: local disk's reservation is `ENOSPC` itself, which cannot be
    /// asked for from here and cannot be wrong.
    pub(crate) async fn an_event_costs_nothing_once_it_is_deleted(
        backend: &dyn WarmStorageBackend,
        stored: impl Fn() -> u64,
    ) {
        let empty = stored();
        let old = 1_000_000_000;
        backend.write_event("cam", &event(old, 40)).await;

        let with_event = stored();
        assert!(
            with_event > empty + 40,
            "an event cost the store {} bytes, which is its video and nothing else — its \
             sidecar and filmstrip frames are stored somewhere this cannot see",
            with_event - empty
        );

        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "the sweep did not delete the event this is about"
        );
        assert_eq!(
            stored(),
            empty,
            "deleting the event left {} bytes behind that nothing will ever reclaim",
            stored() - empty
        );
    }

    /// **Cancellation.** A sweep is many deletions and the drain waits for it,
    /// so a sweep that starts stopped does nothing at all — on either backend,
    /// whether a deletion is an `unlink` or a request that can sit on a timeout.
    pub(crate) async fn a_prune_that_starts_stopped_deletes_nothing(
        backend: &dyn WarmStorageBackend,
    ) {
        let old = 1_000_000_000;
        backend.write_event("cam", &event(old, 40)).await;
        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            1
        );

        // Everything is expired by a retention of one nanosecond.
        backend.prune(1, 1, 1, &AtomicBool::new(true)).await;

        assert_eq!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .len(),
            1,
            "a sweep that started stopped deleted an event anyway"
        );
    }

    /// **Index acceptance, one entry per identity.** Re-writing an event
    /// re-writes the objects it names on both backends — a `PUT` of an existing
    /// key, a `.ts` written over — so the index keeps one entry and the byte
    /// total follows the rewrite instead of counting the same event twice.
    pub(crate) async fn a_rewritten_event_replaces_its_entry(backend: &dyn WarmStorageBackend) {
        backend.write_event("cam", &event(1_000, 40)).await;
        backend.write_event("cam", &event(1_000, 25)).await;

        let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1, "a rewrite added a second entry");
        assert_eq!(
            entries[0].file_size, 25,
            "the index describes the write that was replaced"
        );
    }

    /// **Index acceptance, a classification that only moves one way.** An
    /// upgrade reclassifies the one event it names and leaves it playable; how
    /// the objects get there (a rename between directories, a sidecar rewritten
    /// in place) is the backend's business and not the contract's.
    pub(crate) async fn an_upgrade_reclassifies_the_one_indexed_event(
        backend: &dyn WarmStorageBackend,
    ) {
        backend.write_event("cam", &event(1_000, 40)).await;
        backend.upgrade_event("cam", &upgrade(1_000)).await;

        let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(entries.len(), 1, "the upgrade left two entries behind");
        assert_eq!(entries[0].event_type, EventType::Object);
        assert_eq!(entries[0].object_classes, vec!["person".to_string()]);
        assert_eq!(
            backend
                .read_video("cam", &entries[0], None)
                .await
                .map(|v| v.total_size)
                .unwrap_or(0),
            40,
            "the upgraded event's video is no longer where the index says it is"
        );
    }

    /// **Index acceptance, an event whose objects are gone.** Retention runs
    /// concurrently with the writer on both backends, so an upgrade that is
    /// overtaken by a sweep must leave nothing behind: no index entry claiming
    /// footage that has been deleted, and no metadata resurrected beside it.
    /// Here the sweep has plainly already happened, which is the easy half; each
    /// backend pins the interleaved case in its own terms.
    pub(crate) async fn an_upgrade_of_a_deleted_event_indexes_nothing(
        backend: &dyn WarmStorageBackend,
    ) {
        let old = 1_000_000_000;
        backend.write_event("cam", &event(old, 40)).await;
        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "the sweep did not delete the event this is about"
        );

        backend.upgrade_event("cam", &upgrade(old)).await;

        assert!(
            backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
            "an upgrade re-indexed an event whose footage retention had deleted"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule() -> RetrySchedule {
        RetrySchedule {
            start: Duration::from_millis(1),
            max: Duration::from_millis(4),
        }
    }

    fn policy(attempts: u32) -> RetryPolicy {
        RetryPolicy {
            attempts,
            schedule: schedule(),
        }
    }

    fn raised() -> StopFlag {
        StopFlag::shared(Arc::new(AtomicBool::new(true)))
    }

    /// The whole point of the classification: a refusal the far end will repeat
    /// costs one attempt, not the policy's whole allowance, on a task a camera's
    /// recording is queued behind.
    #[tokio::test]
    async fn a_permanent_failure_is_not_attempted_twice() {
        let tries = AtomicU64::new(0);
        let outcome: Attempted<(), &str> = policy(4)
            .run(
                &StopFlag::never(),
                |_| Recovery::Permanent,
                || async {
                    tries.fetch_add(1, Ordering::Relaxed);
                    Err("refused")
                },
            )
            .await;
        assert!(matches!(outcome, Attempted::Failed("refused")));
        assert_eq!(tries.load(Ordering::Relaxed), 1);
    }

    /// A transient failure spends the whole allowance and no more.
    #[tokio::test]
    async fn a_transient_failure_is_attempted_until_the_allowance_runs_out() {
        let tries = AtomicU64::new(0);
        let outcome: Attempted<(), &str> = policy(3)
            .run(
                &StopFlag::never(),
                |_| Recovery::Transient,
                || async {
                    tries.fetch_add(1, Ordering::Relaxed);
                    Err("reset")
                },
            )
            .await;
        assert!(matches!(outcome, Attempted::Failed("reset")));
        assert_eq!(tries.load(Ordering::Relaxed), 3);
    }

    /// A success on the second attempt is a success, and stops there.
    #[tokio::test]
    async fn a_retry_that_succeeds_stops_retrying() {
        let tries = AtomicU64::new(0);
        let outcome: Attempted<u64, &str> = policy(4)
            .run(
                &StopFlag::never(),
                |_| Recovery::Transient,
                || async {
                    let n = tries.fetch_add(1, Ordering::Relaxed) + 1;
                    if n < 2 {
                        Err("reset")
                    } else {
                        Ok(n)
                    }
                },
            )
            .await;
        assert!(matches!(outcome, Attempted::Done(2)));
        assert_eq!(tries.load(Ordering::Relaxed), 2);
    }

    /// The cancellation rule, at its sharpest: the flag is read before the
    /// *first* attempt, so a stopped policy issues nothing and the drain waits
    /// for nothing.
    #[tokio::test]
    async fn a_stopped_policy_issues_no_request_at_all() {
        let tries = AtomicU64::new(0);
        let outcome: Attempted<(), &str> = policy(4)
            .run(
                &raised(),
                |_| Recovery::Transient,
                || async {
                    tries.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                },
            )
            .await;
        assert!(matches!(outcome, Attempted::Abandoned));
        assert_eq!(tries.load(Ordering::Relaxed), 0);
    }

    /// Shutdown that arrives mid-ladder ends it after the attempt already in
    /// flight, not after the rest of the allowance.
    #[tokio::test]
    async fn shutdown_between_attempts_abandons_the_rest_of_the_ladder() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop = StopFlag::shared(Arc::clone(&flag));
        let tries = AtomicU64::new(0);
        let outcome: Attempted<(), &str> = policy(5)
            .run(
                &stop,
                |_| Recovery::Transient,
                || async {
                    tries.fetch_add(1, Ordering::Relaxed);
                    flag.store(true, Ordering::Relaxed);
                    Err("reset")
                },
            )
            .await;
        assert!(matches!(outcome, Attempted::Abandoned));
        assert_eq!(tries.load(Ordering::Relaxed), 1);
    }

    /// A wait that outlasts the stop flag is the drain waiting too.
    #[tokio::test(start_paused = true)]
    async fn a_wait_ends_when_shutdown_arrives_during_it() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop = StopFlag::shared(Arc::clone(&flag));
        let raiser = {
            let flag = Arc::clone(&flag);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                flag.store(true, Ordering::Relaxed);
            })
        };
        let started = tokio::time::Instant::now();
        stop.sleep_unless_stopped(Duration::from_secs(3600)).await;
        raiser.await.unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "a wait sat out its whole delay through a shutdown"
        );
    }

    /// A flag that is never raised is never raised: [`StopFlag::never`] is for
    /// the startup scan, which runs before there is a drain to be late for.
    #[test]
    fn a_flag_that_was_never_shared_never_stops() {
        assert!(!StopFlag::never().stopped());
        assert!(raised().stopped());
    }

    /// Reservations are what stop two writers walking through one guard
    /// together: each sees the other's claim in the figure it steers by, and the
    /// claim goes back when the guard is dropped.
    #[test]
    fn a_reservation_is_visible_to_everything_that_reads_the_budget() {
        let budget = ByteBudget::new(100);
        assert_eq!(budget.committed(40), 40);
        assert!(!budget.over(40));
        {
            let _one = budget.reserve(40);
            assert_eq!(budget.committed(40), 80);
            assert!(!budget.over(40));
            let _two = budget.reserve(40);
            // 40 stored plus two 40-byte writes in flight is over a cap of 100,
            // which neither writer could tell on its own.
            assert_eq!(budget.committed(40), 120);
            assert!(budget.over(40));
            assert_eq!(budget.remaining(40), 0);
        }
        assert_eq!(budget.committed(40), 40);
        assert_eq!(budget.remaining(40), 60);
    }

    /// The documented sentinel: nothing is capped, nothing is refused, and the
    /// free-space figure never trips a low-space guard.
    #[test]
    fn a_zero_limit_is_unlimited_rather_than_full() {
        let budget = ByteBudget::new(0);
        assert!(budget.unlimited());
        let _held = budget.reserve(u64::MAX);
        assert!(!budget.over(u64::MAX));
        assert_eq!(budget.remaining(u64::MAX), u64::MAX);
    }

    /// A reservation released by a panic unwinding through the write it was
    /// taken for is a reservation released: a phantom occupancy that outlived
    /// its write would shrink the budget for the life of the process.
    #[test]
    fn a_reservation_comes_back_even_when_its_write_panics() {
        let budget = ByteBudget::new(100);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = budget.reserve(60);
            assert_eq!(budget.committed(0), 60);
            panic!("the write gives up half way");
        }));
        assert!(panicked.is_err());
        assert_eq!(budget.committed(0), 0);
    }
}
