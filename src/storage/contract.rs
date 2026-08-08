//! What every warm-storage backend promises, and the mechanics that make two very different
//! backends promise the same thing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::retry::{jittered, RetrySchedule};

/// How often a wait looks up to see whether shutdown has been asked for.
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
#[derive(Clone)]
pub struct StopFlag(Option<Arc<AtomicBool>>);

impl StopFlag {
    /// The process shutdown flag, shared with the drain that raises it.
    pub fn shared(flag: Arc<AtomicBool>) -> Self {
        Self(Some(flag))
    }

    /// A flag that is never raised.
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
    /// Run `op` until it succeeds, until retrying it stops being worth it, or until shutdown
    /// — whichever comes first.
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

/// A client-side cap on stored bytes, and the reservations that keep concurrent writers from
/// walking through it together.
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

    /// Claim `bytes` of the budget for a write that is about to happen. The claim is released
    /// when the returned guard is dropped, whichever way the write ended — the bytes are
    /// either indexed by then (and so counted for real) or never arrived.
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

    /// **Durability.** An event reported `Written` is stored — readable back through the
    /// trait, in full, at once.
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

    /// **Accounting.** Every byte an event costs is accounted for as part of *that event*, so
    /// deleting the event gives all of them back — the sidecar and the filmstrip frames
    /// included.
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

    /// Rewriting one event must replace its index entry and byte accounting.
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

    /// Upgrades reclassify exactly one event without making it unplayable.
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

    /// An upgrade overtaken by retention must not resurrect metadata or an index entry for
    /// deleted footage.
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

    #[test]
    fn a_flag_that_was_never_shared_never_stops() {
        assert!(!StopFlag::never().stopped());
        assert!(raised().stopped());
    }

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
            assert_eq!(budget.committed(40), 120);
            assert!(budget.over(40));
            assert_eq!(budget.remaining(40), 0);
        }
        assert_eq!(budget.committed(40), 40);
        assert_eq!(budget.remaining(40), 60);
    }

    #[test]
    fn a_zero_limit_is_unlimited_rather_than_full() {
        let budget = ByteBudget::new(0);
        assert!(budget.unlimited());
        let _held = budget.reserve(u64::MAX);
        assert!(!budget.over(u64::MAX));
        assert_eq!(budget.remaining(u64::MAX), u64::MAX);
    }

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
