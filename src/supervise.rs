//! Who notices when a long-lived task stops, and what the process does about it.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::locks::MutexExt;
use crate::retry::{jittered, RetrySchedule};

/// How a supervised task left. Distinguished for the report only; the policy
/// treats all three the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The task's future or closure returned.
    Returned,
    /// It panicked, and the unwind is what dropped the guard.
    Panicked,
    /// It was aborted, or dropped without finishing.
    Cancelled,
}

impl Exit {
    /// For a sentence: "the analyzer *was cancelled* while camon was running".
    fn describe(self) -> &'static str {
        match self {
            Exit::Returned => "returned",
            Exit::Panicked => "panicked",
            Exit::Cancelled => "was cancelled",
        }
    }

    /// For a list: `analyzer:front (panicked)`.
    fn kind(self) -> &'static str {
        match self {
            Exit::Returned => "returned",
            Exit::Panicked => "panicked",
            Exit::Cancelled => "cancelled",
        }
    }
}

/// One supervised task's death, as the operator is told about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Death {
    pub task: String,
    pub exit: Exit,
}

impl std::fmt::Display for Death {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.task, self.exit.kind())
    }
}

/// How many consecutive failures a restartable task is allowed before its
/// policy escalates to the fatal one.
pub const PERIODIC_RESTARTS: usize = 3;

/// When a restartable task gives up on itself: how many failures in a row it is allowed, and
/// how long an attempt has to survive to prove the streak is over.
#[derive(Debug, Clone, Copy)]
pub struct RestartLimit {
    /// Consecutive failures allowed. The next one escalates.
    pub max: usize,
    /// The uptime that clears the streak. An attempt that ran at least this
    /// long before it died has proved itself and starts the count over.
    pub healthy_after: Duration,
}

impl RestartLimit {
    /// The limit for a task that does its work once every `cadence`.
    pub fn cycling_every(cadence: Duration) -> Self {
        Self {
            max: PERIODIC_RESTARTS,
            healthy_after: cadence.saturating_mul(2),
        }
    }
}

/// Backoff between restarts of the same task, so a task that panics on sight cannot spin a core
/// through its whole streak.
const RESTART_BACKOFF: RetrySchedule = RetrySchedule {
    start: Duration::from_secs(1),
    max: Duration::from_secs(4),
};

/// What to do about the failure just recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Inside the streak: spawn it again after [`RestartStreak::backoff`].
    Restart,
    /// The streak is spent. Escalate to the fatal policy.
    Exhausted,
}

/// One task's run of failures: how many attempts have died in a row without
/// one of them proving itself, and what the next death means. Pure with
/// respect to the attempt's uptime, so tests need no clock.
pub struct RestartStreak {
    limit: RestartLimit,
    failures: usize,
}

impl RestartStreak {
    pub fn new(limit: RestartLimit) -> Self {
        Self { limit, failures: 0 }
    }

    /// Count a failure, `uptime` being how long the attempt that just died had
    /// been running. An attempt that lasted [`RestartLimit::healthy_after`]
    /// clears whatever streak preceded it.
    pub fn record(&mut self, uptime: Duration) -> Verdict {
        if uptime >= self.limit.healthy_after {
            self.failures = 0;
        }
        self.failures += 1;
        if self.failures > self.limit.max {
            Verdict::Exhausted
        } else {
            Verdict::Restart
        }
    }

    /// How long to wait before spawning the task again: one doubling per
    /// failure in the current streak. Un-jittered — the caller adds that.
    pub fn backoff(&self) -> Duration {
        let mut delay = RESTART_BACKOFF.start;
        for _ in 1..self.failures {
            delay = RESTART_BACKOFF.next(delay);
        }
        delay
    }
}

/// How many deaths are kept for the report: the operator needs the cascade,
/// not an unbounded log. Anything past this is counted and not named.
const MAX_RECORDED_DEATHS: usize = 8;

/// The shared half of the supervisor: the stop flag it reads, the wakeup it
/// uses to cut a backoff short, the way it asks for a drain, and the cascade it
/// has recorded so far.
struct Inner {
    stopping: Arc<AtomicBool>,
    /// The drain's broadcast wakeup, so a task parked in restart backoff when
    /// the stop comes does not hold the drain up for the rest of it.
    wake: Arc<tokio::sync::Notify>,
    request_stop: Box<dyn Fn() + Send + Sync>,
    /// A copy of "a death started this stop" for the drain's watchdog thread, which polls
    /// rather than locks. Written inside the critical section beside the entry that justifies
    /// it, so it can never disagree with the report.
    died: Arc<AtomicBool>,
    /// The one thing every decision here is made against. See [`Report`].
    report: Mutex<Report>,
}

/// What has died so far, and the only place the answer is kept.
#[derive(Default)]
struct Report {
    /// Every death worth attributing, in arrival order. The first decided the
    /// stop; the first is not reliably the origin (see the module header).
    deaths: Vec<Death>,
    /// Deaths past [`MAX_RECORDED_DEATHS`], so the report can admit there were
    /// more rather than quietly truncating.
    unnamed: usize,
}

impl Report {
    /// The snapshot a classification is made from; taking `&self` means it
    /// cannot be reached without holding the lock.
    fn state(&self, stopping: bool) -> State {
        State {
            stopping,
            death_started_it: !self.deaths.is_empty(),
        }
    }

    /// Append a death, bounded. Returns whether it is the first — the one that
    /// asks for the drain.
    fn record(&mut self, name: &str, exit: Exit) -> bool {
        let first = self.deaths.is_empty();
        if self.deaths.len() < MAX_RECORDED_DEATHS {
            self.deaths.push(Death {
                task: name.to_string(),
                exit,
            });
        } else {
            self.unnamed += 1;
        }
        first
    }

    fn lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self.deaths.iter().map(Death::to_string).collect();
        if self.unnamed > 0 {
            lines.push(format!("and {} more", self.unnamed));
        }
        lines
    }
}

/// The two facts a classification turns on, read together or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct State {
    /// The process-wide stop flag.
    stopping: bool,
    /// Whether a supervised death started that stop — the same question as
    /// whether anything has been reported.
    death_started_it: bool,
}

/// What an exit means, from a state that was read all at once. Pure and total,
/// so `the_classification_table_reads_both_facts_together` pins the whole
/// table.
fn meaning(state: State, exit: Exit) -> Meaning {
    if !state.stopping {
        return Meaning::Death;
    }
    // A panic is never a way of being asked to stop, so inside a stop a death
    // started it is the cascade arriving. Inside a *deliberate* stop it is
    // not recorded — that would turn a clean stop nonzero.
    if exit == Exit::Panicked && state.death_started_it {
        return Meaning::Aftermath;
    }
    Meaning::Expected
}

/// Spawns long-lived tasks and outlives them. Cheap to clone — every clone is
/// the same supervisor.
#[derive(Clone)]
pub struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    /// `stopping` is the process-wide stop flag, `wake` the broadcast that goes
    /// with it, and `request_stop` asks for the same graceful drain a signal
    /// does — including waking whoever is waiting to run it.
    pub fn new(
        stopping: Arc<AtomicBool>,
        wake: Arc<tokio::sync::Notify>,
        request_stop: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                stopping,
                wake,
                request_stop: Box::new(request_stop),
                died: Arc::new(AtomicBool::new(false)),
                report: Mutex::new(Report::default()),
            }),
        }
    }

    fn stopping(&self) -> bool {
        self.inner.stopping.load(Ordering::Relaxed)
    }

    /// Raised as soon as a task's death has been accepted, before the drain it
    /// asks for has begun. Polled by the drain's watchdog thread.
    pub fn died(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.inner.died)
    }

    /// The death that decided the stop, if a death did. *Not* a claim about which task failed
    /// first — a cascade's origin routinely arrives second (see the module header), which is
    /// why the operator is shown [`Self::deaths`] and not this.
    pub fn first_failure(&self) -> Option<Death> {
        self.inner.report.lock_recover().deaths.first().cloned()
    }

    /// Every death worth attributing, in arrival order. Empty when the process
    /// is stopping for a reason of its own.
    pub fn deaths(&self) -> Vec<String> {
        self.inner.report.lock_recover().lines()
    }

    /// Report an exit, attribute it if it is worth attributing, and say whether the policy
    /// still has a decision to make about it.
    fn settle(&self, name: &str, exit: Exit) -> bool {
        let meaning = {
            let mut report = self.inner.report.lock_recover();
            let meaning = meaning(report.state(self.stopping()), exit);
            if meaning == Meaning::Aftermath {
                report.record(name, exit);
            }
            meaning
        };
        match meaning {
            Meaning::Death => {
                tracing::error!(
                    task = %name,
                    "supervised task {} while camon was running (any panic message is on the line \
                     above)",
                    exit.describe()
                );
                true
            }
            Meaning::Aftermath => {
                tracing::error!(
                    task = %name,
                    "supervised task panicked while camon was already stopping over another task's \
                     death; this one may well be the cause"
                );
                false
            }
            Meaning::Expected => {
                match exit {
                    Exit::Panicked => tracing::warn!(
                        task = %name,
                        "supervised task panicked while stopping; whatever it still owed the drain \
                         is lost"
                    ),
                    _ => tracing::debug!(task = %name, "supervised task stopped"),
                }
                false
            }
        }
    }

    /// Accept a task's death as the reason the process is stopping: attribute it, arm the
    /// drain's watchdog, and — if it is the first — ask for the drain itself.
    fn fatal(&self, name: &str, exit: Exit) {
        let first = {
            let mut report = self.inner.report.lock_recover();
            let first = report.record(name, exit);
            self.inner.died.store(true, Ordering::Relaxed);
            first
        };
        tracing::error!(
            task = %name,
            "camon cannot run without this task: draining what is in flight and exiting nonzero so \
             the service manager restarts the process"
        );
        if first {
            (self.inner.request_stop)();
        }
    }

    /// Sleep, cut short by a stop: a task parked here when the drain starts
    /// must not spend the drain's budget finishing its nap.
    async fn sleep_or_stop(&self, delay: Duration) {
        let notified = self.inner.wake.notified();
        if self.stopping() {
            return;
        }
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            _ = notified => {}
        }
    }

    /// Spawn a task the process cannot run without: any exit before the stop flag drains and
    /// kills the process.
    pub fn critical<F>(&self, name: impl Into<String>, task: F) -> JoinHandle<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let sup = self.clone();
        let name = name.into();
        tokio::spawn(async move {
            let mut guard = FatalGuard::new(sup, name);
            task.await;
            guard.returned = true;
        })
    }

    /// [`Self::critical`] for a task that runs on a blocking thread — the
    /// motion analyzers, which are CPU-bound decode loops rather than futures.
    pub fn critical_blocking<F>(&self, name: impl Into<String>, task: F) -> JoinHandle<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let sup = self.clone();
        let name = name.into();
        tokio::task::spawn_blocking(move || {
            let mut guard = FatalGuard::new(sup, name);
            task();
            guard.returned = true;
        })
    }

    /// Spawn a periodic task that is put back when it dies, `spawn` building a fresh one each
    /// time.
    pub fn restartable<F, Fut>(
        &self,
        name: impl Into<String>,
        limit: RestartLimit,
        spawn: F,
    ) -> JoinHandle<()>
    where
        F: Fn() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let sup = self.clone();
        let name = name.into();
        tokio::spawn(async move {
            let mut streak = RestartStreak::new(limit);
            loop {
                // tokio's clock, not the system one: a paused test moves this
                // and leaves `std::time::Instant` where it was.
                let started = tokio::time::Instant::now();
                let attempt = tokio::spawn(spawn());
                // Aborting the supervisor must abort the inner task too, or it
                // would be detached and keep running with nobody watching.
                let stop_attempt = AbortOnDrop(attempt.abort_handle());
                let outcome = attempt.await;
                drop(stop_attempt);

                let exit = match outcome {
                    Ok(()) => Exit::Returned,
                    Err(e) if e.is_panic() => Exit::Panicked,
                    Err(_) => Exit::Cancelled,
                };
                if !sup.settle(&name, exit) {
                    return;
                }
                match streak.record(started.elapsed()) {
                    Verdict::Exhausted => {
                        tracing::error!(
                            task = %name,
                            failures = limit.max + 1,
                            healthy_after_secs = limit.healthy_after.as_secs(),
                            "restarting this task is not fixing it: no attempt has stayed up long \
                             enough to do its work"
                        );
                        sup.fatal(&name, exit);
                        return;
                    }
                    Verdict::Restart => {
                        let delay = jittered(streak.backoff());
                        tracing::warn!(
                            task = %name,
                            restart_in_secs = delay.as_secs_f64(),
                            "restarting supervised task"
                        );
                        sup.sleep_or_stop(delay).await;
                        if sup.stopping() {
                            return;
                        }
                    }
                }
            }
        })
    }
}

/// Reports the end of a fatal-policy task from inside the task, whichever way it ends.
/// `returned` is set on the one path that reaches the end of the body; a panic and a
/// cancellation are told apart by [`std::thread::panicking`].
struct FatalGuard {
    sup: Supervisor,
    name: String,
    returned: bool,
}

impl FatalGuard {
    fn new(sup: Supervisor, name: String) -> Self {
        Self {
            sup,
            name,
            returned: false,
        }
    }
}

impl Drop for FatalGuard {
    fn drop(&mut self) {
        let exit = if self.returned {
            Exit::Returned
        } else if std::thread::panicking() {
            Exit::Panicked
        } else {
            Exit::Cancelled
        };
        if self.sup.settle(&self.name, exit) {
            self.sup.fatal(&self.name, exit);
        }
    }
}

/// What an exit means for the process, which is not the same question as how
/// the task left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Meaning {
    /// It happened while camon was running. The policy decides what follows.
    Death,
    /// It landed inside a stop another task's death started, and it panicked —
    /// so it belongs in the report even though the decision is made.
    Aftermath,
    /// It was asked for.
    Expected,
}

/// Aborts a task when the thing that was watching it goes away.
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn supervisor() -> (Supervisor, Arc<AtomicBool>, Arc<AtomicUsize>) {
        let stopping = Arc::new(AtomicBool::new(false));
        let stops = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&stops);
        let flag = Arc::clone(&stopping);
        let sup = Supervisor::new(
            Arc::clone(&stopping),
            Arc::new(tokio::sync::Notify::new()),
            move || {
                counter.fetch_add(1, Ordering::Relaxed);
                flag.store(true, Ordering::Relaxed);
            },
        );
        (sup, stopping, stops)
    }

    #[derive(Clone, Default)]
    struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock_recover().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[tokio::test]
    async fn the_line_an_operator_reads_names_the_task_that_died() {
        let logs = CapturedLog::default();
        let (sup, _, _) = supervisor();
        {
            let _reader = tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .with_writer(logs.clone())
                    .with_max_level(tracing::Level::ERROR)
                    .with_ansi(false)
                    .finish(),
            );
            let _ = sup
                .critical("mqtt-bridge", async { panic!("broker went away") })
                .await;
        }

        let written = String::from_utf8(logs.0.lock_recover().clone()).unwrap();
        let reported = written
            .lines()
            .find(|line| line.contains("panicked"))
            .unwrap_or_else(|| panic!("nothing said a task had panicked: {written}"));
        assert!(reported.contains("mqtt-bridge"), "unnamed: {reported}");
    }

    #[tokio::test]
    async fn a_panicking_task_is_named_and_asks_for_a_stop() {
        let (sup, stopping, stops) = supervisor();
        let handle = sup.critical("cursed-worker", async { panic!("boom") });

        assert!(handle.await.is_err(), "the panic did not reach the handle");
        assert_eq!(sup.deaths(), vec!["cursed-worker (panicked)".to_string()]);
        assert!(
            sup.died().load(Ordering::Relaxed),
            "the watchdog never armed"
        );
        assert!(stopping.load(Ordering::Relaxed), "no drain was asked for");
        assert_eq!(stops.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_task_that_returns_early_is_a_death_too() {
        let (sup, stopping, _) = supervisor();
        sup.critical("quitter", async {}).await.unwrap();

        assert_eq!(sup.deaths(), vec!["quitter (returned)".to_string()]);
        assert!(stopping.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn a_deliberate_stop_is_not_a_death() {
        let (sup, stopping, stops) = supervisor();
        let flag = Arc::clone(&stopping);
        let returning = sup.critical("worker", async move {
            while !flag.load(Ordering::Relaxed) {
                tokio::task::yield_now().await;
            }
        });
        let aborted = sup.critical("aborted-worker", std::future::pending());
        let panicking = sup.critical("late-panicker", async move {
            tokio::task::yield_now().await;
            panic!("on the way out");
        });

        stopping.store(true, Ordering::Relaxed);
        returning.await.unwrap();
        aborted.abort();
        let _ = aborted.await;
        let _ = panicking.await;

        assert!(
            sup.deaths().is_empty(),
            "a clean stop looked like a death: {:?}",
            sup.deaths()
        );
        assert!(!sup.died().load(Ordering::Relaxed));
        assert_eq!(
            stops.load(Ordering::Relaxed),
            0,
            "the supervisor asked for a drain that was already running"
        );
    }

    const CADENCE: Duration = Duration::from_secs(600);

    #[tokio::test(start_paused = true)]
    async fn a_restartable_task_is_put_back_until_the_streak_runs_out() {
        let (sup, stopping, stops) = supervisor();
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let handle = sup.restartable(
            "flaky",
            RestartLimit {
                max: 2,
                healthy_after: CADENCE,
            },
            move || {
                counter.fetch_add(1, Ordering::Relaxed);
                async { panic!("still broken") }
            },
        );

        tokio::time::timeout(CADENCE, handle)
            .await
            .expect("the streak never ran out")
            .unwrap();
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
        assert_eq!(sup.deaths(), vec!["flaky (panicked)".to_string()]);
        assert!(stopping.load(Ordering::Relaxed));
        assert_eq!(stops.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_task_that_fails_at_its_own_cadence_still_escalates() {
        let (sup, stopping, _) = supervisor();
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let handle = sup.restartable(
            "retention",
            RestartLimit::cycling_every(CADENCE),
            move || {
                counter.fetch_add(1, Ordering::Relaxed);
                async move {
                    tokio::time::sleep(CADENCE).await;
                    panic!("the same malformed event, every sweep");
                }
            },
        );

        tokio::time::timeout(CADENCE * (PERIODIC_RESTARTS as u32 + 4), handle)
            .await
            .expect("a task failing every cycle was restarted for ever")
            .unwrap();
        assert_eq!(
            attempts.load(Ordering::Relaxed),
            PERIODIC_RESTARTS + 1,
            "the streak did not escalate at its limit"
        );
        assert_eq!(sup.deaths(), vec!["retention (panicked)".to_string()]);
        assert!(stopping.load(Ordering::Relaxed), "nothing asked for a stop");
    }

    #[tokio::test(start_paused = true)]
    async fn a_long_healthy_attempt_clears_the_streak() {
        let (sup, stopping, _) = supervisor();
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let handle = sup.restartable("anchor", RestartLimit::cycling_every(CADENCE), move || {
            counter.fetch_add(1, Ordering::Relaxed);
            async move {
                tokio::time::sleep(CADENCE * 10).await;
                panic!("once in a blue moon");
            }
        });

        tokio::time::sleep(CADENCE * 10 * (PERIODIC_RESTARTS as u32 + 3)).await;
        assert!(
            attempts.load(Ordering::Relaxed) > PERIODIC_RESTARTS + 1,
            "the task stopped being restarted after {} attempts",
            attempts.load(Ordering::Relaxed)
        );
        assert!(
            !stopping.load(Ordering::Relaxed),
            "a task that works between faults was escalated anyway"
        );
        assert!(sup.deaths().is_empty());
        handle.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn a_task_that_recovers_costs_nothing_but_a_log_line() {
        let (sup, stopping, _) = supervisor();
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let handle = sup.restartable("flaky", RestartLimit::cycling_every(CADENCE), move || {
            let n = counter.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    panic!("one bad sweep");
                }
                std::future::pending::<()>().await;
            }
        });

        tokio::time::sleep(Duration::from_secs(60)).await;
        assert_eq!(attempts.load(Ordering::Relaxed), 2, "it was not restarted");
        assert!(
            !stopping.load(Ordering::Relaxed),
            "one failure killed camon"
        );
        assert!(sup.deaths().is_empty());
        handle.abort();
    }

    #[tokio::test]
    async fn a_cascade_names_the_origin_even_when_it_arrives_second() {
        let (sup, _, stops) = supervisor();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);

        let victim = sup.critical("detection-worker", async move {
            let _ = rx.recv().await;
        });
        drop(tx);
        victim.await.unwrap();

        let origin = sup.critical("analyzer:yard", async { panic!("decoder blew up") });
        let _ = origin.await;

        assert_eq!(
            sup.deaths(),
            vec![
                "detection-worker (returned)".to_string(),
                "analyzer:yard (panicked)".to_string(),
            ],
            "the cascade lost an end of itself"
        );
        assert_eq!(
            sup.first_failure().map(|d| d.task).as_deref(),
            Some("detection-worker"),
            "the decision is first-wins"
        );
        assert_eq!(
            stops.load(Ordering::Relaxed),
            1,
            "the second death started a second drain"
        );
    }

    #[tokio::test]
    async fn a_panic_during_a_deliberate_stop_is_not_recorded() {
        let (sup, stopping, _) = supervisor();
        stopping.store(true, Ordering::Relaxed); // a SIGTERM, just now
        let _ = sup
            .critical("warm-writer:yard", async { panic!("on the way out") })
            .await;

        assert!(
            sup.deaths().is_empty(),
            "a SIGTERM shutdown was reported as a failure: {:?}",
            sup.deaths()
        );
    }

    #[test]
    fn the_report_is_bounded_and_admits_what_it_left_out() {
        let mut report = Report::default();
        for n in 0..MAX_RECORDED_DEATHS + 3 {
            report.record(&format!("task-{n}"), Exit::Panicked);
        }
        let lines = report.lines();
        assert_eq!(lines.len(), MAX_RECORDED_DEATHS + 1);
        assert_eq!(lines[0], "task-0 (panicked)");
        assert_eq!(lines[MAX_RECORDED_DEATHS], "and 3 more");
    }

    #[test]
    fn the_classification_table_reads_both_facts_together() {
        let running = State {
            stopping: false,
            death_started_it: false,
        };
        let stopping_deliberately = State {
            stopping: true,
            death_started_it: false,
        };
        let stopping_over_a_death = State {
            stopping: true,
            death_started_it: true,
        };

        for exit in [Exit::Returned, Exit::Panicked, Exit::Cancelled] {
            assert_eq!(meaning(running, exit), Meaning::Death, "{exit:?}");
            assert_eq!(
                meaning(stopping_deliberately, exit),
                Meaning::Expected,
                "{exit:?}"
            );
        }

        assert_eq!(
            meaning(stopping_over_a_death, Exit::Panicked),
            Meaning::Aftermath
        );
        assert_eq!(
            meaning(stopping_over_a_death, Exit::Returned),
            Meaning::Expected
        );
        assert_eq!(
            meaning(stopping_over_a_death, Exit::Cancelled),
            Meaning::Expected
        );
    }

    #[tokio::test]
    async fn the_report_is_written_before_the_stop_is_asked_for() {
        let stopping = Arc::new(AtomicBool::new(false));
        let supervisor_slot: Arc<std::sync::OnceLock<Supervisor>> = Arc::default();
        let seen = Arc::new(Mutex::new((Vec::new(), false)));

        let sup = Supervisor::new(
            Arc::clone(&stopping),
            Arc::new(tokio::sync::Notify::new()),
            {
                let slot = Arc::clone(&supervisor_slot);
                let seen = Arc::clone(&seen);
                let flag = Arc::clone(&stopping);
                move || {
                    let sup = slot.get().expect("supervisor wired");
                    *seen.lock_recover() = (sup.deaths(), sup.died().load(Ordering::Relaxed));
                    flag.store(true, Ordering::Relaxed);
                }
            },
        );
        let _ = supervisor_slot.set(sup.clone());

        let _ = sup
            .critical("warm-writer:yard", async { panic!("boom") })
            .await;

        let (report, armed) = seen.lock_recover().clone();
        assert_eq!(
            report,
            vec!["warm-writer:yard (panicked)".to_string()],
            "the stop flag went up before the report that explains it"
        );
        assert!(armed, "the stop flag went up before the drain was armed");
    }

    #[tokio::test]
    async fn two_deaths_that_race_the_flag_still_ask_for_one_drain() {
        let stops = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&stops);
        let sup = Supervisor::new(
            Arc::new(AtomicBool::new(false)),
            Arc::new(tokio::sync::Notify::new()),
            move || {
                counter.fetch_add(1, Ordering::Relaxed);
            },
        );

        let _ = sup
            .critical("analyzer:yard", async { panic!("first") })
            .await;
        let _ = sup
            .critical("warm-writer:yard", async { panic!("second") })
            .await;

        assert_eq!(
            sup.deaths(),
            vec![
                "analyzer:yard (panicked)".to_string(),
                "warm-writer:yard (panicked)".to_string(),
            ],
            "a death that lost the race was dropped from the report"
        );
        assert_eq!(
            stops.load(Ordering::Relaxed),
            1,
            "the second death asked for a second drain"
        );
        assert!(sup.died().load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn aborting_the_supervisor_stops_the_task_it_supervises() {
        let (sup, _, _) = supervisor();
        let running = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&running);
        let handle = sup.restartable("endless", RestartLimit::cycling_every(CADENCE), move || {
            let flag = Arc::clone(&flag);
            async move {
                flag.store(true, Ordering::Relaxed);
                let _guard = ClearOnDrop(Arc::clone(&flag));
                std::future::pending::<()>().await;
            }
        });
        while !running.load(Ordering::Relaxed) {
            tokio::task::yield_now().await;
        }

        handle.abort();
        let _ = handle.await;
        for _ in 0..100 {
            if !running.load(Ordering::Relaxed) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the supervised task outlived its supervisor");
    }

    struct ClearOnDrop(Arc<AtomicBool>);

    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Relaxed);
        }
    }

    #[test]
    fn only_a_healthy_attempt_clears_the_streak() {
        let limit = RestartLimit {
            max: 2,
            healthy_after: Duration::from_secs(60),
        };
        let died_at_once = Duration::from_secs(1);

        let mut streak = RestartStreak::new(limit);
        assert_eq!(streak.record(died_at_once), Verdict::Restart);
        assert_eq!(streak.record(died_at_once), Verdict::Restart);
        assert_eq!(streak.record(died_at_once), Verdict::Exhausted);

        let mut healthy = RestartStreak::new(limit);
        for _ in 0..10 {
            assert_eq!(
                healthy.record(Duration::from_secs(60)),
                Verdict::Restart,
                "a task that works between faults spent its streak"
            );
        }

        let mut recovering = RestartStreak::new(limit);
        assert_eq!(recovering.record(died_at_once), Verdict::Restart);
        assert_eq!(recovering.record(died_at_once), Verdict::Restart);
        assert_eq!(
            recovering.record(Duration::from_secs(600)),
            Verdict::Restart
        );
        assert_eq!(recovering.record(died_at_once), Verdict::Restart);
    }

    #[test]
    fn one_cadence_of_uptime_is_not_healthy() {
        let cadence = Duration::from_secs(3600);
        let limit = RestartLimit::cycling_every(cadence);
        let mut streak = RestartStreak::new(limit);
        for expected in [Verdict::Restart, Verdict::Restart, Verdict::Restart] {
            assert_eq!(streak.record(cadence), expected);
        }
        assert_eq!(
            streak.record(cadence),
            Verdict::Exhausted,
            "a task dying on its first tick every cycle restarted for ever"
        );
    }

    #[test]
    fn the_backoff_grows_with_the_streak() {
        let mut streak = RestartStreak::new(RestartLimit::cycling_every(CADENCE));
        streak.record(Duration::ZERO);
        assert_eq!(streak.backoff(), RESTART_BACKOFF.start);
        streak.record(Duration::ZERO);
        assert_eq!(
            streak.backoff(),
            RESTART_BACKOFF.next(RESTART_BACKOFF.start)
        );
    }

    #[test]
    fn the_backoff_cap_is_the_last_delay_the_streak_can_reach() {
        let mut streak = RestartStreak::new(RestartLimit::cycling_every(CADENCE));
        let mut last = Duration::ZERO;
        for _ in 0..PERIODIC_RESTARTS {
            streak.record(Duration::ZERO);
            last = streak.backoff();
        }
        assert_eq!(
            last, RESTART_BACKOFF.max,
            "the backoff cap and the restart limit have drifted apart: {PERIODIC_RESTARTS} \
             restarts reach {last:?} against a cap of {:?}",
            RESTART_BACKOFF.max
        );
    }
}
