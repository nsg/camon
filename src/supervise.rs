//! Who notices when a long-lived task stops, and what the process does about
//! it.
//!
//! Every worker camon depends on used to be spawned and forgotten. A task that
//! panicked took its own job with it and nothing else: the camera whose
//! analyzer died went on filling a hot buffer nobody read, the HTTP server
//! whose bind failed logged one line inside a detached task, and the process
//! stayed up and healthy-looking for as long as the box did. That is the worst
//! shape a fault can take here, because the only recovery either deployment
//! has is a process that *dies*: `Restart=always` in the systemd unit camon
//! writes, and — for the Home Assistant add-on — the Supervisor, which restarts
//! a crashed add-on only when its per-add-on **Watchdog** toggle is on (it is
//! off by default, so an add-on user who wants crash recovery has to turn it on;
//! without it a dead camon stays dead until someone presses start). Either way,
//! staying alive is what prevented the one thing that could have healed the
//! deployment.
//!
//! So every long-lived task is spawned through a [`Supervisor`], which watches
//! it and applies one of two policies:
//!
//! * **Fatal** ([`Supervisor::critical`]) — the task is one the process cannot
//!   do its job without, and it cannot be put back on its own. Its death asks
//!   for the same graceful drain a SIGTERM does, so the footage in flight still
//!   reaches disk (see [`crate::shutdown`]), and the process then exits
//!   nonzero. The service manager restarts it, which rebuilds the whole task
//!   graph from what is on disk — the only recovery that actually works for
//!   these.
//! * **Restart** ([`Supervisor::restartable`]) — the task is periodic and holds
//!   nothing that cannot be reconstructed, so a transient fault in it is not
//!   worth an outage. It is spawned again after a backoff, up to
//!   [`RestartLimit::max`] times *in a row*; the failure after that escalates to
//!   the fatal policy, because a task that will not stay up is a fault the
//!   process cannot fix by trying harder.
//!
//!   Escalating costs something real — a process restart interrupts every
//!   camera for the seconds it takes to drain and come back, while an
//!   in-process restart interrupts nothing else — and it is still right. A
//!   streak with no healthy attempt in it means that task has not done its job
//!   once since the trouble began, and the interruption is bounded and clean:
//!   the M2 drain lands the footage in flight, and the service manager has the
//!   process back in seconds with every one of its tasks rebuilt.
//!
//! # Died, or asked to stop?
//!
//! A drain stops every one of these tasks on purpose, so "the task exited" on
//! its own means nothing. What separates the two is the stop flag: it is raised
//! before anything is asked to finish and it is never lowered, so an exit
//! observed while it is up was asked for, and an exit observed while it is down
//! was not. That is the whole rule for what *decides* anything — not the exit
//! kind, not who joined the handle — which is why a deliberate stop cannot trip
//! the supervisor however the task chose to leave: returning, being aborted, or
//! panicking on its way out.
//!
//! Before the flag, every way out is a death — a panic, obviously, but a clean
//! return too. None of the supervised tasks has a legitimate reason to finish
//! while camon is meant to be running: each is a loop that exits on the stop
//! flag, or a server that only returns on error. A task that returns early is a
//! task that has stopped doing its job, and the process is no more able to
//! notice that later than it was before.
//!
//! # Which task actually failed
//!
//! One fault reaches two guards. An analyzer that panics drops its locals as it
//! unwinds — including the queue sender the detection worker is parked on — so
//! the worker can be woken, see its channel closed and report a clean early
//! return *before* the analyzer's own guard has finished unwinding. The victim
//! gets there first and the origin arrives into a stop that is already under
//! way. No ordering fix is available: the wrapping guard is a local declared
//! before the task it wraps, so it is dropped last by construction.
//!
//! So the decision stays first-wins — one drain, however many tasks fall over —
//! but the *report* is a list. Every flag-down death is recorded, and so is a
//! panic that lands inside a stop another task's death started, which is
//! precisely the origin arriving late. Both names reach
//! [`crate::app::RunError`] and the exit line in arrival order, and the operator
//! reads the whole cascade instead of whichever end of it happened to win a
//! race. A panic during a *deliberate* stop is not recorded: nothing about a
//! SIGTERM asks to be explained, and it must not change the exit status.
//!
//! Telling those apart is a question about two facts at once — whether the
//! process is stopping, and whether a death started that stop — so it is asked
//! once, under one lock, against a state read all at once. [`Report`] is where
//! that lock and the invariants it keeps are written down.
//!
//! # How the exit is seen
//!
//! For the fatal policy the report comes from a guard dropped inside the task
//! itself, not from awaiting its handle. That matters twice over: the handle
//! stays the caller's, so the phased drain in [`crate::app`] still owns the
//! joins, the abort of the detection worker still releases its senders exactly
//! when it used to, and a task abandoned at a phase bound is still abandoned
//! rather than aborted. And the fault is noticed *when it happens* rather than
//! at the stop — the panic that used to be discovered by `let _ = handle.await`
//! minutes or days later, if anyone read the log at all.
//!
//! A guard cannot see the panic payload, so the panic's own message is left to
//! the default panic hook, which has already printed it to stderr by the time
//! the guard runs. What the supervisor adds is the one thing the hook cannot
//! know: which task it was.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::locks::MutexExt;
use crate::retry::{jittered, RetrySchedule};

/// How a supervised task left. Distinguished for the report; the policy treats
/// all three the same, because before the stop flag every one of them is a task
/// that has stopped doing its job.
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
///
/// Three, because the point of a restart is to absorb a fault that is not
/// coming back, and three attempts is as much evidence of that as a fourth
/// would be. A task crashing on sight spends them in seconds — the backoff
/// below is what keeps those seconds from being a busy loop — while a task
/// that fails once and then works is never anywhere near them.
pub const PERIODIC_RESTARTS: usize = 3;

/// When a restartable task gives up on itself: how many failures in a row it is
/// allowed, and how long an attempt has to survive to prove that the streak is
/// over.
///
/// The streak is what makes the escalation reachable at all. Counting failures
/// inside a fixed time window — the first shape of this — could not: every
/// restart rebuilds the task's schedule, so a sweep that panics on every sweep
/// fails an hour apart by construction, and an update check that panics on every
/// check fails twelve hours apart. Any window wide enough to catch them would
/// also punish a task that had failed once a day for a fortnight and worked
/// perfectly in between. A streak asks the question that actually distinguishes
/// the two: *has any attempt worked since the trouble started?*
#[derive(Debug, Clone, Copy)]
pub struct RestartLimit {
    /// Consecutive failures allowed. The next one escalates.
    pub max: usize,
    /// The uptime that clears the streak. An attempt that ran at least this
    /// long before it died has proved itself and starts the count over.
    pub healthy_after: Duration,
}

impl RestartLimit {
    /// The limit for a task that does its work once every `cadence` — the
    /// retention sweep's hour, the anchor's minute, the updater's twelve hours.
    ///
    /// An attempt has to outlive *two* cadences to count as healthy, and the
    /// second one is the whole point: one would be met by a task that dies on
    /// its very first tick every single time, which is exactly the task the
    /// streak has to be able to escalate. Two means it did the work at least
    /// once after the tick that killed its predecessor.
    ///
    /// Taken from the task's own interval constant at the call site rather than
    /// picked here, so a cadence that changes cannot leave this behind.
    pub fn cycling_every(cadence: Duration) -> Self {
        Self {
            max: PERIODIC_RESTARTS,
            healthy_after: cadence.saturating_mul(2),
        }
    }
}

/// Backoff between restarts of the same task, so a task that panics on sight
/// cannot spin a core through its whole streak.
///
/// The cap is the delay the *last* allowed restart reaches and not a second
/// more — with three restarts the doubling runs 1s, 2s, 4s — because a ceiling
/// the streak can never touch is a number that documents nothing and quietly
/// stops matching the policy beside it. Raising [`PERIODIC_RESTARTS`] means
/// raising this too, which
/// `the_backoff_cap_is_the_last_delay_the_streak_can_reach` insists on.
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

/// One task's run of failures: how many attempts have died in a row without one
/// of them proving itself, and what the next death means.
///
/// Pure with respect to the attempt's uptime, like
/// [`crate::shutdown::DrainGate`] is with respect to `now`, so every corner of
/// the escalation can be pinned without a clock at all.
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
    /// clears whatever streak preceded it: it worked, so the trouble that came
    /// before is over and this is the start of a new one.
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
    /// failure in the current streak. Un-jittered — the caller adds that — so
    /// the schedule can be read straight out of the constants.
    pub fn backoff(&self) -> Duration {
        let mut delay = RESTART_BACKOFF.start;
        for _ in 1..self.failures {
            delay = RESTART_BACKOFF.next(delay);
        }
        delay
    }
}

/// How many deaths are kept for the report. One fault can reach a handful of
/// guards — a panicking analyzer takes its detection worker with it — and the
/// operator needs the cascade, not an unbounded log of a process that is on its
/// way out anyway. Anything past this is counted and not named.
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
    /// A copy of "a death started this stop" for the drain's last-resort
    /// watchdog thread, which polls rather than locks. Written inside the
    /// critical section below, beside the entry that justifies it, so it can
    /// never be true with an empty report or false with a full one.
    died: Arc<AtomicBool>,
    /// The one thing every decision here is made against. See [`Report`].
    report: Mutex<Report>,
}

/// What has died so far, and the only place the answer is kept.
///
/// # Why one lock
///
/// Every deciding read happens inside this mutex, because the question
/// "is this exit a death, an aftermath, or asked for?" is a question about two
/// facts at once — whether the process is stopping, and whether a death is what
/// started that stop — and reading them separately can observe an order that
/// never happened. Two relaxed atomics allowed exactly that: a thread could see
/// the stop flag already up and the death that raised it not yet recorded, and
/// classify a genuine cascade origin as "asked for", which drops it from the
/// report and loses the only line naming the real culprit.
///
/// So the state a classification needs is taken as one [`State`] snapshot, and
/// `State` can only be built from a locked `Report` — classifying outside the
/// lock is not something the types let you write.
///
/// # The invariants
///
/// * **A death is in the report before the stop it asks for is visible.**
///   [`Supervisor::fatal`] appends and publishes `died` inside the critical
///   section, and asks for the drain only after releasing it. So any thread
///   that sees the stop flag up and then takes this lock sees the death that
///   raised it, and the misclassification above is unreachable rather than
///   unlikely.
/// * **First-wins is decided here, not raced.** Whether a death is the one that
///   asks for the drain is "was the report empty when I appended?", answered
///   under the same lock, so two tasks dying together still start one drain.
/// * **A non-empty report means `died`, which means exit 1.** The first two are
///   written together under this lock; the third is
///   [`crate::app::stop_outcome`], which returns `Err` for any non-empty
///   report. An operator who is shown a task name always gets a nonzero status
///   with it.
///
/// The stop flag itself is still raised outside this lock — a SIGTERM does not
/// ask permission — so a classification can race *it* and call a death a death
/// when a signal was landing anyway. That direction is harmless: the exit is
/// recorded and explained rather than silently dropped, and it is the same
/// narrow race the exit status already documents.
#[derive(Default)]
struct Report {
    /// Every death worth attributing, in arrival order. The first decided the
    /// stop; the rest are there because the first is not reliably the origin
    /// (see the module header).
    deaths: Vec<Death>,
    /// Deaths past [`MAX_RECORDED_DEATHS`], so the report can admit there were
    /// more rather than quietly truncating.
    unnamed: usize,
}

impl Report {
    /// The snapshot a classification is made from. Private constructor by
    /// virtue of taking `&self`: there is no way to reach one without holding
    /// the lock.
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
    /// Whether a supervised death is what started that stop, which is the same
    /// question as whether anything has been reported.
    death_started_it: bool,
}

/// What an exit means, from a state that was read all at once.
///
/// Pure and total, so the whole table is pinned by
/// `the_classification_table_reads_both_facts_together` rather than inferred
/// from the two or three interleavings a test can actually produce.
fn meaning(state: State, exit: Exit) -> Meaning {
    if !state.stopping {
        return Meaning::Death;
    }
    // A stop that a death started is still being explained, and a panic is
    // never a way of being asked to stop — so this is the cascade arriving, and
    // it is the half most likely to name the real culprit. A panic inside a
    // *deliberate* stop is not this: nothing about a SIGTERM needs explaining,
    // and recording one would turn a clean stop nonzero.
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
    /// asks for has begun. The drain's watchdog thread polls this to know that
    /// nothing outside the process is bounding this stop, and that the exit
    /// status owes the operator a failure.
    pub fn died(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.inner.died)
    }

    /// The death that decided the stop, if a death did.
    ///
    /// First-wins, and deliberately: the drain is asked for once however many
    /// tasks fall over. It is *not* a claim about which task failed first —
    /// the origin of a cascade routinely arrives second (see the module
    /// header), which is why [`Self::deaths`] and not this one is what the
    /// operator is shown.
    pub fn first_failure(&self) -> Option<Death> {
        self.inner.report.lock_recover().deaths.first().cloned()
    }

    /// Every death worth attributing, in arrival order, with a trailing note
    /// when there were more than [`MAX_RECORDED_DEATHS`]. Empty when the
    /// process is stopping for a reason of its own.
    pub fn deaths(&self) -> Vec<String> {
        self.inner.report.lock_recover().lines()
    }

    /// Report an exit, attribute it if it is worth attributing, and say whether
    /// the policy still has a decision to make about it.
    ///
    /// The classification and the recording that follows from it are one
    /// critical section, for the reason written on [`Report`]: read apart, the
    /// two facts behind the decision can describe a moment that never existed.
    /// The logging is done after the lock is released — nothing that formats a
    /// line should be able to hold up a task that is trying to die.
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

    /// Accept a task's death as the reason the process is stopping: attribute
    /// it, arm the drain's watchdog, and — if it is the first — ask for the
    /// drain itself.
    ///
    /// The order is the invariant, not an implementation detail: the report and
    /// `died` are written under the lock, and the stop is asked for only once
    /// that lock is gone. Nothing can therefore observe the stop without also
    /// being able to see what caused it, and a second task dying in the same
    /// instant lands after the first in the report instead of starting a second
    /// drain. Releasing the lock first matters twice over — `request_stop` is
    /// the caller's code, and it must not be able to reach back in.
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

    /// Sleep, cut short by a stop. Same shape as the camera reconnect's wait and
    /// for the same reason: a task parked here when the drain starts must not
    /// spend the drain's budget finishing its nap.
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

    /// Spawn a task the process cannot run without: any exit before the stop
    /// flag drains and kills the process.
    ///
    /// The handle is the task's own, so the caller keeps every join, abort and
    /// abandonment it had before.
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

    /// Spawn a periodic task that is put back when it dies, `spawn` building a
    /// fresh one each time.
    ///
    /// The returned handle belongs to the supervising loop rather than to the
    /// task itself, so it stays valid across restarts and can be joined or
    /// aborted like any other — aborting it stops whichever attempt is running.
    /// A restarted task starts from scratch: its schedule begins again, so the
    /// first tick after a restart is one whole cadence away. That is what
    /// [`RestartLimit::cycling_every`] is measured against — an attempt that
    /// dies on its first tick has an uptime of about one cadence, and the
    /// healthy threshold is two.
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
                // and leaves `std::time::Instant` where it was, which would
                // make every attempt in such a test look instantaneous.
                let started = tokio::time::Instant::now();
                let attempt = tokio::spawn(spawn());
                // Abandoning the supervisor must abandon what it is supervising
                // too: without this the inner task would be detached by the
                // handle's drop and go on running with nobody watching it.
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

/// Reports the end of a fatal-policy task from inside the task, whichever way
/// it ends.
///
/// `returned` is set on the one path that reaches the end of the body, so
/// anything else is a panic or a cancellation — told apart by
/// [`std::thread::panicking`], which is true on this thread exactly while the
/// unwind that is dropping this guard runs.
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

    /// A supervisor over a fresh stop flag, plus the flag and a count of the
    /// drains it asked for.
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

    /// Somewhere for a subscriber to write, so a test can read what an operator
    /// would have seen.
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

    /// The log is the whole interface here. A panic's own message comes from
    /// the panic hook and says nothing about camon; the line that says *which
    /// task* it was is this module's, and an operator with a stack of workers
    /// has nothing else to go on.
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
        // On the line that reports the panic, not merely somewhere in the log:
        // the two are one line apart here and pages apart on a busy box.
        let reported = written
            .lines()
            .find(|line| line.contains("panicked"))
            .unwrap_or_else(|| panic!("nothing said a task had panicked: {written}"));
        assert!(reported.contains("mqtt-bridge"), "unnamed: {reported}");
    }

    /// The failure this whole module exists for: a task that dies while camon
    /// is running is noticed there and then, named, and turned into a stop —
    /// rather than being discovered by whoever eventually joins its handle.
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

    /// A task that returns early has stopped doing its job just as completely
    /// as one that panicked; none of the supervised tasks has a reason to
    /// finish while camon is meant to be running.
    #[tokio::test]
    async fn a_task_that_returns_early_is_a_death_too() {
        let (sup, stopping, _) = supervisor();
        sup.critical("quitter", async {}).await.unwrap();

        assert_eq!(sup.deaths(), vec!["quitter (returned)".to_string()]);
        assert!(stopping.load(Ordering::Relaxed));
    }

    /// The false positive that would make supervision unusable: every one of
    /// these tasks is stopped on purpose by the drain, and none of those exits
    /// may read as a death — not a clean return, not an abort, not even a panic
    /// on the way out. Nothing may ask for a second drain either.
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

    /// A cadence a paused clock can step through, standing in for the sweep's
    /// hour and the updater's twelve.
    const CADENCE: Duration = Duration::from_secs(600);

    /// The restart policy, end to end: the task is put back on its feet, and
    /// the failure past the streak escalates to the fatal one instead of
    /// restarting for ever.
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

        // Bounded: a streak that never escalates restarts for ever, and an
        // unexplained CI hang is a bad way to find that out.
        tokio::time::timeout(CADENCE, handle)
            .await
            .expect("the streak never ran out")
            .unwrap();
        // The first attempt plus its two restarts; the third failure escalated.
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
        assert_eq!(sup.deaths(), vec!["flaky (panicked)".to_string()]);
        assert!(stopping.load(Ordering::Relaxed));
        assert_eq!(stops.load(Ordering::Relaxed), 1);
    }

    /// The case the first design of this could never see, and the reason it was
    /// replaced: a task that fails *at its own cadence* — the retention sweep
    /// that panics on every sweep, the update check that panics on every check.
    /// Counting failures inside a fixed window could never catch it, because
    /// every restart rebuilds the schedule and the failures are therefore a
    /// whole cadence apart by construction. A streak escalates it on the fourth
    /// failure. For a task whose first tick is a full cadence out — the
    /// retention sweep, whose schedule a restart rebuilds — that is about four
    /// cadences in: roughly four hours, and around two days for the
    /// twelve-hourly update check. The watchdog and the anchor tick
    /// immediately on start, so a panic on every check escalates after just
    /// the backoffs — seconds, not minutes.
    ///
    /// Those are the real numbers and they are the right ones. Nothing is
    /// hidden in the meantime — every failure logs an error naming the task,
    /// from the first — and the escalation is the last resort rather than the
    /// alarm: taking a recording NVR down deserves several attempts' worth of
    /// evidence, and the rule stays one rule for four tasks whose cadences
    /// span a minute to half a day. A slow escalation on the updater is
    /// the cost of not having a special case there, and a camon running an old
    /// binary for another day is not what the process restart was for anyway.
    ///
    /// Paused clock: this spends four full cadences of virtual time.
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
                    // Exactly what a real sweep does: wait out its interval, then
                    // fail on the work.
                    tokio::time::sleep(CADENCE).await;
                    panic!("the same malformed event, every sweep");
                }
            },
        );

        // Bounded for the same reason, and generously: this is meant to take
        // four cadences of virtual time.
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

    /// The other half of that rule: an attempt that ran long enough to have
    /// done its work several times over has proved the trouble is over, so its
    /// eventual death starts a new streak rather than continuing an old one. A
    /// task that fails once a fortnight must never accumulate its way into
    /// taking the process down.
    #[tokio::test(start_paused = true)]
    async fn a_long_healthy_attempt_clears_the_streak() {
        let (sup, stopping, _) = supervisor();
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let handle = sup.restartable("anchor", RestartLimit::cycling_every(CADENCE), move || {
            counter.fetch_add(1, Ordering::Relaxed);
            async move {
                // Healthy for many cadences, then a fault — the shape of a
                // task that works and occasionally trips over something.
                tokio::time::sleep(CADENCE * 10).await;
                panic!("once in a blue moon");
            }
        });

        // Long enough for far more failures than the streak allows, if any of
        // them had been allowed to count together.
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

    /// A restart is a restart, not a retry loop that gives up: a task that dies
    /// once and then stays up keeps the process alive and running.
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

    /// One fault, two guards, and the origin is the one that arrives late: a
    /// panicking analyzer drops the queue sender its detection worker is parked
    /// on while it is still unwinding, so the worker can report a clean exit
    /// first and raise the stop flag. The decision is first-wins and stays
    /// there — one drain — but the report has to carry both, or the operator is
    /// handed the victim and told nothing about the cause.
    ///
    /// The two halves are sequenced here rather than raced, because the race is
    /// the thing that cannot be relied on: whichever order it lands in, both
    /// names must come out.
    #[tokio::test]
    async fn a_cascade_names_the_origin_even_when_it_arrives_second() {
        let (sup, _, stops) = supervisor();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);

        // The victim: it ends because its producer's sender went away.
        let victim = sup.critical("detection-worker", async move {
            let _ = rx.recv().await;
        });
        drop(tx);
        victim.await.unwrap();

        // The origin, still unwinding when the victim got there, arriving into
        // the stop its own panic caused.
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

    /// The other side of that rule: a panic during a stop nobody's death
    /// started explains nothing and must not turn a clean stop into a failed
    /// one. It is still logged.
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

    /// The report is bounded — one fault can reach every guard there is — and
    /// says so rather than truncating quietly.
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

    /// The whole classification, from states that were read all at once.
    ///
    /// Written as a table because the interleavings a test can actually produce
    /// are a fraction of the ones a running camon produces, and the bug this
    /// replaced was in a state no test had reached: stop flag up, death not yet
    /// visible, so a real cascade origin read as "asked for" and vanished from
    /// the report. The types make that state unreachable now — a [`State`] can
    /// only come from a locked [`Report`] — and this pins what each of them
    /// means.
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
            // Nothing has asked anything to stop: every way out is a death.
            assert_eq!(meaning(running, exit), Meaning::Death, "{exit:?}");
            // A stop nobody's death started explains itself.
            assert_eq!(
                meaning(stopping_deliberately, exit),
                Meaning::Expected,
                "{exit:?}"
            );
        }

        // Inside a stop a death started, a panic is the cascade still arriving
        // and belongs in the report; the orderly exits around it are the drain
        // doing its job.
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

    /// The ordering the classification rests on: by the time anything can see
    /// the stop flag, the report already says why it went up.
    ///
    /// The probe runs *as* the stop is asked for, which is the earliest moment
    /// any other thread could observe it, and reads the report from there. It
    /// also pins that the lock is not held across that call: a supervisor that
    /// asked for the drain with its own mutex held would deadlock here rather
    /// than quietly inviting the caller to reach back in.
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
        // The watchdog thread reads `died` without taking the lock, so it has
        // to be published with the entry rather than after the stop.
        assert!(armed, "the stop flag went up before the drain was armed");
    }

    /// Two tasks dying before either has managed to raise the flag — the
    /// interleaving a stop flag set outside the lock cannot rule out. The
    /// stand-in below never raises it, which freezes that window open: both
    /// exits classify as deaths, both are reported, and exactly one drain is
    /// asked for, because "am I the first?" is answered under the same lock
    /// that appends.
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

    /// Aborting the supervising loop has to abort whatever it is supervising:
    /// a detached inner task would go on holding whatever it holds — a
    /// warm-writer sender, a store lock — with nobody left to stop it.
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
        // The abort reaches the inner task through the supervising loop's own
        // cancellation, so it takes a poll or two to land.
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

    /// The rule the whole escalation rests on, in isolation: only an attempt
    /// that proved itself clears the count, and how far apart the failures fell
    /// has nothing to do with it.
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

        // The same three failures, each after an attempt that had been up long
        // enough to do its work. Nothing accumulates.
        let mut healthy = RestartStreak::new(limit);
        for _ in 0..10 {
            assert_eq!(
                healthy.record(Duration::from_secs(60)),
                Verdict::Restart,
                "a task that works between faults spent its streak"
            );
        }

        // A single healthy attempt in the middle is enough to start over.
        let mut recovering = RestartStreak::new(limit);
        assert_eq!(recovering.record(died_at_once), Verdict::Restart);
        assert_eq!(recovering.record(died_at_once), Verdict::Restart);
        assert_eq!(
            recovering.record(Duration::from_secs(600)),
            Verdict::Restart
        );
        assert_eq!(recovering.record(died_at_once), Verdict::Restart);
    }

    /// The healthy threshold has to be more than one of the task's own cycles,
    /// or the task it exists to catch — the one that dies on its first tick,
    /// every time — would clear its own streak with an uptime of exactly one
    /// cadence and be restarted for ever.
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

    /// The backoff exists so a task that panics the instant it starts cannot
    /// spin a core through its whole streak. It grows with the streak.
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

    /// A cap the streak could never reach would document nothing and would go
    /// on documenting nothing after someone raised the restart limit. It is the
    /// delay of the last restart the streak allows, exactly.
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
