//! The phased stop: how camon puts a running recording down without cutting
//! the tail off it.
//!
//! One flag used to do the whole job, and that is what lost footage. Producers
//! and consumers polled the same boolean, so the camera thread was still
//! pushing the GOP it had in hand while the analyzer that was supposed to write
//! that GOP had already flushed its open run and exited. Every stop — systemd,
//! the Home Assistant supervisor, a host shutdown, the self-updater's restart —
//! dropped the last seconds of every recording in progress. Reordering the
//! joins does not help: the analyzer was never waiting for the camera, it was
//! racing it, and the race is decided by whichever polls the flag first.
//!
//! So the stop runs in phases, and each one finishes before the next begins:
//!
//! 1. **The producers stop.** The camera threads are told to stop and joined.
//!    Nothing else is asked to finish until they have.
//! 2. **Each camera publishes a terminal watermark** — [`HotBuffer::seal`], the
//!    last sequence that camera will ever produce — and the consumers that read
//!    from it, the analyzers and the continuous recorders, keep working until
//!    they have consumed through that watermark. Only then do they flush and
//!    exit.
//! 3. **The writers drain.** The event channels close and the warm writers
//!    write out everything they were handed, exactly as before.
//!
//! Every phase is bounded, because the second time each of these goes wrong is
//! not a lost tail but an NVR that never stops: a camera thread wedged in a
//! read, a consumer stalled behind a dead decoder, a watermark that is never
//! published because the thread that would publish it is the one that hung. A
//! bound that trips says what it abandoned and lets the next phase start.
//!
//! # The budget
//!
//! Both deployments give the drain 360 seconds — `TimeoutStopSec` in the
//! systemd unit `camon install service` writes, `timeout: 360` in the add-on's
//! `config.yaml`, and [`crate::app::RESTART_DRAIN_DEADLINE`] for the drain the
//! updater starts, which no service manager is watching. That budget is shared
//! with the writers, and a single event bound for a remote stathost can sit on
//! one [`UPLOAD_TIMEOUT`](crate::storage::stathost::UPLOAD_TIMEOUT) of 300
//! seconds. So the phases above spend as little of it as they can and the rest
//! is left to phase 3:
//!
//! ```text
//!   phase 1   camera joins            CAMERA_JOIN_BOUND      10s
//!   phase 2   consumers drain         CONSUMER_JOIN_BOUND    35s
//!             mqtt offline marker     MQTT_SHUTDOWN_TIMEOUT   5s
//!   phase 3   writers, then the       the remainder         300s
//!             retention sweep
//!             handing over            TEARDOWN_MARGIN        10s
//!                                                         ------
//!                                                           360s
//! ```
//!
//! Three hundred seconds is exactly one `UPLOAD_TIMEOUT` and not a second more:
//! the worst case leaves room for one upload that uses its whole timeout, and
//! anything queued behind it is abandoned. That is not a margin, it is the
//! floor, and it is why every one of these constants is pinned by a test
//! against the real budget — widening any of them takes the writers below a
//! single upload, and that has to be a failing build rather than a discovery
//! made during an incident.
//!
//! The worst case is also not the normal one. Phase 3 is deliberately given no
//! constant of its own: it is handed a deadline measured from the start of the
//! drain, so the seconds phases 1 and 2 do not spend — in a healthy stop they
//! spend well under one between them — stay with the uploads instead of being
//! forfeited to a ceiling picked in advance.
//!
//! The scheduled retention sweep is joined inside phase 3's deadline, sharing
//! it with the writers, rather than ahead of the phases where its wait used to
//! sit. A remote sweep can be parked on a request timeout of its own, and 60
//! seconds spent waiting for one before phase 1 had even begun came out of
//! every consumer's [`TAIL_DRAIN_BOUND`] — which is measured from the stop
//! flag — so the cameras were still being joined when the consumers' gates
//! expired. See `app::graceful_shutdown` for why moving the wait is safe.

use std::time::{Duration, Instant};

/// Phase 1: how long the whole drain waits for the camera threads to stop.
///
/// A camera thread is never more than one poll away from noticing: its read
/// loop polls the ffmpeg pipe with a 500 ms timeout and re-reads the stop flag
/// between polls, then kills ffmpeg, reaps it, and joins the stderr reader,
/// which ends on the EOF that the kill produces. A second would cover all of
/// that on an idle box. Ten is the allowance for a loaded one, and it is a
/// deadline shared by every camera rather than one each, so a rack of them
/// cannot multiply it.
///
/// What it does not cover is a thread that will not come back at all — an
/// ffmpeg stuck unkillable in D state, a read that returns to nobody. That
/// camera's watermark is published anyway, from wherever its buffer had got
/// to, and the phases behind it proceed. It is the difference between losing
/// one camera's tail and never restarting.
pub const CAMERA_JOIN_BOUND: Duration = Duration::from_secs(10);

/// Phase 2: how long a consumer keeps trying to reach its camera's watermark
/// after the stop flag goes up.
///
/// Measured from the flag rather than from the watermark, so it also covers the
/// consumer's wait for phase 1 to finish — and so that a watermark which never
/// arrives is a bounded wait rather than a hang. Ten of those thirty seconds
/// are phase 1's; the remaining twenty are the drain itself, which for a
/// consumer running near real time is a couple of segments' work.
pub const TAIL_DRAIN_BOUND: Duration = Duration::from_secs(30);

/// Phase 2's join bound, from the drain's side.
///
/// A consumer bounds its own drain by [`TAIL_DRAIN_BOUND`], so this only has to
/// cover the tick it happens to be in when that trips — a decode, an event
/// assembly, a blocking send to a writer with a full queue. Five seconds of
/// slack over the consumer's own bound.
///
/// Past it the consumer is abandoned where it stands, and abandonment is not
/// abort: the task is left running, still holding its warm-writer sender, and
/// the event it is in the middle of flushing goes down that channel and is
/// written by phase 3 like any other. That is the property this bound rests
/// on, and it is the reason phase 3 has a bound of its own — the same sender
/// that lets a late flush land is a channel that will not close, and a channel
/// that does not close is a writer that never returns.
pub const CONSUMER_JOIN_BOUND: Duration = Duration::from_secs(35);

/// What the drain keeps back from the budget for handing over: the final buffer
/// stats, the line saying the stop is complete, and the process teardown behind
/// it. Everything the budget is not spending on the phases belongs to the
/// writers, so this is small on purpose — and it is subtracted before phase 3's
/// deadline rather than hoped for afterwards, so the restart watchdog's
/// `_exit` never lands on a drain that was about to say what it lost.
pub(crate) const TEARDOWN_MARGIN: Duration = Duration::from_secs(10);

/// A camera's terminal watermark: the handoff from phase 1 to phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Watermark {
    /// One past the last sequence the camera produced.
    pub sequence: u64,
    /// True when the camera thread had *not* actually stopped when this was
    /// published — phase 1's join bound tripped and the drain published it on
    /// the camera's behalf. The camera may still be pushing, so the sequence is
    /// a floor and not a finish line, and a consumer must not treat reaching it
    /// as proof there is nothing more coming.
    pub provisional: bool,
}

/// What a consumer should do about the phase-2 drain right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainStep {
    /// The camera has stopped and nothing it produced is left unconsumed.
    /// Flush and exit.
    Drained,
    /// The watermark is not reached, not published yet, or provisional, and
    /// there is still time. Keep working.
    Continue,
    /// The bound tripped. Whatever is left is left; say so and exit.
    ///
    /// Carries no count. How much is left behind is measured from where the
    /// consumer actually got to, and that is not always the position it hands
    /// [`DrainGate::step`] — an analyzer whose decoder died is *finished*
    /// consuming, whatever sequence it stopped scoring at, and says so to end
    /// the wait. Reporting from the same number would then claim it had kept
    /// up with a camera it had stopped following. The verdict comes from here;
    /// the figure comes from [`shortfall`].
    Abandoned,
}

/// How many sequences a consumer sitting at `next_seq` is leaving unconsumed.
///
/// `None` when the camera never published a watermark at all, so there is no
/// end to measure against and the honest answer is that nobody knows.
/// `Some(0)` is a real answer and not an absence: the consumer reached
/// everything the camera published, and what it could not wait for was the
/// camera itself.
///
/// Shared rather than inlined at the two call sites so the analyzer and the
/// recorder cannot come to mean different things by the same log field.
pub fn shortfall(terminal: Option<Watermark>, next_seq: u64) -> Option<u64> {
    terminal.map(|w| w.sequence.saturating_sub(next_seq))
}

/// Which side of a phase-2 abandonment ran out of time.
///
/// Worth naming because the two read as opposite failures to whoever finds the
/// line afterwards, and the difference is not in the count — a consumer that
/// fell behind and a camera that never stopped can both leave the same number
/// of sequences unconsumed. Each consumer words it in its own terms (an
/// analyzer loses the tail of an *event*, a recorder the tail of a
/// *recording*), but which of the two it is gets decided here, once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stalled {
    /// The camera stopped and said where — this consumer did not get through
    /// what it left in time. Usually a writer queue it was blocking on.
    Consumer,
    /// The camera never finished stopping, so there was no end to reach. The
    /// consumer may well have been keeping up perfectly.
    Camera,
}

/// Read the abandonment off the watermark: only a final one is a camera that
/// did its part.
pub fn who_stalled(terminal: Option<Watermark>) -> Stalled {
    match terminal {
        Some(w) if !w.provisional => Stalled::Consumer,
        _ => Stalled::Camera,
    }
}

/// The phase-2 stall bound, as a deadline a consumer can ask on every tick.
///
/// Created when the consumer first sees the stop flag; consulted after each
/// tick of work with the camera's watermark and the consumer's own position.
/// Pure with respect to `now` so the three answers can be pinned without
/// waiting out a bound.
pub struct DrainGate {
    deadline: Instant,
}

impl DrainGate {
    /// `now` is passed in rather than read here because the async consumers run
    /// on tokio's clock, which a paused test moves without the system clock
    /// moving with it.
    pub fn starting_at(now: Instant, bound: Duration) -> Self {
        Self {
            deadline: now + bound,
        }
    }

    /// Whether the bound has tripped. Asked separately by consumers that owe
    /// themselves one last flush before they abandon the rest.
    pub fn expired(&self, now: Instant) -> bool {
        now >= self.deadline
    }

    /// `terminal` is the camera's watermark, `next_seq` the first sequence this
    /// consumer has not yet consumed. Reaching the watermark wins over the
    /// deadline: a consumer that finished on the very tick its bound ran out
    /// finished.
    ///
    /// A *provisional* watermark never ends the drain, however far past it the
    /// consumer gets. It was published for a camera that missed its join bound
    /// and may still be pushing, so its sequence is a floor: exiting on it
    /// would drop whatever that camera produced next, which is the same loss
    /// the phases exist to prevent, arrived at by a different route. The
    /// consumer stays until its own bound and keeps consuming, which costs the
    /// remainder of [`TAIL_DRAIN_BOUND`] and only in the pathological case.
    pub fn step(&self, terminal: Option<Watermark>, next_seq: u64, now: Instant) -> DrainStep {
        match terminal {
            Some(w) if !w.provisional && next_seq >= w.sequence => DrainStep::Drained,
            Some(_) if self.expired(now) => DrainStep::Abandoned,
            None if self.expired(now) => DrainStep::Abandoned,
            _ => DrainStep::Continue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUND: Duration = Duration::from_secs(30);

    fn gate(now: Instant) -> DrainGate {
        DrainGate::starting_at(now, BOUND)
    }

    /// A camera that stopped on its own, so its watermark is final.
    fn sealed(sequence: u64) -> Option<Watermark> {
        Some(Watermark {
            sequence,
            provisional: false,
        })
    }

    /// A camera that missed its join bound, so the drain published a watermark
    /// on its behalf while it was still running.
    fn provisional(sequence: u64) -> Option<Watermark> {
        Some(Watermark {
            sequence,
            provisional: true,
        })
    }

    /// The whole point of phase 2: a consumer that has not reached the
    /// watermark keeps working, however long it has been stopping for, until
    /// either the watermark or the bound says otherwise.
    #[test]
    fn a_consumer_short_of_the_watermark_keeps_draining() {
        let t0 = Instant::now();
        assert_eq!(
            gate(t0).step(sealed(12), 5, t0 + Duration::from_secs(29)),
            DrainStep::Continue
        );
    }

    /// Consuming through the watermark ends the drain — the consumer does not
    /// sit out the rest of its bound once there is nothing left to consume.
    #[test]
    fn reaching_the_watermark_ends_the_drain() {
        let t0 = Instant::now();
        assert_eq!(gate(t0).step(sealed(12), 12, t0), DrainStep::Drained);
        assert_eq!(gate(t0).step(sealed(12), 15, t0), DrainStep::Drained);
    }

    /// A provisional watermark is a floor, not a finish line. The camera it was
    /// published for is still running, so reaching its sequence proves nothing:
    /// a consumer that exited there would drop whatever that camera pushed
    /// next, which is the loss the phases exist to prevent arrived at by a
    /// different route. It keeps consuming until its own bound instead.
    #[test]
    fn a_provisional_watermark_never_ends_the_drain_early() {
        let t0 = Instant::now();
        assert_eq!(gate(t0).step(provisional(12), 12, t0), DrainStep::Continue);
        assert_eq!(
            gate(t0).step(provisional(12), 99, t0 + BOUND - Duration::from_secs(1)),
            DrainStep::Continue
        );
        // And it ends at the bound, not never.
        assert_eq!(
            gate(t0).step(provisional(12), 12, t0 + BOUND),
            DrainStep::Abandoned
        );
    }

    /// The two abandonments read as opposite failures and the count cannot
    /// tell them apart — both can leave the same number of sequences behind.
    /// A camera that stopped and said where did its part, so what ran out was
    /// the consumer; anything else is a camera that never finished.
    #[test]
    fn an_abandonment_names_whichever_side_ran_out_of_time() {
        assert_eq!(who_stalled(sealed(12)), Stalled::Consumer);
        assert_eq!(who_stalled(provisional(12)), Stalled::Camera);
        assert_eq!(who_stalled(None), Stalled::Camera);
    }

    /// A stalled consumer must not hold the stop open.
    #[test]
    fn a_stalled_consumer_is_abandoned_rather_than_waited_for() {
        let t0 = Instant::now();
        assert_eq!(
            gate(t0).step(sealed(12), 5, t0 + BOUND),
            DrainStep::Abandoned
        );
    }

    /// What the operator is told an abandonment cost, measured from where the
    /// consumer really got to. A camera that published nothing leaves no end to
    /// measure against; one the consumer reached leaves zero, which is an
    /// answer and not an absence.
    #[test]
    fn a_shortfall_counts_from_where_the_consumer_really_got_to() {
        assert_eq!(shortfall(sealed(12), 5), Some(7));
        assert_eq!(shortfall(provisional(12), 12), Some(0));
        assert_eq!(shortfall(None, 5), None);
        // A consumer past a provisional watermark is not owed a negative.
        assert_eq!(shortfall(provisional(12), 99), Some(0));
    }

    /// The camera whose thread never came back publishes no watermark of its
    /// own, so its consumers wait out the bound and then admit they cannot say
    /// what they lost, rather than waiting for ever.
    #[test]
    fn a_watermark_that_never_arrives_is_a_bounded_wait() {
        let t0 = Instant::now();
        let gate = gate(t0);
        assert_eq!(gate.step(None, 5, t0), DrainStep::Continue);
        assert_eq!(gate.step(None, 5, t0 + BOUND), DrainStep::Abandoned);
    }

    /// A consumer that arrives at the watermark on the same tick its bound
    /// expires has drained, not been abandoned: there is nothing left for the
    /// abandonment to describe.
    #[test]
    fn draining_on_the_last_tick_counts_as_drained() {
        let t0 = Instant::now();
        assert_eq!(
            gate(t0).step(sealed(12), 12, t0 + BOUND * 2),
            DrainStep::Drained
        );
    }

    /// The arithmetic in this module's documentation, against the constants
    /// themselves rather than against numbers copied out of it. Every term is
    /// the real one — the phase bounds, the MQTT flush, the teardown margin,
    /// the budget both service managers give the stop, and the upload timeout
    /// the remainder has to cover — so widening any single bound fails the
    /// build here instead of surfacing as a truncated upload during an
    /// incident.
    #[test]
    fn the_phase_bounds_leave_the_writers_a_whole_upload_timeout() {
        let spent_elsewhere = CAMERA_JOIN_BOUND
            + CONSUMER_JOIN_BOUND
            + crate::app::MQTT_SHUTDOWN_TIMEOUT
            + TEARDOWN_MARGIN;
        let left_for_phase_3 = crate::app::RESTART_DRAIN_DEADLINE - spent_elsewhere;
        assert!(
            left_for_phase_3 >= crate::storage::stathost::UPLOAD_TIMEOUT,
            "the drain spends {spent_elsewhere:?} of its {:?} budget outside phase 3, leaving the \
             writers and the retention sweep {left_for_phase_3:?} — less than the {:?} a single \
             stathost upload may take",
            crate::app::RESTART_DRAIN_DEADLINE,
            crate::storage::stathost::UPLOAD_TIMEOUT,
        );
        assert!(
            CONSUMER_JOIN_BOUND > TAIL_DRAIN_BOUND,
            "no slack to join a consumer in after its own bound trips"
        );
        assert!(
            TAIL_DRAIN_BOUND > CAMERA_JOIN_BOUND,
            "a consumer's bound must outlast the phase 1 it waits through"
        );
    }
}
