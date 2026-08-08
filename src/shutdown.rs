//! The phased stop: draining a running recording without cutting off its tail.

use std::time::{Duration, Instant};

/// Phase 1: how long the whole drain waits for the camera threads to stop — one deadline
/// shared by every camera, so a rack of them cannot multiply it.
pub const CAMERA_JOIN_BOUND: Duration = Duration::from_secs(10);

/// Phase 2: how long a consumer keeps trying to reach its camera's watermark.
/// Measured from the stop flag rather than the watermark, so it also covers
/// phase 1 and bounds a watermark that never arrives.
pub const TAIL_DRAIN_BOUND: Duration = Duration::from_secs(30);

/// Phase 2's join bound, from the drain's side: [`TAIL_DRAIN_BOUND`] plus slack for the tick
/// the consumer is in when its own bound trips.
pub const CONSUMER_JOIN_BOUND: Duration = Duration::from_secs(35);

/// Held back from the budget for the final stats and process teardown. Subtracted before phase
/// 3's deadline rather than hoped for afterwards, so the restart watchdog's `_exit` never lands
/// on a drain that was about to say what it lost.
pub(crate) const TEARDOWN_MARGIN: Duration = Duration::from_secs(10);

/// A camera's terminal watermark: the handoff from phase 1 to phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Watermark {
    /// One past the last sequence the camera produced.
    pub sequence: u64,
    /// True when phase 1's join bound tripped and the drain published this on
    /// the camera's behalf. The camera may still be pushing, so the sequence
    /// is a floor, not a finish line.
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
    Abandoned,
}

/// How many sequences a consumer sitting at `next_seq` is leaving unconsumed.
pub fn shortfall(terminal: Option<Watermark>, next_seq: u64) -> Option<u64> {
    terminal.map(|w| w.sequence.saturating_sub(next_seq))
}

/// Which side of a phase-2 abandonment ran out of time. The count cannot tell
/// them apart — a slow consumer and a camera that never stopped can leave the
/// same number of sequences unconsumed — so it is decided here, once.
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

/// The phase-2 stall bound, as a deadline a consumer asks on every tick. Pure
/// with respect to `now` so tests can pin the answers without waiting.
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
    /// themselves one last flush.
    pub fn expired(&self, now: Instant) -> bool {
        now >= self.deadline
    }

    /// `terminal` is the camera's watermark, `next_seq` the first sequence this consumer has
    /// not yet consumed. Reaching the watermark wins over the deadline.
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

    fn sealed(sequence: u64) -> Option<Watermark> {
        Some(Watermark {
            sequence,
            provisional: false,
        })
    }

    fn provisional(sequence: u64) -> Option<Watermark> {
        Some(Watermark {
            sequence,
            provisional: true,
        })
    }

    #[test]
    fn a_consumer_short_of_the_watermark_keeps_draining() {
        let t0 = Instant::now();
        assert_eq!(
            gate(t0).step(sealed(12), 5, t0 + Duration::from_secs(29)),
            DrainStep::Continue
        );
    }

    #[test]
    fn reaching_the_watermark_ends_the_drain() {
        let t0 = Instant::now();
        assert_eq!(gate(t0).step(sealed(12), 12, t0), DrainStep::Drained);
        assert_eq!(gate(t0).step(sealed(12), 15, t0), DrainStep::Drained);
    }

    #[test]
    fn a_provisional_watermark_never_ends_the_drain_early() {
        let t0 = Instant::now();
        assert_eq!(gate(t0).step(provisional(12), 12, t0), DrainStep::Continue);
        assert_eq!(
            gate(t0).step(provisional(12), 99, t0 + BOUND - Duration::from_secs(1)),
            DrainStep::Continue
        );
        assert_eq!(
            gate(t0).step(provisional(12), 12, t0 + BOUND),
            DrainStep::Abandoned
        );
    }

    #[test]
    fn an_abandonment_names_whichever_side_ran_out_of_time() {
        assert_eq!(who_stalled(sealed(12)), Stalled::Consumer);
        assert_eq!(who_stalled(provisional(12)), Stalled::Camera);
        assert_eq!(who_stalled(None), Stalled::Camera);
    }

    #[test]
    fn a_stalled_consumer_is_abandoned_rather_than_waited_for() {
        let t0 = Instant::now();
        assert_eq!(
            gate(t0).step(sealed(12), 5, t0 + BOUND),
            DrainStep::Abandoned
        );
    }

    #[test]
    fn a_shortfall_counts_from_where_the_consumer_really_got_to() {
        assert_eq!(shortfall(sealed(12), 5), Some(7));
        assert_eq!(shortfall(provisional(12), 12), Some(0));
        assert_eq!(shortfall(None, 5), None);
        assert_eq!(shortfall(provisional(12), 99), Some(0));
    }

    #[test]
    fn a_watermark_that_never_arrives_is_a_bounded_wait() {
        let t0 = Instant::now();
        let gate = gate(t0);
        assert_eq!(gate.step(None, 5, t0), DrainStep::Continue);
        assert_eq!(gate.step(None, 5, t0 + BOUND), DrainStep::Abandoned);
    }

    #[test]
    fn draining_on_the_last_tick_counts_as_drained() {
        let t0 = Instant::now();
        assert_eq!(
            gate(t0).step(sealed(12), 12, t0 + BOUND * 2),
            DrainStep::Drained
        );
    }

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
