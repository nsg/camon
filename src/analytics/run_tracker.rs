//! Motion-run lifecycle tracking for event-driven persistence.

use std::time::{Duration, Instant};

/// A finished motion run (or one chunk of a long run), ready for event
/// assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosedRun {
    /// First motion-positive segment of this chunk, including a continuation resumed after
    /// pending padding.
    pub first_motion_seq: u64,
    /// Sequence of the last segment included in the event (motion or
    /// post-padding).
    pub last_seq: u64,
    /// Earliest sequence pre-padding may reach back to, which keeps this event's pre-padding
    /// clear of the previous event or chunk.
    pub min_start_seq: u64,
    /// True when this chunk continues a previous chunk of the same motion run
    /// (produced by the duration cap). Drives the `"continues": true` sidecar
    /// so a UI can stitch the chain back together.
    pub continues: bool,
}

struct OpenRun {
    /// First segment of the current chunk (see [`ClosedRun::first_motion_seq`]).
    first_motion_seq: u64,
    /// When the most recent motion-positive segment was observed. Post-padding
    /// counts from here.
    last_motion_instant: Instant,
    /// Last segment included in the current chunk so far.
    last_seq: u64,
    /// When the current chunk opened. The duration cap counts from here.
    chunk_start_instant: Instant,
    /// Whether the current chunk is itself a follow-on (continuation).
    continues: bool,
}

/// Where a camera's motion run stands. The middle state is the one worth
/// naming: a run can be alive with no chunk open, which is how the cap closes
/// on a padding segment without inventing an event to hold the padding.
enum State {
    /// No run. The next motion segment opens one.
    Idle,
    /// A chunk is open and collecting segments.
    Open(OpenRun),
    /// The cap closed a chunk on a padding segment.
    Pending {
        /// The last motion of the run, still counting the quiet window down.
        last_motion_instant: Instant,
    },
}

pub struct RunTracker {
    post_padding: Duration,
    /// Maximum wall-clock duration of a single event chunk. `Duration::ZERO`
    /// disables chunking (a run grows until motion stops).
    max_event_duration: Duration,
    state: State,
    /// One past the last segment of the previously closed run or chunk.
    barrier_seq: u64,
    /// Identity of the current motion period — bumped every time a run opens
    /// out of [`State::Idle`], never when a chunk rolls within one. See
    /// [`motion_period`](Self::motion_period).
    period: u64,
}

impl RunTracker {
    pub fn new(post_padding: Duration, max_event_duration: Duration) -> Self {
        Self {
            post_padding,
            max_event_duration,
            state: State::Idle,
            barrier_seq: 0,
            period: 0,
        }
    }

    /// Feed one scored segment, observed at monotonic time `now`.
    pub fn observe(&mut self, seq: u64, has_motion: bool, now: Instant) -> Option<ClosedRun> {
        let (last_motion_instant, chunk_start_instant) = match self.state {
            State::Open(ref run) => (run.last_motion_instant, run.chunk_start_instant),
            State::Pending {
                last_motion_instant,
            } => return self.observe_pending(seq, has_motion, now, last_motion_instant),
            State::Idle => {
                if has_motion {
                    // The one place a motion period begins.
                    self.period += 1;
                    self.open_chunk(seq, now, false);
                }
                return None;
            }
        };

        // A non-motion segment past the post-padding window closes the run. The
        // closing segment is excluded; padding segments within the window stay.
        if !has_motion && now.saturating_duration_since(last_motion_instant) > self.post_padding {
            return self.close();
        }

        // The chunk has reached the cap, so it closes here whatever this segment is — its
        // span must stay within the cap.
        if !self.max_event_duration.is_zero()
            && now.saturating_duration_since(chunk_start_instant) >= self.max_event_duration
        {
            return Some(if has_motion {
                self.chunk(seq, now)
            } else {
                self.suspend(last_motion_instant)
            });
        }

        // Extend the current chunk with this segment.
        if let State::Open(ref mut run) = self.state {
            run.last_seq = seq;
            if has_motion {
                run.last_motion_instant = now;
            }
        }
        None
    }

    /// One segment while the run is pending: the quiet window is still counting
    /// down from the run's last motion, and nothing is collecting segments.
    fn observe_pending(
        &mut self,
        seq: u64,
        has_motion: bool,
        now: Instant,
        last_motion_instant: Instant,
    ) -> Option<ClosedRun> {
        if now.saturating_duration_since(last_motion_instant) > self.post_padding {
            // The window elapsed with no motion, so the run is over — and it needs no
            // closing, because the cap already closed its last chunk and advanced the barrier
            // past it.
            self.state = State::Idle;
            return self.observe(seq, has_motion, now);
        }
        if has_motion {
            // Motion inside the window: the chain continues, and the follow-on
            // opens here — on the motion segment, never on the padding before
            // it.
            self.open_chunk(seq, now, true);
        }
        None
    }

    /// Which motion period is alive, or `None` when none is — the physical "is something
    /// moving" the Home Assistant sensor mirrors, as opposed to the event bookkeeping.
    pub fn motion_period(&self) -> Option<u64> {
        match self.state {
            State::Idle => None,
            State::Open(_) | State::Pending { .. } => Some(self.period),
        }
    }

    /// Whether a run is alive — a chunk collecting segments, or one pending
    /// between the cap and whatever comes next.
    pub fn is_open(&self) -> bool {
        self.motion_period().is_some()
    }

    /// Close an open run immediately (shutdown flush — no post-padding wait), extending it
    /// through `through` when that reaches past the last segment the analyzer scored.
    pub fn flush(&mut self, through: Option<u64>) -> Option<ClosedRun> {
        if let (Some(through), State::Open(run)) = (through, &mut self.state) {
            run.last_seq = run.last_seq.max(through);
        }
        self.close()
    }

    /// Open a chunk on `seq`, which is always a motion segment.
    fn open_chunk(&mut self, seq: u64, now: Instant, continues: bool) {
        self.state = State::Open(OpenRun {
            first_motion_seq: seq,
            last_motion_instant: now,
            last_seq: seq,
            chunk_start_instant: now,
            continues,
        });
    }

    /// Close whatever chunk is open and go idle. A pending run has no chunk, so
    /// this ends it without producing one.
    fn close(&mut self) -> Option<ClosedRun> {
        let run = match std::mem::replace(&mut self.state, State::Idle) {
            State::Open(run) => run,
            State::Idle | State::Pending { .. } => return None,
        };
        let closed = ClosedRun {
            first_motion_seq: run.first_motion_seq,
            last_seq: run.last_seq,
            min_start_seq: self.barrier_seq,
            continues: run.continues,
        };
        self.barrier_seq = run.last_seq + 1;
        Some(closed)
    }

    /// Close the current chunk at the duration cap and open its follow-on on `seq`, the motion
    /// segment that crossed the cap.
    fn chunk(&mut self, seq: u64, now: Instant) -> ClosedRun {
        let closed = self.close().expect("chunk requires an open run");
        self.open_chunk(seq, now, true);
        closed
    }

    /// Close the current chunk at the duration cap without opening a follow-on: the segment
    /// that crossed it is padding, and a chunk opened on padding would hold nothing else.
    fn suspend(&mut self, last_motion_instant: Instant) -> ClosedRun {
        let closed = self.close().expect("a cap crossing requires an open run");
        self.state = State::Pending {
            last_motion_instant,
        };
        closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POST: Duration = Duration::from_secs(10);
    const CAP: Duration = Duration::from_secs(120);

    fn base() -> Instant {
        Instant::now()
    }

    #[test]
    fn no_motion_never_opens_a_run() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        for seq in 0..5 {
            assert_eq!(t.observe(seq, false, t0 + Duration::from_secs(seq)), None);
        }
        assert_eq!(t.flush(None), None);
    }

    #[test]
    fn motion_opens_run_and_padding_elapse_closes_it() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, false, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(1)), None);
        assert_eq!(t.observe(2, true, t0 + Duration::from_secs(2)), None);
        assert_eq!(t.observe(3, false, t0 + Duration::from_secs(3)), None);
        let closed = t
            .observe(
                4,
                false,
                t0 + Duration::from_secs(2) + POST + Duration::from_nanos(1),
            )
            .unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 1,
                last_seq: 3,
                min_start_seq: 0,
                continues: false,
            }
        );
        assert_eq!(t.flush(None), None);
    }

    #[test]
    fn motion_within_padding_continues_the_same_run() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, false, t0 + Duration::from_secs(1)), None);
        assert_eq!(t.observe(2, true, t0 + Duration::from_secs(2)), None);
        assert_eq!(
            t.observe(3, false, t0 + Duration::from_secs(2) + POST),
            None
        );
        let closed = t
            .observe(
                4,
                false,
                t0 + Duration::from_secs(2) + POST + Duration::from_nanos(1),
            )
            .unwrap();
        assert_eq!(closed.first_motion_seq, 0);
        assert_eq!(closed.last_seq, 3);
        assert!(!closed.continues);
    }

    #[test]
    fn flush_closes_open_run_immediately() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        t.observe(5, true, t0);
        t.observe(6, false, t0 + Duration::from_secs(1));
        let closed = t.flush(None).unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 5,
                last_seq: 6,
                min_start_seq: 0,
                continues: false,
            }
        );
        assert_eq!(t.flush(None), None);
    }

    #[test]
    fn the_shutdown_flush_closes_through_footage_nobody_scored() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        t.observe(5, true, t0);
        t.observe(6, false, t0 + Duration::from_secs(1));
        let closed = t.flush(Some(11)).unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 5,
                last_seq: 11,
                min_start_seq: 0,
                continues: false,
            }
        );
    }

    #[test]
    fn the_shutdown_flush_never_shortens_a_run() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        t.observe(5, true, t0);
        t.observe(6, true, t0 + Duration::from_secs(1));
        assert_eq!(t.flush(Some(2)).unwrap().last_seq, 6);
    }

    #[test]
    fn the_shutdown_flush_extends_nothing_when_no_run_is_open() {
        let mut t = RunTracker::new(POST, CAP);
        t.observe(5, false, base());
        assert_eq!(t.flush(Some(11)), None);
    }

    #[test]
    fn next_run_cannot_pre_pad_into_previous_event() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        t.observe(0, true, t0);
        let first = t
            .observe(1, false, t0 + POST + Duration::from_nanos(1))
            .unwrap();
        assert_eq!(first.last_seq, 0);
        t.observe(2, true, t0 + POST + Duration::from_secs(2));
        let second = t.flush(None).unwrap();
        assert_eq!(second.min_start_seq, 1);
        assert_eq!(second.first_motion_seq, 2);
    }

    #[test]
    fn cap_chunks_a_long_run_and_opens_a_follow_on() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(60)), None);
        let closed = t.observe(2, true, t0 + CAP).unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 0,
                last_seq: 1,
                min_start_seq: 0,
                continues: false,
            }
        );
        let follow = t.flush(None).unwrap();
        assert_eq!(follow.first_motion_seq, 2);
        assert_eq!(follow.min_start_seq, 2);
        assert!(follow.continues);
    }

    #[test]
    fn chain_of_three_chunks_only_follow_ons_continue() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        let a = t.observe(1, true, t0 + CAP).unwrap();
        assert_eq!(a.first_motion_seq, 0);
        assert_eq!(a.last_seq, 0);
        assert!(!a.continues);
        let b = t.observe(2, true, t0 + CAP + CAP).unwrap();
        assert_eq!(b.first_motion_seq, 1);
        assert_eq!(b.last_seq, 1);
        assert_eq!(b.min_start_seq, 1);
        assert!(b.continues);
        assert_eq!(
            t.observe(3, false, t0 + CAP + CAP + Duration::from_secs(1)),
            None
        );
        let c = t
            .observe(4, false, t0 + CAP + CAP + POST + Duration::from_secs(2))
            .unwrap();
        assert_eq!(c.first_motion_seq, 2);
        assert_eq!(c.last_seq, 3);
        assert_eq!(c.min_start_seq, 2);
        assert!(c.continues);
    }

    #[test]
    fn the_cap_closes_a_motion_chunk_before_a_buffer_wide_quiet_window_can() {
        const WIDE_POST: Duration = Duration::from_secs(600);
        let mut t = RunTracker::new(WIDE_POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(30)), None);
        assert_eq!(t.observe(2, false, t0 + Duration::from_secs(60)), None);
        let closed = t.observe(3, false, t0 + CAP).unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 0,
                last_seq: 2,
                min_start_seq: 0,
                continues: false,
            }
        );
        assert!(t.is_open());
    }

    #[test]
    fn padding_past_the_cap_never_opens_a_motionless_follow_on() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        let last_motion = t0 + CAP - Duration::from_secs(2);
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, last_motion), None);
        let closed = t.observe(2, false, t0 + CAP).unwrap();
        assert_eq!(closed.last_seq, 1);
        assert!(!closed.continues);
        assert_eq!(t.observe(3, false, t0 + CAP + Duration::from_secs(1)), None);
        assert_eq!(t.observe(4, false, t0 + CAP + Duration::from_secs(5)), None);
        assert_eq!(
            t.observe(5, false, last_motion + POST + Duration::from_nanos(1)),
            None
        );
        assert!(!t.is_open());
        assert_eq!(t.flush(None), None);
    }

    #[test]
    fn motion_returning_after_a_cap_on_padding_continues_the_chain() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + CAP - Duration::from_secs(2)), None);
        assert_eq!(t.observe(2, false, t0 + CAP).unwrap().last_seq, 1);
        assert_eq!(t.observe(3, false, t0 + CAP + Duration::from_secs(1)), None);
        assert_eq!(t.observe(4, true, t0 + CAP + Duration::from_secs(2)), None);
        let follow = t.flush(None).unwrap();
        assert_eq!(follow.first_motion_seq, 4);
        assert_eq!(follow.min_start_seq, 2);
        assert!(follow.continues);
    }

    #[test]
    fn a_pending_run_ends_when_its_window_elapses_and_later_motion_starts_a_new_one() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        let last_motion = t0 + CAP - Duration::from_secs(2);
        t.observe(0, true, t0);
        t.observe(1, true, last_motion);
        assert!(t.observe(2, false, t0 + CAP).is_some());
        assert!(t.is_open());
        assert_eq!(
            t.observe(3, true, last_motion + POST + Duration::from_secs(1)),
            None
        );
        let fresh = t.flush(None).unwrap();
        assert_eq!(fresh.first_motion_seq, 3);
        assert_eq!(fresh.min_start_seq, 2);
        assert!(!fresh.continues);
    }

    #[test]
    fn the_shutdown_flush_seals_a_pending_run_without_writing_padding() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        t.observe(0, true, t0);
        t.observe(1, true, t0 + CAP - Duration::from_secs(2));
        assert!(t.observe(2, false, t0 + CAP).is_some());
        assert_eq!(t.flush(Some(99)), None);
        assert!(!t.is_open());
    }

    #[test]
    fn the_motion_period_changes_only_when_one_run_gives_way_to_another() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.motion_period(), None);
        t.observe(0, true, t0);
        let first = t.motion_period().expect("a run is open");
        assert!(t.observe(1, true, t0 + CAP).is_some());
        assert_eq!(t.motion_period(), Some(first));
        let last_motion = t0 + CAP + CAP - Duration::from_secs(2);
        assert_eq!(t.observe(2, true, last_motion), None);
        assert!(t.observe(3, false, t0 + CAP + CAP).is_some());
        assert_eq!(t.motion_period(), Some(first));
        let resumed = last_motion + POST + Duration::from_secs(1);
        t.observe(4, true, resumed);
        assert!(t.is_open());
        assert_ne!(t.motion_period(), Some(first));
        t.observe(5, false, resumed + POST + Duration::from_nanos(1));
        assert_eq!(t.motion_period(), None);
    }

    #[test]
    fn is_open_tracks_the_physical_motion_period() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert!(!t.is_open());
        t.observe(0, true, t0);
        assert!(t.is_open());
        assert!(t.observe(1, true, t0 + CAP).is_some());
        assert!(t.is_open());
        t.observe(2, false, t0 + CAP + POST + Duration::from_nanos(1));
        assert!(!t.is_open());
    }

    #[test]
    fn zero_cap_disables_chunking() {
        let mut t = RunTracker::new(POST, Duration::ZERO);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(3600)), None);
        let closed = t.flush(None).unwrap();
        assert_eq!(closed.first_motion_seq, 0);
        assert_eq!(closed.last_seq, 1);
        assert!(!closed.continues);
    }

    #[test]
    fn pts_jumps_do_not_affect_lifecycle() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(1000, true, t0), None);
        for seq in 1001..1010 {
            assert_eq!(t.observe(seq, true, t0 + Duration::from_secs(1)), None);
        }
        assert_eq!(t.observe(1010, false, t0 + Duration::from_secs(2)), None);
        let closed = t
            .observe(
                1011,
                false,
                t0 + Duration::from_secs(1) + POST + Duration::from_secs(2),
            )
            .unwrap();
        assert_eq!(closed.first_motion_seq, 1000);
        assert_eq!(closed.last_seq, 1010);
    }
}
