//! Motion-run lifecycle tracking for event-driven persistence.
//!
//! The analyzer feeds every scored segment (in sequence order) into the
//! tracker. A run opens on the first motion-positive segment, absorbs
//! non-motion segments while they fall within the post-padding window, and
//! closes when a non-motion segment arrives past the window. The closing
//! segment itself is not part of the event.
//!
//! # Monotonic lifecycle timing
//!
//! All lifecycle decisions — when post-padding elapses, and when a long run
//! hits the duration cap — are driven by `std::time::Instant` (monotonic
//! wall-adjacent time), *not* media PTS. Camera PTS can jump or reset, which
//! would otherwise freeze or prematurely fire these countdowns. The analyzer
//! runs near real time (200 ms poll), so the `Instant` at which it observes a
//! segment is a faithful stand-in for when that segment was captured. `now` is
//! injected by the caller so the state machine stays deterministic under test.
//!
//! This is the *only* place Instant timing is used. Media timing — PTS
//! extraction, segment durations, the pre-padding reach, playlist math and the
//! `{first_pts_ns}_{duration_ms}` filename stem — all stay on media PTS.
//!
//! # Duration cap and chunking
//!
//! Under sustained motion a run would otherwise grow without bound (RAM for the
//! assembled event, giant `.ts` files, and — for runs longer than the hot
//! buffer — gaps as early segments age out before the run closes). When the
//! open chunk reaches `max_event_duration`, the tracker closes it as a
//! *complete, independently playable* event and immediately opens a follow-on
//! chunk continuing the same run. Because chunks split on whole GOP segments,
//! every chunk starts with PAT/PMT + keyframe and decodes on its own.
//!
//! Follow-on chunks are flagged [`ClosedRun::continues`]. They get no
//! pre-padding: the barrier advances past the previous chunk, so assembly
//! cannot reach back into it. The first chunk of a chain keeps normal
//! pre-padding; the final chunk closes through the normal post-padding path.

use std::time::{Duration, Instant};

/// A finished motion run (or one chunk of a long run), ready for event
/// assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosedRun {
    /// Sequence of the first segment of this chunk. For the first chunk this is
    /// the first motion-positive segment; for follow-on chunks it is the
    /// segment that continued the run past the cap.
    pub first_motion_seq: u64,
    /// Sequence of the last segment included in the event (motion or
    /// post-padding).
    pub last_seq: u64,
    /// Earliest sequence pre-padding may reach back to. Prevents the
    /// pre-padding of this event from overlapping the previous event (or, for a
    /// follow-on chunk, the previous chunk — which suppresses pre-padding
    /// entirely).
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

pub struct RunTracker {
    post_padding: Duration,
    /// Maximum wall-clock duration of a single event chunk. `Duration::ZERO`
    /// disables chunking (a run grows until motion stops).
    max_event_duration: Duration,
    open: Option<OpenRun>,
    /// One past the last segment of the previously closed run or chunk.
    barrier_seq: u64,
}

impl RunTracker {
    pub fn new(post_padding: Duration, max_event_duration: Duration) -> Self {
        Self {
            post_padding,
            max_event_duration,
            open: None,
            barrier_seq: 0,
        }
    }

    /// Feed one scored segment, observed at monotonic time `now`. Returns a
    /// `ClosedRun` when this segment ends an open run (post-padding elapsed) or
    /// when it crosses the duration cap (the chunk closes and a follow-on opens
    /// starting at `seq`).
    pub fn observe(&mut self, seq: u64, has_motion: bool, now: Instant) -> Option<ClosedRun> {
        let (last_motion_instant, chunk_start_instant) = match self.open {
            Some(ref run) => (run.last_motion_instant, run.chunk_start_instant),
            None => {
                if has_motion {
                    self.open = Some(OpenRun {
                        first_motion_seq: seq,
                        last_motion_instant: now,
                        last_seq: seq,
                        chunk_start_instant: now,
                        continues: false,
                    });
                }
                return None;
            }
        };

        // A non-motion segment past the post-padding window closes the run. The
        // closing segment is excluded; padding segments within the window stay.
        if !has_motion && now.saturating_duration_since(last_motion_instant) > self.post_padding {
            return self.close();
        }

        // The segment belongs to the run. If the open chunk has reached the cap,
        // close it as a complete event and continue this segment in a new chunk.
        if !self.max_event_duration.is_zero()
            && now.saturating_duration_since(chunk_start_instant) >= self.max_event_duration
        {
            return Some(self.chunk(seq, has_motion, now));
        }

        // Extend the current chunk with this segment.
        if let Some(ref mut run) = self.open {
            run.last_seq = seq;
            if has_motion {
                run.last_motion_instant = now;
            }
        }
        None
    }

    /// Whether a run (or a chunk of one) is currently open. Compared before and
    /// after [`observe`](Self::observe) to spot the physical start and end of
    /// motion: the duration cap closes and reopens within a single call, so a
    /// chunk boundary leaves this `true` throughout and is invisible to
    /// consumers that only care about "is something moving".
    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Close an open run immediately (shutdown flush — no post-padding wait).
    pub fn flush(&mut self) -> Option<ClosedRun> {
        self.close()
    }

    fn close(&mut self) -> Option<ClosedRun> {
        let run = self.open.take()?;
        let closed = ClosedRun {
            first_motion_seq: run.first_motion_seq,
            last_seq: run.last_seq,
            min_start_seq: self.barrier_seq,
            continues: run.continues,
        };
        self.barrier_seq = run.last_seq + 1;
        Some(closed)
    }

    /// Close the current chunk at the duration cap and open a follow-on chunk
    /// beginning at `seq`. The barrier advances past the closed chunk, so the
    /// follow-on gets no pre-padding, and it is flagged `continues`.
    fn chunk(&mut self, seq: u64, has_motion: bool, now: Instant) -> ClosedRun {
        let prev = self.open.take().expect("chunk requires an open run");
        let closed = ClosedRun {
            first_motion_seq: prev.first_motion_seq,
            last_seq: prev.last_seq,
            min_start_seq: self.barrier_seq,
            continues: prev.continues,
        };
        self.barrier_seq = prev.last_seq + 1;
        self.open = Some(OpenRun {
            first_motion_seq: seq,
            // Preserve the last real motion time so the follow-on's post-padding
            // countdown stays correct even if it opens on a padding segment.
            last_motion_instant: if has_motion {
                now
            } else {
                prev.last_motion_instant
            },
            last_seq: seq,
            chunk_start_instant: now,
            continues: true,
        });
        closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POST: Duration = Duration::from_secs(10);
    const CAP: Duration = Duration::from_secs(120);

    /// Fixed base instant; tests advance from it explicitly.
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
        assert_eq!(t.flush(), None);
    }

    #[test]
    fn motion_opens_run_and_padding_elapse_closes_it() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, false, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(1)), None);
        assert_eq!(t.observe(2, true, t0 + Duration::from_secs(2)), None);
        // Non-motion within post-padding keeps the run open.
        assert_eq!(t.observe(3, false, t0 + Duration::from_secs(3)), None);
        // Non-motion past post-padding closes it; the closing segment is
        // excluded, the padding segment stays included.
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
        assert_eq!(t.flush(), None);
    }

    #[test]
    fn motion_within_padding_continues_the_same_run() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, false, t0 + Duration::from_secs(1)), None);
        // New motion inside the padding window extends the run.
        assert_eq!(t.observe(2, true, t0 + Duration::from_secs(2)), None);
        // Padding now counts from the new motion segment.
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
        let closed = t.flush().unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 5,
                last_seq: 6,
                min_start_seq: 0,
                continues: false,
            }
        );
        assert_eq!(t.flush(), None);
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
        // Second run: pre-padding may reach back only to seq 1.
        t.observe(2, true, t0 + POST + Duration::from_secs(2));
        let second = t.flush().unwrap();
        assert_eq!(second.min_start_seq, 1);
        assert_eq!(second.first_motion_seq, 2);
    }

    #[test]
    fn cap_chunks_a_long_run_and_opens_a_follow_on() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        // Motion opens the run; the chunk clock starts at t0.
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(60)), None);
        // At the cap the chunk closes; this segment starts the follow-on.
        let closed = t.observe(2, true, t0 + CAP).unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 0,
                // Chunk closes at the previous segment; seq 2 begins the next.
                last_seq: 1,
                min_start_seq: 0,
                continues: false,
            }
        );
        // The follow-on starts at seq 2, gets no pre-padding (barrier == 2),
        // and is flagged continues when it later closes.
        let follow = t.flush().unwrap();
        assert_eq!(follow.first_motion_seq, 2);
        assert_eq!(follow.min_start_seq, 2);
        assert!(follow.continues);
    }

    #[test]
    fn chain_of_three_chunks_only_follow_ons_continue() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        // First cap: chunk A closes (not a continuation), chunk B opens at 1.
        let a = t.observe(1, true, t0 + CAP).unwrap();
        assert_eq!(a.first_motion_seq, 0);
        assert_eq!(a.last_seq, 0);
        assert!(!a.continues);
        // Second cap: chunk B closes (continuation), chunk C opens at 2.
        let b = t.observe(2, true, t0 + CAP + CAP).unwrap();
        assert_eq!(b.first_motion_seq, 1);
        assert_eq!(b.last_seq, 1);
        assert_eq!(b.min_start_seq, 1);
        assert!(b.continues);
        // Motion stops; chunk C closes normally via post-padding.
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
        // Final chunk of a chain still carries continues (it continues B).
        assert!(c.continues);
    }

    #[test]
    fn is_open_tracks_the_physical_motion_period() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert!(!t.is_open());
        t.observe(0, true, t0);
        assert!(t.is_open());
        // The duration cap closes a chunk and opens its follow-on inside one
        // observe: the physical motion period never appears to end.
        assert!(t.observe(1, true, t0 + CAP).is_some());
        assert!(t.is_open());
        // Post-padding elapsing is the real end.
        t.observe(2, false, t0 + CAP + POST + Duration::from_nanos(1));
        assert!(!t.is_open());
    }

    #[test]
    fn zero_cap_disables_chunking() {
        let mut t = RunTracker::new(POST, Duration::ZERO);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        // Far past any sane cap — must not chunk.
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(3600)), None);
        let closed = t.flush().unwrap();
        assert_eq!(closed.first_motion_seq, 0);
        assert_eq!(closed.last_seq, 1);
        assert!(!closed.continues);
    }

    #[test]
    fn pts_jumps_do_not_affect_lifecycle() {
        // Lifecycle is driven purely by `now`; segment PTS is not even an
        // argument any more. A wildly non-monotonic capture timeline (simulated
        // here by the fact that seq/PTS play no role) must not close or chunk a
        // run — only the injected monotonic clock does.
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(1000, true, t0), None);
        // "PTS" would have jumped backwards/forwards, but time barely moved:
        // the run stays open across many segments.
        for seq in 1001..1010 {
            assert_eq!(t.observe(seq, true, t0 + Duration::from_secs(1)), None);
        }
        // A non-motion segment, still within post-padding by the monotonic
        // clock, keeps it open despite any PTS discontinuity.
        assert_eq!(t.observe(1010, false, t0 + Duration::from_secs(2)), None);
        // Only real elapsed time past post-padding closes it.
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
