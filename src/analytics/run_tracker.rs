//! Motion-run lifecycle tracking for event-driven persistence.
//!
//! The analyzer feeds every scored segment (in sequence order) into the
//! tracker. A run opens on the first motion-positive segment, absorbs
//! non-motion segments while they fall within the post-padding window, and
//! closes when a non-motion segment arrives past the window. The closing
//! segment itself is not part of the event. Padding is PTS-based, matching
//! the segment timeline used everywhere else.

/// A finished motion run, ready for event assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosedRun {
    /// Sequence of the first motion-positive segment.
    pub first_motion_seq: u64,
    /// Sequence of the last segment included in the event (motion or
    /// post-padding).
    pub last_seq: u64,
    /// Earliest sequence pre-padding may reach back to. Prevents the
    /// pre-padding of this event from overlapping the previous event.
    pub min_start_seq: u64,
}

struct OpenRun {
    first_motion_seq: u64,
    last_motion_pts: u64,
    last_seq: u64,
}

pub struct RunTracker {
    post_padding_ns: u64,
    open: Option<OpenRun>,
    /// One past the last segment of the previously closed run.
    barrier_seq: u64,
}

impl RunTracker {
    pub fn new(post_padding_ns: u64) -> Self {
        Self {
            post_padding_ns,
            open: None,
            barrier_seq: 0,
        }
    }

    /// Feed one scored segment. Returns a `ClosedRun` when this segment ends
    /// an open run (post-padding elapsed).
    pub fn observe(&mut self, seq: u64, start_pts: u64, has_motion: bool) -> Option<ClosedRun> {
        if has_motion {
            match self.open {
                Some(ref mut run) => {
                    run.last_motion_pts = start_pts;
                    run.last_seq = seq;
                }
                None => {
                    self.open = Some(OpenRun {
                        first_motion_seq: seq,
                        last_motion_pts: start_pts,
                        last_seq: seq,
                    });
                }
            }
            return None;
        }

        let run = self.open.as_mut()?;
        if start_pts.saturating_sub(run.last_motion_pts) <= self.post_padding_ns {
            run.last_seq = seq;
            None
        } else {
            self.close()
        }
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
        };
        self.barrier_seq = run.last_seq + 1;
        Some(closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: u64 = 1_000_000_000;
    const POST: u64 = 10 * SEC;

    #[test]
    fn no_motion_never_opens_a_run() {
        let mut t = RunTracker::new(POST);
        for seq in 0..5 {
            assert_eq!(t.observe(seq, seq * SEC, false), None);
        }
        assert_eq!(t.flush(), None);
    }

    #[test]
    fn motion_opens_run_and_padding_elapse_closes_it() {
        let mut t = RunTracker::new(POST);
        assert_eq!(t.observe(0, 0, false), None);
        assert_eq!(t.observe(1, SEC, true), None);
        assert_eq!(t.observe(2, 2 * SEC, true), None);
        // Non-motion within post-padding keeps the run open.
        assert_eq!(t.observe(3, 3 * SEC, false), None);
        // Non-motion past post-padding closes it; the closing segment is
        // excluded, the padding segment stays included.
        let closed = t.observe(4, 2 * SEC + POST + 1, false).unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 1,
                last_seq: 3,
                min_start_seq: 0,
            }
        );
        assert_eq!(t.flush(), None);
    }

    #[test]
    fn motion_within_padding_continues_the_same_run() {
        let mut t = RunTracker::new(POST);
        assert_eq!(t.observe(0, 0, true), None);
        assert_eq!(t.observe(1, SEC, false), None);
        // New motion inside the padding window extends the run.
        assert_eq!(t.observe(2, 2 * SEC, true), None);
        // Padding now counts from the new motion segment.
        assert_eq!(t.observe(3, 2 * SEC + POST, false), None);
        let closed = t.observe(4, 2 * SEC + POST + 1, false).unwrap();
        assert_eq!(closed.first_motion_seq, 0);
        assert_eq!(closed.last_seq, 3);
    }

    #[test]
    fn flush_closes_open_run_immediately() {
        let mut t = RunTracker::new(POST);
        t.observe(5, 0, true);
        t.observe(6, SEC, false);
        let closed = t.flush().unwrap();
        assert_eq!(
            closed,
            ClosedRun {
                first_motion_seq: 5,
                last_seq: 6,
                min_start_seq: 0,
            }
        );
        assert_eq!(t.flush(), None);
    }

    #[test]
    fn next_run_cannot_pre_pad_into_previous_event() {
        let mut t = RunTracker::new(POST);
        t.observe(0, 0, true);
        let first = t.observe(1, POST + 1, false).unwrap();
        assert_eq!(first.last_seq, 0);
        // Second run: pre-padding may reach back only to seq 1.
        t.observe(2, POST + 2 * SEC, true);
        let second = t.flush().unwrap();
        assert_eq!(second.min_start_seq, 1);
        assert_eq!(second.first_motion_seq, 2);
    }
}
