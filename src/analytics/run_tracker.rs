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
//! *complete, independently playable* event — always at the cap, never later,
//! because the buffer arithmetic in `config.rs` is derived from that span.
//! Because chunks split on whole GOP segments, every chunk starts with
//! PAT/PMT + keyframe and decodes on its own.
//!
//! What happens next depends on the segment that crossed the cap, and this is
//! the one rule the split obeys above all: **no chunk ever opens on padding.**
//!
//! - It carries motion: a follow-on chunk opens on it immediately, flagged
//!   [`ClosedRun::continues`], and the run rolls on with no visible boundary.
//! - It is padding: nothing opens. The run enters a *pending* state — alive,
//!   because its quiet window has not elapsed, but with no chunk to put
//!   segments in. Opening one there would produce an event holding nothing but
//!   padding: footage of nothing, recorded and retained as if something had
//!   happened in it.
//!
//! Padding that passes while a run is pending belongs to no event at all. That
//! is the honest answer — it was only ever context for motion, and past the cap
//! there is no motion left for it to be context *for*. If motion returns before
//! the quiet window elapses, the follow-on opens on that motion segment and the
//! chain continues; if the window elapses first, the run just ends, with the
//! chunk the cap already closed as its last.
//!
//! Follow-on chunks cannot pre-pad into the chunk before them: the barrier
//! advances past it as it closes. One that opened lazily, after a pending
//! stretch, does get ordinary pre-padding back over that stretch — segments no
//! event holds, sitting between the barrier and the returning motion, which is
//! exactly what pre-padding is for. The first chunk of a chain keeps normal
//! pre-padding; the last closes through the normal post-padding path.

use std::time::{Duration, Instant};

/// A finished motion run (or one chunk of a long run), ready for event
/// assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosedRun {
    /// Sequence of the first segment of this chunk, motion-positive whichever
    /// kind of chunk this is: the segment that opened the run, the motion
    /// segment that carried it past the cap, or — when the cap crossed on
    /// padding and left the run pending — the motion that returned inside the
    /// quiet window and resumed it.
    pub first_motion_seq: u64,
    /// Sequence of the last segment included in the event (motion or
    /// post-padding).
    pub last_seq: u64,
    /// Earliest sequence pre-padding may reach back to, which keeps this
    /// event's pre-padding clear of the previous event or chunk. A follow-on
    /// that opened at the cap boundary sits directly on that barrier and so
    /// gets no pre-padding at all; one that opened later, on motion returning
    /// after a pending stretch, may pre-pad back over the segments that stretch
    /// left to no event.
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
    /// The cap closed a chunk on a padding segment. The run is still alive —
    /// its quiet window has not elapsed — but nothing is collecting segments,
    /// so the padding passing meanwhile lands in no event. Motion returning
    /// before the window elapses opens the follow-on on itself; otherwise the
    /// run ends here, with the chunk the cap closed as its last.
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

    /// Feed one scored segment, observed at monotonic time `now`. Returns a
    /// `ClosedRun` when this segment ends an open run (post-padding elapsed) or
    /// when it crosses the duration cap — in which case it either starts the
    /// follow-on itself (it carries motion) or leaves the run pending (it does
    /// not).
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

        // The chunk has reached the cap, so it closes here whatever this
        // segment is — its span must stay within the cap. Motion carries
        // straight into a follow-on; padding leaves the run pending, because a
        // chunk opened on padding could only ever hold padding.
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
            // The window elapsed with no motion, so the run is over — and it
            // needs no closing, because the cap already closed its last chunk
            // and advanced the barrier past it. This segment is then judged as
            // if the tracker had been idle all along: motion on it starts a new
            // run rather than continuing a dead one.
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

    /// Which motion period is alive, or `None` when none is — the physical
    /// "is something moving" the Home Assistant sensor mirrors, as opposed to
    /// the event bookkeeping.
    ///
    /// Chunk boundaries do not change it: a run rolling at the cap, or pausing
    /// pending between the cap and the motion that resumes it, is one period
    /// throughout, and the sensor must not flicker at either. What does change
    /// it is a period *ending* and another *beginning*, and the caller cannot
    /// always see that as a change in liveness: a pending run whose window
    /// elapses on a motion segment dies and is replaced inside a single
    /// [`observe`](Self::observe), leaving a tracker that was alive before and
    /// alive after with two distinct runs either side. Comparing this value
    /// across the call is what separates the two — same number, nothing
    /// happened; different number, one period ended and the next began, in
    /// that order.
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

    /// Close an open run immediately (shutdown flush — no post-padding wait),
    /// extending it through `through` when that reaches past the last segment
    /// the analyzer scored.
    ///
    /// The extension is what keeps a recording whole when analysis stops before
    /// the footage does — a decoder that died during the drain, a drain that
    /// hit its bound with segments still unscored. Those segments are ordinary
    /// GOPs sitting in the hot buffer and the event is assembled from a
    /// sequence range, so including them costs nothing and keeps the recording
    /// as long as the camera was recording. What ends early is the analysis:
    /// there are no motion scores or detections over the extension, which is
    /// the honest outcome — nothing looked at it.
    ///
    /// A pending run has nothing to extend and nothing to close: its last chunk
    /// is already sealed and written, and the only thing the flush could add is
    /// an event made of the padding that followed it. So it is sealed as ended
    /// and yields no run.
    ///
    /// `None` extends nothing, which is every caller that is not the shutdown
    /// flush.
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

    /// Close the current chunk at the duration cap and open its follow-on on
    /// `seq`, the motion segment that crossed the cap. The barrier advances
    /// past the closed chunk, so the follow-on cannot pre-pad back into it —
    /// and, opening on the very next segment, gets no pre-padding at all.
    fn chunk(&mut self, seq: u64, now: Instant) -> ClosedRun {
        let closed = self.close().expect("chunk requires an open run");
        self.open_chunk(seq, now, true);
        closed
    }

    /// Close the current chunk at the duration cap without opening a follow-on:
    /// the segment that crossed it is padding, and a chunk opened on padding
    /// would hold nothing else. The run stays alive as [`State::Pending`], its
    /// quiet window still counting down from `last_motion_instant`, and the
    /// padding that passes until it elapses belongs to no event.
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
        assert_eq!(t.flush(None), None);
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
        assert_eq!(t.flush(None), None);
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

    /// Analysis can stop before the footage does: a decoder that died during
    /// the shutdown drain, a drain that ran out its bound with segments still
    /// unscored. The run closes through the footage all the same — those
    /// segments are ordinary GOPs and an event is a sequence range, so the
    /// recording stays as long as the camera was recording and only the
    /// scoring ends early. A run cut at the last *analyzed* segment would be a
    /// recording truncated at exactly the moment the drain exists to protect.
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

    /// The extension only ever reaches forward. A `through` behind what was
    /// analyzed — an eviction race, a buffer read a moment stale — would drop
    /// scored footage back out of the event it was already part of.
    #[test]
    fn the_shutdown_flush_never_shortens_a_run() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        t.observe(5, true, t0);
        t.observe(6, true, t0 + Duration::from_secs(1));
        assert_eq!(t.flush(Some(2)).unwrap().last_seq, 6);
    }

    /// Nothing open is nothing to extend: an extension must not conjure an
    /// event out of quiet footage that never had a motion run.
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
        // Second run: pre-padding may reach back only to seq 1.
        t.observe(2, true, t0 + POST + Duration::from_secs(2));
        let second = t.flush(None).unwrap();
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

    /// The shape `config.rs` warns about and pins: a quiet window as wide as
    /// the hot buffer, with a cap well under it. What makes that warning true
    /// is the cap winning the race every time — a chunk holding motion is
    /// assembled while its own footage is still resident. It only wins if it
    /// fires on padding too: a chunk left open through the whole quiet window
    /// would be older than the buffer when it finally closed, and the motion
    /// inside it would have been evicted before assembly could reach it.
    #[test]
    fn the_cap_closes_a_motion_chunk_before_a_buffer_wide_quiet_window_can() {
        const WIDE_POST: Duration = Duration::from_secs(600);
        let mut t = RunTracker::new(WIDE_POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + Duration::from_secs(30)), None);
        assert_eq!(t.observe(2, false, t0 + Duration::from_secs(60)), None);
        // 480s of quiet window still to run, and the chunk closes anyway — at
        // the cap, with every segment in it younger than the cap.
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
        // The run is pending, not finished: motion has not been quiet long
        // enough to end it.
        assert!(t.is_open());
    }

    /// The cap closing on padding must not open a follow-on to put that padding
    /// in. Such an event holds no motion at all — footage of nothing, recorded
    /// and retained as if something had happened in it. The padding belongs to
    /// no event, and the run ends with the chunk the cap closed as its last.
    #[test]
    fn padding_past_the_cap_never_opens_a_motionless_follow_on() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        let last_motion = t0 + CAP - Duration::from_secs(2);
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, last_motion), None);
        // The cap crosses on padding: the chunk closes here, nothing opens.
        let closed = t.observe(2, false, t0 + CAP).unwrap();
        assert_eq!(closed.last_seq, 1);
        assert!(!closed.continues);
        // Padding keeps arriving inside the quiet window and lands nowhere.
        assert_eq!(t.observe(3, false, t0 + CAP + Duration::from_secs(1)), None);
        assert_eq!(t.observe(4, false, t0 + CAP + Duration::from_secs(5)), None);
        // The window elapses: the run ends without a second event.
        assert_eq!(
            t.observe(5, false, last_motion + POST + Duration::from_nanos(1)),
            None
        );
        assert!(!t.is_open());
        assert_eq!(t.flush(None), None);
    }

    /// Motion returning inside the quiet window is the same run: the follow-on
    /// opens on that motion segment and carries `continues`. Its pre-padding
    /// may reach back over the segments no chunk took — that is what
    /// pre-padding is — but not into the chunk the cap closed, which the
    /// barrier still guards.
    #[test]
    fn motion_returning_after_a_cap_on_padding_continues_the_chain() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.observe(0, true, t0), None);
        assert_eq!(t.observe(1, true, t0 + CAP - Duration::from_secs(2)), None);
        assert_eq!(t.observe(2, false, t0 + CAP).unwrap().last_seq, 1);
        assert_eq!(t.observe(3, false, t0 + CAP + Duration::from_secs(1)), None);
        // Still inside the window: this opens the follow-on, on seq 4.
        assert_eq!(t.observe(4, true, t0 + CAP + Duration::from_secs(2)), None);
        let follow = t.flush(None).unwrap();
        assert_eq!(follow.first_motion_seq, 4);
        assert_eq!(follow.min_start_seq, 2);
        assert!(follow.continues);
    }

    /// A pending run whose window elapses is simply over. Motion after that is
    /// a new run — chaining it onto a run that ended while nothing was
    /// recording would claim a continuity that does not exist.
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

    /// The shutdown flush has nothing to close on a pending run: its last chunk
    /// is sealed and written, and the only event left to make would be one of
    /// pure padding. Extending it through unscored footage would make that
    /// event longer, not truer.
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

    /// The period is the physical motion, not the event bookkeeping: rolling a
    /// chunk at the cap and pausing pending on padding both leave it alone,
    /// while a pending run dying and another opening in its place changes it —
    /// which is the only way the caller can tell those two runs apart, since
    /// the tracker is alive on both sides of that single call.
    #[test]
    fn the_motion_period_changes_only_when_one_run_gives_way_to_another() {
        let mut t = RunTracker::new(POST, CAP);
        let t0 = base();
        assert_eq!(t.motion_period(), None);
        t.observe(0, true, t0);
        let first = t.motion_period().expect("a run is open");
        // A cap boundary with motion: same period.
        assert!(t.observe(1, true, t0 + CAP).is_some());
        assert_eq!(t.motion_period(), Some(first));
        // A cap boundary on padding, leaving the run pending: same period.
        let last_motion = t0 + CAP + CAP - Duration::from_secs(2);
        assert_eq!(t.observe(2, true, last_motion), None);
        assert!(t.observe(3, false, t0 + CAP + CAP).is_some());
        assert_eq!(t.motion_period(), Some(first));
        // The window elapses on a motion segment: that run ends and another
        // begins inside the one call, and only the period says so.
        let resumed = last_motion + POST + Duration::from_secs(1);
        t.observe(4, true, resumed);
        assert!(t.is_open());
        assert_ne!(t.motion_period(), Some(first));
        // Post-padding elapsing ends the period outright.
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
        let closed = t.flush(None).unwrap();
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
