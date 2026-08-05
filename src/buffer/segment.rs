use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The longest a single GOP can plausibly have taken. The segmenter's
/// no-segment watchdog (`NO_SEGMENT_TIMEOUT_SECS` in [`crate::camera::rtsp`])
/// tears a stream down once a minute passes without a finished segment, checked
/// between half-second polls, so a real segment reaches a little over 60 s and
/// no more. The slack to 120 s is there so this only ever refuses a
/// measurement, never footage: a span above it means the instrument lied (a
/// wrapped media PTS, or a process stalled so long that the elapsed time holds
/// no frames), not that two minutes of video arrived.
///
/// Raising that watchdog past this bound would start refusing real segments; a
/// compile-time assertion beside the watchdog constant holds the ordering so
/// the two cannot drift apart silently.
pub(crate) const MAX_SEGMENT_SPAN_NS: u64 = 120 * 1_000_000_000;

/// Ticks per second of the MPEG-TS presentation timestamp clock.
const PTS_HZ: u64 = 90_000;

#[derive(Debug, Clone)]
pub struct GopSegment {
    /// Wall clock nanoseconds at which the segmenter cut this GOP: the event's
    /// identity on disk, not a duration anchor. See [`wall_clock_ns`] for what
    /// it reads on a box whose clock is wrong.
    pub start_pts: u64,
    pub duration_ns: u64,
    /// Shared MPEG-TS bytes; cloning a segment only bumps the refcount.
    pub data: Arc<Vec<u8>>,
    pub frame_count: u32,
}

impl GopSegment {
    pub fn new(start_pts: u64) -> Self {
        Self {
            start_pts,
            duration_ns: 0,
            data: Arc::new(Vec::new()),
            frame_count: 0,
        }
    }

    /// Close the segment, measuring how long it ran.
    ///
    /// `open_for` is monotonic — how long the segmenter held this GOP open,
    /// measured with [`std::time::Instant`] rather than from wall clock stamps
    /// at its two ends. The wall clock can step between those ends, and on a
    /// box with no battery-backed clock it *will*: every stamp reads 0 until
    /// NTP lands, then jumps half a century mid-GOP. Nothing here can tell such
    /// a step apart from elapsed time, so nothing here is measured against it.
    ///
    /// Duration falls through two instruments to none:
    ///
    /// 1. the media PTS delta, preferred because it is the encoder's own
    ///    timeline and what a browser's `currentTime` is built from;
    /// 2. the monotonic span, when there is no usable PTS pair — the first
    ///    segment of every connection has no predecessor, and the 33-bit PTS
    ///    field wraps roughly every 26.5 hours;
    /// 3. zero, meaning "not known", when both read implausibly.
    ///
    /// Both are held to [`MAX_SEGMENT_SPAN_NS`], because both can lie and an
    /// absurd duration is not a cosmetic error: the hot buffer evicts by summed
    /// duration, so one segment claiming hours drains the whole buffer on the
    /// next push, and the same number is written into the event's on-disk name
    /// and into the index entry that has to name the same file.
    pub fn finalize_with_media_pts(
        &mut self,
        open_for: Duration,
        media_pts_ticks: Option<u64>,
        prev_media_pts_ticks: Option<u64>,
    ) {
        let monotonic_ns = u64::try_from(open_for.as_nanos()).unwrap_or(u64::MAX);
        self.duration_ns = match media_span_ns(media_pts_ticks, prev_media_pts_ticks) {
            Some(media_ns) if media_ns <= MAX_SEGMENT_SPAN_NS => media_ns,
            _ if monotonic_ns <= MAX_SEGMENT_SPAN_NS => monotonic_ns,
            _ => 0,
        };
    }
}

/// Nanoseconds between two media PTS readings, or `None` when there is no pair
/// to subtract or the timeline did not advance across it (a PTS wrap reads as
/// time running backwards).
///
/// Saturating, so a delta wide enough to overflow the conversion comes out as
/// an obviously implausible span for the caller's bound to refuse, rather than
/// wrapping into a plausible-looking one.
fn media_span_ns(ticks: Option<u64>, prev_ticks: Option<u64>) -> Option<u64> {
    let (ticks, prev_ticks) = (ticks?, prev_ticks?);
    let delta = ticks.checked_sub(prev_ticks).filter(|delta| *delta > 0)?;
    Some(delta.saturating_mul(1_000_000_000) / PTS_HZ)
}

/// Wall clock nanoseconds since the epoch: the clock segments are stamped with,
/// the clock event start times inherit from those stamps, and so the only one
/// their age can be measured against. The single reader of the system clock in
/// the recording path — camera, storage and retention all measure the same
/// timeline, so they cannot disagree about what "now" is.
///
/// A clock reading nonsense never panics and never wraps. Before 1970 it reads
/// 0; past 2554, where the nanosecond count no longer fits a `u64`, it reads
/// `u64::MAX`. A box that boots with no idea what time it is must not lose its
/// cameras over it: the alternative is a pipeline that panics on every segment,
/// restarts, and panics again, recording nothing for as long as the clock stays
/// wrong while the process looks healthy.
///
/// What a wrong clock costs, honestly:
///
/// - **Nothing in the recording path.** The hot buffer orders by sequence, not
///   by stamp, and durations come from the instruments in
///   [`GopSegment::finalize_with_media_pts`], neither of which is this clock. So
///   recording, eviction and chunk rolling all work normally on a box that
///   believes it is 1969.
/// - **Live playback, cosmetically.** The HLS playlist carries each segment's
///   stamp as its `EXT-X-PROGRAM-DATE-TIME`, so a box with an unset clock
///   offers a live stream where every segment claims 1970-01-01 and a player
///   seeking by wall clock has nothing to seek by. The media plays: segments
///   are ordered by media sequence and cut at their real durations, and the
///   playlist's `discontinuous` test refuses to read a break in the timeline
///   out of stamps that carry no timeline.
/// - **Identity.** Every event that starts while the clock reads 0 is named
///   `0_{duration_ms}`, so two of them collide whenever their durations match
///   to the millisecond, and the later one replaces the earlier. In event mode
///   that is occasional; in continuous mode, where every chunk is rolled at one
///   configured cap, the durations cluster tightly enough that most chunks
///   collide and most of the footage is lost. The index stays consistent either
///   way — inserting under a held identity replaces the entry and re-charges
///   the byte total, matching a storage that has one object under one name —
///   but the replaced footage is gone.
/// - **Retention, inverted.** Age expiry is inert while the clock reads 0
///   (`now - start` is 0 for everything), so nothing ages out; but the
///   low-space guard still runs, and it deletes oldest-start-first within each
///   tier. Zero-stamped events are the oldest thing on the box by name, so
///   under sustained space pressure a no-RTC box keeps an archive it can never
///   expire and throws away the footage it is recording right now, retaining
///   only a churn window. That is still recording, and still strictly better
///   than a pipeline that panics forever, but an operator running one of these
///   boxes without NTP should know it is what they have. A far-future clock is
///   not the benign mirror of this: its own `u64::MAX` stamps sort newest,
///   saturate to an age of 0 and are the last thing any pass takes, but every
///   genuinely stamped event is now measured against a saturated "now" and
///   reads as older than any retention — so the real archive is what expires,
///   draining at the sweep's per-pass cap. That is the same path a legitimate
///   forward correction takes, and the cap is what keeps it survivable.
pub(crate) fn wall_clock_ns() -> u64 {
    epoch_ns(SystemTime::now())
}

/// Split out from [`wall_clock_ns`] so the nonsense-clock policy can be tested
/// against a constructed clock; `SystemTime::now()` cannot be moved.
fn epoch_ns(now: SystemTime) -> u64 {
    let since_epoch = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(since_epoch.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: u64 = 1_000_000_000;
    const NO_PTS: Option<u64> = None;
    /// One tick short of the 33-bit PTS field wrapping.
    const PTS_MAX: u64 = (1 << 33) - 1;

    fn segment() -> GopSegment {
        GopSegment::new(0)
    }

    /// Media PTS wins whenever the pair describes a span a GOP could have: it
    /// is the encoder's own timeline, and no clock on this box perturbs it.
    #[test]
    fn a_plausible_media_pts_delta_is_preferred_over_the_monotonic_span() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(5), Some(180_000), Some(90_000));
        assert_eq!(seg.duration_ns, SEC);
    }

    /// The first segment of every connection has no predecessor to subtract.
    #[test]
    fn the_monotonic_span_fills_in_when_there_is_no_media_pts_pair() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(2), Some(90_000), None);
        assert_eq!(seg.duration_ns, 2 * SEC);

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(2), NO_PTS, NO_PTS);
        assert_eq!(seg.duration_ns, 2 * SEC);
    }

    /// A PTS that did not advance is no measurement, and the segment falls
    /// through to the monotonic span rather than to a negative duration.
    #[test]
    fn a_media_pts_that_did_not_advance_falls_through_to_the_monotonic_span() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(3), Some(90_000), Some(90_000));
        assert_eq!(seg.duration_ns, 3 * SEC);

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(3), Some(1), Some(PTS_MAX));
        assert_eq!(seg.duration_ns, 3 * SEC);
    }

    /// The far side of a 33-bit PTS wrap: the delta is arithmetically positive
    /// and describes a day and a half on one GOP. Taking it would drain the hot
    /// buffer on the next push and name the event after a duration no footage
    /// could have, so the monotonic span is used instead.
    #[test]
    fn a_media_pts_delta_across_a_wrap_is_refused_for_the_monotonic_span() {
        const {
            assert!(
                PTS_MAX * SEC / PTS_HZ > 26 * 3600 * SEC,
                "26.5 hours of GOP"
            )
        };

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(1), Some(PTS_MAX), Some(0));
        assert_eq!(seg.duration_ns, SEC);
    }

    /// Both instruments implausible leaves the duration unknown, rather than
    /// letting either absurd number through.
    #[test]
    fn a_segment_with_no_believable_instrument_has_no_duration() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(3600), Some(PTS_MAX), Some(0));
        assert_eq!(seg.duration_ns, 0);
    }

    /// A process stalled past the bound — swapped out, or SIGSTOPped — closes
    /// its GOP after elapsed time that holds no frames.
    #[test]
    fn a_monotonic_span_no_gop_could_have_leaves_the_duration_unknown() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(600), NO_PTS, NO_PTS);
        assert_eq!(seg.duration_ns, 0);
    }

    /// The bound is a real edge, not decoration: a segment that took exactly as
    /// long as a segment may take is kept, and one nanosecond more is refused.
    /// It has to sit above the segmenter's ~60 s no-segment watchdog, or a slow
    /// but genuine GOP would lose its duration.
    #[test]
    fn the_span_bound_accepts_its_own_value_and_refuses_anything_past_it() {
        const {
            assert!(
                MAX_SEGMENT_SPAN_NS > 60 * SEC,
                "below the segmenter watchdog"
            )
        };
        let at_bound = Duration::from_nanos(MAX_SEGMENT_SPAN_NS);

        let mut seg = segment();
        seg.finalize_with_media_pts(at_bound, NO_PTS, NO_PTS);
        assert_eq!(seg.duration_ns, MAX_SEGMENT_SPAN_NS);

        let mut seg = segment();
        seg.finalize_with_media_pts(at_bound + Duration::from_nanos(1), NO_PTS, NO_PTS);
        assert_eq!(seg.duration_ns, 0);

        // The media branch is held to the same edge.
        let at_bound_ticks = MAX_SEGMENT_SPAN_NS / SEC * PTS_HZ;
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::ZERO, Some(at_bound_ticks), Some(0));
        assert_eq!(seg.duration_ns, MAX_SEGMENT_SPAN_NS);

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::ZERO, Some(at_bound_ticks + 1), Some(0));
        assert_eq!(seg.duration_ns, 0, "past the bound and taken anyway");
    }

    /// The whole point of the shared clock: a box that believes it is 1969
    /// records with zero stamps instead of taking the process down.
    #[test]
    fn a_clock_set_before_1970_reads_as_zero_rather_than_panicking() {
        let before_the_epoch = UNIX_EPOCH - Duration::from_secs(365 * 24 * 3600);
        assert_eq!(epoch_ns(before_the_epoch), 0);
    }

    /// Past 2554 the nanosecond count outgrows a `u64`. Saturating keeps the
    /// stamp wrong in the direction the clock is wrong; truncating would fold a
    /// garbage far-future clock into a plausible-looking recent stamp.
    #[test]
    fn a_clock_past_2554_reads_as_the_largest_stamp_rather_than_wrapping() {
        let year_9999 = UNIX_EPOCH + Duration::from_secs(253_370_000_000);
        assert_eq!(epoch_ns(year_9999), u64::MAX);
    }

    #[test]
    fn a_clock_between_those_reads_as_nanoseconds_since_the_epoch() {
        let at = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        assert_eq!(epoch_ns(at), 1_700_000_000 * SEC);
    }

    /// Host-independent, deliberately: asserting a nonzero reading here would
    /// fail on the very box this policy exists for.
    #[test]
    fn the_shared_clock_reads_the_system_clock_through_the_same_policy() {
        let before = epoch_ns(SystemTime::now());
        let read = wall_clock_ns();
        let after = epoch_ns(SystemTime::now());
        assert!((before..=after).contains(&read), "{before} {read} {after}");
    }
}
