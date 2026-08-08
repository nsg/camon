use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The longest a single GOP can plausibly have taken: a span above this means the instrument
/// lied (a wrapped media PTS, a process stalled past any frames) rather than that two minutes
/// of video arrived.
pub(crate) const MAX_SEGMENT_SPAN_NS: u64 = 120 * 1_000_000_000;

/// Ticks per second of the MPEG-TS presentation timestamp clock.
const PTS_HZ: u64 = 90_000;

#[derive(Debug, Clone)]
pub struct GopSegment {
    /// Wall clock nanoseconds at which the segmenter cut this GOP: the event's
    /// identity on disk, not a duration anchor.
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

/// Nanoseconds between two media PTS readings, or `None` when there is no pair to subtract or
/// the timeline did not advance across it (a PTS wrap reads as time running backwards).
fn media_span_ns(ticks: Option<u64>, prev_ticks: Option<u64>) -> Option<u64> {
    let (ticks, prev_ticks) = (ticks?, prev_ticks?);
    let delta = ticks.checked_sub(prev_ticks).filter(|delta| *delta > 0)?;
    Some(delta.saturating_mul(1_000_000_000) / PTS_HZ)
}

/// Wall clock nanoseconds since the epoch. The single reader of the system clock in the
/// recording path, so camera, storage and retention cannot disagree about what "now" is.
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
    const PTS_MAX: u64 = (1 << 33) - 1;

    fn segment() -> GopSegment {
        GopSegment::new(0)
    }

    #[test]
    fn a_plausible_media_pts_delta_is_preferred_over_the_monotonic_span() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(5), Some(180_000), Some(90_000));
        assert_eq!(seg.duration_ns, SEC);
    }

    #[test]
    fn the_monotonic_span_fills_in_when_there_is_no_media_pts_pair() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(2), Some(90_000), None);
        assert_eq!(seg.duration_ns, 2 * SEC);

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(2), NO_PTS, NO_PTS);
        assert_eq!(seg.duration_ns, 2 * SEC);
    }

    #[test]
    fn a_media_pts_that_did_not_advance_falls_through_to_the_monotonic_span() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(3), Some(90_000), Some(90_000));
        assert_eq!(seg.duration_ns, 3 * SEC);

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(3), Some(1), Some(PTS_MAX));
        assert_eq!(seg.duration_ns, 3 * SEC);
    }

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

    #[test]
    fn a_segment_with_no_believable_instrument_has_no_duration() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(3600), Some(PTS_MAX), Some(0));
        assert_eq!(seg.duration_ns, 0);
    }

    #[test]
    fn a_monotonic_span_no_gop_could_have_leaves_the_duration_unknown() {
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::from_secs(600), NO_PTS, NO_PTS);
        assert_eq!(seg.duration_ns, 0);
    }

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

        let at_bound_ticks = MAX_SEGMENT_SPAN_NS / SEC * PTS_HZ;
        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::ZERO, Some(at_bound_ticks), Some(0));
        assert_eq!(seg.duration_ns, MAX_SEGMENT_SPAN_NS);

        let mut seg = segment();
        seg.finalize_with_media_pts(Duration::ZERO, Some(at_bound_ticks + 1), Some(0));
        assert_eq!(seg.duration_ns, 0, "past the bound and taken anyway");
    }

    #[test]
    fn a_clock_set_before_1970_reads_as_zero_rather_than_panicking() {
        let before_the_epoch = UNIX_EPOCH - Duration::from_secs(365 * 24 * 3600);
        assert_eq!(epoch_ns(before_the_epoch), 0);
    }

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

    #[test]
    fn the_shared_clock_reads_the_system_clock_through_the_same_policy() {
        let before = epoch_ns(SystemTime::now());
        let read = wall_clock_ns();
        let after = epoch_ns(SystemTime::now());
        assert!((before..=after).contains(&read), "{before} {read} {after}");
    }
}
