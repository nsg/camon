//! Shared retry policy: backoff schedule, failure-report throttling, and
//! jitter, kept in one place so callers cannot drift apart.

use std::time::Duration;

/// Wait `start` after the first failure, doubling each further one up to `max`.
#[derive(Clone, Copy)]
pub struct RetrySchedule {
    pub start: Duration,
    pub max: Duration,
}

impl RetrySchedule {
    /// Saturating: a doubling that wrapped would turn the longest backoff into
    /// no backoff at all.
    pub fn next(self, current: Duration) -> Duration {
        current.saturating_mul(2).min(self.max)
    }
}

/// Cap on how far apart two reports of one unbroken streak may fall: doubling
/// alone would leave hours of silence about something still broken.
pub const MAX_REPORT_GAP: u32 = 60;

/// Consecutive occurrences of one kind of failure, reported at doubling
/// milestones capped by [`MAX_REPORT_GAP`].
pub struct Streak {
    count: u32,
    next_report: u32,
}

impl Default for Streak {
    fn default() -> Self {
        Self {
            count: 0,
            next_report: 1,
        }
    }
}

impl Streak {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Count one occurrence, returning the streak length when it is one worth
    /// reporting and `None` between milestones.
    pub fn record(&mut self) -> Option<u32> {
        self.count += 1;
        if self.count < self.next_report {
            return None;
        }
        self.next_report = self.count + self.count.min(MAX_REPORT_GAP);
        Some(self.count)
    }
}

/// Apply +/-20% jitter to a delay (in ms). A failure that hits every camera at
/// once would otherwise put every retry on the same tick.
pub fn apply_jitter(base_ms: u64, rand: u64) -> u64 {
    if base_ms == 0 {
        return 0;
    }
    let span = base_ms / 5; // 20%
    if span == 0 {
        return base_ms;
    }
    let offset = (rand % (2 * span + 1)) as i64 - span as i64;
    (base_ms as i64 + offset).max(0) as u64
}

/// A random-enough number without a dependency: `RandomState` is already
/// process-random.
pub fn jitter_source() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish()
}

/// [`apply_jitter`] over a `Duration`.
pub fn jittered(delay: Duration) -> Duration {
    Duration::from_millis(apply_jitter(delay.as_millis() as u64, jitter_source()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_doubles_then_caps() {
        let schedule = RetrySchedule {
            start: Duration::from_secs(5),
            max: Duration::from_secs(60),
        };
        let steps: Vec<u64> =
            std::iter::successors(Some(schedule.start), |&d| Some(schedule.next(d)))
                .take(6)
                .map(|d| d.as_secs())
                .collect();
        assert_eq!(steps, vec![5, 10, 20, 40, 60, 60]);
    }

    #[test]
    fn schedule_doubling_saturates() {
        let schedule = RetrySchedule {
            start: Duration::from_secs(5),
            max: Duration::from_secs(60),
        };
        assert_eq!(schedule.next(Duration::MAX), schedule.max);
    }

    #[test]
    fn streak_reports_on_a_widening_schedule() {
        let mut streak = Streak::new();
        let milestones: Vec<u32> = (0..400).filter_map(|_| streak.record()).collect();
        assert_eq!(&milestones[..7], &[1, 2, 4, 8, 16, 32, 64]);
        assert!(
            milestones.windows(2).all(|p| p[1] - p[0] <= MAX_REPORT_GAP),
            "{milestones:?}"
        );
    }

    #[test]
    fn streak_reset_starts_reporting_again_at_once() {
        let mut streak = Streak::new();
        assert_eq!(streak.record(), Some(1));
        assert_eq!(streak.record(), Some(2));
        assert_eq!(streak.record(), None);
        streak.reset();
        assert_eq!(streak.record(), Some(1));
    }

    #[test]
    fn jitter_stays_within_twenty_percent() {
        let base_ms = 20_000;
        let span = base_ms / 5;
        for rand in [0u64, 1, 7, 12345, u64::MAX] {
            let out = apply_jitter(base_ms, rand);
            assert!(out >= base_ms - span, "{out} below lower bound");
            assert!(out <= base_ms + span, "{out} above upper bound");
        }
    }

    #[test]
    fn jitter_zero_base_is_zero() {
        assert_eq!(apply_jitter(0, 999), 0);
    }

    #[test]
    fn jitter_leaves_a_delay_too_short_to_split_alone() {
        assert_eq!(apply_jitter(4, 999), 4);
    }

    #[test]
    fn jittered_duration_stays_near_its_base() {
        let base = Duration::from_secs(60);
        for _ in 0..50 {
            let out = jittered(base);
            assert!(out >= base * 4 / 5 && out <= base * 6 / 5, "{out:?}");
        }
    }
}
