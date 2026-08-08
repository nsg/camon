//! Skip accounting: footage that left the hot buffer unanalyzed, and the
//! rate-limited reporting of it.

use std::time::{Duration, Instant};

/// Segments that left the hot buffer before the analyzer reached them. Their
/// footage is never scored, so the skip is reported rather than absorbed.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct SkippedSegments {
    pub(super) count: u64,
    pub(super) from_seq: u64,
    pub(super) to_seq: u64,
}

impl SkippedSegments {
    /// The gap between the next sequence the analyzer would have processed and
    /// the oldest one still resident, or `None` when it has kept up.
    pub(super) fn between(last_processed: u64, first_resident: u64) -> Option<Self> {
        if last_processed >= first_resident {
            return None;
        }
        Some(Self {
            count: first_resident - last_processed,
            from_seq: last_processed,
            to_seq: first_resident - 1,
        })
    }

    /// The individual sequences that could not be read. Eviction takes the
    /// oldest first, so these are contiguous in practice and the range says so;
    /// `count` is exact either way.
    pub(super) fn of(sequences: &[u64]) -> Option<Self> {
        Some(Self {
            count: sequences.len() as u64,
            from_seq: *sequences.iter().min()?,
            to_seq: *sequences.iter().max()?,
        })
    }

    fn merged(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            from_seq: self.from_seq.min(other.from_seq),
            to_seq: self.to_seq.max(other.to_seq),
        }
    }
}

pub(super) fn merge_skips(
    a: Option<SkippedSegments>,
    b: Option<SkippedSegments>,
) -> Option<SkippedSegments> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.merged(b)),
        (a, b) => a.or(b),
    }
}

/// Shortest gap between two skipped-footage warnings.
pub(super) const SKIP_REPORT_INTERVAL: Duration = Duration::from_secs(30);

/// Accumulates skipped footage between warnings; see [`SKIP_REPORT_INTERVAL`].
#[derive(Default)]
pub(super) struct SkipReporter {
    pending: Option<SkippedSegments>,
    last_report: Option<Instant>,
}

impl SkipReporter {
    /// Fold one poll's skips in, returning the accumulated report when the
    /// interval has passed. The first skip is always reported: a rare one-off
    /// is exactly the case worth seeing immediately.
    pub(super) fn record(
        &mut self,
        skipped: SkippedSegments,
        now: Instant,
    ) -> Option<SkippedSegments> {
        self.pending = Some(merge_skips(self.pending.take(), Some(skipped))?);
        let due = self
            .last_report
            .is_none_or(|at| now.saturating_duration_since(at) >= SKIP_REPORT_INTERVAL);
        if !due {
            return None;
        }
        self.last_report = Some(now);
        self.pending.take()
    }
}
