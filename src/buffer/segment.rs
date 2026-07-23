use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GopSegment {
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

    pub fn finalize_with_media_pts(
        &mut self,
        wall_clock_end: u64,
        media_pts_ticks: Option<u64>,
        prev_media_pts_ticks: Option<u64>,
    ) {
        // Prefer media PTS for duration (aligns with browser currentTime),
        // fall back to wall-clock if PTS is unavailable.
        if let (Some(cur), Some(prev)) = (media_pts_ticks, prev_media_pts_ticks) {
            if cur > prev {
                // PTS is in 90kHz ticks; convert to nanoseconds
                let delta_ticks = cur - prev;
                self.duration_ns = delta_ticks * 1_000_000_000 / 90_000;
                return;
            }
        }
        if wall_clock_end > self.start_pts {
            self.duration_ns = wall_clock_end - self.start_pts;
        }
    }
}
