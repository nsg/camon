use opencv::{
    core::{Mat, Rect, Vector},
    prelude::*,
    video::{self, BackgroundSubtractorTrait},
    Result as CvResult,
};

const HISTOGRAM_BUCKETS: usize = 100;
const MIN_SAMPLES_FOR_THRESHOLD: u64 = 1000;

pub struct ScoreHistogram {
    buckets: [u64; HISTOGRAM_BUCKETS],
    total_samples: u64,
    target_percentile: f32,
    default_threshold: f32,
}

impl ScoreHistogram {
    pub fn new(target_percentile: f32, default_threshold: f32) -> Self {
        Self {
            buckets: [0; HISTOGRAM_BUCKETS],
            total_samples: 0,
            target_percentile,
            default_threshold,
        }
    }

    pub fn record(&mut self, score: f32) {
        if score <= 0.0 {
            return;
        }
        let bucket = ((score * HISTOGRAM_BUCKETS as f32) as usize).min(HISTOGRAM_BUCKETS - 1);
        self.buckets[bucket] += 1;
        self.total_samples += 1;
    }

    pub fn threshold(&self) -> f32 {
        if self.total_samples < MIN_SAMPLES_FOR_THRESHOLD {
            return self.default_threshold;
        }

        let target_count = (self.total_samples as f32 * self.target_percentile) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return (i as f32 + 0.5) / HISTOGRAM_BUCKETS as f32;
            }
        }

        self.default_threshold
    }

    pub fn samples(&self) -> u64 {
        self.total_samples
    }
}

const WARMUP_FRAMES: u32 = 100;

pub struct MotionDetector {
    mog2: opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
    fg_mask: Mat,
    learning_rate: f64,
    frames_processed: u32,
}

impl MotionDetector {
    pub fn new() -> CvResult<Self> {
        let mog2 = video::create_background_subtractor_mog2(500, 16.0, true)?;
        let fg_mask = Mat::default();

        Ok(Self {
            mog2,
            fg_mask,
            learning_rate: -1.0,
            frames_processed: 0,
        })
    }

    pub fn process_frame(&mut self, frame: &impl opencv::core::ToInputArray) -> CvResult<f32> {
        BackgroundSubtractorTrait::apply(
            &mut self.mog2,
            frame,
            &mut self.fg_mask,
            self.learning_rate,
        )?;

        self.frames_processed += 1;

        // During warmup, return zero score to let background model stabilize
        if self.frames_processed < WARMUP_FRAMES {
            return Ok(0.0);
        }

        let total_pixels = self.fg_mask.rows() * self.fg_mask.cols();
        if total_pixels == 0 {
            return Ok(0.0);
        }

        let fg_pixels = opencv::core::count_non_zero(&self.fg_mask)? as f32;
        let foreground_ratio = fg_pixels / total_pixels as f32;

        Ok((foreground_ratio * 10.0).min(1.0))
    }

    pub fn motion_bbox(&self) -> Option<Rect> {
        let mut points = Vector::<opencv::core::Point>::new();
        opencv::core::find_non_zero(&self.fg_mask, &mut points).ok()?;
        if points.is_empty() {
            return None;
        }
        let rect = opencv::imgproc::bounding_rect(&points).ok()?;
        if rect.width == 0 || rect.height == 0 {
            return None;
        }
        Some(rect)
    }
}
