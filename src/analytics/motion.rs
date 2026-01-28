use opencv::{
    core::{Mat, Rect, Vector},
    imgproc,
    prelude::*,
    video::{self, BackgroundSubtractorTrait},
    Result as CvResult,
};

pub struct MotionScore {
    pub score: f32,
    pub regions: Vec<Rect>,
}

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

    pub fn process_frame(
        &mut self,
        frame: &impl opencv::core::ToInputArray,
    ) -> CvResult<MotionScore> {
        BackgroundSubtractorTrait::apply(
            &mut self.mog2,
            frame,
            &mut self.fg_mask,
            self.learning_rate,
        )?;

        self.frames_processed += 1;

        // During warmup, return zero score to let background model stabilize
        if self.frames_processed < WARMUP_FRAMES {
            return Ok(MotionScore {
                score: 0.0,
                regions: Vec::new(),
            });
        }

        let total_pixels = self.fg_mask.rows() * self.fg_mask.cols();
        if total_pixels == 0 {
            return Ok(MotionScore {
                score: 0.0,
                regions: Vec::new(),
            });
        }

        let fg_pixels = opencv::core::count_non_zero(&self.fg_mask)? as f32;
        let foreground_ratio = fg_pixels / total_pixels as f32;

        let score = (foreground_ratio * 10.0).min(1.0);

        let regions = self.find_motion_regions()?;

        Ok(MotionScore { score, regions })
    }

    fn find_motion_regions(&self) -> CvResult<Vec<Rect>> {
        let mut contours: Vector<Vector<opencv::core::Point>> = Vector::new();
        imgproc::find_contours(
            &self.fg_mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )?;

        let mut regions = Vec::new();
        let min_area = 500.0;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            if area >= min_area {
                let rect = imgproc::bounding_rect(&contour)?;
                regions.push(rect);
            }
        }

        Ok(regions)
    }
}
