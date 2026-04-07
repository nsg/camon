use std::collections::VecDeque;

use opencv::{
    core::{Mat, Rect, Vector},
    imgcodecs,
    prelude::*,
    video::{self, BackgroundSubtractorTrait},
    Result as CvResult,
};

const HISTOGRAM_BUCKETS: usize = 100;
const MIN_SAMPLES_FOR_THRESHOLD: u64 = 1000;
const WINDOW_HOURS: usize = 3;

pub struct ScoreHistogram {
    buckets: [u64; HISTOGRAM_BUCKETS],
    window: VecDeque<u8>,
    window_size: usize,
    target_percentile: f32,
    default_threshold: f32,
}

impl ScoreHistogram {
    pub fn new(target_percentile: f32, default_threshold: f32, sample_fps: u32) -> Self {
        let window_size = WINDOW_HOURS * 60 * 60 * sample_fps as usize;
        Self {
            buckets: [0; HISTOGRAM_BUCKETS],
            window: VecDeque::with_capacity(window_size),
            window_size,
            target_percentile,
            default_threshold,
        }
    }

    pub fn record(&mut self, score: f32) {
        if score <= 0.0 {
            return;
        }
        let bucket = ((score * HISTOGRAM_BUCKETS as f32) as usize).min(HISTOGRAM_BUCKETS - 1);

        if self.window.len() == self.window_size {
            let old = self.window.pop_front().unwrap() as usize;
            self.buckets[old] -= 1;
        }

        self.window.push_back(bucket as u8);
        self.buckets[bucket] += 1;
    }

    pub fn threshold(&self) -> f32 {
        let total = self.window.len() as u64;
        if total < MIN_SAMPLES_FOR_THRESHOLD {
            return self.default_threshold;
        }

        let target_count = (total as f32 * self.target_percentile) as u64;
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
        self.window.len() as u64
    }
}

const WARMUP_FRAMES: u32 = 100;
const SCENE_CHANGE_RATIO: f32 = 0.8;

// MOG2 history: number of frames used to build the background model.
// At ~30fps, 9000 frames ≈ 5 minutes. Persistent motion (tree sway)
// gets absorbed into the background model over this window.
const MOG2_HISTORY: i32 = 9000;

// MOG2 learning rate: controls how fast new patterns are absorbed.
// At 0.003, a pixel needs ~333 consistent frames (~11s at 30fps) to
// enter the background — fast enough to absorb tree sway, slow enough
// that a person lingering won't fade out.
const MOG2_LEARNING_RATE: f64 = 0.003;

pub struct MotionDetector {
    mog2: opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
    fg_mask: Mat,
    learning_rate: f64,
    frames_since_stable: u32,
}

impl MotionDetector {
    pub fn new() -> CvResult<Self> {
        let mog2 = video::create_background_subtractor_mog2(MOG2_HISTORY, 16.0, true)?;
        let fg_mask = Mat::default();

        Ok(Self {
            mog2,
            fg_mask,
            learning_rate: MOG2_LEARNING_RATE,
            frames_since_stable: 0,
        })
    }

    pub fn process_frame(&mut self, frame: &impl opencv::core::ToInputArray) -> CvResult<f32> {
        BackgroundSubtractorTrait::apply(
            &mut self.mog2,
            frame,
            &mut self.fg_mask,
            self.learning_rate,
        )?;

        let total_pixels = self.fg_mask.rows() * self.fg_mask.cols();
        if total_pixels == 0 {
            return Ok(0.0);
        }

        let fg_pixels = opencv::core::count_non_zero(&self.fg_mask)? as f32;
        let foreground_ratio = fg_pixels / total_pixels as f32;

        if foreground_ratio >= SCENE_CHANGE_RATIO {
            self.frames_since_stable = 0;
            return Ok(0.0);
        }

        self.frames_since_stable += 1;

        if self.frames_since_stable < WARMUP_FRAMES {
            return Ok(0.0);
        }

        Ok((foreground_ratio * 10.0).min(1.0))
    }

    /// Returns the current fg_mask as JPEG — shows what MOG2 considers foreground.
    pub fn stability_map_jpeg(&self) -> Option<Vec<u8>> {
        self.fg_mask_jpeg()
    }

    /// Returns MOG2's learned background model as JPEG.
    pub fn background_jpeg(&self) -> Option<Vec<u8>> {
        let mut bg = Mat::default();
        self.mog2.get_background_image(&mut bg).ok()?;
        if bg.empty() {
            return None;
        }
        let mut buf = Vector::<u8>::new();
        let params = Vector::<i32>::new();
        imgcodecs::imencode(".jpg", &bg, &mut buf, &params).ok()?;
        Some(buf.to_vec())
    }

    pub fn fg_mask_jpeg(&self) -> Option<Vec<u8>> {
        if self.fg_mask.empty() {
            return None;
        }
        let mut buf = Vector::<u8>::new();
        let params = Vector::<i32>::new();
        imgcodecs::imencode(".jpg", &self.fg_mask, &mut buf, &params).ok()?;
        Some(buf.to_vec())
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
