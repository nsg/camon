use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use opencv::{
    core::{Mat, Rect, Size, Vector, BORDER_CONSTANT},
    imgcodecs, imgproc,
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

// --- Adaptive parameter defaults and bounds ---

const DEFAULT_VAR_THRESHOLD: f64 = 16.0;
const VAR_THRESHOLD_MIN: f64 = 12.0;
const VAR_THRESHOLD_MAX: f64 = 48.0;
const VAR_THRESHOLD_STEP: f64 = 2.0;

const DEFAULT_LEARNING_RATE: f64 = 0.003;
const LEARNING_RATE_MIN: f64 = 0.001;
const LEARNING_RATE_MAX: f64 = 0.006;
const LEARNING_RATE_STEP: f64 = 0.001;

const DEFAULT_MORPH_KERNEL_SIZE: i32 = 5;
const MORPH_KERNEL_MIN: i32 = 3;
const MORPH_KERNEL_MAX: i32 = 9;
const MORPH_KERNEL_STEP: i32 = 2;

const DEFAULT_MIN_CONTOUR_AREA: f64 = 200.0;
const MIN_CONTOUR_AREA_MIN: f64 = 100.0;
const MIN_CONTOUR_AREA_MAX: f64 = 600.0;
const MIN_CONTOUR_AREA_STEP: f64 = 50.0;

// Tuner timing
const TUNER_EVAL_SECS: u64 = 600; // 10 minutes
const TUNER_COOLDOWN_SECS: u64 = 300; // 5 minutes
const TUNER_STARTUP_GRACE_SECS: u64 = 600; // 10 minutes

// Activity rate targets
const ACTIVITY_RATE_HIGH: f32 = 0.25;
const ACTIVITY_RATE_LOW: f32 = 0.05;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunedParams {
    pub var_threshold: f64,
    pub learning_rate: f64,
    pub morph_kernel_size: i32,
    pub min_contour_area: f64,
}

impl Default for TunedParams {
    fn default() -> Self {
        Self {
            var_threshold: DEFAULT_VAR_THRESHOLD,
            learning_rate: DEFAULT_LEARNING_RATE,
            morph_kernel_size: DEFAULT_MORPH_KERNEL_SIZE,
            min_contour_area: DEFAULT_MIN_CONTOUR_AREA,
        }
    }
}

impl TunedParams {
    fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    tracing::warn!(error = %e, "failed to save motion tuner params");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize motion tuner params");
            }
        }
    }
}

struct MotionTuner {
    camera_id: String,
    params: TunedParams,
    params_path: PathBuf,
    segments_total: u32,
    segments_triggered: u32,
    started_at: Instant,
    eval_start: Instant,
    last_adjustment: Option<Instant>,
}

impl MotionTuner {
    fn new(camera_id: String, data_dir: &Path) -> Self {
        let params_path = data_dir.join(&camera_id).join("motion_tuner.json");
        let params = TunedParams::load(&params_path).unwrap_or_default();

        if params != TunedParams::default() {
            tracing::info!(
                camera = %camera_id,
                var_threshold = params.var_threshold,
                learning_rate = format!("{:.4}", params.learning_rate),
                morph_kernel = params.morph_kernel_size,
                min_contour_area = params.min_contour_area,
                "loaded saved motion tuner params"
            );
        }

        let now = Instant::now();
        Self {
            camera_id,
            params,
            params_path,
            segments_total: 0,
            segments_triggered: 0,
            started_at: now,
            eval_start: now,
            last_adjustment: None,
        }
    }

    fn record_segment(&mut self, had_activity: bool) {
        self.segments_total += 1;
        if had_activity {
            self.segments_triggered += 1;
        }
    }

    fn maybe_tune(&mut self) -> Option<ParamChange> {
        let now = Instant::now();

        // Don't tune during startup grace period
        if now.duration_since(self.started_at).as_secs() < TUNER_STARTUP_GRACE_SECS {
            return None;
        }

        // Don't evaluate until the window is full
        if now.duration_since(self.eval_start).as_secs() < TUNER_EVAL_SECS {
            return None;
        }

        // Don't tune during cooldown
        if let Some(last) = self.last_adjustment {
            if now.duration_since(last).as_secs() < TUNER_COOLDOWN_SECS {
                return None;
            }
        }

        if self.segments_total == 0 {
            self.reset_window();
            return None;
        }

        let activity_rate = self.segments_triggered as f32 / self.segments_total as f32;
        let change = if activity_rate > ACTIVITY_RATE_HIGH {
            self.tighten(activity_rate)
        } else if activity_rate < ACTIVITY_RATE_LOW {
            self.loosen(activity_rate)
        } else {
            None
        };

        if change.is_some() {
            self.last_adjustment = Some(now);
            self.params.save(&self.params_path);
        }

        self.reset_window();
        change
    }

    fn tighten(&mut self, activity_rate: f32) -> Option<ParamChange> {
        // Priority order: var_threshold → min_contour_area → morph_kernel → learning_rate
        if self.params.var_threshold + VAR_THRESHOLD_STEP <= VAR_THRESHOLD_MAX {
            let old = self.params.var_threshold;
            self.params.var_threshold += VAR_THRESHOLD_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = old,
                new = self.params.var_threshold,
                "motion tuner: raising variance threshold — too many frames contain motion"
            );
            return Some(ParamChange::VarThreshold(self.params.var_threshold));
        }

        if self.params.min_contour_area + MIN_CONTOUR_AREA_STEP <= MIN_CONTOUR_AREA_MAX {
            let old = self.params.min_contour_area;
            self.params.min_contour_area += MIN_CONTOUR_AREA_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = old,
                new = self.params.min_contour_area,
                "motion tuner: raising minimum blob size — small noise blobs still triggering"
            );
            return Some(ParamChange::MinContourArea(self.params.min_contour_area));
        }

        if self.params.morph_kernel_size + MORPH_KERNEL_STEP <= MORPH_KERNEL_MAX {
            let old = self.params.morph_kernel_size;
            self.params.morph_kernel_size += MORPH_KERNEL_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = old,
                new = self.params.morph_kernel_size,
                "motion tuner: enlarging noise filter kernel — scattered noise persists after filtering"
            );
            return Some(ParamChange::MorphKernel(self.params.morph_kernel_size));
        }

        if self.params.learning_rate + LEARNING_RATE_STEP <= LEARNING_RATE_MAX {
            let old = self.params.learning_rate;
            self.params.learning_rate += LEARNING_RATE_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = format!("{:.4}", old),
                new = format!("{:.4}", self.params.learning_rate),
                "motion tuner: increasing background absorption rate — persistent motion not being learned"
            );
            return Some(ParamChange::LearningRate(self.params.learning_rate));
        }

        tracing::info!(
            camera = %self.camera_id,
            activity_rate = format!("{:.0}%", activity_rate * 100.0),
            "motion tuner: all parameters at maximum — noise level remains high"
        );
        None
    }

    fn loosen(&mut self, activity_rate: f32) -> Option<ParamChange> {
        // Reverse priority: learning_rate → morph_kernel → min_contour_area → var_threshold
        if self.params.learning_rate - LEARNING_RATE_STEP >= LEARNING_RATE_MIN
            && self.params.learning_rate > DEFAULT_LEARNING_RATE
        {
            let old = self.params.learning_rate;
            self.params.learning_rate -= LEARNING_RATE_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = format!("{:.4}", old),
                new = format!("{:.4}", self.params.learning_rate),
                "motion tuner: reducing background absorption rate — scene is quiet, increasing sensitivity"
            );
            return Some(ParamChange::LearningRate(self.params.learning_rate));
        }

        if self.params.morph_kernel_size - MORPH_KERNEL_STEP >= MORPH_KERNEL_MIN
            && self.params.morph_kernel_size > DEFAULT_MORPH_KERNEL_SIZE
        {
            let old = self.params.morph_kernel_size;
            self.params.morph_kernel_size -= MORPH_KERNEL_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = old,
                new = self.params.morph_kernel_size,
                "motion tuner: shrinking noise filter kernel — scene is quiet, increasing sensitivity"
            );
            return Some(ParamChange::MorphKernel(self.params.morph_kernel_size));
        }

        if self.params.min_contour_area - MIN_CONTOUR_AREA_STEP >= MIN_CONTOUR_AREA_MIN
            && self.params.min_contour_area > DEFAULT_MIN_CONTOUR_AREA
        {
            let old = self.params.min_contour_area;
            self.params.min_contour_area -= MIN_CONTOUR_AREA_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = old,
                new = self.params.min_contour_area,
                "motion tuner: lowering minimum blob size — scene is quiet, increasing sensitivity"
            );
            return Some(ParamChange::MinContourArea(self.params.min_contour_area));
        }

        if self.params.var_threshold - VAR_THRESHOLD_STEP >= VAR_THRESHOLD_MIN
            && self.params.var_threshold > DEFAULT_VAR_THRESHOLD
        {
            let old = self.params.var_threshold;
            self.params.var_threshold -= VAR_THRESHOLD_STEP;
            tracing::info!(
                camera = %self.camera_id,
                activity_rate = format!("{:.0}%", activity_rate * 100.0),
                old = old,
                new = self.params.var_threshold,
                "motion tuner: lowering variance threshold — scene is quiet, increasing sensitivity"
            );
            return Some(ParamChange::VarThreshold(self.params.var_threshold));
        }

        None
    }

    fn reset_window(&mut self) {
        self.segments_total = 0;
        self.segments_triggered = 0;
        self.eval_start = Instant::now();
    }
}

enum ParamChange {
    VarThreshold(f64),
    LearningRate(f64),
    MorphKernel(i32),
    MinContourArea(f64),
}

pub struct MotionDetector {
    mog2: opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
    fg_mask: Mat,
    cleaned_mask: Mat,
    morph_kernel: Mat,
    learning_rate: f64,
    min_contour_area: f64,
    frames_since_stable: u32,
    tuner: MotionTuner,
}

impl MotionDetector {
    pub fn new(camera_id: &str, data_dir: &Path) -> CvResult<Self> {
        let tuner = MotionTuner::new(camera_id.to_string(), data_dir);
        let params = &tuner.params;

        let mog2 =
            video::create_background_subtractor_mog2(MOG2_HISTORY, params.var_threshold, true)?;
        let fg_mask = Mat::default();
        let cleaned_mask = Mat::default();
        let morph_kernel = build_morph_kernel(params.morph_kernel_size)?;

        Ok(Self {
            mog2,
            fg_mask,
            cleaned_mask,
            morph_kernel,
            learning_rate: params.learning_rate,
            min_contour_area: params.min_contour_area,
            frames_since_stable: 0,
            tuner,
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

        // Raw ratio for scene-change detection (before cleanup).
        let raw_fg = opencv::core::count_non_zero(&self.fg_mask)? as f32;
        let raw_ratio = raw_fg / total_pixels as f32;

        if raw_ratio >= SCENE_CHANGE_RATIO {
            self.frames_since_stable = 0;
            return Ok(0.0);
        }

        self.frames_since_stable += 1;

        if self.frames_since_stable < WARMUP_FRAMES {
            return Ok(0.0);
        }

        // Morphological opening: erode then dilate to remove isolated noise pixels.
        let anchor = opencv::core::Point::new(-1, -1);
        imgproc::morphology_ex(
            &self.fg_mask,
            &mut self.cleaned_mask,
            imgproc::MORPH_OPEN,
            &self.morph_kernel,
            anchor,
            1,
            BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;

        // Contour area filter: zero out blobs smaller than min_contour_area.
        let mut contours = Vector::<Vector<opencv::core::Point>>::new();
        imgproc::find_contours(
            &self.cleaned_mask.clone(),
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )?;

        let mut motion_mask = Mat::zeros(
            self.fg_mask.rows(),
            self.fg_mask.cols(),
            opencv::core::CV_8UC1,
        )?
        .to_mat()?;
        for i in 0..contours.len() {
            let area = imgproc::contour_area(&contours.get(i)?, false)?;
            if area >= self.min_contour_area {
                imgproc::draw_contours(
                    &mut motion_mask,
                    &contours,
                    i as i32,
                    opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                    imgproc::FILLED,
                    imgproc::LINE_8,
                    &opencv::core::no_array(),
                    i32::MAX,
                    opencv::core::Point::new(0, 0),
                )?;
            }
        }

        // Swap cleaned result into fg_mask so downstream (bbox, debug JPEG) sees it.
        std::mem::swap(&mut self.fg_mask, &mut motion_mask);

        let fg_pixels = opencv::core::count_non_zero(&self.fg_mask)? as f32;
        let foreground_ratio = fg_pixels / total_pixels as f32;

        Ok((foreground_ratio * 10.0).min(1.0))
    }

    /// Report whether a segment had any motion activity (score > 0).
    /// Measures the noise floor independently of the adaptive threshold.
    pub fn report_segment(&mut self, had_activity: bool) -> CvResult<()> {
        self.tuner.record_segment(had_activity);
        if let Some(change) = self.tuner.maybe_tune() {
            self.apply_param_change(change)?;
        }
        Ok(())
    }

    fn apply_param_change(&mut self, change: ParamChange) -> CvResult<()> {
        match change {
            ParamChange::VarThreshold(val) => {
                self.mog2.set_var_threshold(val)?;
            }
            ParamChange::LearningRate(val) => {
                self.learning_rate = val;
            }
            ParamChange::MorphKernel(size) => {
                self.morph_kernel = build_morph_kernel(size)?;
            }
            ParamChange::MinContourArea(val) => {
                self.min_contour_area = val;
            }
        }
        Ok(())
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

fn build_morph_kernel(size: i32) -> CvResult<Mat> {
    imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(size, size),
        opencv::core::Point::new(-1, -1),
    )
}

impl PartialEq for TunedParams {
    fn eq(&self, other: &Self) -> bool {
        self.var_threshold == other.var_threshold
            && self.learning_rate == other.learning_rate
            && self.morph_kernel_size == other.morph_kernel_size
            && self.min_contour_area == other.min_contour_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuner_does_not_tune_during_grace_period() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());

        // Simulate high activity
        for _ in 0..1000 {
            tuner.record_segment(true);
        }
        // Force eval window to be ready
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        // Still in grace period — should not tune
        assert!(tuner.maybe_tune().is_none());
    }

    #[test]
    fn tuner_tightens_on_high_activity() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        // 50% activity rate — well above 25% threshold
        tuner.segments_total = 1000;
        tuner.segments_triggered = 500;

        let change = tuner.maybe_tune();
        assert!(
            matches!(change, Some(ParamChange::VarThreshold(v)) if v == DEFAULT_VAR_THRESHOLD + VAR_THRESHOLD_STEP)
        );
    }

    #[test]
    fn tuner_loosens_on_low_activity() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        // Tighten first so there's something to loosen
        tuner.params.var_threshold = DEFAULT_VAR_THRESHOLD + VAR_THRESHOLD_STEP;

        // 1% activity rate — below 5% threshold
        tuner.segments_total = 1000;
        tuner.segments_triggered = 10;

        let change = tuner.maybe_tune();
        assert!(matches!(change, Some(ParamChange::VarThreshold(v)) if v == DEFAULT_VAR_THRESHOLD));
    }

    #[test]
    fn tuner_does_nothing_in_target_band() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        // 15% activity rate — in the 5-25% band
        tuner.segments_total = 1000;
        tuner.segments_triggered = 150;

        assert!(tuner.maybe_tune().is_none());
    }

    #[test]
    fn tuner_respects_cooldown() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.last_adjustment = Some(Instant::now()); // just adjusted

        tuner.segments_total = 1000;
        tuner.segments_triggered = 500;

        // Should not tune — still in cooldown
        assert!(tuner.maybe_tune().is_none());
    }

    #[test]
    fn tuner_tightening_priority_order() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);

        let set_high_activity = |t: &mut MotionTuner| {
            t.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
            t.last_adjustment = None;
            t.segments_total = 1000;
            t.segments_triggered = 500;
        };

        // First tightening: var_threshold
        set_high_activity(&mut tuner);
        assert!(matches!(
            tuner.maybe_tune(),
            Some(ParamChange::VarThreshold(_))
        ));

        // Max out var_threshold
        tuner.params.var_threshold = VAR_THRESHOLD_MAX;

        // Next: min_contour_area
        set_high_activity(&mut tuner);
        assert!(matches!(
            tuner.maybe_tune(),
            Some(ParamChange::MinContourArea(_))
        ));

        // Max out min_contour_area
        tuner.params.min_contour_area = MIN_CONTOUR_AREA_MAX;

        // Next: morph_kernel
        set_high_activity(&mut tuner);
        assert!(matches!(
            tuner.maybe_tune(),
            Some(ParamChange::MorphKernel(_))
        ));

        // Max out morph_kernel
        tuner.params.morph_kernel_size = MORPH_KERNEL_MAX;

        // Next: learning_rate
        set_high_activity(&mut tuner);
        assert!(matches!(
            tuner.maybe_tune(),
            Some(ParamChange::LearningRate(_))
        ));

        // Max out learning_rate
        tuner.params.learning_rate = LEARNING_RATE_MAX;

        // All maxed — returns None
        set_high_activity(&mut tuner);
        assert!(tuner.maybe_tune().is_none());
    }

    #[test]
    fn tuner_loosening_does_not_go_below_defaults() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        // All at defaults, very low activity
        tuner.segments_total = 1000;
        tuner.segments_triggered = 10;

        // Should not loosen below defaults
        assert!(tuner.maybe_tune().is_none());
    }

    #[test]
    fn tuner_persistence_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let params = TunedParams {
            var_threshold: 24.0,
            learning_rate: 0.004,
            morph_kernel_size: 7,
            min_contour_area: 350.0,
        };
        let path = dir.path().join("test").join("motion_tuner.json");
        params.save(&path);

        let loaded = TunedParams::load(&path).unwrap();
        assert_eq!(params, loaded);
    }
}
