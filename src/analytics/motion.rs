use std::path::{Path, PathBuf};
use std::time::Instant;

use opencv::{
    core::{Mat, Rect, Size, Vector, BORDER_CONSTANT},
    imgcodecs, imgproc,
    prelude::*,
    video::{self, BackgroundSubtractorTrait},
    Result as CvResult,
};

const WARMUP_FRAMES: u32 = 100;
const SCENE_CHANGE_RATIO: f32 = 0.8;

// MOG2 history: number of frames used to build the background model.
// At ~30fps, 9000 frames ≈ 5 minutes. Persistent motion (tree sway)
// gets absorbed into the background model over this window.
const MOG2_HISTORY: i32 = 9000;

// --- Adaptive parameter defaults and bounds ---

const DEFAULT_VAR_THRESHOLD: f64 = 16.0;
const VAR_THRESHOLD_MAX: f64 = 96.0;
const VAR_THRESHOLD_STEP: f64 = 4.0;

const DEFAULT_LEARNING_RATE: f64 = 0.003;
const LEARNING_RATE_MAX: f64 = 0.010;
const LEARNING_RATE_STEP: f64 = 0.001;

const DEFAULT_MORPH_KERNEL_SIZE: i32 = 5;
const MORPH_KERNEL_MAX: i32 = 15;
const MORPH_KERNEL_STEP: i32 = 2;

// On a 320x240 analysis frame a person is ~4000px. Max 2000 leaves a
// safe margin while filtering out wind/leaf blobs that are a few hundred px.
const DEFAULT_MIN_CONTOUR_AREA: f64 = 200.0;
const MIN_CONTOUR_AREA_MAX: f64 = 2000.0;
const MIN_CONTOUR_AREA_STEP: f64 = 100.0;

// Tuner timing
const TUNER_EVAL_SECS: u64 = 600; // 10 minutes
const TUNER_COOLDOWN_SECS: u64 = 300; // 5 minutes
const TUNER_STARTUP_GRACE_SECS: u64 = 600; // 10 minutes

// Noise threshold: movement-only events per hour above this trigger tightening.
// Events with object detections are excluded (they're real, not noise).
const NOISE_EVENTS_PER_HOUR_HIGH: f32 = 10.0;

// How many consecutive quiet windows (zero noise) before relaxing one step.
// At 10-min eval windows, 6 windows = 1 hour of silence before relaxing.
const RELAX_QUIET_WINDOWS: u32 = 6;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TunerStats {
    pub var_threshold: f64,
    pub learning_rate: f64,
    pub morph_kernel_size: i32,
    pub min_contour_area: f64,
    pub noise_events: u32,
    pub quiet_windows: u32,
}

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

// Tightening weights: how fast each parameter advances per cycle.
// 1.0 = full step every cycle. Lower = slower.
const WEIGHT_VAR_THRESHOLD: f32 = 1.0;
const WEIGHT_MIN_CONTOUR_AREA: f32 = 0.5;
const WEIGHT_MORPH_KERNEL: f32 = 0.33;
const WEIGHT_LEARNING_RATE: f32 = 0.25;

struct MotionTuner {
    camera_id: String,
    params: TunedParams,
    params_path: PathBuf,
    noise_events: u32,
    quiet_windows: u32,
    started_at: Instant,
    eval_start: Instant,
    last_adjustment: Option<Instant>,
    // Accumulators for weighted stepping
    acc_var_threshold: f32,
    acc_min_contour_area: f32,
    acc_morph_kernel: f32,
    acc_learning_rate: f32,
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
            noise_events: 0,
            quiet_windows: 0,
            started_at: now,
            eval_start: now,
            last_adjustment: None,
            acc_var_threshold: 0.0,
            acc_min_contour_area: 0.0,
            acc_morph_kernel: 0.0,
            acc_learning_rate: 0.0,
        }
    }

    fn record_motion_event(&mut self) {
        self.noise_events += 1;
    }

    fn record_positive_detection(&mut self) {
        self.noise_events = self.noise_events.saturating_sub(1);
    }

    fn maybe_tune(&mut self) -> Vec<ParamChange> {
        let now = Instant::now();

        if now.duration_since(self.started_at).as_secs() < TUNER_STARTUP_GRACE_SECS {
            return Vec::new();
        }

        let window_secs = now.duration_since(self.eval_start).as_secs();
        if window_secs < TUNER_EVAL_SECS {
            return Vec::new();
        }

        if let Some(last) = self.last_adjustment {
            if now.duration_since(last).as_secs() < TUNER_COOLDOWN_SECS {
                return Vec::new();
            }
        }

        let window_hours = window_secs as f32 / 3600.0;
        let noise_per_hour = self.noise_events as f32 / window_hours;

        let changes = if noise_per_hour > NOISE_EVENTS_PER_HOUR_HIGH {
            self.quiet_windows = 0;
            self.tighten(noise_per_hour)
        } else if self.noise_events == 0 && self.params != TunedParams::default() {
            self.quiet_windows += 1;
            if self.quiet_windows >= RELAX_QUIET_WINDOWS {
                self.quiet_windows = 0;
                self.relax().into_iter().collect()
            } else {
                Vec::new()
            }
        } else {
            self.quiet_windows = 0;
            Vec::new()
        };

        if !changes.is_empty() {
            self.last_adjustment = Some(now);
            self.params.save(&self.params_path);
        }

        self.reset_window();
        changes
    }

    fn tighten(&mut self, noise_per_hour: f32) -> Vec<ParamChange> {
        let mut changes = Vec::new();

        self.acc_var_threshold += WEIGHT_VAR_THRESHOLD;
        self.acc_min_contour_area += WEIGHT_MIN_CONTOUR_AREA;
        self.acc_morph_kernel += WEIGHT_MORPH_KERNEL;
        self.acc_learning_rate += WEIGHT_LEARNING_RATE;

        if self.acc_var_threshold >= 1.0
            && self.params.var_threshold + VAR_THRESHOLD_STEP <= VAR_THRESHOLD_MAX
        {
            self.acc_var_threshold -= 1.0;
            let old = self.params.var_threshold;
            self.params.var_threshold += VAR_THRESHOLD_STEP;
            tracing::info!(
                camera = %self.camera_id,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                old = old, new = self.params.var_threshold,
                "motion tuner: raising variance threshold"
            );
            changes.push(ParamChange::VarThreshold(self.params.var_threshold));
        }

        if self.acc_min_contour_area >= 1.0
            && self.params.min_contour_area + MIN_CONTOUR_AREA_STEP <= MIN_CONTOUR_AREA_MAX
        {
            self.acc_min_contour_area -= 1.0;
            let old = self.params.min_contour_area;
            self.params.min_contour_area += MIN_CONTOUR_AREA_STEP;
            tracing::info!(
                camera = %self.camera_id,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                old = old, new = self.params.min_contour_area,
                "motion tuner: raising minimum blob size"
            );
            changes.push(ParamChange::MinContourArea(self.params.min_contour_area));
        }

        if self.acc_morph_kernel >= 1.0
            && self.params.morph_kernel_size + MORPH_KERNEL_STEP <= MORPH_KERNEL_MAX
        {
            self.acc_morph_kernel -= 1.0;
            let old = self.params.morph_kernel_size;
            self.params.morph_kernel_size += MORPH_KERNEL_STEP;
            tracing::info!(
                camera = %self.camera_id,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                old = old, new = self.params.morph_kernel_size,
                "motion tuner: enlarging noise filter kernel"
            );
            changes.push(ParamChange::MorphKernel(self.params.morph_kernel_size));
        }

        if self.acc_learning_rate >= 1.0
            && self.params.learning_rate + LEARNING_RATE_STEP <= LEARNING_RATE_MAX
        {
            self.acc_learning_rate -= 1.0;
            let old = self.params.learning_rate;
            self.params.learning_rate += LEARNING_RATE_STEP;
            tracing::info!(
                camera = %self.camera_id,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                old = format!("{:.4}", old), new = format!("{:.4}", self.params.learning_rate),
                "motion tuner: increasing background absorption rate"
            );
            changes.push(ParamChange::LearningRate(self.params.learning_rate));
        }

        if changes.is_empty() {
            tracing::info!(
                camera = %self.camera_id,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                "motion tuner: all parameters at maximum"
            );
        }

        changes
    }

    fn relax(&mut self) -> Option<ParamChange> {
        // Reverse priority: undo the most aggressive changes first
        if self.params.learning_rate > DEFAULT_LEARNING_RATE {
            let old = self.params.learning_rate;
            self.params.learning_rate =
                (self.params.learning_rate - LEARNING_RATE_STEP).max(DEFAULT_LEARNING_RATE);
            tracing::info!(
                camera = %self.camera_id,
                old = format!("{:.4}", old),
                new = format!("{:.4}", self.params.learning_rate),
                "motion tuner: relaxing background absorption rate after sustained quiet"
            );
            return Some(ParamChange::LearningRate(self.params.learning_rate));
        }

        if self.params.morph_kernel_size > DEFAULT_MORPH_KERNEL_SIZE {
            let old = self.params.morph_kernel_size;
            self.params.morph_kernel_size =
                (self.params.morph_kernel_size - MORPH_KERNEL_STEP).max(DEFAULT_MORPH_KERNEL_SIZE);
            tracing::info!(
                camera = %self.camera_id,
                old = old,
                new = self.params.morph_kernel_size,
                "motion tuner: relaxing noise filter kernel after sustained quiet"
            );
            return Some(ParamChange::MorphKernel(self.params.morph_kernel_size));
        }

        if self.params.min_contour_area > DEFAULT_MIN_CONTOUR_AREA {
            let old = self.params.min_contour_area;
            self.params.min_contour_area = (self.params.min_contour_area - MIN_CONTOUR_AREA_STEP)
                .max(DEFAULT_MIN_CONTOUR_AREA);
            tracing::info!(
                camera = %self.camera_id,
                old = old,
                new = self.params.min_contour_area,
                "motion tuner: relaxing minimum blob size after sustained quiet"
            );
            return Some(ParamChange::MinContourArea(self.params.min_contour_area));
        }

        if self.params.var_threshold > DEFAULT_VAR_THRESHOLD {
            let old = self.params.var_threshold;
            self.params.var_threshold =
                (self.params.var_threshold - VAR_THRESHOLD_STEP).max(DEFAULT_VAR_THRESHOLD);
            tracing::info!(
                camera = %self.camera_id,
                old = old,
                new = self.params.var_threshold,
                "motion tuner: relaxing variance threshold after sustained quiet"
            );
            return Some(ParamChange::VarThreshold(self.params.var_threshold));
        }

        None
    }

    fn reset_window(&mut self) {
        self.noise_events = 0;
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

        // MOG2 with shadow detection marks shadows as 127, foreground as 255.
        // Threshold to keep only true foreground before cleanup.
        imgproc::threshold(
            &self.fg_mask.clone(),
            &mut self.fg_mask,
            200.0,
            255.0,
            imgproc::THRESH_BINARY,
        )?;

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

    /// Report a motion event (segment above threshold). The tuner counts these
    /// as potential noise unless offset by `report_positive_detection`.
    pub fn report_motion_event(&mut self) {
        self.tuner.record_motion_event();
    }

    /// Evaluate tuner — must be called every analysis cycle regardless of motion.
    pub fn maybe_tune(&mut self) -> CvResult<()> {
        for change in self.tuner.maybe_tune() {
            self.apply_param_change(change)?;
        }
        Ok(())
    }

    /// An object detection confirmed this motion was real — subtract one noise event.
    pub fn report_positive_detection(&mut self) {
        self.tuner.record_positive_detection();
    }

    pub fn tuner_stats(&self) -> TunerStats {
        TunerStats {
            var_threshold: self.tuner.params.var_threshold,
            learning_rate: self.tuner.params.learning_rate,
            morph_kernel_size: self.tuner.params.morph_kernel_size,
            min_contour_area: self.tuner.params.min_contour_area,
            noise_events: self.tuner.noise_events,
            quiet_windows: self.tuner.quiet_windows,
        }
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
        let mut contours = Vector::<Vector<opencv::core::Point>>::new();
        imgproc::find_contours(
            &self.fg_mask.clone(),
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .ok()?;

        let mut best_rect: Option<Rect> = None;
        let mut best_area = 0.0;
        for i in 0..contours.len() {
            let area = imgproc::contour_area(&contours.get(i).ok()?, false).ok()?;
            if area > best_area {
                best_area = area;
                best_rect = imgproc::bounding_rect(&contours.get(i).ok()?).ok();
            }
        }

        best_rect.filter(|r| r.width > 0 && r.height > 0)
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

        tuner.noise_events = 100;
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        assert!(tuner.maybe_tune().is_empty());
    }

    #[test]
    fn tuner_tightens_on_high_noise() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        tuner.noise_events = 30;

        let changes = tuner.maybe_tune();
        // First cycle: var_threshold always steps (weight 1.0)
        assert!(changes
            .iter()
            .any(|c| matches!(c, ParamChange::VarThreshold(_))));
        assert_eq!(
            tuner.params.var_threshold,
            DEFAULT_VAR_THRESHOLD + VAR_THRESHOLD_STEP
        );
    }

    #[test]
    fn tuner_weighted_stepping() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);

        let set_high_noise = |t: &mut MotionTuner| {
            t.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
            t.last_adjustment = None;
            t.noise_events = 100;
        };

        // Cycle 1: var_threshold steps (weight 1.0)
        set_high_noise(&mut tuner);
        let c1 = tuner.maybe_tune();
        assert!(c1.iter().any(|c| matches!(c, ParamChange::VarThreshold(_))));
        assert!(!c1
            .iter()
            .any(|c| matches!(c, ParamChange::MinContourArea(_))));

        // Cycle 2: var_threshold steps again, min_contour_area steps (weight 0.5, acc now 1.0)
        set_high_noise(&mut tuner);
        let c2 = tuner.maybe_tune();
        assert!(c2.iter().any(|c| matches!(c, ParamChange::VarThreshold(_))));
        assert!(c2
            .iter()
            .any(|c| matches!(c, ParamChange::MinContourArea(_))));

        // Cycle 3: var_threshold steps, morph_kernel steps (weight 0.33, acc ~0.99)
        set_high_noise(&mut tuner);
        let c3 = tuner.maybe_tune();
        assert!(c3.iter().any(|c| matches!(c, ParamChange::VarThreshold(_))));
        // morph_kernel should step on cycle 3 (acc 0.33 + 0.33 + 0.33 = 0.99, not yet 1.0)
        // Actually it steps on cycle 4 (acc reaches 1.32)

        // Cycle 4: var_threshold + learning_rate (weight 0.25, acc 1.0)
        set_high_noise(&mut tuner);
        let c4 = tuner.maybe_tune();
        assert!(c4.iter().any(|c| matches!(c, ParamChange::VarThreshold(_))));
        assert!(c4.iter().any(|c| matches!(c, ParamChange::LearningRate(_))));
    }

    #[test]
    fn tuner_does_nothing_when_quiet() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        tuner.noise_events = 1;

        assert!(tuner.maybe_tune().is_empty());
    }

    #[test]
    fn tuner_positive_detections_offset_noise() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        for _ in 0..30 {
            tuner.record_motion_event();
        }
        for _ in 0..25 {
            tuner.record_positive_detection();
        }
        assert!(tuner.noise_events == 5);
    }

    #[test]
    fn tuner_respects_cooldown() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.last_adjustment = Some(Instant::now());

        tuner.noise_events = 100;

        assert!(tuner.maybe_tune().is_empty());
    }

    #[test]
    fn tuner_all_maxed_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);

        tuner.params.var_threshold = VAR_THRESHOLD_MAX;
        tuner.params.min_contour_area = MIN_CONTOUR_AREA_MAX;
        tuner.params.morph_kernel_size = MORPH_KERNEL_MAX;
        tuner.params.learning_rate = LEARNING_RATE_MAX;
        tuner.noise_events = 100;

        assert!(tuner.maybe_tune().is_empty());
    }

    #[test]
    fn tuner_relaxes_after_sustained_quiet() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);

        tuner.params.var_threshold = VAR_THRESHOLD_MAX;
        tuner.params.learning_rate = LEARNING_RATE_MAX;

        for _ in 0..RELAX_QUIET_WINDOWS - 1 {
            tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
            tuner.last_adjustment = None;
            tuner.noise_events = 0;
            assert!(tuner.maybe_tune().is_empty());
        }

        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.last_adjustment = None;
        tuner.noise_events = 0;
        let changes = tuner.maybe_tune();
        assert!(changes
            .iter()
            .any(|c| matches!(c, ParamChange::LearningRate(_))));
        assert_eq!(
            tuner.params.learning_rate,
            LEARNING_RATE_MAX - LEARNING_RATE_STEP
        );
        assert_eq!(tuner.params.var_threshold, VAR_THRESHOLD_MAX);
    }

    #[test]
    fn tuner_does_not_relax_at_defaults() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.quiet_windows = RELAX_QUIET_WINDOWS;

        tuner.noise_events = 0;

        assert!(tuner.maybe_tune().is_empty());
    }

    #[test]
    fn tuner_any_noise_resets_quiet_counter() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.params.var_threshold = VAR_THRESHOLD_MAX;

        tuner.quiet_windows = RELAX_QUIET_WINDOWS - 1;

        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.last_adjustment = None;
        tuner.noise_events = 1;
        assert!(tuner.maybe_tune().is_empty());
        assert_eq!(tuner.quiet_windows, 0);

        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.noise_events = 0;
        assert!(tuner.maybe_tune().is_empty());
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
