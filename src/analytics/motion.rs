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

// --- Adaptive parameter defaults, bounds, and per-cycle increments ---

const DEFAULT_VAR_THRESHOLD: f64 = 16.0;
const VAR_THRESHOLD_MAX: f64 = 96.0;
const VAR_THRESHOLD_INCREMENT: f64 = 4.0;

const DEFAULT_LEARNING_RATE: f64 = 0.003;
const LEARNING_RATE_MAX: f64 = 0.010;
const LEARNING_RATE_INCREMENT: f64 = 0.00025;

const DEFAULT_MORPH_KERNEL: f64 = 5.0;
const MORPH_KERNEL_MAX: f64 = 15.0;
const MORPH_KERNEL_INCREMENT: f64 = 0.66;

// On a 320x240 analysis frame a person is ~4000px. Max 2000 leaves a
// safe margin while filtering out wind/leaf blobs that are a few hundred px.
const DEFAULT_MIN_CONTOUR_AREA: f64 = 200.0;
const MIN_CONTOUR_AREA_MAX: f64 = 2000.0;
const MIN_CONTOUR_AREA_INCREMENT: f64 = 50.0;

// Per-region contour area tuning grid. 4x3 on 320x240 → 80x80px per cell.
const REGION_COLS: usize = 4;
const REGION_ROWS: usize = 3;
const REGION_COUNT: usize = REGION_COLS * REGION_ROWS;

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
    pub morph_kernel: f64,
    pub noise_events: u32,
    pub quiet_windows: u32,
    pub region_min_contour_areas: Vec<f64>,
    pub region_noise_events: Vec<u32>,
    pub region_cols: usize,
    pub region_rows: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TunedParams {
    pub var_threshold: f64,
    pub learning_rate: f64,
    pub morph_kernel: f64,
    /// Legacy global min_contour_area — kept for backward compat in serialization.
    /// At runtime, `region_min_contour_areas` is used instead.
    #[serde(default = "default_min_contour_area")]
    pub min_contour_area: f64,
    #[serde(default = "default_region_min_contour_areas")]
    pub region_min_contour_areas: Vec<f64>,
}

fn default_min_contour_area() -> f64 {
    DEFAULT_MIN_CONTOUR_AREA
}

fn default_region_min_contour_areas() -> Vec<f64> {
    vec![DEFAULT_MIN_CONTOUR_AREA; REGION_COUNT]
}

impl TunedParams {
    fn morph_kernel_size(&self) -> i32 {
        let v = (self.morph_kernel.floor() as i32) | 1; // ensure odd
        v.max(3)
    }

    fn ensure_regions(&mut self) {
        if self.region_min_contour_areas.len() != REGION_COUNT {
            let fill = if self.min_contour_area != DEFAULT_MIN_CONTOUR_AREA {
                self.min_contour_area
            } else {
                DEFAULT_MIN_CONTOUR_AREA
            };
            self.region_min_contour_areas = vec![fill; REGION_COUNT];
        } else if self.min_contour_area != DEFAULT_MIN_CONTOUR_AREA
            && self
                .region_min_contour_areas
                .iter()
                .all(|&v| v == DEFAULT_MIN_CONTOUR_AREA)
        {
            // Loaded from format that had no region data — backfill from global.
            self.region_min_contour_areas = vec![self.min_contour_area; REGION_COUNT];
        }
    }

    fn is_at_defaults(&self) -> bool {
        let d = Self::default();
        self.var_threshold == d.var_threshold
            && self.learning_rate == d.learning_rate
            && self.morph_kernel == d.morph_kernel
            && self
                .region_min_contour_areas
                .iter()
                .all(|&v| v == DEFAULT_MIN_CONTOUR_AREA)
    }
}

impl Default for TunedParams {
    fn default() -> Self {
        Self {
            var_threshold: DEFAULT_VAR_THRESHOLD,
            learning_rate: DEFAULT_LEARNING_RATE,
            morph_kernel: DEFAULT_MORPH_KERNEL,
            min_contour_area: DEFAULT_MIN_CONTOUR_AREA,
            region_min_contour_areas: vec![DEFAULT_MIN_CONTOUR_AREA; REGION_COUNT],
        }
    }
}

impl TunedParams {
    fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        if let Ok(mut params) = serde_json::from_str::<Self>(&data) {
            params.ensure_regions();
            return Some(params);
        }
        // Migrate old format (morph_kernel_size: i32 → morph_kernel: f64)
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
            let global_mca = v["min_contour_area"]
                .as_f64()
                .unwrap_or(DEFAULT_MIN_CONTOUR_AREA);
            let mut params = Self {
                var_threshold: v["var_threshold"].as_f64().unwrap_or(DEFAULT_VAR_THRESHOLD),
                learning_rate: v["learning_rate"].as_f64().unwrap_or(DEFAULT_LEARNING_RATE),
                morph_kernel: v["morph_kernel_size"]
                    .as_i64()
                    .map(|v| v as f64)
                    .or_else(|| v["morph_kernel"].as_f64())
                    .unwrap_or(DEFAULT_MORPH_KERNEL),
                min_contour_area: global_mca,
                region_min_contour_areas: vec![global_mca; REGION_COUNT],
            };
            params.ensure_regions();
            tracing::info!(path = %path.display(), "migrated old motion tuner format");
            params.save(path);
            return Some(params);
        }
        // Corrupt file — remove it
        let _ = std::fs::remove_file(path);
        None
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
    region_noise_events: [u32; REGION_COUNT],
    region_quiet_windows: [u32; REGION_COUNT],
    started_at: Instant,
    eval_start: Instant,
    last_adjustment: Option<Instant>,
}

impl MotionTuner {
    fn new(camera_id: String, data_dir: &Path) -> Self {
        let params_path = data_dir.join(&camera_id).join("motion_tuner.json");
        let mut params = TunedParams::load(&params_path).unwrap_or_default();
        params.ensure_regions();

        if !params.is_at_defaults() {
            tracing::info!(
                camera = %camera_id,
                var_threshold = params.var_threshold,
                learning_rate = format!("{:.4}", params.learning_rate),
                morph_kernel = format!("{:.1}", params.morph_kernel),
                min_contour_area = params.min_contour_area,
                "loaded saved motion tuner params"
            );
        }

        let now = Instant::now();
        Self {
            camera_id,
            params,
            params_path,
            region_noise_events: [0; REGION_COUNT],
            region_quiet_windows: [0; REGION_COUNT],
            started_at: now,
            eval_start: now,
            last_adjustment: None,
        }
    }

    fn total_noise_events(&self) -> u32 {
        self.region_noise_events.iter().sum()
    }

    fn record_motion_event(&mut self, regions: &[usize]) {
        for &r in regions {
            if r < REGION_COUNT {
                self.region_noise_events[r] += 1;
            }
        }
    }

    fn record_positive_detection(&mut self, regions: &[usize]) {
        for &r in regions {
            if r < REGION_COUNT {
                self.region_noise_events[r] = self.region_noise_events[r].saturating_sub(1);
            }
        }
    }

    fn maybe_tune(&mut self) -> bool {
        let now = Instant::now();

        if now.duration_since(self.started_at).as_secs() < TUNER_STARTUP_GRACE_SECS {
            return false;
        }

        let window_secs = now.duration_since(self.eval_start).as_secs();
        if window_secs < TUNER_EVAL_SECS {
            return false;
        }

        if let Some(last) = self.last_adjustment {
            if now.duration_since(last).as_secs() < TUNER_COOLDOWN_SECS {
                return false;
            }
        }

        let window_hours = window_secs as f32 / 3600.0;
        let total_noise = self.total_noise_events();
        let aggregate_noise_per_hour = total_noise as f32 / window_hours;

        let mut changed = false;

        // Global params (var_threshold, learning_rate, morph_kernel) use aggregate noise.
        if aggregate_noise_per_hour > NOISE_EVENTS_PER_HOUR_HIGH {
            changed |= self.tighten_global(aggregate_noise_per_hour);
        }

        // Per-region min_contour_area tuning.
        for region in 0..REGION_COUNT {
            let region_noise = self.region_noise_events[region];
            let region_noise_per_hour = region_noise as f32 / window_hours;

            if region_noise_per_hour > NOISE_EVENTS_PER_HOUR_HIGH {
                self.region_quiet_windows[region] = 0;
                changed |= self.tighten_region(region, region_noise_per_hour);
            } else if region_noise == 0 {
                self.region_quiet_windows[region] += 1;
                if self.params.region_min_contour_areas[region] != DEFAULT_MIN_CONTOUR_AREA
                    && self.region_quiet_windows[region] >= RELAX_QUIET_WINDOWS
                {
                    self.region_quiet_windows[region] = 0;
                    changed |= self.relax_region(region);
                }
            } else {
                self.region_quiet_windows[region] = 0;
            }
        }

        // Relax global params after all regions have been quiet long enough.
        if total_noise == 0 && !self.params.is_at_defaults() {
            let all_quiet = self
                .region_quiet_windows
                .iter()
                .all(|&w| w >= RELAX_QUIET_WINDOWS);
            if all_quiet {
                changed |= self.relax_global();
            }
        }

        if changed {
            self.last_adjustment = Some(now);
            // Update legacy global field to max of regional values.
            self.params.min_contour_area = self
                .params
                .region_min_contour_areas
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
            self.params.save(&self.params_path);
        }

        self.reset_window();
        changed
    }

    fn tighten_global(&mut self, noise_per_hour: f32) -> bool {
        let p = &mut self.params;
        let before_vt = p.var_threshold;
        let before_mk = p.morph_kernel;
        let before_lr = p.learning_rate;

        p.var_threshold = (p.var_threshold + VAR_THRESHOLD_INCREMENT).min(VAR_THRESHOLD_MAX);
        p.morph_kernel = (p.morph_kernel + MORPH_KERNEL_INCREMENT).min(MORPH_KERNEL_MAX);
        p.learning_rate = (p.learning_rate + LEARNING_RATE_INCREMENT).min(LEARNING_RATE_MAX);

        let changed = p.var_threshold != before_vt
            || p.morph_kernel != before_mk
            || p.learning_rate != before_lr;

        if changed {
            tracing::info!(
                camera = %self.camera_id,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                var_threshold = format!("{:.1}", p.var_threshold),
                morph_kernel = p.morph_kernel_size(),
                learning_rate = format!("{:.4}", p.learning_rate),
                "motion tuner: tightening global params"
            );
        }
        changed
    }

    fn tighten_region(&mut self, region: usize, noise_per_hour: f32) -> bool {
        let v = &mut self.params.region_min_contour_areas[region];
        let before = *v;
        *v = (*v + MIN_CONTOUR_AREA_INCREMENT).min(MIN_CONTOUR_AREA_MAX);
        let changed = *v != before;
        if changed {
            tracing::info!(
                camera = %self.camera_id,
                region,
                noise_per_hour = format!("{:.1}", noise_per_hour),
                min_contour_area = format!("{:.0}", *v),
                "motion tuner: tightening region"
            );
        }
        changed
    }

    fn relax_global(&mut self) -> bool {
        const RELAX_DIVISOR: f64 = 6.0;
        let p = &mut self.params;
        let before_vt = p.var_threshold;
        let before_mk = p.morph_kernel;
        let before_lr = p.learning_rate;

        p.var_threshold =
            (p.var_threshold - VAR_THRESHOLD_INCREMENT / RELAX_DIVISOR).max(DEFAULT_VAR_THRESHOLD);
        p.morph_kernel =
            (p.morph_kernel - MORPH_KERNEL_INCREMENT / RELAX_DIVISOR).max(DEFAULT_MORPH_KERNEL);
        p.learning_rate =
            (p.learning_rate - LEARNING_RATE_INCREMENT / RELAX_DIVISOR).max(DEFAULT_LEARNING_RATE);

        let changed = p.var_threshold != before_vt
            || p.morph_kernel != before_mk
            || p.learning_rate != before_lr;

        if changed {
            tracing::info!(
                camera = %self.camera_id,
                var_threshold = format!("{:.1}", p.var_threshold),
                morph_kernel = p.morph_kernel_size(),
                learning_rate = format!("{:.4}", p.learning_rate),
                "motion tuner: relaxing global params"
            );
        }
        changed
    }

    fn relax_region(&mut self, region: usize) -> bool {
        const RELAX_DIVISOR: f64 = 6.0;
        let v = &mut self.params.region_min_contour_areas[region];
        let before = *v;
        *v = (*v - MIN_CONTOUR_AREA_INCREMENT / RELAX_DIVISOR).max(DEFAULT_MIN_CONTOUR_AREA);
        let changed = *v != before;
        if changed {
            tracing::info!(
                camera = %self.camera_id,
                region,
                min_contour_area = format!("{:.0}", *v),
                "motion tuner: relaxing region"
            );
        }
        changed
    }

    fn reset_window(&mut self) {
        self.region_noise_events = [0; REGION_COUNT];
        self.eval_start = Instant::now();
    }
}

pub struct MotionDetector {
    mog2: opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
    fg_mask: Mat,
    cleaned_mask: Mat,
    morph_kernel: Mat,
    learning_rate: f64,
    region_min_contour_areas: [f64; REGION_COUNT],
    frames_since_stable: u32,
    tuner: MotionTuner,
    raw_mog2_mask: Mat,
    no_shadow_mask: Mat,
    morph_mask: Mat,
}

impl MotionDetector {
    pub fn new(camera_id: &str, data_dir: &Path) -> CvResult<Self> {
        let tuner = MotionTuner::new(camera_id.to_string(), data_dir);
        let params = &tuner.params;

        let mog2 =
            video::create_background_subtractor_mog2(MOG2_HISTORY, params.var_threshold, true)?;
        let fg_mask = Mat::default();
        let cleaned_mask = Mat::default();
        let morph_kernel = build_morph_kernel(params.morph_kernel_size())?;

        let mut region_areas = [DEFAULT_MIN_CONTOUR_AREA; REGION_COUNT];
        for (i, &v) in params.region_min_contour_areas.iter().enumerate() {
            if i < REGION_COUNT {
                region_areas[i] = v;
            }
        }

        Ok(Self {
            mog2,
            fg_mask,
            cleaned_mask,
            morph_kernel,
            learning_rate: params.learning_rate,
            region_min_contour_areas: region_areas,
            frames_since_stable: 0,
            tuner,
            raw_mog2_mask: Mat::default(),
            no_shadow_mask: Mat::default(),
            morph_mask: Mat::default(),
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

        // Snapshot: raw MOG2 output (shadows=127, foreground=255).
        self.raw_mog2_mask = self.fg_mask.clone();

        // MOG2 with shadow detection marks shadows as 127, foreground as 255.
        // Threshold to keep only true foreground before cleanup.
        imgproc::threshold(
            &self.fg_mask.clone(),
            &mut self.fg_mask,
            200.0,
            255.0,
            imgproc::THRESH_BINARY,
        )?;

        // Snapshot: after shadow removal (only true foreground remains).
        self.no_shadow_mask = self.fg_mask.clone();

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

        // Snapshot: after morphological opening.
        self.morph_mask = self.cleaned_mask.clone();

        // Contour area filter: zero out blobs smaller than min_contour_area.
        let mut contours = Vector::<Vector<opencv::core::Point>>::new();
        imgproc::find_contours(
            &self.cleaned_mask.clone(),
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )?;

        let frame_w = self.fg_mask.cols();
        let frame_h = self.fg_mask.rows();
        let mut motion_mask = Mat::zeros(frame_h, frame_w, opencv::core::CV_8UC1)?.to_mat()?;
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            let rect = imgproc::bounding_rect(&contour)?;
            let region = contour_region(&rect, frame_w, frame_h);
            let threshold = self.region_min_contour_areas[region];
            if area >= threshold {
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

    /// Report a motion event for regions covered by the given bounding boxes
    /// (in analysis-frame pixel coordinates).
    pub fn report_motion_event(&mut self, bboxes: &[Rect], frame_w: i32, frame_h: i32) {
        let regions = unique_regions_from_rects(bboxes, frame_w, frame_h);
        self.tuner.record_motion_event(&regions);
    }

    /// Evaluate tuner — must be called every analysis cycle regardless of motion.
    pub fn maybe_tune(&mut self) -> CvResult<()> {
        if self.tuner.maybe_tune() {
            self.apply_tuned_params()?;
        }
        Ok(())
    }

    /// An object detection confirmed this motion was real — subtract noise for
    /// the regions covered by the given normalized bounding boxes.
    pub fn report_positive_detection(&mut self, normalized_rects: &[(f32, f32, f32, f32)]) {
        let regions = unique_regions_from_normalized(normalized_rects);
        self.tuner.record_positive_detection(&regions);
    }

    pub fn tuner_stats(&self) -> TunerStats {
        TunerStats {
            var_threshold: self.tuner.params.var_threshold,
            learning_rate: self.tuner.params.learning_rate,
            morph_kernel: self.tuner.params.morph_kernel,
            noise_events: self.tuner.total_noise_events(),
            quiet_windows: *self.tuner.region_quiet_windows.iter().min().unwrap_or(&0),
            region_min_contour_areas: self.tuner.params.region_min_contour_areas.clone(),
            region_noise_events: self.tuner.region_noise_events.to_vec(),
            region_cols: REGION_COLS,
            region_rows: REGION_ROWS,
        }
    }

    fn apply_tuned_params(&mut self) -> CvResult<()> {
        let p = &self.tuner.params;
        self.mog2.set_var_threshold(p.var_threshold)?;
        self.learning_rate = p.learning_rate;
        for (i, &v) in p.region_min_contour_areas.iter().enumerate() {
            if i < REGION_COUNT {
                self.region_min_contour_areas[i] = v;
            }
        }
        self.morph_kernel = build_morph_kernel(p.morph_kernel_size())?;
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
        Self::mat_to_jpeg(&self.fg_mask)
    }

    /// Raw MOG2 output including shadow pixels.
    pub fn raw_mog2_mask_jpeg(&self) -> Option<Vec<u8>> {
        Self::mat_to_jpeg(&self.raw_mog2_mask)
    }

    /// After shadow removal, before morphological cleanup.
    pub fn no_shadow_mask_jpeg(&self) -> Option<Vec<u8>> {
        Self::mat_to_jpeg(&self.no_shadow_mask)
    }

    /// After morphological opening, before contour area filtering.
    pub fn morph_mask_jpeg(&self) -> Option<Vec<u8>> {
        Self::mat_to_jpeg(&self.morph_mask)
    }

    fn mat_to_jpeg(mat: &Mat) -> Option<Vec<u8>> {
        if mat.empty() {
            return None;
        }
        let mut buf = Vector::<u8>::new();
        let params = Vector::<i32>::new();
        imgcodecs::imencode(".jpg", mat, &mut buf, &params).ok()?;
        Some(buf.to_vec())
    }

    /// Returns bounding rects for ALL contours in the foreground mask.
    /// The mask is already filtered to min_contour_area by process_frame().
    pub fn motion_bboxes(&self) -> Vec<Rect> {
        let mut contours = Vector::<Vector<opencv::core::Point>>::new();
        if imgproc::find_contours(
            &self.fg_mask.clone(),
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .is_err()
        {
            return Vec::new();
        }

        let mut rects = Vec::new();
        for i in 0..contours.len() {
            if let Ok(c) = contours.get(i) {
                if let Ok(r) = imgproc::bounding_rect(&c) {
                    if r.width > 0 && r.height > 0 {
                        rects.push(r);
                    }
                }
            }
        }
        rects
    }
}

fn build_morph_kernel(size: i32) -> CvResult<Mat> {
    imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(size, size),
        opencv::core::Point::new(-1, -1),
    )
}

fn contour_region(rect: &Rect, frame_w: i32, frame_h: i32) -> usize {
    let cx = rect.x + rect.width / 2;
    let cy = rect.y + rect.height / 2;
    let col = ((cx as usize) * REGION_COLS / frame_w.max(1) as usize).min(REGION_COLS - 1);
    let row = ((cy as usize) * REGION_ROWS / frame_h.max(1) as usize).min(REGION_ROWS - 1);
    row * REGION_COLS + col
}

fn unique_regions_from_rects(rects: &[Rect], frame_w: i32, frame_h: i32) -> Vec<usize> {
    let mut seen = [false; REGION_COUNT];
    let mut result = Vec::new();
    for r in rects {
        let region = contour_region(r, frame_w, frame_h);
        if !seen[region] {
            seen[region] = true;
            result.push(region);
        }
    }
    result
}

fn unique_regions_from_normalized(rects: &[(f32, f32, f32, f32)]) -> Vec<usize> {
    let mut seen = [false; REGION_COUNT];
    let mut result = Vec::new();
    for &(x, y, w, h) in rects {
        let cx = x + w / 2.0;
        let cy = y + h / 2.0;
        let col = ((cx * REGION_COLS as f32) as usize).min(REGION_COLS - 1);
        let row = ((cy * REGION_ROWS as f32) as usize).min(REGION_ROWS - 1);
        let region = row * REGION_COLS + col;
        if !seen[region] {
            seen[region] = true;
            result.push(region);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tuner_past_grace(dir: &Path) -> MotionTuner {
        let mut tuner = MotionTuner::new("test".into(), dir);
        tuner.started_at =
            Instant::now() - std::time::Duration::from_secs(TUNER_STARTUP_GRACE_SECS + 1);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner
    }

    fn add_noise(tuner: &mut MotionTuner, region: usize, count: u32) {
        for _ in 0..count {
            tuner.record_motion_event(&[region]);
        }
    }

    #[test]
    fn tuner_does_not_tune_during_grace_period() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());
        add_noise(&mut tuner, 0, 100);
        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        assert!(!tuner.maybe_tune());
    }

    #[test]
    fn tuner_tightens_global_and_noisy_region() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        // Concentrate all noise in region 0
        add_noise(&mut tuner, 0, 30);

        assert!(tuner.maybe_tune());
        assert!(tuner.params.var_threshold > DEFAULT_VAR_THRESHOLD);
        assert!(tuner.params.morph_kernel > DEFAULT_MORPH_KERNEL);
        assert!(tuner.params.learning_rate > DEFAULT_LEARNING_RATE);
        // Only region 0 should have tightened
        assert!(tuner.params.region_min_contour_areas[0] > DEFAULT_MIN_CONTOUR_AREA);
        // Other regions should stay at default
        assert_eq!(
            tuner.params.region_min_contour_areas[1],
            DEFAULT_MIN_CONTOUR_AREA
        );
        assert_eq!(
            tuner.params.region_min_contour_areas[5],
            DEFAULT_MIN_CONTOUR_AREA
        );
    }

    #[test]
    fn tuner_increments_are_proportional() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        add_noise(&mut tuner, 0, 100);

        tuner.maybe_tune();

        let vt_progress = (tuner.params.var_threshold - DEFAULT_VAR_THRESHOLD)
            / (VAR_THRESHOLD_MAX - DEFAULT_VAR_THRESHOLD);
        let lr_progress = (tuner.params.learning_rate - DEFAULT_LEARNING_RATE)
            / (LEARNING_RATE_MAX - DEFAULT_LEARNING_RATE);
        assert!(
            vt_progress > lr_progress,
            "var_threshold should advance faster than learning_rate"
        );
    }

    #[test]
    fn tuner_does_nothing_when_quiet() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        add_noise(&mut tuner, 3, 1);
        assert!(!tuner.maybe_tune());
    }

    #[test]
    fn tuner_positive_detections_offset_noise() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = MotionTuner::new("test".into(), dir.path());

        for _ in 0..30 {
            tuner.record_motion_event(&[2]);
        }
        for _ in 0..25 {
            tuner.record_positive_detection(&[2]);
        }
        assert_eq!(tuner.region_noise_events[2], 5);
        assert_eq!(tuner.total_noise_events(), 5);
    }

    #[test]
    fn tuner_respects_cooldown() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        tuner.last_adjustment = Some(Instant::now());
        add_noise(&mut tuner, 0, 100);
        assert!(!tuner.maybe_tune());
    }

    #[test]
    fn tuner_all_maxed() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        tuner.params.var_threshold = VAR_THRESHOLD_MAX;
        tuner.params.morph_kernel = MORPH_KERNEL_MAX;
        tuner.params.learning_rate = LEARNING_RATE_MAX;
        for v in &mut tuner.params.region_min_contour_areas {
            *v = MIN_CONTOUR_AREA_MAX;
        }
        add_noise(&mut tuner, 0, 100);
        assert!(!tuner.maybe_tune());
    }

    #[test]
    fn tuner_region_relaxes_independently() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());

        // Tighten region 3 only
        tuner.params.region_min_contour_areas[3] = MIN_CONTOUR_AREA_MAX;

        for _ in 0..RELAX_QUIET_WINDOWS - 1 {
            tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
            tuner.last_adjustment = None;
            // No noise anywhere → region quiet windows accumulate
            assert!(!tuner.maybe_tune());
        }

        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.last_adjustment = None;
        assert!(tuner.maybe_tune());
        assert!(tuner.params.region_min_contour_areas[3] < MIN_CONTOUR_AREA_MAX);
        // Other regions should still be at default
        assert_eq!(
            tuner.params.region_min_contour_areas[0],
            DEFAULT_MIN_CONTOUR_AREA
        );
    }

    #[test]
    fn tuner_relaxes_global_after_sustained_quiet() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        tuner.params.var_threshold = VAR_THRESHOLD_MAX;
        tuner.params.learning_rate = LEARNING_RATE_MAX;
        tuner.params.morph_kernel = MORPH_KERNEL_MAX;

        for _ in 0..RELAX_QUIET_WINDOWS - 1 {
            tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
            tuner.last_adjustment = None;
            assert!(!tuner.maybe_tune());
        }

        tuner.eval_start = Instant::now() - std::time::Duration::from_secs(TUNER_EVAL_SECS + 1);
        tuner.last_adjustment = None;
        assert!(tuner.maybe_tune());
        assert!(tuner.params.var_threshold < VAR_THRESHOLD_MAX);
        assert!(tuner.params.learning_rate < LEARNING_RATE_MAX);
    }

    #[test]
    fn tuner_does_not_relax_at_defaults() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        for rqw in &mut tuner.region_quiet_windows {
            *rqw = RELAX_QUIET_WINDOWS;
        }
        assert!(!tuner.maybe_tune());
    }

    #[test]
    fn tuner_noise_in_region_resets_its_quiet_counter() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut tuner = make_tuner_past_grace(dir.path());
        tuner.params.region_min_contour_areas[5] = 500.0;
        tuner.region_quiet_windows[5] = RELAX_QUIET_WINDOWS - 1;

        // Add low noise (below threshold) in region 5
        add_noise(&mut tuner, 5, 1);
        assert!(!tuner.maybe_tune());
        assert_eq!(tuner.region_quiet_windows[5], 0);
    }

    #[test]
    fn tuner_morph_kernel_rounds_to_odd() {
        let params = TunedParams {
            morph_kernel: 6.3,
            ..TunedParams::default()
        };
        assert_eq!(params.morph_kernel_size(), 7);
    }

    #[test]
    fn tuner_persistence_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut params = TunedParams {
            var_threshold: 24.0,
            learning_rate: 0.004,
            morph_kernel: 7.5,
            min_contour_area: 350.0,
            region_min_contour_areas: vec![DEFAULT_MIN_CONTOUR_AREA; REGION_COUNT],
        };
        params.region_min_contour_areas[2] = 450.0;
        params.region_min_contour_areas[7] = 800.0;

        let path = dir.path().join("test").join("motion_tuner.json");
        params.save(&path);

        let loaded = TunedParams::load(&path).unwrap();
        assert_eq!(params, loaded);
        assert_eq!(loaded.region_min_contour_areas[2], 450.0);
        assert_eq!(loaded.region_min_contour_areas[7], 800.0);
    }

    #[test]
    fn tuner_loads_legacy_format_without_regions() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test").join("motion_tuner.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            r#"{"var_threshold":28.0,"learning_rate":0.00375,"morph_kernel":6.98,"min_contour_area":350.0}"#,
        ).unwrap();

        let loaded = TunedParams::load(&path).unwrap();
        assert_eq!(loaded.var_threshold, 28.0);
        // All regions should be initialized from the global value
        assert!(loaded.region_min_contour_areas.iter().all(|&v| v == 350.0));
    }

    #[test]
    fn contour_region_boundaries() {
        // Top-left corner
        let r = Rect::new(0, 0, 10, 10);
        assert_eq!(contour_region(&r, 320, 240), 0);

        // Bottom-right corner
        let r = Rect::new(310, 230, 10, 10);
        assert_eq!(contour_region(&r, 320, 240), REGION_COUNT - 1);

        // Center of frame
        let r = Rect::new(155, 115, 10, 10);
        assert_eq!(contour_region(&r, 320, 240), REGION_COLS + 2); // row 1, col 2
    }

    #[test]
    fn unique_regions_deduplicates() {
        let rects = vec![
            (0.1, 0.1, 0.05, 0.05),   // region 0
            (0.15, 0.15, 0.05, 0.05), // also region 0
            (0.9, 0.9, 0.05, 0.05),   // region 11
        ];
        let regions = unique_regions_from_normalized(&rects);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0], 0);
        assert_eq!(regions[1], REGION_COUNT - 1);
    }
}
