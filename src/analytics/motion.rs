use std::path::{Path, PathBuf};
use std::time::Instant;

use super::ccl::{Component, ConnectedComponents};
use super::mog2::Mog2;
use super::morph::{self, StructuringElement};

// The analyzer decodes keyframes only, and both production cameras emit a 1 s
// GOP (measured 2026-07-23 with ffprobe: keyframes at exactly 1.000 s
// intervals), so the effective sample rate of this detector is 1 frame per
// second. All time constants below are scaled for 1 fps.

// Frames to suppress after startup or a scene change while the background
// model warms up. ~10 seconds at 1 fps.
const WARMUP_FRAMES: u32 = 10;
const SCENE_CHANGE_RATIO: f32 = 0.8;

// MOG2 history: number of frames used to build the background model.
// At 1 fps, 300 frames ≈ 5 minutes of background memory. Persistent motion
// (tree sway) gets absorbed into the background model over this window.
const MOG2_HISTORY: u32 = 300;

// --- Adaptive parameter defaults, bounds, and per-cycle increments ---

const DEFAULT_VAR_THRESHOLD: f64 = 16.0;
const VAR_THRESHOLD_MAX: f64 = 96.0;
const VAR_THRESHOLD_INCREMENT: f64 = 4.0;

// Learning rate (alpha) per analyzed frame. While the tuned rate stays at or
// below the automatic floor of 1/MOG2_HISTORY (≈0.0033), the detector follows
// OpenCV's auto schedule 1/min(2·t, history) — fast adaptation while the
// model is young, settling at the 5-minute memory above. The tuner can raise
// the rate above the floor to absorb persistent noise faster: 0.010 at 1 fps
// shortens the background memory to ~100 s.
const DEFAULT_LEARNING_RATE: f64 = 0.003;
const LEARNING_RATE_MAX: f64 = 0.010;
const LEARNING_RATE_INCREMENT: f64 = 0.00025;

const DEFAULT_MORPH_KERNEL: f64 = 5.0;
const MORPH_KERNEL_MAX: f64 = 15.0;
const MORPH_KERNEL_INCREMENT: f64 = 0.66;

// On a 320x240 analysis frame a person is ~4000px. Max 2000 leaves a
// safe margin while filtering out wind/leaf blobs that are a few hundred px.
// Area is the component's foreground pixel count.
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

/// Axis-aligned bounding box of a motion region, in analysis-frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

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

/// Pure-Rust motion detector: Zivkovic MOG2 background subtraction →
/// morphological opening → connected-component area filtering. Consumes plain
/// grayscale frames (no OpenCV types anywhere in the motion path).
pub struct MotionDetector {
    mog2: Mog2,
    learning_rate: f64,
    kernel: StructuringElement,
    region_min_contour_areas: [f64; REGION_COUNT],
    frames_since_stable: u32,
    tuner: MotionTuner,

    width: usize,
    height: usize,
    /// Raw MOG2 foreground mask (0/255), refreshed every frame.
    raw_mask: Vec<u8>,
    /// After morphological opening. Stale during warmup.
    morph_mask: Vec<u8>,
    /// After component area filtering — the final motion mask. Stale during
    /// warmup.
    final_mask: Vec<u8>,
    erode_buf: Vec<u8>,
    ccl: ConnectedComponents,
    components: Vec<Component>,
    retained: Vec<bool>,
    bboxes: Vec<MotionBox>,
}

impl MotionDetector {
    pub fn new(camera_id: &str, data_dir: &Path) -> Self {
        let tuner = MotionTuner::new(camera_id.to_string(), data_dir);
        let params = &tuner.params;

        let mog2 = Mog2::new(MOG2_HISTORY, params.var_threshold);
        let kernel = StructuringElement::ellipse(params.morph_kernel_size());
        let learning_rate = params.learning_rate;

        let mut region_areas = [DEFAULT_MIN_CONTOUR_AREA; REGION_COUNT];
        for (i, &v) in params.region_min_contour_areas.iter().enumerate() {
            if i < REGION_COUNT {
                region_areas[i] = v;
            }
        }

        Self {
            mog2,
            learning_rate,
            kernel,
            region_min_contour_areas: region_areas,
            frames_since_stable: 0,
            tuner,
            width: 0,
            height: 0,
            raw_mask: Vec::new(),
            morph_mask: Vec::new(),
            final_mask: Vec::new(),
            erode_buf: Vec::new(),
            ccl: ConnectedComponents::new(),
            components: Vec::new(),
            retained: Vec::new(),
            bboxes: Vec::new(),
        }
    }

    /// Process one grayscale analysis frame (row-major, `width * height`
    /// bytes) and return the motion score in `0.0..=1.0`
    /// (foreground ratio × 10, capped).
    pub fn process_frame(&mut self, gray: &[u8], width: usize, height: usize) -> f32 {
        let total_pixels = width * height;
        if total_pixels == 0 || gray.len() < total_pixels {
            return 0.0;
        }
        self.width = width;
        self.height = height;

        let lr = self.effective_learning_rate();
        self.mog2.apply(gray, width, height, lr, &mut self.raw_mask);

        // Raw ratio for scene-change detection (before cleanup).
        let raw_fg = self.raw_mask.iter().filter(|&&v| v != 0).count();
        let raw_ratio = raw_fg as f32 / total_pixels as f32;

        if raw_ratio >= SCENE_CHANGE_RATIO {
            self.frames_since_stable = 0;
            self.bboxes.clear();
            return 0.0;
        }

        self.frames_since_stable += 1;

        if self.frames_since_stable < WARMUP_FRAMES {
            self.bboxes.clear();
            return 0.0;
        }

        // Morphological opening: erode then dilate to remove isolated noise
        // pixels.
        morph::open(
            &self.raw_mask,
            width,
            height,
            &self.kernel,
            &mut self.erode_buf,
            &mut self.morph_mask,
        );

        // Component area filter: drop blobs smaller than the (per-region)
        // minimum area; keep bounding boxes of what remains.
        self.ccl
            .label(&self.morph_mask, width, height, &mut self.components);
        self.retained.clear();
        self.retained.resize(self.components.len(), false);
        self.bboxes.clear();
        for (i, c) in self.components.iter().enumerate() {
            let bbox = MotionBox {
                x: c.min_x as i32,
                y: c.min_y as i32,
                width: (c.max_x - c.min_x + 1) as i32,
                height: (c.max_y - c.min_y + 1) as i32,
            };
            let region = contour_region(&bbox, width as i32, height as i32);
            if f64::from(c.area) >= self.region_min_contour_areas[region] {
                self.retained[i] = true;
                self.bboxes.push(bbox);
            }
        }

        self.final_mask.clear();
        self.final_mask.resize(total_pixels, 0);
        let mut fg_pixels = 0u32;
        for (out, &label) in self.final_mask.iter_mut().zip(self.ccl.labels()) {
            if label != 0 && self.retained[(label - 1) as usize] {
                *out = 255;
                fg_pixels += 1;
            }
        }

        let foreground_ratio = fg_pixels as f32 / total_pixels as f32;
        (foreground_ratio * 10.0).min(1.0)
    }

    /// Learning rate for the upcoming frame. OpenCV's auto schedule
    /// (`1/min(2·t, history)`) acts as a floor so the model initializes
    /// quickly; a tuned rate above the floor takes over once the model is old
    /// enough.
    fn effective_learning_rate(&self) -> f64 {
        let t = self.mog2.frame_count() + 1;
        let auto = 1.0 / (2 * t).min(u64::from(MOG2_HISTORY)) as f64;
        if self.learning_rate > auto {
            self.learning_rate
        } else {
            -1.0 // negative → MOG2 auto schedule
        }
    }

    /// Report a motion event for regions covered by the given bounding boxes
    /// (in analysis-frame pixel coordinates).
    pub fn report_motion_event(&mut self, bboxes: &[MotionBox], frame_w: i32, frame_h: i32) {
        let regions = unique_regions_from_rects(bboxes, frame_w, frame_h);
        self.tuner.record_motion_event(&regions);
    }

    /// Evaluate tuner — must be called every analysis cycle regardless of motion.
    pub fn maybe_tune(&mut self) {
        if self.tuner.maybe_tune() {
            self.apply_tuned_params();
        }
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

    fn apply_tuned_params(&mut self) {
        let p = &self.tuner.params;
        self.mog2.set_var_threshold(p.var_threshold);
        self.learning_rate = p.learning_rate;
        for (i, &v) in p.region_min_contour_areas.iter().enumerate() {
            if i < REGION_COUNT {
                self.region_min_contour_areas[i] = v;
            }
        }
        self.kernel = StructuringElement::ellipse(p.morph_kernel_size());
    }

    fn mask<'a>(&self, data: &'a [u8]) -> Option<(&'a [u8], usize, usize)> {
        let npixels = self.width * self.height;
        if npixels == 0 || data.len() != npixels {
            return None;
        }
        Some((data, self.width, self.height))
    }

    /// Final motion mask (after opening and component filtering) — what the
    /// score and bounding boxes are computed from. `None` until the detector
    /// has processed a frame past warmup.
    pub fn fg_mask(&self) -> Option<(&[u8], usize, usize)> {
        self.mask(&self.final_mask)
    }

    /// Raw MOG2 foreground mask, refreshed every frame (including warmup).
    /// The pure-Rust detector has no shadow class, so this is a plain 0/255
    /// mask.
    pub fn raw_mask(&self) -> Option<(&[u8], usize, usize)> {
        self.mask(&self.raw_mask)
    }

    /// Historical "after shadow removal" stage. Shadow detection is gone
    /// (chromaticity-based, meaningless on grayscale), so this is now an
    /// alias for the raw MOG2 mask — kept so the debug API/UI stage names
    /// stay stable.
    pub fn no_shadow_mask(&self) -> Option<(&[u8], usize, usize)> {
        self.raw_mask()
    }

    /// Mask after morphological opening, before component area filtering.
    pub fn morph_mask(&self) -> Option<(&[u8], usize, usize)> {
        self.mask(&self.morph_mask)
    }

    /// Render the learned background model into `out`; returns dimensions.
    pub fn background_into(&self, out: &mut Vec<u8>) -> Option<(usize, usize)> {
        self.mog2.background_into(out)
    }

    /// Bounding boxes of the motion components retained by the area filter in
    /// the last processed frame (empty during warmup / scene change).
    pub fn motion_bboxes(&self) -> &[MotionBox] {
        &self.bboxes
    }

    /// Whether the last frame was fully processed. False during model warmup
    /// and right after a scene change, while scores are suppressed to 0 and
    /// the final/morph masks are not refreshed.
    #[allow(dead_code)] // used through the library crate root (examples/)
    pub fn is_warmed_up(&self) -> bool {
        self.frames_since_stable >= WARMUP_FRAMES
    }
}

fn contour_region(rect: &MotionBox, frame_w: i32, frame_h: i32) -> usize {
    let cx = rect.x + rect.width / 2;
    let cy = rect.y + rect.height / 2;
    let col = ((cx as usize) * REGION_COLS / frame_w.max(1) as usize).min(REGION_COLS - 1);
    let row = ((cy as usize) * REGION_ROWS / frame_h.max(1) as usize).min(REGION_ROWS - 1);
    row * REGION_COLS + col
}

fn unique_regions_from_rects(rects: &[MotionBox], frame_w: i32, frame_h: i32) -> Vec<usize> {
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
        let r = MotionBox {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        };
        assert_eq!(contour_region(&r, 320, 240), 0);

        // Bottom-right corner
        let r = MotionBox {
            x: 310,
            y: 230,
            width: 10,
            height: 10,
        };
        assert_eq!(contour_region(&r, 320, 240), REGION_COUNT - 1);

        // Center of frame
        let r = MotionBox {
            x: 155,
            y: 115,
            width: 10,
            height: 10,
        };
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

    // --- End-to-end detector tests on synthetic frames ---

    const W: usize = 320;
    const H: usize = 240;

    fn detector() -> (tempfile::TempDir, MotionDetector) {
        let dir = tempfile::TempDir::new().unwrap();
        let det = MotionDetector::new("test", dir.path());
        (dir, det)
    }

    fn static_frame() -> Vec<u8> {
        vec![60u8; W * H]
    }

    fn frame_with_blob(x0: usize, y0: usize, size: usize) -> Vec<u8> {
        let mut f = static_frame();
        for y in y0..(y0 + size).min(H) {
            for x in x0..(x0 + size).min(W) {
                f[y * W + x] = 200;
            }
        }
        f
    }

    /// Feed enough static frames to get past scene-change reset + warmup.
    fn warm_up(det: &mut MotionDetector) {
        let frame = static_frame();
        // Frame 1 is all-foreground (fresh model) → scene-change reset, then
        // WARMUP_FRAMES suppressed frames.
        for _ in 0..(WARMUP_FRAMES as usize + 2) {
            det.process_frame(&frame, W, H);
        }
    }

    #[test]
    fn detector_suppresses_scores_during_warmup() {
        let (_dir, mut det) = detector();
        let frame = static_frame();
        for _ in 0..=WARMUP_FRAMES {
            let score = det.process_frame(&frame, W, H);
            assert_eq!(score, 0.0);
            assert!(det.motion_bboxes().is_empty());
        }
    }

    #[test]
    fn detector_finds_moving_blob_with_correct_bbox() {
        let (_dir, mut det) = detector();
        warm_up(&mut det);

        let score = det.process_frame(&frame_with_blob(100, 80, 24), W, H);
        assert!(score > 0.0, "24x24 blob must score");
        let bboxes = det.motion_bboxes();
        assert_eq!(bboxes.len(), 1, "exactly one component");
        assert_eq!(
            bboxes[0],
            MotionBox {
                x: 100,
                y: 80,
                width: 24,
                height: 24
            }
        );
    }

    #[test]
    fn detector_area_filter_drops_small_blobs() {
        let (_dir, mut det) = detector();
        warm_up(&mut det);

        // 12x12 = 144 px, < 200 after opening → filtered out.
        let score = det.process_frame(&frame_with_blob(50, 50, 12), W, H);
        assert_eq!(score, 0.0);
        assert!(det.motion_bboxes().is_empty());
    }

    #[test]
    fn detector_ignores_salt_noise() {
        let (_dir, mut det) = detector();
        warm_up(&mut det);

        let mut frame = static_frame();
        for i in 0..60 {
            frame[(i * 1237 + 101) % (W * H)] = 220;
        }
        let score = det.process_frame(&frame, W, H);
        assert_eq!(score, 0.0, "isolated pixels removed by opening");
    }

    #[test]
    fn detector_scene_change_restarts_warmup() {
        let (_dir, mut det) = detector();
        warm_up(&mut det);

        // Full-frame change → scene change, score 0.
        let inverted = vec![210u8; W * H];
        assert_eq!(det.process_frame(&inverted, W, H), 0.0);

        // Back to the old scene: everything below warmup is suppressed, even
        // an obvious blob.
        for _ in 0..(WARMUP_FRAMES as usize - 1) {
            let score = det.process_frame(&frame_with_blob(10, 10, 30), W, H);
            assert_eq!(score, 0.0, "warmup after scene change");
        }
    }

    #[test]
    fn detector_masks_available_after_processing() {
        let (_dir, mut det) = detector();
        assert!(det.fg_mask().is_none());
        assert!(det.raw_mask().is_none());
        warm_up(&mut det);
        det.process_frame(&frame_with_blob(100, 80, 24), W, H);

        let (fg, w, h) = det.fg_mask().unwrap();
        assert_eq!((w, h), (W, H));
        // Opening rounds the blob's corners, so slightly under 24x24 pixels.
        let area = fg.iter().filter(|&&v| v != 0).count();
        assert!((540..=576).contains(&area), "final mask area {area}");
        assert!(det.raw_mask().is_some());
        assert!(det.morph_mask().is_some());
        assert!(det.no_shadow_mask().is_some());

        let mut bg = Vec::new();
        let (w, h) = det.background_into(&mut bg).unwrap();
        assert_eq!((w, h), (W, H));
        // The background model should still show the static scene value.
        assert!((bg[0] as i32 - 60).abs() <= 1);
    }
}
