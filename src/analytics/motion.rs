use super::ccl::{Component, ConnectedComponents};
use super::mog2::Mog2;
use super::morph::{self, StructuringElement};
use super::motion_settings::{MASK_CELLS, MASK_COLS, MASK_ROWS};

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

// Learning rate (alpha) per analyzed frame. At or below the automatic floor of
// 1/MOG2_HISTORY (≈0.0033) the detector follows OpenCV's auto schedule
// 1/min(2·t, history) — fast adaptation while the model is young, settling at
// the 5-minute memory above.
const LEARNING_RATE: f64 = 0.003;

// Morphological opening kernel size (odd). Fixed: noise rejection is now the
// job of the (user-tunable) var_threshold and min-object-size controls, not a
// self-adjusting kernel.
const MORPH_KERNEL_SIZE: i32 = 5;

/// Axis-aligned bounding box of a motion region, in analysis-frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Pure-Rust motion detector: Zivkovic MOG2 background subtraction →
/// static ignore-mask → morphological opening → connected-component area
/// filtering. Consumes plain grayscale frames (no OpenCV types anywhere in the
/// motion path). All behaviour is deterministic and driven by three
/// user-controlled settings (var_threshold, min_contour_area, ignore mask);
/// there is no hidden auto-tuning.
pub struct MotionDetector {
    mog2: Mog2,
    learning_rate: f64,
    kernel: StructuringElement,
    min_contour_area: f64,
    /// Ignore mask: one bool per 16x12 cell (row-major), `true` = excluded.
    /// Applied to the raw MOG2 foreground mask before scene-change evaluation,
    /// morphology, and connected-component labeling.
    mask: Vec<bool>,
    /// True once any mask cell is set — skips the masking pass entirely when the
    /// whole frame is active.
    mask_active: bool,
    frames_since_stable: u32,

    width: usize,
    height: usize,
    /// Raw MOG2 foreground mask (0/255) with the ignore mask applied, refreshed
    /// every frame.
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
    /// Create a detector with the given sensitivity (`var_threshold`) and
    /// minimum object size (`min_contour_area`). The ignore mask starts empty;
    /// set it with [`set_mask`](Self::set_mask).
    pub fn new(var_threshold: f64, min_contour_area: f64) -> Self {
        let mog2 = Mog2::new(MOG2_HISTORY, var_threshold);
        let kernel = StructuringElement::ellipse(MORPH_KERNEL_SIZE);

        Self {
            mog2,
            learning_rate: LEARNING_RATE,
            kernel,
            min_contour_area,
            mask: vec![false; MASK_CELLS],
            mask_active: false,
            frames_since_stable: 0,
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

    /// Sensitivity: higher = less sensitive. Takes effect on the next frame.
    pub fn set_var_threshold(&mut self, var_threshold: f64) {
        self.mog2.set_var_threshold(var_threshold);
    }

    /// Minimum connected-component area (foreground pixels) to report as motion.
    pub fn set_min_contour_area(&mut self, min_contour_area: f64) {
        self.min_contour_area = min_contour_area;
    }

    /// Replace the ignore mask (one bool per 16x12 cell, row-major). Cells set
    /// to `true` are excluded from motion detection. A wrong-length slice is
    /// ignored.
    pub fn set_mask(&mut self, mask: &[bool]) {
        if mask.len() != MASK_CELLS {
            return;
        }
        self.mask.copy_from_slice(mask);
        self.mask_active = mask.iter().any(|&m| m);
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

        // Static ignore mask: zero masked cells in the raw MOG2 foreground mask
        // BEFORE scene-change evaluation, morphological opening, and CCL, so
        // masked regions can neither raise the score nor trip scene-change
        // suppression. Deterministic — the same mask always ignores the same
        // pixels.
        self.apply_ignore_mask();

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

        // Component area filter: drop blobs smaller than the minimum area; keep
        // bounding boxes of what remains.
        self.ccl
            .label(&self.morph_mask, width, height, &mut self.components);
        self.retained.clear();
        self.retained.resize(self.components.len(), false);
        self.bboxes.clear();
        for (i, c) in self.components.iter().enumerate() {
            if f64::from(c.area) >= self.min_contour_area {
                self.retained[i] = true;
                self.bboxes.push(MotionBox {
                    x: c.min_x as i32,
                    y: c.min_y as i32,
                    width: (c.max_x - c.min_x + 1) as i32,
                    height: (c.max_y - c.min_y + 1) as i32,
                });
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

    /// Zero every pixel of `raw_mask` that falls inside a masked cell.
    fn apply_ignore_mask(&mut self) {
        if !self.mask_active {
            return;
        }
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return;
        }
        for y in 0..h {
            let row = (y * MASK_ROWS / h).min(MASK_ROWS - 1);
            let base = y * w;
            let cell_base = row * MASK_COLS;
            for x in 0..w {
                let col = (x * MASK_COLS / w).min(MASK_COLS - 1);
                if self.mask[cell_base + col] {
                    self.raw_mask[base + x] = 0;
                }
            }
        }
    }

    /// Learning rate for the upcoming frame. OpenCV's auto schedule
    /// (`1/min(2·t, history)`) acts as a floor so the model initializes
    /// quickly.
    fn effective_learning_rate(&self) -> f64 {
        let t = self.mog2.frame_count() + 1;
        let auto = 1.0 / (2 * t).min(u64::from(MOG2_HISTORY)) as f64;
        if self.learning_rate > auto {
            self.learning_rate
        } else {
            -1.0 // negative → MOG2 auto schedule
        }
    }

    fn mask_view<'a>(&self, data: &'a [u8]) -> Option<(&'a [u8], usize, usize)> {
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
        self.mask_view(&self.final_mask)
    }

    /// Raw MOG2 foreground mask (with the ignore mask already applied),
    /// refreshed every frame including warmup. The pure-Rust detector has no
    /// shadow class, so this is a plain 0/255 mask.
    pub fn raw_mask(&self) -> Option<(&[u8], usize, usize)> {
        self.mask_view(&self.raw_mask)
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
        self.mask_view(&self.morph_mask)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::motion_settings::{
        MotionSettings, DEFAULT_MIN_CONTOUR_AREA, DEFAULT_VAR_THRESHOLD,
    };

    // --- End-to-end detector tests on synthetic frames ---

    const W: usize = 320;
    const H: usize = 240;

    fn detector() -> MotionDetector {
        MotionDetector::new(DEFAULT_VAR_THRESHOLD, DEFAULT_MIN_CONTOUR_AREA)
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

    /// Build a 16x12 mask that covers the single cell containing (px, py).
    fn mask_covering(px: usize, py: usize) -> Vec<bool> {
        let mut mask = vec![false; MASK_CELLS];
        let col = (px * MASK_COLS / W).min(MASK_COLS - 1);
        let row = (py * MASK_ROWS / H).min(MASK_ROWS - 1);
        mask[row * MASK_COLS + col] = true;
        mask
    }

    #[test]
    fn detector_suppresses_scores_during_warmup() {
        let mut det = detector();
        let frame = static_frame();
        for _ in 0..=WARMUP_FRAMES {
            let score = det.process_frame(&frame, W, H);
            assert_eq!(score, 0.0);
            assert!(det.motion_bboxes().is_empty());
        }
    }

    #[test]
    fn detector_finds_moving_blob_with_correct_bbox() {
        let mut det = detector();
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
        let mut det = detector();
        warm_up(&mut det);

        // 12x12 = 144 px, < 200 after opening → filtered out.
        let score = det.process_frame(&frame_with_blob(50, 50, 12), W, H);
        assert_eq!(score, 0.0);
        assert!(det.motion_bboxes().is_empty());
    }

    #[test]
    fn detector_ignores_salt_noise() {
        let mut det = detector();
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
        let mut det = detector();
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
        let mut det = detector();
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

    #[test]
    fn detector_ignore_mask_suppresses_motion_in_masked_cell() {
        let mut det = detector();
        // Mask the cell containing the blob at (100,80).
        det.set_mask(&mask_covering(112, 92));
        warm_up(&mut det);

        let score = det.process_frame(&frame_with_blob(100, 80, 24), W, H);
        assert_eq!(score, 0.0, "motion inside a masked cell must be ignored");
        assert!(det.motion_bboxes().is_empty());
    }

    #[test]
    fn detector_ignore_mask_leaves_unmasked_motion() {
        let mut det = detector();
        // Mask a far-away cell; the blob at (100,80) is elsewhere.
        det.set_mask(&mask_covering(300, 220));
        warm_up(&mut det);

        let score = det.process_frame(&frame_with_blob(100, 80, 24), W, H);
        assert!(
            score > 0.0,
            "motion outside the mask must still be detected"
        );
        assert_eq!(det.motion_bboxes().len(), 1);
    }

    #[test]
    fn detector_clearing_mask_reenables_detection() {
        let mut det = detector();
        det.set_mask(&mask_covering(112, 92));
        // An all-false mask clears masking.
        det.set_mask(&[false; MASK_CELLS]);
        warm_up(&mut det);

        let score = det.process_frame(&frame_with_blob(100, 80, 24), W, H);
        assert!(score > 0.0, "cleared mask must restore detection");
    }

    /// The defect this detector was blinded by: `min_contour_area = nan` in
    /// config (TOML has a `nan` literal) survived `f64::clamp`, which returns
    /// NaN for NaN, and landed here — where `area >= min_contour_area` is false
    /// for every component ever labeled, so no blob is ever retained, no motion
    /// is ever scored, and no event is ever recorded, with nothing logged and a
    /// camera that looks like it is working. Config now refuses the value, and
    /// the settings that seed the detector replace it with the default, so the
    /// same input leaves detection intact.
    #[test]
    fn a_min_contour_area_that_is_not_a_number_cannot_blind_the_detector() {
        let settings = MotionSettings::from_defaults(DEFAULT_VAR_THRESHOLD, f64::NAN);
        let mut det = MotionDetector::new(settings.var_threshold, settings.min_contour_area);
        warm_up(&mut det);

        let score = det.process_frame(&frame_with_blob(100, 80, 24), W, H);
        assert!(score > 0.0, "a blob this size has to score");
        assert_eq!(det.motion_bboxes().len(), 1);

        // The unprotected value, for contrast: every comparison against it is
        // false, so the same frame is silently still.
        let mut blind = MotionDetector::new(DEFAULT_VAR_THRESHOLD, f64::NAN);
        warm_up(&mut blind);
        assert_eq!(
            blind.process_frame(&frame_with_blob(100, 80, 24), W, H),
            0.0
        );
        assert!(blind.motion_bboxes().is_empty());
    }

    #[test]
    fn detector_rejects_wrong_length_mask() {
        let mut det = detector();
        det.set_mask(&mask_covering(112, 92));
        // Wrong length is ignored, so the earlier mask stays in effect.
        det.set_mask(&[true, false, true]);
        warm_up(&mut det);
        let score = det.process_frame(&frame_with_blob(100, 80, 24), W, H);
        assert_eq!(score, 0.0, "wrong-length mask must not disturb the mask");
    }
}
