//! Pure-Rust port of the Zivkovic adaptive Gaussian-mixture background
//! subtractor ("MOG2") for single-channel (grayscale) input, matching the
//! semantics of OpenCV's `BackgroundSubtractorMOG2` with shadow detection
//! disabled.
//!
//! References:
//! - Z. Zivkovic, "Improved adaptive Gaussian mixture model for background
//!   subtraction", ICPR 2004.
//! - Z. Zivkovic, F. van der Heijden, "Efficient adaptive density estimation
//!   per image pixel for the task of background subtraction", Pattern
//!   Recognition Letters 27(7), 2006.
//!
//! The per-pixel update loop is a faithful port of OpenCV's implementation
//! (`modules/video/src/bgfg_gaussmix2.cpp`), including its mode ordering,
//! complexity-reduction pruning (CT), and auto learning-rate schedule, so the
//! output can be validated against OpenCV frame-by-frame.

/// Maximum number of Gaussian components per pixel (OpenCV `nmixtures`).
const NMIXTURES: usize = 5;

/// TB (OpenCV `backgroundRatio`): a pixel matches the background when it fits
/// a component whose cumulative preceding weight (sorted by weight,
/// descending) is below this threshold.
const BACKGROUND_RATIO: f32 = 0.9;

/// Tg (OpenCV `varThresholdGen`): squared-distance threshold, in units of the
/// component variance, for updating an existing component instead of
/// generating a new one.
const VAR_THRESHOLD_GEN: f32 = 9.0;

/// Initial variance for newly created components (OpenCV `fVarInit`).
const VAR_INIT: f32 = 15.0;
const VAR_MIN: f32 = 4.0;
const VAR_MAX: f32 = 5.0 * VAR_INIT;

/// CT (OpenCV `fCT`): complexity-reduction prior. Components whose evidence
/// stays below `CT * alpha` worth of weight are pruned, keeping the mixture
/// small on stable pixels.
const COMPLEXITY_REDUCTION: f32 = 0.05;

/// Mask value for foreground pixels. Background is 0. (No shadow class: shadow
/// detection is chromaticity-based and meaningless on grayscale input, where
/// its distance test degenerates to always-true for any darker pixel.)
const FOREGROUND: u8 = 255;

#[derive(Clone, Copy, Default)]
struct Mode {
    weight: f32,
    var: f32,
    mean: f32,
}

/// Adaptive per-pixel Gaussian mixture background model.
pub struct Mog2 {
    history: u32,
    var_threshold: f32, // Tb
    width: usize,
    height: usize,
    frames: u64,
    /// Number of active components per pixel.
    nmodes: Vec<u8>,
    /// `NMIXTURES` components per pixel, sorted by weight (descending).
    modes: Vec<Mode>,
}

impl Mog2 {
    pub fn new(history: u32, var_threshold: f64) -> Self {
        Self {
            history: history.max(1),
            var_threshold: var_threshold as f32,
            width: 0,
            height: 0,
            frames: 0,
            nmodes: Vec::new(),
            modes: Vec::new(),
        }
    }

    /// Tb: squared-distance threshold (in variances) for the
    /// background/foreground decision.
    pub fn set_var_threshold(&mut self, v: f64) {
        self.var_threshold = v as f32;
    }

    /// Number of frames absorbed into the model.
    pub fn frame_count(&self) -> u64 {
        self.frames
    }

    /// Update the model with one grayscale frame and write the foreground
    /// mask (0 = background, 255 = foreground) into `fg_mask`.
    ///
    /// `learning_rate` follows OpenCV semantics: a negative value selects the
    /// automatic schedule `alpha = 1 / min(2 * frame_count, history)`, which
    /// adapts fast while the model is young and settles at `1 / history`.
    ///
    /// A dimension change reinitializes the model.
    pub fn apply(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        learning_rate: f64,
        fg_mask: &mut Vec<u8>,
    ) {
        let npixels = width * height;
        assert!(
            frame.len() >= npixels,
            "frame buffer smaller than dimensions"
        );

        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.frames = 0;
            self.nmodes.clear();
            self.nmodes.resize(npixels, 0);
            self.modes.clear();
            self.modes.resize(npixels * NMIXTURES, Mode::default());
        }

        self.frames += 1;
        let auto = 1.0 / (2 * self.frames).min(u64::from(self.history)) as f64;
        let alpha = if learning_rate >= 0.0 && self.frames > 1 {
            learning_rate
        } else {
            auto
        } as f32;

        fg_mask.clear();
        fg_mask.resize(npixels, 0);

        let tb = self.var_threshold;
        let alpha1 = 1.0 - alpha;
        let prune = -alpha * COMPLEXITY_REDUCTION;

        let pixels = frame[..npixels].iter().zip(fg_mask.iter_mut());
        let model = self
            .nmodes
            .iter_mut()
            .zip(self.modes.chunks_exact_mut(NMIXTURES));
        for ((&pix, out), (nmodes, g)) in pixels.zip(model) {
            let data = pix as f32;
            let mut nm = *nmodes as usize;
            let mut fits = false;
            let mut background = false;
            let mut total_weight = 0.0f32;

            let mut mode = 0;
            while mode < nm {
                // Weight decay plus the constant complexity-reduction prune term.
                let mut weight = alpha1 * g[mode].weight + prune;
                let mut swap_count = 0;

                if !fits {
                    let var = g[mode].var;
                    let diff = g[mode].mean - data;
                    let dist2 = diff * diff;

                    // Background test (Tb) against components still inside the
                    // cumulative background portion of the mixture.
                    if total_weight < BACKGROUND_RATIO && dist2 < tb * var {
                        background = true;
                    }

                    // Component-match test (Tg): update this component.
                    if dist2 < VAR_THRESHOLD_GEN * var {
                        fits = true;
                        weight += alpha;
                        let k = alpha / weight;
                        g[mode].mean -= k * diff;
                        g[mode].var = (var + k * (dist2 - var)).clamp(VAR_MIN, VAR_MAX);

                        // Bubble the strengthened component towards the front
                        // to keep the ordering by weight.
                        for j in (1..=mode).rev() {
                            if weight < g[j - 1].weight {
                                break;
                            }
                            swap_count += 1;
                            g.swap(j, j - 1);
                        }
                    }
                }

                // Prune components with too little evidence.
                if weight < -prune {
                    weight = 0.0;
                    nm -= 1;
                }

                g[mode - swap_count].weight = weight;
                total_weight += weight;
                mode += 1;
            }

            // Renormalize weights to sum to 1.
            if total_weight > 0.0 {
                let inv = 1.0 / total_weight;
                for m in &mut g[..nm] {
                    m.weight *= inv;
                }
            }

            // No component matched: create one (replacing the weakest when the
            // mixture is full).
            if !fits && alpha > 0.0 {
                let mode = if nm == NMIXTURES {
                    NMIXTURES - 1
                } else {
                    nm += 1;
                    nm - 1
                };

                if nm == 1 {
                    g[mode].weight = 1.0;
                } else {
                    g[mode].weight = alpha;
                    for m in &mut g[..nm - 1] {
                        m.weight *= alpha1;
                    }
                }

                g[mode].mean = data;
                g[mode].var = VAR_INIT;

                for j in (1..nm).rev() {
                    if alpha < g[j - 1].weight {
                        break;
                    }
                    g.swap(j, j - 1);
                }
            }

            *nmodes = nm as u8;
            *out = if background { 0 } else { FOREGROUND };
        }
    }

    /// Render the learned background (weighted mean of the components that
    /// make up the background portion of each pixel's mixture) into `out`.
    /// Returns the image dimensions, or `None` before the first frame.
    pub fn background_into(&self, out: &mut Vec<u8>) -> Option<(usize, usize)> {
        if self.frames == 0 {
            return None;
        }
        out.clear();
        out.reserve(self.width * self.height);
        for (&nm, g) in self.nmodes.iter().zip(self.modes.chunks_exact(NMIXTURES)) {
            let mut total_weight = 0.0f32;
            let mut mean = 0.0f32;
            for m in &g[..nm as usize] {
                mean += m.weight * m.mean;
                total_weight += m.weight;
                if total_weight > BACKGROUND_RATIO {
                    break;
                }
            }
            let v = if total_weight > 0.0 {
                (mean / total_weight).round().clamp(0.0, 255.0)
            } else {
                0.0
            };
            out.push(v as u8);
        }
        Some((self.width, self.height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 16;
    const H: usize = 12;

    fn apply(mog2: &mut Mog2, frame: &[u8]) -> Vec<u8> {
        let mut mask = Vec::new();
        mog2.apply(frame, W, H, -1.0, &mut mask);
        mask
    }

    #[test]
    fn first_frame_is_foreground_then_static_scene_converges() {
        let mut mog2 = Mog2::new(300, 16.0);
        let frame = vec![80u8; W * H];

        let mask = apply(&mut mog2, &frame);
        assert!(mask.iter().all(|&v| v == FOREGROUND), "no model yet");

        let mask = apply(&mut mog2, &frame);
        assert!(mask.iter().all(|&v| v == 0), "static scene is background");

        for _ in 0..50 {
            let mask = apply(&mut mog2, &frame);
            assert!(mask.iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn small_deviation_is_background_large_is_foreground() {
        let mut mog2 = Mog2::new(300, 16.0);
        let frame = vec![100u8; W * H];
        for _ in 0..100 {
            apply(&mut mog2, &frame);
        }
        // Converged variance is VAR_MIN = 4; Tb = 16 → |diff| < 8 is background.
        let mask = apply(&mut mog2, &[105u8; W * H]);
        assert!(mask.iter().all(|&v| v == 0), "within Tb·var is background");

        let mask = apply(&mut mog2, &[130u8; W * H]);
        assert!(
            mask.iter().all(|&v| v == FOREGROUND),
            "outside Tb·var is foreground"
        );
    }

    #[test]
    fn step_change_adapts_at_history_rate() {
        let mut mog2 = Mog2::new(300, 16.0);
        let old = vec![60u8; W * H];
        // Long static period: auto learning rate settles at 1/history.
        for _ in 0..400 {
            apply(&mut mog2, &old);
        }

        // Permanent step change. The new component's weight grows as
        // 1 - (1 - 1/300)^t and the pixel flips to background once it exceeds
        // 1 - BACKGROUND_RATIO = 0.1, i.e. after ≈ 300·ln(1/0.9) ≈ 32 frames.
        let new = vec![180u8; W * H];
        let mut flipped_at = None;
        for t in 1..=80 {
            let mask = apply(&mut mog2, &new);
            if mask.iter().all(|&v| v == 0) {
                flipped_at = Some(t);
                break;
            }
        }
        let flipped_at = flipped_at.expect("step change must be absorbed");
        assert!(
            (20..=45).contains(&flipped_at),
            "absorbed after {flipped_at} frames, expected ≈32"
        );
    }

    #[test]
    fn background_image_tracks_static_input() {
        let mut mog2 = Mog2::new(300, 16.0);
        let mut frame = vec![0u8; W * H];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = (i % 200) as u8;
        }
        for _ in 0..30 {
            apply(&mut mog2, &frame);
        }
        let mut bg = Vec::new();
        let (w, h) = mog2.background_into(&mut bg).unwrap();
        assert_eq!((w, h), (W, H));
        for (&b, &f) in bg.iter().zip(&frame) {
            assert!((b as i32 - f as i32).abs() <= 1, "bg {b} vs input {f}");
        }
    }

    #[test]
    fn transient_flicker_does_not_disturb_background() {
        let mut mog2 = Mog2::new(300, 16.0);
        let frame = vec![50u8; W * H];
        for _ in 0..100 {
            apply(&mut mog2, &frame);
        }
        // One-frame flash, then back to normal: the model must still call the
        // original scene background.
        apply(&mut mog2, &[250u8; W * H]);
        let mask = apply(&mut mog2, &frame);
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn dimension_change_reinitializes() {
        let mut mog2 = Mog2::new(300, 16.0);
        let frame = vec![50u8; W * H];
        for _ in 0..10 {
            apply(&mut mog2, &frame);
        }
        let mut mask = Vec::new();
        mog2.apply(&[50u8; 8 * 8], 8, 8, -1.0, &mut mask);
        assert_eq!(mask.len(), 64);
        assert!(mask.iter().all(|&v| v == FOREGROUND), "fresh model");
        assert_eq!(mog2.frame_count(), 1);
    }
}
