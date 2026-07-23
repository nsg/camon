//! Deterministic, user-controlled motion-detection settings, one set per
//! camera. This replaces the removed self-tuning `MotionTuner` and the learned
//! `DetectionGrid`: the values here only ever change when a human moves a
//! slider or paints the ignore mask in the web UI — there is no hidden feedback
//! loop that can silently reduce sensitivity.
//!
//! Settings are persisted to `{data_dir}/{camera}/motion_settings.json` and
//! shared with the analyzer through an `Arc`-backed store, so edits made via
//! the API take effect on the next analysis tick without a restart.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::locks::LockExt;

/// Ignore-mask grid geometry. 16x12 over a 320x240 analysis frame gives 20x20px
/// cells — the same resolution the old detection-grid overlay rendered at.
pub const MASK_COLS: usize = 16;
pub const MASK_ROWS: usize = 12;
pub const MASK_CELLS: usize = MASK_COLS * MASK_ROWS;

/// Sensitivity (MOG2 `var_threshold`) bounds. Higher = less sensitive.
pub const VAR_THRESHOLD_MIN: f64 = 4.0;
pub const VAR_THRESHOLD_MAX: f64 = 96.0;
pub const DEFAULT_VAR_THRESHOLD: f64 = 16.0;

/// Minimum-object-size (`min_contour_area`, foreground pixel count) bounds.
pub const MIN_CONTOUR_AREA_MIN: f64 = 50.0;
pub const MIN_CONTOUR_AREA_MAX: f64 = 2000.0;
pub const DEFAULT_MIN_CONTOUR_AREA: f64 = 200.0;

fn default_var_threshold() -> f64 {
    DEFAULT_VAR_THRESHOLD
}

fn default_min_contour_area() -> f64 {
    DEFAULT_MIN_CONTOUR_AREA
}

fn default_mask() -> Vec<bool> {
    vec![false; MASK_CELLS]
}

/// The three deterministic controls for one camera's motion detection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionSettings {
    /// MOG2 var_threshold. Higher = less sensitive.
    #[serde(default = "default_var_threshold")]
    pub var_threshold: f64,
    /// Minimum connected-component area (foreground pixels).
    #[serde(default = "default_min_contour_area")]
    pub min_contour_area: f64,
    /// One bool per 16x12 cell, row-major. `true` = ignored: the cell is
    /// excluded from motion detection deterministically.
    #[serde(default = "default_mask")]
    pub mask: Vec<bool>,
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            var_threshold: DEFAULT_VAR_THRESHOLD,
            min_contour_area: DEFAULT_MIN_CONTOUR_AREA,
            mask: default_mask(),
        }
    }
}

impl MotionSettings {
    /// Build settings from configured defaults, clamped into range.
    pub fn from_defaults(var_threshold: f64, min_contour_area: f64) -> Self {
        let mut s = Self {
            var_threshold,
            min_contour_area,
            mask: default_mask(),
        };
        s.clamp();
        s
    }

    /// Clamp the sliders into their valid ranges and normalize the mask length.
    /// Applied on load and on every update so out-of-range API/config input and
    /// stale on-disk state can never reach the detector.
    pub fn clamp(&mut self) {
        self.var_threshold = self
            .var_threshold
            .clamp(VAR_THRESHOLD_MIN, VAR_THRESHOLD_MAX);
        self.min_contour_area = self
            .min_contour_area
            .clamp(MIN_CONTOUR_AREA_MIN, MIN_CONTOUR_AREA_MAX);
        if self.mask.len() != MASK_CELLS {
            self.mask.resize(MASK_CELLS, false);
        }
    }
}

/// Partial update accepted by the settings API. Absent fields are left
/// unchanged.
#[derive(Debug, Default, Deserialize)]
pub struct SettingsUpdate {
    pub var_threshold: Option<f64>,
    pub min_contour_area: Option<f64>,
    pub mask: Option<Vec<bool>>,
}

struct CameraSettings {
    settings: MotionSettings,
    path: PathBuf,
}

/// Shared, `Arc`-backed store of per-camera motion settings. Cloned into both
/// the analyzer (which reads it each tick) and the HTTP layer (which serves and
/// updates it).
#[derive(Clone)]
pub struct MotionSettingsStore {
    cameras: Arc<HashMap<String, RwLock<CameraSettings>>>,
}

impl MotionSettingsStore {
    /// Load each camera's settings from disk (falling back to the configured
    /// defaults) and delete any stale learned-state files left by the removed
    /// auto-tuner / detection grid.
    pub fn new(
        camera_ids: &[String],
        data_dir: &Path,
        default_var_threshold: f64,
        default_min_contour_area: f64,
    ) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            remove_stale_learned_state(data_dir, id);
            let path = settings_path(data_dir, id);
            let settings = load(&path).unwrap_or_else(|| {
                MotionSettings::from_defaults(default_var_threshold, default_min_contour_area)
            });
            cameras.insert(id.clone(), RwLock::new(CameraSettings { settings, path }));
        }
        Self {
            cameras: Arc::new(cameras),
        }
    }

    /// Current settings for a camera, or `None` if it is unknown.
    pub fn get(&self, camera_id: &str) -> Option<MotionSettings> {
        let lock = self.cameras.get(camera_id)?;
        Some(lock.read_recover().settings.clone())
    }

    /// Apply a partial update, clamp, persist to disk, and return the new
    /// settings. `None` if the camera is unknown.
    pub fn update(&self, camera_id: &str, update: SettingsUpdate) -> Option<MotionSettings> {
        let lock = self.cameras.get(camera_id)?;
        let (path, settings) = {
            let mut cam = lock.write_recover();
            if let Some(v) = update.var_threshold {
                cam.settings.var_threshold = v;
            }
            if let Some(v) = update.min_contour_area {
                cam.settings.min_contour_area = v;
            }
            if let Some(m) = update.mask {
                cam.settings.mask = m;
            }
            cam.settings.clamp();
            (cam.path.clone(), cam.settings.clone())
        };
        save(&path, &settings);
        Some(settings)
    }
}

fn settings_path(data_dir: &Path, camera_id: &str) -> PathBuf {
    data_dir.join(camera_id).join("motion_settings.json")
}

fn load(path: &Path) -> Option<MotionSettings> {
    let data = std::fs::read_to_string(path).ok()?;
    let mut settings: MotionSettings = serde_json::from_str(&data).ok()?;
    settings.clamp();
    tracing::info!(path = %path.display(), "loaded motion settings");
    Some(settings)
}

fn save(path: &Path, settings: &MotionSettings) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                tracing::warn!(error = %e, "failed to save motion settings");
            }
        }
        Err(e) => tracing::warn!(error = %e, "failed to serialize motion settings"),
    }
}

/// Delete the persisted state of the removed auto-tuner and detection grid.
/// Called once per camera at startup so stale self-blinding state cannot linger.
fn remove_stale_learned_state(data_dir: &Path, camera_id: &str) {
    for name in ["motion_tuner.json", "detection_grid.json"] {
        let path = data_dir.join(camera_id).join(name);
        if !path.exists() {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(
                camera = %camera_id,
                file = name,
                "removed stale learned-state file (feature removed)"
            ),
            Err(e) => tracing::warn!(
                camera = %camera_id,
                file = name,
                error = %e,
                "failed to remove stale learned-state file"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn defaults_are_in_range() {
        let s = MotionSettings::default();
        assert_eq!(s.var_threshold, DEFAULT_VAR_THRESHOLD);
        assert_eq!(s.min_contour_area, DEFAULT_MIN_CONTOUR_AREA);
        assert_eq!(s.mask.len(), MASK_CELLS);
        assert!(s.mask.iter().all(|&m| !m));
    }

    #[test]
    fn clamp_bounds_sliders() {
        let mut s = MotionSettings {
            var_threshold: 1000.0,
            min_contour_area: 0.0,
            mask: vec![true; 3],
        };
        s.clamp();
        assert_eq!(s.var_threshold, VAR_THRESHOLD_MAX);
        assert_eq!(s.min_contour_area, MIN_CONTOUR_AREA_MIN);
        // Mask length normalized, existing cells preserved where they fit.
        assert_eq!(s.mask.len(), MASK_CELLS);
        assert!(s.mask[0] && s.mask[1] && s.mask[2]);
        assert!(!s.mask[3]);

        let mut low = MotionSettings {
            var_threshold: -5.0,
            min_contour_area: 99999.0,
            mask: default_mask(),
        };
        low.clamp();
        assert_eq!(low.var_threshold, VAR_THRESHOLD_MIN);
        assert_eq!(low.min_contour_area, MIN_CONTOUR_AREA_MAX);
    }

    #[test]
    fn json_roundtrip() {
        let mut s = MotionSettings {
            var_threshold: 24.0,
            min_contour_area: 350.0,
            mask: default_mask(),
        };
        s.mask[5] = true;
        s.mask[MASK_CELLS - 1] = true;

        let json = serde_json::to_string(&s).unwrap();
        let back: MotionSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
        assert!(back.mask[5] && back.mask[MASK_CELLS - 1]);
    }

    #[test]
    fn store_persists_and_reloads() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);

        let mut mask = default_mask();
        mask[10] = true;
        let updated = store
            .update(
                "cam1",
                SettingsUpdate {
                    var_threshold: Some(48.0),
                    min_contour_area: Some(500.0),
                    mask: Some(mask),
                },
            )
            .unwrap();
        assert_eq!(updated.var_threshold, 48.0);
        assert!(updated.mask[10]);

        // A fresh store over the same dir must see the persisted values.
        let store2 = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);
        let loaded = store2.get("cam1").unwrap();
        assert_eq!(loaded.var_threshold, 48.0);
        assert_eq!(loaded.min_contour_area, 500.0);
        assert!(loaded.mask[10]);
    }

    #[test]
    fn update_clamps_out_of_range() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);
        let updated = store
            .update(
                "cam1",
                SettingsUpdate {
                    var_threshold: Some(9999.0),
                    min_contour_area: Some(1.0),
                    mask: None,
                },
            )
            .unwrap();
        assert_eq!(updated.var_threshold, VAR_THRESHOLD_MAX);
        assert_eq!(updated.min_contour_area, MIN_CONTOUR_AREA_MIN);
    }

    #[test]
    fn unknown_camera_returns_none() {
        let dir = TempDir::new().unwrap();
        let store = MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        assert!(store.get("nope").is_none());
        assert!(store.update("nope", SettingsUpdate::default()).is_none());
    }

    #[test]
    fn removes_stale_learned_state_files() {
        let dir = TempDir::new().unwrap();
        let cam_dir = dir.path().join("cam1");
        std::fs::create_dir_all(&cam_dir).unwrap();
        let tuner = cam_dir.join("motion_tuner.json");
        let grid = cam_dir.join("detection_grid.json");
        std::fs::write(&tuner, "{}").unwrap();
        std::fs::write(&grid, "{}").unwrap();

        let _ = MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        assert!(!tuner.exists(), "motion_tuner.json should be deleted");
        assert!(!grid.exists(), "detection_grid.json should be deleted");
    }
}
