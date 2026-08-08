//! Deterministic, user-controlled motion-detection settings, one set per camera.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};

use crate::durable::{create_dir_all_synced, sync_dir, tmp_path, write_synced};
use crate::locks::{LockExt, MutexExt};

/// Ignore-mask geometry: 16x12 cells over a 320x240 analysis frame.
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
    /// excluded from motion detection deterministically. This is the
    /// "movement mask": nothing ever moves here.
    #[serde(default = "default_mask")]
    pub mask: Vec<bool>,
    /// One bool per 16x12 cell, row-major.
    #[serde(default = "default_mask")]
    pub detection_mask: Vec<bool>,
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            var_threshold: DEFAULT_VAR_THRESHOLD,
            min_contour_area: DEFAULT_MIN_CONTOUR_AREA,
            mask: default_mask(),
            detection_mask: default_mask(),
        }
    }
}

impl MotionSettings {
    /// Build settings from configured defaults, forced into range.
    pub fn from_defaults(var_threshold: f64, min_contour_area: f64) -> Self {
        let mut s = Self {
            var_threshold,
            min_contour_area,
            mask: default_mask(),
            detection_mask: default_mask(),
        };
        s.sanitize();
        s
    }

    /// Force the sliders into their valid ranges and normalize the mask length. Applied on load
    /// and on every update so out-of-range API/config input and stale on-disk state can never
    /// reach the detector.
    pub fn sanitize(&mut self) {
        self.var_threshold = bounded(
            self.var_threshold,
            DEFAULT_VAR_THRESHOLD,
            VAR_THRESHOLD_MIN,
            VAR_THRESHOLD_MAX,
        );
        self.min_contour_area = bounded(
            self.min_contour_area,
            DEFAULT_MIN_CONTOUR_AREA,
            MIN_CONTOUR_AREA_MIN,
            MIN_CONTOUR_AREA_MAX,
        );
        if self.mask.len() != MASK_CELLS {
            self.mask.resize(MASK_CELLS, false);
        }
        if self.detection_mask.len() != MASK_CELLS {
            self.detection_mask.resize(MASK_CELLS, false);
        }
    }
}

/// One slider held to `min..=max`, with a value that is not a real number
/// replaced by `default` rather than clamped — see [`MotionSettings::sanitize`].
fn bounded(value: f64, default: f64, min: f64, max: f64) -> f64 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        default
    }
}

/// Why an update did not fully succeed.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("camera not found")]
    UnknownCamera,
    /// The new settings are live in the running detector but did not reach
    /// disk, so they are lost on the next restart.
    #[error("settings applied to the running detector but not saved: {0}")]
    NotPersisted(#[source] std::io::Error),
    /// A slider was sent a value that is not a real number.
    #[error("{field} must be a real number, got {value}")]
    NotANumber { field: &'static str, value: f64 },
}

/// Partial update accepted by the settings API. Absent fields are left
/// unchanged.
#[derive(Debug, Default, Deserialize)]
pub struct SettingsUpdate {
    pub var_threshold: Option<f64>,
    pub min_contour_area: Option<f64>,
    pub mask: Option<Vec<bool>>,
    pub detection_mask: Option<Vec<bool>>,
}

struct CameraSettings {
    settings: MotionSettings,
    path: PathBuf,
}

struct Camera {
    state: RwLock<CameraSettings>,
    /// Serializes update-then-persist for this camera.
    persist: Mutex<()>,
}

/// Shared, `Arc`-backed store of per-camera motion settings. Cloned into both
/// the analyzer (which reads it each tick) and the HTTP layer (which serves and
/// updates it).
#[derive(Clone)]
pub struct MotionSettingsStore {
    cameras: Arc<HashMap<String, Camera>>,
}

impl MotionSettingsStore {
    /// Load each camera's settings from disk (falling back to the configured defaults) and
    /// delete any stale learned-state files left by the removed auto-tuner / detection grid.
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
            let defaults =
                || MotionSettings::from_defaults(default_var_threshold, default_min_contour_area);
            let settings = match load(&path) {
                Persisted::Settings(settings) => settings,
                Persisted::Absent => defaults(),
                Persisted::Corrupt => MotionSettings {
                    detection_mask: vec![true; MASK_CELLS],
                    ..defaults()
                },
            };
            cameras.insert(
                id.clone(),
                Camera {
                    state: RwLock::new(CameraSettings { settings, path }),
                    persist: Mutex::new(()),
                },
            );
        }
        Self {
            cameras: Arc::new(cameras),
        }
    }

    /// Current settings for a camera, or `None` if it is unknown.
    pub fn get(&self, camera_id: &str) -> Option<MotionSettings> {
        let cam = self.cameras.get(camera_id)?;
        Some(cam.state.read_recover().settings.clone())
    }

    /// Apply a partial update, clamp, persist to disk, and return the new settings.
    pub fn update(
        &self,
        camera_id: &str,
        update: SettingsUpdate,
    ) -> Result<MotionSettings, UpdateError> {
        let cam = self
            .cameras
            .get(camera_id)
            .ok_or(UpdateError::UnknownCamera)?;
        // Before anything is locked or mutated: a rejected update leaves the
        // live settings exactly as the detector already had them.
        for (field, value) in [
            ("var_threshold", update.var_threshold),
            ("min_contour_area", update.min_contour_area),
        ] {
            if let Some(value) = value.filter(|v| !v.is_finite()) {
                return Err(UpdateError::NotANumber { field, value });
            }
        }
        // Taken before the mutation, not just around the write, so that two
        // concurrent updates are applied and persisted in the same order.
        let _persist = cam.persist.lock_recover();
        let (path, settings) = {
            let mut state = cam.state.write_recover();
            if let Some(v) = update.var_threshold {
                state.settings.var_threshold = v;
            }
            if let Some(v) = update.min_contour_area {
                state.settings.min_contour_area = v;
            }
            if let Some(m) = update.mask {
                state.settings.mask = m;
            }
            if let Some(m) = update.detection_mask {
                state.settings.detection_mask = m;
            }
            state.settings.sanitize();
            (state.path.clone(), state.settings.clone())
        };
        match save(&path, &settings) {
            Ok(()) => Ok(settings),
            Err(e) => {
                tracing::warn!(camera = %camera_id, path = %path.display(), error = %e,
                    "failed to save motion settings");
                Err(UpdateError::NotPersisted(e))
            }
        }
    }
}

fn settings_path(data_dir: &Path, camera_id: &str) -> PathBuf {
    data_dir.join(camera_id).join("motion_settings.json")
}

/// What one camera's `motion_settings.json` turned out to be.
enum Persisted {
    Settings(MotionSettings),
    Absent,
    Corrupt,
}

/// Read one camera's persisted settings.
fn load(path: &Path) -> Persisted {
    let data = match std::fs::read_to_string(path) {
        Ok(data) => data,
        // A camera seen for the first time has no file yet; that is the
        // ordinary case, not a fault.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Persisted::Absent,
        Err(e) => {
            tracing::warn!(
                path = %path.display(), error = %e,
                "motion settings file cannot be read; starting this camera from the configured \
                 defaults with its detection mask fully painted, so no frame reaches the \
                 vision model until the mask is set again"
            );
            return Persisted::Corrupt;
        }
    };
    let mut settings: MotionSettings = match serde_json::from_str(&data) {
        Ok(settings) => settings,
        Err(e) => {
            tracing::warn!(
                path = %path.display(), error = %e,
                "motion settings file cannot be parsed; starting this camera from the \
                 configured defaults with its detection mask fully painted, so no frame \
                 reaches the vision model until the mask is set again"
            );
            return Persisted::Corrupt;
        }
    };
    settings.sanitize();
    tracing::info!(path = %path.display(), "loaded motion settings");
    Persisted::Settings(settings)
}

/// Persist settings the way the storage layer commits an event: stage into
/// `motion_settings.json.tmp`, fsync it, rename, then fsync the directory.
fn save(path: &Path, settings: &MotionSettings) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    create_dir_all_synced(dir)?;

    let json = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let tmp = tmp_path(path);
    if let Err(e) = write_synced(&tmp, json.as_bytes()).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    sync_dir(dir)
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
    fn sanitize_bounds_sliders() {
        let mut s = MotionSettings {
            var_threshold: 1000.0,
            min_contour_area: 0.0,
            mask: vec![true; 3],
            detection_mask: vec![true; 5],
        };
        s.sanitize();
        assert_eq!(s.var_threshold, VAR_THRESHOLD_MAX);
        assert_eq!(s.min_contour_area, MIN_CONTOUR_AREA_MIN);
        assert_eq!(s.mask.len(), MASK_CELLS);
        assert!(s.mask[0] && s.mask[1] && s.mask[2]);
        assert!(!s.mask[3]);
        assert_eq!(s.detection_mask.len(), MASK_CELLS);
        assert!(s.detection_mask[0] && s.detection_mask[4]);
        assert!(!s.detection_mask[5]);

        let mut low = MotionSettings {
            var_threshold: -5.0,
            min_contour_area: 99999.0,
            mask: default_mask(),
            detection_mask: default_mask(),
        };
        low.sanitize();
        assert_eq!(low.var_threshold, VAR_THRESHOLD_MIN);
        assert_eq!(low.min_contour_area, MIN_CONTOUR_AREA_MAX);
    }

    #[test]
    fn json_roundtrip() {
        let mut s = MotionSettings {
            var_threshold: 24.0,
            min_contour_area: 350.0,
            mask: default_mask(),
            detection_mask: default_mask(),
        };
        s.mask[5] = true;
        s.mask[MASK_CELLS - 1] = true;
        s.detection_mask[7] = true;

        let json = serde_json::to_string(&s).unwrap();
        let back: MotionSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
        assert!(back.mask[5] && back.mask[MASK_CELLS - 1]);
        assert!(back.detection_mask[7] && !back.detection_mask[5]);
    }

    #[test]
    fn detection_mask_defaults_when_absent() {
        let json = r#"{
            "var_threshold": 20.0,
            "min_contour_area": 300.0,
            "mask": [true, false, true]
        }"#;
        let mut s: MotionSettings = serde_json::from_str(json).unwrap();
        s.sanitize();
        assert_eq!(s.detection_mask.len(), MASK_CELLS);
        assert!(s.detection_mask.iter().all(|&m| !m));
        assert_eq!(s.mask.len(), MASK_CELLS);
        assert!(s.mask[0] && s.mask[2]);
    }

    #[test]
    fn update_detection_mask_independent_of_movement_mask() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);

        let mut movement = default_mask();
        movement[1] = true;
        store
            .update(
                "cam1",
                SettingsUpdate {
                    mask: Some(movement),
                    ..Default::default()
                },
            )
            .unwrap();

        let mut detection = default_mask();
        detection[2] = true;
        let updated = store
            .update(
                "cam1",
                SettingsUpdate {
                    detection_mask: Some(detection),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(updated.mask[1], "movement mask preserved");
        assert!(updated.detection_mask[2], "detection mask applied");
        assert!(!updated.detection_mask[1]);
        assert!(!updated.mask[2]);

        let store2 = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);
        let loaded = store2.get("cam1").unwrap();
        assert!(loaded.mask[1]);
        assert!(loaded.detection_mask[2]);
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
                    detection_mask: None,
                },
            )
            .unwrap();
        assert_eq!(updated.var_threshold, 48.0);
        assert!(updated.mask[10]);

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
                    detection_mask: None,
                },
            )
            .unwrap();
        assert_eq!(updated.var_threshold, VAR_THRESHOLD_MAX);
        assert_eq!(updated.min_contour_area, MIN_CONTOUR_AREA_MIN);
    }

    #[test]
    fn a_slider_that_is_not_a_number_becomes_its_default() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut s = MotionSettings {
                var_threshold: bad,
                min_contour_area: bad,
                mask: default_mask(),
                detection_mask: default_mask(),
            };
            s.sanitize();
            assert_eq!(s.var_threshold, DEFAULT_VAR_THRESHOLD, "from {bad}");
            assert_eq!(s.min_contour_area, DEFAULT_MIN_CONTOUR_AREA, "from {bad}");
            assert_ne!(s.var_threshold, VAR_THRESHOLD_MAX, "clamped, not defaulted");
            assert_ne!(
                s.min_contour_area, MIN_CONTOUR_AREA_MIN,
                "clamped, not defaulted"
            );
        }

        let seeded = MotionSettings::from_defaults(f64::NAN, f64::NAN);
        assert_eq!(seeded.var_threshold, DEFAULT_VAR_THRESHOLD);
        assert_eq!(seeded.min_contour_area, DEFAULT_MIN_CONTOUR_AREA);
    }

    #[test]
    fn an_update_that_is_not_a_number_is_refused_and_changes_nothing() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);
        store
            .update(
                "cam1",
                SettingsUpdate {
                    var_threshold: Some(48.0),
                    min_contour_area: Some(500.0),
                    ..Default::default()
                },
            )
            .unwrap();

        for (field, update) in [
            (
                "var_threshold",
                SettingsUpdate {
                    var_threshold: Some(f64::NAN),
                    min_contour_area: Some(600.0),
                    ..Default::default()
                },
            ),
            (
                "min_contour_area",
                SettingsUpdate {
                    min_contour_area: Some(f64::INFINITY),
                    ..Default::default()
                },
            ),
        ] {
            let err = store
                .update("cam1", update)
                .expect_err("accepted a non-number");
            assert!(
                matches!(err, UpdateError::NotANumber { field: f, .. } if f == field),
                "got {err:?}"
            );

            let live = store.get("cam1").unwrap();
            assert_eq!(live.var_threshold, 48.0, "live settings survived {field}");
            assert_eq!(live.min_contour_area, 500.0, "no half-applied update");
        }

        let reloaded = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);
        assert_eq!(reloaded.get("cam1").unwrap().var_threshold, 48.0);
    }

    fn store_over_settings_file(dir: &TempDir, content: &str) -> MotionSettingsStore {
        let path = settings_path(dir.path(), "cam1");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0)
    }

    #[test]
    fn an_unreadable_settings_file_starts_from_the_defaults() {
        let ids = vec!["cam1".to_string()];
        for content in [
            r#"{"var_threshold": null, "min_contour_area": 200.0}"#,
            "{ truncated",
            "",
        ] {
            let dir = TempDir::new().unwrap();
            let store = store_over_settings_file(&dir, content);
            let settings = store.get("cam1").expect("camera dropped over a bad file");
            assert_eq!(settings.var_threshold, 16.0, "{content:?}");
            assert_eq!(settings.min_contour_area, 200.0, "{content:?}");

            store
                .update(
                    "cam1",
                    SettingsUpdate {
                        var_threshold: Some(32.0),
                        ..Default::default()
                    },
                )
                .unwrap();
            let reloaded = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);
            assert_eq!(reloaded.get("cam1").unwrap().var_threshold, 32.0);
        }
    }

    #[test]
    fn a_lost_settings_file_hides_the_camera_from_the_model_but_keeps_it_recording() {
        let dir = TempDir::new().unwrap();
        let store = store_over_settings_file(&dir, "{ truncated");
        let settings = store.get("cam1").unwrap();

        assert!(
            settings.detection_mask.iter().all(|&c| c),
            "the model must see nothing until the mask is painted again"
        );
        assert!(
            settings.mask.iter().all(|&c| !c),
            "a closed movement mask would stop the camera recording"
        );
        assert_eq!(settings.var_threshold, 16.0);
        assert_eq!(settings.min_contour_area, 200.0);

        let updated = store
            .update(
                "cam1",
                SettingsUpdate {
                    detection_mask: Some(default_mask()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(updated.detection_mask.iter().all(|&c| !c));
        let reloaded = MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        assert!(reloaded
            .get("cam1")
            .unwrap()
            .detection_mask
            .iter()
            .all(|&c| !c));
    }

    #[test]
    fn a_camera_with_no_settings_file_yet_starts_unmasked() {
        let dir = TempDir::new().unwrap();
        let store = MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        let settings = store.get("cam1").unwrap();
        assert!(settings.detection_mask.iter().all(|&c| !c));
        assert!(settings.mask.iter().all(|&c| !c));
    }

    #[test]
    fn a_settings_path_that_cannot_be_read_fails_closed_the_same_way() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(settings_path(dir.path(), "cam1")).unwrap();

        let store = MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        let settings = store.get("cam1").expect("camera dropped over an I/O error");
        assert!(settings.detection_mask.iter().all(|&c| c));
        assert!(settings.mask.iter().all(|&c| !c));
        assert_eq!(settings.var_threshold, 16.0);
    }

    #[derive(Clone, Default)]
    struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock_recover().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn every_corrupt_settings_file_is_reported_at_warn() {
        for (case, content) in [("parse", "{ truncated"), ("io", "")] {
            let dir = TempDir::new().unwrap();
            let logs = CapturedLog::default();
            {
                let _reader = tracing::subscriber::set_default(
                    tracing_subscriber::fmt()
                        .with_writer(logs.clone())
                        .with_max_level(tracing::Level::WARN)
                        .with_ansi(false)
                        .finish(),
                );
                if case == "io" {
                    std::fs::create_dir_all(settings_path(dir.path(), "cam1")).unwrap();
                    MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
                } else {
                    store_over_settings_file(&dir, content);
                }
            }

            let written = String::from_utf8(logs.0.lock_recover().clone()).unwrap();
            assert!(written.contains("WARN"), "{case}: not a warning: {written}");
            assert!(
                written.contains("motion_settings.json"),
                "{case}: does not name the file: {written}"
            );
            assert!(
                written.contains("detection mask"),
                "{case}: does not say what it did: {written}"
            );
        }
    }

    #[test]
    fn a_camera_with_no_settings_file_is_not_warned_about() {
        let dir = TempDir::new().unwrap();
        let logs = CapturedLog::default();
        {
            let _reader = tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .with_writer(logs.clone())
                    .with_max_level(tracing::Level::WARN)
                    .with_ansi(false)
                    .finish(),
            );
            MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        }
        let written = String::from_utf8(logs.0.lock_recover().clone()).unwrap();
        assert!(written.is_empty(), "unexpected log: {written}");
    }

    #[test]
    fn unknown_camera_returns_none() {
        let dir = TempDir::new().unwrap();
        let store = MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0);
        assert!(store.get("nope").is_none());
        assert!(matches!(
            store.update("nope", SettingsUpdate::default()),
            Err(UpdateError::UnknownCamera)
        ));
    }

    fn unwritable_store(dir: &TempDir) -> MotionSettingsStore {
        std::fs::write(dir.path().join("cam1"), b"not a directory").unwrap();
        MotionSettingsStore::new(&["cam1".to_string()], dir.path(), 16.0, 200.0)
    }

    #[test]
    fn update_reports_persistence_failure_instead_of_success() {
        let dir = TempDir::new().unwrap();
        let store = unwritable_store(&dir);

        let mut mask = default_mask();
        mask[4] = true;
        let err = store
            .update(
                "cam1",
                SettingsUpdate {
                    detection_mask: Some(mask),
                    ..Default::default()
                },
            )
            .expect_err("unsaved settings reported as success");
        assert!(matches!(err, UpdateError::NotPersisted(_)), "{err}");

        assert!(store.get("cam1").unwrap().detection_mask[4]);
    }

    #[test]
    fn concurrent_updates_persist_exactly_one_of_them() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);

        let writers: Vec<_> = (0..8)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    let mask: Vec<bool> = (0..MASK_CELLS).map(|c| c % 8 == i).collect();
                    for _ in 0..50 {
                        store
                            .update(
                                "cam1",
                                SettingsUpdate {
                                    var_threshold: Some(20.0 + i as f64),
                                    detection_mask: Some(mask.clone()),
                                    ..Default::default()
                                },
                            )
                            .expect("a save collided with a concurrent one");
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }

        let Persisted::Settings(persisted) = load(&settings_path(dir.path(), "cam1")) else {
            panic!("no readable settings file");
        };
        let i = (persisted.var_threshold - 20.0) as usize;
        assert!(
            i < 8,
            "var_threshold {} is no writer's",
            persisted.var_threshold
        );
        let expected: Vec<bool> = (0..MASK_CELLS).map(|c| c % 8 == i).collect();
        assert_eq!(persisted.detection_mask, expected, "file mixes two updates");

        assert_eq!(persisted, store.get("cam1").unwrap());
    }

    #[test]
    fn save_replaces_by_rename_and_never_writes_the_live_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("cam1").join("motion_settings.json");
        let mut first = MotionSettings::default();
        first.detection_mask[3] = true;
        save(&path, &first).unwrap();
        let before = std::fs::read_to_string(&path).unwrap();
        assert!(!tmp_path(&path).exists(), "staging file left behind");

        std::fs::create_dir(tmp_path(&path)).unwrap();
        let mut second = MotionSettings::default();
        second.detection_mask[7] = true;
        assert!(save(&path, &second).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
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
