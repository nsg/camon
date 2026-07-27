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
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};

use crate::locks::{LockExt, MutexExt};

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
    /// excluded from motion detection deterministically. This is the
    /// "movement mask": nothing ever moves here.
    #[serde(default = "default_mask")]
    pub mask: Vec<bool>,
    /// One bool per 16x12 cell, row-major. `true` = blacked out: the cell's
    /// pixels are set to black in every frame handed to the vision model, so
    /// stationary nuisance objects never reach classification. This is the
    /// "detection mask": the model never sees these pixels. It has no effect
    /// on motion detection. Defaults to all-false so pre-existing
    /// `motion_settings.json` files load unchanged.
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
    /// Build settings from configured defaults, clamped into range.
    pub fn from_defaults(var_threshold: f64, min_contour_area: f64) -> Self {
        let mut s = Self {
            var_threshold,
            min_contour_area,
            mask: default_mask(),
            detection_mask: default_mask(),
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
        if self.detection_mask.len() != MASK_CELLS {
            self.detection_mask.resize(MASK_CELLS, false);
        }
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
    /// Serializes update-then-persist for this camera. Held across the disk
    /// write — which `state` deliberately is not, so the analyzer's per-tick
    /// read never waits on I/O — so that the order updates are applied is the
    /// order they reach disk. Without it two concurrent updates stage through
    /// the same `.tmp` path, and the later mutation can lose the rename to the
    /// earlier one, leaving the file disagreeing with the live settings.
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

    /// Apply a partial update, clamp, persist to disk, and return the new
    /// settings.
    ///
    /// The live state is updated before persistence and is *kept* even when the
    /// write fails: a mask is a privacy control that has to take effect on the
    /// next analysis tick, and refusing to apply it because the disk is full
    /// would leave the operator unable to stop the model seeing a sensitive
    /// area at all. `NotPersisted` says exactly that — applied now, gone on
    /// restart — instead of the silent success this used to report.
    pub fn update(
        &self,
        camera_id: &str,
        update: SettingsUpdate,
    ) -> Result<MotionSettings, UpdateError> {
        let cam = self
            .cameras
            .get(camera_id)
            .ok_or(UpdateError::UnknownCamera)?;
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
            state.settings.clamp();
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

fn load(path: &Path) -> Option<MotionSettings> {
    let data = std::fs::read_to_string(path).ok()?;
    let mut settings: MotionSettings = serde_json::from_str(&data).ok()?;
    settings.clamp();
    tracing::info!(path = %path.display(), "loaded motion settings");
    Some(settings)
}

/// Persist settings the way the storage layer commits an event: stage into
/// `motion_settings.json.tmp`, fsync it, then rename. `load` falls back to
/// unmasked defaults on any parse error, so a torn write would silently drop a
/// privacy mask; the rename makes that unreachable — a crash can only leave a
/// stale `.tmp` beside an intact previous file.
///
/// The containing directory is fsynced after the rename as well. `sync_all` on
/// the staging file only makes its *contents* durable; the directory entry the
/// rename swaps is not, so without this a crash just after a success response
/// could still bring back the previous mask, or lose a first-ever settings file
/// outright. Nothing sweeps a stray `motion_settings.json.tmp` at startup — the
/// warm storage recovery pass only walks the per-event-type subdirectories — so
/// there is no second chance here of the kind the event path has.
fn save(path: &Path, settings: &MotionSettings) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    // A directory that has just come into existence is itself only durable once
    // its own parent is synced; the settings file inside it would go with it.
    let dir_is_new = !dir.exists();
    std::fs::create_dir_all(dir)?;
    if dir_is_new {
        if let Some(grandparent) = dir.parent() {
            sync_dir(grandparent)?;
        }
    }

    let json = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let tmp = tmp_path(path);
    if let Err(e) = write_synced(&tmp, json.as_bytes()).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    sync_dir(dir)
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn write_synced(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(data)?;
    f.sync_all()
}

/// fsync a directory so a rename or creation inside it survives a crash.
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
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
            detection_mask: vec![true; 5],
        };
        s.clamp();
        assert_eq!(s.var_threshold, VAR_THRESHOLD_MAX);
        assert_eq!(s.min_contour_area, MIN_CONTOUR_AREA_MIN);
        // Mask length normalized, existing cells preserved where they fit.
        assert_eq!(s.mask.len(), MASK_CELLS);
        assert!(s.mask[0] && s.mask[1] && s.mask[2]);
        assert!(!s.mask[3]);
        // Detection mask is normalized independently.
        assert_eq!(s.detection_mask.len(), MASK_CELLS);
        assert!(s.detection_mask[0] && s.detection_mask[4]);
        assert!(!s.detection_mask[5]);

        let mut low = MotionSettings {
            var_threshold: -5.0,
            min_contour_area: 99999.0,
            mask: default_mask(),
            detection_mask: default_mask(),
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
        // A `motion_settings.json` written before the detection mask existed
        // has no `detection_mask` field; it must load with an all-false mask.
        let json = r#"{
            "var_threshold": 20.0,
            "min_contour_area": 300.0,
            "mask": [true, false, true]
        }"#;
        let mut s: MotionSettings = serde_json::from_str(json).unwrap();
        s.clamp();
        assert_eq!(s.detection_mask.len(), MASK_CELLS);
        assert!(s.detection_mask.iter().all(|&m| !m));
        // The movement mask still loads and is length-normalized.
        assert_eq!(s.mask.len(), MASK_CELLS);
        assert!(s.mask[0] && s.mask[2]);
    }

    #[test]
    fn update_detection_mask_independent_of_movement_mask() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);

        // Paint a movement-mask cell first.
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

        // A partial update touching only the detection mask must leave the
        // movement mask untouched.
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

        // Persisted and reloaded independently.
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
                    detection_mask: None,
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
        assert!(matches!(
            store.update("nope", SettingsUpdate::default()),
            Err(UpdateError::UnknownCamera)
        ));
    }

    /// A regular file where the camera directory belongs: nothing can be
    /// persisted underneath it, whatever user the test runs as.
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

        // Deliberate: the mask is live even though it is not durable, and the
        // error is what says so.
        assert!(store.get("cam1").unwrap().detection_mask[4]);
    }

    /// Every writer stages through the same `{name}.tmp`, so without the
    /// per-camera persistence lock one could rename a file another was still
    /// writing through — a live file holding a mix of two updates — or land an
    /// older write after a newer one and leave the file disagreeing with the
    /// settings the detector is using.
    #[test]
    fn concurrent_updates_persist_exactly_one_of_them() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let store = MotionSettingsStore::new(&ids, dir.path(), 16.0, 200.0);

        let writers: Vec<_> = (0..8)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    // A whole-mask pattern keyed to the writer, so a file mixing
                    // two of them cannot match any single writer's update.
                    let mask: Vec<bool> = (0..MASK_CELLS).map(|c| c % 8 == i).collect();
                    store
                        .update(
                            "cam1",
                            SettingsUpdate {
                                var_threshold: Some(20.0 + i as f64),
                                detection_mask: Some(mask),
                                ..Default::default()
                            },
                        )
                        .unwrap();
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }

        let persisted =
            load(&settings_path(dir.path(), "cam1")).expect("no readable settings file");
        let i = (persisted.var_threshold - 20.0) as usize;
        assert!(
            i < 8,
            "var_threshold {} is no writer's",
            persisted.var_threshold
        );
        let expected: Vec<bool> = (0..MASK_CELLS).map(|c| c % 8 == i).collect();
        assert_eq!(persisted.detection_mask, expected, "file mixes two updates");

        // The last writer to persist is also the last to have mutated, so the
        // file can never be an older update than what the detector is reading.
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

        // A directory in the staging path blocks the write. Were the settings
        // written in place instead of staged and renamed, this would succeed
        // and `before` would change — and a crash at that point is exactly what
        // truncates the file into an unmasked default.
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
