use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::locks::LockExt;

const GRID_COLS: usize = 16;
const GRID_ROWS: usize = 12;
const GRID_SIZE: usize = GRID_COLS * GRID_ROWS;

/// Cell value above which detections are suppressed.
const ABSORPTION_THRESHOLD: f32 = 0.6;
/// Decay applied per minute. At 0.001/min a fully absorbed cell (0.6)
/// takes ~10 hours to decay back to novel after the object leaves.
const DECAY_PER_MINUTE: f32 = 0.001;
/// Minimum interval between decay ticks.
const DECAY_INTERVAL_SECS: u64 = 60;
/// Increment per detection hit. At 0.15 per hit, 4 detections at the
/// same grid cell reach the absorption threshold (0.6). With motion-gated
/// detection rates of ~1-3/hour, a stationary object absorbs in 1-4 hours.
const HIT_INCREMENT: f32 = 0.15;

#[derive(Serialize, Deserialize, Clone)]
struct ClassGrid {
    cells: Vec<f32>,
}

impl ClassGrid {
    fn new() -> Self {
        Self {
            cells: vec![0.0; GRID_SIZE],
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct CameraGrid {
    classes: HashMap<String, ClassGrid>,
}

impl CameraGrid {
    fn new() -> Self {
        Self {
            classes: HashMap::new(),
        }
    }
}

struct CameraState {
    grid: CameraGrid,
    last_decay: Instant,
}

#[derive(Clone)]
pub struct DetectionGrid {
    cameras: Arc<HashMap<String, RwLock<CameraState>>>,
    data_dir: PathBuf,
}

/// JSON response for the API endpoint.
#[derive(Serialize)]
pub struct GridResponse {
    pub cols: usize,
    pub rows: usize,
    pub classes: HashMap<String, Vec<f32>>,
}

impl DetectionGrid {
    pub fn new(camera_ids: &[String], data_dir: PathBuf) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            let grid = Self::load_grid(&data_dir, id).unwrap_or_else(CameraGrid::new);
            cameras.insert(
                id.clone(),
                RwLock::new(CameraState {
                    grid,
                    last_decay: Instant::now(),
                }),
            );
        }
        Self {
            cameras: Arc::new(cameras),
            data_dir,
        }
    }

    /// Record a detection across all grid cells overlapping the given
    /// normalized rects (x, y, w, h in 0.0-1.0). Returns true if any
    /// touched cell was below the absorption threshold (i.e. novel).
    pub fn record_rects(
        &self,
        camera_id: &str,
        class: &str,
        rects: &[(f32, f32, f32, f32)],
    ) -> bool {
        if rects.is_empty() {
            return true;
        }

        let lock = match self.cameras.get(camera_id) {
            Some(l) => l,
            None => return true,
        };

        let mut state = lock.write_recover();
        let class_grid = state
            .grid
            .classes
            .entry(class.to_string())
            .or_insert_with(ClassGrid::new);

        let mut any_novel = false;

        for &(x, y, w, h) in rects {
            let col_min = ((x * GRID_COLS as f32) as usize).min(GRID_COLS - 1);
            let col_max = (((x + w) * GRID_COLS as f32).ceil() as usize).min(GRID_COLS);
            let row_min = ((y * GRID_ROWS as f32) as usize).min(GRID_ROWS - 1);
            let row_max = (((y + h) * GRID_ROWS as f32).ceil() as usize).min(GRID_ROWS);

            for row in row_min..row_max {
                for col in col_min..col_max {
                    let idx = row * GRID_COLS + col;
                    class_grid.cells[idx] = (class_grid.cells[idx] + HIT_INCREMENT).min(1.0);
                    if class_grid.cells[idx] < ABSORPTION_THRESHOLD {
                        any_novel = true;
                    }
                }
            }
        }

        any_novel
    }

    /// Decay all cells. Safe to call frequently — only applies decay
    /// when at least DECAY_INTERVAL_SECS has elapsed since the last tick.
    pub fn decay(&self, camera_id: &str) {
        let lock = match self.cameras.get(camera_id) {
            Some(l) => l,
            None => return,
        };

        let mut state = lock.write_recover();
        let elapsed = state.last_decay.elapsed();
        if elapsed.as_secs() < DECAY_INTERVAL_SECS {
            return;
        }
        let minutes = elapsed.as_secs_f32() / 60.0;
        let decay = DECAY_PER_MINUTE * minutes;
        state.last_decay = Instant::now();

        for class_grid in state.grid.classes.values_mut() {
            for cell in &mut class_grid.cells {
                *cell = (*cell - decay).max(0.0);
            }
        }
    }

    /// Get grid state for the API.
    pub fn get_grid(&self, camera_id: &str) -> Option<GridResponse> {
        let lock = self.cameras.get(camera_id)?;
        let state = lock.read_recover();
        let classes: HashMap<String, Vec<f32>> = state
            .grid
            .classes
            .iter()
            .filter(|(_, g)| g.cells.iter().any(|&v| v > 0.0))
            .map(|(class, g)| (class.clone(), g.cells.clone()))
            .collect();
        Some(GridResponse {
            cols: GRID_COLS,
            rows: GRID_ROWS,
            classes,
        })
    }

    /// Save grid state to disk for a specific camera.
    pub fn save(&self, camera_id: &str) {
        let lock = match self.cameras.get(camera_id) {
            Some(l) => l,
            None => return,
        };
        let state = lock.read_recover();
        let path = self.grid_path(camera_id);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string(&state.grid) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    tracing::warn!(camera = %camera_id, error = %e, "failed to save detection grid");
                }
            }
            Err(e) => {
                tracing::warn!(camera = %camera_id, error = %e, "failed to serialize detection grid");
            }
        }
    }

    fn load_grid(data_dir: &std::path::Path, camera_id: &str) -> Option<CameraGrid> {
        let path = data_dir.join(camera_id).join("detection_grid.json");
        let data = std::fs::read_to_string(&path).ok()?;
        let grid: CameraGrid = serde_json::from_str(&data).ok()?;
        tracing::info!(camera = %camera_id, classes = grid.classes.len(), "loaded detection grid");
        Some(grid)
    }

    fn grid_path(&self, camera_id: &str) -> PathBuf {
        self.data_dir.join(camera_id).join("detection_grid.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_grid(cameras: &[&str]) -> (DetectionGrid, TempDir) {
        let dir = TempDir::new().unwrap();
        let ids: Vec<String> = cameras.iter().map(|s| s.to_string()).collect();
        let grid = DetectionGrid::new(&ids, dir.path().to_path_buf());
        (grid, dir)
    }

    /// Build a tiny rect that covers exactly one grid cell at the given
    /// normalized coordinate, matching the semantics of the old `record()`.
    fn point_rect(cx: f32, cy: f32) -> Vec<(f32, f32, f32, f32)> {
        let cw = 1.0 / GRID_COLS as f32;
        let ch = 1.0 / GRID_ROWS as f32;
        let col = (cx * GRID_COLS as f32) as usize;
        let row = (cy * GRID_ROWS as f32) as usize;
        vec![(col as f32 * cw, row as f32 * ch, cw * 0.5, ch * 0.5)]
    }

    fn cell_idx(cx: f32, cy: f32) -> usize {
        let col = ((cx * GRID_COLS as f32) as usize).min(GRID_COLS - 1);
        let row = ((cy * GRID_ROWS as f32) as usize).min(GRID_ROWS - 1);
        row * GRID_COLS + col
    }

    #[test]
    fn record_returns_novel_until_absorbed() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let r = point_rect(0.5, 0.5);

        // Each hit adds 0.15. Threshold is 0.6. So 4 hits = 0.60.
        for i in 0..3 {
            let novel = grid.record_rects("cam1", "car", &r);
            assert!(novel, "should be novel at hit {i}");
        }

        // 4th hit reaches 0.60 — absorbed
        let novel = grid.record_rects("cam1", "car", &r);
        assert!(!novel, "should be absorbed after 4 hits");
    }

    #[test]
    fn different_classes_are_independent() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let r = point_rect(0.5, 0.5);

        for _ in 0..4 {
            grid.record_rects("cam1", "car", &r);
        }

        let novel = grid.record_rects("cam1", "person", &r);
        assert!(novel, "different class should be independent");
    }

    #[test]
    fn different_cells_are_independent() {
        let (grid, _dir) = make_grid(&["cam1"]);

        for _ in 0..4 {
            grid.record_rects("cam1", "car", &point_rect(0.1, 0.1));
        }

        let novel = grid.record_rects("cam1", "car", &point_rect(0.9, 0.9));
        assert!(novel, "different cell should be independent");
    }

    #[test]
    fn decay_is_time_gated() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let r = point_rect(0.5, 0.5);

        for _ in 0..3 {
            grid.record_rects("cam1", "car", &r);
        }

        // Calling decay immediately should not reduce values (interval not reached)
        grid.decay("cam1");

        let response = grid.get_grid("cam1").unwrap();
        let val = response.classes["car"][cell_idx(0.5, 0.5)];
        assert!(
            (val - 0.45).abs() < 0.01,
            "value should be ~0.45 (3 * 0.15) with no decay yet, got {val}"
        );
    }

    #[test]
    fn decay_applies_after_interval() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let r = point_rect(0.5, 0.5);

        for _ in 0..4 {
            grid.record_rects("cam1", "car", &r);
        }
        // Value is 0.60

        // Force the last_decay timestamp back to trigger decay
        {
            let lock = grid.cameras.get("cam1").unwrap();
            let mut state = lock.write_recover();
            state.last_decay = Instant::now() - std::time::Duration::from_secs(120);
        }

        grid.decay("cam1");

        let response = grid.get_grid("cam1").unwrap();
        let val = response.classes["car"][cell_idx(0.5, 0.5)];
        // 2 minutes of decay: 0.60 - (0.001 * 2) = 0.598
        assert!(
            (val - 0.598).abs() < 0.01,
            "value should be ~0.598 after 2 min decay, got {val}"
        );
    }

    #[test]
    fn decay_does_not_go_below_zero() {
        let (grid, _dir) = make_grid(&["cam1"]);
        grid.record_rects("cam1", "car", &point_rect(0.5, 0.5));

        // Force a very long elapsed time
        {
            let lock = grid.cameras.get("cam1").unwrap();
            let mut state = lock.write_recover();
            state.last_decay = Instant::now() - std::time::Duration::from_secs(86400);
        }
        grid.decay("cam1");

        let response = grid.get_grid("cam1").unwrap();
        for cells in response.classes.values() {
            for &v in cells {
                assert!(v >= 0.0, "cell value should not be negative");
            }
        }
    }

    #[test]
    fn value_capped_at_one() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let r = point_rect(0.5, 0.5);

        for _ in 0..100 {
            grid.record_rects("cam1", "car", &r);
        }

        let response = grid.get_grid("cam1").unwrap();
        let cells = &response.classes["car"];
        let idx = cell_idx(0.5, 0.5);
        assert!(cells[idx] <= 1.0, "value should be capped at 1.0");
        assert!(cells[idx] >= 0.99, "value should be near 1.0");
    }

    #[test]
    fn unknown_camera_returns_novel() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let novel = grid.record_rects("unknown", "car", &point_rect(0.5, 0.5));
        assert!(novel, "unknown camera should always return novel");
    }

    #[test]
    fn get_grid_returns_none_for_unknown_camera() {
        let (grid, _dir) = make_grid(&["cam1"]);
        assert!(grid.get_grid("unknown").is_none());
    }

    #[test]
    fn get_grid_filters_empty_classes() {
        let (grid, _dir) = make_grid(&["cam1"]);

        grid.record_rects("cam1", "car", &point_rect(0.5, 0.5));
        let response = grid.get_grid("cam1").unwrap();
        assert!(response.classes.contains_key("car"));

        // Force long elapsed time and decay
        {
            let lock = grid.cameras.get("cam1").unwrap();
            let mut state = lock.write_recover();
            state.last_decay = Instant::now() - std::time::Duration::from_secs(86400);
        }
        grid.decay("cam1");

        let response = grid.get_grid("cam1").unwrap();
        assert!(
            !response.classes.contains_key("car"),
            "fully decayed class should be filtered"
        );
    }

    #[test]
    fn grid_dimensions_are_correct() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let response = grid.get_grid("cam1").unwrap();
        assert_eq!(response.cols, GRID_COLS);
        assert_eq!(response.rows, GRID_ROWS);
    }

    #[test]
    fn coordinate_mapping_edges() {
        let (grid, _dir) = make_grid(&["cam1"]);

        grid.record_rects("cam1", "car", &point_rect(0.0, 0.0));
        grid.record_rects("cam1", "car", &[(0.9, 0.9, 0.1, 0.1)]);

        let response = grid.get_grid("cam1").unwrap();
        let cells = &response.classes["car"];

        assert!(cells[0] > 0.0, "top-left should be recorded");

        let last_idx = (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1);
        assert!(cells[last_idx] > 0.0, "bottom-right should be recorded");
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];
        let r_car = point_rect(0.3, 0.7);
        let r_person = point_rect(0.8, 0.2);

        let grid = DetectionGrid::new(&ids, dir.path().to_path_buf());
        for _ in 0..10 {
            grid.record_rects("cam1", "car", &r_car);
        }
        for _ in 0..5 {
            grid.record_rects("cam1", "person", &r_person);
        }
        grid.save("cam1");

        let grid2 = DetectionGrid::new(&ids, dir.path().to_path_buf());
        let response = grid2.get_grid("cam1").unwrap();

        assert!(response.classes.contains_key("car"));
        assert!(response.classes.contains_key("person"));

        let car_val = response.classes["car"][cell_idx(0.3, 0.7)];
        // 10 * 0.15 = 1.5, capped at 1.0
        assert!(
            (car_val - 1.0).abs() < 0.01,
            "car value should be 1.0 (10 * 0.15 capped), got {car_val}"
        );
    }

    #[test]
    fn record_rects_marks_multiple_cells() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Rect spanning columns 4-7, rows 3-5 (normalized: x=0.25..0.5, y=0.25..0.5)
        let novel = grid.record_rects("cam1", "car", &[(0.25, 0.25, 0.25, 0.25)]);
        assert!(novel);

        let response = grid.get_grid("cam1").unwrap();
        let cells = &response.classes["car"];

        // Check that cells in the rect region got incremented
        for row in 3..6 {
            for col in 4..8 {
                let idx = row * GRID_COLS + col;
                assert!(
                    cells[idx] > 0.0,
                    "cell ({col}, {row}) should be marked, got {}",
                    cells[idx]
                );
            }
        }

        // Check that a cell outside the rect was NOT marked
        assert!(
            cells[0] == 0.0,
            "cell (0,0) should be unmarked, got {}",
            cells[0]
        );
    }

    #[test]
    fn record_rects_empty_returns_novel() {
        let (grid, _dir) = make_grid(&["cam1"]);
        assert!(grid.record_rects("cam1", "car", &[]));
    }

    #[test]
    fn record_rects_absorbs_after_repeated_hits() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let rect = &[(0.0, 0.0, 0.0625, 0.0833)]; // Single cell (col 0, row 0)

        for _ in 0..3 {
            assert!(grid.record_rects("cam1", "car", rect));
        }
        // 4th hit reaches absorption (4 * 0.15 = 0.60)
        assert!(
            !grid.record_rects("cam1", "car", rect),
            "should be absorbed after 4 hits"
        );
    }

    #[test]
    fn record_rects_novel_if_any_cell_below_threshold() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Absorb cell (0,0) via single-cell rect
        let small = &[(0.0, 0.0, 0.0625, 0.0833)];
        for _ in 0..4 {
            grid.record_rects("cam1", "car", small);
        }

        // Now record a wider rect that includes (0,0) plus fresh cells
        let wide = &[(0.0, 0.0, 0.25, 0.0833)];
        assert!(
            grid.record_rects("cam1", "car", wide),
            "should be novel because fresh cells are included"
        );
    }
}
