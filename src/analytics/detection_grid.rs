use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

const GRID_COLS: usize = 16;
const GRID_ROWS: usize = 12;
const GRID_SIZE: usize = GRID_COLS * GRID_ROWS;

/// How many detection cycles a cell needs to be "seen" before it's absorbed.
const ABSORPTION_THRESHOLD: f32 = 0.6;
/// Decay per analysis cycle for cells with no detection.
const DECAY_RATE: f32 = 0.005;
/// Increment per detection hit.
const HIT_INCREMENT: f32 = 0.03;

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

#[derive(Clone)]
pub struct DetectionGrid {
    cameras: Arc<HashMap<String, RwLock<CameraGrid>>>,
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
            cameras.insert(id.clone(), RwLock::new(grid));
        }
        Self {
            cameras: Arc::new(cameras),
            data_dir,
        }
    }

    /// Record a detection at normalized coordinates (0.0-1.0).
    /// Returns true if this detection is novel (not yet absorbed).
    pub fn record(&self, camera_id: &str, class: &str, cx: f32, cy: f32) -> bool {
        let lock = match self.cameras.get(camera_id) {
            Some(l) => l,
            None => return true,
        };

        let col = ((cx * GRID_COLS as f32) as usize).min(GRID_COLS - 1);
        let row = ((cy * GRID_ROWS as f32) as usize).min(GRID_ROWS - 1);
        let idx = row * GRID_COLS + col;

        let mut grid = lock.write().unwrap();
        let class_grid = grid
            .classes
            .entry(class.to_string())
            .or_insert_with(ClassGrid::new);

        class_grid.cells[idx] = (class_grid.cells[idx] + HIT_INCREMENT).min(1.0);

        class_grid.cells[idx] < ABSORPTION_THRESHOLD
    }

    /// Decay all cells. Call once per analysis cycle.
    pub fn decay(&self, camera_id: &str) {
        let lock = match self.cameras.get(camera_id) {
            Some(l) => l,
            None => return,
        };

        let mut grid = lock.write().unwrap();
        for class_grid in grid.classes.values_mut() {
            for cell in &mut class_grid.cells {
                *cell = (*cell - DECAY_RATE).max(0.0);
            }
        }
    }

    /// Get grid state for the API.
    pub fn get_grid(&self, camera_id: &str) -> Option<GridResponse> {
        let lock = self.cameras.get(camera_id)?;
        let grid = lock.read().unwrap();
        let classes: HashMap<String, Vec<f32>> = grid
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
        let grid = lock.read().unwrap();
        let path = self.grid_path(camera_id);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string(&*grid) {
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

    #[test]
    fn record_returns_novel_until_absorbed() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Each hit adds 0.03. Threshold is 0.6.
        // With f32 precision, 20 * 0.03 lands just under 0.6,
        // so it takes 21 hits to cross the threshold.
        for i in 0..20 {
            let novel = grid.record("cam1", "car", 0.5, 0.5);
            assert!(novel, "should be novel at hit {i}");
        }

        // 21st hit crosses 0.6 in f32
        let novel = grid.record("cam1", "car", 0.5, 0.5);
        assert!(!novel, "should be absorbed after 21 hits");
    }

    #[test]
    fn different_classes_are_independent() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Fill up "car" at a cell
        for _ in 0..20 {
            grid.record("cam1", "car", 0.5, 0.5);
        }

        // "person" at the same cell should still be novel
        let novel = grid.record("cam1", "person", 0.5, 0.5);
        assert!(novel, "different class should be independent");
    }

    #[test]
    fn different_cells_are_independent() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Fill up one cell
        for _ in 0..20 {
            grid.record("cam1", "car", 0.1, 0.1);
        }

        // Different cell should still be novel
        let novel = grid.record("cam1", "car", 0.9, 0.9);
        assert!(novel, "different cell should be independent");
    }

    #[test]
    fn decay_reduces_values() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Record 10 hits = 0.30
        for _ in 0..10 {
            grid.record("cam1", "car", 0.5, 0.5);
        }

        // Decay 60 times: 60 * 0.005 = 0.30, should bring it back to ~0
        for _ in 0..60 {
            grid.decay("cam1");
        }

        // Should be novel again
        let novel = grid.record("cam1", "car", 0.5, 0.5);
        assert!(novel, "should be novel after full decay");
    }

    #[test]
    fn decay_does_not_go_below_zero() {
        let (grid, _dir) = make_grid(&["cam1"]);

        grid.record("cam1", "car", 0.5, 0.5);

        // Decay way more than needed
        for _ in 0..1000 {
            grid.decay("cam1");
        }

        let response = grid.get_grid("cam1").unwrap();
        // All cells should be 0 or empty
        for (_, cells) in &response.classes {
            for &v in cells {
                assert!(v >= 0.0, "cell value should not be negative");
            }
        }
    }

    #[test]
    fn value_capped_at_one() {
        let (grid, _dir) = make_grid(&["cam1"]);

        // Hit 100 times: 100 * 0.03 = 3.0, but should cap at 1.0
        for _ in 0..100 {
            grid.record("cam1", "car", 0.5, 0.5);
        }

        let response = grid.get_grid("cam1").unwrap();
        let cells = &response.classes["car"];
        let col = (0.5 * GRID_COLS as f32) as usize;
        let row = (0.5 * GRID_ROWS as f32) as usize;
        let idx = row * GRID_COLS + col;
        assert!(cells[idx] <= 1.0, "value should be capped at 1.0");
        assert!(cells[idx] >= 0.99, "value should be near 1.0");
    }

    #[test]
    fn unknown_camera_returns_novel() {
        let (grid, _dir) = make_grid(&["cam1"]);
        let novel = grid.record("unknown", "car", 0.5, 0.5);
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

        grid.record("cam1", "car", 0.5, 0.5);
        let response = grid.get_grid("cam1").unwrap();
        assert!(response.classes.contains_key("car"));

        // Decay until zero
        for _ in 0..200 {
            grid.decay("cam1");
        }

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

        // Top-left corner
        grid.record("cam1", "car", 0.0, 0.0);
        // Bottom-right corner
        grid.record("cam1", "car", 1.0, 1.0);

        let response = grid.get_grid("cam1").unwrap();
        let cells = &response.classes["car"];

        // (0,0) -> cell index 0
        assert!(cells[0] > 0.0, "top-left should be recorded");

        // (1.0, 1.0) -> clamped to (15, 11) -> index 11*16+15 = 191
        let last_idx = (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1);
        assert!(cells[last_idx] > 0.0, "bottom-right should be recorded");
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let ids = vec!["cam1".to_string()];

        // Create grid and record some data
        let grid = DetectionGrid::new(&ids, dir.path().to_path_buf());
        for _ in 0..15 {
            grid.record("cam1", "car", 0.3, 0.7);
        }
        for _ in 0..5 {
            grid.record("cam1", "person", 0.8, 0.2);
        }
        grid.save("cam1");

        // Load into a new grid instance
        let grid2 = DetectionGrid::new(&ids, dir.path().to_path_buf());
        let response = grid2.get_grid("cam1").unwrap();

        assert!(response.classes.contains_key("car"));
        assert!(response.classes.contains_key("person"));

        let col = (0.3 * GRID_COLS as f32) as usize;
        let row = (0.7 * GRID_ROWS as f32) as usize;
        let idx = row * GRID_COLS + col;
        let car_val = response.classes["car"][idx];
        assert!(
            (car_val - 0.45).abs() < 0.01,
            "car value should be ~0.45 (15 * 0.03), got {car_val}"
        );
    }
}
