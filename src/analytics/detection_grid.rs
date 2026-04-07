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
