use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_ENTRIES: usize = 50;

pub struct DebugEntry {
    pub id: u64,
    pub timestamp: u64,
    pub grid_jpeg: Vec<u8>,
    pub raw_response: String,
    pub model: String,
    pub detection_count: usize,
}

pub struct DebugSnapshot {
    pub id: u64,
    pub timestamp: u64,
    pub raw_response: String,
    pub model: String,
    pub detection_count: usize,
}

pub struct DetectionDebugStore {
    cameras: Arc<HashMap<String, RwLock<VecDeque<DebugEntry>>>>,
    next_id: Arc<AtomicU64>,
}

impl DetectionDebugStore {
    pub fn new(camera_ids: &[String]) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(VecDeque::new()));
        }
        Self {
            cameras: Arc::new(cameras),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn insert(
        &self,
        camera_id: &str,
        grid_jpeg: Vec<u8>,
        raw_response: String,
        model: String,
        detection_count: usize,
    ) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write().unwrap();
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            entries.push_back(DebugEntry {
                id,
                timestamp,
                grid_jpeg,
                raw_response,
                model,
                detection_count,
            });
            while entries.len() > MAX_ENTRIES {
                entries.pop_front();
            }
        }
    }

    pub fn list(&self, camera_id: &str) -> Vec<DebugSnapshot> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read().unwrap();
                entries
                    .iter()
                    .map(|e| DebugSnapshot {
                        id: e.id,
                        timestamp: e.timestamp,
                        raw_response: e.raw_response.clone(),
                        model: e.model.clone(),
                        detection_count: e.detection_count,
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn get_grid_jpeg(&self, camera_id: &str, id: u64) -> Option<Vec<u8>> {
        self.cameras.get(camera_id).and_then(|lock| {
            let entries = lock.read().unwrap();
            entries
                .iter()
                .find(|e| e.id == id)
                .map(|e| e.grid_jpeg.clone())
        })
    }
}

impl Clone for DetectionDebugStore {
    fn clone(&self) -> Self {
        Self {
            cameras: Arc::clone(&self.cameras),
            next_id: Arc::clone(&self.next_id),
        }
    }
}
