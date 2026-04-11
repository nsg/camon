use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_ENTRIES: usize = 50;

pub struct DebugEntry {
    pub id: u64,
    pub timestamp: u64,
    pub frame_jpegs: Vec<Vec<u8>>,
    pub raw_responses: Vec<String>,
    pub model: String,
    pub detection_count: usize,
}

pub struct DebugSnapshot {
    pub id: u64,
    pub timestamp: u64,
    pub raw_responses: Vec<String>,
    pub model: String,
    pub detection_count: usize,
    pub frame_count: usize,
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
        frame_jpegs: Vec<Vec<u8>>,
        raw_responses: Vec<String>,
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
                frame_jpegs,
                raw_responses,
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
                        raw_responses: e.raw_responses.clone(),
                        model: e.model.clone(),
                        detection_count: e.detection_count,
                        frame_count: e.frame_jpegs.len(),
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn get_frame_jpeg(&self, camera_id: &str, id: u64, frame_index: usize) -> Option<Vec<u8>> {
        self.cameras.get(camera_id).and_then(|lock| {
            let entries = lock.read().unwrap();
            entries
                .iter()
                .find(|e| e.id == id)
                .and_then(|e| e.frame_jpegs.get(frame_index).cloned())
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
