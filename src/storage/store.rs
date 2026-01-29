use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

pub struct MotionEntry {
    pub segment_sequence: u64,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub motion_score: f32,
    pub mask_jpeg: Option<Vec<u8>>,
}

pub struct MotionStore {
    cameras: Arc<HashMap<String, RwLock<VecDeque<MotionEntry>>>>,
}

impl MotionStore {
    pub fn new(camera_ids: &[String]) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(VecDeque::new()));
        }
        Self {
            cameras: Arc::new(cameras),
        }
    }

    pub fn insert(&self, camera_id: &str, entry: MotionEntry) {
        if let Some(lock) = self.cameras.get(camera_id) {
            lock.write().unwrap().push_back(entry);
        }
    }

    pub fn get_motion(&self, camera_id: &str) -> Vec<MotionSnapshot> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read().unwrap();
                entries
                    .iter()
                    .map(|e| MotionSnapshot {
                        segment_sequence: e.segment_sequence,
                        duration_ns: e.end_time_ns - e.start_time_ns,
                        motion_score: e.motion_score,
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn get_mask(&self, camera_id: &str, segment_sequence: u64) -> Option<Vec<u8>> {
        let lock = self.cameras.get(camera_id)?;
        let entries = lock.read().unwrap();
        entries
            .iter()
            .find(|e| e.segment_sequence == segment_sequence)
            .and_then(|e| e.mask_jpeg.clone())
    }

    pub fn cleanup(&self, camera_id: &str, min_sequence: u64) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write().unwrap();
            while let Some(front) = entries.front() {
                if front.segment_sequence < min_sequence {
                    entries.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    pub fn has_motion(&self, camera_id: &str, segment_sequence: u64) -> bool {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read().unwrap();
                entries
                    .iter()
                    .any(|e| e.segment_sequence == segment_sequence && e.motion_score > 0.0)
            }
            None => false,
        }
    }

    pub fn last_sequence(&self, camera_id: &str) -> Option<u64> {
        self.cameras
            .get(camera_id)?
            .read()
            .unwrap()
            .back()
            .map(|e| e.segment_sequence)
    }
}

impl Clone for MotionStore {
    fn clone(&self) -> Self {
        Self {
            cameras: Arc::clone(&self.cameras),
        }
    }
}

pub struct MotionSnapshot {
    pub segment_sequence: u64,
    pub duration_ns: u64,
    pub motion_score: f32,
}
