use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;

pub struct DetectionEntry {
    pub id: u64,
    pub segment_sequence: u64,
    pub object_class: String,
    pub confidence: f32,
    pub frame_jpeg: Arc<Vec<u8>>,
    pub backend: String,
    pub model: String,
}

pub struct DetectionSnapshot {
    pub id: u64,
    pub segment_sequence: u64,
    pub object_class: String,
    pub confidence: f32,
}

pub struct DetectionInfo {
    pub object_class: String,
    pub confidence: f32,
    pub backend: String,
    pub model: String,
}

pub struct DetectionStore {
    cameras: Arc<HashMap<String, RwLock<VecDeque<DetectionEntry>>>>,
    next_id: Arc<AtomicU64>,
}

impl DetectionStore {
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

    pub fn insert(&self, camera_id: &str, entry: DetectionEntry) -> u64 {
        let id = entry.id;
        if let Some(lock) = self.cameras.get(camera_id) {
            lock.write_recover().push_back(entry);
        }
        id
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn get_detections(&self, camera_id: &str) -> Vec<DetectionSnapshot> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
                entries
                    .iter()
                    .map(|e| DetectionSnapshot {
                        id: e.id,
                        segment_sequence: e.segment_sequence,
                        object_class: e.object_class.clone(),
                        confidence: e.confidence,
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn get_frame(&self, camera_id: &str, detection_id: u64) -> Option<Arc<Vec<u8>>> {
        self.cameras.get(camera_id).and_then(|lock| {
            let entries = lock.read_recover();
            entries
                .iter()
                .find(|e| e.id == detection_id)
                .map(|e| Arc::clone(&e.frame_jpeg))
        })
    }

    pub fn get_detection_info(&self, camera_id: &str, segment_sequence: u64) -> Vec<DetectionInfo> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
                entries
                    .iter()
                    .filter(|e| e.segment_sequence == segment_sequence)
                    .map(|e| DetectionInfo {
                        object_class: e.object_class.clone(),
                        confidence: e.confidence,
                        backend: e.backend.clone(),
                        model: e.model.clone(),
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn cleanup(&self, camera_id: &str, min_sequence: u64) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write_recover();
            while let Some(front) = entries.front() {
                if front.segment_sequence < min_sequence {
                    entries.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

impl Clone for DetectionStore {
    fn clone(&self) -> Self {
        Self {
            cameras: Arc::clone(&self.cameras),
            next_id: Arc::clone(&self.next_id),
        }
    }
}
