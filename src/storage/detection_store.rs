use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub struct DetectionEntry {
    pub id: u64,
    pub segment_sequence: u64,
    pub object_class: String,
    pub confidence: f32,
    pub frame_jpeg: Vec<u8>,
}

pub struct DetectionSnapshot {
    pub id: u64,
    pub segment_sequence: u64,
    pub object_class: String,
    pub confidence: f32,
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

    pub fn insert(
        &self,
        camera_id: &str,
        segment_sequence: u64,
        object_class: String,
        confidence: f32,
        frame_jpeg: Vec<u8>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Some(lock) = self.cameras.get(camera_id) {
            lock.write().unwrap().push_back(DetectionEntry {
                id,
                segment_sequence,
                object_class,
                confidence,
                frame_jpeg,
            });
        }
        id
    }

    pub fn get_detections(&self, camera_id: &str) -> Vec<DetectionSnapshot> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read().unwrap();
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

    pub fn get_frame(&self, camera_id: &str, detection_id: u64) -> Option<Vec<u8>> {
        self.cameras.get(camera_id).and_then(|lock| {
            let entries = lock.read().unwrap();
            entries
                .iter()
                .find(|e| e.id == detection_id)
                .map(|e| e.frame_jpeg.clone())
        })
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
}

impl Clone for DetectionStore {
    fn clone(&self) -> Self {
        Self {
            cameras: Arc::clone(&self.cameras),
            next_id: Arc::clone(&self.next_id),
        }
    }
}
