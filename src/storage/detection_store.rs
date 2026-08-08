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

    /// Drop every row whose segment has aged out of the hot buffer.
    pub fn cleanup(&self, camera_id: &str, min_sequence: u64) {
        if let Some(lock) = self.cameras.get(camera_id) {
            lock.write_recover()
                .retain(|entry| entry.segment_sequence >= min_sequence);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> DetectionStore {
        DetectionStore::new(&["cam".to_string()])
    }

    fn row(store: &DetectionStore, segment_sequence: u64) -> DetectionEntry {
        DetectionEntry {
            id: store.next_id(),
            segment_sequence,
            object_class: "person".to_string(),
            confidence: 0.9,
            frame_jpeg: Arc::new(Vec::new()),
            backend: "ollama".to_string(),
            model: "test".to_string(),
        }
    }

    fn sequences(store: &DetectionStore) -> Vec<u64> {
        store
            .get_detections("cam")
            .iter()
            .map(|d| d.segment_sequence)
            .collect()
    }

    #[test]
    fn cleanup_drops_an_aged_row_stranded_behind_a_newer_one() {
        let store = store();
        store.insert("cam", row(&store, 20));
        store.insert("cam", row(&store, 5));
        store.cleanup("cam", 10);
        assert_eq!(sequences(&store), vec![20]);
        assert!(store.get_detection_info("cam", 5).is_empty());
    }

    #[test]
    fn cleanup_empties_a_deque_whose_out_of_order_rows_all_aged_out() {
        let store = store();
        store.insert("cam", row(&store, 20));
        store.insert("cam", row(&store, 5));
        store.insert("cam", row(&store, 12));
        store.cleanup("cam", 30);
        assert!(sequences(&store).is_empty());
    }

    #[test]
    fn cleanup_keeps_in_window_rows_wherever_they_sit() {
        let store = store();
        store.insert("cam", row(&store, 3));
        store.insert("cam", row(&store, 10));
        store.insert("cam", row(&store, 40));
        store.insert("cam", row(&store, 7));
        store.insert("cam", row(&store, 25));
        store.cleanup("cam", 10);
        assert_eq!(sequences(&store), vec![10, 40, 25]);
    }
}
