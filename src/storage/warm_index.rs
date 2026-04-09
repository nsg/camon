use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    Movement,
    Object,
}

impl EventType {
    fn dir_name(self) -> &'static str {
        match self {
            EventType::Movement => "movements",
            EventType::Object => "objects",
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WarmEventEntry {
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    pub event_type: EventType,
    pub file_size: u64,
    pub object_classes: Vec<String>,
}

#[derive(Clone)]
pub struct WarmEventIndex {
    cameras: Arc<HashMap<String, RwLock<Vec<WarmEventEntry>>>>,
    data_dir: PathBuf,
}

impl WarmEventIndex {
    pub fn new(camera_ids: &[String], data_dir: PathBuf) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(Vec::new()));
        }
        Self {
            cameras: Arc::new(cameras),
            data_dir,
        }
    }

    pub fn scan(&self) {
        let start = std::time::Instant::now();
        let mut total_events = 0;
        for (camera_id, lock) in self.cameras.iter() {
            let mut entries = Vec::new();
            for event_type in &[EventType::Movement, EventType::Object] {
                let dir = self.data_dir.join(camera_id).join(event_type.dir_name());
                let read_dir = match std::fs::read_dir(&dir) {
                    Ok(rd) => rd,
                    Err(_) => continue,
                };
                for entry in read_dir.flatten() {
                    let path = entry.path();
                    let ext = path.extension().and_then(|e| e.to_str());
                    if ext != Some("ts") {
                        continue;
                    }
                    let stem = match path.file_stem().and_then(|s| s.to_str()) {
                        Some(s) => s,
                        None => continue,
                    };
                    let (start_str, dur_str) = match stem.split_once('_') {
                        Some(pair) => pair,
                        None => continue,
                    };
                    let start_pts_ns: u64 = match start_str.parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let duration_ms: u32 = match dur_str.parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    let object_classes = Self::load_classes(&path.with_extension("json"));
                    entries.push(WarmEventEntry {
                        start_pts_ns,
                        duration_ms,
                        event_type: *event_type,
                        file_size,
                        object_classes,
                    });
                }
            }
            entries.sort_by_key(|e| e.start_pts_ns);
            let count = entries.len();
            *lock.write().unwrap() = entries;
            total_events += count;
            if count > 0 {
                tracing::info!(camera = %camera_id, events = count, "scanned warm events");
            }
        }
        tracing::info!(
            total_events,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "warm index scan complete"
        );
    }

    fn load_classes(path: &std::path::Path) -> Vec<String> {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let parsed: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        parsed["classes"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn insert(&self, camera_id: &str, entry: WarmEventEntry) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write().unwrap();
            let pos = entries
                .binary_search_by_key(&entry.start_pts_ns, |e| e.start_pts_ns)
                .unwrap_or_else(|p| p);
            entries.insert(pos, entry);
        }
    }

    pub fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read().unwrap();
                let start = entries.partition_point(|e| {
                    e.start_pts_ns + (e.duration_ms as u64) * 1_000_000 < from_ns
                });
                let end = entries.partition_point(|e| e.start_pts_ns <= to_ns);
                entries[start..end].to_vec()
            }
            None => Vec::new(),
        }
    }

    pub fn find_event(&self, camera_id: &str, start_pts_ns: u64) -> Option<WarmEventEntry> {
        let lock = self.cameras.get(camera_id)?;
        let entries = lock.read().unwrap();
        entries
            .binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns)
            .ok()
            .map(|i| entries[i].clone())
    }

    pub fn resolve_file_path(&self, camera_id: &str, entry: &WarmEventEntry) -> PathBuf {
        let dir = self
            .data_dir
            .join(camera_id)
            .join(entry.event_type.dir_name());
        dir.join(format!("{}_{}.ts", entry.start_pts_ns, entry.duration_ms))
    }

    pub async fn prune(&self, movement_max_age_ns: u64, object_max_age_ns: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        for (camera_id, lock) in self.cameras.iter() {
            let expired: Vec<WarmEventEntry> = {
                let entries = lock.read().unwrap();
                entries
                    .iter()
                    .filter(|e| {
                        let max_age = match e.event_type {
                            EventType::Movement => movement_max_age_ns,
                            EventType::Object => object_max_age_ns,
                        };
                        now_ns.saturating_sub(e.start_pts_ns) > max_age
                    })
                    .cloned()
                    .collect()
            };

            if expired.is_empty() {
                continue;
            }

            let mut deleted = 0u64;
            for entry in &expired {
                let path = self.resolve_file_path(camera_id, entry);
                let thumb = path.with_extension("jpg");
                if let Err(e) = tokio::fs::remove_file(&path).await {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(
                            camera = %camera_id,
                            path = %path.display(),
                            error = %e,
                            "failed to delete expired warm event"
                        );
                    }
                } else {
                    deleted += 1;
                }
                let _ = tokio::fs::remove_file(&thumb).await;
                let _ = tokio::fs::remove_file(&path.with_extension("json")).await;
            }

            {
                let cutoff_movement = now_ns.saturating_sub(movement_max_age_ns);
                let cutoff_object = now_ns.saturating_sub(object_max_age_ns);
                let mut entries = lock.write().unwrap();
                entries.retain(|e| match e.event_type {
                    EventType::Movement => e.start_pts_ns >= cutoff_movement,
                    EventType::Object => e.start_pts_ns >= cutoff_object,
                });
            }

            if deleted > 0 {
                tracing::info!(
                    camera = %camera_id,
                    deleted,
                    "pruned expired warm events"
                );
            }
        }
    }
}
