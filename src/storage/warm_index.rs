use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;

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
pub struct DetectionDetail {
    pub class: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WarmEventEntry {
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    pub event_type: EventType,
    pub file_size: u64,
    pub object_classes: Vec<String>,
    pub backend: Option<String>,
    pub model: Option<String>,
    pub detections: Vec<DetectionDetail>,
    /// TODO(2026-04-26): remove — all events now generate filmstrips.
    /// Only needed for events saved before this change.
    pub has_filmstrip: bool,
    /// True when this event is a follow-on chunk of a longer motion run split
    /// at the duration cap (from the sidecar `"continues"` flag).
    pub continues: bool,
}

#[derive(Clone)]
pub struct WarmEventIndex {
    cameras: Arc<HashMap<String, RwLock<Vec<WarmEventEntry>>>>,
    data_dir: PathBuf,
}

struct SidecarData {
    classes: Vec<String>,
    backend: Option<String>,
    model: Option<String>,
    detections: Vec<DetectionDetail>,
    continues: bool,
}

fn parse_event_filename(stem: &str) -> Option<(u64, u32)> {
    let (start_str, dur_str) = stem.split_once('_')?;
    let start_pts_ns: u64 = start_str.parse().ok()?;
    let duration_ms: u32 = dur_str.parse().ok()?;
    Some((start_pts_ns, duration_ms))
}

fn parse_sidecar_json(parsed: &serde_json::Value) -> SidecarData {
    let backend = parsed["backend"].as_str().map(String::from);
    let model = parsed["model"].as_str().map(String::from);
    // Present only on follow-on chunks; absent (→ false) on every other sidecar.
    let continues = parsed["continues"].as_bool().unwrap_or(false);

    // New format: {"backend": ..., "detections": [{class, confidence}]}
    if let Some(dets) = parsed["detections"].as_array() {
        let detections: Vec<DetectionDetail> = dets
            .iter()
            .filter_map(|d| {
                Some(DetectionDetail {
                    class: d["class"].as_str()?.to_string(),
                    confidence: d["confidence"].as_f64()? as f32,
                })
            })
            .collect();
        let classes = detections.iter().map(|d| d.class.clone()).collect();
        return SidecarData {
            classes,
            backend,
            model,
            detections,
            continues,
        };
    }

    // Old format: {"classes": ["person", "car"]}
    let classes = parsed["classes"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    SidecarData {
        classes,
        backend,
        model,
        detections: Vec::new(),
        continues,
    }
}

fn load_sidecar(path: &std::path::Path) -> SidecarData {
    let empty = SidecarData {
        classes: Vec::new(),
        backend: None,
        model: None,
        detections: Vec::new(),
        continues: false,
    };
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return empty,
    };
    match serde_json::from_str(&data) {
        Ok(parsed) => parse_sidecar_json(&parsed),
        Err(_) => empty,
    }
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
            let entries = self.scan_camera(camera_id);
            let count = entries.len();
            *lock.write_recover() = entries;
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

    fn scan_camera(&self, camera_id: &str) -> Vec<WarmEventEntry> {
        let mut entries = Vec::new();
        for event_type in &[EventType::Movement, EventType::Object] {
            let dir = self.data_dir.join(camera_id).join(event_type.dir_name());
            let read_dir = match std::fs::read_dir(&dir) {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            for entry in read_dir.flatten() {
                if let Some(warm_entry) = self.scan_entry(&entry, *event_type) {
                    entries.push(warm_entry);
                }
            }
        }
        entries.sort_by_key(|e| e.start_pts_ns);
        entries
    }

    fn scan_entry(
        &self,
        entry: &std::fs::DirEntry,
        event_type: EventType,
    ) -> Option<WarmEventEntry> {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ts") {
            return None;
        }
        let stem = path.file_stem()?.to_str()?;
        let (start_pts_ns, duration_ms) = parse_event_filename(stem)?;
        let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let sidecar = load_sidecar(&path.with_extension("json"));
        let has_filmstrip = path
            .with_file_name(format!("{}_thumb_0.jpg", stem))
            .exists();

        Some(WarmEventEntry {
            start_pts_ns,
            duration_ms,
            event_type,
            file_size,
            object_classes: sidecar.classes,
            backend: sidecar.backend,
            model: sidecar.model,
            detections: sidecar.detections,
            has_filmstrip,
            continues: sidecar.continues,
        })
    }

    pub fn insert(&self, camera_id: &str, entry: WarmEventEntry) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut entries = lock.write_recover();
            let pos = entries
                .binary_search_by_key(&entry.start_pts_ns, |e| e.start_pts_ns)
                .unwrap_or_else(|p| p);
            entries.insert(pos, entry);
        }
    }

    pub fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
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
        let entries = lock.read_recover();
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
                let entries = lock.read_recover();
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
                // Clean up filmstrip thumbnails
                let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
                let dir = path.parent().unwrap_or(&self.data_dir);
                for i in 0..4 {
                    let _ =
                        tokio::fs::remove_file(dir.join(format!("{}_thumb_{}.jpg", stem, i))).await;
                }
            }

            {
                let cutoff_movement = now_ns.saturating_sub(movement_max_age_ns);
                let cutoff_object = now_ns.saturating_sub(object_max_age_ns);
                let mut entries = lock.write_recover();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_event_files(dir: &std::path::Path, subdir: &str, stem: &str, sidecar: Option<&str>) {
        let d = dir.join("cam").join(subdir);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(format!("{stem}.ts")), b"tsdata").unwrap();
        if let Some(json) = sidecar {
            std::fs::write(d.join(format!("{stem}.json")), json).unwrap();
        }
    }

    fn entries(index: &WarmEventIndex) -> Vec<WarmEventEntry> {
        index.query("cam", 0, u64::MAX)
    }

    #[test]
    fn scan_round_trips_continues_and_keeps_type_from_directory() {
        let dir = tempfile::tempdir().unwrap();

        // A movement-only follow-on chunk: minimal sidecar with just continues.
        write_event_files(
            dir.path(),
            "movements",
            "1000_5000",
            Some(r#"{"detections":[],"continues":true}"#),
        );
        // A plain movement first chunk: no sidecar at all.
        write_event_files(dir.path(), "movements", "2000_5000", None);
        // An object follow-on chunk: detections plus continues.
        write_event_files(
            dir.path(),
            "objects",
            "3000_5000",
            Some(
                r#"{"backend":"ollama","model":"m","detections":[{"class":"person","confidence":0.9}],"continues":true}"#,
            ),
        );

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        let events = entries(&index);
        assert_eq!(events.len(), 3);

        let movement_chunk = index.find_event("cam", 1000).unwrap();
        // Type comes from the directory, not the sidecar presence.
        assert_eq!(movement_chunk.event_type, EventType::Movement);
        assert!(movement_chunk.continues);
        assert!(movement_chunk.object_classes.is_empty());

        let plain = index.find_event("cam", 2000).unwrap();
        assert_eq!(plain.event_type, EventType::Movement);
        assert!(!plain.continues);

        let object_chunk = index.find_event("cam", 3000).unwrap();
        assert_eq!(object_chunk.event_type, EventType::Object);
        assert!(object_chunk.continues);
        assert_eq!(object_chunk.detections.len(), 1);
        assert_eq!(object_chunk.backend.as_deref(), Some("ollama"));
    }

    #[test]
    fn scan_defaults_continues_false_for_legacy_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        // Legacy object sidecar without a continues field.
        write_event_files(
            dir.path(),
            "objects",
            "1000_5000",
            Some(r#"{"classes":["car"]}"#),
        );
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        let e = index.find_event("cam", 1000).unwrap();
        assert!(!e.continues);
        assert_eq!(e.object_classes, vec!["car".to_string()]);
    }
}
