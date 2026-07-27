use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    Movement,
    Object,
    /// A chunk of gapless continuous recording (analytics disabled — "dumb NVR"
    /// mode). Not motion-gated; every segment reaches disk.
    Continuous,
}

impl EventType {
    pub(crate) fn dir_name(self) -> &'static str {
        match self {
            EventType::Movement => "movements",
            EventType::Object => "objects",
            EventType::Continuous => "continuous",
        }
    }

    /// Wire name used by the remote (stathost) backend's sidecar, which carries
    /// the event type in JSON rather than in a directory name.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            EventType::Movement => "movement",
            EventType::Object => "object",
            EventType::Continuous => "continuous",
        }
    }

    /// Parse a wire name back into an [`EventType`]; unknown strings yield `None`.
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "movement" => Some(EventType::Movement),
            "object" => Some(EventType::Object),
            "continuous" => Some(EventType::Continuous),
            _ => None,
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
    /// Number of filmstrip thumbnail frames on disk for this event
    /// (`{stem}_thumb_{0..n-1}.jpg`). A motion run is subsampled to at most 4
    /// frames, and short runs yield fewer — the UI renders exactly this many.
    pub filmstrip_frames: usize,
    /// True when this event is a follow-on chunk of a longer motion run split
    /// at the duration cap (from the sidecar `"continues"` flag).
    pub continues: bool,
    /// True when this event was salvaged from an orphaned `.ts.tmp` at startup
    /// after a crash or power cut (from the sidecar `"recovered"` flag). The
    /// tail may be truncated at the last intact packet.
    pub recovered: bool,
}

#[derive(Clone)]
pub struct WarmEventIndex {
    cameras: Arc<HashMap<String, RwLock<Vec<WarmEventEntry>>>>,
    data_dir: PathBuf,
}

pub(crate) struct SidecarData {
    pub(crate) classes: Vec<String>,
    pub(crate) backend: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) detections: Vec<DetectionDetail>,
    pub(crate) continues: bool,
    pub(crate) recovered: bool,
    /// Event type carried in the sidecar. Local mode leaves this `None` and
    /// derives the type from the on-disk directory; the remote (stathost)
    /// backend, which has no directories, reads it from here.
    pub(crate) event_type: Option<EventType>,
}

pub(crate) fn parse_event_filename(stem: &str) -> Option<(u64, u32)> {
    let (start_str, dur_str) = stem.split_once('_')?;
    let start_pts_ns: u64 = start_str.parse().ok()?;
    let duration_ms: u32 = dur_str.parse().ok()?;
    Some((start_pts_ns, duration_ms))
}

pub(crate) fn parse_sidecar_json(parsed: &serde_json::Value) -> SidecarData {
    let backend = parsed["backend"].as_str().map(String::from);
    let model = parsed["model"].as_str().map(String::from);
    // Present only on follow-on chunks; absent (→ false) on every other sidecar.
    let continues = parsed["continues"].as_bool().unwrap_or(false);
    // Present only on events salvaged by startup orphan recovery.
    let recovered = parsed["recovered"].as_bool().unwrap_or(false);
    // Present only on remote (stathost) sidecars, which have no directory to
    // carry the type; unknown/absent → None (local mode ignores it entirely).
    let event_type = parsed["event_type"].as_str().and_then(EventType::from_str);

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
            recovered,
            event_type,
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
        recovered,
        event_type,
    }
}

fn load_sidecar(path: &std::path::Path) -> SidecarData {
    let empty = SidecarData {
        classes: Vec::new(),
        backend: None,
        model: None,
        detections: Vec::new(),
        continues: false,
        recovered: false,
        event_type: None,
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
        for event_type in &[
            EventType::Movement,
            EventType::Object,
            EventType::Continuous,
        ] {
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
        // Filmstrip frames are numbered contiguously from 0; count until the
        // first gap. The pipeline writes at most 4.
        let mut filmstrip_frames = 0;
        while path
            .with_file_name(format!("{}_thumb_{}.jpg", stem, filmstrip_frames))
            .exists()
        {
            filmstrip_frames += 1;
        }

        Some(WarmEventEntry {
            start_pts_ns,
            duration_ms,
            event_type,
            file_size,
            object_classes: sidecar.classes,
            backend: sidecar.backend,
            model: sidecar.model,
            detections: sidecar.detections,
            filmstrip_frames,
            continues: sidecar.continues,
            recovered: sidecar.recovered,
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

    /// Every event overlapping `[from_ns, to_ns]`. An inverted range is empty.
    ///
    /// Entries are ordered by start PTS only, so the upper bound binary-searches
    /// but the lower one cannot: a long event (a continuous chunk) can start
    /// far before the window and still reach into it, and "ends after `from_ns`"
    /// is not monotone in start order. The candidate prefix is filtered instead.
    pub fn query(&self, camera_id: &str, from_ns: u64, to_ns: u64) -> Vec<WarmEventEntry> {
        if from_ns > to_ns {
            return Vec::new();
        }
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
                let end = entries.partition_point(|e| e.start_pts_ns <= to_ns);
                entries[..end]
                    .iter()
                    .filter(|e| {
                        e.start_pts_ns
                            .saturating_add((e.duration_ms as u64) * 1_000_000)
                            >= from_ns
                    })
                    .cloned()
                    .collect()
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

    /// Mutate the entry with the given start PTS in place (used by the
    /// post-hoc movement→object upgrade; the sort key never changes).
    /// Returns false when no such event is indexed.
    pub fn update_event(
        &self,
        camera_id: &str,
        start_pts_ns: u64,
        f: impl FnOnce(&mut WarmEventEntry),
    ) -> bool {
        let Some(lock) = self.cameras.get(camera_id) else {
            return false;
        };
        let mut entries = lock.write_recover();
        match entries.binary_search_by_key(&start_pts_ns, |e| e.start_pts_ns) {
            Ok(i) => {
                f(&mut entries[i]);
                true
            }
            Err(_) => false,
        }
    }

    pub fn resolve_file_path(&self, camera_id: &str, entry: &WarmEventEntry) -> PathBuf {
        let dir = self
            .data_dir
            .join(camera_id)
            .join(entry.event_type.dir_name());
        dir.join(format!("{}_{}.ts", entry.start_pts_ns, entry.duration_ms))
    }

    pub async fn prune(
        &self,
        movement_max_age_ns: u64,
        object_max_age_ns: u64,
        continuous_max_age_ns: u64,
    ) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let max_age = |t: EventType| match t {
            EventType::Movement => movement_max_age_ns,
            EventType::Object => object_max_age_ns,
            EventType::Continuous => continuous_max_age_ns,
        };

        for (camera_id, lock) in self.cameras.iter() {
            let expired: Vec<WarmEventEntry> = {
                let entries = lock.read_recover();
                entries
                    .iter()
                    .filter(|e| now_ns.saturating_sub(e.start_pts_ns) > max_age(e.event_type))
                    .cloned()
                    .collect()
            };

            if expired.is_empty() {
                continue;
            }

            let mut deleted = 0u64;
            for entry in &expired {
                if self.remove_event_files(camera_id, entry).await {
                    deleted += 1;
                }
            }

            {
                let mut entries = lock.write_recover();
                entries.retain(|e| e.start_pts_ns >= now_ns.saturating_sub(max_age(e.event_type)));
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

    /// Delete every file belonging to one event (.ts, sidecar, thumbnails).
    /// Returns true if the video file itself was actually deleted.
    async fn remove_event_files(&self, camera_id: &str, entry: &WarmEventEntry) -> bool {
        let path = self.resolve_file_path(camera_id, entry);
        let thumb = path.with_extension("jpg");
        let removed = match tokio::fs::remove_file(&path).await {
            Ok(()) => true,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        camera = %camera_id,
                        path = %path.display(),
                        error = %e,
                        "failed to delete warm event file"
                    );
                }
                false
            }
        };
        let _ = tokio::fs::remove_file(&thumb).await;
        let _ = tokio::fs::remove_file(&path.with_extension("json")).await;
        // Clean up filmstrip thumbnails
        let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
        let dir = path.parent().unwrap_or(&self.data_dir);
        for i in 0..4 {
            let _ = tokio::fs::remove_file(dir.join(format!("{}_thumb_{}.jpg", stem, i))).await;
        }
        removed
    }

    /// Emergency prune for low-disk-space conditions: delete the oldest events
    /// first, cheapest-to-lose tier first (continuous → movements → objects),
    /// until `satisfied()` reports the pressure is gone (in production: free
    /// space back above `min_free_bytes`) or nothing is left to delete.
    ///
    /// Returns the number of events deleted.
    pub async fn emergency_prune<F: FnMut() -> bool>(&self, mut satisfied: F) -> u64 {
        let mut deleted = 0u64;
        for tier in [
            EventType::Continuous,
            EventType::Movement,
            EventType::Object,
        ] {
            // Snapshot this tier's candidates across all cameras, oldest first.
            let mut candidates: Vec<(String, WarmEventEntry)> = Vec::new();
            for (camera_id, lock) in self.cameras.iter() {
                let entries = lock.read_recover();
                candidates.extend(
                    entries
                        .iter()
                        .filter(|e| e.event_type == tier)
                        .cloned()
                        .map(|e| (camera_id.clone(), e)),
                );
            }
            candidates.sort_by_key(|(_, e)| e.start_pts_ns);

            for (camera_id, entry) in candidates {
                if satisfied() {
                    return deleted;
                }
                self.remove_event_files(&camera_id, &entry).await;
                if let Some(lock) = self.cameras.get(&camera_id) {
                    lock.write_recover().retain(|e| {
                        !(e.start_pts_ns == entry.start_pts_ns && e.event_type == entry.event_type)
                    });
                }
                deleted += 1;
                tracing::warn!(
                    camera = %camera_id,
                    start_pts_ns = entry.start_pts_ns,
                    event_type = ?entry.event_type,
                    "emergency prune: deleted event to reclaim disk space"
                );
            }
        }
        deleted
    }
}

/// Free bytes available to unprivileged writes on the filesystem holding
/// `path` (statvfs `f_bavail * f_frsize`). Small wrapper so the low-space
/// guard's threshold logic stays testable without touching a real disk.
pub(crate) fn free_space_bytes(path: &std::path::Path) -> std::io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut st) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(st.f_bavail as u64 * st.f_frsize as u64)
}

/// Threshold decision for the low-space guard. `min_free_bytes == 0` disables
/// the guard entirely.
pub(crate) fn should_emergency_prune(free_bytes: u64, min_free_bytes: u64) -> bool {
    min_free_bytes > 0 && free_bytes < min_free_bytes
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

    const SEC: u64 = 1_000_000_000;

    fn indexed(spans: &[(u64, u32)]) -> WarmEventIndex {
        let index = WarmEventIndex::new(&["cam".to_string()], PathBuf::from("/nonexistent"));
        for &(start_pts_ns, duration_ms) in spans {
            index.insert(
                "cam",
                WarmEventEntry {
                    start_pts_ns,
                    duration_ms,
                    event_type: EventType::Continuous,
                    file_size: 0,
                    object_classes: Vec::new(),
                    backend: None,
                    model: None,
                    detections: Vec::new(),
                    filmstrip_frames: 0,
                    continues: false,
                    recovered: false,
                },
            );
        }
        index
    }

    #[test]
    fn query_returns_long_events_that_started_before_the_window() {
        // A 100s chunk starting at 0, then two 1s events that end long before
        // the window: sorted by start, "ends before from" is false-then-true,
        // so a binary search on it skips right past the chunk that does overlap.
        let index = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let hits = index.query("cam", 50 * SEC, 60 * SEC);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_pts_ns, 0);
    }

    #[test]
    fn query_returns_every_overlapping_event_in_start_order() {
        let index = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let starts: Vec<u64> = index
            .query("cam", 0, u64::MAX)
            .iter()
            .map(|e| e.start_pts_ns)
            .collect();
        assert_eq!(starts, vec![0, 10 * SEC, 20 * SEC]);
        assert!(index.query("unknown", 0, u64::MAX).is_empty());
    }

    #[test]
    fn zero_duration_events_are_found_at_their_start() {
        let index = indexed(&[(10 * SEC, 0)]);
        assert_eq!(index.query("cam", 10 * SEC, 10 * SEC).len(), 1);
        assert!(index.query("cam", 10 * SEC + 1, 20 * SEC).is_empty());
    }

    #[test]
    fn query_bounds_include_events_that_only_touch_them() {
        let index = indexed(&[(10 * SEC, 5_000)]);
        // Ends exactly at from_ns.
        assert_eq!(index.query("cam", 15 * SEC, 20 * SEC).len(), 1);
        assert!(index.query("cam", 15 * SEC + 1, 20 * SEC).is_empty());
        // Starts exactly at to_ns.
        assert_eq!(index.query("cam", 0, 10 * SEC).len(), 1);
        assert!(index.query("cam", 0, 10 * SEC - 1).is_empty());
    }

    #[test]
    fn query_with_an_inverted_range_is_empty() {
        // These bounds used to be computed independently and sliced, which
        // panicked here with start > end.
        let index = indexed(&[(0, 100_000), (10 * SEC, 1_000)]);
        assert!(index.query("cam", u64::MAX, 0).is_empty());
        assert!(index.query("cam", 20 * SEC, 5 * SEC).is_empty());
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
    fn scan_picks_up_continuous_chunks_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        // First continuous chunk: no sidecar. Follow-on: continues sidecar.
        write_event_files(dir.path(), "continuous", "1000_5000", None);
        write_event_files(
            dir.path(),
            "continuous",
            "2000_5000",
            Some(r#"{"detections":[],"continues":true}"#),
        );

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let first = index.find_event("cam", 1000).unwrap();
        assert_eq!(first.event_type, EventType::Continuous);
        assert!(!first.continues);
        let follow = index.find_event("cam", 2000).unwrap();
        assert_eq!(follow.event_type, EventType::Continuous);
        assert!(follow.continues);
        // Continuous chunks resolve back into continuous/.
        assert_eq!(
            index.resolve_file_path("cam", &follow),
            dir.path()
                .join("cam")
                .join("continuous")
                .join("2000_5000.ts")
        );
    }

    #[tokio::test]
    async fn prune_honors_the_continuous_retention() {
        let dir = tempfile::tempdir().unwrap();
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let day_ns = 86_400 * 1_000_000_000u64;

        // A movement event 3 days old and a continuous chunk 2 days old, both
        // named with real wall-clock start times so prune's now-based age works.
        let movement_pts = now_ns - 3 * day_ns;
        let continuous_pts = now_ns - 2 * day_ns;
        write_event_files(
            dir.path(),
            "movements",
            &format!("{movement_pts}_5000"),
            None,
        );
        write_event_files(
            dir.path(),
            "continuous",
            &format!("{continuous_pts}_5000"),
            None,
        );

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(entries(&index).len(), 2);

        // Movement retention 7d (keep the 3d-old movement), continuous 1d (drop
        // the 2d-old chunk). Object retention irrelevant here.
        index.prune(7 * day_ns, 14 * day_ns, day_ns).await;

        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].event_type, EventType::Movement);
        // The continuous file and its in-memory entry are both gone.
        assert!(index.find_event("cam", continuous_pts).is_none());
        assert!(!dir
            .path()
            .join("cam")
            .join("continuous")
            .join(format!("{continuous_pts}_5000.ts"))
            .exists());
    }

    #[test]
    fn free_space_threshold_uses_injected_value() {
        assert!(should_emergency_prune(0, 100));
        assert!(should_emergency_prune(99, 100));
        assert!(!should_emergency_prune(100, 100));
        assert!(!should_emergency_prune(u64::MAX, 100));
        // 0 disables the guard, even with nothing free.
        assert!(!should_emergency_prune(0, 0));
    }

    #[test]
    fn free_space_bytes_reports_nonzero_for_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(free_space_bytes(dir.path()).unwrap() > 0);
        assert!(free_space_bytes(std::path::Path::new("/nonexistent-camon")).is_err());
    }

    #[tokio::test]
    async fn emergency_prune_deletes_cheapest_and_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        // Two continuous chunks (one older), one older-still movement, one
        // ancient object event. Tier order must beat age order.
        write_event_files(dir.path(), "continuous", "5000_1000", None);
        write_event_files(dir.path(), "continuous", "4000_1000", None);
        write_event_files(dir.path(), "movements", "3000_1000", None);
        write_event_files(dir.path(), "objects", "1000_1000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        // "Pressure gone" after three deletions: both continuous chunks
        // (oldest first) and the movement go; the object survives even though
        // it is the oldest file on disk.
        let mut checks = 0;
        let deleted = index
            .emergency_prune(|| {
                checks += 1;
                checks > 3
            })
            .await;
        assert_eq!(deleted, 3);

        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].event_type, EventType::Object);
        assert_eq!(remaining[0].start_pts_ns, 1000);
        // Files gone from disk too.
        assert!(!dir
            .path()
            .join("cam")
            .join("continuous")
            .join("4000_1000.ts")
            .exists());
        assert!(!dir
            .path()
            .join("cam")
            .join("movements")
            .join("3000_1000.ts")
            .exists());
        assert!(dir
            .path()
            .join("cam")
            .join("objects")
            .join("1000_1000.ts")
            .exists());
    }

    #[tokio::test]
    async fn emergency_prune_stops_when_nothing_left() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        // Never satisfied: deletes everything it has, then gives up.
        let deleted = index.emergency_prune(|| false).await;
        assert_eq!(deleted, 1);
        assert!(entries(&index).is_empty());
    }

    #[tokio::test]
    async fn emergency_prune_immediately_satisfied_deletes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(index.emergency_prune(|| true).await, 0);
        assert_eq!(entries(&index).len(), 1);
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
