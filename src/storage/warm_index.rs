//! The local-disk warm index: the filesystem half of
//! [`LocalDiskBackend`](crate::storage::LocalDiskBackend) — the directory scan, the sidecar
//! format, and the unlinks retention performs — over the shared in-RAM [`EventIndex`].

use std::path::PathBuf;

use crate::buffer::wall_clock_ns;
use crate::storage::event_index::{
    deduplicate_detections, evict_tiers, filmstrip_frame_count, sweep_expired, DetectionDetail,
    EmergencyOutcome, EventIndex, EventPage, EventRef, EventType, EvictionPolicy, Removal,
    WarmEventEntry, MAX_FILMSTRIP_FRAMES,
};

/// Local disk's event identity:
/// `{data_dir}/{camera}/{event_type}/{start_pts_ns}_{duration_ms}.ts` names exactly one file,
/// and the event type is a directory in it.
type EventKey = (u64, EventType, u32);

/// What an API request ([`EventRef`]) names here: all three fields, because all
/// three are in the path of the file it asks for.
fn event_key(event: EventRef) -> EventKey {
    (event.start_pts_ns, event.event_type, event.duration_ms)
}

pub struct WarmEventIndex {
    events: EventIndex<EventKey>,
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

/// Render a sidecar, for fresh writes and post-hoc upgrades on either backend.
pub(crate) fn sidecar_json(
    event_type: Option<EventType>,
    backend: Option<&str>,
    model: Option<&str>,
    detection_details: &[DetectionDetail],
    continues: bool,
) -> String {
    let mut meta = serde_json::Map::new();
    if let Some(event_type) = event_type {
        meta.insert(
            "event_type".to_string(),
            serde_json::json!(event_type.as_str()),
        );
    }
    if let Some(backend) = backend {
        meta.insert("backend".to_string(), serde_json::json!(backend));
    }
    if let Some(model) = model {
        meta.insert("model".to_string(), serde_json::json!(model));
    }

    let deduped = deduplicate_detections(detection_details);
    let detections: Vec<serde_json::Value> = deduped
        .iter()
        .map(|(class, confidence)| serde_json::json!({"class": class, "confidence": confidence}))
        .collect();
    meta.insert("detections".to_string(), serde_json::json!(detections));

    // Only follow-on chunks carry `continues`; omit it otherwise so ordinary
    // sidecars stay unchanged.
    if continues {
        meta.insert("continues".to_string(), serde_json::json!(true));
    }

    serde_json::to_string(&meta).unwrap()
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

/// Walk one camera's three tier directories and build an index entry for every
/// event file in them. Pure blocking filesystem work over a path — no index, no
/// `self` — so it can be handed whole to a blocking thread.
fn scan_camera_dirs(data_dir: &std::path::Path, camera_id: &str) -> Vec<WarmEventEntry> {
    let mut entries = Vec::new();
    for event_type in [
        EventType::Movement,
        EventType::Object,
        EventType::Continuous,
    ] {
        let dir = data_dir.join(camera_id).join(event_type.dir_name());
        let read_dir = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            // NotFound is the ordinary case (a tier never written). Anything
            // else is a storage fault that would otherwise scan back as an
            // empty archive, indistinguishable from a fresh install.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(
                    camera = %camera_id,
                    dir = %dir.display(),
                    error = %e,
                    "cannot read a stored event directory; its events are missing from \
                     the index and will not be served, pruned or counted"
                );
                continue;
            }
        };

        scan_dir_entries(read_dir, event_type, camera_id, &dir, &mut entries);
    }
    entries
}

/// Index every event file a directory listing names, and report the entries the listing itself
/// could not produce — dropping those silently is how an event disappears with nothing to say
/// it existed.
fn scan_dir_entries(
    listing: impl Iterator<Item = std::io::Result<std::fs::DirEntry>>,
    event_type: EventType,
    camera_id: &str,
    dir: &std::path::Path,
    entries: &mut Vec<WarmEventEntry>,
) {
    let mut unreadable = 0usize;
    let mut first_error = None;
    for entry in listing {
        match entry {
            Ok(entry) => {
                if let Some(warm_entry) = scan_entry(&entry, event_type) {
                    entries.push(warm_entry);
                }
            }
            Err(e) => {
                unreadable += 1;
                first_error.get_or_insert(e);
            }
        }
    }
    if let Some(error) = first_error {
        tracing::warn!(
            camera = %camera_id,
            dir = %dir.display(),
            unreadable,
            error = %error,
            "entries in a stored event directory could not be read; any events among them \
             are missing from the index and will not be served, pruned or counted (first \
             error shown)"
        );
    }
}

/// One directory entry's index entry, or `None` when it is not an event file.
fn scan_entry(entry: &std::fs::DirEntry, event_type: EventType) -> Option<WarmEventEntry> {
    let path = entry.path();
    if path.extension().and_then(|e| e.to_str()) != Some("ts") {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    let (start_pts_ns, duration_ms) = parse_event_filename(stem)?;
    let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
    let sidecar = load_sidecar(&path.with_extension("json"));
    let filmstrip_frames = filmstrip_frame_count(|i| {
        path.with_file_name(format!("{stem}_thumb_{i}.jpg"))
            .exists()
    });

    Some(WarmEventEntry {
        start_pts_ns,
        duration_ms,
        event_type,
        file_size,
        // Zero, deliberately: `statvfs` is this backend's accounting
        // authority — see [`crate::storage::contract`].
        sidecar_bytes: 0,
        thumbnail_bytes: 0,
        object_classes: sidecar.classes,
        backend: sidecar.backend,
        model: sidecar.model,
        detections: sidecar.detections,
        filmstrip_frames,
        continues: sidecar.continues,
        recovered: sidecar.recovered,
        delete_failed: false,
    })
}

impl WarmEventIndex {
    pub fn new(camera_ids: &[String], data_dir: PathBuf) -> Self {
        Self {
            events: EventIndex::new(camera_ids),
            data_dir,
        }
    }

    /// The same rebuild, on the calling thread — reachable only from a test, deliberately:
    /// the walk is blocking filesystem work, and production must go through
    /// [`scan_off_thread`](Self::scan_off_thread).
    #[cfg(test)]
    pub(crate) fn scan(&self) {
        let start = std::time::Instant::now();
        let mut total_events = 0;
        for camera_id in self.scan_order() {
            let entries = scan_camera_dirs(&self.data_dir, &camera_id);
            total_events += self.take_scanned_camera(&camera_id, entries);
        }
        Self::report_scan(total_events, start);
    }

    /// Rebuild the index from the directory tree, with the walk handed to a blocking thread —
    /// the only way production runs it.
    pub async fn scan_off_thread(&self) {
        self.scan_off_thread_with(scan_camera_dirs).await
    }

    /// The body of [`scan_off_thread`](Self::scan_off_thread), over an
    /// injectable walk so a test can hold the walk still and watch the runtime
    /// carry on without it.
    async fn scan_off_thread_with<W>(&self, walk: W)
    where
        W: Fn(&std::path::Path, &str) -> Vec<WarmEventEntry> + Clone + Send + 'static,
    {
        let start = std::time::Instant::now();
        let mut total_events = 0;
        for camera_id in self.scan_order() {
            let data_dir = self.data_dir.clone();
            let walking = camera_id.clone();
            let walk = walk.clone();
            let walked = tokio::task::spawn_blocking(move || walk(&data_dir, &walking)).await;
            match walked {
                Ok(entries) => total_events += self.take_scanned_camera(&camera_id, entries),
                // The walk panicked: one camera's events go missing rather
                // than the whole scan; the panic hook already reported it.
                Err(e) => tracing::warn!(
                    camera = %camera_id,
                    error = %e,
                    "the directory walk for a camera did not finish; its events are missing \
                     from the index and will not be served, pruned or counted"
                ),
            }
        }
        Self::report_scan(total_events, start);
    }

    /// The cameras to walk, as owned ids: `replace_camera` takes the write lock
    /// the index's own iterator would otherwise hold open across a whole
    /// directory walk.
    fn scan_order(&self) -> Vec<String> {
        self.events.camera_ids().map(String::from).collect()
    }

    /// Hand one camera's freshly walked events to the index, and say how many.
    fn take_scanned_camera(&self, camera_id: &str, entries: Vec<WarmEventEntry>) -> usize {
        let count = entries.len();
        self.events.replace_camera(camera_id, entries);
        if count > 0 {
            tracing::info!(camera = %camera_id, events = count, "scanned warm events");
        }
        count
    }

    fn report_scan(total_events: usize, start: std::time::Instant) {
        tracing::info!(
            total_events,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "warm index scan complete"
        );
    }

    pub fn insert(&self, camera_id: &str, entry: WarmEventEntry) {
        self.events.insert(camera_id, entry);
    }

    pub fn query(&self, camera_id: &str, page: EventPage) -> Vec<WarmEventEntry> {
        self.events.query(camera_id, page)
    }

    pub fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        self.events.newest_event_end_ns(camera_id)
    }

    pub fn find_event(&self, camera_id: &str, event: EventRef) -> Option<WarmEventEntry> {
        self.events.find(camera_id, event_key(event))
    }

    /// Apply the post-hoc movement→object upgrade to the indexed entry.
    pub fn update_event(
        &self,
        camera_id: &str,
        start_pts_ns: u64,
        duration_ms: u32,
        f: impl FnOnce(&mut WarmEventEntry),
    ) -> bool {
        let key = (start_pts_ns, EventType::Movement, duration_ms);
        self.events.reidentify(camera_id, key, f)
    }

    pub fn resolve_file_path(&self, camera_id: &str, entry: &WarmEventEntry) -> PathBuf {
        let dir = self
            .data_dir
            .join(camera_id)
            .join(entry.event_type.dir_name());
        dir.join(format!("{}_{}.ts", entry.start_pts_ns, entry.duration_ms))
    }

    /// Delete events past their per-class retention, up to this sweep's share of the camera.
    pub async fn prune<F: FnMut() -> bool>(
        &self,
        movement_max_age_ns: u64,
        object_max_age_ns: u64,
        continuous_max_age_ns: u64,
        cancel: F,
    ) -> u64 {
        self.prune_at(
            wall_clock_ns(),
            movement_max_age_ns,
            object_max_age_ns,
            continuous_max_age_ns,
            cancel,
        )
        .await
    }

    /// [`prune`](Self::prune) with the clock handed in, so a sweep can be run
    /// at an arbitrary "now" — including one on the far side of a clock jump —
    /// without waiting for the hour or touching the system clock.
    async fn prune_at<F: FnMut() -> bool>(
        &self,
        now_ns: u64,
        movement_max_age_ns: u64,
        object_max_age_ns: u64,
        continuous_max_age_ns: u64,
        mut cancel: F,
    ) -> u64 {
        let max_age = |t: EventType| match t {
            EventType::Movement => movement_max_age_ns,
            EventType::Object => object_max_age_ns,
            EventType::Continuous => continuous_max_age_ns,
        };

        let mut total_deleted = 0u64;
        for camera_id in self.events.camera_ids() {
            if cancel() {
                break;
            }
            let expired = self
                .events
                .expired_for_sweep(camera_id, now_ns, |e| max_age(e.event_type));
            if expired.is_empty() {
                continue;
            }

            let outcome = sweep_expired(
                &self.events,
                camera_id,
                expired,
                &mut cancel,
                |entry| async move { self.remove_event_files(camera_id, &entry).await },
            )
            .await;

            total_deleted += outcome.deleted;
            if outcome.deleted > 0 {
                tracing::info!(
                    camera = %camera_id,
                    deleted = outcome.deleted,
                    "pruned expired warm events"
                );
            }
            if outcome.failed > 0 {
                tracing::warn!(
                    camera = %camera_id,
                    failed = outcome.failed,
                    "expired warm events are still on disk after a failed delete, \
                     kept indexed for the next prune (paths at debug level)"
                );
            }
        }
        total_deleted
    }

    /// Delete every file belonging to one event — sidecar and thumbnails first, the video
    /// last.
    async fn remove_event_files(&self, camera_id: &str, entry: &WarmEventEntry) -> Removal {
        let path = self.resolve_file_path(camera_id, entry);
        // Best effort, and deliberately not a reason to keep the video:
        // holding an expired recording because a thumbnail resisted would
        // break the retention promise for kilobytes.
        let _ = tokio::fs::remove_file(&path.with_extension("jpg")).await;
        let _ = tokio::fs::remove_file(&path.with_extension("json")).await;
        let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
        let dir = path.parent().unwrap_or(&self.data_dir);
        for i in 0..MAX_FILMSTRIP_FRAMES {
            let _ = tokio::fs::remove_file(dir.join(format!("{}_thumb_{}.jpg", stem, i))).await;
        }

        match tokio::fs::remove_file(&path).await {
            Ok(()) => Removal::Deleted,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Removal::Missing,
            Err(e) => {
                // Per-file detail is debug: a persistently broken disk hits
                // this on every sweep and every low-space pass, and the
                // callers already report one aggregate warning per pass.
                tracing::debug!(
                    camera = %camera_id,
                    path = %path.display(),
                    error = %e,
                    "failed to delete warm event file"
                );
                // The video is still on disk, so the entry stays indexed for
                // the next attempt — now a bare `.ts` with its metadata gone.
                Removal::Failed
            }
        }
    }

    /// Emergency prune for low disk space: delete the oldest events, cheapest tier first, until
    /// `satisfied()` reports the pressure is gone or nothing is left.
    pub async fn emergency_prune<F: FnMut() -> bool>(&self, satisfied: F) -> EmergencyOutcome {
        evict_tiers(
            &self.events,
            EvictionPolicy {
                skip_failed: true,
                stop_on_failure: false,
                reason: "emergency prune: deleted event to reclaim disk space",
            },
            |_, entry| entry.event_type,
            satisfied,
            // Never cancelled: an eviction here is a handful of `unlink`s, so
            // the cancellation guarantee is discharged structurally (see
            // [`crate::storage::contract`]).
            || false,
            |camera_id, entry| async move { self.remove_event_files(&camera_id, &entry).await },
        )
        .await
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
        index.query("cam", EventPage::unbounded(0, u64::MAX))
    }

    #[test]
    fn newest_event_end_is_the_end_of_the_last_event() {
        let index = indexed(&[(0, 1000), (10 * SEC, 5000), (5 * SEC, 1000)]);
        assert_eq!(
            index.newest_event_end_ns("cam"),
            Some(10 * SEC + 5 * SEC),
            "not the end of the latest-starting event"
        );
    }

    #[test]
    fn newest_event_end_is_none_without_events_or_camera() {
        assert_eq!(indexed(&[]).newest_event_end_ns("cam"), None);
        assert_eq!(indexed(&[(0, 1000)]).newest_event_end_ns("other"), None);
    }

    fn running() -> impl FnMut() -> bool {
        || false
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
                    sidecar_bytes: 0,
                    thumbnail_bytes: 0,
                    object_classes: Vec::new(),
                    backend: None,
                    model: None,
                    detections: Vec::new(),
                    filmstrip_frames: 0,
                    continues: false,
                    recovered: false,
                    delete_failed: false,
                },
            );
        }
        index
    }

    #[test]
    fn query_returns_long_events_that_started_before_the_window() {
        let index = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let hits = index.query("cam", EventPage::unbounded(50 * SEC, 60 * SEC));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].start_pts_ns, 0);
    }

    #[test]
    fn query_returns_every_overlapping_event_in_start_order() {
        let index = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
        let starts: Vec<u64> = index
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .iter()
            .map(|e| e.start_pts_ns)
            .collect();
        assert_eq!(starts, vec![0, 10 * SEC, 20 * SEC]);
        assert!(index
            .query("unknown", EventPage::unbounded(0, u64::MAX))
            .is_empty());
    }

    #[test]
    fn zero_duration_events_are_found_at_their_start() {
        let index = indexed(&[(10 * SEC, 0)]);
        assert_eq!(
            index
                .query("cam", EventPage::unbounded(10 * SEC, 10 * SEC))
                .len(),
            1
        );
        assert!(index
            .query("cam", EventPage::unbounded(10 * SEC + 1, 20 * SEC))
            .is_empty());
    }

    #[test]
    fn query_bounds_include_events_that_only_touch_them() {
        let index = indexed(&[(10 * SEC, 5_000)]);
        assert_eq!(
            index
                .query("cam", EventPage::unbounded(15 * SEC, 20 * SEC))
                .len(),
            1
        );
        assert!(index
            .query("cam", EventPage::unbounded(15 * SEC + 1, 20 * SEC))
            .is_empty());
        assert_eq!(
            index.query("cam", EventPage::unbounded(0, 10 * SEC)).len(),
            1
        );
        assert!(index
            .query("cam", EventPage::unbounded(0, 10 * SEC - 1))
            .is_empty());
    }

    #[test]
    fn query_with_an_inverted_range_is_empty() {
        let index = indexed(&[(0, 100_000), (10 * SEC, 1_000)]);
        assert!(index
            .query("cam", EventPage::unbounded(u64::MAX, 0))
            .is_empty());
        assert!(index
            .query("cam", EventPage::unbounded(20 * SEC, 5 * SEC))
            .is_empty());
    }

    #[test]
    fn scan_round_trips_continues_and_keeps_type_from_directory() {
        let dir = tempfile::tempdir().unwrap();

        write_event_files(
            dir.path(),
            "movements",
            "1000_5000",
            Some(r#"{"detections":[],"continues":true}"#),
        );
        write_event_files(dir.path(), "movements", "2000_5000", None);
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

        let movement_chunk = index
            .find_event("cam", EventRef::new(1000, 5000, EventType::Movement))
            .unwrap();
        assert_eq!(movement_chunk.event_type, EventType::Movement);
        assert!(movement_chunk.continues);
        assert!(movement_chunk.object_classes.is_empty());

        let plain = index
            .find_event("cam", EventRef::new(2000, 5000, EventType::Movement))
            .unwrap();
        assert_eq!(plain.event_type, EventType::Movement);
        assert!(!plain.continues);

        let object_chunk = index
            .find_event("cam", EventRef::new(3000, 5000, EventType::Object))
            .unwrap();
        assert_eq!(object_chunk.event_type, EventType::Object);
        assert!(object_chunk.continues);
        assert_eq!(object_chunk.detections.len(), 1);
        assert_eq!(object_chunk.backend.as_deref(), Some("ollama"));
    }

    #[test]
    fn scan_picks_up_continuous_chunks_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_5000", None);
        write_event_files(
            dir.path(),
            "continuous",
            "2000_5000",
            Some(r#"{"detections":[],"continues":true}"#),
        );

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let first = index
            .find_event("cam", EventRef::new(1000, 5000, EventType::Continuous))
            .unwrap();
        assert_eq!(first.event_type, EventType::Continuous);
        assert!(!first.continues);
        let follow = index
            .find_event("cam", EventRef::new(2000, 5000, EventType::Continuous))
            .unwrap();
        assert_eq!(follow.event_type, EventType::Continuous);
        assert!(follow.continues);
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

        index
            .prune(7 * day_ns, 14 * day_ns, day_ns, running())
            .await;

        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].event_type, EventType::Movement);
        assert!(index
            .find_event(
                "cam",
                EventRef::new(continuous_pts, 5000, EventType::Continuous)
            )
            .is_none());
        assert!(!dir
            .path()
            .join("cam")
            .join("continuous")
            .join(format!("{continuous_pts}_5000.ts"))
            .exists());
    }

    const DAY: u64 = 86_400 * SEC;
    const NOW: u64 = 1_700_000_000 * SEC;

    fn archive(dir: &std::path::Path, starts: &[u64]) -> WarmEventIndex {
        for start in starts {
            write_event_files(dir, "continuous", &format!("{start}_1000"), None);
        }
        let index = WarmEventIndex::new(&["cam".to_string()], dir.to_path_buf());
        index.scan();
        index
    }

    #[tokio::test]
    async fn no_forward_clock_jump_empties_the_archive() {
        let starts: Vec<u64> = (0..48).map(|i| NOW - (48 - i) * 3600 * SEC).collect();
        let retention = 2 * DAY;
        for jump in [
            retention / 2,
            retention,
            retention + retention / 4,
            2 * retention,
            30 * 365 * DAY,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let index = archive(dir.path(), &starts);
            assert_eq!(
                index
                    .prune_at(NOW, retention, retention, retention, running())
                    .await,
                0
            );

            let deleted = index
                .prune_at(NOW + jump, retention, retention, retention, running())
                .await;
            assert_eq!(deleted, 12, "jump {jump}: the cap is a quarter of 48");
            assert_eq!(
                entries(&index).len(),
                36,
                "jump {jump}: the sweep deleted past the cap"
            );
            assert_eq!(entries(&index)[0].start_pts_ns, starts[12], "jump {jump}");
        }
    }

    #[tokio::test]
    async fn a_shortened_retention_still_drains() {
        let dir = tempfile::tempdir().unwrap();
        let starts: Vec<u64> = (0..30).map(|i| NOW - (30 - i) * DAY).collect();
        let index = archive(dir.path(), &starts);

        let sweep = || index.prune_at(NOW, 2 * DAY, 2 * DAY, 2 * DAY, running());
        assert_eq!(sweep().await, 8, "28 expired, a quarter of 30 may go");

        let mut sweeps = 1;
        while entries(&index).len() > 2 {
            sweep().await;
            sweeps += 1;
            assert!(sweeps < 20, "the retention change never drained");
        }
        assert_eq!(sweeps, 6, "the drain took an unexpected number of sweeps");
        let kept: Vec<u64> = entries(&index).iter().map(|e| e.start_pts_ns).collect();
        assert_eq!(kept, vec![starts[28], starts[29]]);
    }

    #[tokio::test]
    async fn ordinary_hourly_expiry_is_not_capped() {
        let dir = tempfile::tempdir().unwrap();
        let starts: Vec<u64> = (1..=50).map(|i| NOW - i * 3600 * SEC).rev().collect();
        let index = archive(dir.path(), &starts);

        let retention = 2 * DAY;
        let deleted = index
            .prune_at(NOW, retention, retention, retention, running())
            .await;
        assert_eq!(deleted, 2);
        assert_eq!(entries(&index).len(), 48);
        assert_eq!(entries(&index)[0].start_pts_ns, starts[2]);
    }

    #[tokio::test]
    async fn undeletable_events_do_not_consume_the_cap_forever() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..4u64 {
            write_undeletable_event(dir.path(), "continuous", &format!("{}_1000", 1000 + i));
        }
        for i in 4..8u64 {
            write_event_files(
                dir.path(),
                "continuous",
                &format!("{}_1000", 1000 + i),
                None,
            );
        }
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        assert_eq!(index.prune(1, 1, 1, running()).await, 0);
        assert_eq!(entries(&index).len(), 8);

        assert_eq!(index.prune(1, 1, 1, running()).await, 4);
        assert_eq!(entries(&index).len(), 4);
    }

    fn write_undeletable_event(dir: &std::path::Path, subdir: &str, stem: &str) {
        let path = dir.join("cam").join(subdir).join(format!("{stem}.ts"));
        std::fs::create_dir_all(&path).unwrap();
    }

    #[tokio::test]
    async fn prune_keeps_events_it_could_not_delete_and_retries_them() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        write_undeletable_event(dir.path(), "continuous", "2000_1000");

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(entries(&index).len(), 2);

        assert_eq!(index.prune(1, 1, 1, running()).await, 1);
        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0].start_pts_ns, 2000,
            "a failed deletion was unindexed, leaking the file"
        );

        let path = dir
            .path()
            .join("cam")
            .join("continuous")
            .join("2000_1000.ts");
        std::fs::remove_dir(&path).unwrap();
        std::fs::write(&path, b"tsdata").unwrap();

        assert_eq!(
            index.prune(1, 1, 1, running()).await,
            1,
            "the retry deleted nothing"
        );
        assert!(entries(&index).is_empty());
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn prune_deletes_the_sidecar_and_thumbnails_with_the_video() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", Some("{}"));
        let d = dir.path().join("cam").join("continuous");
        std::fs::write(d.join("1000_1000.jpg"), b"jpg").unwrap();
        for i in 0..4 {
            std::fs::write(d.join(format!("1000_1000_thumb_{i}.jpg")), b"jpg").unwrap();
        }

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(index.prune(1, 1, 1, running()).await, 1);

        let left: Vec<_> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(left.is_empty(), "orphaned by the delete: {left:?}");
    }

    #[tokio::test]
    async fn a_delete_that_cannot_finish_leaves_the_video_rather_than_its_metadata() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "continuous", "1000_1000");
        let d = dir.path().join("cam").join("continuous");
        std::fs::write(d.join("1000_1000.json"), "{}").unwrap();
        std::fs::write(d.join("1000_1000_thumb_0.jpg"), b"jpg").unwrap();

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(index.prune(1, 1, 1, running()).await, 0);

        assert!(d.join("1000_1000.ts").is_dir(), "the video was deleted");
        assert!(
            !d.join("1000_1000.json").exists(),
            "the sidecar was left for the video's sake"
        );
        assert!(!d.join("1000_1000_thumb_0.jpg").exists());
        assert_eq!(
            entries(&index).len(),
            1,
            "an event still on disk was unindexed"
        );
    }

    #[tokio::test]
    async fn prune_unindexes_events_whose_files_already_vanished() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        std::fs::remove_file(
            dir.path()
                .join("cam")
                .join("continuous")
                .join("1000_1000.ts"),
        )
        .unwrap();

        assert_eq!(
            index.prune(1, 1, 1, running()).await,
            0,
            "an already-gone file was counted as reclaimed"
        );
        assert!(entries(&index).is_empty());
    }

    #[tokio::test]
    async fn prune_stops_between_events_once_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..4 {
            write_event_files(
                dir.path(),
                "continuous",
                &format!("{}_1000", 1000 + i),
                None,
            );
        }
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let mut checks = 0;
        let deleted = index
            .prune(1, 1, 1, || {
                checks += 1;
                checks > 2
            })
            .await;
        assert_eq!(deleted, 1, "the sweep ignored the flag between events");
        assert_eq!(entries(&index).len(), 3);

        assert_eq!(index.prune(1, 1, 1, || true).await, 0);
        assert_eq!(entries(&index).len(), 3);

        assert_eq!(index.prune(1, 1, 1, running()).await, 3);
        assert!(entries(&index).is_empty());
    }

    #[tokio::test]
    async fn prune_does_not_unindex_a_different_event_sharing_a_start() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "movements", "1000_1000");
        write_event_files(dir.path(), "objects", "1000_2000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(entries(&index).len(), 2);

        assert_eq!(index.prune(1, 1, 1, running()).await, 1);
        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1, "an unrelated entry was unindexed");
        assert_eq!(remaining[0].event_type, EventType::Movement);
        assert_eq!(remaining[0].duration_ms, 1000);
    }

    #[tokio::test]
    async fn prune_distinguishes_events_that_differ_only_in_duration() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "continuous", "1000_1000");
        write_event_files(dir.path(), "continuous", "1000_2000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(entries(&index).len(), 2);

        assert_eq!(index.prune(1, 1, 1, running()).await, 1);
        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1, "an unrelated entry was unindexed");
        assert_eq!(remaining[0].duration_ms, 1000);
    }

    #[tokio::test]
    async fn emergency_prune_distinguishes_events_that_differ_only_in_duration() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "continuous", "1000_1000");
        write_event_files(dir.path(), "continuous", "1000_2000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        assert_eq!(index.emergency_prune(|| false).await.deleted, 1);
        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1, "an unrelated entry was unindexed");
        assert_eq!(remaining[0].duration_ms, 1000);
    }

    #[test]
    fn an_upgraded_event_no_longer_answers_to_its_pre_upgrade_key() {
        let index = WarmEventIndex::new(&["cam".to_string()], PathBuf::from("/nonexistent"));
        index.insert(
            "cam",
            WarmEventEntry {
                start_pts_ns: 1000,
                duration_ms: 5000,
                event_type: EventType::Movement,
                file_size: 0,
                sidecar_bytes: 0,
                thumbnail_bytes: 0,
                object_classes: Vec::new(),
                backend: None,
                model: None,
                detections: Vec::new(),
                filmstrip_frames: 0,
                continues: false,
                recovered: false,
                delete_failed: false,
            },
        );
        let snapshot = EventRef::new(1000, 5000, EventType::Movement);
        assert!(index.find_event("cam", snapshot).is_some());

        index.update_event("cam", 1000, 5000, |e| e.event_type = EventType::Object);

        assert!(
            index.find_event("cam", snapshot).is_none(),
            "an upgraded event would be unindexed by the sweep it raced"
        );
        assert!(index
            .find_event("cam", EventRef::new(1000, 5000, EventType::Object))
            .is_some());
    }

    #[tokio::test]
    async fn emergency_prune_keeps_events_it_could_not_delete() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "continuous", "1000_1000");
        write_event_files(dir.path(), "continuous", "2000_1000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let outcome = index.emergency_prune(|| false).await;
        assert_eq!(
            outcome,
            EmergencyOutcome {
                deleted: 1,
                failed: 1,
                missing: 0
            },
            "an undeletable event was reported as reclaimed"
        );
        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].start_pts_ns, 1000);
        assert!(dir
            .path()
            .join("cam")
            .join("continuous")
            .join("1000_1000.ts")
            .exists());
    }

    #[test]
    fn free_space_threshold_uses_injected_value() {
        assert!(should_emergency_prune(0, 100));
        assert!(should_emergency_prune(99, 100));
        assert!(!should_emergency_prune(100, 100));
        assert!(!should_emergency_prune(u64::MAX, 100));
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
        write_event_files(dir.path(), "continuous", "5000_1000", None);
        write_event_files(dir.path(), "continuous", "4000_1000", None);
        write_event_files(dir.path(), "movements", "3000_1000", None);
        write_event_files(dir.path(), "objects", "1000_1000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let mut checks = 0;
        let outcome = index
            .emergency_prune(|| {
                checks += 1;
                checks > 3
            })
            .await;
        assert_eq!(outcome.deleted, 3);

        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].event_type, EventType::Object);
        assert_eq!(remaining[0].start_pts_ns, 1000);
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
        let outcome = index.emergency_prune(|| false).await;
        assert_eq!(outcome.deleted, 1);
        assert!(entries(&index).is_empty());
    }

    #[tokio::test]
    async fn emergency_prune_immediately_satisfied_deletes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(
            index.emergency_prune(|| true).await,
            EmergencyOutcome::default()
        );
        assert_eq!(entries(&index).len(), 1);
    }

    #[tokio::test]
    async fn emergency_prune_unindexes_vanished_events_without_counting_them() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        write_event_files(dir.path(), "continuous", "2000_1000", None);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let continuous = dir.path().join("cam").join("continuous");
        std::fs::remove_file(continuous.join("1000_1000.ts")).unwrap();
        std::fs::remove_file(continuous.join("2000_1000.ts")).unwrap();

        assert_eq!(
            index.emergency_prune(|| false).await,
            EmergencyOutcome {
                deleted: 0,
                failed: 0,
                missing: 2
            },
            "already-gone events were counted as reclaimed space"
        );
        assert!(entries(&index).is_empty());
    }

    #[tokio::test]
    async fn emergency_prune_reclaims_past_undeletable_oldest_events() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..8 {
            write_undeletable_event(dir.path(), "continuous", &format!("{}_1000", 1000 + i));
        }
        for i in 0..20 {
            write_event_files(
                dir.path(),
                "continuous",
                &format!("{}_1000", 2000 + i),
                None,
            );
        }
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        let outcome = index.emergency_prune(|| false).await;
        assert_eq!(
            outcome.deleted, 20,
            "undeletable oldest events blocked every newer deletable one"
        );
        assert_eq!(outcome.failed, 8);
        assert_eq!(entries(&index).len(), 8);
    }

    #[tokio::test]
    async fn emergency_prune_stops_retrying_known_failures_but_the_sweep_does_not() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "continuous", "1000_1000");
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        assert_eq!(index.emergency_prune(|| false).await.failed, 1);
        assert_eq!(
            index.emergency_prune(|| false).await,
            EmergencyOutcome::default(),
            "the guard re-attempted a file the filesystem had already refused"
        );
        assert_eq!(entries(&index).len(), 1);

        assert_eq!(index.prune(1, 1, 1, running()).await, 0);
        let path = dir
            .path()
            .join("cam")
            .join("continuous")
            .join("1000_1000.ts");
        std::fs::remove_dir(&path).unwrap();
        std::fs::write(&path, b"tsdata").unwrap();
        assert_eq!(index.prune(1, 1, 1, running()).await, 1);
        assert!(entries(&index).is_empty());
    }

    fn warnings_from(body: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct CapturedLog(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

        impl std::io::Write for CapturedLog {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let logs = CapturedLog::default();
        {
            let _reader = tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .with_writer(logs.clone())
                    .with_max_level(tracing::Level::WARN)
                    .with_ansi(false)
                    .finish(),
            );
            body();
        }
        let written = logs.0.lock().unwrap().clone();
        String::from_utf8(written).unwrap()
    }

    #[test]
    fn unreadable_directory_entries_cost_one_warning_and_the_scan_carries_on() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "movements", "1000_5000", None);
        let movements = dir.path().join("cam").join("movements");

        const FAILURES: usize = 10_000;
        let failing = std::iter::repeat_with(|| {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "EACCES",
            ))
        })
        .take(FAILURES);

        let mut entries = Vec::new();
        let warned = warnings_from(|| {
            scan_dir_entries(
                failing.chain(std::fs::read_dir(&movements).unwrap()),
                EventType::Movement,
                "cam",
                &movements,
                &mut entries,
            );
        });

        assert_eq!(entries.len(), 1, "the scan gave up at the first failure");
        assert_eq!(entries[0].start_pts_ns, 1000);
        assert_eq!(warned.matches("WARN").count(), 1, "{warned}");
        assert!(
            warned.contains(&format!("unreadable={FAILURES}")),
            "{warned}"
        );
        assert!(warned.contains("EACCES"), "{warned}");
    }

    #[test]
    fn a_healthy_scan_warns_about_nothing() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "movements", "1000_5000", None);
        write_event_files(dir.path(), "objects", "2000_5000", Some("{}"));
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        let warned = warnings_from(|| index.scan());
        assert!(warned.is_empty(), "{warned}");
        assert_eq!(entries(&index).len(), 2);
    }

    #[test]
    fn a_filmstrip_with_a_gap_still_counts_the_frames_that_are_there() {
        for (present, expected) in [
            (vec![0, 1, 2, 3], 4),
            (vec![1, 2, 3], 4), // the first frame lost to a crash
            (vec![0, 2], 3),    // a hole in the middle
            (vec![3], 4),       // only the last survived
            (vec![0], 1),
            (vec![], 0),
        ] {
            let dir = tempfile::tempdir().unwrap();
            write_event_files(dir.path(), "movements", "1000_5000", None);
            let d = dir.path().join("cam").join("movements");
            for i in &present {
                std::fs::write(d.join(format!("1000_5000_thumb_{i}.jpg")), b"jpg").unwrap();
            }

            let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
            index.scan();
            let event = index
                .find_event("cam", EventRef::new(1000, 5000, EventType::Movement))
                .unwrap();
            assert_eq!(event.filmstrip_frames, expected, "frames {present:?}");
        }
    }

    #[tokio::test]
    async fn a_delete_reaches_the_frames_above_a_gap() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "movements", "1000_5000", None);
        let d = dir.path().join("cam").join("movements");
        for i in [1, 3] {
            std::fs::write(d.join(format!("1000_5000_thumb_{i}.jpg")), b"jpg").unwrap();
        }

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        assert_eq!(index.prune(1, 1, 1, running()).await, 1);

        let left: Vec<_> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(
            left.is_empty(),
            "frames above the gap were leaked: {left:?}"
        );
    }

    #[test]
    fn a_directory_walk_that_blocks_does_not_wedge_the_runtime() {
        let (released_tx, released_rx) = std::sync::mpsc::channel::<()>();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel::<bool>();

        let runner = std::thread::spawn(move || {
            let dir = tempfile::tempdir().unwrap();
            write_event_files(dir.path(), "movements", "1000_5000", None);
            let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
            let held = std::sync::Arc::new(std::sync::Mutex::new(Some(released_rx)));

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let walked = tokio::sync::Mutex::new(false);
                tokio::join!(
                    async {
                        index
                            .scan_off_thread_with(move |data_dir, camera_id| {
                                if let Some(rx) = held.lock().unwrap().take() {
                                    let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
                                }
                                scan_camera_dirs(data_dir, camera_id)
                            })
                            .await;
                    },
                    async {
                        tokio::task::yield_now().await;
                        released_tx.send(()).unwrap();
                        *walked.lock().await = true;
                    }
                );
                let _ = finished_tx.send(*walked.lock().await);
            });
            assert_eq!(entries(&index).len(), 1);
        });

        let released = finished_rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("the blocking walk wedged the runtime worker it was awaited on");
        assert!(released, "the walk finished without anything releasing it");
        runner.join().unwrap();
    }

    #[test]
    fn scan_defaults_continues_false_for_legacy_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(
            dir.path(),
            "objects",
            "1000_5000",
            Some(r#"{"classes":["car"]}"#),
        );
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();
        let e = index
            .find_event("cam", EventRef::new(1000, 5000, EventType::Object))
            .unwrap();
        assert!(!e.continues);
        assert_eq!(e.object_classes, vec!["car".to_string()]);
    }
}
