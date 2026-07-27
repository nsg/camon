use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionDetail {
    pub class: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
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
    /// Set once a deletion of this event has failed. Purely in-RAM (a restart
    /// clears it, and the scan retries everything): it takes the event out of
    /// *emergency* candidate selection, so a low-space pass ahead of every
    /// write does not re-attempt a file the filesystem has already refused.
    /// The hourly sweep ignores this and keeps retrying — that is where a
    /// transient failure gets its second chance.
    pub delete_failed: bool,
}

/// What identifies one indexed event, and with it exactly one file on disk:
/// `{data_dir}/{camera}/{event_type}/{start_pts_ns}_{duration_ms}.ts`.
///
/// The start PTS alone does not. Nothing enforces its uniqueness — `scan`
/// happily indexes the same start under two event types with two durations,
/// and `insert` never checks — and a movement→object upgrade changes an
/// entry's type while a prune is mid-flight. Unindexing on the start alone
/// would then drop a surviving entry on the strength of some other entry's
/// delete, which is the leak this whole path exists to prevent.
type EventKey = (u64, EventType, u32);

fn event_key(entry: &WarmEventEntry) -> EventKey {
    (entry.start_pts_ns, entry.event_type, entry.duration_ms)
}

/// Outcome of deleting one indexed event's files.
enum Removal {
    /// The video file was deleted; its bytes are back.
    Deleted,
    /// The video file was already gone. Nothing was reclaimed, but the index
    /// entry has to go too — it describes a file that does not exist.
    Missing,
    /// The video file is still on disk (EACCES, EIO, ...). The entry stays
    /// indexed so the next prune retries it instead of leaking the file.
    ///
    /// The cost is deliberate: such an event stays listed, and stays offered
    /// for playback, indefinitely past its configured retention — for as long
    /// as the deletion keeps failing. A visible retention violation an
    /// operator can see and act on beats a file that is gone from the index,
    /// still eating disk, and never retried by anything.
    Failed,
}

/// What one pass of [`WarmEventIndex::emergency_prune`] actually achieved.
/// The three counts are distinct outcomes with distinct operator meanings —
/// "nothing to delete", "deletions are failing", and "someone else already
/// reclaimed it" all produce zero deleted events and call for different
/// reactions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EmergencyOutcome {
    /// Events deleted; the only count that reflects reclaimed bytes.
    pub deleted: u64,
    /// Events whose deletion failed. They are still on disk and still indexed.
    pub failed: u64,
    /// Events whose file had already vanished. Nothing was reclaimed here, but
    /// the stale index entries were dropped.
    pub missing: u64,
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
            delete_failed: false,
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

    /// End of the newest indexed event, in wall-clock nanoseconds. Entries are
    /// kept sorted by start, so the last one is the newest.
    pub fn newest_event_end_ns(&self, camera_id: &str) -> Option<u64> {
        let entries = self.cameras.get(camera_id)?.read_recover();
        entries.last().map(|e| {
            e.start_pts_ns
                .saturating_add((e.duration_ms as u64) * 1_000_000)
        })
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

    /// Delete events past their per-class retention, up to this sweep's share
    /// of the camera (see [`cap_sweep_deletions`]), and drop each one from the
    /// index once its video is gone — an event whose files could not be deleted
    /// stays listed for the next sweep to retry. Returns the number of events
    /// actually deleted.
    ///
    /// `cancel` is polled between events and between cameras so a shutdown
    /// does not wait out a whole sweep. Never mid-event: an event that lost
    /// its `.ts` but kept its sidecar and thumbnails is invisible to the
    /// startup scan, which only looks at `.ts` files, so those files would
    /// leak exactly as the ones this method refuses to unindex would.
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
        for (camera_id, lock) in self.cameras.iter() {
            if cancel() {
                break;
            }
            let (indexed, expired) = {
                let entries = lock.read_recover();
                let expired: Vec<WarmEventEntry> = entries
                    .iter()
                    .filter(|e| now_ns.saturating_sub(e.start_pts_ns) > max_age(e.event_type))
                    .cloned()
                    .collect();
                (entries.len(), expired)
            };

            if expired.is_empty() {
                continue;
            }
            let expired = cap_sweep_deletions(camera_id, indexed, expired);

            let mut deleted = 0u64;
            let mut failed = 0u64;
            let mut unindex: HashSet<EventKey> = HashSet::new();
            for entry in &expired {
                if cancel() {
                    break;
                }
                match self.remove_event_files(camera_id, entry).await {
                    Removal::Deleted => {
                        deleted += 1;
                        unindex.insert(event_key(entry));
                    }
                    Removal::Missing => {
                        unindex.insert(event_key(entry));
                    }
                    Removal::Failed => {
                        failed += 1;
                        self.mark_delete_failed(camera_id, event_key(entry));
                    }
                }
            }

            {
                let mut entries = lock.write_recover();
                entries.retain(|e| !unindex.contains(&event_key(e)));
            }

            total_deleted += deleted;
            if deleted > 0 {
                tracing::info!(
                    camera = %camera_id,
                    deleted,
                    "pruned expired warm events"
                );
            }
            if failed > 0 {
                tracing::warn!(
                    camera = %camera_id,
                    failed,
                    "expired warm events are still on disk after a failed delete, \
                     kept indexed for the next prune (paths at debug level)"
                );
            }
        }
        total_deleted
    }

    /// Delete every file belonging to one event (.ts, sidecar, thumbnails).
    async fn remove_event_files(&self, camera_id: &str, entry: &WarmEventEntry) -> Removal {
        let path = self.resolve_file_path(camera_id, entry);
        let thumb = path.with_extension("jpg");
        let removed = match tokio::fs::remove_file(&path).await {
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
                Removal::Failed
            }
        };
        // The video is still on disk, so the entry stays indexed for the next
        // attempt and its metadata is left intact — stripping the sidecar and
        // thumbnails off a file that is still there helps nobody. Being
        // indexed is not a promise that it still reads: whatever blocked the
        // delete may well block playback too.
        if matches!(removed, Removal::Failed) {
            return removed;
        }
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

    /// Remember that this event resisted deletion, so the low-space guard
    /// stops offering it as reclaimable space. Keyed on the full
    /// [`EventKey`]: an entry that has been upgraded since is a different
    /// event and must not inherit the flag.
    fn mark_delete_failed(&self, camera_id: &str, key: EventKey) {
        let Some(lock) = self.cameras.get(camera_id) else {
            return;
        };
        let mut entries = lock.write_recover();
        // Entries are sorted by start PTS, which repeats across event types, so
        // walk the run of equal starts rather than trusting one hit.
        let from = entries.partition_point(|e| e.start_pts_ns < key.0);
        for entry in entries[from..]
            .iter_mut()
            .take_while(|e| e.start_pts_ns == key.0)
        {
            if event_key(entry) == key {
                entry.delete_failed = true;
                return;
            }
        }
    }

    /// Emergency prune for low-disk-space conditions: delete the oldest events
    /// first, cheapest-to-lose tier first (continuous → movements → objects),
    /// until `satisfied()` reports the pressure is gone (in production: free
    /// space back above `min_free_bytes`) or nothing is left to delete.
    ///
    /// Reports what it achieved: see [`EmergencyOutcome`]. A pass ends when the
    /// pressure is gone or its candidates are exhausted — never on a failure
    /// count, which would let the oldest few undeletable events starve every
    /// newer deletable one and stop recording outright.
    ///
    /// This path is deliberately outside [`cap_sweep_deletions`]: space
    /// pressure is not clock-derived, and a full disk stops recording
    /// altogether. So a low-space pass during a held-back drain can delete the
    /// very footage the sweep's cap is holding.
    ///
    /// Candidates exclude events whose deletion has already failed once
    /// ([`WarmEventEntry::delete_failed`]). That is what bounds the work: this
    /// runs ahead of every single write, and re-attempting a file the
    /// filesystem has refused costs a syscall to learn nothing. The hourly
    /// sweep does the retrying.
    pub async fn emergency_prune<F: FnMut() -> bool>(&self, mut satisfied: F) -> EmergencyOutcome {
        let mut outcome = EmergencyOutcome::default();
        'tiers: for tier in [
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
                        .filter(|e| e.event_type == tier && !e.delete_failed)
                        .cloned()
                        .map(|e| (camera_id.clone(), e)),
                );
            }
            candidates.sort_by_key(|(_, e)| e.start_pts_ns);

            for (camera_id, entry) in candidates {
                if satisfied() {
                    break 'tiers;
                }
                // A failed delete keeps its entry: the file is still on disk and
                // occupying the space this prune is trying to reclaim, so the
                // hourly sweep must see it again. One poisoned file must not
                // block the rest, hence `continue` and not `break` — the
                // events behind it are the space this pass exists to reclaim.
                let removal = self.remove_event_files(&camera_id, &entry).await;
                if matches!(removal, Removal::Failed) {
                    outcome.failed += 1;
                    self.mark_delete_failed(&camera_id, event_key(&entry));
                    continue;
                }
                let key = event_key(&entry);
                if let Some(lock) = self.cameras.get(&camera_id) {
                    lock.write_recover().retain(|e| event_key(e) != key);
                }
                if matches!(removal, Removal::Deleted) {
                    outcome.deleted += 1;
                    tracing::warn!(
                        camera = %camera_id,
                        start_pts_ns = entry.start_pts_ns,
                        event_type = ?entry.event_type,
                        "emergency prune: deleted event to reclaim disk space"
                    );
                } else {
                    outcome.missing += 1;
                }
            }
        }
        outcome
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

/// Wall-clock nanoseconds since the epoch: the clock event start times are
/// stamped with, and so the only one their age can be measured against. A clock
/// set before 1970 reads as 0 rather than panicking — a box booting with no
/// idea what time it is, which is the scenario [`cap_sweep_deletions`] exists
/// for, must not take the process down.
pub(crate) fn wall_clock_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Share of a camera's archive one sweep may delete — a quarter.
const SWEEP_DELETE_SHARE: usize = 4;

/// Floor under the share, so an archive of a handful of events still expires in
/// one sweep. Losing four events is not the loss the cap exists to prevent, and
/// dribbling them out an hour at a time would only make retention look broken
/// on small installs.
const SWEEP_DELETE_FLOOR: usize = 4;

fn sweep_delete_limit(indexed: usize) -> usize {
    indexed.div_ceil(SWEEP_DELETE_SHARE).max(SWEEP_DELETE_FLOOR)
}

/// Hold back the tail of an over-large expiry, so no single sweep can empty an
/// archive.
///
/// An event's start time is wall clock at keyframe time and its age is
/// `now - start`, so a forward clock correction of J ages every stored event by
/// J at once. A box with no battery-backed RTC does exactly that on an ordinary
/// boot: systemd-timesyncd (and fake-hwclock) restore the clock they saved at
/// shutdown, so the box comes up as far behind as it was switched off and jumps
/// forward when NTP lands — J is the off-time, and a box switched off over a
/// weekend jumps a weekend. Once J reaches the retention window, every event
/// ever recorded is expired in the same sweep. That sweep used to delete the
/// lot, and report it at `info!`.
///
/// So one flat cap covers every expiry: at most [`sweep_delete_limit`] events
/// per camera per sweep, oldest first, with a `warn!` naming the counts
/// whenever anything is held back. The cap is deliberately blind to *why* an
/// event expired. A clock jump, a shortened retention and a long outage are
/// indistinguishable from inside a sweep, and every test that tries to tell
/// them apart is a hole at the jump sizes it guesses wrong about — "how far
/// past due is it" leaves J up to 1.25 retention windows uncapped, and "does
/// anything recent survive" disengages the moment the first post-jump event is
/// recorded, which is within seconds.
///
/// Ordinary retention never comes near the cap: an hourly sweep of an R-day
/// retention expires 1/(24R) of an archive, under 4% at the one-day minimum
/// camon accepts. What does reach it is a real mass expiry — retention cut from
/// 30 days to 2, a camera whose whole archive aged out while it was offline —
/// and that still drains completely, a quarter of what is left per sweep (never
/// fewer than [`SWEEP_DELETE_FLOOR`]) until only what the retention keeps
/// remains. It takes a working day instead of one pass, and says so at `warn!`
/// the whole way down.
///
/// Two paths deliberately bypass the cap, because neither is clock-derived: the
/// low-space emergency prune and the stathost storage budget. So a disk filling
/// up during a held-back drain can still delete the footage this cap is holding
/// — running out of space stops recording altogether, which is the more urgent
/// failure.
///
/// Events whose deletion already failed do not count against the cap: they were
/// let through an earlier sweep's cap and are only being retried, and charging
/// them again would let a few undeletable events at the head of the queue
/// starve every deletion behind them for good. That relies on the caller
/// marking its failures ([`WarmEventEntry::delete_failed`]); both backends do.
pub(crate) fn cap_sweep_deletions(
    camera_id: &str,
    indexed: usize,
    expired: Vec<WarmEventEntry>,
) -> Vec<WarmEventEntry> {
    let expired_count = expired.len();
    let limit = sweep_delete_limit(indexed);
    let mut budget = limit;
    let mut held_back = 0usize;
    // Filtering in place keeps the index's oldest-first order, so the oldest
    // footage goes first and a sweep cut short by shutdown has still deleted
    // the events nearest their retention.
    let deleting: Vec<WarmEventEntry> = expired
        .into_iter()
        .filter(|entry| {
            if entry.delete_failed {
                return true;
            }
            if budget > 0 {
                budget -= 1;
                return true;
            }
            held_back += 1;
            false
        })
        .collect();

    let deleting_count = deleting.len();
    if held_back > 0 {
        tracing::warn!(
            camera = %camera_id,
            indexed,
            expired = expired_count,
            deleting = deleting_count,
            held_back,
            "more warm events expired at once than one sweep may delete — a forward clock \
             jump, a shortened retention and a long outage all look like this. Deleting the \
             oldest {deleting_count} of {expired_count} expired; the {held_back} held back \
             follow on later sweeps"
        );
    }
    deleting
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

    /// Seeds the recording watchdog, so it has to be the END of the newest
    /// event: a camera mid-way through a long continuous chunk has not been
    /// silent since that chunk started.
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

    /// A cancel predicate that never fires, for the sweeps that are not about
    /// shutdown.
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
        index
            .prune(7 * day_ns, 14 * day_ns, day_ns, running())
            .await;

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

    const DAY: u64 = 86_400 * SEC;
    /// A plausible wall clock (2023-11-14), so the ages in these tests are the
    /// ages a real sweep would compute.
    const NOW: u64 = 1_700_000_000 * SEC;

    /// One-second continuous chunks on disk at `starts`, indexed by a scan.
    fn archive(dir: &std::path::Path, starts: &[u64]) -> WarmEventIndex {
        for start in starts {
            write_event_files(dir, "continuous", &format!("{start}_1000"), None);
        }
        let index = WarmEventIndex::new(&["cam".to_string()], dir.to_path_buf());
        index.scan();
        index
    }

    /// The catastrophic case: a box with no battery-backed RTC boots at the
    /// clock timesyncd saved at shutdown and NTP jumps it forward by the
    /// off-time, which ages every stored event at once. No jump size may empty
    /// the archive in one sweep — least of all a jump the size of the retention
    /// window itself, which is what being switched off for exactly that long
    /// produces, and which an "how overdue is it" test would wave through.
    #[tokio::test]
    async fn no_forward_clock_jump_empties_the_archive() {
        // Two days of footage, one chunk an hour, against a two-day retention.
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
            // Nothing is due before the correction lands.
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
            // Oldest first, so what survives a jump is the newest footage.
            assert_eq!(entries(&index)[0].start_pts_ns, starts[12], "jump {jump}");
        }
    }

    /// The cap only slows a mass expiry down. An operator who cuts retention
    /// from 30 days to 2 still gets their disk back, over sweeps rather than in
    /// one silent pass.
    #[tokio::test]
    async fn a_shortened_retention_still_drains() {
        let dir = tempfile::tempdir().unwrap();
        // A month of footage, one chunk a day, ages 30 days down to 1. Against
        // the new two-day retention all but the newest two are expired.
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
        // What is left is exactly what the new retention keeps.
        let kept: Vec<u64> = entries(&index).iter().map(|e| e.start_pts_ns).collect();
        assert_eq!(kept, vec![starts[28], starts[29]]);
    }

    /// Ordinary retention never comes near the cap: an hourly sweep of an
    /// R-day retention expires 1/(24R) of the archive, and the cap is a
    /// quarter of it.
    #[tokio::test]
    async fn ordinary_hourly_expiry_is_not_capped() {
        let dir = tempfile::tempdir().unwrap();
        // Fifty hours of footage, one chunk an hour, against a two-day
        // retention: the two oldest chunks aged out in the last two hours.
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

    /// The cap must not be spent, sweep after sweep, on the same files that
    /// refuse to be deleted — a stuck head of the queue would starve every
    /// deletion behind it for as long as the failure lasts.
    #[tokio::test]
    async fn undeletable_events_do_not_consume_the_cap_forever() {
        let dir = tempfile::tempdir().unwrap();
        // Eight events, all long past due; the four oldest cannot be deleted.
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

        // The first sweep spends its whole cap on the four that refuse to go.
        assert_eq!(index.prune(1, 1, 1, running()).await, 0);
        assert_eq!(entries(&index).len(), 8);

        // Retrying those is free, so the next sweep reaches the four behind them.
        assert_eq!(index.prune(1, 1, 1, running()).await, 4);
        assert_eq!(entries(&index).len(), 4);
    }

    /// Put something undeletable where an event file belongs: on Linux
    /// `unlink` on a directory fails with EISDIR for every user, root
    /// included, which a permission bit could not guarantee.
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

        // Both are expired, only one can actually go.
        assert_eq!(index.prune(1, 1, 1, running()).await, 1);
        let remaining = entries(&index);
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0].start_pts_ns, 2000,
            "a failed deletion was unindexed, leaking the file"
        );

        // Whatever blocked the delete clears — the path is a normal file again
        // — and the next prune actually deletes it, and says so.
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
    async fn prune_unindexes_events_whose_files_already_vanished() {
        let dir = tempfile::tempdir().unwrap();
        write_event_files(dir.path(), "continuous", "1000_1000", None);
        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        // Deleted behind camon's back: nothing to reclaim, but an entry
        // pointing at a file that does not exist must not linger forever.
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

    /// Shutdown asks the sweep to stop rather than cancelling it mid-event, so
    /// the flag has to be honoured inside `prune` — and between *events*, not
    /// just once per camera: a camera with a day of footage is one iteration
    /// of the outer loop and thousands of the inner one.
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

        // Cancelled part-way in: the per-camera check passes, the first event
        // is deleted, and the rest of that camera's sweep is abandoned.
        let mut checks = 0;
        let deleted = index
            .prune(1, 1, 1, || {
                checks += 1;
                checks > 2
            })
            .await;
        assert_eq!(deleted, 1, "the sweep ignored the flag between events");
        assert_eq!(entries(&index).len(), 3);

        // Cancelled before it starts: nothing at all.
        assert_eq!(index.prune(1, 1, 1, || true).await, 0);
        assert_eq!(entries(&index).len(), 3);

        // Uncancelled, the same sweep takes the rest.
        assert_eq!(index.prune(1, 1, 1, running()).await, 3);
        assert!(entries(&index).is_empty());
    }

    /// The retain key has to identify a file, not just a start time: `scan`
    /// indexes the same start under two event types, and an upgrade moves an
    /// entry between them while a prune is in flight.
    #[tokio::test]
    async fn prune_does_not_unindex_a_different_event_sharing_a_start() {
        let dir = tempfile::tempdir().unwrap();
        // Same start PTS, two event types, two durations — one deletable, one
        // not. Only the deletable one may leave the index.
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

    /// Same start, same event type, different duration: two distinct files,
    /// so the duration has to carry its own weight in the key.
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

    /// The same collision through the emergency path, which keys its retain
    /// separately.
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

    /// The race the key guards against: a sweep snapshots a movement, the
    /// writer upgrades that same event to an object mid-sweep, and the sweep
    /// then finds `movements/{stem}.ts` gone. The surviving object entry must
    /// not answer to the key of the movement that was pruned.
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
        let snapshot_key = event_key(&index.find_event("cam", 1000).unwrap());

        index.update_event("cam", 1000, |e| e.event_type = EventType::Object);

        let after = event_key(&index.find_event("cam", 1000).unwrap());
        assert_ne!(
            snapshot_key, after,
            "an upgraded event would be unindexed by the sweep it raced"
        );
    }

    #[tokio::test]
    async fn emergency_prune_keeps_events_it_could_not_delete() {
        let dir = tempfile::tempdir().unwrap();
        write_undeletable_event(dir.path(), "continuous", "1000_1000");
        write_event_files(dir.path(), "continuous", "2000_1000", None);

        let index = WarmEventIndex::new(&["cam".to_string()], dir.path().to_path_buf());
        index.scan();

        // Never satisfied: it tries both, oldest first. Only the second frees
        // anything, so only that one may be counted and unindexed.
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

    /// Reclaiming nothing has three meanings and the caller logs three
    /// different things: this is the one where a concurrent guard already
    /// deleted the files. Stale entries go; the "deleted" count must not move.
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

    /// The starvation case: the oldest events cannot be deleted and plenty of
    /// newer ones can. Head-of-line blocking here is fatal — the guard runs
    /// ahead of every write, so a pass that reclaims nothing means every event
    /// is dropped from then on and recording stops for good.
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

    /// What bounds the work instead of a failure counter: an event that has
    /// refused deletion once stops being offered to the *guard*, which runs
    /// ahead of every write — while the hourly sweep keeps retrying it.
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

        // The hourly sweep excludes nothing: it retries, and once the
        // obstruction clears it finishes the job.
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
