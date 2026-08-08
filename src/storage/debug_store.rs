//! The frames and model answers behind the detector's debug view.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::locks::LockExt;

const MAX_ENTRIES: usize = 50;

/// How long a request for a camera's debug view keeps its frames being produced and kept.
const DEBUG_DEMAND_WINDOW: Duration = Duration::from_secs(30);

pub struct DebugEntry {
    pub id: u64,
    pub timestamp: u64,
    pub frame_jpegs: Vec<Arc<Vec<u8>>>,
    pub raw_responses: Vec<String>,
    pub model: String,
    pub detection_count: usize,
    pub full_frame_jpeg: Option<Arc<Vec<u8>>>,
    /// Individual motion bounding boxes in normalized coords (x, y, w, h).
    pub motion_rects: Vec<(f32, f32, f32, f32)>,
    /// Union crop region sent to Ollama in normalized coords.
    pub crop_rect: Option<(f32, f32, f32, f32)>,
    /// Ollama-returned bboxes mapped to full-frame normalized coords.
    pub ollama_rects: Vec<(String, f32, f32, f32, f32)>,
}

pub struct DebugSnapshot {
    pub id: u64,
    pub timestamp: u64,
    pub raw_responses: Vec<String>,
    pub model: String,
    pub detection_count: usize,
    pub frame_count: usize,
    pub has_full_frame: bool,
    pub motion_rects: Vec<(f32, f32, f32, f32)>,
    pub crop_rect: Option<(f32, f32, f32, f32)>,
    pub ollama_rects: Vec<(String, f32, f32, f32, f32)>,
}

/// One camera's debug entries, and when the API last asked for them (which includes requests
/// answered with an empty list).
#[derive(Default)]
struct CameraDebug {
    entries: RwLock<VecDeque<DebugEntry>>,
    last_request: RwLock<Option<Instant>>,
}

pub struct DetectionDebugStore {
    cameras: Arc<HashMap<String, CameraDebug>>,
    next_id: Arc<AtomicU64>,
}

impl DetectionDebugStore {
    pub fn new(camera_ids: &[String]) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), CameraDebug::default());
        }
        Self {
            cameras: Arc::new(cameras),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Whether this camera's debug frames were asked for recently enough to be worth producing.
    pub fn wanted(&self, camera_id: &str) -> bool {
        self.cameras.get(camera_id).is_some_and(demand_is_open)
    }

    /// Store one classification run's frames and answers — or, when nobody is watching, store
    /// nothing and drop whatever an earlier viewer left behind.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &self,
        camera_id: &str,
        frame_jpegs: Vec<Arc<Vec<u8>>>,
        raw_responses: Vec<String>,
        model: String,
        detection_count: usize,
        full_frame_jpeg: Option<Arc<Vec<u8>>>,
        motion_rects: Vec<(f32, f32, f32, f32)>,
        crop_rect: Option<(f32, f32, f32, f32)>,
        ollama_rects: Vec<(String, f32, f32, f32, f32)>,
    ) {
        let Some(camera) = self.cameras.get(camera_id) else {
            return;
        };
        let mut entries = camera.entries.write_recover();
        if !demand_is_open(camera) {
            release(camera_id, &mut entries);
            return;
        }
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
            full_frame_jpeg,
            motion_rects,
            crop_rect,
            ollama_rects,
        });
        while entries.len() > MAX_ENTRIES {
            entries.pop_front();
        }
    }

    /// The camera's entries, and a registration that somebody wants them.
    pub fn list(&self, camera_id: &str) -> Vec<DebugSnapshot> {
        let Some(camera) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        // The ended session is dropped before the request is recorded, or the window this poll
        // opens would make its entries look current.
        expire_unwatched_entries(camera_id, camera);
        *camera.last_request.write_recover() = Some(Instant::now());
        camera
            .entries
            .read_recover()
            .iter()
            .map(|e| DebugSnapshot {
                id: e.id,
                timestamp: e.timestamp,
                raw_responses: e.raw_responses.clone(),
                model: e.model.clone(),
                detection_count: e.detection_count,
                frame_count: e.frame_jpegs.len(),
                has_full_frame: e.full_frame_jpeg.is_some(),
                motion_rects: e.motion_rects.clone(),
                crop_rect: e.crop_rect,
                ollama_rects: e.ollama_rects.clone(),
            })
            .collect()
    }

    /// Free a camera's entries once its demand window has closed.
    pub fn expire_unwatched(&self, camera_id: &str) {
        if let Some(camera) = self.cameras.get(camera_id) {
            expire_unwatched_entries(camera_id, camera);
        }
    }

    /// Images are fetched as a consequence of a list the viewer already has, so reading one is
    /// not itself evidence that anybody is watching and does not arm production.
    pub fn get_full_frame_jpeg(&self, camera_id: &str, id: u64) -> Option<Arc<Vec<u8>>> {
        self.cameras.get(camera_id).and_then(|camera| {
            let entries = camera.entries.read_recover();
            entries
                .iter()
                .find(|e| e.id == id)
                .and_then(|e| e.full_frame_jpeg.clone())
        })
    }

    pub fn get_frame_jpeg(
        &self,
        camera_id: &str,
        id: u64,
        frame_index: usize,
    ) -> Option<Arc<Vec<u8>>> {
        self.cameras.get(camera_id).and_then(|camera| {
            let entries = camera.entries.read_recover();
            entries
                .iter()
                .find(|e| e.id == id)
                .and_then(|e| e.frame_jpegs.get(frame_index).cloned())
        })
    }

    /// Back-date a camera's last request, so tests can reach across the demand
    /// window without waiting it out.
    #[cfg(test)]
    pub(crate) fn mark_requested_ago(&self, camera_id: &str, ago: Duration) {
        if let Some(camera) = self.cameras.get(camera_id) {
            *camera.last_request.write_recover() = Some(super::back_date(ago));
        }
    }

    /// How many entries a camera is holding, for the tests that assert on the
    /// memory rather than on what a reader is shown — [`Self::list`] arms the
    /// window it is meant to be measuring across.
    #[cfg(test)]
    pub(crate) fn stored(&self, camera_id: &str) -> usize {
        self.cameras
            .get(camera_id)
            .map_or(0, |camera| camera.entries.read_recover().len())
    }
}

fn demand_is_open(camera: &CameraDebug) -> bool {
    camera
        .last_request
        .read_recover()
        .is_some_and(|at| at.elapsed() <= DEBUG_DEMAND_WINDOW)
}

/// Give back what a closed window was holding — deciding whether it has closed while holding
/// the entries themselves.
fn expire_unwatched_entries(camera_id: &str, camera: &CameraDebug) {
    let mut entries = camera.entries.write_recover();
    if demand_is_open(camera) {
        return;
    }
    release(camera_id, &mut entries);
}

/// Drop the entries under a lock the caller already holds.
fn release(camera_id: &str, entries: &mut VecDeque<DebugEntry>) {
    if entries.is_empty() {
        return;
    }
    tracing::debug!(
        camera = %camera_id,
        entries = entries.len(),
        "nobody is watching the detection debug view; dropped the frames it was holding"
    );
    entries.clear();
}

impl Clone for DetectionDebugStore {
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

    fn store() -> DetectionDebugStore {
        DetectionDebugStore::new(&["cam".to_string(), "other".to_string()])
    }

    fn insert_one(store: &DetectionDebugStore, camera_id: &str) {
        store.insert(
            camera_id,
            vec![Arc::new(vec![0xaa])],
            vec!["{}".to_string()],
            "test-model".to_string(),
            0,
            Some(Arc::new(vec![0xbb])),
            Vec::new(),
            None,
            Vec::new(),
        );
    }

    #[test]
    fn nothing_is_stored_while_nobody_is_watching() {
        let store = store();
        assert!(!store.wanted("cam"));

        for _ in 0..10 {
            insert_one(&store, "cam");
        }
        assert_eq!(
            store.stored("cam"),
            0,
            "kept a megabyte per motion run for a view nobody has open"
        );
    }

    #[test]
    fn a_request_opens_the_window_and_the_entries_flow() {
        let store = store();
        store.list("cam");
        assert!(store.wanted("cam"));

        insert_one(&store, "cam");
        assert_eq!(store.stored("cam"), 1);

        let listed = store.list("cam");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].frame_count, 1);
        assert!(listed[0].has_full_frame);
    }

    #[test]
    fn what_was_stored_is_freed_once_the_window_closes() {
        let store = store();
        store.list("cam");
        for _ in 0..5 {
            insert_one(&store, "cam");
        }
        assert_eq!(store.stored("cam"), 5);

        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW / 2);
        store.expire_unwatched("cam");
        assert_eq!(
            store.stored("cam"),
            5,
            "dropped a watching viewer's frames mid-window"
        );

        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);
        store.expire_unwatched("cam");
        assert_eq!(
            store.stored("cam"),
            0,
            "kept the frames of a session that ended"
        );
    }

    #[test]
    fn a_poll_after_the_window_closed_starts_from_nothing() {
        let store = store();
        store.list("cam");
        insert_one(&store, "cam");

        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);
        assert!(store.list("cam").is_empty(), "served an ended session");
        assert_eq!(store.stored("cam"), 0, "hid the old entries but kept them");
    }

    #[test]
    fn a_sweep_that_started_before_a_viewer_returned_leaves_the_new_session_alone() {
        let store = store();
        store.list("cam");
        insert_one(&store, "cam");
        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);

        store.list("cam");
        insert_one(&store, "cam");
        store.expire_unwatched("cam");
        assert_eq!(store.stored("cam"), 1, "swept a session that had restarted");

        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);
        let sweeping = {
            let held = store.cameras["cam"].entries.write_recover();
            let sweeper = store.clone();
            let sweeping = std::thread::spawn(move || sweeper.expire_unwatched("cam"));
            std::thread::sleep(Duration::from_millis(50));
            store.mark_requested_ago("cam", Duration::ZERO);
            drop(held);
            sweeping
        };
        sweeping.join().expect("sweep");

        assert_eq!(
            store.stored("cam"),
            1,
            "a sweep decided against the session that ended cleared the one that had started"
        );
    }

    #[test]
    fn the_window_re_arms_on_the_next_request() {
        let store = store();
        store.list("cam");
        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);
        assert!(!store.wanted("cam"));

        store.list("cam");
        assert!(store.wanted("cam"));
        insert_one(&store, "cam");
        assert_eq!(store.stored("cam"), 1);
    }

    #[test]
    fn demand_does_not_spread_between_cameras() {
        let store = store();
        store.list("cam");
        assert!(!store.wanted("other"));
        assert!(!store.wanted("no-such-camera"));

        insert_one(&store, "other");
        assert_eq!(store.stored("other"), 0);
    }

    #[test]
    fn a_watched_camera_holds_no_more_than_the_cap() {
        let store = store();
        store.list("cam");
        for _ in 0..MAX_ENTRIES * 2 {
            insert_one(&store, "cam");
        }
        assert_eq!(store.stored("cam"), MAX_ENTRIES);
    }

    #[test]
    fn stored_frames_are_the_handles_that_were_inserted() {
        let store = store();
        store.list("cam");
        let crop = Arc::new(vec![0xaa]);
        let full = Arc::new(vec![0xbb]);
        store.insert(
            "cam",
            vec![Arc::clone(&crop)],
            Vec::new(),
            "test-model".to_string(),
            0,
            Some(Arc::clone(&full)),
            Vec::new(),
            None,
            Vec::new(),
        );

        let id = store.list("cam")[0].id;
        assert!(Arc::ptr_eq(
            &store.get_frame_jpeg("cam", id, 0).unwrap(),
            &crop
        ));
        assert!(Arc::ptr_eq(
            &store.get_full_frame_jpeg("cam", id).unwrap(),
            &full
        ));
    }
}
