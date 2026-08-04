//! The frames and model answers behind the detector's debug view.
//!
//! Everything here exists for one page of the UI, and that page is opened for
//! a few minutes when somebody is tuning a camera. Kept unconditionally it is
//! the largest thing this process holds: one entry is a full 1080p frame plus
//! up to four crops, and a crop is not smaller than that frame in the worst
//! case — motion spanning the scene degenerates the crop back to the whole
//! frame — so the cap per entry is five near-1080p JPEGs at quality 90, on the
//! order of 2.5 MB, and [`MAX_ENTRIES`] of them are retained per camera. Held
//! for the life of the process against a page nobody opened, that is the
//! largest waste in the process. Losing that memory is not only waste: the
//! process that is killed for it is killed with SIGKILL, so it never runs the
//! shutdown drain, and the footage still in the warm writers dies with it.
//!
//! So the frames are produced and kept only while somebody is watching — the
//! same bargain [`crate::storage::MotionStore::map_wanted`] strikes for the
//! stage overlays. A request on the debug endpoint opens a
//! [`DEBUG_DEMAND_WINDOW`]; the analyzer asks before encoding a full frame and
//! the detection worker asks before storing anything; and when the window
//! closes what was stored is dropped rather than kept for a viewer who has
//! gone.
//!
//! The arithmetic that leaves is a cap, not an average: with a view open,
//! `MAX_ENTRIES` × ~2.5 MB ≈ 125 MB for the camera being watched, and ~250 MB
//! if somebody has both of the production site's cameras open at once. That
//! ceiling is deliberate — it is reached only while an operator is looking at
//! the page, it lasts as long as they look, and it is the price of the view
//! being useful at all. A typical entry is nearer 1 MB; the bound is what this
//! file promises.
//!
//! With nobody watching: zero, and not one generation behind zero. The entries
//! go back within one tick of an analyzer that is ticking — the two states
//! where one briefly is not are the shutdown drain, where the process is going
//! away regardless, and an analyzer parked on a warm writer that has stopped
//! draining, which is bounded and which supervision makes fatal.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::locks::LockExt;

const MAX_ENTRIES: usize = 50;

/// How long a request for a camera's debug view keeps its frames being
/// produced and kept.
///
/// The debug view polls the entry list every 5 s and arms the next poll only
/// once the previous one has settled, so consecutive requests are 5 s apart
/// plus however long a round trip takes. That is the same poll the stage
/// overlays run on, and the window is deliberately the same 30 s as
/// [`crate::storage::MotionStore::map_wanted`]'s: a shorter one would blank
/// the view on a slow round, and this list has less to lose from overshooting
/// than the overlays do — a few extra entries outlive the viewer by half a
/// minute and are then dropped.
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

/// One camera's debug entries, and when the API last asked for them (which
/// includes requests answered with an empty list). The request time is what
/// tells the analyzer and the detection worker whether these frames are worth
/// producing at all — see [`DetectionDebugStore::wanted`].
///
/// Entries and demand are separate locks for the same reason the stage maps
/// keep them apart: the API takes the demand lock for writing on every poll,
/// and the detection worker must not have to wait behind a poll that is still
/// building its answer.
///
/// Where both are needed, the order is entries first and demand inside it,
/// never the other way round. That order is what makes the transitions atomic:
/// deciding whether a session is over and acting on that decision happen in one
/// critical section, so an expiry cannot be decided against a window that has
/// closed and then land on the entries of a session that has since begun.
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

    /// Whether this camera's debug frames were asked for recently enough to be
    /// worth producing.
    ///
    /// Producing them costs a 1080p JPEG encode per motion run and about a
    /// megabyte of resident memory per entry, and outside the debug view
    /// nothing ever reads the result. Demand is tracked per camera, so tuning
    /// one camera does not put the others to work.
    pub fn wanted(&self, camera_id: &str) -> bool {
        self.cameras.get(camera_id).is_some_and(demand_is_open)
    }

    /// Store one classification run's frames and answers — or, when nobody is
    /// watching, store nothing and drop whatever an earlier viewer left behind.
    ///
    /// The producers ask [`Self::wanted`] first and normally do not get here at
    /// all; the check is repeated because the bound is this store's to keep,
    /// not its callers'. It is repeated under the entries lock, so an entry is
    /// admitted or refused against the window as it stands at the moment the
    /// entry is added, not as it stood a moment before.
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
    ///
    /// A list served after the window closed answers for the session that is
    /// starting, not the one that ended: what an earlier viewer left behind is
    /// dropped here rather than shown, because the view presents itself as the
    /// tail of what the detector is doing now and cannot say that its entries
    /// stopped arriving. The request is recorded either way, so this call is
    /// also what puts production back on for the next run.
    pub fn list(&self, camera_id: &str) -> Vec<DebugSnapshot> {
        let Some(camera) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        // The ended session is dropped before the request is recorded, or the
        // window this poll opens would make its entries look current. Both
        // steps are ordered by the entries lock: no producer can slip an entry
        // of the new session in between them, because the clear is still
        // holding the lock it would need.
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
    ///
    /// The producers stop calling [`Self::insert`] the moment nobody is
    /// watching, so the store cannot rely on the next insert to notice: the
    /// analyzer calls this every tick instead, and the tens of megabytes an
    /// ended session left behind go back within a poll interval of it ending
    /// rather than at the next detection — which may be hours away, or never.
    pub fn expire_unwatched(&self, camera_id: &str) {
        if let Some(camera) = self.cameras.get(camera_id) {
            expire_unwatched_entries(camera_id, camera);
        }
    }

    /// Images are fetched as a consequence of a list the viewer already has,
    /// so reading one is not itself evidence that anybody is watching and does
    /// not arm production. An entry that expired between the list and the image
    /// is simply gone, and the view's next poll no longer offers it.
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

/// Give back what a closed window was holding — deciding whether it has closed
/// while holding the entries themselves.
///
/// The decision cannot be taken first and acted on afterwards. A viewer who
/// returns in between re-arms the window and the detector starts filling it
/// again, and a clear ordered against the session that ended would take the new
/// one's first entries with it. Under this lock the two orders are the only two
/// there are: either the window is already open when the decision is read and
/// nothing is dropped, or the clear finishes before any producer can add to
/// what it is clearing.
fn expire_unwatched_entries(camera_id: &str, camera: &CameraDebug) {
    let mut entries = camera.entries.write_recover();
    if demand_is_open(camera) {
        return;
    }
    release(camera_id, &mut entries);
}

/// Drop the entries under a lock the caller already holds. Silent when there is
/// nothing to drop — the common case, several times a second per camera — and a
/// debug line otherwise: dropping these is what the store is designed to do,
/// not something going wrong.
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

    /// One entry's worth of the real thing: a full frame and one crop.
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

    /// The core of it: an entry is about a megabyte of JPEG, and detection runs
    /// all night whether or not anybody ever opens the page it is for.
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

    /// Production stopping is only half of it: the frames stored while somebody
    /// was watching are the memory that has to come back when they stop.
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

    /// The analyzer's tick is what normally frees an ended session, but a poll
    /// that arrives before it must not resurrect what that session left: the
    /// view presents its entries as the live tail.
    #[test]
    fn a_poll_after_the_window_closed_starts_from_nothing() {
        let store = store();
        store.list("cam");
        insert_one(&store, "cam");

        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);
        assert!(store.list("cam").is_empty(), "served an ended session");
        assert_eq!(store.stored("cam"), 0, "hid the old entries but kept them");
    }

    /// The window closing and a viewer returning are the two things that race
    /// here, and the sweep that gives the memory back runs several times a
    /// second per camera. Its decision must not survive the moment it was taken
    /// in: a viewer who returns between the decision and the clear has a live
    /// session, and clearing it takes the frames out from under a page that is
    /// on screen.
    ///
    /// The sweep is held on the entries lock while the return happens, which is
    /// exactly the interleaving — the decision it would have acted on was taken
    /// before the window re-opened.
    #[test]
    fn a_sweep_that_started_before_a_viewer_returned_leaves_the_new_session_alone() {
        let store = store();
        store.list("cam");
        insert_one(&store, "cam");
        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);

        // Plainly first: the poll that reopens the window clears the ended
        // session, and the entry stored after it survives the sweeps that
        // follow.
        store.list("cam");
        insert_one(&store, "cam");
        store.expire_unwatched("cam");
        assert_eq!(store.stored("cam"), 1, "swept a session that had restarted");

        // Now the same thing with the sweep genuinely in flight. The lock is
        // taken here, so the sweep parks inside `expire_unwatched` having read
        // nothing yet; the window is re-armed while it waits.
        store.mark_requested_ago("cam", DEBUG_DEMAND_WINDOW * 2);
        let sweeping = {
            let held = store.cameras["cam"].entries.write_recover();
            let sweeper = store.clone();
            let sweeping = std::thread::spawn(move || sweeper.expire_unwatched("cam"));
            // Long enough for the sweep to have reached the lock — and, in a
            // version that decides before taking it, to have decided.
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

    /// A viewer that comes back has to get frames again — the window closing is
    /// not a one-way door.
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

    /// Watching one camera must not put the others to work.
    #[test]
    fn demand_does_not_spread_between_cameras() {
        let store = store();
        store.list("cam");
        assert!(!store.wanted("other"));
        assert!(!store.wanted("no-such-camera"));

        insert_one(&store, "other");
        assert_eq!(store.stored("other"), 0);
    }

    /// The cap is what the arithmetic in this file's header rests on.
    #[test]
    fn a_watched_camera_holds_no_more_than_the_cap() {
        let store = store();
        store.list("cam");
        for _ in 0..MAX_ENTRIES * 2 {
            insert_one(&store, "cam");
        }
        assert_eq!(store.stored("cam"), MAX_ENTRIES);
    }

    /// The API hands the JPEG bytes straight out; nothing here may copy them.
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
