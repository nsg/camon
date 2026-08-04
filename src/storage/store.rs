use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::locks::LockExt;

/// Number of recent motion entries to retain mask JPEGs for
const MASK_RETAIN_COUNT: usize = 60;

/// How long a request for a stage keeps that stage being produced.
///
/// The debug UI polls the maps every 5 s and gives an image that never
/// arrives 15 s before abandoning it, so two consecutive requests can be
/// 20 s apart while somebody is watching. 30 s clears that with room to
/// spare; overshooting only costs a few encodes after the view is closed,
/// while undershooting would blank the overlay on a slow round.
const MAP_DEMAND_WINDOW: Duration = Duration::from_secs(30);

pub struct MotionEntry {
    pub segment_sequence: u64,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub motion_score: f32,
    pub mask_jpeg: Option<Vec<u8>>,
}

/// One of the detector's pipeline-stage views, published as a JPEG for the
/// debug UI. The string form is the stage's URL segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapKind {
    Stability,
    Background,
    RawMog2,
    NoShadow,
    Morph,
}

impl MapKind {
    const COUNT: usize = 5;

    pub const ALL: [MapKind; Self::COUNT] = [
        Self::Stability,
        Self::Background,
        Self::RawMog2,
        Self::NoShadow,
        Self::Morph,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stability => "stability",
            Self::Background => "background",
            Self::RawMog2 => "raw",
            Self::NoShadow => "no-shadow",
            Self::Morph => "morph",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

/// A stage view and the moment it was published. The timestamp is what lets
/// [`MotionStore::get_map`] tell a current view from one left behind when the
/// stage stopped being produced.
struct PublishedMap {
    jpeg: Vec<u8>,
    at: Instant,
}

/// One stage's latest view, and when the API last had a request for it (which
/// includes requests answered `404`). The request time is what tells the
/// analyzer whether the stage is worth encoding at all — see
/// [`MotionStore::map_wanted`].
#[derive(Default)]
struct MapSlot {
    published: RwLock<Option<PublishedMap>>,
    last_request: RwLock<Option<Instant>>,
}

/// Stage views for one camera: an image lock and a demand lock per stage. The
/// analyzer writes one stage at a time, and a debug-UI poll takes the demand
/// lock for writing before it reads the image, so keeping the two apart keeps
/// a JPEG clone in the API from standing between the analyzer and the next
/// stage it publishes.
type CameraMaps = [MapSlot; MapKind::COUNT];

#[derive(Clone)]
pub struct MotionStore {
    cameras: Arc<HashMap<String, RwLock<VecDeque<MotionEntry>>>>,
    maps: Arc<HashMap<String, CameraMaps>>,
}

impl MotionStore {
    pub fn new(camera_ids: &[String]) -> Self {
        let mut cameras = HashMap::new();
        let mut maps = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(VecDeque::new()));
            let camera_maps: CameraMaps = std::array::from_fn(|_| MapSlot::default());
            maps.insert(id.clone(), camera_maps);
        }
        Self {
            cameras: Arc::new(cameras),
            maps: Arc::new(maps),
        }
    }

    pub fn insert(&self, camera_id: &str, entry: MotionEntry) {
        if let Some(lock) = self.cameras.get(camera_id) {
            lock.write_recover().push_back(entry);
        }
    }

    pub fn get_motion(&self, camera_id: &str) -> Vec<MotionSnapshot> {
        match self.cameras.get(camera_id) {
            Some(lock) => {
                let entries = lock.read_recover();
                entries
                    .iter()
                    .map(|e| MotionSnapshot {
                        segment_sequence: e.segment_sequence,
                        duration_ns: e.end_time_ns - e.start_time_ns,
                        motion_score: e.motion_score,
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn get_mask(&self, camera_id: &str, segment_sequence: u64) -> Option<Vec<u8>> {
        let lock = self.cameras.get(camera_id)?;
        let entries = lock.read_recover();
        entries
            .iter()
            .find(|e| e.segment_sequence == segment_sequence)
            .and_then(|e| e.mask_jpeg.clone())
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
            // Keep mask JPEGs only for the most recent entries
            let len = entries.len();
            if len > MASK_RETAIN_COUNT {
                for entry in entries.iter_mut().take(len - MASK_RETAIN_COUNT) {
                    entry.mask_jpeg = None;
                }
            }
        }
    }

    pub fn set_map(&self, camera_id: &str, kind: MapKind, jpeg: Vec<u8>) {
        if let Some(maps) = self.maps.get(camera_id) {
            *maps[kind as usize].published.write_recover() = Some(PublishedMap {
                jpeg,
                at: Instant::now(),
            });
        }
    }

    /// Read a stage's latest view, and register that somebody wants it.
    ///
    /// The answer is a current view or nothing at all. Production stops once
    /// demand lapses, and the last image stays in the slot, so without the age
    /// check a view of the scene as it was an hour ago would be served as if it
    /// were live — and the debug overlays cannot show that difference: a frame
    /// that has quietly stopped updating looks exactly like a scene that has
    /// stopped changing. Anything older than [`MAP_DEMAND_WINDOW`] is therefore
    /// withheld. The request is recorded either way, so a stage that has aged
    /// out is put back into production and the next tick refills it, exactly as
    /// for a stage that was never published at all.
    pub fn get_map(&self, camera_id: &str, kind: MapKind) -> Option<Vec<u8>> {
        let slot = &self.maps.get(camera_id)?[kind as usize];
        *slot.last_request.write_recover() = Some(Instant::now());
        slot.published
            .read_recover()
            .as_ref()
            .filter(|map| map.at.elapsed() <= MAP_DEMAND_WINDOW)
            .map(|map| map.jpeg.clone())
    }

    /// Whether a stage was asked for recently enough to be worth producing.
    ///
    /// Each stage costs a greyscale JPEG encode per camera per analysis tick,
    /// and outside the debug view nothing ever reads the result. Demand is
    /// tracked per camera *and* per stage, so watching one camera's overlay
    /// does not put the others back to work.
    pub fn map_wanted(&self, camera_id: &str, kind: MapKind) -> bool {
        let Some(maps) = self.maps.get(camera_id) else {
            return false;
        };
        let last = *maps[kind as usize].last_request.read_recover();
        last.is_some_and(|at| at.elapsed() <= MAP_DEMAND_WINDOW)
    }

    /// Back-date a stage's last request, so tests can reach across the demand
    /// window without waiting it out.
    #[cfg(test)]
    pub(crate) fn mark_map_requested_ago(&self, camera_id: &str, kind: MapKind, ago: Duration) {
        if let Some(maps) = self.maps.get(camera_id) {
            *maps[kind as usize].last_request.write_recover() = Some(super::back_date(ago));
        }
    }

    /// Back-date a stage's published view, so tests can age one out of currency
    /// without waiting the window out.
    #[cfg(test)]
    pub(crate) fn mark_map_published_ago(&self, camera_id: &str, kind: MapKind, ago: Duration) {
        if let Some(maps) = self.maps.get(camera_id) {
            if let Some(map) = maps[kind as usize].published.write_recover().as_mut() {
                map.at = super::back_date(ago);
            }
        }
    }

    pub fn last_sequence(&self, camera_id: &str) -> Option<u64> {
        self.cameras
            .get(camera_id)?
            .read_recover()
            .back()
            .map(|e| e.segment_sequence)
    }
}

pub struct MotionSnapshot {
    pub segment_sequence: u64,
    pub duration_ns: u64,
    pub motion_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MotionStore {
        MotionStore::new(&["cam".to_string(), "other".to_string()])
    }

    #[test]
    fn every_stage_keeps_its_own_map() {
        let store = store();
        for kind in MapKind::ALL {
            store.set_map("cam", kind, kind.as_str().as_bytes().to_vec());
        }
        for kind in MapKind::ALL {
            assert_eq!(
                store.get_map("cam", kind).as_deref(),
                Some(kind.as_str().as_bytes()),
                "{} read back another stage's map",
                kind.as_str()
            );
        }
    }

    #[test]
    fn maps_are_per_camera_and_absent_until_published() {
        let store = store();
        store.set_map("cam", MapKind::Morph, vec![1]);
        assert_eq!(store.get_map("other", MapKind::Morph), None);
        assert_eq!(store.get_map("cam", MapKind::Stability), None);
        assert_eq!(store.get_map("no-such-camera", MapKind::Morph), None);
    }

    /// The overlays claim to show what the detector sees right now, and the
    /// client cannot tell a retained frame from a live one. Since production
    /// stops when nobody is looking, the image left behind has to stop being
    /// served too.
    #[test]
    fn a_view_that_stopped_being_refreshed_is_withheld() {
        let store = store();
        store.set_map("cam", MapKind::Background, vec![1]);
        assert_eq!(
            store.get_map("cam", MapKind::Background).as_deref(),
            Some(&[1][..])
        );

        store.mark_map_published_ago("cam", MapKind::Background, MAP_DEMAND_WINDOW / 2);
        assert!(
            store.get_map("cam", MapKind::Background).is_some(),
            "dropped a view that was still being refreshed"
        );

        store.mark_map_published_ago("cam", MapKind::Background, MAP_DEMAND_WINDOW * 2);
        assert_eq!(
            store.get_map("cam", MapKind::Background),
            None,
            "served a background model from a session that ended long ago"
        );
    }

    /// A request that comes back empty because the view aged out has to arm
    /// production just as one that finds nothing at all does — otherwise the
    /// stage could never come back.
    #[test]
    fn a_request_that_ages_out_still_arms_production() {
        let store = store();
        store.set_map("cam", MapKind::Morph, vec![1]);
        store.mark_map_published_ago("cam", MapKind::Morph, MAP_DEMAND_WINDOW * 2);

        assert_eq!(store.get_map("cam", MapKind::Morph), None);
        assert!(
            store.map_wanted("cam", MapKind::Morph),
            "a stale read left the stage out of production, so it can never refill"
        );
    }

    /// Producing a stage costs a JPEG encode per camera per analysis tick, so
    /// the analyzer asks first — and the answer has to be "no" until somebody
    /// looks, and "no" again once they stop.
    #[test]
    fn a_stage_is_wanted_only_while_requests_keep_arriving() {
        let store = store();
        for kind in MapKind::ALL {
            assert!(
                !store.map_wanted("cam", kind),
                "{} wanted before anyone asked for it",
                kind.as_str()
            );
        }

        store.get_map("cam", MapKind::Morph);
        assert!(store.map_wanted("cam", MapKind::Morph));

        store.mark_map_requested_ago("cam", MapKind::Morph, MAP_DEMAND_WINDOW / 2);
        assert!(
            store.map_wanted("cam", MapKind::Morph),
            "gave up mid-window"
        );

        store.mark_map_requested_ago("cam", MapKind::Morph, MAP_DEMAND_WINDOW * 2);
        assert!(
            !store.map_wanted("cam", MapKind::Morph),
            "kept encoding long after the last request"
        );
    }

    /// Watching one camera, or one stage, must not put the rest back to work.
    #[test]
    fn demand_for_a_stage_does_not_spread() {
        let store = store();
        store.get_map("cam", MapKind::Morph);
        assert!(!store.map_wanted("other", MapKind::Morph));
        assert!(!store.map_wanted("cam", MapKind::Stability));
        assert!(!store.map_wanted("no-such-camera", MapKind::Morph));
    }

    /// The stage names are URL segments: renaming one silently breaks the debug
    /// UI, which requests them by name.
    #[test]
    fn stage_names_round_trip_and_are_distinct() {
        let mut names: Vec<&str> = MapKind::ALL.iter().map(|k| k.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), MapKind::ALL.len());
        for kind in MapKind::ALL {
            assert_eq!(MapKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(MapKind::parse("stability/raw"), None);
    }
}
