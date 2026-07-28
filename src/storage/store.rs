use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;

/// Number of recent motion entries to retain mask JPEGs for
const MASK_RETAIN_COUNT: usize = 60;

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

/// Stage views for one camera, one lock per stage. Writers touch a single
/// stage at a time on the analyzer thread and readers are debug-UI polls, so
/// keeping the locks separate costs nothing and keeps a JPEG clone in the API
/// from standing between the analyzer and the next stage it publishes.
type CameraMaps = [RwLock<Option<Vec<u8>>>; MapKind::COUNT];

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
            let camera_maps: CameraMaps = std::array::from_fn(|_| RwLock::new(None));
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
            *maps[kind as usize].write_recover() = Some(jpeg);
        }
    }

    pub fn get_map(&self, camera_id: &str, kind: MapKind) -> Option<Vec<u8>> {
        self.maps.get(camera_id)?[kind as usize]
            .read_recover()
            .clone()
    }

    pub fn set_stability_map(&self, camera_id: &str, jpeg: Vec<u8>) {
        self.set_map(camera_id, MapKind::Stability, jpeg);
    }

    pub fn set_background_map(&self, camera_id: &str, jpeg: Vec<u8>) {
        self.set_map(camera_id, MapKind::Background, jpeg);
    }

    pub fn set_raw_mog2_map(&self, camera_id: &str, jpeg: Vec<u8>) {
        self.set_map(camera_id, MapKind::RawMog2, jpeg);
    }

    pub fn set_no_shadow_map(&self, camera_id: &str, jpeg: Vec<u8>) {
        self.set_map(camera_id, MapKind::NoShadow, jpeg);
    }

    pub fn set_morph_map(&self, camera_id: &str, jpeg: Vec<u8>) {
        self.set_map(camera_id, MapKind::Morph, jpeg);
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
