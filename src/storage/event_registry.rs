//! Registry of recently written warm events, keyed by segment-sequence range.
//!
//! The analyzer records an entry right after handing a finished event to the
//! warm writer; the detection worker consults it when an Ollama verdict
//! arrives after the covering event already reached disk, to decide whether a
//! post-hoc movement→object upgrade is due.
//!
//! Race with event assembly (verdict arrives while the analyzer is emitting
//! the event) — the ordering that makes the outcome deterministic:
//!
//! 1. analyzer reads the detection store and assembles the event;
//! 2. analyzer enqueues the write message to the warm writer;
//! 3. analyzer records the registry entry (this module).
//!
//! The worker inserts detections into the store first, then claims covering
//! entries here. Because the registry entry is recorded only *after* the
//! write message is enqueued, any upgrade message the worker sends is
//! guaranteed to arrive at the writer behind the write itself (same channel,
//! FIFO). The one losing interleaving: the verdict lands after the analyzer
//! read the detection store but before the entry was recorded — then no
//! upgrade fires and the event stays movement-classified, with the
//! detections still visible in the detection store and API. That is the
//! documented worst case; footage is never lost.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;

/// Entries kept per camera. Events are written at most every few seconds and
/// verdicts lag by at most a few minutes of queue depth, so a handful of
/// entries is plenty; the bound only guards memory.
const MAX_RECORDS_PER_CAMERA: usize = 32;

#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Start PTS of the written event — identifies the file stem together
    /// with `duration_ms`.
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    /// Inclusive segment-sequence range the event covers (motion range; the
    /// pre-padding reach-back is irrelevant for verdict matching).
    pub first_motion_seq: u64,
    pub last_seq: u64,
    /// Classification at write time. `true` means it already went to
    /// `objects/` and needs no upgrade.
    pub has_objects: bool,
    /// Follow-on chunk flag, carried into the upgraded sidecar.
    pub continues: bool,
}

impl EventRecord {
    fn covers_any(&self, seqs: &[u64]) -> bool {
        seqs.iter()
            .any(|&s| s >= self.first_motion_seq && s <= self.last_seq)
    }
}

#[derive(Clone)]
pub struct EventRegistry {
    cameras: Arc<HashMap<String, RwLock<VecDeque<EventRecord>>>>,
}

impl EventRegistry {
    pub fn new(camera_ids: &[String]) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(VecDeque::new()));
        }
        Self {
            cameras: Arc::new(cameras),
        }
    }

    /// Record a just-written event. Oldest entries fall off past the bound.
    pub fn record(&self, camera_id: &str, record: EventRecord) {
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut records = lock.write_recover();
            records.push_back(record);
            while records.len() > MAX_RECORDS_PER_CAMERA {
                records.pop_front();
            }
        }
    }

    /// Movement-classified events whose sequence range intersects `seqs`.
    /// Each returned record is atomically marked `has_objects` under the
    /// write lock, so a second verdict for the same run cannot trigger a
    /// duplicate upgrade.
    pub fn claim_movement_events(&self, camera_id: &str, seqs: &[u64]) -> Vec<EventRecord> {
        let Some(lock) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        let mut records = lock.write_recover();
        let mut claimed = Vec::new();
        for record in records.iter_mut() {
            if !record.has_objects && record.covers_any(seqs) {
                record.has_objects = true;
                claimed.push(record.clone());
            }
        }
        // The claimed clones still say has_objects = true; callers only need
        // the identity fields, but reset for clarity of intent.
        for record in &mut claimed {
            record.has_objects = false;
        }
        claimed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(start_pts_ns: u64, first: u64, last: u64, has_objects: bool) -> EventRecord {
        EventRecord {
            start_pts_ns,
            duration_ms: 5000,
            first_motion_seq: first,
            last_seq: last,
            has_objects,
            continues: false,
        }
    }

    #[test]
    fn claim_finds_covering_movement_event() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        registry.record("cam", record(1000, 10, 20, false));
        let claimed = registry.claim_movement_events("cam", &[12, 13]);
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].start_pts_ns, 1000);
    }

    #[test]
    fn claim_ignores_non_overlapping_ranges() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        registry.record("cam", record(1000, 10, 20, false));
        assert!(registry.claim_movement_events("cam", &[21, 25]).is_empty());
        assert!(registry.claim_movement_events("cam", &[5, 9]).is_empty());
        // Boundary sequences do match.
        assert_eq!(registry.claim_movement_events("cam", &[20]).len(), 1);
    }

    #[test]
    fn claim_skips_object_classified_events() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        registry.record("cam", record(1000, 10, 20, true));
        assert!(registry.claim_movement_events("cam", &[15]).is_empty());
    }

    #[test]
    fn claim_is_idempotent() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        registry.record("cam", record(1000, 10, 20, false));
        assert_eq!(registry.claim_movement_events("cam", &[15]).len(), 1);
        // Second verdict for the same run: already claimed, no double upgrade.
        assert!(registry.claim_movement_events("cam", &[16]).is_empty());
    }

    #[test]
    fn claim_matches_multiple_chunks_of_one_run() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        // A capped run split into two chained chunks.
        registry.record("cam", record(1000, 10, 20, false));
        registry.record("cam", record(2000, 21, 30, false));
        let claimed = registry.claim_movement_events("cam", &[19, 20, 21, 22]);
        assert_eq!(claimed.len(), 2);
    }

    #[test]
    fn registry_is_bounded() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        for i in 0..100u64 {
            registry.record("cam", record(i * 1000, i * 10, i * 10 + 5, false));
        }
        // The earliest entries were evicted.
        assert!(registry.claim_movement_events("cam", &[0]).is_empty());
        // The most recent survive.
        assert_eq!(registry.claim_movement_events("cam", &[990]).len(), 1);
    }

    #[test]
    fn unknown_camera_is_a_no_op() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        registry.record("ghost", record(1000, 10, 20, false));
        assert!(registry.claim_movement_events("ghost", &[15]).is_empty());
    }
}
