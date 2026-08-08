//! Registry of the events a camera has in flight to warm storage, and of the object-detection
//! verdicts that decide how long each of them is kept.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::locks::LockExt;
use crate::storage::event_index::DetectionDetail;

/// More unresolved records than the detection queue can account for. Reported
/// once per camera as it is crossed, and never acted on — see the module docs
/// on what bounds the memory.
const RECORD_ALARM: usize = 256;

/// What a detection verdict says about one motion run, less the identity of
/// whichever event it turns out to land on.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub object_classes: Vec<String>,
    pub detections: Vec<DetectionDetail>,
    pub backend: String,
    pub model: String,
}

/// The identity of an event whose write is already on its way to disk: the
/// file stem (`{start_pts_ns}_{duration_ms}`) the writer will find it under,
/// plus the chain flag a sidecar rewrite has to preserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpgradeTarget {
    pub start_pts_ns: u64,
    pub duration_ms: u32,
    pub continues: bool,
}

/// Handle on a crop job the detection queue has accepted. While one of these
/// is outstanding, every record whose sequences the job covers is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerdictId(u64);

/// Where one event is between the analyzer assembling it and its class being
/// settled for good. See the module docs for why the middle state exists.
enum State {
    /// The analyzer holds it; a verdict arriving now parks here.
    Pending(Option<Verdict>),
    /// The write is queued; a verdict arriving now is upgraded on the spot.
    Committed { start_pts_ns: u64, duration_ms: u32 },
    /// Object-classified. Terminal.
    Classified,
}

struct TrackedEvent {
    id: u64,
    /// Inclusive segment-sequence range the event covers (motion range; the
    /// pre-padding reach-back is irrelevant for verdict matching).
    first_motion_seq: u64,
    last_seq: u64,
    /// Follow-on chunk flag, carried into the upgraded sidecar.
    continues: bool,
    state: State,
}

impl TrackedEvent {
    fn covers_any(&self, seqs: &[u64]) -> bool {
        seqs.iter()
            .any(|&s| s >= self.first_motion_seq && s <= self.last_seq)
    }

    fn overlaps(&self, (first, last): (u64, u64)) -> bool {
        first <= self.last_seq && last >= self.first_motion_seq
    }
}

#[derive(Default)]
struct CameraEvents {
    records: VecDeque<TrackedEvent>,
    /// Crop jobs the detection queue has accepted and not yet answered, by the
    /// inclusive sequence range each covers.
    outstanding: HashMap<u64, (u64, u64)>,
    /// Edge trigger for [`RECORD_ALARM`], so a camera that stays over the mark
    /// says so once rather than once per event.
    alarmed: bool,
}

impl CameraEvents {
    /// Drop every record whose classification question is settled: object
    /// events (nothing more can happen to them) and movement events no
    /// outstanding crop job can still cover.
    fn forget_resolved(&mut self) {
        let outstanding: Vec<(u64, u64)> = self.outstanding.values().copied().collect();
        self.records.retain(|record| match record.state {
            State::Pending(_) => true,
            State::Classified => false,
            State::Committed { .. } => outstanding.iter().any(|&range| record.overlaps(range)),
        });
    }
}

#[derive(Clone)]
pub struct EventRegistry {
    cameras: Arc<HashMap<String, RwLock<CameraEvents>>>,
    /// One counter for both record and job handles; they only ever need to be
    /// distinct from their own kind.
    next_id: Arc<AtomicU64>,
}

impl EventRegistry {
    pub fn new(camera_ids: &[String]) -> Self {
        let mut cameras = HashMap::new();
        for id in camera_ids {
            cameras.insert(id.clone(), RwLock::new(CameraEvents::default()));
        }
        Self {
            cameras: Arc::new(cameras),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Open a record for an event the analyzer is about to assemble.
    pub fn open(
        &self,
        camera_id: &str,
        first_motion_seq: u64,
        last_seq: u64,
        continues: bool,
    ) -> PendingEvent {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Some(lock) = self.cameras.get(camera_id) {
            let mut camera = lock.write_recover();
            camera.forget_resolved();
            camera.records.push_back(TrackedEvent {
                id,
                first_motion_seq,
                last_seq,
                continues,
                state: State::Pending(None),
            });
            let held = camera.records.len();
            if held > RECORD_ALARM && !camera.alarmed {
                camera.alarmed = true;
                tracing::warn!(
                    camera = %camera_id,
                    records = held,
                    "event registry holding more unclassified events than the detection queue \
                     can explain; none are being dropped, but object upgrades are lagging far \
                     behind the recordings they belong to"
                );
            } else if held <= RECORD_ALARM {
                camera.alarmed = false;
            }
        }
        PendingEvent {
            registry: self.clone(),
            camera_id: camera_id.to_string(),
            id,
            resolved: false,
        }
    }

    /// Deliver one run's verdict to every event whose sequence range it covers.
    pub fn deliver_verdict(
        &self,
        camera_id: &str,
        seqs: &[u64],
        verdict: &Verdict,
    ) -> Vec<UpgradeTarget> {
        let Some(lock) = self.cameras.get(camera_id) else {
            return Vec::new();
        };
        let mut camera = lock.write_recover();
        let mut targets = Vec::new();
        for record in camera.records.iter_mut() {
            if !record.covers_any(seqs) {
                continue;
            }
            match record.state {
                // First verdict wins, as it does for a written event: the
                // upgrade rewrites the whole sidecar, so a second one would be
                // the same rewrite with a different opinion.
                State::Pending(ref mut parked) => {
                    parked.get_or_insert_with(|| verdict.clone());
                }
                State::Committed {
                    start_pts_ns,
                    duration_ms,
                } => {
                    targets.push(UpgradeTarget {
                        start_pts_ns,
                        duration_ms,
                        continues: record.continues,
                    });
                    record.state = State::Classified;
                }
                State::Classified => {}
            }
        }
        targets
    }

    /// A crop job has entered the detection queue: until it comes back, no
    /// record whose sequences it covers may be forgotten.
    pub fn expect_verdict(&self, camera_id: &str, seqs: &[u64]) -> Option<VerdictId> {
        let lock = self.cameras.get(camera_id)?;
        let first = seqs.iter().copied().min()?;
        let last = seqs.iter().copied().max()?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        lock.write_recover().outstanding.insert(id, (first, last));
        Some(VerdictId(id))
    }

    /// That job has been answered. Whether it produced a verdict, produced nothing, or was
    /// dropped from the queue unprocessed does not matter here; what matters is that nothing
    /// more is coming from it, so the records it was holding open can go.
    pub fn verdict_settled(&self, camera_id: &str, id: Option<VerdictId>) {
        let (Some(lock), Some(VerdictId(id))) = (self.cameras.get(camera_id), id) else {
            return;
        };
        let mut camera = lock.write_recover();
        camera.outstanding.remove(&id);
        camera.forget_resolved();
    }

    fn commit(
        &self,
        camera_id: &str,
        id: u64,
        start_pts_ns: u64,
        duration_ms: u32,
        has_objects: bool,
    ) -> Option<Verdict> {
        let lock = self.cameras.get(camera_id)?;
        let mut camera = lock.write_recover();
        let record = camera.records.iter_mut().find(|record| record.id == id)?;
        let parked = match record.state {
            State::Pending(ref mut parked) => parked.take(),
            // Only the handle reaches this, and it commits at most once.
            _ => None,
        };
        // Written straight to `objects/`: the file already says everything an upgrade would,
        // including for a verdict that parked while it was being written — that verdict is in
        // the detection store the assembly read from.
        if has_objects {
            record.state = State::Classified;
            return None;
        }
        match parked {
            Some(verdict) => {
                record.state = State::Classified;
                Some(verdict)
            }
            None => {
                record.state = State::Committed {
                    start_pts_ns,
                    duration_ms,
                };
                None
            }
        }
    }

    fn forget(&self, camera_id: &str, id: u64) {
        if let Some(lock) = self.cameras.get(camera_id) {
            lock.write_recover()
                .records
                .retain(|record| record.id != id);
        }
    }

    /// How many records this camera is holding, for tests that need to see a
    /// record appear or be dropped without racing on a log line.
    #[cfg(test)]
    pub fn held(&self, camera_id: &str) -> usize {
        self.cameras
            .get(camera_id)
            .map(|lock| lock.read_recover().records.len())
            .unwrap_or(0)
    }
}

/// The analyzer's claim on one event, from before the detection store is read until the write
/// is in the writer's queue.
pub struct PendingEvent {
    registry: EventRegistry,
    camera_id: String,
    id: u64,
    resolved: bool,
}

impl PendingEvent {
    /// The write is enqueued: the record takes upgrades of its own from here.
    pub fn commit(
        mut self,
        start_pts_ns: u64,
        duration_ms: u32,
        has_objects: bool,
    ) -> Option<Verdict> {
        self.resolved = true;
        self.registry.commit(
            &self.camera_id,
            self.id,
            start_pts_ns,
            duration_ms,
            has_objects,
        )
    }
}

impl Drop for PendingEvent {
    fn drop(&mut self) {
        if !self.resolved {
            self.registry.forget(&self.camera_id, self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> EventRegistry {
        EventRegistry::new(&["cam".to_string()])
    }

    fn verdict(class: &str) -> Verdict {
        Verdict {
            object_classes: vec![class.to_string()],
            detections: vec![DetectionDetail {
                class: class.to_string(),
                confidence: 0.9,
            }],
            backend: "ollama".to_string(),
            model: "test-model".to_string(),
        }
    }

    fn written(registry: &EventRegistry, start_pts_ns: u64, first: u64, last: u64) {
        let pending = registry.open("cam", first, last, false);
        assert_eq!(pending.commit(start_pts_ns, 5000, false), None);
    }

    #[test]
    fn a_verdict_finds_the_written_event_covering_its_run() {
        let registry = registry();
        written(&registry, 1000, 10, 20);
        let targets = registry.deliver_verdict("cam", &[12, 13], &verdict("person"));
        assert_eq!(
            targets,
            vec![UpgradeTarget {
                start_pts_ns: 1000,
                duration_ms: 5000,
                continues: false,
            }]
        );
    }

    #[test]
    fn a_verdict_ignores_events_outside_its_run() {
        let registry = registry();
        written(&registry, 1000, 10, 20);
        assert!(registry
            .deliver_verdict("cam", &[21, 25], &verdict("person"))
            .is_empty());
        assert!(registry
            .deliver_verdict("cam", &[5, 9], &verdict("person"))
            .is_empty());
        assert_eq!(
            registry
                .deliver_verdict("cam", &[20], &verdict("person"))
                .len(),
            1
        );
    }

    #[test]
    fn an_event_written_with_its_detections_needs_no_upgrade() {
        let registry = registry();
        let pending = registry.open("cam", 10, 20, false);
        assert_eq!(pending.commit(1000, 5000, true), None);
        assert!(registry
            .deliver_verdict("cam", &[15], &verdict("person"))
            .is_empty());
    }

    #[test]
    fn a_second_verdict_for_one_event_upgrades_nothing() {
        let registry = registry();
        written(&registry, 1000, 10, 20);
        assert_eq!(
            registry
                .deliver_verdict("cam", &[15], &verdict("person"))
                .len(),
            1
        );
        assert!(registry
            .deliver_verdict("cam", &[16], &verdict("car"))
            .is_empty());
    }

    #[test]
    fn a_verdict_reaches_every_chunk_of_one_capped_run() {
        let registry = registry();
        let job = registry.expect_verdict("cam", &[19, 20, 21, 22]);
        written(&registry, 1000, 10, 20);
        written(&registry, 2000, 21, 30);
        assert_eq!(
            registry
                .deliver_verdict("cam", &[19, 20, 21, 22], &verdict("person"))
                .len(),
            2
        );
        registry.verdict_settled("cam", job);
    }

    #[test]
    fn an_unknown_camera_is_a_no_op() {
        let registry = registry();
        let pending = registry.open("ghost", 10, 20, false);
        assert_eq!(pending.commit(1000, 5000, false), None);
        assert!(registry
            .deliver_verdict("ghost", &[15], &verdict("person"))
            .is_empty());
    }

    #[test]
    fn a_verdict_that_lands_before_the_write_is_enqueued_is_handed_back_at_commit() {
        let registry = registry();
        let pending = registry.open("cam", 10, 20, false);

        let targets = registry.deliver_verdict("cam", &[15], &verdict("person"));
        assert!(
            targets.is_empty(),
            "an upgrade was sent for an event whose write is not queued yet"
        );

        assert_eq!(pending.commit(1000, 5000, false), Some(verdict("person")));
    }

    #[test]
    fn a_verdict_before_the_commit_and_one_after_it_settle_the_same_way() {
        let target = UpgradeTarget {
            start_pts_ns: 1000,
            duration_ms: 5000,
            continues: true,
        };

        let before = registry();
        let pending = before.open("cam", 10, 20, true);
        assert!(before
            .deliver_verdict("cam", &[15], &verdict("person"))
            .is_empty());
        let parked = pending.commit(1000, 5000, false);
        assert_eq!(parked, Some(verdict("person")));

        let after = registry();
        let pending = after.open("cam", 10, 20, true);
        assert_eq!(pending.commit(1000, 5000, false), None);
        assert_eq!(
            after.deliver_verdict("cam", &[15], &verdict("person")),
            [target]
        );

        for registry in [&before, &after] {
            assert!(registry
                .deliver_verdict("cam", &[16], &verdict("car"))
                .is_empty());
        }
    }

    #[test]
    fn an_event_whose_verdict_is_still_outstanding_survives_a_backlog() {
        let registry = registry();
        let job = registry.expect_verdict("cam", &[10, 11, 12]);
        written(&registry, 1000, 10, 20);

        for i in 1..500u64 {
            written(&registry, 1000 + i * 1000, 100 + i * 10, 105 + i * 10);
        }

        assert_eq!(
            registry
                .deliver_verdict("cam", &[11], &verdict("person"))
                .len(),
            1,
            "the record was evicted while its verdict was still outstanding"
        );
        registry.verdict_settled("cam", job);
    }

    #[test]
    fn nothing_is_ever_evicted_to_stay_under_a_count() {
        let registry = registry();
        let held = RECORD_ALARM + 50;
        let jobs: Vec<_> = (0..held as u64)
            .map(|i| {
                let job = registry.expect_verdict("cam", &[i * 10, i * 10 + 1]);
                written(&registry, 1000 + i, i * 10, i * 10 + 1);
                job
            })
            .collect();
        assert_eq!(registry.held("cam"), held);

        for (i, job) in jobs.into_iter().enumerate() {
            assert_eq!(
                registry
                    .deliver_verdict("cam", &[i as u64 * 10], &verdict("person"))
                    .len(),
                1,
                "the record for event {i} was evicted to keep the count down"
            );
            registry.verdict_settled("cam", job);
        }
    }

    #[test]
    fn records_are_forgotten_once_nothing_can_still_classify_them() {
        let registry = registry();
        let job = registry.expect_verdict("cam", &[10, 11, 12]);
        written(&registry, 1000, 10, 20);
        assert_eq!(registry.held("cam"), 1);

        registry.verdict_settled("cam", job);
        assert_eq!(
            registry.held("cam"),
            0,
            "a record was kept after the only job that could classify it was answered"
        );
    }

    #[test]
    fn a_job_that_never_ran_still_releases_the_records_it_held() {
        let registry = registry();
        let dropped = registry.expect_verdict("cam", &[10, 11]);
        let answered = registry.expect_verdict("cam", &[12, 13]);
        written(&registry, 1000, 10, 20);

        registry.verdict_settled("cam", dropped);
        assert_eq!(registry.held("cam"), 1, "the other job still covers it");
        registry.verdict_settled("cam", answered);
        assert_eq!(registry.held("cam"), 0);
    }

    #[test]
    fn an_abandoned_event_leaves_no_record_for_a_verdict_to_upgrade() {
        let registry = registry();
        let job = registry.expect_verdict("cam", &[10, 11]);
        {
            let _pending = registry.open("cam", 10, 20, false);
            assert_eq!(registry.held("cam"), 1);
        }
        assert_eq!(
            registry.held("cam"),
            0,
            "a record outlived the event it was opened for"
        );
        assert!(registry
            .deliver_verdict("cam", &[15], &verdict("person"))
            .is_empty());
        registry.verdict_settled("cam", job);
    }

    #[test]
    fn events_stay_committed_when_their_verdicts_are_never_coming() {
        let registry = registry();
        let _never_answered = registry.expect_verdict("cam", &[10, 11]);
        written(&registry, 1000, 10, 20);
        written(&registry, 2000, 21, 30);
        assert_eq!(registry.held("cam"), 2);
    }
}
