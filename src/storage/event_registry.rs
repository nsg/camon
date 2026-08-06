//! Registry of the events a camera has in flight to warm storage, and of the
//! object-detection verdicts that decide how long each of them is kept.
//!
//! What is at stake is retention, not labelling. A movement event is deleted
//! after `movement_retention_days` and an object event after
//! `object_retention_days` — two days against fourteen on the production box —
//! so an event that should have been classified as an object and was not loses
//! its footage twelve days early, and nothing ever revisits it. The rule this
//! module exists to keep is therefore:
//!
//! > a verdict that arrives for a motion run reaches the event covering it,
//! > whichever order the two happen in.
//!
//! Two producers race for that. The analyzer reads the detection store,
//! assembles the event, hands the write to the warm writer, and only then
//! knows the file stem the event will have. The detection worker inserts its
//! verdict into the detection store and then looks for a written event to
//! upgrade. The window between the analyzer's read and its handoff is not a
//! microsecond: the handoff is a blocking send on a short channel whose
//! consumer is one slow remote store away, and it can sit there for minutes. A
//! verdict landing anywhere inside that window used to hit nothing on either
//! path — the store read had already happened, and no entry existed yet to
//! claim.
//!
//! So a record is opened BEFORE the detection store is read, and it lives
//! through both halves of the event's life:
//!
//! * [`State::Pending`] — the analyzer owns it. No file exists yet and the
//!   identity may not even be known, so a verdict landing here is *parked* on
//!   the record. It must not be sent: an upgrade message reaching the writer
//!   ahead of the write it refers to would go looking for a file nobody has
//!   created.
//! * [`State::Committed`] — the write is in the writer's queue and the
//!   identity is known, so a verdict landing here becomes an upgrade the
//!   detection worker sends down that same channel, guaranteed to arrive
//!   behind the write itself (FIFO).
//! * [`State::Classified`] — the event is an object event, either written that
//!   way or upgraded once already. Terminal, so a second verdict for the same
//!   run cannot trigger a second whole-sidecar rewrite.
//!
//! [`EventRegistry::open`] hands out the [`PendingEvent`] handle and
//! [`PendingEvent::commit`] is the hinge: under one write lock it takes
//! whatever parked and moves the record on. Whichever side of the handoff a
//! verdict arrives on, at most one upgrade per covering event is sent (a
//! verdict straddling a capped run's chunks upgrades each chunk it covers,
//! once), and the thread that sends
//! it is the one that can — the analyzer for a verdict that parked, because
//! only it can queue a message behind its own write; the worker for one that
//! arrives after. A handle dropped without committing (assembly found the
//! segments evicted, the writer was already gone) takes its record with it:
//! there will be no file for a verdict to upgrade.
//!
//! ## What bounds the memory
//!
//! Records are forgotten by resolution, never by count. A record is resolved
//! when it is classified, or when it is committed and no crop job that could
//! still cover it remains — which the registry knows because the detection
//! queue tells it: [`EventRegistry::expect_verdict`] when a job is accepted,
//! [`EventRegistry::verdict_settled`] when the worker has answered it or the
//! queue dropped it unprocessed.
//!
//! "No outstanding job overlaps this range" is a fact rather than a guess, and
//! what makes it one is the statement order in
//! `analytics::pipeline::MotionAnalyzer::process_new_segments`: a sequence is
//! analyzed once, every crop job covering the batch just analyzed is dispatched
//! — and so registered here — and only then are the runs that closed in that
//! batch emitted as events. A run closing mid-batch does *not* have all its
//! jobs enqueued when it closes; it has them all enqueued by the time its
//! record is opened, which is the property this depends on. Moving the emit
//! ahead of the dispatch would leave a record open with a job for its own
//! sequences still to arrive, and the next event to close would forget it. The
//! comment at that call site says so, at the line someone reordering it has to
//! edit.
//!
//! That bounds the set. The detection queue holds at most
//! `DETECT_QUEUE_PER_CAMERA_CAP` jobs per camera plus the one in flight, each
//! covering one contiguous run, so only a handful of records per camera can be
//! unresolved at any moment — at a few dozen bytes each, memory is not the
//! constraint. A count that says otherwise is a bug somewhere else, so it is
//! reported ([`RECORD_ALARM`]) rather than trimmed away: dropping a record
//! whose verdict is still outstanding is exactly how footage gets deleted
//! twelve days early, and this module would rather hold a kilobyte it does not
//! need than be the thing that does that.
//!
//! ## Deliberately not solved here
//!
//! The registry is in RAM, so a restart forgets every record. Nothing is lost
//! with them, because a restart also loses the crop jobs those verdicts would
//! have come from — the detection worker is aborted during shutdown and its
//! queue goes with the process — so no upgrade is ever left half applied.
//!
//! However an upgrade this module asked for fails to reach the writer, it
//! ends the same way — the event keeps movement retention — and none of it is
//! repairable from here, because the record is already `Classified` by the
//! time the send is attempted. What the log shows differs by path: the
//! analyzer's loss is an error naming only the camera; the worker's is a
//! warning, naming the event when the channel was merely full and only the
//! camera when it was closed — a closed channel means that camera's writer is
//! already dead, which supervision treats as fatal, so the process is on its
//! way down with it.
//!
//! A verdict that parked during the handoff can lose its upgrade at shutdown,
//! in one narrow window: the analyzer gets its write into the writer's queue,
//! and the writer finishes and exits before the analyzer can send the upgrade
//! behind it. It is consistent with what the stop already treats as droppable —
//! the whole detection queue is aborted a moment later — and the footage itself
//! is through, so the alternative (holding the drain open for a message the
//! writer may never take) would cost more than it saves.
//!
//! A verdict that arrives after the commit loses its upgrade whenever the
//! camera's writer channel is full, which needs no shutdown at all: the
//! detection worker offers that message rather than waiting for room, because
//! ONE task sends them for every camera and waiting would stop object
//! detection site-wide while a single writer finishes an upload. That is a
//! routine loss under a slow store, not an edge case — the trade and what it
//! costs are set out at `analytics::detect_worker`'s
//! `DetectionWorker::upgrade_covering_events`.
//!
//! An event whose store sidecar reads object while this process's index still
//! reads movement is a different failure, from a write that committed after
//! reporting failure, and the healing scan's `join_object_type` is what
//! repairs it.

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
    ///
    /// Called before the detection store is read, which is what closes the
    /// race this module is about: from here until the handle is committed or
    /// dropped, a verdict for these sequences has somewhere to land.
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
    ///
    /// Returns the events to upgrade *now* — those whose write is already in
    /// the writer's queue, so an upgrade message can only arrive behind it. An
    /// event still pending has no file to upgrade and no place in the channel
    /// yet, so its verdict is parked on the record and the analyzer sends it
    /// the moment the write is enqueued. Either way the record is marked
    /// classified under the same write lock, so a second verdict for the same
    /// event can never produce a second upgrade.
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

    /// That job has been answered. Whether it produced a verdict, produced
    /// nothing, or was dropped from the queue unprocessed does not matter
    /// here; what matters is that nothing more is coming from it, so the
    /// records it was holding open can go.
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
        // Written straight to `objects/`: the file already says everything an
        // upgrade would, including for a verdict that parked while it was
        // being written — that verdict is in the detection store the assembly
        // read from.
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

/// The analyzer's claim on one event, from before the detection store is read
/// until the write is in the writer's queue.
///
/// Dropped without [`PendingEvent::commit`], it abandons its record: every
/// path that drops it is a path where no file will exist, so there is nothing
/// a later verdict could upgrade. That also covers the paths nobody wrote —
/// an early return added later, a panic on the analyzer thread — which is the
/// point of it being a guard rather than a pair of calls.
pub struct PendingEvent {
    registry: EventRegistry,
    camera_id: String,
    id: u64,
    resolved: bool,
}

impl PendingEvent {
    /// The write is enqueued: the record takes upgrades of its own from here.
    ///
    /// Returns the verdict that landed while the analyzer was assembling or
    /// blocked on the handoff, if one did. That one is the caller's to send,
    /// because only a message queued after its own write can find the file the
    /// write creates.
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

    /// An event written and handed off with nothing to say about objects —
    /// the ordinary movement event a later verdict has to be able to find.
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
        // Boundary sequences do match.
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

    /// The duration cap splits a long run into chained chunks, and one crop
    /// job's sequences can straddle the split. Both chunks are the same
    /// footage as far as retention is concerned, so both are upgraded.
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

    /// The race this module exists for, in the small: the verdict lands while
    /// the analyzer is still assembling or still blocked handing off the
    /// write. It cannot be sent from there — the file does not exist — so it
    /// parks, and the commit hands it back to the one thread that can queue it
    /// behind the write.
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

    /// The same verdict, on the other side of the handoff, has to end in the
    /// same place: one upgrade, for the same event, and nothing left for a
    /// second verdict to do.
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

        // Whichever way round it went, the event is classified and a second
        // verdict has nothing to do.
        for registry in [&before, &after] {
            assert!(registry
                .deliver_verdict("cam", &[16], &verdict("car"))
                .is_empty());
        }
    }

    /// The fixed 32-entry ring this replaced dropped the oldest record under
    /// backlog, and a slow model is exactly when backlog and late verdicts
    /// happen together. A record whose crop job is still on the queue survives
    /// any number of later events.
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

    /// The bound is resolution, not a count. A camera whose verdicts are all
    /// outstanding at once keeps every record, past the mark where the
    /// registry starts saying it does not like the look of this — trimming to
    /// a count is precisely the thing that deleted footage twelve days early.
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

    /// The other half of that bargain: once the job is answered, the record it
    /// was holding open goes, so a camera that records all day does not
    /// accumulate one record per event for the life of the process.
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

    /// A job dropped at the queue cap settles the same way an answered one
    /// does — nothing is coming from it either — so it must not pin records
    /// for the life of the process.
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

    /// Assembly found the segments gone, or the writer was already gone: no
    /// file will exist under this identity, so the record goes with the handle
    /// and a verdict that arrives afterwards finds nothing to rewrite.
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

    /// Shutdown aborts the detection worker with jobs still queued, so their
    /// records are never settled. Nothing about that may block or wedge: the
    /// registry has no waits, and the events themselves are already on their
    /// way to disk.
    #[test]
    fn events_stay_committed_when_their_verdicts_are_never_coming() {
        let registry = registry();
        let _never_answered = registry.expect_verdict("cam", &[10, 11]);
        written(&registry, 1000, 10, 20);
        // Whatever the analyzer flushes on the way out is still accepted.
        written(&registry, 2000, 21, 30);
        assert_eq!(registry.held("cam"), 2);
    }
}
