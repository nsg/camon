//! Global object-detection worker.
//!
//! ONE async task serves all cameras. Analyzers flag motion, drop a crop job
//! on the [`DetectQueue`], and continue immediately — motion detection never
//! stalls on the vision model. The worker drains the queue strictly
//! serially: at most ONE in-flight request to Ollama at any time, across all
//! cameras, and the frames of a single run are sent one after another (the
//! production GPU is old and degrades badly under parallel load).
//!
//! The queue holds a bounded stack of jobs per camera and serves cameras
//! round-robin, newest job first. Fairness keeps one spammy camera from
//! drowning the others; newest-first keeps verdicts about what is happening
//! NOW instead of grinding through a backlog. When a camera overflows its
//! stack, its oldest job is dropped with a warning. That is explicitly
//! acceptable: the motion event still persists to `movements/`; only the
//! object upgrade is lost.
//!
//! Verdicts are handled in two ways:
//! - always written to the [`DetectionStore`], where the event-assembly path
//!   and the API read them exactly as before;
//! - delivered to the [`EventRegistry`], which decides what the covering
//!   events still need. One whose write is already queued gets an upgrade
//!   message on its camera's warm writer channel, which owns all file
//!   mutations; one the analyzer is still assembling parks the verdict and
//!   sends that message itself. See `storage::event_registry`.
//!
//! Every job is reported back to the registry when the worker is done with
//! it, verdict or no verdict, because a record covering its sequences is
//! being kept alive until it is — see [`EventRegistry::verdict_settled`].

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tokio::sync::{mpsc, Notify};

use crate::buffer::warm::{EventUpgrade, WriterMessage};
use crate::mqtt::{send_event, MqttEvent, Sighting};
use crate::storage::event_index::DetectionDetail;
use crate::storage::{
    DetectionDebugStore, DetectionEntry, DetectionStore, EventRegistry, Verdict, VerdictId,
};

use super::ollama::{Detection, OllamaClient};

/// Jobs held per camera before the oldest is evicted. A job is roughly 1 MB
/// (up to four crop JPEGs plus a full 1080p frame), so even every camera at
/// its cap stays around 100–200 MB — fine on the production box. The cap
/// exists to bound staleness, not memory: under sustained motion the oldest
/// jobs are the ones that stopped mattering.
pub const DETECT_QUEUE_PER_CAMERA_CAP: usize = 32;

/// At most this many frames of a run are sent to the model.
const MAX_FRAMES_PER_RUN: usize = 4;

/// One crop job: everything the worker needs to classify a motion run and
/// attribute the verdict back to segments and events.
pub struct DetectionJob {
    pub camera_id: String,
    /// Segment sequences of the motion run this job covers (ascending).
    pub seqs: Vec<u64>,
    /// Cropped frames, JPEG-encoded, at most 4 are used.
    pub crop_jpegs: Vec<Vec<u8>>,
    /// A full (uncropped) frame for the debug overlay.
    pub full_frame_jpeg: Option<Vec<u8>>,
    /// Individual motion boxes in normalized full-frame coords.
    pub motion_rects: Vec<(f32, f32, f32, f32)>,
    /// The union crop region the frames were cropped to, normalized.
    pub run_crop: Option<(f32, f32, f32, f32)>,
    /// The registry's handle on this job, stamped by [`DetectQueueSender::send`]
    /// as the job is accepted and handed back when the job leaves the system —
    /// answered here, or dropped at the queue cap. `None` when there is no
    /// registry to hold records for (warm storage disabled).
    pub verdict_id: Option<VerdictId>,
}

pub struct DetectionWorker {
    client: OllamaClient,
    detection_store: DetectionStore,
    debug_store: Option<DetectionDebugStore>,
    /// Registry + writer channels are only present when warm storage is
    /// enabled; without them verdicts still land in the detection store.
    event_registry: Option<EventRegistry>,
    event_senders: HashMap<String, mpsc::Sender<WriterMessage>>,
    /// Verdicts for the Home Assistant MQTT bridge. `None` when MQTT is off.
    mqtt_tx: Option<mpsc::Sender<MqttEvent>>,
}

impl DetectionWorker {
    pub fn new(
        client: OllamaClient,
        detection_store: DetectionStore,
        debug_store: Option<DetectionDebugStore>,
        event_registry: Option<EventRegistry>,
        event_senders: HashMap<String, mpsc::Sender<WriterMessage>>,
        mqtt_tx: Option<mpsc::Sender<MqttEvent>>,
    ) -> Self {
        Self {
            client,
            detection_store,
            debug_store,
            event_registry,
            event_senders,
            mqtt_tx,
        }
    }

    /// Worker main loop. Exits when every job sender (the analyzers) is
    /// gone and the queue is drained; at shutdown the task is aborted
    /// instead — pending jobs and even an in-flight request are droppable
    /// by design.
    pub async fn run(self, queue: Arc<DetectQueue>) {
        // Surface a typo'd model name in seconds rather than as a string of
        // silent detection failures.
        self.client.check_models().await;
        tracing::info!(model = %self.client.model(), "detection worker started (serial, one in-flight request)");
        while let Some(job) = queue.recv().await {
            self.process_job(job).await;
        }
        tracing::info!("detection worker stopped");
    }

    /// Classify one job and then tell the registry the question is answered,
    /// on every path out of the classification — a job that timed out, one the
    /// model saw nothing in, one whose camera has no registry at all. The
    /// records it was holding open are only released by this call, so it
    /// wraps the work rather than living inside it.
    async fn process_job(&self, job: DetectionJob) {
        let camera_id = job.camera_id.clone();
        let verdict_id = job.verdict_id;
        self.classify_job(job).await;
        if let Some(ref registry) = self.event_registry {
            registry.verdict_settled(&camera_id, verdict_id);
        }
    }

    async fn classify_job(&self, job: DetectionJob) {
        // Kept per frame rather than flattened so each detection stays
        // attributable to the crop it came from: entry `i` holds the verdict
        // for `job.crop_jpegs[i]`, including for frames the model failed on.
        let mut per_frame: Vec<Vec<Detection>> = Vec::new();
        let mut raw_responses = Vec::new();
        let mut model = self.client.model().to_string();

        // Strictly sequential: each frame waits for the previous response.
        for jpeg in job.crop_jpegs.iter().take(MAX_FRAMES_PER_RUN) {
            match self.client.detect_jpeg(jpeg).await {
                Ok(result) => {
                    per_frame.push(result.detections);
                    raw_responses.push(result.raw_response);
                    model = result.model;
                }
                Err(e) => {
                    // A timeout or server error costs only the upgrade.
                    tracing::warn!(camera = %job.camera_id, error = %e, "frame detection failed");
                    per_frame.push(Vec::new());
                    raw_responses.push(format!("ERROR: {e}"));
                }
            }
        }
        let detections: Vec<Detection> = per_frame.concat();

        self.store_debug_entry(&job, &detections, raw_responses, &model);

        if detections.is_empty() {
            return;
        }
        tracing::debug!(
            camera = %job.camera_id,
            count = detections.len(),
            classes = ?detections.iter().map(|d| &d.class_name).collect::<Vec<_>>(),
            "ollama detections"
        );

        let (classes, confidences) = deduplicate_by_class(&detections);
        // Everything reaching here already passed the client's allowed-class
        // and confidence-threshold filtering (`ollama::parse_detections`), so
        // the deduped classes are exactly the verdict — the same set that goes
        // into the detection store and the event upgrade.
        if let Some(ref tx) = self.mqtt_tx {
            send_event(
                tx,
                MqttEvent::Detections {
                    camera_id: job.camera_id.clone(),
                    sightings: build_sightings(
                        &classes,
                        &per_frame,
                        &job.crop_jpegs,
                        job.full_frame_jpeg.as_deref(),
                    ),
                },
            );
        }
        self.store_detections(&job, &classes, &confidences, &model);
        self.upgrade_covering_events(&job, &detections, &classes, &model)
            .await;
    }

    /// Store one detection row per (segment, class) so the event-assembly
    /// and API paths see exactly what they saw before the rework.
    fn store_detections(
        &self,
        job: &DetectionJob,
        classes: &[String],
        confidences: &[f32],
        model: &str,
    ) {
        // Thumbnail: the second frame when there is one (an inner frame of
        // the run reads better than the leading edge).
        let best_idx = if job.crop_jpegs.len() > 1 { 1 } else { 0 };
        let frame_jpeg = Arc::new(
            job.crop_jpegs
                .get(best_idx)
                .or_else(|| job.crop_jpegs.first())
                .cloned()
                .unwrap_or_default(),
        );

        for &seq in &job.seqs {
            for (class, &confidence) in classes.iter().zip(confidences) {
                self.detection_store.insert(
                    &job.camera_id,
                    DetectionEntry {
                        id: self.detection_store.next_id(),
                        segment_sequence: seq,
                        object_class: class.clone(),
                        confidence,
                        frame_jpeg: Arc::clone(&frame_jpeg),
                        backend: "ollama".to_string(),
                        model: model.to_string(),
                    },
                );
            }
        }
    }

    /// Hand this run's verdict to the registry and send the upgrades it asks
    /// for.
    ///
    /// The registry answers for every event the run covers, so the ones that
    /// come back are exactly those already in the writer's queue — an upgrade
    /// for them is guaranteed to arrive behind the write it refers to. An
    /// event the analyzer is still assembling comes back from nowhere: the
    /// verdict parks on its record, and the analyzer sends it once the write
    /// is queued, because a message this worker sent now would reach the
    /// writer first and find no file.
    async fn upgrade_covering_events(
        &self,
        job: &DetectionJob,
        detections: &[Detection],
        classes: &[String],
        model: &str,
    ) {
        let Some(ref registry) = self.event_registry else {
            return;
        };
        let verdict = Verdict {
            object_classes: classes.to_vec(),
            detections: detections
                .iter()
                .map(|d| DetectionDetail {
                    class: d.class_name.clone(),
                    confidence: d.confidence,
                })
                .collect(),
            backend: "ollama".to_string(),
            model: model.to_string(),
        };
        let targets = registry.deliver_verdict(&job.camera_id, &job.seqs, &verdict);
        if targets.is_empty() {
            return;
        }
        let Some(tx) = self.event_senders.get(&job.camera_id) else {
            return;
        };

        for target in targets {
            let upgrade = EventUpgrade::for_event(target, verdict.clone());
            if tx.send(WriterMessage::Upgrade(upgrade)).await.is_err() {
                tracing::warn!(camera = %job.camera_id, "warm writer gone, upgrade lost");
            }
        }
    }

    fn store_debug_entry(
        &self,
        job: &DetectionJob,
        detections: &[Detection],
        raw_responses: Vec<String>,
        model: &str,
    ) {
        let Some(ref debug_store) = self.debug_store else {
            return;
        };
        if job.crop_jpegs.is_empty() {
            return;
        }
        // Map ollama bboxes from crop space back to full-frame space.
        let ollama_rects: Vec<(String, f32, f32, f32, f32)> = detections
            .iter()
            .filter_map(|d| {
                let (x, y, w, h) = match (d.bbox, job.run_crop) {
                    (Some((bx, by, bw, bh)), Some((cx, cy, cw, ch))) => {
                        (cx + bx * cw, cy + by * ch, bw * cw, bh * ch)
                    }
                    (Some(b), None) => b,
                    (None, Some(c)) => c,
                    (None, None) => return None,
                };
                Some((d.class_name.clone(), x, y, w, h))
            })
            .collect();
        debug_store.insert(
            &job.camera_id,
            job.crop_jpegs.iter().cloned().map(Arc::new).collect(),
            raw_responses,
            model.to_string(),
            detections.len(),
            job.full_frame_jpeg.clone().map(Arc::new),
            job.motion_rects.clone(),
            job.run_crop,
            ollama_rects,
        );
    }
}

/// Deduplicate detections by class, keeping the best confidence per class.
fn deduplicate_by_class(detections: &[Detection]) -> (Vec<String>, Vec<f32>) {
    let mut best: HashMap<&str, f32> = HashMap::new();
    for d in detections {
        best.entry(d.class_name.as_str())
            .and_modify(|c| {
                if d.confidence > *c {
                    *c = d.confidence;
                }
            })
            .or_insert(d.confidence);
    }
    let classes: Vec<String> = best.keys().map(|k| k.to_string()).collect();
    let confidences: Vec<f32> = classes.iter().map(|c| best[c.as_str()]).collect();
    (classes, confidences)
}

/// The frame that best evidences each class: the index of the frame holding
/// that class's highest-confidence detection. `per_frame[i]` is the verdict for
/// crop `i`, so the returned index addresses `crop_jpegs` directly.
fn best_frame_per_class(per_frame: &[Vec<Detection>]) -> HashMap<&str, usize> {
    let mut best: HashMap<&str, (usize, f32)> = HashMap::new();
    for (idx, frame) in per_frame.iter().enumerate() {
        for d in frame {
            let entry = best
                .entry(d.class_name.as_str())
                .or_insert((idx, d.confidence));
            if d.confidence > entry.1 {
                *entry = (idx, d.confidence);
            }
        }
    }
    best.into_iter()
        .map(|(class, (idx, _))| (class, idx))
        .collect()
}

/// Pair every class of the verdict with the picture behind it, for the Home
/// Assistant bridge to publish retained: the crop the model classified when
/// picking that class, the run's full frame when the crops are gone, and
/// nothing at all when the job carried no frame.
fn build_sightings(
    classes: &[String],
    per_frame: &[Vec<Detection>],
    crop_jpegs: &[Vec<u8>],
    full_frame_jpeg: Option<&[u8]>,
) -> Vec<Sighting> {
    let best = best_frame_per_class(per_frame);
    classes
        .iter()
        .map(|class| Sighting {
            class: class.clone(),
            frame_jpeg: best
                .get(class.as_str())
                .and_then(|&idx| crop_jpegs.get(idx))
                .map(Vec::as_slice)
                .or(full_frame_jpeg)
                .map(<[u8]>::to_vec),
        })
        .collect()
}

/// Create the crop-job queue: a sender for the analyzers (clone one per
/// camera) and the shared queue for the worker. The queue closes when the
/// last sender is dropped, mirroring channel semantics.
///
/// The registry is held here because this is the one place that sees a job's
/// whole life on the queue: accepted (a verdict is now expected, and records
/// covering the job's sequences are kept until it comes) and, for the jobs
/// that never reach the worker, dropped at the cap.
pub fn detect_queue(
    event_registry: Option<EventRegistry>,
) -> (DetectQueueSender, Arc<DetectQueue>) {
    let queue = Arc::new(DetectQueue {
        state: Mutex::new(QueueState::default()),
        notify: Notify::new(),
        senders: AtomicUsize::new(1),
        event_registry,
    });
    (
        DetectQueueSender {
            queue: Arc::clone(&queue),
        },
        queue,
    )
}

/// Fair, freshness-first crop-job queue: one bounded stack per camera,
/// served round-robin with the newest job first.
pub struct DetectQueue {
    state: Mutex<QueueState>,
    notify: Notify,
    senders: AtomicUsize,
    event_registry: Option<EventRegistry>,
}

#[derive(Default)]
struct QueueState {
    /// Per-camera stacks: enqueued at the back, served from the back
    /// (newest first), evicted from the front (oldest) on overflow.
    jobs: HashMap<String, VecDeque<DetectionJob>>,
    /// Round-robin order, extended lazily on a camera's first job.
    cameras: Vec<String>,
    next_camera: usize,
    closed: bool,
}

impl QueueState {
    /// Newest job of the next camera (round-robin) that has one.
    fn pop_fair(&mut self) -> Option<DetectionJob> {
        for _ in 0..self.cameras.len() {
            let camera = &self.cameras[self.next_camera];
            self.next_camera = (self.next_camera + 1) % self.cameras.len();
            if let Some(job) = self.jobs.get_mut(camera).and_then(VecDeque::pop_back) {
                return Some(job);
            }
        }
        None
    }
}

impl DetectQueue {
    /// Next job by fairness policy, or `None` once every sender is gone and
    /// the queue is drained. Single consumer.
    pub async fn recv(&self) -> Option<DetectionJob> {
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self.lock_state();
                if let Some(job) = state.pop_fair() {
                    return Some(job);
                }
                if state.closed {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// Every mutation under this lock is a single-step queue edit, so a
    /// poisoned lock is recoverable for the same reason as in [`crate::locks`].
    fn lock_state(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub struct DetectQueueSender {
    queue: Arc<DetectQueue>,
}

impl DetectQueueSender {
    /// Enqueue a job without ever blocking the analyzer. The new job is
    /// always accepted; a camera past its cap loses its OLDEST queued job
    /// instead — the motion event still persists, only that object upgrade
    /// is lost.
    ///
    /// The registry is told before the job is queued and after one is dropped,
    /// in that order, so a record covering these sequences is never held open
    /// by a job that has already left and never forgotten while one is still
    /// on the queue.
    pub fn send(&self, mut job: DetectionJob) {
        if let Some(ref registry) = self.queue.event_registry {
            job.verdict_id = registry.expect_verdict(&job.camera_id, &job.seqs);
        }
        let dropped = {
            let mut state = self.queue.lock_state();
            if !state.jobs.contains_key(&job.camera_id) {
                state.cameras.push(job.camera_id.clone());
            }
            let stack = state.jobs.entry(job.camera_id.clone()).or_default();
            stack.push_back(job);
            if stack.len() > DETECT_QUEUE_PER_CAMERA_CAP {
                stack.pop_front()
            } else {
                None
            }
        };
        if let Some(old) = dropped {
            if let Some(ref registry) = self.queue.event_registry {
                registry.verdict_settled(&old.camera_id, old.verdict_id);
            }
            tracing::warn!(
                camera = %old.camera_id,
                first_seq = old.seqs.first().copied().unwrap_or_default(),
                "camera at detection queue cap, dropped its oldest crop job (motion event still recorded)"
            );
        }
        self.queue.notify.notify_one();
    }
}

impl Clone for DetectQueueSender {
    fn clone(&self) -> Self {
        self.queue.senders.fetch_add(1, Ordering::Relaxed);
        Self {
            queue: Arc::clone(&self.queue),
        }
    }
}

impl Drop for DetectQueueSender {
    fn drop(&mut self) {
        if self.queue.senders.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.queue.lock_state().closed = true;
            self.queue.notify.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(camera_id: &str, seqs: Vec<u64>) -> DetectionJob {
        DetectionJob {
            camera_id: camera_id.to_string(),
            seqs,
            crop_jpegs: vec![vec![0xff]],
            full_frame_jpeg: None,
            motion_rects: Vec::new(),
            run_crop: None,
            verdict_id: None,
        }
    }

    #[tokio::test]
    async fn queue_serves_cameras_round_robin_newest_first() {
        let (tx, queue) = detect_queue(None);
        tx.send(job("spammy", vec![1]));
        tx.send(job("spammy", vec![2]));
        tx.send(job("spammy", vec![3]));
        tx.send(job("quiet", vec![10]));
        drop(tx);

        let order: Vec<Vec<u64>> = [
            queue.recv().await.unwrap().seqs,
            queue.recv().await.unwrap().seqs,
            queue.recv().await.unwrap().seqs,
            queue.recv().await.unwrap().seqs,
        ]
        .into();
        // The quiet camera is not drowned out, and within a camera the
        // newest job comes first.
        assert_eq!(order, vec![vec![3], vec![10], vec![2], vec![1]]);
        assert!(queue.recv().await.is_none());
    }

    #[tokio::test]
    async fn overflowing_camera_loses_its_oldest_job() {
        let (tx, queue) = detect_queue(None);
        for seq in 0..=DETECT_QUEUE_PER_CAMERA_CAP as u64 {
            tx.send(job("cam", vec![seq]));
        }
        drop(tx);

        let mut seqs = Vec::new();
        while let Some(job) = queue.recv().await {
            seqs.push(job.seqs[0]);
        }
        // Job 0 was evicted; the rest arrive newest first.
        let expected: Vec<u64> = (1..=DETECT_QUEUE_PER_CAMERA_CAP as u64).rev().collect();
        assert_eq!(seqs, expected);
    }

    /// A job sitting on the queue is a verdict still to come, and the record
    /// it will land on has to survive until then. The queue is what says so,
    /// as it accepts the job — without that, the very next event to close
    /// forgets the record, and a verdict arriving minutes later against a slow
    /// model finds nothing to upgrade.
    #[tokio::test]
    async fn a_queued_job_keeps_alive_the_record_its_verdict_will_land_on() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        let (tx, _queue) = detect_queue(Some(registry.clone()));

        tx.send(job("cam", vec![0, 1]));
        registry.open("cam", 0, 1, false).commit(1000, 5000, false);

        // A later event closing is when settled records are forgotten. This
        // one is not settled: its job has not even been served yet.
        registry
            .open("cam", 100, 100, false)
            .commit(2000, 5000, false);
        assert_eq!(
            registry.held("cam"),
            2,
            "the record was forgotten while its crop job was still on the queue"
        );
    }

    /// The registry keeps an event's record alive until every crop job that
    /// could still classify it has come back. A job dropped at the cap never
    /// will, so the drop has to report it — otherwise the camera least able to
    /// afford it, the one at its queue cap, accumulates a record per dropped
    /// job for the life of the process.
    #[tokio::test]
    async fn a_job_dropped_at_the_cap_releases_the_records_it_was_holding() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        let (tx, _queue) = detect_queue(Some(registry.clone()));

        tx.send(job("cam", vec![0]));
        registry.open("cam", 0, 0, false).commit(1000, 5000, false);
        assert_eq!(registry.held("cam"), 1);

        // Fill the camera's stack past its cap: the oldest job — the only one
        // that covered seq 0 — falls off.
        for seq in 1..=DETECT_QUEUE_PER_CAMERA_CAP as u64 + 1 {
            tx.send(job("cam", vec![seq]));
        }

        // Nothing can classify that first event any more, so the next event to
        // arrive forgets it instead of holding it forever.
        registry
            .open("cam", 100, 100, false)
            .commit(2000, 5000, false);
        assert_eq!(
            registry.held("cam"),
            1,
            "a job that was dropped unprocessed went on pinning its records"
        );
    }

    /// The same obligation on the worker's side, on the path with nothing to
    /// report: a job whose frames all failed, or that the model saw nothing
    /// in, leaves through the early return in `classify_job` — and still has
    /// to release what it was holding.
    #[tokio::test]
    async fn a_job_the_model_answered_nothing_for_is_still_reported_back() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        // Nothing is listening on this port, so every frame of the job fails
        // and it reaches the end of processing with no verdict at all.
        let client = OllamaClient::new(
            "http://127.0.0.1:1",
            "test-model",
            1,
            0.5,
            vec!["person".to_string()],
            None,
        )
        .expect("client");
        let worker = DetectionWorker::new(
            client,
            DetectionStore::new(&["cam".to_string()]),
            None,
            Some(registry.clone()),
            HashMap::new(),
            None,
        );

        let mut job = job("cam", vec![0]);
        job.verdict_id = registry.expect_verdict("cam", &[0]);
        registry.open("cam", 0, 0, false).commit(1000, 5000, false);
        worker.process_job(job).await;

        registry
            .open("cam", 100, 100, false)
            .commit(2000, 5000, false);
        assert_eq!(
            registry.held("cam"),
            1,
            "a job that produced no verdict never released the record it was holding"
        );
    }

    #[tokio::test]
    async fn queue_closes_when_last_sender_drops() {
        let (tx, queue) = detect_queue(None);
        let tx2 = tx.clone();
        tx.send(job("cam", vec![1]));
        drop(tx);

        // A sender clone still exists: the queued job is served.
        assert_eq!(queue.recv().await.unwrap().seqs, vec![1]);

        let recv = tokio::spawn(async move { queue.recv().await.is_none() });
        drop(tx2);
        assert!(recv.await.unwrap());
    }

    #[test]
    fn dedup_keeps_best_confidence_per_class() {
        let detections = vec![
            Detection {
                class_name: "person".to_string(),
                confidence: 0.7,
                bbox: None,
            },
            Detection {
                class_name: "person".to_string(),
                confidence: 0.9,
                bbox: None,
            },
            Detection {
                class_name: "car".to_string(),
                confidence: 0.6,
                bbox: None,
            },
        ];
        let (classes, confidences) = deduplicate_by_class(&detections);
        assert_eq!(classes.len(), 2);
        let person_idx = classes.iter().position(|c| c == "person").unwrap();
        assert!((confidences[person_idx] - 0.9).abs() < 0.001);
    }

    fn detection(class: &str, confidence: f32) -> Detection {
        Detection {
            class_name: class.to_string(),
            confidence,
            bbox: None,
        }
    }

    #[test]
    fn best_frame_is_the_argmax_of_confidence_per_class() {
        let per_frame = vec![
            vec![detection("person", 0.6), detection("cat", 0.9)],
            vec![],
            vec![detection("person", 0.8), detection("cat", 0.7)],
        ];
        let best = best_frame_per_class(&per_frame);
        assert_eq!(best["person"], 2);
        assert_eq!(best["cat"], 0);
        // A class nobody saw has no frame.
        assert!(!best.contains_key("car"));
    }

    #[test]
    fn sighting_carries_the_crop_the_class_was_seen_in() {
        let per_frame = vec![
            vec![detection("person", 0.6), detection("cat", 0.9)],
            vec![detection("person", 0.8)],
        ];
        let crops = vec![vec![0xaa], vec![0xbb]];
        let sightings = build_sightings(
            &["person".to_string(), "cat".to_string()],
            &per_frame,
            &crops,
            Some(&[0xcc]),
        );
        assert_eq!(
            sightings,
            vec![
                Sighting {
                    class: "person".to_string(),
                    frame_jpeg: Some(vec![0xbb]),
                },
                Sighting {
                    class: "cat".to_string(),
                    frame_jpeg: Some(vec![0xaa]),
                },
            ]
        );
    }

    #[test]
    fn sighting_falls_back_to_the_full_frame_then_to_nothing() {
        let classes = ["person".to_string()];
        // No crops to point at: the full frame stands in.
        let sightings = build_sightings(&classes, &[], &[], Some(&[0xcc]));
        assert_eq!(sightings[0].frame_jpeg, Some(vec![0xcc]));

        // Neither: nothing is published for this sighting.
        let sightings = build_sightings(&classes, &[], &[], None);
        assert_eq!(sightings[0].class, "person");
        assert_eq!(sightings[0].frame_jpeg, None);
    }
}
