//! Global object-detection worker.

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

/// Jobs held per camera before the oldest is evicted.
pub const DETECT_QUEUE_PER_CAMERA_CAP: usize = 32;

/// At most this many frames of a run are sent to the model.
const MAX_FRAMES_PER_RUN: usize = 4;

/// One crop job: everything the worker needs to classify a motion run and
/// attribute the verdict back to segments and events.
pub struct DetectionJob {
    pub camera_id: String,
    /// Segment sequences of the motion run this job covers (ascending).
    pub seqs: Vec<u64>,
    /// Cropped frames, JPEG-encoded, at most 4 are used. Held by handle
    /// because the debug store outlives the job: it takes a share of these
    /// bytes rather than a second copy of them.
    pub crop_jpegs: Vec<Arc<Vec<u8>>>,
    /// A full (uncropped) frame for the debug overlay, which is what reads it — so the
    /// analyzer encodes one only while somebody is watching that view, and this is `None` the
    /// rest of the time.
    pub full_frame_jpeg: Option<Arc<Vec<u8>>>,
    /// Individual motion boxes in normalized full-frame coords.
    pub motion_rects: Vec<(f32, f32, f32, f32)>,
    /// The union crop region the frames were cropped to, normalized.
    pub run_crop: Option<(f32, f32, f32, f32)>,
    /// The registry's handle on this job, stamped by [`DetectQueueSender::send`] as the job is
    /// accepted and handed back when the job leaves the system — answered here, or dropped at
    /// the queue cap.
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

    /// Worker main loop. Exits when every job sender (the analyzers) is gone and the queue is
    /// drained; at shutdown the task is aborted instead — pending jobs and even an in-flight
    /// request are droppable by design.
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

    /// Settle the registry expectation exactly once after classification, including every
    /// failure or empty-verdict path.
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
        // Parsing already applied class and confidence filters, so this deduped set is the
        // complete verdict used by every consumer.
        if let Some(ref tx) = self.mqtt_tx {
            send_event(
                tx,
                MqttEvent::Detections {
                    camera_id: job.camera_id.clone(),
                    sightings: build_sightings(
                        &classes,
                        &per_frame,
                        &job.crop_jpegs,
                        job.full_frame_jpeg.as_ref().map(|f| f.as_slice()),
                    ),
                },
            );
        }
        self.store_detections(&job, &classes, &confidences, &model);
        self.upgrade_covering_events(&job, &detections, &classes, &model);
    }

    /// Store one detection row per segment and class for event assembly and the API.
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
        let frame_jpeg = job
            .crop_jpegs
            .get(best_idx)
            .or_else(|| job.crop_jpegs.first())
            .map(Arc::clone)
            .unwrap_or_default();

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

    /// Hand this run's verdict to the registry and send the upgrades it asks for — offering
    /// them to the writer rather than waiting for room.
    fn upgrade_covering_events(
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
            match tx.try_send(WriterMessage::Upgrade(upgrade)) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => tracing::warn!(
                    camera = %job.camera_id,
                    start_pts_ns = target.start_pts_ns,
                    "warm writer backlogged, dropped an object upgrade (the event stays a \
                     movement event and expires on the shorter movement retention; its \
                     detections are readable for as long as its footage is still in the \
                     hot buffer, and nowhere after that)"
                ),
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    tracing::warn!(camera = %job.camera_id, "warm writer gone, upgrade lost");
                }
            }
        }
    }

    /// Publish this run to the detector's debug view — but only while somebody has that view
    /// open.
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
        if !debug_store.wanted(&job.camera_id) {
            return;
        }
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
            // Both of these clone `Arc` handles, not JPEGs.
            job.crop_jpegs.clone(),
            raw_responses,
            model.to_string(),
            detections.len(),
            job.full_frame_jpeg.clone(),
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

/// Pair every class of the verdict with the picture behind it, for the Home Assistant bridge to
/// publish retained: the crop the model classified when picking that class, the run's full
/// frame when the crops are gone, and nothing at all when the job carried no frame.
fn build_sightings(
    classes: &[String],
    per_frame: &[Vec<Detection>],
    crop_jpegs: &[Arc<Vec<u8>>],
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
                .map(|jpeg| jpeg.as_slice())
                .or(full_frame_jpeg)
                .map(<[u8]>::to_vec),
        })
        .collect()
}

/// Create the crop-job queue: a sender for the analyzers (clone one per camera) and the shared
/// queue for the worker. The queue closes when the last sender is dropped, mirroring channel
/// semantics.
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
    /// Enqueue a job without ever blocking the analyzer. The new job is always accepted; a
    /// camera past its cap loses its OLDEST queued job instead — the motion event still
    /// persists, only that object upgrade is lost.
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
            crop_jpegs: vec![Arc::new(vec![0xff])],
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
        let expected: Vec<u64> = (1..=DETECT_QUEUE_PER_CAMERA_CAP as u64).rev().collect();
        assert_eq!(seqs, expected);
    }

    #[tokio::test]
    async fn a_queued_job_keeps_alive_the_record_its_verdict_will_land_on() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        let (tx, _queue) = detect_queue(Some(registry.clone()));

        tx.send(job("cam", vec![0, 1]));
        registry.open("cam", 0, 1, false).commit(1000, 5000, false);

        registry
            .open("cam", 100, 100, false)
            .commit(2000, 5000, false);
        assert_eq!(
            registry.held("cam"),
            2,
            "the record was forgotten while its crop job was still on the queue"
        );
    }

    #[tokio::test]
    async fn a_job_dropped_at_the_cap_releases_the_records_it_was_holding() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        let (tx, _queue) = detect_queue(Some(registry.clone()));

        tx.send(job("cam", vec![0]));
        registry.open("cam", 0, 0, false).commit(1000, 5000, false);
        assert_eq!(registry.held("cam"), 1);

        for seq in 1..=DETECT_QUEUE_PER_CAMERA_CAP as u64 + 1 {
            tx.send(job("cam", vec![seq]));
        }

        registry
            .open("cam", 100, 100, false)
            .commit(2000, 5000, false);
        assert_eq!(
            registry.held("cam"),
            1,
            "a job that was dropped unprocessed went on pinning its records"
        );
    }

    #[tokio::test]
    async fn a_job_the_model_answered_nothing_for_is_still_reported_back() {
        let registry = EventRegistry::new(&["cam".to_string()]);
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

    async fn ollama_that_always_sees_a_person() -> String {
        let app = axum::Router::new()
            .route(
                "/api/chat",
                axum::routing::post(|| async {
                    axum::Json(serde_json::json!({
                        "message": {
                            "content": "{\"detections\":[{\"class\":\"person\",\
                                        \"confidence\":0.9,\"x\":0.1,\"y\":0.1,\
                                        \"w\":0.2,\"h\":0.2}]}"
                        }
                    }))
                }),
            )
            .route(
                "/api/tags",
                axum::routing::get(|| async {
                    axum::Json(serde_json::json!({"models": [{"name": "test-model"}]}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn worker_for(
        url: &str,
        cameras: &[&str],
        registry: &EventRegistry,
        event_senders: HashMap<String, mpsc::Sender<WriterMessage>>,
    ) -> DetectionWorker {
        let ids: Vec<String> = cameras.iter().map(|c| (*c).to_string()).collect();
        let client = OllamaClient::new(url, "test-model", 5, 0.5, vec!["person".to_string()], None)
            .expect("client");
        DetectionWorker::new(
            client,
            DetectionStore::new(&ids),
            None,
            Some(registry.clone()),
            event_senders,
            None,
        )
    }

    fn filler() -> WriterMessage {
        WriterMessage::Upgrade(EventUpgrade::for_event(
            crate::storage::UpgradeTarget {
                start_pts_ns: u64::MAX,
                duration_ms: 0,
                continues: false,
            },
            Verdict {
                object_classes: Vec::new(),
                detections: Vec::new(),
                backend: String::new(),
                model: String::new(),
            },
        ))
    }

    fn a_writer_that_is_not_draining(
    ) -> (mpsc::Sender<WriterMessage>, mpsc::Receiver<WriterMessage>) {
        let (tx, rx) = mpsc::channel(1);
        tx.try_send(filler()).expect("a fresh channel has room");
        (tx, rx)
    }

    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(10);

    async fn next_message(rx: &mut mpsc::Receiver<WriterMessage>, expected: &str) -> WriterMessage {
        match tokio::time::timeout(PATIENCE, rx.recv()).await {
            Ok(Some(message)) => message,
            Ok(None) => panic!("the writer channel closed before {expected}"),
            Err(_) => panic!("{expected} never arrived"),
        }
    }

    #[tokio::test]
    async fn a_writer_that_is_not_draining_does_not_stall_the_camera_behind_it() {
        let url = ollama_that_always_sees_a_person().await;
        let registry = EventRegistry::new(&["blocked".to_string(), "next".to_string()]);
        let (blocked_tx, mut blocked_rx) = a_writer_that_is_not_draining();
        let (next_tx, mut next_rx) = mpsc::channel(1);
        let worker = worker_for(
            &url,
            &["blocked", "next"],
            &registry,
            HashMap::from([
                ("blocked".to_string(), blocked_tx),
                ("next".to_string(), next_tx),
            ]),
        );

        let (tx, queue) = detect_queue(Some(registry.clone()));
        tx.send(job("blocked", vec![0]));
        tx.send(job("next", vec![0]));
        drop(tx);
        registry
            .open("blocked", 0, 0, false)
            .commit(1000, 5000, false);
        registry.open("next", 0, 0, false).commit(2000, 5000, false);

        let worker_task = tokio::spawn(worker.run(queue));
        match next_message(&mut next_rx, "the second camera's upgrade").await {
            WriterMessage::Upgrade(upgrade) => assert_eq!(upgrade.start_pts_ns, 2000),
            WriterMessage::Event(_) => panic!("the worker sent an event"),
        }

        match blocked_rx.try_recv() {
            Ok(WriterMessage::Upgrade(held)) => assert_eq!(held.start_pts_ns, u64::MAX),
            _ => panic!("the filler holding the channel full went missing"),
        }
        assert!(
            blocked_rx.try_recv().is_err(),
            "the upgrade found room after all, so nothing here was ever blocked"
        );
        worker_task.await.expect("the worker panicked");
    }

    #[tokio::test]
    async fn a_dropped_upgrade_still_settles_what_the_registry_expects() {
        let url = ollama_that_always_sees_a_person().await;
        let registry = EventRegistry::new(&["cam".to_string()]);
        let (tx, mut rx) = a_writer_that_is_not_draining();
        let worker = worker_for(
            &url,
            &["cam"],
            &registry,
            HashMap::from([("cam".to_string(), tx)]),
        );

        let mut job = job("cam", vec![0]);
        job.verdict_id = registry.expect_verdict("cam", &[0]);
        registry.open("cam", 0, 0, false).commit(1000, 5000, false);
        worker.process_job(job).await;

        match rx.try_recv() {
            Ok(WriterMessage::Upgrade(held)) => assert_eq!(held.start_pts_ns, u64::MAX),
            _ => panic!("the filler holding the channel full went missing"),
        }
        assert!(rx.try_recv().is_err(), "the upgrade was queued after all");

        registry.open("cam", 0, 0, false).commit(2000, 5000, false);
        registry
            .open("cam", 100, 100, false)
            .commit(3000, 5000, false);
        assert_eq!(
            registry.held("cam"),
            1,
            "the dropped upgrade left its verdict outstanding, pinning records for the life \
             of the process"
        );
    }

    #[derive(Clone)]
    struct Logged {
        level: tracing::Level,
        message: String,
        fields: Vec<(String, String)>,
    }

    impl Logged {
        fn field(&self, name: &str) -> Option<&str> {
            self.fields
                .iter()
                .find(|(recorded, _)| recorded == name)
                .map(|(_, value)| value.as_str())
        }
    }

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<Logged>>>);

    #[derive(Default)]
    struct Fields {
        message: String,
        rest: Vec<(String, String)>,
    }

    impl Fields {
        fn put(&mut self, field: &tracing::field::Field, value: String) {
            if field.name() == "message" {
                self.message = value;
            } else {
                self.rest.push((field.name().to_string(), value));
            }
        }
    }

    impl tracing::field::Visit for Fields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.put(field, format!("{value:?}"));
        }

        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            self.put(field, value.to_string());
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Captured {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut fields = Fields::default();
            event.record(&mut fields);
            self.0.lock().expect("log capture poisoned").push(Logged {
                level: *event.metadata().level(),
                message: fields.message,
                fields: fields.rest,
            });
        }
    }

    fn capture(body: impl FnOnce()) -> Vec<Logged> {
        use tracing_subscriber::layer::SubscriberExt;
        let captured = Captured::default();
        let subscriber = tracing_subscriber::registry().with(captured.clone());
        tracing::subscriber::with_default(subscriber, body);
        let logs = captured.0.lock().expect("log capture poisoned").clone();
        logs
    }

    #[test]
    fn a_dropped_upgrade_is_reported_the_way_this_file_reports_its_other_drops() {
        let registry = EventRegistry::new(&["cam".to_string()]);
        let (tx, rx) = a_writer_that_is_not_draining();
        let worker = worker_for(
            "http://127.0.0.1:1",
            &["cam"],
            &registry,
            HashMap::from([("cam".to_string(), tx)]),
        );
        let seen = [detection("person", 0.9)];
        let classes = ["person".to_string()];

        registry.open("cam", 0, 0, false).commit(1000, 5000, false);
        let backlogged = capture(|| {
            worker.upgrade_covering_events(&job("cam", vec![0]), &seen, &classes, "test-model");
        });
        let [warning] = &backlogged[..] else {
            panic!(
                "a lost object upgrade was reported {} times, not once",
                backlogged.len()
            );
        };
        assert_eq!(
            warning.level,
            tracing::Level::WARN,
            "a lost object upgrade was reported below the production log level"
        );
        assert!(
            warning.message.contains("dropped an object upgrade"),
            "the backlogged writer was not named as the reason: {}",
            warning.message
        );
        assert_eq!(
            warning.field("camera"),
            Some("cam"),
            "the warning does not carry the camera that lost the upgrade"
        );
        assert_eq!(
            warning.field("start_pts_ns"),
            Some("1000"),
            "the warning does not carry the event that lost the upgrade"
        );

        drop(rx);
        registry.open("cam", 1, 1, false).commit(2000, 5000, false);
        let gone = capture(|| {
            worker.upgrade_covering_events(&job("cam", vec![1]), &seen, &classes, "test-model");
        });
        let [warning] = &gone[..] else {
            panic!(
                "a vanished writer was reported {} times, not once",
                gone.len()
            );
        };
        assert_eq!(warning.level, tracing::Level::WARN);
        assert!(
            warning.message.contains("warm writer gone"),
            "a vanished writer was reported as a backlogged one: {}",
            warning.message
        );
        assert_eq!(warning.field("camera"), Some("cam"));
    }

    fn worker_with(debug_store: &DetectionDebugStore) -> DetectionWorker {
        let client = OllamaClient::new(
            "http://127.0.0.1:1",
            "test-model",
            1,
            0.5,
            vec!["person".to_string()],
            None,
        )
        .expect("client");
        DetectionWorker::new(
            client,
            DetectionStore::new(&["cam".to_string()]),
            Some(debug_store.clone()),
            None,
            HashMap::new(),
            None,
        )
    }

    fn job_with_pictures(crop: &Arc<Vec<u8>>, full: &Arc<Vec<u8>>) -> DetectionJob {
        let mut job = job("cam", vec![0]);
        job.crop_jpegs = vec![Arc::clone(crop)];
        job.full_frame_jpeg = Some(Arc::clone(full));
        job
    }

    #[test]
    fn neither_debug_picture_is_kept_until_somebody_is_watching() {
        let debug_store = DetectionDebugStore::new(&["cam".to_string()]);
        let worker = worker_with(&debug_store);
        let (crop, full) = (Arc::new(vec![0xaa]), Arc::new(vec![0xbb]));

        worker.store_debug_entry(&job_with_pictures(&crop, &full), &[], Vec::new(), "m");
        assert_eq!(
            debug_store.stored("cam"),
            0,
            "kept a run's frames for a debug view nobody has open"
        );

        debug_store.list("cam");
        worker.store_debug_entry(&job_with_pictures(&crop, &full), &[], Vec::new(), "m");
        assert_eq!(debug_store.stored("cam"), 1);
        let id = debug_store.list("cam")[0].id;
        assert!(
            debug_store.get_frame_jpeg("cam", id, 0).is_some(),
            "the crops never reached the open view"
        );
        assert!(
            debug_store.get_full_frame_jpeg("cam", id).is_some(),
            "the full frame never reached the open view"
        );
    }

    #[test]
    fn the_debug_entry_shares_the_jobs_frames_rather_than_copying_them() {
        let debug_store = DetectionDebugStore::new(&["cam".to_string()]);
        let worker = worker_with(&debug_store);
        let (crop, full) = (Arc::new(vec![0xaa]), Arc::new(vec![0xbb]));

        debug_store.list("cam");
        worker.store_debug_entry(&job_with_pictures(&crop, &full), &[], Vec::new(), "m");

        let id = debug_store.list("cam")[0].id;
        assert!(
            Arc::ptr_eq(&debug_store.get_frame_jpeg("cam", id, 0).unwrap(), &crop),
            "the crop was copied into the store instead of shared"
        );
        assert!(
            Arc::ptr_eq(&debug_store.get_full_frame_jpeg("cam", id).unwrap(), &full),
            "the full frame was copied into the store instead of shared"
        );
    }

    #[tokio::test]
    async fn queue_closes_when_last_sender_drops() {
        let (tx, queue) = detect_queue(None);
        let tx2 = tx.clone();
        tx.send(job("cam", vec![1]));
        drop(tx);

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
        assert!(!best.contains_key("car"));
    }

    #[test]
    fn sighting_carries_the_crop_the_class_was_seen_in() {
        let per_frame = vec![
            vec![detection("person", 0.6), detection("cat", 0.9)],
            vec![detection("person", 0.8)],
        ];
        let crops = vec![Arc::new(vec![0xaa]), Arc::new(vec![0xbb])];
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
        let sightings = build_sightings(&classes, &[], &[], Some(&[0xcc]));
        assert_eq!(sightings[0].frame_jpeg, Some(vec![0xcc]));

        let sightings = build_sightings(&classes, &[], &[], None);
        assert_eq!(sightings[0].class, "person");
        assert_eq!(sightings[0].frame_jpeg, None);
    }
}
