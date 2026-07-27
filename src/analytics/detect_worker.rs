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
//! - if the covering event already reached disk as a movement event (looked
//!   up in the [`EventRegistry`]), an upgrade message is sent to that
//!   camera's warm writer, which owns all file mutations. See
//!   `storage::event_registry` for the race analysis with event assembly.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tokio::sync::{mpsc, Notify};

use crate::buffer::warm::{EventUpgrade, WriterMessage};
use crate::mqtt::{send_event, MqttEvent, Sighting};
use crate::storage::warm_index::DetectionDetail;
use crate::storage::{DetectionDebugStore, DetectionEntry, DetectionStore, EventRegistry};

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

    async fn process_job(&self, job: DetectionJob) {
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

    /// If any covering event already reached disk movement-classified,
    /// request its post-hoc upgrade from the owning warm writer.
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
        let records = registry.claim_movement_events(&job.camera_id, &job.seqs);
        if records.is_empty() {
            return;
        }
        let Some(tx) = self.event_senders.get(&job.camera_id) else {
            return;
        };

        let details: Vec<DetectionDetail> = detections
            .iter()
            .map(|d| DetectionDetail {
                class: d.class_name.clone(),
                confidence: d.confidence,
            })
            .collect();

        for record in records {
            let upgrade = EventUpgrade {
                start_pts_ns: record.start_pts_ns,
                duration_ms: record.duration_ms,
                object_classes: classes.to_vec(),
                detections: details.clone(),
                backend: "ollama".to_string(),
                model: model.to_string(),
                continues: record.continues,
            };
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
pub fn detect_queue() -> (DetectQueueSender, Arc<DetectQueue>) {
    let queue = Arc::new(DetectQueue {
        state: Mutex::new(QueueState::default()),
        notify: Notify::new(),
        senders: AtomicUsize::new(1),
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
    pub fn send(&self, job: DetectionJob) {
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
        }
    }

    #[tokio::test]
    async fn queue_serves_cameras_round_robin_newest_first() {
        let (tx, queue) = detect_queue();
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
        let (tx, queue) = detect_queue();
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

    #[tokio::test]
    async fn queue_closes_when_last_sender_drops() {
        let (tx, queue) = detect_queue();
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
