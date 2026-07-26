//! Global object-detection worker.
//!
//! ONE async task serves all cameras. Analyzers flag motion, drop a crop job
//! on a small bounded queue, and continue immediately — motion detection
//! never stalls on the vision model. The worker drains the queue strictly
//! serially: at most ONE in-flight request to Ollama at any time, across all
//! cameras, and the frames of a single run are sent one after another (the
//! production GPU is old and degrades badly under parallel load).
//!
//! When the queue is full the job is dropped with a warning. That is
//! explicitly acceptable: the motion event still persists to `movements/`;
//! only the object upgrade is lost.
//!
//! Verdicts are handled in two ways:
//! - always written to the [`DetectionStore`], where the event-assembly path
//!   and the API read them exactly as before;
//! - if the covering event already reached disk as a movement event (looked
//!   up in the [`EventRegistry`]), an upgrade message is sent to that
//!   camera's warm writer, which owns all file mutations. See
//!   `storage::event_registry` for the race analysis with event assembly.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::buffer::warm::{EventUpgrade, WriterMessage};
use crate::mqtt::{send_event, MqttEvent};
use crate::storage::warm_index::DetectionDetail;
use crate::storage::{DetectionDebugStore, DetectionEntry, DetectionStore, EventRegistry};

use super::ollama::{Detection, OllamaClient};

/// Depth of the global crop-job queue. Small on purpose: a slow GPU cannot
/// work through a backlog anyway, and a dropped job only costs the object
/// upgrade of an already-persisted motion event. During sustained motion the
/// queue simply stays full and later jobs from the same run take the slots.
pub const DETECT_QUEUE_CAPACITY: usize = 8;

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
    /// gone; at shutdown the task is aborted instead — pending jobs and even
    /// an in-flight request are droppable by design.
    pub async fn run(self, mut rx: mpsc::Receiver<DetectionJob>) {
        // Surface a typo'd model name in seconds rather than as a string of
        // silent detection failures.
        self.client.check_models().await;
        tracing::info!(model = %self.client.model(), "detection worker started (serial, one in-flight request)");
        while let Some(job) = rx.recv().await {
            self.process_job(job).await;
        }
        tracing::info!("detection worker stopped");
    }

    async fn process_job(&self, job: DetectionJob) {
        let mut detections: Vec<Detection> = Vec::new();
        let mut raw_responses = Vec::new();
        let mut model = self.client.model().to_string();

        // Strictly sequential: each frame waits for the previous response.
        for jpeg in job.crop_jpegs.iter().take(MAX_FRAMES_PER_RUN) {
            match self.client.detect_jpeg(jpeg).await {
                Ok(result) => {
                    detections.extend(result.detections);
                    raw_responses.push(result.raw_response);
                    model = result.model;
                }
                Err(e) => {
                    // A timeout or server error costs only the upgrade.
                    tracing::warn!(camera = %job.camera_id, error = %e, "frame detection failed");
                    raw_responses.push(format!("ERROR: {e}"));
                }
            }
        }

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
                    classes: classes.clone(),
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

/// Enqueue a job without ever blocking the analyzer. A full queue drops the
/// job with a warning — the motion event still persists, only the object
/// upgrade is lost.
pub fn enqueue_job(tx: &mpsc::Sender<DetectionJob>, job: DetectionJob) -> bool {
    match tx.try_send(job) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(job)) => {
            tracing::warn!(
                camera = %job.camera_id,
                first_seq = job.seqs.first().copied().unwrap_or_default(),
                "detection queue full, dropping crop job (motion event still recorded)"
            );
            false
        }
        Err(mpsc::error::TrySendError::Closed(job)) => {
            tracing::warn!(camera = %job.camera_id, "detection worker gone, dropping crop job");
            false
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
    async fn enqueue_drops_when_queue_full() {
        let (tx, mut rx) = mpsc::channel(1);
        assert!(enqueue_job(&tx, job("cam", vec![1])));
        // Queue full: second job dropped, analyzer never blocks.
        assert!(!enqueue_job(&tx, job("cam", vec![2])));
        // Only the first job is in the queue.
        assert_eq!(rx.recv().await.unwrap().seqs, vec![1]);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn enqueue_drops_when_worker_gone() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        assert!(!enqueue_job(&tx, job("cam", vec![1])));
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
}
