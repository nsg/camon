use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;

use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::storage::{DetectionStore, MotionEntry, MotionStore};

use super::decoder::{DetectionDecoder, FrameDecoder};
use super::motion::{MotionDetector, ScoreHistogram};
use super::object::ObjectDetector;

const MOTION_PERCENTILE: f32 = 0.90;
const DEFAULT_MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);

pub struct MotionAnalyzer {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    detection_decoder: Option<DetectionDecoder>,
    object_detector: Option<ObjectDetector>,
    last_processed: u64,
    score_histogram: ScoreHistogram,
}

impl MotionAnalyzer {
    fn new(
        camera_id: String,
        buffer: Arc<RwLock<HotBuffer>>,
        motion_store: MotionStore,
        detection_store: Option<DetectionStore>,
        object_detector: Option<ObjectDetector>,
        config: AnalyticsConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let detector = MotionDetector::new()?;
        let decoder = FrameDecoder::new(config.sample_fps)?;

        let detection_decoder = if object_detector.is_some() {
            Some(DetectionDecoder::new(config.sample_fps)?)
        } else {
            None
        };

        let last_processed = motion_store
            .last_sequence(&camera_id)
            .map(|s| s + 1)
            .unwrap_or(0);

        let score_histogram = ScoreHistogram::new(MOTION_PERCENTILE, DEFAULT_MOTION_THRESHOLD);

        Ok(Self {
            camera_id,
            buffer,
            motion_store,
            detection_store,
            config,
            detector,
            decoder,
            detection_decoder,
            object_detector,
            last_processed,
            score_histogram,
        })
    }

    fn run(mut self, shutdown: Arc<AtomicBool>) {
        tracing::info!(camera = %self.camera_id, "motion analyzer started");

        while !shutdown.load(Ordering::Relaxed) {
            if !self.decoder.is_alive() {
                tracing::warn!(camera = %self.camera_id, "decoder process died, restarting");
                match FrameDecoder::new(self.config.sample_fps) {
                    Ok(d) => self.decoder = d,
                    Err(e) => {
                        tracing::error!(camera = %self.camera_id, error = %e, "failed to restart decoder");
                        thread::sleep(Duration::from_secs(5));
                        continue;
                    }
                }
            }

            if let Some(ref mut dd) = self.detection_decoder {
                if !dd.is_alive() {
                    tracing::warn!(camera = %self.camera_id, "detection decoder died, restarting");
                    match DetectionDecoder::new(self.config.sample_fps) {
                        Ok(d) => self.detection_decoder = Some(d),
                        Err(e) => {
                            tracing::error!(camera = %self.camera_id, error = %e, "failed to restart detection decoder");
                        }
                    }
                }
            }

            if let Err(e) = self.process_new_segments() {
                tracing::error!(
                    camera = %self.camera_id,
                    error = %e,
                    "motion analysis error"
                );
            }

            thread::sleep(POLL_INTERVAL);
        }

        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    fn process_new_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let segments_to_process = {
            let buffer = self.buffer.read().map_err(|_| "buffer lock poisoned")?;
            let first_seq = buffer.first_sequence();
            let last_seq = buffer.last_sequence();

            if first_seq > 0 {
                self.motion_store.cleanup(&self.camera_id, first_seq);
                if let Some(ref ds) = self.detection_store {
                    ds.cleanup(&self.camera_id, first_seq);
                }
            }

            if self.last_processed < first_seq {
                self.last_processed = first_seq;
            }

            let mut segments = Vec::new();
            for seq in self.last_processed..last_seq {
                if let Some(segment) = buffer.get_segment_by_sequence(seq) {
                    segments.push((
                        seq,
                        segment.data.clone(),
                        segment.start_pts,
                        segment.duration_ns,
                    ));
                }
            }
            segments
        };

        for (seq, data, start_pts, duration_ns) in segments_to_process {
            let score = self.analyze_segment(&data, duration_ns)?;

            self.score_histogram.record(score);
            let threshold = self.score_histogram.threshold();

            if score >= threshold {
                self.motion_store.insert(
                    &self.camera_id,
                    MotionEntry {
                        segment_sequence: seq,
                        start_time_ns: start_pts,
                        end_time_ns: start_pts + duration_ns,
                        motion_score: score,
                    },
                );

                tracing::debug!(
                    camera = %self.camera_id,
                    sequence = seq,
                    score = format!("{:.3}", score),
                    threshold = format!("{:.3}", threshold),
                    samples = self.score_histogram.samples(),
                    "motion detected"
                );

                if self.object_detector.is_some() && self.detection_store.is_some() {
                    self.run_object_detection(&data, seq, duration_ns);
                }
            }

            self.last_processed = seq + 1;
        }

        Ok(())
    }

    fn analyze_segment(
        &mut self,
        data: &[u8],
        duration_ns: u64,
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let raw_frames = self.decoder.decode_segment(data, duration_ns);

        if raw_frames.is_empty() {
            return Ok(0.0);
        }

        let height = self.decoder.height() as i32;
        let mut total_score = 0.0f32;
        let mut frame_count = 0u32;

        for frame_data in &raw_frames {
            let mat = Mat::from_slice(frame_data)?;
            let mat = mat.reshape(1, height)?;

            match self.detector.process_frame(&mat) {
                Ok(score) => {
                    total_score += score;
                    frame_count += 1;
                }
                Err(e) => {
                    tracing::trace!(error = %e, "frame processing error");
                }
            }
        }

        if frame_count == 0 {
            return Ok(0.0);
        }

        Ok(total_score / frame_count as f32)
    }

    fn run_object_detection(&mut self, data: &[u8], seq: u64, duration_ns: u64) {
        let detection_decoder = match &self.detection_decoder {
            Some(d) => d,
            None => return,
        };
        let detection_store = match &self.detection_store {
            Some(s) => s.clone(),
            None => return,
        };

        let raw_frames = detection_decoder.decode_segment(data, duration_ns);
        if raw_frames.is_empty() {
            return;
        }

        let height = detection_decoder.height() as i32;
        let _width = detection_decoder.width() as i32;

        for frame_data in raw_frames.iter() {
            let mat = match Mat::from_slice(frame_data) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let reshaped = match mat.reshape(3, height) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mat_owned = match reshaped.try_clone() {
                Ok(m) => m,
                Err(_) => continue,
            };

            let object_detector = match &mut self.object_detector {
                Some(d) => d,
                None => return,
            };

            let detections = match object_detector.detect(&mat_owned) {
                Ok(d) => d,
                Err(e) => {
                    tracing::trace!(error = %e, "object detection error");
                    continue;
                }
            };

            if detections.is_empty() {
                continue;
            }

            let frame_jpeg = match encode_jpeg(&mat_owned) {
                Some(j) => j,
                None => continue,
            };

            for det in detections {
                detection_store.insert(
                    &self.camera_id,
                    seq,
                    det.class_name.clone(),
                    det.confidence,
                    frame_jpeg.clone(),
                );

                tracing::debug!(
                    camera = %self.camera_id,
                    sequence = seq,
                    class = %det.class_name,
                    confidence = format!("{:.2}", det.confidence),
                    "object detected"
                );
            }
        }
    }
}

fn encode_jpeg(mat: &Mat) -> Option<Vec<u8>> {
    let mut buf = Vector::<u8>::new();
    let params = Vector::<i32>::new();
    imgcodecs::imencode(".jpg", mat, &mut buf, &params).ok()?;
    Some(buf.to_vec())
}

pub fn spawn_analyzer(
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    object_detector: Option<ObjectDetector>,
    config: AnalyticsConfig,
    shutdown: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        match MotionAnalyzer::new(
            camera_id.clone(),
            buffer,
            motion_store,
            detection_store,
            object_detector,
            config,
        ) {
            Ok(analyzer) => analyzer.run(shutdown),
            Err(e) => {
                tracing::error!(camera = %camera_id, error = %e, "failed to create motion analyzer");
            }
        }
    })
}
