use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use opencv::core::{Mat, Rect, Size, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use crate::analytics::detection_grid::DetectionGrid;
use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::storage::{DetectionStore, MotionEntry, MotionStore};

use super::decoder::{CropDecoder, FrameDecoder};
use super::motion::MotionDetector;
use super::object::ObjectDetector;

const DETECTION_WIDTH: i32 = 640;
const DETECTION_HEIGHT: i32 = 480;
const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;
const CROP_DECODE_WIDTH: i32 = 1920;
const CROP_DECODE_HEIGHT: i32 = 1080;

// Fixed motion threshold — a score of 0.05 means 0.5% of pixels survived
// morphological opening and contour area filtering. Below this is too small
// to be meaningful motion.
const MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);

struct MotionSegment {
    seq: u64,
    data: Vec<u8>,
    duration_ns: u64,
}

struct SegmentDetectionResult {
    classes: Vec<String>,
    confidences: Vec<f32>,
    centers: Vec<(f32, f32)>,
    frame_jpeg: Vec<u8>,
}

pub struct MotionAnalyzer {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    has_object_detection: bool,
    object_detector: Option<ObjectDetector>,
    last_processed: u64,
    last_motion_bbox: Option<Rect>,
    detection_grid: Option<DetectionGrid>,
    grid_save_counter: u32,
}

impl MotionAnalyzer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        camera_id: String,
        buffer: Arc<RwLock<HotBuffer>>,
        motion_store: MotionStore,
        detection_store: Option<DetectionStore>,
        object_detector: Option<ObjectDetector>,
        config: AnalyticsConfig,
        detection_grid: Option<DetectionGrid>,
        data_dir: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let detector = MotionDetector::new(&camera_id, &data_dir)?;
        let decoder = FrameDecoder::new()?;

        let has_object_detection = object_detector.is_some();

        let last_processed = motion_store
            .last_sequence(&camera_id)
            .map(|s| s + 1)
            .unwrap_or(0);

        Ok(Self {
            camera_id,
            buffer,
            motion_store,
            detection_store,
            config,
            detector,
            decoder,
            has_object_detection,
            object_detector,
            last_processed,
            last_motion_bbox: None,
            detection_grid,
            grid_save_counter: 0,
        })
    }

    fn run(mut self, shutdown: Arc<AtomicBool>) {
        tracing::info!(camera = %self.camera_id, "motion analyzer started");

        while !shutdown.load(Ordering::Relaxed) {
            if !self.decoder.is_alive() {
                tracing::warn!(camera = %self.camera_id, "decoder process died, restarting");
                match FrameDecoder::new() {
                    Ok(d) => self.decoder = d,
                    Err(e) => {
                        tracing::error!(camera = %self.camera_id, error = %e, "failed to restart decoder");
                        thread::sleep(Duration::from_secs(5));
                        continue;
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

            // Save detection grid every ~5 minutes (1500 cycles × 200ms)
            self.grid_save_counter += 1;
            if self.grid_save_counter >= 1500 {
                self.grid_save_counter = 0;
                if let Some(ref grid) = self.detection_grid {
                    grid.save(&self.camera_id);
                }
            }

            thread::sleep(POLL_INTERVAL);
        }

        // Save on shutdown
        if let Some(ref grid) = self.detection_grid {
            grid.save(&self.camera_id);
        }
        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    fn process_new_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Grab only the sequence range under the lock, release immediately
        let (first_seq, last_seq) = {
            let buffer = self.buffer.read().map_err(|_| "buffer lock poisoned")?;
            (buffer.first_sequence(), buffer.last_sequence())
        };

        if first_seq > 0 {
            self.motion_store.cleanup(&self.camera_id, first_seq);
            if let Some(ref ds) = self.detection_store {
                ds.cleanup(&self.camera_id, first_seq);
            }
        }

        if self.last_processed < first_seq {
            self.last_processed = first_seq;
        }

        // Fetch and clone each segment with a short-lived lock
        let mut segments_to_process = Vec::new();
        for seq in self.last_processed..last_seq {
            let segment = {
                let buffer = self.buffer.read().map_err(|_| "buffer lock poisoned")?;
                buffer
                    .get_segment_by_sequence(seq)
                    .map(|s| (seq, s.data.clone(), s.start_pts, s.duration_ns))
            };
            if let Some(seg) = segment {
                segments_to_process.push(seg);
            }
        }

        let has_detection = self.has_object_detection && self.detection_store.is_some();
        let mut motion_segments = Vec::new();

        // Phase 1: Motion analysis
        for (seq, data, start_pts, duration_ns) in segments_to_process {
            let score = self.analyze_segment(&data)?;

            if let Some(jpeg) = self.detector.stability_map_jpeg() {
                self.motion_store.set_stability_map(&self.camera_id, jpeg);
            }
            if let Some(jpeg) = self.detector.background_jpeg() {
                self.motion_store.set_background_map(&self.camera_id, jpeg);
            }

            if score >= MOTION_THRESHOLD {
                if let Err(e) = self.detector.report_motion_event() {
                    tracing::warn!(camera = %self.camera_id, error = %e, "tuner update failed");
                }
                let mask_jpeg = self.detector.fg_mask_jpeg();
                self.motion_store.insert(
                    &self.camera_id,
                    MotionEntry {
                        segment_sequence: seq,
                        start_time_ns: start_pts,
                        end_time_ns: start_pts + duration_ns,
                        motion_score: score,
                        mask_jpeg,
                    },
                );

                tracing::debug!(
                    camera = %self.camera_id,
                    sequence = seq,
                    score = format!("{:.3}", score),
                    "motion detected"
                );

                if has_detection {
                    motion_segments.push(MotionSegment {
                        seq,
                        data,
                        duration_ns,
                    });
                }
            }

            self.last_processed = seq + 1;
        }

        // Decay detection grid each cycle
        if let Some(ref grid) = self.detection_grid {
            grid.decay(&self.camera_id);
        }

        // Phase 2: Sampled object detection
        if !motion_segments.is_empty() {
            self.run_sampled_detections(motion_segments);
        }

        Ok(())
    }

    fn analyze_segment(
        &mut self,
        data: &[u8],
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let raw_frames = self.decoder.decode_segment(data);

        if raw_frames.is_empty() {
            return Ok(0.0);
        }

        let height = self.decoder.height() as i32;
        let mut total_score = 0.0f32;
        let mut frame_count = 0u32;
        let mut last_bbox = None;

        for frame_data in &raw_frames {
            let mat = Mat::from_slice(frame_data)?;
            let mat = mat.reshape(1, height)?;

            match self.detector.process_frame(&mat) {
                Ok(score) => {
                    total_score += score;
                    frame_count += 1;
                    if let Some(bbox) = self.detector.motion_bbox() {
                        last_bbox = Some(bbox);
                    }
                }
                Err(e) => {
                    tracing::trace!(error = %e, "frame processing error");
                }
            }
        }

        self.last_motion_bbox = last_bbox;

        if frame_count == 0 {
            return Ok(0.0);
        }

        Ok(total_score / frame_count as f32)
    }

    fn detect_segment(
        &mut self,
        data: &[u8],
        duration_ns: u64,
        crop_decoder: &CropDecoder,
    ) -> Option<SegmentDetectionResult> {
        let raw_frames = crop_decoder.decode_segment(data, duration_ns);
        if raw_frames.is_empty() {
            return None;
        }

        let height = crop_decoder.height() as i32;
        let crop_rect = self.crop_region();

        for frame_data in raw_frames.iter() {
            let mat = match Mat::from_slice(frame_data) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let reshaped = match mat.reshape(3, height) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let detection_input = match crop_rect {
                Some(rect) => match Mat::roi(&reshaped, rect) {
                    Ok(roi) => match roi.try_clone() {
                        Ok(m) => m,
                        Err(_) => continue,
                    },
                    Err(_) => continue,
                },
                None => {
                    let mut resized = Mat::default();
                    if imgproc::resize(
                        &reshaped,
                        &mut resized,
                        Size::new(DETECTION_WIDTH, DETECTION_HEIGHT),
                        0.0,
                        0.0,
                        imgproc::INTER_LINEAR,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    resized
                }
            };

            let object_detector = match &mut self.object_detector {
                Some(d) => d,
                None => return None,
            };

            let detections = match object_detector.detect(&detection_input) {
                Ok(d) => d,
                Err(e) => {
                    tracing::trace!(error = %e, "object detection error");
                    continue;
                }
            };

            if detections.is_empty() {
                continue;
            }

            let frame_jpeg = match encode_jpeg(&detection_input) {
                Some(j) => j,
                None => continue,
            };

            let mut classes = Vec::with_capacity(detections.len());
            let mut confidences = Vec::with_capacity(detections.len());
            let mut centers = Vec::with_capacity(detections.len());
            for det in detections {
                classes.push(det.class_name);
                confidences.push(det.confidence);
                centers.push((det.cx, det.cy));
            }

            return Some(SegmentDetectionResult {
                classes,
                confidences,
                centers,
                frame_jpeg,
            });
        }

        None
    }

    fn store_detection_result(&mut self, seq: u64, result: &SegmentDetectionResult) {
        let detection_store = match &self.detection_store {
            Some(s) => s,
            None => return,
        };

        let mut stored_any = false;
        for (i, (class, &confidence)) in result.classes.iter().zip(&result.confidences).enumerate()
        {
            let (cx, cy) = result.centers.get(i).copied().unwrap_or((0.5, 0.5));

            // Check detection grid — skip absorbed (stationary) objects
            if let Some(ref grid) = self.detection_grid {
                if !grid.record(&self.camera_id, class, cx, cy) {
                    tracing::trace!(
                        camera = %self.camera_id,
                        class = %class,
                        "detection suppressed (absorbed by grid)"
                    );
                    continue;
                }
            }

            detection_store.insert(
                &self.camera_id,
                seq,
                class.clone(),
                confidence,
                result.frame_jpeg.clone(),
            );
            stored_any = true;

            tracing::debug!(
                camera = %self.camera_id,
                sequence = seq,
                class = %class,
                confidence = format!("{:.2}", confidence),
                "object detected"
            );
        }

        if stored_any {
            self.detector.report_positive_detection();
        }
    }

    fn run_sampled_detections(&mut self, segments: Vec<MotionSegment>) {
        let crop_decoder = match CropDecoder::new(self.config.sample_fps) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(camera = %self.camera_id, error = %e, "failed to create crop decoder");
                return;
            }
        };

        let runs = group_contiguous_runs(segments);
        for run in runs {
            self.detect_run(run, &crop_decoder);
        }
        // crop_decoder is dropped here, killing the ffmpeg process
    }

    fn detect_run(&mut self, run: Vec<MotionSegment>, crop_decoder: &CropDecoder) {
        let len = run.len();
        if len <= 2 {
            for seg in &run {
                if let Some(result) = self.detect_segment(&seg.data, seg.duration_ns, crop_decoder)
                {
                    self.store_detection_result(seg.seq, &result);
                }
            }
            return;
        }

        let first_result = self.detect_segment(&run[0].data, run[0].duration_ns, crop_decoder);
        let last_result =
            self.detect_segment(&run[len - 1].data, run[len - 1].duration_ns, crop_decoder);

        let boundaries_agree = match (&first_result, &last_result) {
            (Some(first), Some(last)) => {
                let first_classes: BTreeSet<&str> =
                    first.classes.iter().map(|s| s.as_str()).collect();
                let last_classes: BTreeSet<&str> =
                    last.classes.iter().map(|s| s.as_str()).collect();
                first_classes == last_classes
            }
            _ => false,
        };

        if boundaries_agree {
            let first_result = first_result.unwrap();
            let last_result = last_result.unwrap();

            self.store_detection_result(run[0].seq, &first_result);
            self.store_detection_result(run[len - 1].seq, &last_result);

            let min_confidences: Vec<f32> = first_result
                .confidences
                .iter()
                .zip(&last_result.confidences)
                .map(|(&a, &b)| a.min(b))
                .collect();

            let mid = len / 2;
            for (i, seg) in run.iter().enumerate().take(len - 1).skip(1) {
                let nearest = if i <= mid {
                    &first_result
                } else {
                    &last_result
                };

                let propagated = SegmentDetectionResult {
                    classes: first_result.classes.clone(),
                    confidences: min_confidences.clone(),
                    centers: nearest.centers.clone(),
                    frame_jpeg: nearest.frame_jpeg.clone(),
                };

                self.store_detection_result(seg.seq, &propagated);

                tracing::debug!(
                    camera = %self.camera_id,
                    sequence = seg.seq,
                    "detection propagated from boundary"
                );
            }
        } else {
            // Boundaries disagree or empty — split in half and recurse
            if let Some(result) = first_result {
                self.store_detection_result(run[0].seq, &result);
            }
            if let Some(result) = last_result {
                self.store_detection_result(run[len - 1].seq, &result);
            }

            let mut inner: Vec<MotionSegment> = run.into_iter().skip(1).collect();
            inner.pop(); // remove last (already stored)

            if !inner.is_empty() {
                let mid = inner.len() / 2;
                let right = inner.split_off(mid);
                self.detect_run(inner, crop_decoder);
                self.detect_run(right, crop_decoder);
            }
        }
    }

    fn crop_region(&self) -> Option<Rect> {
        let bbox = self.last_motion_bbox?;

        let scale_x = CROP_DECODE_WIDTH as f32 / ANALYSIS_WIDTH as f32;
        let scale_y = CROP_DECODE_HEIGHT as f32 / ANALYSIS_HEIGHT as f32;

        let center_x = ((bbox.x as f32 + bbox.width as f32 / 2.0) * scale_x) as i32;
        let center_y = ((bbox.y as f32 + bbox.height as f32 / 2.0) * scale_y) as i32;

        let scaled_w = (bbox.width as f32 * scale_x) as i32;
        let scaled_h = (bbox.height as f32 * scale_y) as i32;

        if scaled_w > DETECTION_WIDTH || scaled_h > DETECTION_HEIGHT {
            return None;
        }

        let x = (center_x - DETECTION_WIDTH / 2).clamp(0, CROP_DECODE_WIDTH - DETECTION_WIDTH);
        let y = (center_y - DETECTION_HEIGHT / 2).clamp(0, CROP_DECODE_HEIGHT - DETECTION_HEIGHT);

        Some(Rect::new(x, y, DETECTION_WIDTH, DETECTION_HEIGHT))
    }
}

fn group_contiguous_runs(segments: Vec<MotionSegment>) -> Vec<Vec<MotionSegment>> {
    let mut runs: Vec<Vec<MotionSegment>> = Vec::new();

    for seg in segments {
        let start_new = match runs.last() {
            Some(run) => {
                let last_seq = run.last().unwrap().seq;
                seg.seq != last_seq + 1
            }
            None => true,
        };

        if start_new {
            runs.push(vec![seg]);
        } else {
            runs.last_mut().unwrap().push(seg);
        }
    }

    runs
}

fn encode_jpeg(mat: &Mat) -> Option<Vec<u8>> {
    let mut buf = Vector::<u8>::new();
    let params = Vector::<i32>::new();
    imgcodecs::imencode(".jpg", mat, &mut buf, &params).ok()?;
    Some(buf.to_vec())
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_analyzer(
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    object_detector: Option<ObjectDetector>,
    config: AnalyticsConfig,
    shutdown: Arc<AtomicBool>,
    detection_grid: Option<DetectionGrid>,
    data_dir: PathBuf,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        match MotionAnalyzer::new(
            camera_id.clone(),
            buffer,
            motion_store,
            detection_store,
            object_detector,
            config,
            detection_grid,
            data_dir,
        ) {
            Ok(analyzer) => analyzer.run(shutdown),
            Err(e) => {
                tracing::error!(camera = %camera_id, error = %e, "failed to create motion analyzer");
            }
        }
    })
}
