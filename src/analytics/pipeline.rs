use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use opencv::core::{Mat, Rect, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;

use crate::analytics::detection_grid::DetectionGrid;
use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::storage::{DetectionEntry, DetectionStore, MotionEntry, MotionStore};

use super::decoder::{CropDecoder, FrameDecoder};
use super::motion::MotionDetector;
use super::ollama::{Detection, OllamaDetector};

const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;

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

pub struct AnalyzerContext {
    pub camera_id: String,
    pub buffer: Arc<RwLock<HotBuffer>>,
    pub motion_store: MotionStore,
    pub detection_store: Option<DetectionStore>,
    pub object_detector: Option<OllamaDetector>,
    pub config: AnalyticsConfig,
    pub detection_grid: Option<DetectionGrid>,
    pub data_dir: PathBuf,
}

pub struct MotionAnalyzer {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    object_detector: Option<OllamaDetector>,
    last_processed: u64,
    last_motion_bbox: Option<Rect>,
    detection_grid: Option<DetectionGrid>,
    grid_save_counter: u32,
}

impl MotionAnalyzer {
    fn new(ctx: AnalyzerContext) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let detector = MotionDetector::new(&ctx.camera_id, &ctx.data_dir)?;
        let decoder = FrameDecoder::new()?;

        let last_processed = ctx
            .motion_store
            .last_sequence(&ctx.camera_id)
            .map(|s| s + 1)
            .unwrap_or(0);

        Ok(Self {
            camera_id: ctx.camera_id,
            buffer: ctx.buffer,
            motion_store: ctx.motion_store,
            detection_store: ctx.detection_store,
            config: ctx.config,
            detector,
            decoder,
            object_detector: ctx.object_detector,
            last_processed,
            last_motion_bbox: None,
            detection_grid: ctx.detection_grid,
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

            self.grid_save_counter += 1;
            if self.grid_save_counter >= 1500 {
                self.grid_save_counter = 0;
                if let Some(ref grid) = self.detection_grid {
                    grid.save(&self.camera_id);
                }
            }

            thread::sleep(POLL_INTERVAL);
        }

        if let Some(ref grid) = self.detection_grid {
            grid.save(&self.camera_id);
        }
        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    fn process_new_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

        let has_detection = self.object_detector.is_some() && self.detection_store.is_some();
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
            self.motion_store
                .set_tuner_stats(&self.camera_id, self.detector.tuner_stats());

            if score >= MOTION_THRESHOLD {
                self.detector.report_motion_event();
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

        // Evaluate tuner every cycle (not just on motion events)
        if let Err(e) = self.detector.maybe_tune() {
            tracing::warn!(camera = %self.camera_id, error = %e, "tuner update failed");
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

    // --- Phase 2: Generic frame extraction + detection ---

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
    }

    fn extract_run_frames(&self, run: &[MotionSegment], crop_decoder: &CropDecoder) -> Vec<Mat> {
        let indices = sample_indices(run.len());
        let height = crop_decoder.height() as i32;
        let mut all_frames = Vec::new();

        // Feed preceding segments to prime the decoder
        let first_seq = run[0].seq;
        if first_seq >= 3 {
            if let Ok(buffer) = self.buffer.read() {
                for prime_seq in (first_seq - 3)..first_seq {
                    if let Some(seg) = buffer.get_segment_by_sequence(prime_seq) {
                        decode_to_mats(
                            crop_decoder,
                            &seg.data,
                            seg.duration_ns,
                            height,
                            &mut all_frames,
                        );
                    }
                }
            }
        }

        for &idx in &indices {
            let seg = &run[idx];
            decode_to_mats(
                crop_decoder,
                &seg.data,
                seg.duration_ns,
                height,
                &mut all_frames,
            );
        }

        subsample_frames(all_frames)
    }

    fn detect_run(&mut self, run: Vec<MotionSegment>, crop_decoder: &CropDecoder) {
        if run.is_empty() {
            return;
        }

        let frames = self.extract_run_frames(&run, crop_decoder);
        if frames.is_empty() {
            return;
        }

        // Encode filmstrip JPEGs and store in detection store
        let filmstrip_jpegs: Vec<Vec<u8>> = frames.iter().filter_map(encode_jpeg).collect();
        if let Some(ref ds) = self.detection_store {
            let filmstrip = Arc::new(filmstrip_jpegs.clone());
            for seg in &run {
                ds.insert_filmstrip(&self.camera_id, seg.seq, Arc::clone(&filmstrip));
            }
        }

        // Get motion position from bbox
        let (cx, cy) = self
            .last_motion_bbox
            .map(|bbox| {
                let cx = (bbox.x as f32 + bbox.width as f32 / 2.0) / ANALYSIS_WIDTH as f32;
                let cy = (bbox.y as f32 + bbox.height as f32 / 2.0) / ANALYSIS_HEIGHT as f32;
                (cx.clamp(0.0, 1.0), cy.clamp(0.0, 1.0))
            })
            .unwrap_or((0.5, 0.5));

        let (detections, best_frame_idx) = self.detect_ollama(&frames, cx, cy);

        if detections.is_empty() {
            return;
        }

        // Use the best frame's JPEG as the detection thumbnail
        let frame_jpeg = filmstrip_jpegs
            .get(best_frame_idx)
            .cloned()
            .or_else(|| filmstrip_jpegs.first().cloned())
            .unwrap_or_default();

        let result = SegmentDetectionResult {
            classes: detections.iter().map(|d| d.class_name.clone()).collect(),
            confidences: detections.iter().map(|d| d.confidence).collect(),
            centers: detections.iter().map(|d| (d.cx, d.cy)).collect(),
            frame_jpeg,
        };

        // Store for all segments in the run
        let mid_seq = run[run.len() / 2].seq;
        self.store_detection_result(mid_seq, &result);
        for seg in &run {
            if seg.seq == mid_seq {
                continue;
            }
            let propagated = SegmentDetectionResult {
                classes: result.classes.clone(),
                confidences: result.confidences.clone(),
                centers: result.centers.clone(),
                frame_jpeg: result.frame_jpeg.clone(),
            };
            self.store_detection_result(seg.seq, &propagated);
        }
    }

    fn detect_ollama(&self, frames: &[Mat], cx: f32, cy: f32) -> (Vec<Detection>, usize) {
        let ollama = match &self.object_detector {
            Some(d) => d,
            None => return (Vec::new(), 0),
        };

        let detections = match ollama.detect_grid(frames, cx, cy) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    camera = %self.camera_id,
                    error = %e,
                    "ollama detection error"
                );
                return (Vec::new(), 0);
            }
        };

        if !detections.is_empty() {
            tracing::debug!(
                camera = %self.camera_id,
                count = detections.len(),
                classes = ?detections.iter().map(|d| &d.class_name).collect::<Vec<_>>(),
                "ollama detections"
            );
        }

        // Use second frame (index 1) as thumbnail — inner frame
        let best_idx = if frames.len() > 1 { 1 } else { 0 };
        (detections, best_idx)
    }

    fn store_detection_result(&mut self, seq: u64, result: &SegmentDetectionResult) {
        let detection_store = match &self.detection_store {
            Some(s) => s,
            None => return,
        };

        let (backend, model) = self
            .object_detector
            .as_ref()
            .map(|d| ("ollama".to_string(), d.model().to_string()))
            .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));

        let mut stored_any = false;
        for (i, (class, &confidence)) in result.classes.iter().zip(&result.confidences).enumerate()
        {
            let (cx, cy) = result.centers.get(i).copied().unwrap_or((0.5, 0.5));

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
                DetectionEntry {
                    id: detection_store.next_id(),
                    segment_sequence: seq,
                    object_class: class.clone(),
                    confidence,
                    frame_jpeg: result.frame_jpeg.clone(),
                    backend: backend.clone(),
                    model: model.clone(),
                },
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
}

fn sample_indices(len: usize) -> Vec<usize> {
    if len <= 4 {
        (0..len).collect()
    } else {
        vec![0, len / 3, 2 * len / 3, len - 1]
    }
}

fn decode_to_mats(
    decoder: &CropDecoder,
    data: &[u8],
    duration_ns: u64,
    height: i32,
    out: &mut Vec<Mat>,
) {
    for frame_data in &decoder.decode_segment(data, duration_ns) {
        if let Ok(mat) = Mat::from_slice(frame_data) {
            if let Ok(reshaped) = mat.reshape(3, height) {
                if let Ok(cloned) = reshaped.try_clone() {
                    out.push(cloned);
                }
            }
        }
    }
}

fn subsample_frames(frames: Vec<Mat>) -> Vec<Mat> {
    if frames.len() <= 4 {
        return frames;
    }
    let n = frames.len();
    [0, n / 3, 2 * n / 3, n - 1]
        .iter()
        .map(|&i| frames[i].try_clone().unwrap_or_default())
        .collect()
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

pub fn spawn_analyzer(
    ctx: AnalyzerContext,
    shutdown: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    let camera_id = ctx.camera_id.clone();
    tokio::task::spawn_blocking(move || match MotionAnalyzer::new(ctx) {
        Ok(analyzer) => analyzer.run(shutdown),
        Err(e) => {
            tracing::error!(camera = %camera_id, error = %e, "failed to create motion analyzer");
        }
    })
}
