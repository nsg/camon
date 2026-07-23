use std::collections::HashMap;
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
use crate::storage::{
    DetectionDebugStore, DetectionEntry, DetectionStore, MotionEntry, MotionStore,
};

use super::decoder::{CropDecoder, FrameDecoder};
use super::motion::MotionDetector;
use super::ollama::{DetectResult, Detection, OllamaDetector};

const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;

const MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);
const GRID_SAVE_INTERVAL: u32 = 1500; // ~5 minutes at 200ms poll interval
const CROP_PADDING: f32 = 0.2;
const MIN_CROP_FRACTION: f32 = 0.15;

#[derive(Clone, Copy)]
struct NormalizedRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn normalize_rect(r: Rect, frame_w: i32, frame_h: i32) -> NormalizedRect {
    NormalizedRect {
        x: r.x as f32 / frame_w as f32,
        y: r.y as f32 / frame_h as f32,
        w: r.width as f32 / frame_w as f32,
        h: r.height as f32 / frame_h as f32,
    }
}

fn union_rects_padded(rects: &[NormalizedRect], padding: f32) -> Option<NormalizedRect> {
    if rects.is_empty() {
        return None;
    }
    let min_x = rects.iter().map(|r| r.x).fold(f32::MAX, f32::min);
    let min_y = rects.iter().map(|r| r.y).fold(f32::MAX, f32::min);
    let max_x = rects.iter().map(|r| r.x + r.w).fold(0.0f32, f32::max);
    let max_y = rects.iter().map(|r| r.y + r.h).fold(0.0f32, f32::max);

    let w = (max_x - min_x).max(MIN_CROP_FRACTION);
    let h = (max_y - min_y).max(MIN_CROP_FRACTION);
    let pad_x = w * padding;
    let pad_y = h * padding;

    let x = (min_x - pad_x).max(0.0);
    let y = (min_y - pad_y).max(0.0);
    Some(NormalizedRect {
        x,
        y,
        w: (w + 2.0 * pad_x).min(1.0 - x),
        h: (h + 2.0 * pad_y).min(1.0 - y),
    })
}

fn union_two_rects(a: NormalizedRect, b: NormalizedRect) -> NormalizedRect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let max_x = (a.x + a.w).max(b.x + b.w);
    let max_y = (a.y + a.h).max(b.y + b.h);
    NormalizedRect {
        x,
        y,
        w: max_x - x,
        h: max_y - y,
    }
}

fn crop_mat(frame: &Mat, region: &NormalizedRect) -> Option<Mat> {
    let cols = frame.cols();
    let rows = frame.rows();
    if cols == 0 || rows == 0 {
        return None;
    }

    let x = (region.x * cols as f32) as i32;
    let y = (region.y * rows as f32) as i32;
    let w = (region.w * cols as f32) as i32;
    let h = (region.h * rows as f32) as i32;

    let roi = Rect::new(
        x.max(0),
        y.max(0),
        w.min(cols - x.max(0)),
        h.min(rows - y.max(0)),
    );
    if roi.width <= 0 || roi.height <= 0 {
        return None;
    }

    Mat::roi(frame, roi).ok()?.try_clone().ok()
}

struct MotionSegment {
    seq: u64,
    data: Vec<u8>,
    duration_ns: u64,
}

struct PendingSegment {
    seq: u64,
    data: Vec<u8>,
    start_pts: u64,
    duration_ns: u64,
}

struct SegmentDetectionResult {
    classes: Vec<String>,
    confidences: Vec<f32>,
    /// Per-class bounding rects in full-frame normalized coordinates.
    class_rects: Vec<Vec<(f32, f32, f32, f32)>>,
    frame_jpeg: Vec<u8>,
}

pub struct AnalyzerContext {
    pub camera_id: String,
    pub buffer: Arc<RwLock<HotBuffer>>,
    pub motion_store: MotionStore,
    pub detection_store: Option<DetectionStore>,
    pub debug_store: Option<DetectionDebugStore>,
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
    debug_store: Option<DetectionDebugStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    object_detector: Option<OllamaDetector>,
    last_processed: u64,
    detection_grid: Option<DetectionGrid>,
    grid_save_counter: u32,
    segment_crops: HashMap<u64, NormalizedRect>,
    segment_motion_rects: HashMap<u64, Vec<NormalizedRect>>,
    last_run_motion_rects: Vec<(f32, f32, f32, f32)>,
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
            debug_store: ctx.debug_store,
            config: ctx.config,
            detector,
            decoder,
            object_detector: ctx.object_detector,
            last_processed,
            detection_grid: ctx.detection_grid,
            grid_save_counter: 0,
            segment_crops: HashMap::new(),
            segment_motion_rects: HashMap::new(),
            last_run_motion_rects: Vec::new(),
        })
    }

    fn run(mut self, shutdown: Arc<AtomicBool>) {
        tracing::info!(camera = %self.camera_id, "motion analyzer started");

        while !shutdown.load(Ordering::Relaxed) {
            if !self.ensure_decoder_alive() {
                continue;
            }

            if let Err(e) = self.process_new_segments() {
                tracing::error!(
                    camera = %self.camera_id,
                    error = %e,
                    "motion analysis error"
                );
            }

            self.maybe_save_grid();
            thread::sleep(POLL_INTERVAL);
        }

        self.save_grid();
        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    fn ensure_decoder_alive(&mut self) -> bool {
        if self.decoder.is_alive() {
            return true;
        }
        tracing::warn!(camera = %self.camera_id, "decoder process died, restarting");
        match FrameDecoder::new() {
            Ok(d) => {
                self.decoder = d;
                true
            }
            Err(e) => {
                tracing::error!(camera = %self.camera_id, error = %e, "failed to restart decoder");
                thread::sleep(Duration::from_secs(5));
                false
            }
        }
    }

    fn maybe_save_grid(&mut self) {
        self.grid_save_counter += 1;
        if self.grid_save_counter >= GRID_SAVE_INTERVAL {
            self.grid_save_counter = 0;
            self.save_grid();
        }
    }

    fn save_grid(&self) {
        if let Some(ref grid) = self.detection_grid {
            grid.save(&self.camera_id);
        }
    }

    fn process_new_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (first_seq, last_seq) = {
            let buffer = self.buffer.read().map_err(|_| "buffer lock poisoned")?;
            (buffer.first_sequence(), buffer.last_sequence())
        };

        self.cleanup_old_data(first_seq);
        let segments = self.collect_pending_segments(last_seq)?;
        let motion_segments = self.run_motion_analysis(segments)?;

        if let Err(e) = self.detector.maybe_tune() {
            tracing::warn!(camera = %self.camera_id, error = %e, "tuner update failed");
        }
        if let Some(ref grid) = self.detection_grid {
            grid.decay(&self.camera_id);
        }

        if !motion_segments.is_empty() {
            self.run_sampled_detections(motion_segments);
        }

        Ok(())
    }

    fn cleanup_old_data(&mut self, first_seq: u64) {
        if first_seq > 0 {
            self.motion_store.cleanup(&self.camera_id, first_seq);
            if let Some(ref ds) = self.detection_store {
                ds.cleanup(&self.camera_id, first_seq);
            }
            self.segment_crops.retain(|&seq, _| seq >= first_seq);
            self.segment_motion_rects.retain(|&seq, _| seq >= first_seq);
        }
        if self.last_processed < first_seq {
            self.last_processed = first_seq;
        }
    }

    fn collect_pending_segments(
        &self,
        last_seq: u64,
    ) -> Result<Vec<PendingSegment>, Box<dyn std::error::Error + Send + Sync>> {
        let mut segments = Vec::new();
        for seq in self.last_processed..last_seq {
            let segment = {
                let buffer = self.buffer.read().map_err(|_| "buffer lock poisoned")?;
                buffer.get_segment_by_sequence(seq).map(|s| PendingSegment {
                    seq,
                    data: s.data.clone(),
                    start_pts: s.start_pts,
                    duration_ns: s.duration_ns,
                })
            };
            if let Some(seg) = segment {
                segments.push(seg);
            }
        }
        Ok(segments)
    }

    fn run_motion_analysis(
        &mut self,
        segments: Vec<PendingSegment>,
    ) -> Result<Vec<MotionSegment>, Box<dyn std::error::Error + Send + Sync>> {
        let has_detection = self.object_detector.is_some() && self.detection_store.is_some();
        let capture_frames = self.detection_store.is_some();
        let mut motion_segments = Vec::new();
        let mut motion_frames: Vec<(u64, Vec<u8>)> = Vec::new();

        for seg in segments {
            let (score, crop, motion_rects, frame_jpeg) =
                self.analyze_segment(&seg.data, capture_frames)?;
            self.publish_debug_maps();

            if score >= MOTION_THRESHOLD {
                self.record_motion(seg.seq, seg.start_pts, seg.duration_ns, score);
                if let Some(jpeg) = frame_jpeg {
                    motion_frames.push((seg.seq, jpeg));
                }
                if has_detection {
                    if let Some(crop) = crop {
                        self.segment_crops.insert(seg.seq, crop);
                    }
                    if !motion_rects.is_empty() {
                        self.segment_motion_rects.insert(seg.seq, motion_rects);
                    }
                    motion_segments.push(MotionSegment {
                        seq: seg.seq,
                        data: seg.data,
                        duration_ns: seg.duration_ns,
                    });
                }
            }

            self.last_processed = seg.seq + 1;
        }

        if !motion_frames.is_empty() {
            self.store_movement_filmstrips(motion_frames);
        }

        Ok(motion_segments)
    }

    fn store_movement_filmstrips(&self, frames: Vec<(u64, Vec<u8>)>) {
        let ds = match self.detection_store {
            Some(ref ds) => ds,
            None => return,
        };

        // Group contiguous sequences into runs
        let mut runs: Vec<Vec<(u64, Vec<u8>)>> = Vec::new();
        for item in frames {
            let start_new = match runs.last() {
                Some(run) => item.0 != run.last().unwrap().0 + 1,
                None => true,
            };
            if start_new {
                runs.push(vec![item]);
            } else {
                runs.last_mut().unwrap().push(item);
            }
        }

        for run in runs {
            let seqs: Vec<u64> = run.iter().map(|(seq, _)| *seq).collect();
            let all_jpegs: Vec<Vec<u8>> = run.into_iter().map(|(_, jpeg)| jpeg).collect();

            // Subsample to at most 4 representative frames
            let filmstrip_jpegs: Vec<Vec<u8>> = if all_jpegs.len() <= 4 {
                all_jpegs
            } else {
                let n = all_jpegs.len();
                [0, n / 3, 2 * n / 3, n - 1]
                    .iter()
                    .map(|&i| all_jpegs[i].clone())
                    .collect()
            };

            let filmstrip = Arc::new(filmstrip_jpegs);
            for seq in &seqs {
                ds.insert_filmstrip(&self.camera_id, *seq, Arc::clone(&filmstrip));
            }
        }
    }

    fn publish_debug_maps(&mut self) {
        if let Some(jpeg) = self.detector.stability_map_jpeg() {
            self.motion_store.set_stability_map(&self.camera_id, jpeg);
        }
        if let Some(jpeg) = self.detector.background_jpeg() {
            self.motion_store.set_background_map(&self.camera_id, jpeg);
        }
        if let Some(jpeg) = self.detector.raw_mog2_mask_jpeg() {
            self.motion_store.set_raw_mog2_map(&self.camera_id, jpeg);
        }
        if let Some(jpeg) = self.detector.no_shadow_mask_jpeg() {
            self.motion_store.set_no_shadow_map(&self.camera_id, jpeg);
        }
        if let Some(jpeg) = self.detector.morph_mask_jpeg() {
            self.motion_store.set_morph_map(&self.camera_id, jpeg);
        }
        self.motion_store
            .set_tuner_stats(&self.camera_id, self.detector.tuner_stats());
    }

    fn record_motion(&mut self, seq: u64, start_pts: u64, duration_ns: u64, score: f32) {
        let bboxes = self.detector.motion_bboxes();
        self.detector
            .report_motion_event(&bboxes, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
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
    }

    #[allow(clippy::type_complexity)]
    fn analyze_segment(
        &mut self,
        data: &[u8],
        capture_frame: bool,
    ) -> Result<
        (
            f32,
            Option<NormalizedRect>,
            Vec<NormalizedRect>,
            Option<Vec<u8>>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let raw_frames = self.decoder.decode_segment(data);

        if raw_frames.is_empty() {
            return Ok((0.0, None, Vec::new(), None));
        }

        let height = self.decoder.height() as i32;
        let mut total_score = 0.0f32;
        let mut frame_count = 0u32;
        let mut all_rects = Vec::new();
        let mut last_mat = None;

        for frame_data in &raw_frames {
            let mat = Mat::from_slice(frame_data)?;
            let mat = mat.reshape(1, height)?;

            match self.detector.process_frame(&mat) {
                Ok(score) => {
                    total_score += score;
                    frame_count += 1;
                    for r in self.detector.motion_bboxes() {
                        all_rects.push(normalize_rect(r, ANALYSIS_WIDTH, ANALYSIS_HEIGHT));
                    }
                    if capture_frame {
                        last_mat = mat.try_clone().ok();
                    }
                }
                Err(e) => {
                    tracing::trace!(error = %e, "frame processing error");
                }
            }
        }

        let crop = union_rects_padded(&all_rects, CROP_PADDING);

        if frame_count == 0 {
            return Ok((0.0, None, Vec::new(), None));
        }

        let frame_jpeg = if capture_frame {
            last_mat.and_then(|m| encode_jpeg(&m))
        } else {
            None
        };

        Ok((
            total_score / frame_count as f32,
            crop,
            all_rects,
            frame_jpeg,
        ))
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

    fn extract_run_frames(
        &self,
        run: &[MotionSegment],
        crop_decoder: &CropDecoder,
    ) -> Vec<(Mat, Option<NormalizedRect>)> {
        let indices = sample_indices(run.len());
        let height = crop_decoder.height() as i32;
        let mut all_frames: Vec<(Mat, Option<NormalizedRect>)> = Vec::new();

        // Feed preceding segments to prime the decoder (no crop — not motion-positive)
        let first_seq = run[0].seq;
        if first_seq >= 3 {
            if let Ok(buffer) = self.buffer.read() {
                for prime_seq in (first_seq - 3)..first_seq {
                    if let Some(seg) = buffer.get_segment_by_sequence(prime_seq) {
                        let before = all_frames.len();
                        decode_to_mats_tagged(
                            crop_decoder,
                            &seg.data,
                            seg.duration_ns,
                            height,
                            None,
                            &mut all_frames,
                        );
                        let _ = before; // priming frames have None crop
                    }
                }
            }
        }

        for &idx in &indices {
            let seg = &run[idx];
            let crop = self.segment_crops.get(&seg.seq).copied();
            decode_to_mats_tagged(
                crop_decoder,
                &seg.data,
                seg.duration_ns,
                height,
                crop,
                &mut all_frames,
            );
        }

        subsample_tagged(all_frames)
    }

    fn detect_run(&mut self, run: Vec<MotionSegment>, crop_decoder: &CropDecoder) {
        if run.is_empty() {
            return;
        }

        let tagged_frames = self.extract_run_frames(&run, crop_decoder);
        if tagged_frames.is_empty() {
            return;
        }

        // Collect motion rects and crop before consuming them
        let mut all_motion_rects: Vec<(f32, f32, f32, f32)> = Vec::new();
        let mut run_crop: Option<NormalizedRect> = None;
        for seg in &run {
            if let Some(rects) = self.segment_motion_rects.get(&seg.seq) {
                for r in rects {
                    all_motion_rects.push((r.x, r.y, r.w, r.h));
                }
            }
            if let Some(&crop) = self.segment_crops.get(&seg.seq) {
                run_crop = Some(match run_crop {
                    Some(existing) => union_two_rects(existing, crop),
                    None => crop,
                });
            }
        }

        // Encode a full (uncropped) frame for debug overlay.
        // Pick the first tagged frame that has a crop (i.e. had motion).
        let full_frame_jpeg = tagged_frames
            .iter()
            .find(|(_, crop)| crop.is_some())
            .and_then(|(frame, _)| encode_jpeg(frame));

        // Apply per-frame crops
        let cropped: Vec<Mat> = tagged_frames
            .iter()
            .map(|(frame, crop)| {
                crop.and_then(|r| crop_mat(frame, &r))
                    .unwrap_or_else(|| frame.try_clone().unwrap_or_default())
            })
            .collect();

        let filmstrip_jpegs: Vec<Vec<u8>> = cropped.iter().filter_map(encode_jpeg).collect();
        self.store_filmstrip_for_run(&run, &filmstrip_jpegs);

        let (detections, best_frame_idx) =
            self.detect_ollama(&cropped, full_frame_jpeg, &all_motion_rects, run_crop);

        // Remove consumed segment data
        for seg in &run {
            self.segment_crops.remove(&seg.seq);
            self.segment_motion_rects.remove(&seg.seq);
        }

        if detections.is_empty() {
            return;
        }

        self.last_run_motion_rects = all_motion_rects;
        let result =
            build_detection_result(&detections, &filmstrip_jpegs, best_frame_idx, run_crop);
        self.propagate_detection(&run, &result);
    }

    fn store_filmstrip_for_run(&self, run: &[MotionSegment], jpegs: &[Vec<u8>]) {
        if let Some(ref ds) = self.detection_store {
            let filmstrip = Arc::new(jpegs.to_vec());
            for seg in run {
                ds.insert_filmstrip(&self.camera_id, seg.seq, Arc::clone(&filmstrip));
            }
        }
    }

    fn propagate_detection(&mut self, run: &[MotionSegment], result: &SegmentDetectionResult) {
        let mid_seq = run[run.len() / 2].seq;
        self.store_detection_result(mid_seq, result);
        for seg in run {
            if seg.seq == mid_seq {
                continue;
            }
            let propagated = SegmentDetectionResult {
                classes: result.classes.clone(),
                confidences: result.confidences.clone(),
                class_rects: result.class_rects.clone(),
                frame_jpeg: result.frame_jpeg.clone(),
            };
            self.store_detection_result(seg.seq, &propagated);
        }
    }

    fn detect_ollama(
        &self,
        frames: &[Mat],
        full_frame_jpeg: Option<Vec<u8>>,
        motion_rects: &[(f32, f32, f32, f32)],
        run_crop: Option<NormalizedRect>,
    ) -> (Vec<Detection>, usize) {
        let ollama = match &self.object_detector {
            Some(d) => d,
            None => return (Vec::new(), 0),
        };

        let result: DetectResult = match ollama.detect_frames(frames) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    camera = %self.camera_id,
                    error = %e,
                    "ollama detection error"
                );
                return (Vec::new(), 0);
            }
        };

        // Store debug entry regardless of detection outcome
        if let Some(ref debug_store) = self.debug_store {
            if !result.frame_jpegs.is_empty() {
                // Map ollama bboxes from crop space to full-frame space for debug
                let ollama_rects: Vec<(String, f32, f32, f32, f32)> = result
                    .detections
                    .iter()
                    .filter_map(|d| {
                        let (x, y, w, h) = match (d.bbox, run_crop) {
                            (Some((bx, by, bw, bh)), Some(c)) => {
                                (c.x + bx * c.w, c.y + by * c.h, bw * c.w, bh * c.h)
                            }
                            (Some(b), None) => b,
                            (None, Some(c)) => (c.x, c.y, c.w, c.h),
                            (None, None) => return None,
                        };
                        Some((d.class_name.clone(), x, y, w, h))
                    })
                    .collect();
                let crop_tuple = run_crop.map(|c| (c.x, c.y, c.w, c.h));
                debug_store.insert(
                    &self.camera_id,
                    result.frame_jpegs,
                    result.raw_responses,
                    result.model,
                    result.detections.len(),
                    full_frame_jpeg,
                    motion_rects.to_vec(),
                    crop_tuple,
                    ollama_rects,
                );
            }
        }

        if !result.detections.is_empty() {
            tracing::debug!(
                camera = %self.camera_id,
                count = result.detections.len(),
                classes = ?result.detections.iter().map(|d| &d.class_name).collect::<Vec<_>>(),
                "ollama detections"
            );
        }

        // Use second frame (index 1) as thumbnail — inner frame
        let best_idx = if frames.len() > 1 { 1 } else { 0 };
        (result.detections, best_idx)
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
            let rects = &result.class_rects[i];
            if let Some(ref grid) = self.detection_grid {
                if !grid.record_rects(&self.camera_id, class, rects) {
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
            self.detector
                .report_positive_detection(&self.last_run_motion_rects);
        }
    }
}

fn build_detection_result(
    detections: &[Detection],
    filmstrip_jpegs: &[Vec<u8>],
    best_frame_idx: usize,
    crop: Option<NormalizedRect>,
) -> SegmentDetectionResult {
    let frame_jpeg = filmstrip_jpegs
        .get(best_frame_idx)
        .cloned()
        .or_else(|| filmstrip_jpegs.first().cloned())
        .unwrap_or_default();

    // Deduplicate by class — keep highest confidence per class,
    // and collect all bboxes for each class.
    let mut best_conf: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();
    let mut class_bboxes: std::collections::HashMap<&str, Vec<(f32, f32, f32, f32)>> =
        std::collections::HashMap::new();

    for d in detections {
        best_conf
            .entry(d.class_name.as_str())
            .and_modify(|c| {
                if d.confidence > *c {
                    *c = d.confidence;
                }
            })
            .or_insert(d.confidence);

        // Map Ollama bbox from crop space to full-frame space.
        // If no bbox from Ollama, fall back to the crop rect.
        let rect = match (d.bbox, crop) {
            (Some((bx, by, bw, bh)), Some(c)) => {
                // Ollama bbox is in cropped image coords → map to full frame
                (c.x + bx * c.w, c.y + by * c.h, bw * c.w, bh * c.h)
            }
            (Some(b), None) => b, // No crop, bbox is already in full frame
            (None, Some(c)) => (c.x, c.y, c.w, c.h), // No bbox, use crop
            (None, None) => continue, // No location info at all
        };

        class_bboxes
            .entry(d.class_name.as_str())
            .or_default()
            .push(rect);
    }

    let classes: Vec<String> = best_conf.keys().map(|k| k.to_string()).collect();
    let confidences: Vec<f32> = classes.iter().map(|c| best_conf[c.as_str()]).collect();
    let class_rects: Vec<Vec<(f32, f32, f32, f32)>> = classes
        .iter()
        .map(|c| class_bboxes.remove(c.as_str()).unwrap_or_default())
        .collect();

    SegmentDetectionResult {
        classes,
        confidences,
        class_rects,
        frame_jpeg,
    }
}

fn sample_indices(len: usize) -> Vec<usize> {
    if len <= 4 {
        (0..len).collect()
    } else {
        vec![0, len / 3, 2 * len / 3, len - 1]
    }
}

fn decode_to_mats_tagged(
    decoder: &CropDecoder,
    data: &[u8],
    duration_ns: u64,
    height: i32,
    crop: Option<NormalizedRect>,
    out: &mut Vec<(Mat, Option<NormalizedRect>)>,
) {
    for frame_data in &decoder.decode_segment(data, duration_ns) {
        if let Ok(mat) = Mat::from_slice(frame_data) {
            if let Ok(reshaped) = mat.reshape(3, height) {
                if let Ok(cloned) = reshaped.try_clone() {
                    out.push((cloned, crop));
                }
            }
        }
    }
}

fn subsample_tagged(
    frames: Vec<(Mat, Option<NormalizedRect>)>,
) -> Vec<(Mat, Option<NormalizedRect>)> {
    if frames.len() <= 4 {
        return frames;
    }
    let n = frames.len();
    [0, n / 3, 2 * n / 3, n - 1]
        .iter()
        .map(|&i| {
            let (ref mat, crop) = frames[i];
            (mat.try_clone().unwrap_or_default(), crop)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_rect_maps_to_unit_coords() {
        let r = Rect::new(80, 60, 160, 120);
        let n = normalize_rect(r, 320, 240);
        assert!((n.x - 0.25).abs() < 0.01);
        assert!((n.y - 0.25).abs() < 0.01);
        assert!((n.w - 0.50).abs() < 0.01);
        assert!((n.h - 0.50).abs() < 0.01);
    }

    #[test]
    fn union_rects_empty_returns_none() {
        assert!(union_rects_padded(&[], 0.2).is_none());
    }

    #[test]
    fn union_rects_single_rect_with_padding() {
        let r = NormalizedRect {
            x: 0.4,
            y: 0.4,
            w: 0.2,
            h: 0.2,
        };
        let u = union_rects_padded(&[r], 0.2).unwrap();
        // w=0.2, pad_x=0.04, so x=0.36, w=0.28
        assert!((u.x - 0.36).abs() < 0.01);
        assert!((u.w - 0.28).abs() < 0.01);
    }

    #[test]
    fn union_rects_clamps_to_bounds() {
        let r = NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 0.1,
            h: 0.1,
        };
        let u = union_rects_padded(&[r], 0.5).unwrap();
        assert!(u.x >= 0.0);
        assert!(u.y >= 0.0);
        assert!(u.x + u.w <= 1.0);
        assert!(u.y + u.h <= 1.0);
    }

    #[test]
    fn union_rects_merges_two_rects() {
        let rects = vec![
            NormalizedRect {
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            },
            NormalizedRect {
                x: 0.6,
                y: 0.6,
                w: 0.2,
                h: 0.2,
            },
        ];
        let u = union_rects_padded(&rects, 0.0).unwrap();
        // Union spans 0.1..0.8 in both axes = 0.7
        assert!((u.x - 0.1).abs() < 0.01);
        assert!((u.y - 0.1).abs() < 0.01);
        assert!((u.w - 0.7).abs() < 0.01);
        assert!((u.h - 0.7).abs() < 0.01);
    }

    #[test]
    fn union_rects_enforces_minimum_size() {
        let r = NormalizedRect {
            x: 0.5,
            y: 0.5,
            w: 0.01,
            h: 0.01,
        };
        let u = union_rects_padded(&[r], 0.0).unwrap();
        assert!(u.w >= MIN_CROP_FRACTION);
        assert!(u.h >= MIN_CROP_FRACTION);
    }

    #[test]
    fn crop_mat_extracts_region() {
        let mat = Mat::zeros(100, 200, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let region = NormalizedRect {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let cropped = crop_mat(&mat, &region).unwrap();
        assert_eq!(cropped.cols(), 100);
        assert_eq!(cropped.rows(), 50);
    }

    #[test]
    fn crop_mat_clamps_at_edge() {
        let mat = Mat::zeros(100, 200, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let region = NormalizedRect {
            x: 0.8,
            y: 0.8,
            w: 0.5,
            h: 0.5,
        };
        let cropped = crop_mat(&mat, &region).unwrap();
        // Should clamp: x=160, w=min(100, 200-160)=40; y=80, h=min(50, 100-80)=20
        assert_eq!(cropped.cols(), 40);
        assert_eq!(cropped.rows(), 20);
    }
}
