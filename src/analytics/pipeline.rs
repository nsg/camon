use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::analytics::motion_settings::{
    MotionSettingsStore, DEFAULT_MIN_CONTOUR_AREA, DEFAULT_VAR_THRESHOLD,
};
use crate::buffer::warm::{assemble_event, WriterMessage};
use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::locks::LockExt;
use crate::storage::{DetectionStore, EventRecord, EventRegistry, MotionEntry, MotionStore};

use super::decoder::{CropDecoder, FrameDecoder};
use super::detect_worker::{enqueue_job, DetectionJob};
use super::motion::{MotionBox, MotionDetector};
use super::run_tracker::{ClosedRun, RunTracker};

const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;

const MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);
const CROP_PADDING: f32 = 0.2;
const MIN_CROP_FRACTION: f32 = 0.15;

#[derive(Clone, Copy)]
struct NormalizedRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn normalize_rect(r: MotionBox, frame_w: i32, frame_h: i32) -> NormalizedRect {
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

/// A raw 8-bit RGB frame (3 bytes per pixel, row-major, no padding), as
/// produced by the crop decoder's ffmpeg pipe.
#[derive(Clone)]
struct RgbFrame {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

/// Cut a normalized region out of a frame with pure row copying. The region
/// is clamped to the frame bounds; a region that leaves no visible area
/// yields `None`.
fn crop_frame(frame: &RgbFrame, region: &NormalizedRect) -> Option<RgbFrame> {
    let cols = frame.width as i32;
    let rows = frame.height as i32;
    if cols == 0 || rows == 0 {
        return None;
    }

    let x = ((region.x * cols as f32) as i32).max(0);
    let y = ((region.y * rows as f32) as i32).max(0);
    let w = ((region.w * cols as f32) as i32).min(cols - x);
    let h = ((region.h * rows as f32) as i32).min(rows - y);
    if w <= 0 || h <= 0 {
        return None;
    }

    let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        let start = (row * frame.width + x) * 3;
        data.extend_from_slice(&frame.data[start..start + w * 3]);
    }
    Some(RgbFrame {
        data,
        width: w,
        height: h,
    })
}

struct MotionSegment {
    seq: u64,
    data: Arc<Vec<u8>>,
    duration_ns: u64,
}

struct PendingSegment {
    seq: u64,
    data: Arc<Vec<u8>>,
    start_pts: u64,
    duration_ns: u64,
}

pub struct AnalyzerContext {
    pub camera_id: String,
    pub buffer: Arc<RwLock<HotBuffer>>,
    pub motion_store: MotionStore,
    pub detection_store: Option<DetectionStore>,
    /// Crop jobs for the global (serial) detection worker. `None` when
    /// object detection is disabled. The analyzer only ever `try_send`s —
    /// motion detection never stalls on the vision model.
    pub detect_tx: Option<tokio::sync::mpsc::Sender<DetectionJob>>,
    /// Recently written events, recorded here for the detection worker's
    /// post-hoc upgrade lookup. `None` when warm storage or detection is off.
    pub event_registry: Option<EventRegistry>,
    pub config: AnalyticsConfig,
    /// Deterministic per-camera motion settings (sensitivity, min object size,
    /// ignore mask). Shared so live edits apply without a restart.
    pub motion_settings: MotionSettingsStore,
    /// Finished events go to the warm writer over this channel. `None` when
    /// warm storage is disabled.
    pub event_tx: Option<tokio::sync::mpsc::Sender<WriterMessage>>,
    /// Pre-padding reach, in media PTS nanoseconds. Media timing — stays PTS.
    pub pre_padding_ns: u64,
    /// Post-padding window, as monotonic wall time. Lifecycle timing — Instant.
    pub post_padding: Duration,
    /// Duration cap per event chunk, as monotonic wall time. `Duration::ZERO`
    /// disables chunking. Lifecycle timing — Instant.
    pub max_event_duration: Duration,
}

pub struct MotionAnalyzer {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    detect_tx: Option<tokio::sync::mpsc::Sender<DetectionJob>>,
    event_registry: Option<EventRegistry>,
    last_processed: u64,
    motion_settings: MotionSettingsStore,
    segment_crops: HashMap<u64, NormalizedRect>,
    segment_motion_rects: HashMap<u64, Vec<NormalizedRect>>,
    run_tracker: RunTracker,
    event_tx: Option<tokio::sync::mpsc::Sender<WriterMessage>>,
    pre_padding_ns: u64,
}

impl MotionAnalyzer {
    fn new(ctx: AnalyzerContext) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Seed the detector from the persisted (or default) per-camera settings;
        // subsequent live edits are picked up each tick in `sync_settings`.
        let settings = ctx.motion_settings.get(&ctx.camera_id);
        let (var_threshold, min_contour_area) = settings
            .as_ref()
            .map(|s| (s.var_threshold, s.min_contour_area))
            .unwrap_or((DEFAULT_VAR_THRESHOLD, DEFAULT_MIN_CONTOUR_AREA));
        let mut detector = MotionDetector::new(var_threshold, min_contour_area);
        if let Some(s) = settings.as_ref() {
            detector.set_mask(&s.mask);
        }
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
            detect_tx: ctx.detect_tx,
            event_registry: ctx.event_registry,
            last_processed,
            motion_settings: ctx.motion_settings,
            segment_crops: HashMap::new(),
            segment_motion_rects: HashMap::new(),
            run_tracker: RunTracker::new(ctx.post_padding, ctx.max_event_duration),
            event_tx: ctx.event_tx,
            pre_padding_ns: ctx.pre_padding_ns,
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

            thread::sleep(POLL_INTERVAL);
        }

        self.flush_open_run();
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

    /// Pull the latest deterministic settings from the shared store and apply
    /// them to the detector. Cheap (a lock read + a 192-byte mask copy), run
    /// every tick so slider/mask edits take effect without a restart.
    fn sync_settings(&mut self) {
        if let Some(s) = self.motion_settings.get(&self.camera_id) {
            self.detector.set_var_threshold(s.var_threshold);
            self.detector.set_min_contour_area(s.min_contour_area);
            self.detector.set_mask(&s.mask);
        }
    }

    fn process_new_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.sync_settings();

        let (first_seq, last_seq) = {
            let buffer = self.buffer.read_recover();
            (buffer.first_sequence(), buffer.last_sequence())
        };

        self.cleanup_old_data(first_seq);
        let segments = self.collect_pending_segments(last_seq)?;
        let (motion_segments, closed_runs) = self.run_motion_analysis(segments)?;

        if !motion_segments.is_empty() {
            self.run_sampled_detections(motion_segments);
        }

        // Emit after detection so runs that close in the same batch as their
        // motion segments still get object metadata.
        for run in closed_runs {
            self.emit_event(run);
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
                let buffer = self.buffer.read_recover();
                buffer.get_segment_by_sequence(seq).map(|s| PendingSegment {
                    seq,
                    data: Arc::clone(&s.data),
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
    ) -> Result<(Vec<MotionSegment>, Vec<ClosedRun>), Box<dyn std::error::Error + Send + Sync>>
    {
        let has_detection = self.detect_tx.is_some() && self.detection_store.is_some();
        let capture_frames = self.detection_store.is_some();
        let mut motion_segments = Vec::new();
        let mut motion_frames: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut closed_runs = Vec::new();

        // Lifecycle timing is monotonic: the analyzer runs near real time, so
        // the instant it observes a segment stands in for capture time. One
        // reading per poll batch is enough — batches are ~one segment.
        let now = Instant::now();

        for seg in segments {
            let (score, crop, motion_rects, frame_jpeg) =
                self.analyze_segment(&seg.data, capture_frames)?;
            self.publish_debug_maps();

            let has_motion = score >= MOTION_THRESHOLD;
            if let Some(run) = self.run_tracker.observe(seg.seq, has_motion, now) {
                closed_runs.push(run);
            }

            if has_motion {
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

        Ok((motion_segments, closed_runs))
    }

    /// Assemble and hand off a finished event the moment its run closes.
    /// All segments in range are still hot and the metadata stores have not
    /// been cleaned up for them yet, so everything is read fresh here.
    fn emit_event(&self, run: ClosedRun) {
        let tx = match self.event_tx {
            Some(ref tx) => tx,
            None => return,
        };

        let event = {
            let buffer = self.buffer.read_recover();
            assemble_event(
                &buffer,
                self.detection_store.as_ref(),
                &self.camera_id,
                run.first_motion_seq,
                run.last_seq,
                run.min_start_seq,
                self.pre_padding_ns,
                run.continues,
            )
        };
        let event = match event {
            Some(event) => event,
            None => {
                tracing::warn!(
                    camera = %self.camera_id,
                    first_motion_seq = run.first_motion_seq,
                    "event segments no longer in hot buffer, skipping event"
                );
                return;
            }
        };

        let start_pts_ns = event.first_pts;
        let duration_ms = event.duration_ms() as u32;
        let has_objects = event.has_objects;

        // Events are durability-critical: block this analyzer thread until
        // the writer has room rather than dropping the event.
        if tx.blocking_send(WriterMessage::Event(event)).is_err() {
            tracing::error!(camera = %self.camera_id, "warm writer gone, event lost");
            return;
        }

        // Record the event AFTER enqueuing the write, so any upgrade the
        // detection worker derives from this record is guaranteed to reach
        // the writer behind the write itself (same channel, FIFO). See
        // `storage::event_registry` for the full race analysis.
        if let Some(ref registry) = self.event_registry {
            registry.record(
                &self.camera_id,
                EventRecord {
                    start_pts_ns,
                    duration_ms,
                    first_motion_seq: run.first_motion_seq,
                    last_seq: run.last_seq,
                    has_objects,
                    continues: run.continues,
                },
            );
        }
    }

    fn flush_open_run(&mut self) {
        if let Some(run) = self.run_tracker.flush() {
            tracing::info!(
                camera = %self.camera_id,
                first_motion_seq = run.first_motion_seq,
                "flushing open motion event at shutdown"
            );
            self.emit_event(run);
        }
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

    /// Publish the detector's pipeline-stage views for the debug UI:
    /// - stability: final motion mask (after opening + area filter)
    /// - raw: raw MOG2 foreground mask
    /// - no-shadow: alias of raw (the pure-Rust detector has no shadow class;
    ///   the stage name is kept so the API/UI stay stable)
    /// - morph: after morphological opening
    /// - background: the learned MOG2 background model
    fn publish_debug_maps(&mut self) {
        if let Some(jpeg) = self.detector.fg_mask().and_then(gray_jpeg) {
            self.motion_store.set_stability_map(&self.camera_id, jpeg);
        }
        let mut bg = Vec::new();
        if let Some((w, h)) = self.detector.background_into(&mut bg) {
            if let Some(jpeg) = gray_jpeg((&bg, w, h)) {
                self.motion_store.set_background_map(&self.camera_id, jpeg);
            }
        }
        if let Some(jpeg) = self.detector.raw_mask().and_then(gray_jpeg) {
            self.motion_store.set_raw_mog2_map(&self.camera_id, jpeg);
        }
        if let Some(jpeg) = self.detector.no_shadow_mask().and_then(gray_jpeg) {
            self.motion_store.set_no_shadow_map(&self.camera_id, jpeg);
        }
        if let Some(jpeg) = self.detector.morph_mask().and_then(gray_jpeg) {
            self.motion_store.set_morph_map(&self.camera_id, jpeg);
        }
    }

    fn record_motion(&mut self, seq: u64, start_pts: u64, duration_ns: u64, score: f32) {
        let mask_jpeg = self.detector.fg_mask().and_then(gray_jpeg);
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

        let (w, h) = (ANALYSIS_WIDTH as usize, ANALYSIS_HEIGHT as usize);
        let mut total_score = 0.0f32;
        let mut frame_count = 0u32;
        let mut all_rects = Vec::new();
        let mut last_frame: Option<&Vec<u8>> = None;

        for frame_data in &raw_frames {
            let score = self.detector.process_frame(frame_data, w, h);
            total_score += score;
            frame_count += 1;
            for &r in self.detector.motion_bboxes() {
                all_rects.push(normalize_rect(r, ANALYSIS_WIDTH, ANALYSIS_HEIGHT));
            }
            if capture_frame {
                last_frame = Some(frame_data);
            }
        }

        let crop = union_rects_padded(&all_rects, CROP_PADDING);

        let frame_jpeg = last_frame.and_then(|f| gray_jpeg((f.as_slice(), w, h)));

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
    ) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
        let indices = sample_indices(run.len());
        let mut all_frames: Vec<(RgbFrame, Option<NormalizedRect>)> = Vec::new();

        // Feed preceding segments to prime the decoder (no crop — not motion-positive)
        let first_seq = run[0].seq;
        if first_seq >= 3 {
            let buffer = self.buffer.read_recover();
            for prime_seq in (first_seq - 3)..first_seq {
                if let Some(seg) = buffer.get_segment_by_sequence(prime_seq) {
                    decode_to_frames_tagged(
                        crop_decoder,
                        &seg.data,
                        seg.duration_ns,
                        None,
                        &mut all_frames,
                    );
                }
            }
        }

        for &idx in &indices {
            let seg = &run[idx];
            let crop = self.segment_crops.get(&seg.seq).copied();
            decode_to_frames_tagged(
                crop_decoder,
                &seg.data,
                seg.duration_ns,
                crop,
                &mut all_frames,
            );
        }

        subsample_tagged(all_frames)
    }

    /// Prepare a crop job for one contiguous motion run and hand it to the
    /// global detection worker. This never blocks: the frames are extracted,
    /// cropped, and JPEG-encoded here, then `try_send`-queued. If the queue
    /// is full the job is dropped with a warning — the motion event still
    /// persists; only the object upgrade is lost.
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
            .and_then(|(frame, _)| rgb_jpeg(frame));

        // Apply per-frame crops
        let cropped: Vec<RgbFrame> = tagged_frames
            .iter()
            .map(|(frame, crop)| {
                crop.and_then(|r| crop_frame(frame, &r))
                    .unwrap_or_else(|| frame.clone())
            })
            .collect();

        let filmstrip_jpegs: Vec<Vec<u8>> = cropped.iter().filter_map(rgb_jpeg).collect();
        self.store_filmstrip_for_run(&run, &filmstrip_jpegs);

        // Remove consumed segment data
        for seg in &run {
            self.segment_crops.remove(&seg.seq);
            self.segment_motion_rects.remove(&seg.seq);
        }

        if let Some(ref tx) = self.detect_tx {
            enqueue_job(
                tx,
                DetectionJob {
                    camera_id: self.camera_id.clone(),
                    seqs: run.iter().map(|seg| seg.seq).collect(),
                    crop_jpegs: filmstrip_jpegs,
                    full_frame_jpeg,
                    motion_rects: all_motion_rects,
                    run_crop: run_crop.map(|c| (c.x, c.y, c.w, c.h)),
                },
            );
        }
    }

    fn store_filmstrip_for_run(&self, run: &[MotionSegment], jpegs: &[Vec<u8>]) {
        if let Some(ref ds) = self.detection_store {
            let filmstrip = Arc::new(jpegs.to_vec());
            for seg in run {
                ds.insert_filmstrip(&self.camera_id, seg.seq, Arc::clone(&filmstrip));
            }
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

fn decode_to_frames_tagged(
    decoder: &CropDecoder,
    data: &[u8],
    duration_ns: u64,
    crop: Option<NormalizedRect>,
    out: &mut Vec<(RgbFrame, Option<NormalizedRect>)>,
) {
    let width = decoder.width() as usize;
    let height = decoder.height() as usize;
    for frame_data in decoder.decode_segment(data, duration_ns) {
        // The pipe delivers exact fixed-size frames; anything else is a
        // torn read from a dying ffmpeg and gets skipped.
        if frame_data.len() == width * height * 3 {
            out.push((
                RgbFrame {
                    data: frame_data,
                    width,
                    height,
                },
                crop,
            ));
        }
    }
}

fn subsample_tagged(
    frames: Vec<(RgbFrame, Option<NormalizedRect>)>,
) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
    if frames.len() <= 4 {
        return frames;
    }
    let n = frames.len();
    [0, n / 3, 2 * n / 3, n - 1]
        .iter()
        .map(|&i| frames[i].clone())
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

/// JPEG quality for frames sent to the vision model and served to the UI.
/// High enough that compression artifacts don't cost the model detections.
const JPEG_QUALITY: u8 = 90;

fn encode_jpeg_raw(
    data: &[u8],
    width: usize,
    height: usize,
    color: image::ExtendedColorType,
) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    encoder
        .encode(data, width as u32, height as u32, color)
        .ok()?;
    Some(buf)
}

/// Encode a raw RGB frame as JPEG.
fn rgb_jpeg(frame: &RgbFrame) -> Option<Vec<u8>> {
    if frame.data.len() != frame.width * frame.height * 3 {
        return None;
    }
    encode_jpeg_raw(
        &frame.data,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgb8,
    )
}

/// Encode an 8-bit grayscale buffer (detector masks, background model) as
/// JPEG for the debug endpoints.
fn gray_jpeg((data, w, h): (&[u8], usize, usize)) -> Option<Vec<u8>> {
    if w == 0 || h == 0 || data.len() != w * h {
        return None;
    }
    encode_jpeg_raw(data, w, h, image::ExtendedColorType::L8)
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
        let r = MotionBox {
            x: 80,
            y: 60,
            width: 160,
            height: 120,
        };
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

    /// A 200x100 frame where each pixel encodes its own coordinates:
    /// R = column % 256, G = row, B = 0. Makes copy errors visible.
    fn coordinate_frame() -> RgbFrame {
        let (width, height) = (200usize, 100usize);
        let mut data = Vec::with_capacity(width * height * 3);
        for row in 0..height {
            for col in 0..width {
                data.extend_from_slice(&[(col % 256) as u8, row as u8, 0]);
            }
        }
        RgbFrame {
            data,
            width,
            height,
        }
    }

    #[test]
    fn crop_frame_extracts_region() {
        let frame = coordinate_frame();
        let region = NormalizedRect {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let cropped = crop_frame(&frame, &region).unwrap();
        assert_eq!(cropped.width, 100);
        assert_eq!(cropped.height, 50);
        assert_eq!(cropped.data.len(), 100 * 50 * 3);
        // Top-left pixel of the crop is source pixel (col 50, row 25).
        assert_eq!(&cropped.data[0..3], &[50, 25, 0]);
        // Bottom-right pixel of the crop is source pixel (col 149, row 74).
        let last = cropped.data.len() - 3;
        assert_eq!(&cropped.data[last..], &[149, 74, 0]);
    }

    #[test]
    fn crop_frame_clamps_at_edge() {
        let frame = coordinate_frame();
        let region = NormalizedRect {
            x: 0.8,
            y: 0.8,
            w: 0.5,
            h: 0.5,
        };
        let cropped = crop_frame(&frame, &region).unwrap();
        // Should clamp: x=160, w=min(100, 200-160)=40; y=80, h=min(50, 100-80)=20
        assert_eq!(cropped.width, 40);
        assert_eq!(cropped.height, 20);
        assert_eq!(&cropped.data[0..3], &[160, 80, 0]);
    }

    #[test]
    fn crop_frame_fully_outside_is_none() {
        let frame = coordinate_frame();
        let region = NormalizedRect {
            x: 1.0,
            y: 0.0,
            w: 0.5,
            h: 0.5,
        };
        assert!(crop_frame(&frame, &region).is_none());
    }

    #[test]
    fn crop_frame_empty_frame_is_none() {
        let frame = RgbFrame {
            data: Vec::new(),
            width: 0,
            height: 0,
        };
        let region = NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
        assert!(crop_frame(&frame, &region).is_none());
    }

    #[test]
    fn rgb_jpeg_round_trips_through_image_crate() {
        let frame = coordinate_frame();
        let jpeg = rgb_jpeg(&frame).unwrap();
        // JPEG magic bytes.
        assert_eq!(&jpeg[0..2], &[0xff, 0xd8]);
        let decoded = image::load_from_memory(&jpeg).unwrap();
        assert_eq!(decoded.width(), 200);
        assert_eq!(decoded.height(), 100);
    }

    #[test]
    fn rgb_jpeg_rejects_mismatched_buffer() {
        let frame = RgbFrame {
            data: vec![0; 10],
            width: 200,
            height: 100,
        };
        assert!(rgb_jpeg(&frame).is_none());
    }

    #[test]
    fn gray_jpeg_round_trips_through_image_crate() {
        let (w, h) = (32usize, 16usize);
        let data: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let jpeg = gray_jpeg((&data, w, h)).unwrap();
        assert_eq!(&jpeg[0..2], &[0xff, 0xd8]);
        let decoded = image::load_from_memory(&jpeg).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn gray_jpeg_rejects_bad_dimensions() {
        assert!(gray_jpeg((&[0u8; 10], 3, 3)).is_none());
        assert!(gray_jpeg((&[], 0, 0)).is_none());
    }
}
