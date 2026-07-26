use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::analytics::motion_settings::{
    MotionSettingsStore, DEFAULT_MIN_CONTOUR_AREA, DEFAULT_VAR_THRESHOLD, MASK_CELLS, MASK_COLS,
    MASK_ROWS,
};
use crate::buffer::warm::{assemble_event, WriterMessage};
use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::locks::LockExt;
use crate::mqtt::{send_event, MqttEvent};
use crate::storage::{DetectionStore, EventRecord, EventRegistry, MotionEntry, MotionStore};

use super::decoder::{
    CropDecoder, DecodeOutcome, FrameDecoder, DETECTION_CROP_SIZE, THUMBNAIL_CROP_SIZE,
};
use super::detect_worker::{enqueue_job, DetectionJob};
use super::motion::{MotionBox, MotionDetector};
use super::run_tracker::{ClosedRun, RunTracker};

const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;

const MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);
const CROP_PADDING: f32 = 0.2;
const MIN_CROP_FRACTION: f32 = 0.15;

/// Consecutive zero-frame decodes tolerated before the decoder is declared
/// blind. A segment is one GOP and always opens on a keyframe, so a healthy
/// decode yields at least one I-frame — but pipe buffering can push a frame
/// past the read timeout into the next segment's read window, so a single
/// empty decode proves nothing. Only an unbroken streak does, and at roughly
/// one segment per second thirty of them is about half a minute of blindness:
/// long enough that no buffering hiccup explains it, short enough that little
/// motion is missed before the respawn.
const BLIND_DECODER_STREAK: u32 = 30;

/// Counts consecutive zero-frame decodes so an ffmpeg that consumes input but
/// emits nothing is caught. Without it the analyzer scores empty frame lists
/// forever, silently: the child is alive, so the liveness check never fires.
#[derive(Default)]
struct ZeroFrameTripwire {
    streak: u32,
}

impl ZeroFrameTripwire {
    /// Record one decode's frame count. Returns `true` when the streak reaches
    /// [`BLIND_DECODER_STREAK`], which also resets it — a decoder still blind
    /// after its respawn trips again rather than going quiet.
    fn observe(&mut self, frames: usize) -> bool {
        if frames > 0 {
            self.streak = 0;
            return false;
        }
        self.streak += 1;
        if self.streak >= BLIND_DECODER_STREAK {
            self.streak = 0;
            return true;
        }
        false
    }

    fn reset(&mut self) {
        self.streak = 0;
    }
}

#[derive(Clone, Copy)]
struct NormalizedRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// The whole frame in normalized coordinates. Used as the crop region for the
/// full-frame fallback (a frame with no motion crop, or a lighting-driven crop
/// that spans the entire frame) so the detection mask is applied consistently.
const FULL_FRAME: NormalizedRect = NormalizedRect {
    x: 0.0,
    y: 0.0,
    w: 1.0,
    h: 1.0,
};

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

/// Black out (set to RGB black) every pixel of `frame` that belongs to a
/// painted detection-mask cell. `frame` is a crop covering the normalized
/// full-frame region `crop`; the 16x12 detection mask is defined over the full
/// frame, so each painted cell's rectangle is intersected with the crop and
/// translated into the crop's own pixel space. Cells that fall entirely
/// outside the crop contribute nothing.
///
/// The vision model must never see a masked pixel regardless of crop geometry,
/// so intersections are rounded outward (start floored, end ceiled): a painted
/// cell is always fully covered even when its edges land between pixels.
fn apply_detection_mask(frame: &mut RgbFrame, crop: NormalizedRect, mask: &[bool]) {
    if mask.len() != MASK_CELLS
        || mask.iter().all(|&m| !m)
        || crop.w <= 0.0
        || crop.h <= 0.0
        || frame.width == 0
        || frame.height == 0
    {
        return;
    }
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    for row in 0..MASK_ROWS {
        for col in 0..MASK_COLS {
            if !mask[row * MASK_COLS + col] {
                continue;
            }
            // Cell rectangle in full-frame normalized coordinates.
            let cx0 = col as f32 / MASK_COLS as f32;
            let cx1 = (col + 1) as f32 / MASK_COLS as f32;
            let cy0 = row as f32 / MASK_ROWS as f32;
            let cy1 = (row + 1) as f32 / MASK_ROWS as f32;
            // Intersect with the crop region.
            let ix0 = cx0.max(crop.x);
            let ix1 = cx1.min(crop.x + crop.w);
            let iy0 = cy0.max(crop.y);
            let iy1 = cy1.min(crop.y + crop.h);
            if ix1 <= ix0 || iy1 <= iy0 {
                continue;
            }
            // Translate into crop-local pixel coordinates, rounding outward.
            let px0 = ((((ix0 - crop.x) / crop.w) * fw).floor() as i64).clamp(0, frame.width as i64)
                as usize;
            let px1 = ((((ix1 - crop.x) / crop.w) * fw).ceil() as i64).clamp(0, frame.width as i64)
                as usize;
            let py0 = ((((iy0 - crop.y) / crop.h) * fh).floor() as i64)
                .clamp(0, frame.height as i64) as usize;
            let py1 = ((((iy1 - crop.y) / crop.h) * fh).ceil() as i64).clamp(0, frame.height as i64)
                as usize;
            for py in py0..py1 {
                let start = (py * frame.width + px0) * 3;
                let end = (py * frame.width + px1) * 3;
                for b in &mut frame.data[start..end] {
                    *b = 0;
                }
            }
        }
    }
}

struct MotionSegment {
    seq: u64,
    data: Arc<Vec<u8>>,
    duration_ns: u64,
}

/// The JPEG thumbnails of one closed motion run, shared with the event that
/// carries them to warm storage.
type Filmstrip = Arc<Vec<Vec<u8>>>;

/// Frames kept per event once the run closes.
const FILMSTRIP_FRAMES: usize = 4;
/// Working size of an open run's accumulator. A run can last for hours, so
/// past this the strip is halved rather than grown.
const FILMSTRIP_ACCUMULATOR_CAP: usize = 8;

/// Thumbnails extracted so far for the motion run that is currently open.
/// Frames arrive batch by batch and belong to the run as a whole, not to any
/// single segment, so they live here until the run closes.
#[derive(Default)]
struct RunFilmstrip {
    frames: Vec<Vec<u8>>,
}

impl RunFilmstrip {
    /// Add a batch's frames, dropping every second frame whenever the
    /// accumulator outgrows its cap. Halving keeps the whole run covered at
    /// coarser spacing instead of truncating it to its beginning or end.
    fn push(&mut self, frames: Vec<Vec<u8>>) {
        self.frames.extend(frames);
        if self.frames.len() > FILMSTRIP_ACCUMULATOR_CAP {
            let mut keep = 0;
            self.frames.retain(|_| {
                keep += 1;
                keep % 2 == 1
            });
        }
    }

    /// Snapshot and reset, subsampled to at most [`FILMSTRIP_FRAMES`] frames
    /// spread from the first to the last. `None` when nothing was extracted.
    fn take(&mut self) -> Option<Filmstrip> {
        let frames = std::mem::take(&mut self.frames);
        if frames.is_empty() {
            return None;
        }
        Some(Arc::new(subsample_filmstrip(frames)))
    }
}

/// What this camera's color frames are extracted for. Detection needs the
/// vision model's input resolution; thumbnails alone are far cheaper to decode,
/// and with no consumer at all the crop decoder never runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameUse {
    None,
    Thumbnails,
    Detection,
}

impl FrameUse {
    fn of(records_events: bool, detects_objects: bool) -> Self {
        match (records_events, detects_objects) {
            (_, true) => Self::Detection,
            (true, false) => Self::Thumbnails,
            (false, false) => Self::None,
        }
    }

    fn crop_size(self) -> (u32, u32) {
        match self {
            Self::Detection => DETECTION_CROP_SIZE,
            _ => THUMBNAIL_CROP_SIZE,
        }
    }
}

fn subsample_filmstrip(frames: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let n = frames.len();
    if n <= FILMSTRIP_FRAMES {
        return frames;
    }
    let picks = [0, n / 3, 2 * n / 3, n - 1];
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, frame)| frame)
        .collect()
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
    /// Motion lifecycle events for the Home Assistant MQTT bridge. `None` when
    /// MQTT is disabled. Only ever `try_send`, never awaited: the analyzer is a
    /// blocking loop and must not stall on the bridge.
    pub mqtt_tx: Option<tokio::sync::mpsc::Sender<MqttEvent>>,
}

pub struct MotionAnalyzer {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    motion_store: MotionStore,
    detection_store: Option<DetectionStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    /// Watches for a decoder that consumes segments but returns no frames. The
    /// detector above is deliberately not part of the decoder, so a respawn
    /// leaves the learned MOG2 background model intact.
    zero_frames: ZeroFrameTripwire,
    detect_tx: Option<tokio::sync::mpsc::Sender<DetectionJob>>,
    event_registry: Option<EventRegistry>,
    last_processed: u64,
    motion_settings: MotionSettingsStore,
    /// Per-camera "detection mask": 16x12 row-major cells, `true` = blacked
    /// out of every frame sent to the vision model. Refreshed each tick in
    /// `sync_settings` so paint edits apply live, exactly like the movement
    /// mask and the sliders.
    detection_mask: Vec<bool>,
    segment_crops: HashMap<u64, NormalizedRect>,
    segment_motion_rects: HashMap<u64, Vec<NormalizedRect>>,
    run_tracker: RunTracker,
    frame_use: FrameUse,
    run_filmstrip: RunFilmstrip,
    event_tx: Option<tokio::sync::mpsc::Sender<WriterMessage>>,
    mqtt_tx: Option<tokio::sync::mpsc::Sender<MqttEvent>>,
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
        let detection_mask = settings
            .as_ref()
            .map(|s| s.detection_mask.clone())
            .unwrap_or_else(|| vec![false; MASK_CELLS]);
        let decoder = FrameDecoder::new()?;
        let frame_use = FrameUse::of(ctx.event_tx.is_some(), ctx.detect_tx.is_some());

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
            zero_frames: ZeroFrameTripwire::default(),
            detect_tx: ctx.detect_tx,
            event_registry: ctx.event_registry,
            last_processed,
            motion_settings: ctx.motion_settings,
            detection_mask,
            segment_crops: HashMap::new(),
            segment_motion_rects: HashMap::new(),
            run_tracker: RunTracker::new(ctx.post_padding, ctx.max_event_duration),
            frame_use,
            run_filmstrip: RunFilmstrip::default(),
            event_tx: ctx.event_tx,
            mqtt_tx: ctx.mqtt_tx,
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
            self.detection_mask = s.detection_mask;
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
            self.process_motion_runs(motion_segments);
        }

        // Emit after detection so runs that close in the same batch as their
        // motion segments still get object metadata.
        for (run, filmstrip) in closed_runs {
            self.emit_event(run, filmstrip);
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

    #[allow(clippy::type_complexity)]
    fn run_motion_analysis(
        &mut self,
        segments: Vec<PendingSegment>,
    ) -> Result<
        (Vec<MotionSegment>, Vec<(ClosedRun, Option<Filmstrip>)>),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let extract_frames = self.frame_use != FrameUse::None;
        let mut motion_segments = Vec::new();
        let mut closed_runs = Vec::new();

        // Lifecycle timing is monotonic: the analyzer runs near real time, so
        // the instant it observes a segment stands in for capture time. One
        // reading per poll batch is enough — batches are ~one segment.
        let now = Instant::now();

        for seg in segments {
            let (score, crop, motion_rects) = self.analyze_segment(&seg.data)?;
            self.publish_debug_maps();

            let has_motion = score >= MOTION_THRESHOLD;
            // The physical motion period, as opposed to the event chunking:
            // the duration cap closes one chunk and opens the next inside a
            // single `observe`, so a chunk boundary leaves the tracker open
            // and produces no MQTT transition.
            let was_open = self.run_tracker.is_open();
            // Whatever has accumulated belongs to the run that just closed:
            // this batch's own frames are extracted later, in
            // `process_motion_runs`.
            if let Some(run) = self.run_tracker.observe(seg.seq, has_motion, now) {
                let filmstrip = self.run_filmstrip.take();
                closed_runs.push((run, filmstrip));
            }
            match (was_open, self.run_tracker.is_open()) {
                (false, true) => self.send_motion_event(MqttEvent::MotionStart {
                    camera_id: self.camera_id.clone(),
                }),
                (true, false) => self.send_motion_event(MqttEvent::MotionEnd {
                    camera_id: self.camera_id.clone(),
                }),
                _ => {}
            }

            if has_motion {
                self.record_motion(seg.seq, seg.start_pts, seg.duration_ns, score);
                if extract_frames {
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

        Ok((motion_segments, closed_runs))
    }

    /// Assemble and hand off a finished event the moment its run closes.
    /// All segments in range are still hot and the metadata stores have not
    /// been cleaned up for them yet, so everything is read fresh here.
    fn emit_event(&self, run: ClosedRun, filmstrip: Option<Filmstrip>) {
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
                filmstrip,
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

    /// Hand a motion transition to the MQTT bridge. This runs on the blocking
    /// analyzer thread, so it must never await: a full or closed queue drops
    /// the event rather than stalling motion detection.
    fn send_motion_event(&self, event: MqttEvent) {
        if let Some(ref tx) = self.mqtt_tx {
            send_event(tx, event);
        }
    }

    fn flush_open_run(&mut self) {
        if let Some(run) = self.run_tracker.flush() {
            tracing::info!(
                camera = %self.camera_id,
                first_motion_seq = run.first_motion_seq,
                "flushing open motion event at shutdown"
            );
            let filmstrip = self.run_filmstrip.take();
            self.emit_event(run, filmstrip);
            // The run never saw its post-padding close, so HA would otherwise
            // be left with a motion sensor stuck ON across the restart.
            self.send_motion_event(MqttEvent::MotionEnd {
                camera_id: self.camera_id.clone(),
            });
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
    ) -> Result<
        (f32, Option<NormalizedRect>, Vec<NormalizedRect>),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let raw_frames = match self.decoder.decode_segment(data) {
            DecodeOutcome::Frames(frames) => frames,
            DecodeOutcome::Wedged => {
                tracing::warn!(
                    camera = %self.camera_id,
                    "decoder stopped consuming input, restarting"
                );
                // The respawned decoder starts from a clean slate, so the
                // streak the wedge interrupted says nothing about it.
                self.zero_frames.reset();
                self.decoder.kill();
                return Ok((0.0, None, Vec::new()));
            }
        };

        if self.zero_frames.observe(raw_frames.len()) {
            tracing::error!(
                camera = %self.camera_id,
                segments = BLIND_DECODER_STREAK,
                "decoder produced no frames for consecutive segments, restarting"
            );
            self.decoder.kill();
        }

        if raw_frames.is_empty() {
            return Ok((0.0, None, Vec::new()));
        }

        let (w, h) = (ANALYSIS_WIDTH as usize, ANALYSIS_HEIGHT as usize);
        let mut total_score = 0.0f32;
        let mut frame_count = 0u32;
        let mut all_rects = Vec::new();

        for frame_data in &raw_frames {
            let score = self.detector.process_frame(frame_data, w, h);
            total_score += score;
            frame_count += 1;
            for &r in self.detector.motion_bboxes() {
                all_rects.push(normalize_rect(r, ANALYSIS_WIDTH, ANALYSIS_HEIGHT));
            }
        }

        let crop = union_rects_padded(&all_rects, CROP_PADDING);

        Ok((total_score / frame_count as f32, crop, all_rects))
    }

    // --- Phase 2: Generic frame extraction + detection ---

    fn process_motion_runs(&mut self, segments: Vec<MotionSegment>) {
        let crop_decoder = match CropDecoder::new(
            self.config.sample_fps,
            self.frame_use.crop_size(),
        ) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(camera = %self.camera_id, error = %e, "failed to create crop decoder");
                return;
            }
        };

        let runs = group_contiguous_runs(segments);
        for run in runs {
            self.process_run(run, &crop_decoder);
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

    /// Extract, crop and JPEG-encode the color frames of one contiguous motion
    /// run. They become the filmstrip of the event the run belongs to, and —
    /// when object detection is on — a crop job for the global detection
    /// worker. Handing that job off never blocks: it is `try_send`-queued, and
    /// a full queue drops it with a warning, costing the object upgrade but
    /// never the event.
    fn process_run(&mut self, run: Vec<MotionSegment>, crop_decoder: &CropDecoder) {
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

        // Encode a full (uncropped) frame for debug overlay. Pick the first
        // tagged frame that has a crop (i.e. had motion). The detection mask
        // is blacked out here too so the debug UI shows exactly what the model
        // could not see. Only the detection debug UI reads it, so it is not
        // encoded at all without object detection.
        let full_frame_jpeg = self.detect_tx.as_ref().and_then(|_| {
            tagged_frames
                .iter()
                .find(|(_, crop)| crop.is_some())
                .and_then(|(frame, _)| {
                    let mut f = frame.clone();
                    apply_detection_mask(&mut f, FULL_FRAME, &self.detection_mask);
                    rgb_jpeg(&f)
                })
        });

        // Apply per-frame crops, then black out any painted detection-mask
        // cells so masked pixels reach neither the model nor a stored
        // thumbnail. A frame with no crop
        // falls back to the whole frame (region [0,0,1,1]); the mask is
        // applied in that region's coordinate space either way.
        let cropped: Vec<RgbFrame> = tagged_frames
            .iter()
            .map(|(frame, crop)| {
                let region = crop.unwrap_or(FULL_FRAME);
                // If the crop degenerates to nothing the full frame is used
                // instead, so the mask must be applied in full-frame space —
                // never a smaller region's — or the blackout lands on the
                // wrong pixels.
                let (mut out, region) = match crop_frame(frame, &region) {
                    Some(cropped) => (cropped, region),
                    None => (frame.clone(), FULL_FRAME),
                };
                apply_detection_mask(&mut out, region, &self.detection_mask);
                out
            })
            .collect();

        let filmstrip_jpegs: Vec<Vec<u8>> = cropped.iter().filter_map(rgb_jpeg).collect();

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
                    crop_jpegs: filmstrip_jpegs.clone(),
                    full_frame_jpeg,
                    motion_rects: all_motion_rects,
                    run_crop: run_crop.map(|c| (c.x, c.y, c.w, c.h)),
                },
            );
        }

        self.run_filmstrip.push(filmstrip_jpegs);
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

    /// A solid white frame whose dimensions are an exact multiple of the mask
    /// grid, so each 16x12 cell maps to a clean integer pixel block.
    fn white_frame(width: usize, height: usize) -> RgbFrame {
        RgbFrame {
            data: vec![255u8; width * height * 3],
            width,
            height,
        }
    }

    fn empty_mask() -> Vec<bool> {
        vec![false; MASK_CELLS]
    }

    /// True if the pixel at (col, row) is pure black.
    fn is_black(frame: &RgbFrame, col: usize, row: usize) -> bool {
        let i = (row * frame.width + col) * 3;
        frame.data[i] == 0 && frame.data[i + 1] == 0 && frame.data[i + 2] == 0
    }

    #[test]
    fn detection_mask_noop_when_empty() {
        let mut frame = white_frame(160, 120);
        apply_detection_mask(&mut frame, FULL_FRAME, &empty_mask());
        assert!(frame.data.iter().all(|&b| b == 255), "frame untouched");
    }

    #[test]
    fn detection_mask_full_frame_blacks_exact_cell() {
        // 160x120 => each of the 16x12 cells is a 10x10 pixel block.
        let mut frame = white_frame(160, 120);
        let mut mask = empty_mask();
        // Paint the top-left cell (col 0, row 0) and an interior cell (col 3,
        // row 2).
        mask[0] = true;
        mask[2 * MASK_COLS + 3] = true;
        apply_detection_mask(&mut frame, FULL_FRAME, &mask);

        // Top-left 10x10 block is black; the pixel just past it is not.
        assert!(is_black(&frame, 0, 0));
        assert!(is_black(&frame, 9, 9));
        assert!(!is_black(&frame, 10, 0));
        assert!(!is_black(&frame, 0, 10));

        // Interior cell (col 3, row 2) covers pixels x=30..40, y=20..30.
        assert!(is_black(&frame, 30, 20));
        assert!(is_black(&frame, 39, 29));
        assert!(!is_black(&frame, 29, 20));
        assert!(!is_black(&frame, 40, 20));
        assert!(!is_black(&frame, 30, 19));
    }

    #[test]
    fn detection_mask_intersects_partial_crop() {
        // Crop the right half of the full frame: x in [0.5,1.0]. The crop
        // frame is 80x120 covering full-frame columns 8..16.
        let crop = NormalizedRect {
            x: 0.5,
            y: 0.0,
            w: 0.5,
            h: 1.0,
        };
        let mut frame = white_frame(80, 120);
        let mut mask = empty_mask();
        // Cell (col 0, row 0) is entirely OUTSIDE the crop -> no effect.
        mask[0] = true;
        // Cell (col 8, row 0) is the first column INSIDE the crop; it maps to
        // crop-local x=0..10.
        mask[8] = true;
        apply_detection_mask(&mut frame, crop, &mask);

        // The out-of-crop cell painted nothing.
        // The in-crop cell blacked the leftmost 10px column of the crop.
        assert!(is_black(&frame, 0, 0));
        assert!(is_black(&frame, 9, 0));
        assert!(!is_black(&frame, 10, 0));
        // A cell far to the right of column 8 stays white.
        assert!(!is_black(&frame, 79, 0));
    }

    #[test]
    fn detection_mask_full_frame_crop_matches_uncropped() {
        // A lighting-driven crop that spans the whole frame ([0,0,1,1]) must
        // black the same pixels as the FULL_FRAME fallback.
        let mut frame = white_frame(160, 120);
        let mut mask = empty_mask();
        mask[MASK_COLS + 1] = true; // col 1, row 1 => x=10..20, y=10..20
        let spanning = NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
        apply_detection_mask(&mut frame, spanning, &mask);
        assert!(is_black(&frame, 10, 10));
        assert!(is_black(&frame, 19, 19));
        assert!(!is_black(&frame, 9, 10));
        assert!(!is_black(&frame, 20, 20));
    }

    /// One-byte stand-ins for JPEGs, tagged so subsampling order is visible.
    fn frames(tags: &[u8]) -> Vec<Vec<u8>> {
        tags.iter().map(|&t| vec![t]).collect()
    }

    fn tags(frames: &[Vec<u8>]) -> Vec<u8> {
        frames.iter().map(|f| f[0]).collect()
    }

    #[test]
    fn frames_are_extracted_for_events_without_object_detection() {
        assert_eq!(FrameUse::of(true, false), FrameUse::Thumbnails);
        assert_eq!(FrameUse::of(true, true), FrameUse::Detection);
        // Detection without warm storage still needs its input frames.
        assert_eq!(FrameUse::of(false, true), FrameUse::Detection);
        // Nothing consumes them: no crop decoding at all.
        assert_eq!(FrameUse::of(false, false), FrameUse::None);
    }

    #[test]
    fn thumbnails_only_decode_smaller_frames() {
        assert_eq!(FrameUse::Detection.crop_size(), DETECTION_CROP_SIZE);
        assert_eq!(FrameUse::Thumbnails.crop_size(), THUMBNAIL_CROP_SIZE);
        assert!(THUMBNAIL_CROP_SIZE.0 < DETECTION_CROP_SIZE.0);
    }

    #[test]
    fn run_filmstrip_accumulates_across_batches() {
        let mut strip = RunFilmstrip::default();
        strip.push(frames(&[1, 2]));
        strip.push(frames(&[3]));
        let taken = strip.take().unwrap();
        assert_eq!(tags(&taken), vec![1, 2, 3]);
    }

    #[test]
    fn run_filmstrip_take_resets_and_is_none_when_empty() {
        let mut strip = RunFilmstrip::default();
        assert!(strip.take().is_none());
        strip.push(frames(&[1]));
        assert!(strip.take().is_some());
        assert!(strip.take().is_none());
    }

    #[test]
    fn run_filmstrip_halves_past_the_cap() {
        let mut strip = RunFilmstrip::default();
        for batch in 0..6u8 {
            strip.push(frames(&[batch * 2, batch * 2 + 1]));
        }
        assert!(strip.frames.len() <= FILMSTRIP_ACCUMULATOR_CAP);
        let taken = strip.take().unwrap();
        assert_eq!(taken.len(), FILMSTRIP_FRAMES);
        // Coverage still spans the run: first frame kept, last batch present.
        assert_eq!(taken[0], vec![0]);
        assert!(taken[FILMSTRIP_FRAMES - 1][0] >= 8);
    }

    #[test]
    fn run_filmstrip_subsamples_spread_over_the_run() {
        let mut strip = RunFilmstrip::default();
        strip.push(frames(&[0, 1, 2, 3, 4, 5]));
        let taken = strip.take().unwrap();
        assert_eq!(tags(&taken), vec![0, 2, 4, 5]);
    }

    #[test]
    fn run_filmstrip_close_does_not_steal_the_next_runs_frames() {
        // Batch order in the analyzer: a run closes (take) before this batch's
        // own frames are extracted (push).
        let mut strip = RunFilmstrip::default();
        strip.push(frames(&[1, 2]));
        let closed = strip.take().unwrap();
        strip.push(frames(&[3]));
        assert_eq!(tags(&closed), vec![1, 2]);
        assert_eq!(tags(&strip.take().unwrap()), vec![3]);
    }

    #[test]
    fn zero_frame_tripwire_resets_on_any_decoded_frame() {
        let mut tripwire = ZeroFrameTripwire::default();
        for _ in 0..BLIND_DECODER_STREAK - 1 {
            assert!(!tripwire.observe(0));
        }
        assert!(!tripwire.observe(1), "a decoded frame clears the streak");
        for _ in 0..BLIND_DECODER_STREAK - 1 {
            assert!(!tripwire.observe(0), "streak restarts from zero");
        }
    }

    #[test]
    fn zero_frame_tripwire_trips_at_the_threshold_and_rearms() {
        let mut tripwire = ZeroFrameTripwire::default();
        for _ in 0..BLIND_DECODER_STREAK - 1 {
            assert!(!tripwire.observe(0));
        }
        assert!(
            tripwire.observe(0),
            "trips on the threshold-th empty decode"
        );
        // A decoder that stays blind after its respawn must trip again.
        for _ in 0..BLIND_DECODER_STREAK - 1 {
            assert!(!tripwire.observe(0));
        }
        assert!(tripwire.observe(0));
    }

    #[test]
    fn zero_frame_tripwire_reset_clears_a_partial_streak() {
        let mut tripwire = ZeroFrameTripwire::default();
        for _ in 0..BLIND_DECODER_STREAK - 1 {
            assert!(!tripwire.observe(0));
        }
        tripwire.reset();
        assert!(
            !tripwire.observe(0),
            "reset streak needs the full run again"
        );
    }

    #[test]
    fn detection_mask_ignores_wrong_length() {
        let mut frame = white_frame(160, 120);
        apply_detection_mask(&mut frame, FULL_FRAME, &[true, false, true]);
        assert!(frame.data.iter().all(|&b| b == 255));
    }
}
