//! Motion analysis: one blocking loop per camera, turning hot-buffer segments into motion
//! scores, event filmstrips and the crop jobs the vision model runs on.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::analytics::motion_settings::{
    MotionSettingsStore, DEFAULT_MIN_CONTOUR_AREA, DEFAULT_VAR_THRESHOLD, MASK_CELLS,
};
use crate::buffer::warm::{assemble_event, EventUpgrade, WriterMessage};
use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::locks::LockExt;
use crate::mqtt::{send_event, MqttEvent};
use crate::retry::RetrySchedule;
use crate::shutdown::{shortfall, who_stalled, DrainGate, DrainStep, Stalled, TAIL_DRAIN_BOUND};
use crate::storage::{
    DetectionDebugStore, DetectionStore, EventRegistry, MapKind, MotionEntry, MotionStore,
    UpgradeTarget,
};

use super::decoder::{
    CropDecoder, DecodeOutcome, FrameDecoder, DETECTION_CROP_SIZE, THUMBNAIL_CROP_SIZE,
};
use super::detect_worker::{DetectQueueSender, DetectionJob};
use super::motion::MotionDetector;
use super::run_tracker::{ClosedRun, RunTracker};

const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;

const MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);

mod decoder_slot;
mod framing;
mod sampling;
mod skips;
#[cfg(test)]
mod tests;

use decoder_slot::{
    ensure_long_lived, sleep_unless_shutdown, sleep_unless_shutdown_watching, DecoderSpawnRetry,
    LongLived, ZeroFrameTripwire, BLIND_DECODER_STREAK, DECODER_SPAWN_SCHEDULE, PRIMING_SEGMENTS,
};
use framing::{
    apply_detection_mask, crop_frame, normalize_rect, union_rects_padded, union_two_rects,
    NormalizedRect, RgbFrame, CROP_PADDING, FULL_FRAME,
};
pub(crate) use sampling::FILMSTRIP_FRAMES;
use sampling::{gray_jpeg, rgb_jpeg, sample_run_frames, Filmstrip, RunFilmstrip};
use skips::{merge_skips, SkipReporter, SkippedSegments};

struct MotionSegment {
    seq: u64,
    data: Arc<Vec<u8>>,
    duration_ns: u64,
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

/// One segment's motion verdict.
struct SegmentAnalysis {
    score: f32,
    crop: Option<NormalizedRect>,
    motion_rects: Vec<NormalizedRect>,
}

impl SegmentAnalysis {
    fn has_motion(&self) -> bool {
        self.score >= MOTION_THRESHOLD
    }
}

struct PendingSegment {
    seq: u64,
    data: Arc<Vec<u8>>,
    start_pts: u64,
    duration_ns: u64,
}

/// Cloneable so a failed construction can be retried with it. Every field is
/// either a handle (`Arc`, channel sender, store) or small config, so a clone
/// costs nothing worth avoiding.
#[derive(Clone)]
pub struct AnalyzerContext {
    pub camera_id: String,
    pub buffer: Arc<RwLock<HotBuffer>>,
    pub motion_store: MotionStore,
    pub detection_store: Option<DetectionStore>,
    /// The detector's debug view.
    pub debug_store: Option<DetectionDebugStore>,
    /// Crop jobs for the global (serial) detection worker. `None` when
    /// object detection is disabled. Sends never block — motion detection
    /// never stalls on the vision model.
    pub detect_tx: Option<DetectQueueSender>,
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
    debug_store: Option<DetectionDebugStore>,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    /// The camera's crop decoder, forked on its first motion and kept until its
    /// child dies — never per batch. Empty until then, and empty again after a
    /// fork that failed.
    crop_decoder: LongLived<CropDecoder>,
    /// Backoff and log escalation for a decoder that will not respawn.
    decoder_retry: DecoderSpawnRetry,
    /// Watches for a decoder that consumes segments but returns no frames. The
    /// detector above is deliberately not part of the decoder, so a respawn
    /// leaves the learned MOG2 background model intact.
    zero_frames: ZeroFrameTripwire,
    detect_tx: Option<DetectQueueSender>,
    event_registry: Option<EventRegistry>,
    last_processed: u64,
    /// Whether `last_processed` reflects a sequence this analyzer actually
    /// reached, as opposed to the estimate it started from. Gates the
    /// skipped-footage warnings; see [`MotionAnalyzer::report_skip`].
    observed_sequences: bool,
    skip_reporter: SkipReporter,
    motion_settings: MotionSettingsStore,
    /// Per-camera "detection mask": 16x12 row-major cells, `true` = blacked out of every frame
    /// sent to the vision model. Refreshed each tick in `sync_settings` so paint edits apply
    /// live, exactly like the movement mask and the sliders.
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
        let decoder = FrameDecoder::new()?;
        Ok(Self::with_decoder(ctx, decoder))
    }

    /// Everything but the fork. Split out so the shutdown tests can build an
    /// analyzer around a decoder that is already dead — see
    /// [`FrameDecoder::dead`] — without an ffmpeg on the box.
    fn with_decoder(ctx: AnalyzerContext, decoder: FrameDecoder) -> Self {
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
        let frame_use = FrameUse::of(ctx.event_tx.is_some(), ctx.detect_tx.is_some());

        // An estimate: the motion store only records motion-positive segments,
        // so an analyzed quiet stretch leaves no trace here.
        // `observed_sequences` keeps it from being reported as skipped footage.
        let last_processed = ctx
            .motion_store
            .last_sequence(&ctx.camera_id)
            .map(|s| s + 1)
            .unwrap_or(0);

        Self {
            camera_id: ctx.camera_id,
            buffer: ctx.buffer,
            motion_store: ctx.motion_store,
            detection_store: ctx.detection_store,
            debug_store: ctx.debug_store,
            config: ctx.config,
            detector,
            decoder,
            crop_decoder: LongLived::default(),
            decoder_retry: DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE),
            zero_frames: ZeroFrameTripwire::default(),
            detect_tx: ctx.detect_tx,
            event_registry: ctx.event_registry,
            last_processed,
            observed_sequences: false,
            skip_reporter: SkipReporter::default(),
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
        }
    }

    fn run(mut self, shutdown: Arc<AtomicBool>) {
        tracing::info!(camera = %self.camera_id, "motion analyzer started");

        while !shutdown.load(Ordering::Relaxed) {
            if self.tick(&shutdown) {
                thread::sleep(POLL_INTERVAL);
            }
        }

        // The stop flag only means a stop has *begun*: the camera is being
        // joined right now and the GOP it has in hand is still on its way, so
        // drain before flushing or the recording loses its last seconds.
        self.drain_tail(DrainGate::starting_at(Instant::now(), TAIL_DRAIN_BOUND));
        self.flush_open_run();
        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    /// One pass of the analyzer. Returns whether the caller should wait out the poll interval
    /// — a pass that gave up on the decoder has already waited its own respawn backoff.
    fn tick(&mut self, shutdown: &AtomicBool) -> bool {
        if let Some(ref debug_store) = self.debug_store {
            debug_store.expire_unwatched(&self.camera_id);
        }
        self.release_idle_crop_frames();

        // This check covers only the frame decoder's fork directly below. The
        // crop decoder forks at the far end of the pass, so it is handed the
        // flag rather than this reading of it; see [`ensure_long_lived`].
        if shutdown.load(Ordering::Relaxed) || !self.ensure_decoder_alive(shutdown) {
            return false;
        }

        if let Err(e) = self.process_new_segments(Some(shutdown)) {
            tracing::error!(
                camera = %self.camera_id,
                error = %e,
                "motion analysis error"
            );
        }
        true
    }

    /// Phase 2 of the stop: keep analyzing until the camera's terminal watermark has been
    /// consumed, so the tail it pushed on its way out lands in the event about to be flushed.
    fn drain_tail(&mut self, gate: DrainGate) {
        let mut said_the_decoder_was_gone = false;
        loop {
            // A dead decoder is not respawned here (no forking during a drain), but the loop
            // still waits: `flush_open_run` closes the run through where the camera says it
            // stopped, and leaving early would cut the recording a GOP short of its tail.
            let decoding = self.decoder.is_alive();
            if decoding {
                if let Err(e) = self.process_new_segments(None) {
                    tracing::error!(
                        camera = %self.camera_id,
                        error = %e,
                        "motion analysis error while draining"
                    );
                }
            } else if !said_the_decoder_was_gone {
                said_the_decoder_was_gone = true;
                tracing::warn!(
                    camera = %self.camera_id,
                    last_analyzed = self.last_processed.saturating_sub(1),
                    "decoder gone at shutdown; waiting out the camera so the recording keeps its \
                     tail, but nothing past this sequence is scored for motion or objects"
                );
            }

            // Without a decoder nothing more will ever be consumed, so this
            // analyzer is as caught up as it is ever going to be: all it is
            // waiting for is the camera to say where it stopped.
            let position = if decoding {
                self.last_processed
            } else {
                u64::MAX
            };
            let terminal = self.buffer.read_recover().terminal_watermark();
            match gate.step(terminal, position, Instant::now()) {
                DrainStep::Drained => return,
                DrainStep::Abandoned => {
                    let ran_out_of = match who_stalled(terminal) {
                        Stalled::Consumer => {
                            "the analyzer could not catch up with the camera's last segment before \
                             the shutdown drain bound; the tail of this event is missing"
                        }
                        Stalled::Camera => {
                            "gave up waiting for a camera that never finished stopping; whatever \
                             it records past this point is not in the event"
                        }
                    };
                    tracing::warn!(
                        camera = %self.camera_id,
                        last_processed = self.last_processed,
                        // Measured from where scoring stopped, not from the
                        // position handed to the gate: a dead decoder reports
                        // itself finished to end the wait.
                        segments_abandoned = shortfall(terminal, self.last_processed),
                        "{ran_out_of}"
                    );
                    return;
                }
                DrainStep::Continue => thread::sleep(POLL_INTERVAL),
            }
        }
    }

    fn ensure_decoder_alive(&mut self, shutdown: &AtomicBool) -> bool {
        self.ensure_decoder_alive_with(shutdown, FrameDecoder::new)
    }

    /// Replace the frame decoder if its child is gone, waiting out the backoff here when the
    /// fork fails — which is why [`MotionAnalyzer::tick`] returns without a poll of its own
    /// afterwards.
    fn ensure_decoder_alive_with(
        &mut self,
        shutdown: &AtomicBool,
        spawn: impl FnOnce() -> Result<FrameDecoder, std::io::Error>,
    ) -> bool {
        if self.decoder.is_alive() {
            return true;
        }
        tracing::warn!(camera = %self.camera_id, "decoder process died, restarting");
        match spawn() {
            Ok(d) => {
                self.decoder = d;
                self.decoder_retry.succeeded();
                true
            }
            Err(e) => {
                let (delay, report) = self.decoder_retry.failed();
                if let Some(attempts) = report {
                    tracing::error!(
                        camera = %self.camera_id,
                        error = %e,
                        attempts,
                        retry_in_secs = delay.as_secs(),
                        "failed to restart decoder"
                    );
                }
                sleep_unless_shutdown_watching(delay, shutdown, || {
                    self.release_idle_crop_frames();
                });
                false
            }
        }
    }

    /// Whether anybody is looking at this camera's detection debug view. The
    /// answer comes from the last time the API was asked for it, exactly as
    /// [`MotionStore::map_wanted`] answers for a stage overlay.
    fn debug_view_wanted(&self) -> bool {
        self.debug_store
            .as_ref()
            .is_some_and(|store| store.wanted(&self.camera_id))
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

    /// `stop` is the crop decoder's fork window; `None` — the drain's — closes
    /// it for good. See [`ensure_long_lived`].
    fn process_new_segments(
        &mut self,
        stop: Option<&AtomicBool>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.sync_settings();

        let (first_seq, last_seq) = {
            let buffer = self.buffer.read_recover();
            (buffer.first_sequence(), buffer.last_sequence())
        };

        let aged_out = self.cleanup_old_data(first_seq);
        let (segments, evicted) = self.collect_pending_segments(last_seq)?;
        // Both losses are the same event seen at two moments of the same poll,
        // so they are reported together rather than as two warnings.
        if let Some(skipped) = merge_skips(aged_out, evicted) {
            self.report_skip(skipped);
        }
        let (motion_segments, closed_runs) = self.run_motion_analysis(segments)?;

        if !motion_segments.is_empty() {
            self.process_motion_runs(motion_segments, stop);
        }

        // Dispatch detections first so a closing run's registry record knows every covering job.
        for (run, filmstrip) in closed_runs {
            self.emit_event(run, filmstrip);
        }

        Ok(())
    }

    /// Drop metadata for segments the hot buffer no longer holds, returning
    /// what aged out before the analyzer reached it.
    fn cleanup_old_data(&mut self, first_seq: u64) -> Option<SkippedSegments> {
        if first_seq > 0 {
            self.motion_store.cleanup(&self.camera_id, first_seq);
            if let Some(ref ds) = self.detection_store {
                ds.cleanup(&self.camera_id, first_seq);
            }
            self.segment_crops.retain(|&seq, _| seq >= first_seq);
            self.segment_motion_rects.retain(|&seq, _| seq >= first_seq);
        }
        let skipped = SkippedSegments::between(self.last_processed, first_seq)?;
        self.last_processed = first_seq;
        Some(skipped)
    }

    /// Report footage that was never analyzed — but only once the analyzer has actually
    /// observed a sequence. Until then `last_processed` is a reconstruction from the motion
    /// store, so an early range would be invented, not measured.
    fn report_skip(&mut self, skipped: SkippedSegments) {
        if !self.observed_sequences {
            tracing::debug!(
                camera = %self.camera_id,
                segments = skipped.count,
                "skipping segments predating the analyzer's first pass"
            );
            return;
        }
        if let Some(total) = self.skip_reporter.record(skipped, Instant::now()) {
            tracing::warn!(
                camera = %self.camera_id,
                segments = total.count,
                from_seq = total.from_seq,
                to_seq = total.to_seq,
                "analyzer fell behind, segments passed through the hot buffer unanalyzed"
            );
        }
    }

    /// Read every pending segment still resident, along with any that were
    /// evicted before this loop reached them.
    #[allow(clippy::type_complexity)]
    fn collect_pending_segments(
        &self,
        last_seq: u64,
    ) -> Result<
        (Vec<PendingSegment>, Option<SkippedSegments>),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut segments = Vec::new();
        // The hot buffer can evict while this loop runs: `first_sequence` was sampled before
        // it, and each segment is fetched under its own lock.
        let mut evicted = Vec::new();
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
            match segment {
                Some(seg) => segments.push(seg),
                None => evicted.push(seq),
            }
        }
        Ok((segments, SkippedSegments::of(&evicted)))
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

        let observed_at = batch_instants(&segments, Instant::now());

        for (seg, now) in segments.into_iter().zip(observed_at) {
            let analysis = match self.analyze_segment(&seg.data)? {
                Some(analysis) => analysis,
                None => {
                    // Not scored quiet — see [`SegmentAnalysis`]. The
                    // zero-frame tripwire notices a decoder that stays blind.
                    tracing::debug!(
                        camera = %self.camera_id,
                        sequence = seg.seq,
                        "segment decoded no frames, not analyzed"
                    );
                    self.observed_sequences = true;
                    self.last_processed = seg.seq + 1;
                    continue;
                }
            };
            publish_debug_maps(&self.motion_store, &self.camera_id, &self.detector);

            let has_motion = analysis.has_motion();
            let SegmentAnalysis {
                score,
                crop,
                motion_rects,
            } = analysis;
            // Whatever has accumulated belongs to the run that just closed:
            // this batch's own frames are extracted later, in
            // `process_motion_runs`.
            if let Some(run) = self.observe_run(seg.seq, has_motion, now) {
                let filmstrip = self.run_filmstrip.take();
                closed_runs.push((run, filmstrip));
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

            self.observed_sequences = true;
            self.last_processed = seg.seq + 1;
        }

        Ok((motion_segments, closed_runs))
    }

    /// Assemble and hand off a finished event the moment its run closes; all segments in range
    /// are still hot and the metadata stores not yet cleaned up, so everything is read fresh
    /// here.
    fn emit_event(&self, run: ClosedRun, filmstrip: Option<Filmstrip>) {
        let tx = match self.event_tx {
            Some(ref tx) => tx,
            None => return,
        };

        // Every path below that returns without committing drops this, which
        // abandons the record — right, because no file ever appears under
        // this identity on those paths.
        let pending = self.event_registry.as_ref().map(|registry| {
            registry.open(
                &self.camera_id,
                run.first_motion_seq,
                run.last_seq,
                run.continues,
            )
        });

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

        // The write is in the channel, so upgrades the detection worker sends from here on
        // arrive behind it (FIFO).
        let Some(verdict) =
            pending.and_then(|pending| pending.commit(start_pts_ns, duration_ms, has_objects))
        else {
            return;
        };
        let upgrade = EventUpgrade::for_event(
            UpgradeTarget {
                start_pts_ns,
                duration_ms,
                continues: run.continues,
            },
            verdict,
        );
        // Losing this costs the event twelve days of retention, so it gets the
        // same blocking send the write itself did.
        if tx.blocking_send(WriterMessage::Upgrade(upgrade)).is_err() {
            tracing::error!(
                camera = %self.camera_id,
                "warm writer gone, object upgrade lost: the event keeps movement retention"
            );
        }
    }

    /// Feed one scored segment to the run tracker and send whatever the motion sensor owes as a
    /// result, returning the chunk that closed (if any).
    fn observe_run(&mut self, seq: u64, has_motion: bool, now: Instant) -> Option<ClosedRun> {
        let before = self.run_tracker.motion_period();
        let closed = self.run_tracker.observe(seq, has_motion, now);
        let after = self.run_tracker.motion_period();
        if before.is_some() && before != after {
            self.send_motion_event(MqttEvent::MotionEnd {
                camera_id: self.camera_id.clone(),
            });
        }
        if after.is_some() && before != after {
            self.send_motion_event(MqttEvent::MotionStart {
                camera_id: self.camera_id.clone(),
            });
        }
        closed
    }

    /// Hand a motion transition to the MQTT bridge. This runs on the blocking
    /// analyzer thread, so it must never await: a full or closed queue drops
    /// the event rather than stalling motion detection.
    fn send_motion_event(&self, event: MqttEvent) {
        if let Some(ref tx) = self.mqtt_tx {
            send_event(tx, event);
        }
    }

    /// Close whatever run is still open as a complete event, without waiting out its
    /// post-padding.
    fn flush_open_run(&mut self) {
        let through = self.buffer.read_recover().last_sequence().checked_sub(1);
        let was_moving = self.run_tracker.motion_period().is_some();
        if let Some(run) = self.run_tracker.flush(through) {
            tracing::info!(
                camera = %self.camera_id,
                first_motion_seq = run.first_motion_seq,
                "flushing open motion event at shutdown"
            );
            let filmstrip = self.run_filmstrip.take();
            self.emit_event(run, filmstrip);
        }
        // The run never saw its post-padding close, so nothing else clears the motion sensor
        // before camon comes back.
        if was_moving {
            self.send_motion_event(MqttEvent::MotionEnd {
                camera_id: self.camera_id.clone(),
            });
        }
    }

    fn record_motion(&mut self, seq: u64, start_pts: u64, duration_ns: u64, score: f32) {
        let mask_jpeg = self.detector.fg_mask().and_then(gray_jpeg);
        self.motion_store.insert(
            &self.camera_id,
            MotionEntry::spanning(seq, start_pts, duration_ns, score, mask_jpeg),
        );
        tracing::debug!(
            camera = %self.camera_id,
            sequence = seq,
            score = format!("{:.3}", score),
            "motion detected"
        );
    }

    /// Score one segment, or `None` when the decoder produced no frames for it
    /// — see [`SegmentAnalysis`].
    fn analyze_segment(
        &mut self,
        data: &Arc<Vec<u8>>,
    ) -> Result<Option<SegmentAnalysis>, Box<dyn std::error::Error + Send + Sync>> {
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
                return Ok(None);
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
            return Ok(None);
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

        Ok(Some(SegmentAnalysis {
            score: total_score / frame_count as f32,
            crop,
            motion_rects: all_rects,
        }))
    }

    fn process_motion_runs(&mut self, segments: Vec<MotionSegment>, stop: Option<&AtomicBool>) {
        let runs = group_contiguous_runs(segments);
        let (sample_fps, crop_size) = (self.config.sample_fps, self.frame_use.crop_size());
        // The slot is emptied for the batch (analyzer and decoder are needed
        // mutably at once) and must be refilled on the way out, or the child
        // dies and per-batch forking silently returns.
        let mut crop = std::mem::take(&mut self.crop_decoder);
        if ensure_long_lived(
            &mut crop,
            stop,
            &self.camera_id,
            CropDecoder::is_alive,
            || CropDecoder::new(sample_fps, crop_size),
        ) {
            for run in runs {
                self.process_run(run, &mut crop);
            }
        }
        self.crop_decoder = crop;
    }

    /// Let go of whatever the crop decoder emitted after the last motion batch stopped reading.
    /// Called at the top of every pass and at every wakeup of the respawn backoff — between
    /// them, every wait this analyzer sets for itself.
    fn release_idle_crop_frames(&self) -> usize {
        let Some(decoder) = self.crop_decoder.decoder.as_ref() else {
            return 0;
        };
        let dropped = decoder.drain();
        if dropped > 0 {
            tracing::debug!(
                camera = %self.camera_id,
                frames = dropped,
                "released the frames the last motion batch left in the crop decoder"
            );
        }
        dropped
    }

    /// Feed the segments preceding `first_seq` to `decode`, so a freshly forked ffmpeg spends
    /// its stream probe on footage nobody wants. The pictures predate the motion, so whatever
    /// comes out of them is dropped.
    fn prime_with<D>(
        &self,
        crop: &mut LongLived<D>,
        first_seq: u64,
        mut decode: impl FnMut(&mut D, &Arc<Vec<u8>>, u64),
    ) {
        if crop.primed() {
            return;
        }
        let Some(decoder) = crop.decoder.as_mut() else {
            return;
        };
        let segments: Vec<(Arc<Vec<u8>>, u64)> = match first_seq.checked_sub(PRIMING_SEGMENTS) {
            Some(from) => {
                let buffer = self.buffer.read_recover();
                (from..first_seq)
                    .filter_map(|seq| buffer.get_segment_by_sequence(seq))
                    .map(|seg| (Arc::clone(&seg.data), seg.duration_ns))
                    .collect()
            }
            None => Vec::new(),
        };
        for (data, duration_ns) in &segments {
            decode(decoder, data, *duration_ns);
        }
        crop.primed_with += segments.len() as u64;
    }

    /// Decode the sampled segments of one run down to the handful of frames
    /// [`pick_four`](sampling::pick_four) can still use, holding no more than
    /// [`RUN_FRAME_ACCUMULATOR_CAP`] of them at once.
    fn extract_run_frames(
        &self,
        run: &[MotionSegment],
        crop: &mut LongLived<CropDecoder>,
    ) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
        self.prime_with(crop, run[0].seq, |decoder, data, duration_ns| {
            decoder.decode_segment(data, duration_ns, |_| {});
        });
        let Some(decoder) = crop.decoder.as_mut() else {
            return Vec::new();
        };
        let stale = decoder.drain();
        if stale > 0 {
            tracing::debug!(
                camera = %self.camera_id,
                frames = stale,
                "dropped frames predating the motion run"
            );
        }

        let (width, height) = (decoder.width() as usize, decoder.height() as usize);
        sample_run_frames(
            run,
            &self.segment_crops,
            width,
            height,
            |data, duration_ns, sink| decoder.decode_segment(data, duration_ns, sink),
        )
    }

    /// The full (uncropped) frame the detection debug view draws its overlay on: the first
    /// frame of the run that has a crop, i.e. that had motion, with the detection mask blacked
    /// out so the view shows exactly what the model could not see.
    fn debug_overlay_frame(
        &self,
        tagged_frames: &[(RgbFrame, Option<NormalizedRect>)],
    ) -> Option<Arc<Vec<u8>>> {
        if self.detect_tx.is_none() || !self.debug_view_wanted() {
            return None;
        }
        tagged_frames
            .iter()
            .find(|(_, crop)| crop.is_some())
            .and_then(|(frame, _)| {
                let mut f = frame.clone();
                apply_detection_mask(&mut f, FULL_FRAME, &self.detection_mask);
                rgb_jpeg(&f)
            })
            .map(Arc::new)
    }

    /// Extract, crop and JPEG-encode the color frames of one contiguous motion run.
    fn process_run(&mut self, run: Vec<MotionSegment>, crop: &mut LongLived<CropDecoder>) {
        if run.is_empty() {
            return;
        }

        let tagged_frames = self.extract_run_frames(&run, crop);
        if tagged_frames.is_empty() {
            return;
        }

        self.process_run_frames(run, tagged_frames);
    }

    /// Everything the run's frames are turned into, once they have been extracted. Split from
    /// the extraction above so what the strip, the job and the debug overlay are built out of
    /// can be driven from a test without an ffmpeg to decode with.
    fn process_run_frames(
        &mut self,
        run: Vec<MotionSegment>,
        tagged_frames: Vec<(RgbFrame, Option<NormalizedRect>)>,
    ) {
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

        let full_frame_jpeg = self.debug_overlay_frame(&tagged_frames);

        // Crop, then black out painted detection-mask cells so masked pixels reach neither the
        // model nor a stored thumbnail.
        let cropped: Vec<RgbFrame> = tagged_frames
            .iter()
            .map(|(frame, crop)| {
                let region = crop.unwrap_or(FULL_FRAME);
                let (mut out, region) = match crop_frame(frame, &region) {
                    Some(cropped) => (cropped, region),
                    None => (frame.clone(), FULL_FRAME),
                };
                apply_detection_mask(&mut out, region, &self.detection_mask);
                out
            })
            .collect();

        let filmstrip_jpegs: Vec<Vec<u8>> = cropped.iter().filter_map(rgb_jpeg).collect();

        for seg in &run {
            self.segment_crops.remove(&seg.seq);
            self.segment_motion_rects.remove(&seg.seq);
        }

        if let Some(ref tx) = self.detect_tx {
            tx.send(DetectionJob {
                camera_id: self.camera_id.clone(),
                seqs: run.iter().map(|seg| seg.seq).collect(),
                // Copied once for the job's independent ownership; past this
                // point the model, thumbnail and debug store all share these
                // bytes by handle.
                crop_jpegs: filmstrip_jpegs.iter().cloned().map(Arc::new).collect(),
                full_frame_jpeg,
                motion_rects: all_motion_rects,
                run_crop: run_crop.map(|c| (c.x, c.y, c.w, c.h)),
                // Stamped by the queue as it accepts the job.
                verdict_id: None,
            });
        }

        self.run_filmstrip.push(filmstrip_jpegs);
    }
}

/// Monotonic capture instant per segment of one batch: the last segment ends at `now` and the
/// others are placed back along their own media durations, so a burst of backlog spans the same
/// time the footage did and post-padding and the duration cap behave the same either way.
fn batch_instants(segments: &[PendingSegment], now: Instant) -> Vec<Instant> {
    let mut times = Vec::with_capacity(segments.len());
    let mut at = now;
    for seg in segments.iter().rev() {
        times.push(at);
        at = at
            .checked_sub(Duration::from_nanos(seg.duration_ns))
            .unwrap_or(at);
    }
    times.reverse();
    times
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

/// Publish the detector's pipeline-stage views for the debug UI: stability (final mask), raw
/// MOG2, no-shadow (alias of raw), morph, and background.
fn publish_debug_maps(store: &MotionStore, camera_id: &str, detector: &MotionDetector) {
    if store.map_wanted(camera_id, MapKind::Stability) {
        if let Some(jpeg) = detector.fg_mask().and_then(gray_jpeg) {
            store.set_map(camera_id, MapKind::Stability, jpeg);
        }
    }
    if store.map_wanted(camera_id, MapKind::Background) {
        let mut bg = Vec::new();
        if let Some((w, h)) = detector.background_into(&mut bg) {
            if let Some(jpeg) = gray_jpeg((&bg, w, h)) {
                store.set_map(camera_id, MapKind::Background, jpeg);
            }
        }
    }
    // `no_shadow_mask()` is the raw mask, so the two stages share one encode —
    // and the motion overlay asks for both.
    let raw_wanted = store.map_wanted(camera_id, MapKind::RawMog2);
    let no_shadow_wanted = store.map_wanted(camera_id, MapKind::NoShadow);
    if raw_wanted || no_shadow_wanted {
        if let Some(jpeg) = detector.raw_mask().and_then(gray_jpeg) {
            if no_shadow_wanted {
                store.set_map(camera_id, MapKind::NoShadow, jpeg.clone());
            }
            if raw_wanted {
                store.set_map(camera_id, MapKind::RawMog2, jpeg);
            }
        }
    }
    if store.map_wanted(camera_id, MapKind::Morph) {
        if let Some(jpeg) = detector.morph_mask().and_then(gray_jpeg) {
            store.set_map(camera_id, MapKind::Morph, jpeg);
        }
    }
}

/// Build until it succeeds or shutdown is requested; `None` means only the
/// latter. The sleep between attempts is shutdown-aware, so a camera stuck in
/// here never holds the drain up.
fn build_with_retry<T, E: std::fmt::Display>(
    camera_id: &str,
    what: &str,
    shutdown: &AtomicBool,
    schedule: RetrySchedule,
    mut build: impl FnMut() -> Result<T, E>,
) -> Option<T> {
    let mut retry = DecoderSpawnRetry::new(schedule);
    while !shutdown.load(Ordering::Relaxed) {
        match build() {
            Ok(built) => return Some(built),
            Err(e) => {
                let (delay, report) = retry.failed();
                if let Some(attempts) = report {
                    tracing::error!(
                        camera = %camera_id,
                        error = %e,
                        attempts,
                        retry_in_secs = delay.as_secs(),
                        "failed to create {what}, retrying"
                    );
                }
                sleep_unless_shutdown(delay, shutdown);
            }
        }
    }
    None
}

/// Construction is retried rather than fatal: it fails for the same reasons a
/// running decoder dies, and a camera left with no analyzer records nothing in
/// event mode for the rest of the process.
pub fn spawn_analyzer(
    ctx: AnalyzerContext,
    shutdown: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(analyzer_body(ctx, shutdown))
}

/// The analyzer's whole life as a closure, for callers that spawn it themselves —
/// [`crate::supervise::Supervisor::critical_blocking`] does, so that an analyzer which dies is
/// noticed while camon is running rather than by whoever joins its handle at the stop.
pub fn analyzer_body(
    ctx: AnalyzerContext,
    shutdown: Arc<AtomicBool>,
) -> impl FnOnce() + Send + 'static {
    let camera_id = ctx.camera_id.clone();
    move || {
        let analyzer = build_with_retry(
            &camera_id,
            "motion analyzer",
            &shutdown,
            DECODER_SPAWN_SCHEDULE,
            || MotionAnalyzer::new(ctx.clone()),
        );
        // The analyzer holds its own clone of every sender in `ctx`. Shutdown
        // drains the warm writers by waiting for the channel to close, so a
        // duplicate sender outliving construction is a hang.
        drop(ctx);
        if let Some(analyzer) = analyzer {
            analyzer.run(shutdown);
        }
    }
}
