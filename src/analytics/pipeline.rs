//! Motion analysis: one blocking loop per camera, turning hot-buffer segments
//! into motion scores, event filmstrips and the crop jobs the vision model runs
//! on.
//!
//! Two rules hold this loop together, and both exist because the camera thread
//! filling the hot buffer is on the other side of them.
//!
//! **The hot buffer's read lock is only ever held to collect handles.** Every
//! read here — the sequence bounds, a pending segment, a run's priming
//! segments, an event's assembly — takes the lock, copies out `Arc`s and
//! scalars, and drops it. The worst case is [`assemble_event`]'s walk over one
//! event's segments, bounded by the event duration cap, forking nothing,
//! decoding nothing and touching no disk; everything else is a handful of map
//! lookups. A camera thread's `push` needs the *write* lock and so waits out
//! every reader, which is why nothing under this one may be slower than a walk
//! over pointers. The priming loop below used to decode three segments of video
//! under it — hundreds of milliseconds per motion batch, seconds while ffmpeg
//! was still probing its input — and it did so *during motion*, when the
//! footage the camera is trying to push is the footage that matters most.
//!
//! **ffmpeg children belong to the camera, not to the batch.** Both decoders —
//! the frame decoder that scores motion and the crop decoder that cuts the
//! frames an event keeps — are spawned once and kept until their child dies.
//! The crop decoder used to be forked, primed and killed per motion batch, up
//! to five fork/exec/kill cycles a second per camera, each one paying ffmpeg's
//! stream probe again. Death is noticed where it was always noticed, at the
//! next use; see [`ensure_long_lived`] for what a respawn costs and for the one
//! thing it will not do, which is start an ffmpeg for a camera that has already
//! been asked to stop.
//!
//! What a child that outlives the batch inherits is the batch's timeline, and a
//! per-batch fork never had one. This decoder is handed only the segments motion
//! asked for, so a quiet stretch reaches it as a gap in its input's timestamps,
//! and a camera up for a day and a half reaches it as an MPEG-TS timestamp
//! running out of bits and starting over. Neither is expensive, and the reason
//! is not symmetry: a gap wide enough to read as a break in the stream is
//! rebased by libavformat before the `fps` filter is shown it, while a gap
//! narrow enough to read as ordinary lateness is filled at one duplicated
//! picture per step — so the *small* gaps are the ones that cost, and they are
//! small. What bounds either is structural rather than argued. The frame
//! channel holds four and a full one backpressures ffmpeg instead of growing;
//! every extraction opens by draining what is queued, and what is left is let
//! go of at every pass and at every wakeup of the backoff a pass can be parked
//! in — so none of the waits this analyzer sets for *itself*, poll or backoff,
//! leaves those frames longer than a poll, not even on a camera whose decoder
//! will never come back. A wait it inherits from a stalled consumer downstream
//! is another matter, and a small term of a larger one;
//! [`MotionAnalyzer::release_idle_crop_frames`] names both such waits and why
//! neither is worth interrupting. And a child that stops consuming its input
//! is killed where the wedge is found. The worst
//! case is one killed child and one lost
//! batch of event frames — never a stalled analyzer, and never memory that
//! grows with how long the camera has been up. Those are the claims
//! [`crate::analytics::decoder`]'s gated tests answer against a real ffmpeg
//! rather than reason about.

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

/// One segment's motion verdict. Absent — `analyze_segment` returning `None` —
/// means the decoder produced no frames for it, which is *not* the same as no
/// motion: the segment was never looked at, and scoring it quiet would feed the
/// run tracker evidence of stillness that nothing supports.
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
    /// The detector's debug view. Held here for its demand window, not to
    /// write to it: the analyzer asks whether anybody is watching before
    /// encoding a full frame for it, and frees what an ended session left
    /// behind. `None` when there is no debug view to feed (the tests).
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

        // An estimate, not a record of what was analyzed: the motion store only
        // ever sees motion-positive segments, so an analyzed quiet stretch
        // leaves no trace here. `observed_sequences` keeps that estimate from
        // being reported as skipped footage.
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

        // The stop flag alone means only that a stop has *begun*: the camera
        // feeding this buffer is being joined right now and the GOP it has in
        // hand is still on its way. Flushing here — which is all this used to
        // do — is what dropped the last seconds of every recording in progress.
        self.drain_tail(DrainGate::starting_at(Instant::now(), TAIL_DRAIN_BOUND));
        self.flush_open_run();
        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    /// One pass of the analyzer. Returns whether the caller should wait out the
    /// poll interval — a pass that gave up on the decoder has already waited
    /// its own respawn backoff.
    ///
    /// Two things are given back before the decoder is so much as looked at,
    /// and they are the two that have to survive a decoder which cannot be
    /// brought back: the debug view's frames, and whatever the crop decoder has
    /// emitted since the last motion batch. A camera looping on a respawn that
    /// keeps failing analyzes nothing, and that is precisely the state where
    /// the box is short enough of memory for a fork to fail — so it is
    /// precisely the state in which neither the debug view's tens of megabytes
    /// nor the crop channel's four raw frames may stay pinned. Below the gate
    /// they would: a pass that gives up on the decoder returns from it, and the
    /// backoff between attempts escalates to a minute and repeats for as long
    /// as the fork keeps failing.
    ///
    /// Both are reachable without a frame ever being decoded, which is what
    /// puts them above the gate. Everything below it needs frames to do
    /// anything at all.
    fn tick(&mut self, shutdown: &AtomicBool) -> bool {
        if let Some(ref debug_store) = self.debug_store {
            debug_store.expire_unwatched(&self.camera_id);
        }
        self.release_idle_crop_frames();

        // The stop can be requested while a pass is running, and a decoder
        // forked for a pass that will not happen is the fork the drain path
        // already refuses to make. This check covers the frame decoder's fork
        // and nothing else — it sits directly above it. The crop decoder's fork
        // is at the far end of the pass, seconds away on a camera with a
        // backlog, so it is handed the flag rather than this reading of it; see
        // [`ensure_long_lived`].
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

    /// Phase 2 of the stop: keep analyzing until the camera's terminal
    /// watermark has been consumed, so the tail it pushed on its way out is
    /// part of the event that is about to be flushed rather than footage that
    /// arrived one poll too late.
    ///
    /// Bounded because a consumer that cannot finish must not be the reason an
    /// NVR never restarts. `gate` carries that bound — [`TAIL_DRAIN_BOUND`] at
    /// the one call site — and it covers the wait for phase 1 as well as the
    /// drain itself, so a camera that never comes back, and therefore never
    /// gets a final watermark, costs this analyzer the bound and no more. It is
    /// a parameter so a test can trip that bound without waiting out half a
    /// minute of it.
    ///
    /// Nothing forks from here on. That was already true of the frame decoder
    /// below, which this loop declines to respawn; the passes it runs carry no
    /// stop flag to extend it to the crop decoder, which used to fork per batch
    /// straight through the drain. `None` is the only thing this function can
    /// pass — it has no stop flag to offer and wants none — so the invariant is
    /// the signature's rather than a flag's.
    fn drain_tail(&mut self, gate: DrainGate) {
        let mut said_the_decoder_was_gone = false;
        loop {
            // A decoder that died is not respawned here: forking ffmpeg during
            // a drain is the one thing the analyzer's construction path already
            // refuses to do, and without one no further segment can be scored.
            //
            // What it does not do is leave. The camera is still finishing, and
            // the sequence `flush_open_run` extends the open run through is only
            // the end of the footage once the camera has said where it stopped
            // — returning here would sample it a GOP early and close the
            // recording exactly short of the tail this phase exists to keep. So
            // the wait is the same wait, held without decoding.
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
                    // Whose bound this was, said plainly. A camera that stopped
                    // and published a final watermark did its part, and what ran
                    // out was this analyzer's ability to keep up with it —
                    // usually a writer queue it is blocking on. Anything else is
                    // a camera that never finished stopping.
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
                        // From where scoring actually stopped, never from the
                        // position handed to the gate: a dead decoder reports
                        // itself finished to end the wait, and measuring from
                        // that would say it kept up with a camera it had
                        // stopped following.
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

    /// Replace the frame decoder if its child is gone, waiting out the backoff
    /// here when the fork fails — which is what makes a pass that gave up cost
    /// the caller nothing further, and why [`MotionAnalyzer::tick`] returns
    /// without a poll of its own afterwards.
    ///
    /// The wait releases crop frames as it surfaces. Nothing else will while it
    /// runs: this is the one place in a pass that can take a minute, the pass
    /// above it has already done its release, and the crop decoder's reader
    /// thread does not stop handing frames over just because the analyzer has
    /// stopped asking for them. Left to the next pass, four frames — 24 MB at
    /// the detection crop size — would sit through a backoff that widens
    /// towards a minute, on the box that is failing to fork ffmpeg. Which is
    /// the whole reason the release was hoisted above the gate in the first
    /// place, undone by the sleep below the gate.
    ///
    /// `spawn` is a parameter for the same reason [`build_with_retry`] takes
    /// one: the path worth pinning is the one where the fork keeps failing, and
    /// a test cannot make a real fork fail.
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

        // Emit after detection so runs that close in the same batch as their
        // motion segments still get object metadata.
        //
        // The registry's memory bound rests on this order too, and less
        // visibly. `process_motion_runs` dispatches the crop jobs for this
        // batch, and dispatching is what registers the verdicts they owe; a
        // record opened below is then guaranteed that every job which can ever
        // cover its sequences already exists. Emit first and a record can be
        // opened while a job for its own sequences is still to be dispatched —
        // the record looks resolved, the next event to close forgets it, and
        // the verdict arrives to find nothing. See
        // [`crate::storage::event_registry`].
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

    /// Report footage that was never analyzed — but only once the analyzer has
    /// actually observed a sequence. Until then `last_processed` is a
    /// reconstruction from the motion store, which records motion-positive
    /// segments only: a quiet segment that *was* analyzed is indistinguishable
    /// there from one that never was, so an early range would be invented, not
    /// measured.
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
        // The hot buffer can evict while this loop runs: `first_sequence` was
        // sampled before it, and each segment is fetched under its own lock.
        // Sequences that vanish in between are gone unanalyzed and nothing
        // later can notice — `last_processed` advances past the gap.
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

        // Lifecycle timing is monotonic: the analyzer runs near real time, so
        // the instant it observes a segment stands in for capture time. A
        // backlog is the exception — after a decoder or writer stall a batch
        // can hold minutes of footage at once — so each segment is dated by
        // its own media duration instead of sharing one reading.
        let observed_at = batch_instants(&segments, Instant::now());

        for (seg, now) in segments.into_iter().zip(observed_at) {
            let analysis = match self.analyze_segment(&seg.data)? {
                Some(analysis) => analysis,
                None => {
                    // No frames came out for this segment, so nothing is known
                    // about it: a quiet verdict here would count as evidence of
                    // stillness and could close an open run on footage that was
                    // never looked at. Skip it and move on — the zero-frame
                    // tripwire is what notices a decoder that stays blind.
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

    /// Assemble and hand off a finished event the moment its run closes.
    /// All segments in range are still hot and the metadata stores have not
    /// been cleaned up for them yet, so everything is read fresh here.
    ///
    /// The registry record is opened BEFORE the detection store is read and
    /// committed only once the write is in the writer's queue, so there is no
    /// instant in between — and the blocking send in the middle of it can last
    /// minutes — at which a verdict for this run arrives to find nothing to
    /// land on. See [`crate::storage::event_registry`] for the whole
    /// reconciliation; what this function owns is the one case the detection
    /// worker cannot handle itself, a verdict that landed before the write was
    /// queued and so has to be sent from here, behind it.
    ///
    /// Those minutes are the analyzer's longest wait, and the one place its
    /// crop decoder's frames are held without a release reaching them — a
    /// small term beside the event this thread is holding meanwhile, which is
    /// the point [`MotionAnalyzer::release_idle_crop_frames`] makes at length.
    fn emit_event(&self, run: ClosedRun, filmstrip: Option<Filmstrip>) {
        let tx = match self.event_tx {
            Some(ref tx) => tx,
            None => return,
        };

        // Every path below that returns without committing drops this, and
        // dropping it abandons the record — which is right, because those are
        // the paths where no file ever appears under this identity.
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

        // The write is in the channel, so from here the detection worker can
        // derive upgrades from this record itself: they go down the same
        // channel and so arrive behind the write (FIFO). What comes back is a
        // verdict that landed before that was true — while this thread was
        // assembling, or blocked on the send above — and it is this thread's
        // to send, because only a message queued after the write can find the
        // file the write creates.
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

    /// Feed one scored segment to the run tracker and send whatever the motion
    /// sensor owes as a result, returning the chunk that closed (if any) for
    /// the caller to emit.
    ///
    /// The sensor tracks the *physical* motion period, which is not the same
    /// question as "did a chunk close" and cannot be read off the tracker's
    /// liveness alone. Two shapes make that concrete, and both are why this
    /// compares [`RunTracker::motion_period`] rather than `is_open`:
    ///
    /// - The cap rolls a chunk, or suspends one on padding. A chunk closed and
    ///   perhaps another opened, but nothing physically stopped moving, and the
    ///   period is the same on both sides — so the sensor hears nothing.
    /// - A pending run's quiet window elapses on a motion segment. The old run
    ///   dies and a new one opens inside that single call, so the tracker is
    ///   alive before and alive after while the runs either side are strangers
    ///   to each other. Two periods, in order: the old one ended, then the new
    ///   one began, and the sensor is told both. Inferring from liveness would
    ///   fuse them into one and leave Home Assistant showing continuous motion
    ///   across a gap that the events themselves report as separate.
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

    /// Close whatever run is still open as a complete event, without waiting
    /// out its post-padding.
    ///
    /// Reached after [`MotionAnalyzer::drain_tail`], and closed through the last
    /// segment the camera actually produced rather than through the last one
    /// this analyzer managed to score. On the drain's normal path those are the
    /// same sequence. On the paths where they are not — a decoder that died
    /// mid-drain, a drain that ran out its bound — the difference is footage
    /// that is sitting in the hot buffer with nothing wrong with it except that
    /// nobody looked at it, and an event that stopped short of it would be a
    /// recording cut off at exactly the moment this whole drain exists to
    /// protect. The analysis ends early; the recording does not.
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
        // The run never saw its post-padding close, so nothing else would clear
        // the motion sensor. The bridge restates every entity on its next
        // connect, but that only helps if camon comes back — and it leaves HA
        // holding movement until it does.
        //
        // Driven off the period, not off the flush returning something: a run
        // suspended pending by the cap has no chunk left to write and flushes
        // nothing, while the sensor it turned on is still on. That shape —
        // shutdown inside the quiet window of a run longer than the cap — is
        // exactly when restarts and upgrades sample a busy camera.
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

    // --- Phase 2: Generic frame extraction + detection ---

    fn process_motion_runs(&mut self, segments: Vec<MotionSegment>, stop: Option<&AtomicBool>) {
        let runs = group_contiguous_runs(segments);
        let (sample_fps, crop_size) = (self.config.sample_fps, self.frame_use.crop_size());
        // The slot is emptied for the length of the batch — extracting a run
        // needs the analyzer and its decoder mutably at once — and refilled on
        // the way out: a batch that dropped the decoder instead would kill the
        // child and put per-batch forking back, silently.
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

    /// Let go of whatever the crop decoder emitted after the last motion batch
    /// stopped reading. Called from the two places a pass can be, and for the
    /// same reason in both: at the top, above the gate that ends a pass with no
    /// decoder to analyze with (see [`MotionAnalyzer::tick`]), and at every
    /// wakeup of the respawn backoff below that gate (see
    /// [`MotionAnalyzer::ensure_decoder_alive_with`]), which is the only part
    /// of a pass that can last longer than a poll.
    ///
    /// Those frames are already forfeit: the next batch opens by draining them,
    /// for the reason [`MotionAnalyzer::extract_run_frames`] gives. All that is
    /// at stake is *when*, and until this the answer was "whenever the camera
    /// next moves" — which on a quiet night is the morning. The channel holds
    /// four, and four is 24 MB at the detection crop size, per camera, held
    /// against a box whose analytics memory has already been the problem once.
    ///
    /// The top of a pass rather than the end of a batch, because at the end of
    /// a batch ffmpeg is still working through the segments it was just fed and
    /// refills the channel behind the drain. A pass begins a whole
    /// [`POLL_INTERVAL`] after the last one ended, with no batch in flight and
    /// — if the camera has gone quiet — nothing left for the child to make a
    /// frame from, so this is the drain that sticks. It costs one `try_recv` on
    /// an empty channel per poll.
    ///
    /// What an outage keeps, then, is one idle ffmpeg per camera and a channel
    /// nothing accumulates in: a frame the reader hands over during a backoff
    /// is taken at the next wakeup, so the residency through an outage is what
    /// arrives in a [`POLL_INTERVAL`] and not what arrives in a backoff that
    /// widens towards a minute. Killing that child too would free its process
    /// while the primary decoder is down, and is not done: respawn backoffs
    /// start at seconds, so a transient outage would cost a fork/exec cycle on
    /// the way back in exactly the conditions where forks are what is failing —
    /// and the memory that actually grows with an outage is the frames, which
    /// this releases.
    ///
    /// What it promises is worth stating exactly, because it is not "the
    /// channel is empty afterwards". [`CropDecoder::drain`] stops at the first
    /// empty channel, and the pipe's reader thread is a thread: it can be
    /// holding a frame it is about to hand over at that instant, and hand it
    /// over immediately after. So one call is best-effort — it takes what has
    /// arrived, and can leave one slot behind. What closes that is repetition
    /// rather than synchronisation, and the repetition is what the two call
    /// sites are for between them — though only over the waits this analyzer
    /// sets for itself. Those it does cover completely: neither the poll
    /// between passes nor a respawn backoff a minute wide goes longer than a
    /// [`POLL_INTERVAL`] without one of them, and the child that produced a
    /// straggler has no more input to make another from. Across those, the
    /// worst residency is what a child can emit in one poll interval, which
    /// the four-slot channel caps at four frames, and it is that for a poll
    /// rather than for a night or a backoff. Reaching for cross-thread
    /// bookkeeping to shave the last frame would buy a poll interval at the
    /// price of a debt ledger between the analyzer and the reader thread.
    ///
    /// Two waits have neither call site, and both are deliberate.
    ///
    /// One is the shutdown drain's: bounded by [`TAIL_DRAIN_BOUND`], nothing
    /// feeds the crop decoder during it, and the analyzer — child and channel
    /// together — is dropped the moment it ends. Releasing there would be
    /// freeing frames seconds before killing the process holding them.
    ///
    /// The other is [`MotionAnalyzer::emit_event`]'s blocking sends, which are
    /// not this analyzer's wait but the warm writer's, and can last minutes
    /// while that writer is stalled. They fall at the worst moment for this —
    /// immediately after a motion batch, with the crop child still refilling
    /// behind it — so four frames do sit through them. What decides it is what
    /// else sits through them: the `FinishedEvent` in flight holds that event's
    /// whole segment list, tens to hundreds of megabytes of footage handles,
    /// for the identical duration and by the same durability argument. The crop
    /// channel's 24 MB is a subordinate term of a residency this pipeline
    /// already has, already documents, and already bounds through the writer's
    /// own drain — and slicing a send that an event's survival depends on into
    /// interleaved releases would complicate that path to buy back the smaller
    /// of the two numbers.
    ///
    /// Reports how many it let go of, which is how a test sees the difference
    /// between a pass that released them and one that only had nothing to
    /// release.
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

    /// Feed the segments preceding `first_seq` to `decode`, so a freshly forked
    /// ffmpeg spends its stream probe on footage nobody wants before it reaches
    /// the footage somebody does. Those pictures predate the motion, so
    /// whatever comes back out of them is dropped.
    ///
    /// Once per child, and the probe belongs to one particular ffmpeg — so a
    /// decoder that has had [`PRIMING_SEGMENTS`] of footage is left alone from
    /// then on, and one that has had less is fed again on the next run. What a
    /// run can offer is not always the whole window: a run at the very start of
    /// a camera's buffer has nothing behind it, a run near the buffer's edge
    /// has only the part that has not aged out yet, and a camera configured
    /// with seconds of retention may never hold the whole window behind any run
    /// at all. All three are the same case. Feed what there is — it is footage
    /// nobody keeps either way, and it still spends some of the probe — and add
    /// it to what this child has already had, which is what makes even the
    /// third of those converge instead of re-feeding forever.
    ///
    /// The handles are collected under the hot buffer's read lock and decoded
    /// after it has been released, which is the whole point of the split.
    /// Decoded under it, as this loop used to be, a camera thread's `push`
    /// waited out three ffmpeg decodes on every motion batch — and only motion
    /// gets here, so the stall landed on the footage that matters most.
    ///
    /// `decode` is a parameter for the same reason [`sample_run_frames`] takes
    /// one: it lets a test put something that takes the buffer's *write* lock
    /// where ffmpeg goes — the camera thread's exact position — and have it
    /// succeed.
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
    /// [`subsample_tagged`] can still use, holding no more than
    /// [`RUN_FRAME_ACCUMULATOR_CAP`] of them at once.
    ///
    /// A child that is not yet past its stream probe is fed towards it first —
    /// on the first run it decodes, and on the runs after that only for as long
    /// as the buffer has been too short to finish the job in one go; never on
    /// every batch, which is what it used to be. Everything that
    /// arrived by the end of that is then drained, and so is everything left
    /// over from the previous batch: a leftover taken by the first sampled
    /// segment's read would be tagged with a crop measured on a different
    /// picture. That drain is the one thing a decoder kept across batches needs
    /// which a per-batch one got for free from being new; what the same decoder
    /// is left holding between passes is
    /// [`MotionAnalyzer::release_idle_crop_frames`]'s to let go of. It reaches
    /// only what has arrived, so it narrows the window rather than closing it;
    /// [`frames_per_segment`] keeps more than one frame per segment so a lagged
    /// pipe costs the strip a frame instead of all of them.
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

    /// The full (uncropped) frame the detection debug view draws its overlay
    /// on: the first frame of the run that has a crop, i.e. that had motion.
    /// The detection mask is blacked out here too, so the view shows exactly
    /// what the model could not see.
    ///
    /// Nothing but that view ever reads this frame, so it is encoded only while
    /// somebody has the view open — the same bargain the stage overlays strike
    /// in [`publish_debug_maps`]. Unconditionally it is a 1080p encode per
    /// motion run all night, a megabyte carried through the detection queue and
    /// then pinned in the debug store, for a page nobody opened. With detection
    /// off there is no job to carry it at all.
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

    /// Extract, crop and JPEG-encode the color frames of one contiguous motion
    /// run. They become the filmstrip of the event the run belongs to, and —
    /// when object detection is on — a crop job for the global detection
    /// worker. Handing that job off never blocks: a camera past its queue cap
    /// loses its oldest queued job instead, costing that object upgrade but
    /// never the event.
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

    /// Everything the run's frames are turned into, once they have been
    /// extracted. Split from the extraction above so what the strip, the job
    /// and the debug overlay are built out of can be driven from a test
    /// without an ffmpeg to decode with.
    fn process_run_frames(
        &mut self,
        run: Vec<MotionSegment>,
        tagged_frames: Vec<(RgbFrame, Option<NormalizedRect>)>,
    ) {
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

        let full_frame_jpeg = self.debug_overlay_frame(&tagged_frames);

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
            tx.send(DetectionJob {
                camera_id: self.camera_id.clone(),
                seqs: run.iter().map(|seg| seg.seq).collect(),
                // The strip has two independent owners — this event and the
                // detection job — so it is copied once, here. Past this point
                // the job's copy is shared by handle: what the model reads, the
                // thumbnail keeps and the debug store holds are all these
                // bytes, never another copy of them.
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

/// Monotonic capture instant per segment of one batch: the last segment ends
/// at `now` and the others are placed back along their own media durations, so
/// a batch of backlog spans the same time the footage did. Post-padding and the
/// event duration cap then behave the same whether segments arrive one per poll
/// or as a burst after a stall.
///
/// Walking backwards from `now` keeps every instant at or before it whatever
/// the durations say. An instant in the future would be worse than the single
/// shared reading this replaces: the tracker's elapsed math saturates at zero
/// until wall time catches up, freezing both countdowns. Backlog older than the
/// monotonic epoch (a hot buffer inherited by a just-started process) stops at
/// that floor instead of wrapping.
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

/// Publish the detector's pipeline-stage views for the debug UI:
/// - stability: final motion mask (after opening + area filter)
/// - raw: raw MOG2 foreground mask
/// - no-shadow: alias of raw (the pure-Rust detector has no shadow class;
///   the stage name is kept so the API/UI stay stable)
/// - morph: after morphological opening
/// - background: the learned MOG2 background model
///
/// A stage is only encoded while somebody is looking at it: five greyscale
/// JPEGs per camera per analysis tick is real CPU, and outside the debug view
/// nothing ever reads them. [`MotionStore::map_wanted`] answers from the last
/// time the API was asked for that stage on that camera, so the encode — and
/// the background copy that feeds it — is skipped outright rather than thrown
/// away afterwards.
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
    // `no_shadow_mask()` is the raw mask, so the two stages are one encode. The
    // motion overlay asks for both, which makes doing it twice the common case
    // rather than the exotic one.
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

/// Construction is retried rather than fatal because it fails for the same
/// reasons a running decoder dies — which [`MotionAnalyzer::ensure_decoder_alive`]
/// already respawns through. Giving up here instead left the camera with no
/// analyzer at all, and in event mode nothing else writes events: it recorded
/// nothing for the rest of the process after a single line at startup.
pub fn spawn_analyzer(
    ctx: AnalyzerContext,
    shutdown: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(analyzer_body(ctx, shutdown))
}

/// The analyzer's whole life as a closure, for callers that spawn it
/// themselves — [`crate::supervise::Supervisor::critical_blocking`] does, so
/// that an analyzer which dies is noticed while camon is running rather than
/// by whoever joins its handle at the stop.
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
        // The retry needs the context to still be here, but the analyzer now
        // holds its own clone of every sender in it. Shutdown drains the warm
        // writers by dropping the last sender and waiting for the channel to
        // close, so a duplicate that outlives construction is a hang waiting
        // for the one camera whose analyzer takes longest to stop.
        drop(ctx);
        if let Some(analyzer) = analyzer {
            analyzer.run(shutdown);
        }
    }
}
