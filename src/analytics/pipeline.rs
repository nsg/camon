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
    MotionSettingsStore, DEFAULT_MIN_CONTOUR_AREA, DEFAULT_VAR_THRESHOLD, MASK_CELLS, MASK_COLS,
    MASK_ROWS,
};
use crate::buffer::warm::{assemble_event, EventUpgrade, WriterMessage};
use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::locks::LockExt;
use crate::mqtt::{send_event, MqttEvent};
use crate::retry::{jittered, RetrySchedule, Streak};
use crate::shutdown::{shortfall, who_stalled, DrainGate, DrainStep, Stalled, TAIL_DRAIN_BOUND};
use crate::storage::{
    DetectionDebugStore, DetectionStore, EventRegistry, MapKind, MotionEntry, MotionStore,
    UpgradeTarget,
};

use super::decoder::{
    CropDecoder, DecodeOutcome, FrameDecoder, DETECTION_CROP_SIZE, THUMBNAIL_CROP_SIZE,
};
use super::detect_worker::{DetectQueueSender, DetectionJob};
use super::motion::{MotionBox, MotionDetector};
use super::run_tracker::{ClosedRun, RunTracker};

const ANALYSIS_WIDTH: i32 = 320;
const ANALYSIS_HEIGHT: i32 = 240;

const MOTION_THRESHOLD: f32 = 0.05;

const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// How long to wait before trying a dead decoder again.
const DECODER_RESTART_BACKOFF: Duration = Duration::from_secs(5);

/// Ceiling on the decoder-spawn backoff. Spawning a decoder fails for two very
/// different reasons: a transient one (a fork that lost a race for memory, an
/// exhausted fd table) that clears on its own, and a permanent one (no ffmpeg on
/// PATH) that never does. Doubling from [`DECODER_RESTART_BACKOFF`] to a minute
/// serves both — the same shape the camera pipeline's reconnect uses — so the
/// first recovers within a minute of clearing and the second stops costing an
/// ffmpeg fork every five seconds.
const DECODER_SPAWN_BACKOFF_MAX: Duration = Duration::from_secs(60);

const DECODER_SPAWN_SCHEDULE: RetrySchedule = RetrySchedule {
    start: DECODER_RESTART_BACKOFF,
    max: DECODER_SPAWN_BACKOFF_MAX,
};

/// The one policy for failing to spawn a decoder, used by both places that do
/// it: building an analyzer and replacing one whose decoder died. They fail for
/// identical reasons, so a missing ffmpeg must not produce a line a minute
/// through one path and twelve through the other.
///
/// Reporting escalates rather than repeating: something permanently broken
/// stays visible without burying every other line in the log.
struct DecoderSpawnRetry {
    schedule: RetrySchedule,
    backoff: Duration,
    streak: Streak,
}

impl DecoderSpawnRetry {
    fn new(schedule: RetrySchedule) -> Self {
        Self {
            schedule,
            backoff: schedule.start,
            streak: Streak::new(),
        }
    }

    /// Record a failed spawn. Returns how long to wait, and the streak length
    /// when this failure is one worth a log line.
    fn failed(&mut self) -> (Duration, Option<u32>) {
        let delay = jittered(self.backoff);
        self.backoff = self.schedule.next(self.backoff);
        (delay, self.streak.record())
    }

    fn succeeded(&mut self) {
        self.backoff = self.schedule.start;
        self.streak.reset();
    }
}

const CROP_PADDING: f32 = 0.2;
const MIN_CROP_FRACTION: f32 = 0.15;

/// Segments fed to a freshly forked crop decoder before its frames are kept.
/// Three seconds of footage is enough to carry ffmpeg past the stream probe it
/// swallows its first input for; see [`MotionAnalyzer::prime_with`].
const PRIMING_SEGMENTS: u64 = 3;

/// Consecutive zero-frame decodes tolerated before the decoder is declared
/// blind. A segment is one GOP and always opens on a keyframe, so a healthy
/// decode yields at least one I-frame — but a freshly spawned ffmpeg swallows
/// several seconds of input while it probes the stream, so a single empty
/// decode proves nothing. Only an unbroken streak does, and at roughly
/// one segment per second thirty of them is about half a minute of blindness:
/// long enough that no buffering hiccup explains it, short enough that little
/// motion is missed before the respawn.
const BLIND_DECODER_STREAK: u32 = 30;

/// Sleep up to `total`, returning early once shutdown is requested, so a backoff
/// never holds the drain up. The analyzer body runs on a blocking thread and so
/// cannot select against the shutdown notify the async tasks use; polling the
/// same flag it already polls every tick is the equivalent.
fn sleep_unless_shutdown(total: Duration, shutdown: &AtomicBool) {
    sleep_unless_shutdown_watching(total, shutdown, || {});
}

/// The same sleep, with something to do each time it surfaces.
///
/// A sleep in this analyzer is not idle time. The ffmpeg children go on running
/// through it — the crop decoder's reader thread keeps handing over frames from
/// segments it was fed before the analyzer stopped asking — so a caller parked
/// here is a caller not collecting them. The wait is already cut into
/// [`POLL_INTERVAL`] slices to notice a shutdown; `at_each_wakeup` rides the
/// same slices, which is what keeps the longest a backoff can leave those
/// frames unattended at one slice rather than at the whole backoff.
///
/// The sleep's own semantics are untouched by it: same deadline arithmetic,
/// same early return the moment a stop is requested, and nothing runs after
/// that return.
fn sleep_unless_shutdown_watching(
    total: Duration,
    shutdown: &AtomicBool,
    mut at_each_wakeup: impl FnMut(),
) {
    let deadline = Instant::now() + total;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || shutdown.load(Ordering::Relaxed) {
            return;
        }
        thread::sleep(remaining.min(POLL_INTERVAL));
        at_each_wakeup();
    }
}

/// Counts consecutive zero-frame decodes so an ffmpeg that consumes input but
/// emits nothing is caught. A single empty decode is normal and simply leaves
/// that segment unanalyzed, so nothing else notices a decoder that never
/// recovers: it analyzes nothing for ever while the child stays alive, past
/// every liveness check there is.
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

/// Whether this analyzer may start an ffmpeg *now* — asked where a fork would
/// happen, not remembered from where the pass began.
///
/// The distinction is the whole point. A pass reaches the crop decoder at its
/// very end, after every segment of the batch has been decoded and scored,
/// which on a camera working through a backlog is seconds after the pass
/// started. A stop requested in between has to be seen there, so `Open` carries
/// the stop flag itself and reads it at the fork rather than passing a copy
/// down. (The frame decoder needs none of this: its fork sits microseconds
/// after the check at the top of the pass, and it is left alone.)
///
/// `Closed` is what the shutdown drain runs in, and the reason this is an enum
/// rather than a flag on the analyzer. [`MotionAnalyzer::drain_tail`] holds no
/// stop flag and needs none — nothing forks from there whatever any flag says —
/// so the variant it passes carries nothing that could later be made to say
/// otherwise.
#[derive(Clone, Copy)]
enum ForkWindow<'a> {
    Open(&'a AtomicBool),
    Closed,
}

impl ForkWindow<'_> {
    fn is_open(self) -> bool {
        match self {
            Self::Open(stop_requested) => !stop_requested.load(Ordering::Relaxed),
            Self::Closed => false,
        }
    }
}

/// A child process the analyzer keeps rather than re-forks. One method, because
/// one question is all the reuse policy asks of it — and because a test can
/// answer that question from a counter where production answers it from
/// `waitpid`.
trait Respawnable {
    fn is_alive(&mut self) -> bool;
}

impl Respawnable for CropDecoder {
    fn is_alive(&mut self) -> bool {
        CropDecoder::is_alive(self)
    }
}

/// A decoder that outlives the batch which first needed it, together with the
/// one thing its current child has to be told before it is useful.
struct LongLived<D> {
    decoder: Option<D>,
    /// How many segments the child in `decoder` has been fed to carry it past
    /// ffmpeg's stream probe. Reset by nothing except a respawn, which is what
    /// makes the probe a cost per child instead of a cost per batch:
    /// re-priming a healthy decoder would decode three segments of footage on
    /// every batch purely to throw all of it away.
    ///
    /// A count rather than a flag because the window a run can offer is not
    /// always whole — the buffer's oldest sequences age out, and a camera
    /// configured with seconds of retention may never hold
    /// [`PRIMING_SEGMENTS`] of them behind a run at all. What the probe cares
    /// about is how much input it has been given, not which sequences it came
    /// from, so partial windows add up and a child that can only ever be fed
    /// one segment at a time is primed by the third run instead of never.
    primed_with: u64,
}

impl<D> LongLived<D> {
    /// Whether this child has had enough input to be past its probe, and so
    /// whether the next run may keep what it decodes.
    fn primed(&self) -> bool {
        self.primed_with >= PRIMING_SEGMENTS
    }
}

/// Empty, and so unprimed: the state before the camera's first motion, and the
/// state a batch leaves behind when its decoder could not be replaced.
impl<D> Default for LongLived<D> {
    fn default() -> Self {
        Self {
            decoder: None,
            primed_with: 0,
        }
    }
}

/// Make sure `slot` holds a decoder whose child is alive, forking one only when
/// it does not, and reporting whether this batch has one to decode with.
///
/// A living child is handed straight back — that is the whole point of the slot
/// — so the fork happens on the camera's first motion and then only when the
/// child is really gone: it crashed, or [`CropDecoder::decode_segment`] killed
/// it for wedging. A respawn clears the priming count, because the protocol is
/// addressed to a particular ffmpeg's stream probe and the replacement has its
/// own.
///
/// A closed [`ForkWindow`] is the one refusal. Once a stop has been requested
/// this analyzer starts no new ffmpeg — the same promise
/// [`MotionAnalyzer::drain_tail`] makes for the frame decoder, and now kept for
/// this one too, which used to fork per batch straight through the drain. A
/// decoder already running is still used, so a stop only loses the frames of a
/// camera whose crop decoder happened to die on the way out.
fn ensure_long_lived<D: Respawnable>(
    slot: &mut LongLived<D>,
    window: ForkWindow,
    camera_id: &str,
    spawn: impl FnOnce() -> Result<D, std::io::Error>,
) -> bool {
    if slot.decoder.as_mut().is_some_and(D::is_alive) {
        return true;
    }
    if !window.is_open() {
        return false;
    }
    match spawn() {
        Ok(decoder) => {
            *slot = LongLived {
                decoder: Some(decoder),
                primed_with: 0,
            };
            true
        }
        Err(e) => {
            // Dropping the dead one kills whatever is left of it, and leaves
            // the next batch to try the fork again.
            *slot = LongLived::default();
            tracing::error!(camera = %camera_id, error = %e, "failed to create crop decoder");
            false
        }
    }
}

/// What keeping a decoder across batches asks of whoever keeps it: the slot it
/// lives in, and the camera a failed fork is reported under.
///
/// A trait for one reason. The reuse policy is the part of this worth pinning
/// and no test can fork an ffmpeg, so the code a camera really runs is written
/// against a stand-in that can count forks instead of making them — including
/// the step that is easiest to lose and hardest to see, which is giving the
/// decoder back at the end of the batch.
trait KeepsDecoder<D> {
    fn slot(&mut self) -> &mut LongLived<D>;
    fn camera(&self) -> &str;
}

/// Lend a batch the decoder its camera keeps, and take it back.
///
/// The slot is emptied for the length of the batch because extracting a run
/// needs the analyzer and its decoder mutably at the same moment, which a field
/// cannot give. Returning it is not a separate step and must not become one: a
/// batch that dropped the decoder instead of handing it back would kill the
/// child on the way out and leave the next batch to fork another — per-batch
/// forking again, silently, which is the whole thing the slot exists to
/// prevent.
///
/// `batch` runs only when there is a decoder to run it with; see
/// [`ensure_long_lived`] for when there is not.
fn lend_for_batch<O, D>(
    owner: &mut O,
    window: ForkWindow,
    spawn: impl FnOnce() -> Result<D, std::io::Error>,
    batch: impl FnOnce(&mut O, &mut LongLived<D>),
) where
    O: KeepsDecoder<D>,
    D: Respawnable,
{
    let mut lent = std::mem::take(owner.slot());
    if ensure_long_lived(&mut lent, window, owner.camera(), spawn) {
        batch(owner, &mut lent);
    }
    *owner.slot() = lent;
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
///
/// Visible to the crate because storage counts the frames back off the store
/// and has to know where to stop looking: see
/// [`MAX_FILMSTRIP_FRAMES`](crate::storage::event_index::MAX_FILMSTRIP_FRAMES),
/// which is pinned to this number rather than agreeing with it by convention.
pub(crate) const FILMSTRIP_FRAMES: usize = 4;
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

/// Drop every second entry once `acc` outgrows `cap`. Halving keeps the whole
/// span covered at coarser spacing instead of truncating it to its beginning or
/// end, and the first entry always survives.
///
/// Used wherever the final length is not known while the entries arrive: an
/// open run lasts for as many batches as the motion does, and a segment holds
/// `sample_fps` frames per second of footage, which is config with no ceiling.
/// Where the count *is* known up front, thinning to it directly beats halving
/// down to it — see [`frames_per_segment`].
fn halve_past<T>(acc: &mut Vec<T>, cap: usize) {
    if acc.len() <= cap {
        return;
    }
    let mut seen = 0;
    acc.retain(|_| {
        seen += 1;
        seen % 2 == 1
    });
}

impl RunFilmstrip {
    /// Add a batch's frames, halving the accumulator whenever it outgrows its
    /// cap.
    fn push(&mut self, frames: Vec<Vec<u8>>) {
        self.frames.extend(frames);
        halve_past(&mut self.frames, FILMSTRIP_ACCUMULATOR_CAP);
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

/// Segments that left the hot buffer before the analyzer reached them. Their
/// footage is never scored, so the skip is reported rather than absorbed.
#[derive(Debug, PartialEq, Eq)]
struct SkippedSegments {
    count: u64,
    from_seq: u64,
    to_seq: u64,
}

impl SkippedSegments {
    /// The gap between the next sequence the analyzer would have processed and
    /// the oldest one still resident, or `None` when it has kept up.
    fn between(last_processed: u64, first_resident: u64) -> Option<Self> {
        if last_processed >= first_resident {
            return None;
        }
        Some(Self {
            count: first_resident - last_processed,
            from_seq: last_processed,
            to_seq: first_resident - 1,
        })
    }

    /// The individual sequences that could not be read. Eviction takes the
    /// oldest first, so these are contiguous in practice and the range says so;
    /// `count` is exact either way.
    fn of(sequences: &[u64]) -> Option<Self> {
        Some(Self {
            count: sequences.len() as u64,
            from_seq: *sequences.iter().min()?,
            to_seq: *sequences.iter().max()?,
        })
    }

    fn merged(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            from_seq: self.from_seq.min(other.from_seq),
            to_seq: self.to_seq.max(other.to_seq),
        }
    }
}

fn merge_skips(a: Option<SkippedSegments>, b: Option<SkippedSegments>) -> Option<SkippedSegments> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.merged(b)),
        (a, b) => a.or(b),
    }
}

/// Shortest gap between two skipped-footage warnings. An analyzer that stays
/// behind skips something on most of its 200 ms polls, and warnings are the
/// level this project keeps enabled in release, so the reports are accumulated
/// and released as one line per interval instead of per poll.
const SKIP_REPORT_INTERVAL: Duration = Duration::from_secs(30);

/// Accumulates skipped footage between warnings; see [`SKIP_REPORT_INTERVAL`].
#[derive(Default)]
struct SkipReporter {
    pending: Option<SkippedSegments>,
    last_report: Option<Instant>,
}

impl SkipReporter {
    /// Fold one poll's skips in, returning the accumulated report when the
    /// interval has passed. The first skip is always reported: a rare one-off
    /// is exactly the case worth seeing immediately.
    fn record(&mut self, skipped: SkippedSegments, now: Instant) -> Option<SkippedSegments> {
        self.pending = Some(merge_skips(self.pending.take(), Some(skipped))?);
        let due = self
            .last_report
            .is_none_or(|at| now.saturating_duration_since(at) >= SKIP_REPORT_INTERVAL);
        if !due {
            return None;
        }
        self.last_report = Some(now);
        self.pending.take()
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

impl KeepsDecoder<CropDecoder> for MotionAnalyzer {
    fn slot(&mut self) -> &mut LongLived<CropDecoder> {
        &mut self.crop_decoder
    }

    fn camera(&self) -> &str {
        &self.camera_id
    }
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
        // [`ForkWindow`].
        if shutdown.load(Ordering::Relaxed) || !self.ensure_decoder_alive(shutdown) {
            return false;
        }

        if let Err(e) = self.process_new_segments(ForkWindow::Open(shutdown)) {
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
    /// below, which this loop declines to respawn; the passes it runs carry a
    /// closed [`ForkWindow`] to extend it to the crop decoder, which used to
    /// fork per batch straight through the drain. Closed is the only thing this
    /// function can pass — it has no stop flag to offer and wants none — so the
    /// invariant is the signature's rather than a flag's.
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
                if let Err(e) = self.process_new_segments(ForkWindow::Closed) {
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

    fn process_new_segments(
        &mut self,
        window: ForkWindow,
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
            self.process_motion_runs(motion_segments, window);
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

    fn process_motion_runs(&mut self, segments: Vec<MotionSegment>, window: ForkWindow) {
        let runs = group_contiguous_runs(segments);
        let (sample_fps, crop_size) = (self.config.sample_fps, self.frame_use.crop_size());
        lend_for_batch(
            self,
            window,
            || CropDecoder::new(sample_fps, crop_size),
            |analyzer, crop| {
                for run in runs {
                    analyzer.process_run(run, crop);
                }
            },
        );
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

fn sample_indices(len: usize) -> Vec<usize> {
    if len <= 4 {
        (0..len).collect()
    } else {
        vec![0, len / 3, 2 * len / 3, len - 1]
    }
}

/// Frames kept out of one segment's decode, given how many segments the run
/// contributes. [`sample_indices`] has already spread those segments over the
/// run and the final pick is positional — it never compares pixels — so one
/// frame per segment is all [`subsample_tagged`] strictly needs; the spare is
/// what keeps a segment that decoded short, or a pipe running a frame behind
/// its segments, from costing the strip a picture it cannot backfill.
fn frames_per_segment(segments: usize) -> usize {
    FILMSTRIP_FRAMES.div_ceil(segments.max(1)) + 1
}

/// Live raw frames one run may hold while its filmstrip is chosen. Not a policy
/// of its own — the largest [`frames_per_segment`] ever grants across the
/// segments [`sample_indices`] yields — but pinned so the peak stays a number
/// this file states rather than one a reader has to derive.
const RUN_FRAME_ACCUMULATOR_CAP: usize = 9;

/// Keep at most `keep` of one segment's frames, spread from its first to its
/// last. Both ends are kept because they are the two the run's selection can
/// least afford to lose: the first frame is the segment's keyframe, the one the
/// crop tag was measured on, and the last is the furthest whatever moved has
/// travelled by the time the next segment starts.
///
/// The two picks over a whole run — [`subsample_filmstrip`] and
/// [`subsample_tagged`] — instead space themselves at `n/3` and land on the
/// last frame only by way of `n - 1`. They are picking moments out of an event,
/// where the exact endpoints carry nothing in particular; this is picking
/// frames out of one segment, where they carry the two things above.
fn thin_evenly<T>(frames: Vec<T>, keep: usize) -> Vec<T> {
    let n = frames.len();
    if n <= keep {
        return frames;
    }
    let span = keep.saturating_sub(1).max(1);
    let picks: Vec<usize> = (0..keep).map(|k| k * (n - 1) / span).collect();
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, frame)| frame)
        .collect()
}

/// Decode the sampled segments of `run` and reduce them to the frames the event
/// filmstrip and the vision model get, each tagged with its own segment's crop.
///
/// `decode` is a parameter so the selection can be driven without an ffmpeg;
/// [`MotionAnalyzer::extract_run_frames`] passes the crop decoder.
///
/// No segment is ever materialised whole. Frames are thinned as they arrive,
/// through an accumulator [`halve_past`] holds just above `keep` — a segment
/// owns `sample_fps` frames per second and `sample_fps` has no configured
/// ceiling, so the live cost has to be a function of what is kept rather than
/// of what is decoded.
fn sample_run_frames(
    run: &[MotionSegment],
    crops: &HashMap<u64, NormalizedRect>,
    width: usize,
    height: usize,
    mut decode: impl FnMut(&Arc<Vec<u8>>, u64, &mut dyn FnMut(Vec<u8>)),
) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
    let indices = sample_indices(run.len());
    let keep = frames_per_segment(indices.len());
    let mut all_frames: Vec<(RgbFrame, Option<NormalizedRect>)> =
        Vec::with_capacity(RUN_FRAME_ACCUMULATOR_CAP);

    for &idx in &indices {
        let seg = &run[idx];
        let crop = crops.get(&seg.seq).copied();
        // One frame past `keep`, because `halve_past` needs one in hand beyond
        // its cap to halve against, and one more transiently while it is held.
        // Stated once: the two must not drift apart, and every frame either
        // covers is 6 MB at the detection crop size. The width is invisible in
        // the result — the final thin lands on the same frames whatever it is —
        // so it is purely how much memory the decode is allowed to use.
        let reservoir = keep + 1;
        let mut held: Vec<Vec<u8>> = Vec::with_capacity(reservoir + 1);
        decode(&seg.data, seg.duration_ns, &mut |frame_data: Vec<u8>| {
            // The pipe delivers exact fixed-size frames; anything else is a
            // torn read from a dying ffmpeg and gets skipped.
            if frame_data.len() == width * height * 3 {
                held.push(frame_data);
                halve_past(&mut held, reservoir);
            }
        });
        all_frames.extend(thin_evenly(held, keep).into_iter().map(|data| {
            (
                RgbFrame {
                    data,
                    width,
                    height,
                },
                crop,
            )
        }));
    }

    subsample_tagged(all_frames)
}

fn subsample_tagged(
    frames: Vec<(RgbFrame, Option<NormalizedRect>)>,
) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
    if frames.len() <= 4 {
        return frames;
    }
    let n = frames.len();
    // Moved out rather than indexed and cloned, as in [`subsample_filmstrip`]:
    // a kept frame is a whole raw RGB image, several megabytes at the detection
    // crop size.
    let picks = [0, n / 3, 2 * n / 3, n - 1];
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, tagged)| tagged)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// A detector past warmup with a blob in view, so every stage has
    /// something to publish.
    fn detector_with_masks() -> MotionDetector {
        const W: usize = 64;
        const H: usize = 48;
        let mut detector = MotionDetector::new(16.0, 4.0);
        let still = vec![60u8; W * H];
        for _ in 0..20 {
            detector.process_frame(&still, W, H);
        }
        let mut moved = still.clone();
        for y in 8..24 {
            for x in 8..24 {
                moved[y * W + x] = 220;
            }
        }
        detector.process_frame(&moved, W, H);
        detector
    }

    /// Five JPEG encodes per camera per tick, for a view that is almost never
    /// open: a camera nobody has asked about recently gets none of them.
    #[test]
    fn debug_maps_are_only_encoded_while_somebody_is_watching() {
        let store = MotionStore::new(&["watched".to_string(), "idle".to_string()]);
        let detector = detector_with_masks();
        for kind in MapKind::ALL {
            store.mark_map_requested_ago("watched", kind, Duration::ZERO);
            store.mark_map_requested_ago("idle", kind, Duration::from_secs(600));
        }

        publish_debug_maps(&store, "watched", &detector);
        publish_debug_maps(&store, "idle", &detector);

        for kind in MapKind::ALL {
            assert!(
                store.get_map("watched", kind).is_some(),
                "{} was not published for the camera being watched",
                kind.as_str()
            );
            assert!(
                store.get_map("idle", kind).is_none(),
                "{} was encoded for a camera nobody is watching",
                kind.as_str()
            );
        }
    }

    /// The overlays are toggled one at a time, so demand is tracked that way
    /// too — a request for one stage does not pay for the other four.
    #[test]
    fn debug_map_demand_is_tracked_per_stage() {
        let store = MotionStore::new(&["cam".to_string()]);
        let detector = detector_with_masks();
        store.mark_map_requested_ago("cam", MapKind::Background, Duration::ZERO);

        publish_debug_maps(&store, "cam", &detector);

        assert!(store.get_map("cam", MapKind::Background).is_some());
        for kind in MapKind::ALL
            .into_iter()
            .filter(|k| *k != MapKind::Background)
        {
            assert!(
                store.get_map("cam", kind).is_none(),
                "{} rode along on a request for another stage",
                kind.as_str()
            );
        }
    }

    /// The whole point of the gate is that demand expires. If publishing were
    /// ever to count as a request, the first tick after somebody looked once
    /// would hold the gate open forever and every other test here would still
    /// pass.
    #[test]
    fn publishing_a_map_does_not_renew_its_own_demand() {
        let store = MotionStore::new(&["cam".to_string()]);
        let detector = detector_with_masks();
        store.mark_map_requested_ago("cam", MapKind::Stability, Duration::ZERO);

        publish_debug_maps(&store, "cam", &detector);
        store.mark_map_requested_ago("cam", MapKind::Stability, Duration::from_secs(600));

        assert!(
            !store.map_wanted("cam", MapKind::Stability),
            "publishing latched the gate open, so the encode never stops"
        );
    }

    /// `no-shadow` is an alias of `raw`, and the motion overlay asks for both,
    /// so the common case must not pay for the same JPEG twice.
    #[test]
    fn the_raw_and_no_shadow_stages_share_one_encode() {
        let store = MotionStore::new(&["cam".to_string()]);
        let detector = detector_with_masks();
        store.mark_map_requested_ago("cam", MapKind::RawMog2, Duration::ZERO);
        store.mark_map_requested_ago("cam", MapKind::NoShadow, Duration::ZERO);

        publish_debug_maps(&store, "cam", &detector);

        let raw = store.get_map("cam", MapKind::RawMog2);
        assert!(raw.is_some());
        assert_eq!(raw, store.get_map("cam", MapKind::NoShadow));
    }

    /// Sharing the encode must not merge the two stages: each still answers to
    /// its own demand, and one asked for alone leaves the other empty.
    #[test]
    fn the_raw_and_no_shadow_stages_are_still_gated_apart() {
        let store = MotionStore::new(&["cam".to_string()]);
        let detector = detector_with_masks();

        store.mark_map_requested_ago("cam", MapKind::NoShadow, Duration::ZERO);
        publish_debug_maps(&store, "cam", &detector);
        assert!(store.get_map("cam", MapKind::NoShadow).is_some());
        assert!(
            store.get_map("cam", MapKind::RawMog2).is_none(),
            "raw was filled by a request for no-shadow"
        );

        let store = MotionStore::new(&["cam".to_string()]);
        store.mark_map_requested_ago("cam", MapKind::RawMog2, Duration::ZERO);
        publish_debug_maps(&store, "cam", &detector);
        assert!(store.get_map("cam", MapKind::RawMog2).is_some());
        assert!(
            store.get_map("cam", MapKind::NoShadow).is_none(),
            "no-shadow was filled by a request for raw"
        );
    }

    /// The decoder-restart backoff must not outlive a shutdown request: the
    /// analyzer is joined by the drain, and five seconds per camera is time the
    /// service manager does not always give.
    #[test]
    fn decoder_restart_backoff_ends_when_shutdown_is_requested() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let signaller = Arc::clone(&shutdown);
        std::thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            signaller.store(true, Ordering::Relaxed);
        });

        let started = Instant::now();
        sleep_unless_shutdown(DECODER_RESTART_BACKOFF, &shutdown);
        assert!(
            started.elapsed() < DECODER_RESTART_BACKOFF / 2,
            "backoff outlived the shutdown request"
        );
    }

    #[test]
    fn decoder_restart_backoff_is_skipped_when_shutdown_is_already_requested() {
        let started = Instant::now();
        sleep_unless_shutdown(DECODER_RESTART_BACKOFF, &AtomicBool::new(true));
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn decoder_restart_backoff_runs_to_completion_without_a_shutdown() {
        let started = Instant::now();
        sleep_unless_shutdown(Duration::from_millis(250), &AtomicBool::new(false));
        assert!(started.elapsed() >= Duration::from_millis(250));
    }

    /// Milliseconds, not the production seconds: the schedule is a parameter so
    /// the retry can be exercised without the test waiting out a real backoff.
    const TEST_RETRY: RetrySchedule = RetrySchedule {
        start: Duration::from_millis(5),
        max: Duration::from_millis(20),
    };

    fn failing_build(attempts: &AtomicU32) -> impl FnMut() -> Result<(), &'static str> + '_ {
        || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err("no ffmpeg")
        }
    }

    /// The failure this retries — a decoder that would not spawn — used to end
    /// the analyzer task for good, leaving the camera recording nothing.
    #[test]
    fn analyzer_construction_is_retried_until_it_succeeds() {
        let attempts = AtomicU32::new(0);
        let built = build_with_retry(
            "cam",
            "motion analyzer",
            &AtomicBool::new(false),
            TEST_RETRY,
            || match attempts.fetch_add(1, Ordering::Relaxed) {
                0 | 1 => Err("no ffmpeg"),
                _ => Ok("analyzer"),
            },
        );
        assert_eq!(built, Some("analyzer"));
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn analyzer_construction_retry_ends_when_shutdown_is_requested() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let signaller = Arc::clone(&shutdown);
        std::thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            signaller.store(true, Ordering::Relaxed);
        });

        let attempts = AtomicU32::new(0);
        let started = Instant::now();
        let built = build_with_retry(
            "cam",
            "motion analyzer",
            &shutdown,
            TEST_RETRY,
            failing_build(&attempts),
        );
        assert!(built.is_none());
        assert!(
            attempts.load(Ordering::Relaxed) > 1,
            "gave up after a single attempt instead of retrying"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "retry loop outlived the shutdown request"
        );
    }

    /// Delays widen towards the ceiling and reports thin out, so a decoder that
    /// will never spawn costs a handful of lines rather than one a minute for
    /// as long as the process lives.
    #[test]
    fn a_decoder_that_never_spawns_backs_off_and_stops_repeating_itself() {
        let mut retry = DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE);
        let mut reported = Vec::new();
        let mut delays = Vec::new();
        for _ in 0..200 {
            let (delay, report) = retry.failed();
            delays.push(delay);
            if let Some(attempts) = report {
                reported.push(attempts);
            }
        }

        assert_eq!(&reported[..5], &[1, 2, 4, 8, 16]);
        assert!(
            reported.len() < 15,
            "200 failures produced {} lines",
            reported.len()
        );
        // Jitter is +/-20%, so the ceiling is a band rather than a value.
        let last = *delays.last().unwrap();
        assert!(
            last >= DECODER_SPAWN_BACKOFF_MAX * 4 / 5 && last <= DECODER_SPAWN_BACKOFF_MAX * 6 / 5,
            "{last:?} is not near the ceiling"
        );
        assert!(delays[0] < delays[3], "backoff did not widen");
    }

    /// Both decoder-spawn sites share one policy, so a working spawn puts the
    /// next failure back at the start of the schedule either way.
    #[test]
    fn a_successful_spawn_clears_the_backoff_and_the_streak() {
        let mut retry = DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE);
        for _ in 0..20 {
            retry.failed();
        }
        retry.succeeded();

        let (delay, report) = retry.failed();
        assert_eq!(report, Some(1), "escalation did not reset");
        assert!(delay <= DECODER_SPAWN_SCHEDULE.start * 6 / 5, "{delay:?}");
    }

    /// Jitter exists so a failure that hits every camera at once — an fd table
    /// that filled up — does not put every camera's retry on the same tick.
    #[test]
    fn two_cameras_failing_together_do_not_retry_in_lockstep() {
        let delays: std::collections::HashSet<Duration> = (0..16)
            .map(|_| DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE).failed().0)
            .collect();
        assert!(delays.len() > 1, "every camera drew the same delay");
    }

    /// A camera must not spawn an ffmpeg during the drain.
    #[test]
    fn analyzer_construction_is_not_attempted_once_shutdown_is_requested() {
        let attempts = AtomicU32::new(0);
        let built = build_with_retry(
            "cam",
            "motion analyzer",
            &AtomicBool::new(true),
            DECODER_SPAWN_SCHEDULE,
            failing_build(&attempts),
        );
        assert!(built.is_none());
        assert_eq!(attempts.load(Ordering::Relaxed), 0);
    }

    // ---- The crop decoder's life, without an ffmpeg to fork ----

    /// A child whose liveness is whatever the test says it is, so the reuse
    /// policy can be driven past deaths and respawns in the suite that gates
    /// commits rather than only behind `--ignored`.
    struct CountedChild {
        alive: bool,
        /// Which fork produced it. A pid is how the gated tests tell a kept
        /// child from its replacement; this is how the others do.
        generation: u32,
    }

    impl Respawnable for CountedChild {
        fn is_alive(&mut self) -> bool {
            self.alive
        }
    }

    /// A camera that keeps a decoder, standing in for the analyzer so the code
    /// a real batch runs can be driven without an ffmpeg to fork.
    struct TestCamera {
        slot: LongLived<CountedChild>,
    }

    impl KeepsDecoder<CountedChild> for TestCamera {
        fn slot(&mut self) -> &mut LongLived<CountedChild> {
            &mut self.slot
        }

        fn camera(&self) -> &str {
            "cam"
        }
    }

    /// A stop that has not been requested: what every ordinary pass runs under.
    fn running() -> AtomicBool {
        AtomicBool::new(false)
    }

    /// What one motion batch asks of the slot, with the fork counted instead of
    /// made.
    fn ensure_counted(
        slot: &mut LongLived<CountedChild>,
        window: ForkWindow,
        forks: &mut u32,
    ) -> bool {
        ensure_long_lived(slot, window, "cam", || {
            *forks += 1;
            Ok(CountedChild {
                alive: true,
                generation: *forks,
            })
        })
    }

    /// One whole motion batch, borrowed decoder and all.
    fn batch_on(
        camera: &mut TestCamera,
        window: ForkWindow,
        forks: &mut u32,
        batch: impl FnOnce(&mut LongLived<CountedChild>),
    ) {
        lend_for_batch(
            camera,
            window,
            || {
                *forks += 1;
                Ok(CountedChild {
                    alive: true,
                    generation: *forks,
                })
            },
            |_, crop| batch(crop),
        );
    }

    /// Up to five fork/exec/kill cycles a second per camera is what a crop
    /// decoder built per batch cost, each one paying ffmpeg's stream probe
    /// again. A batch that finds a living child must use it.
    ///
    /// Driven through [`lend_for_batch`] rather than the policy alone, because
    /// the half of this that no policy can state is the return: a batch that
    /// kept the decoder to itself would leave the camera empty and the next
    /// batch would fork. That is per-batch forking again, and the only thing
    /// that used to notice was a test needing a real ffmpeg and two pids.
    #[test]
    fn a_crop_decoder_is_forked_once_and_kept_across_batches() {
        let stop = running();
        let mut camera = TestCamera {
            slot: LongLived::default(),
        };
        let mut forks = 0;
        let mut children = Vec::new();
        let window = ForkWindow::Open(&stop);
        for _ in 0..5 {
            batch_on(&mut camera, window, &mut forks, |crop| {
                let child = crop.decoder.as_ref().expect("the batch got no decoder");
                children.push(child.generation);
                // What an extraction does with the child it was handed.
                crop.primed_with = PRIMING_SEGMENTS;
            });
        }

        assert_eq!(forks, 1, "the crop decoder was forked per batch");
        assert_eq!(children, [1; 5], "the batches ran on different children");
        assert!(
            camera.slot.decoder.is_some(),
            "the batch kept the camera's crop decoder instead of giving it back"
        );
        assert!(
            camera.slot.primed(),
            "the camera lost what its child had already been told"
        );
    }

    /// Priming is addressed to one particular ffmpeg's stream probe, so it
    /// follows the child rather than the batch: paid when the child appears,
    /// paid again when it is replaced, and never in between.
    #[test]
    fn a_respawned_crop_decoder_is_primed_again_and_only_then() {
        let stop = running();
        let window = ForkWindow::Open(&stop);
        let mut slot = LongLived::default();
        let mut forks = 0;

        assert!(ensure_counted(&mut slot, window, &mut forks));
        assert!(
            !slot.primed(),
            "a fresh child arrived claiming to be primed"
        );
        slot.primed_with = PRIMING_SEGMENTS;

        assert!(ensure_counted(&mut slot, window, &mut forks));
        assert!(slot.primed(), "re-primed a child that never died");
        assert_eq!(forks, 1);

        slot.decoder.as_mut().unwrap().alive = false;
        assert!(ensure_counted(&mut slot, window, &mut forks));
        assert_eq!(forks, 2, "a dead child was not replaced");
        assert!(!slot.primed(), "the replacement child was left unprimed");
    }

    /// The drain's promise, now kept for this decoder too: what is already
    /// running is used to the end, and nothing new is started. A crop decoder
    /// used to be forked per batch straight through the shutdown.
    #[test]
    fn the_drain_uses_a_running_crop_decoder_and_forks_no_other() {
        let mut forks = 0;

        let mut empty = LongLived::default();
        assert!(!ensure_counted(&mut empty, ForkWindow::Closed, &mut forks));

        let mut dead = LongLived {
            decoder: Some(CountedChild {
                alive: false,
                generation: 0,
            }),
            primed_with: PRIMING_SEGMENTS,
        };
        assert!(!ensure_counted(&mut dead, ForkWindow::Closed, &mut forks));
        assert_eq!(forks, 0, "the drain forked a crop decoder");

        let mut running_child = LongLived {
            decoder: Some(CountedChild {
                alive: true,
                generation: 0,
            }),
            primed_with: PRIMING_SEGMENTS,
        };
        assert!(
            ensure_counted(&mut running_child, ForkWindow::Closed, &mut forks),
            "the drain refused a decoder it was already holding"
        );
        assert_eq!(forks, 0);
    }

    /// The stop this refusal turns on is read where the fork is, not where the
    /// pass began. Between the two lies a whole batch of segments — every one
    /// of them decoded and scored, seconds of it on a camera working through a
    /// backlog — and a stop requested in that window is one this camera has
    /// already been told about by the time it forks.
    #[test]
    fn a_stop_requested_after_the_pass_began_forks_no_crop_decoder() {
        let stop = running();
        // What the top of the pass saw, carried down the way a pass carries it.
        let window = ForkWindow::Open(&stop);
        assert!(window.is_open(), "nothing had been requested yet");

        // ... the batch is decoded, and the stop arrives while it is.
        stop.store(true, Ordering::Relaxed);

        let mut camera = TestCamera {
            slot: LongLived::default(),
        };
        let mut forks = 0;
        batch_on(&mut camera, window, &mut forks, |_| {
            panic!("a batch ran on a decoder forked after the stop")
        });
        assert_eq!(forks, 0, "a stop mid-pass still forked a crop decoder");
        assert!(camera.slot.decoder.is_none());
    }

    /// The priming loop's whole reason for this shape: the hot buffer's read
    /// lock is held long enough to collect [`PRIMING_SEGMENTS`] handles and
    /// released before ffmpeg is handed the first of them.
    ///
    /// Pinned from the camera thread's own position — the decode does what a
    /// `push` does and takes the write lock. Under the old shape it could not
    /// have: the read guard was still alive, and ingest waited out three
    /// decodes on every motion batch.
    #[test]
    fn priming_decodes_with_the_hot_buffer_lock_already_released() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = test_context("cam", dir.path());
        let buffer = Arc::clone(&ctx.buffer);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..5 {
                buf.push(gop(seq));
            }
        }
        let analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

        let mut slot = LongLived {
            decoder: Some(CountedChild {
                alive: true,
                generation: 1,
            }),
            primed_with: 0,
        };
        let mut primed: Vec<(Arc<Vec<u8>>, u64)> = Vec::new();
        analyzer.prime_with(&mut slot, 3, |_, data, duration_ns| {
            assert!(
                buffer.try_write().is_ok(),
                "the hot buffer was still locked while a priming segment was decoded"
            );
            primed.push((Arc::clone(data), duration_ns));
        });

        assert_eq!(primed.len(), PRIMING_SEGMENTS as usize);
        assert_eq!(primed[0].1, SEC);
        // Shared with the buffer rather than copied out of it: a GOP is
        // megabytes, and the lock is held for the pointer.
        let resident = buffer.read_recover();
        assert!(
            Arc::ptr_eq(
                &primed[0].0,
                &resident.get_segment_by_sequence(0).expect("seq 0").data
            ),
            "the priming segments were copied out from under the lock"
        );
        drop(resident);

        // The next run of the same batch, and every run after it, finds the
        // child already past its probe.
        assert!(slot.primed());
        analyzer.prime_with(&mut slot, 4, |_, _, _| {
            panic!("primed a child that had already been primed")
        });
    }

    /// Priming is a claim about one child having been carried past its stream
    /// probe, and only the whole window carries it there. A run at the very
    /// start of a camera's buffer has nothing behind it; a run just past the
    /// buffer's edge has some of it, because the rest aged out while the
    /// analyzer worked through the batch. Neither is past the probe, and
    /// neither may leave the child recorded as though it were — the next run is
    /// further in and finds the whole window resident, so the claim is only
    /// deferred.
    #[test]
    fn a_child_primed_on_part_of_the_window_tries_again_on_the_next_run() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = test_context("cam", dir.path());
        let buffer = Arc::clone(&ctx.buffer);
        {
            // More footage than the hot buffer's retention holds, so the oldest
            // sequences are gone exactly as they go in production.
            let mut buf = buffer.write_recover();
            for seq in 0..33 {
                buf.push(gop(seq));
            }
        }
        let first_resident = buffer.read_recover().first_sequence();
        assert!(
            first_resident > 0,
            "nothing was evicted, so nothing is partial"
        );
        let analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

        let mut slot = LongLived {
            decoder: Some(CountedChild {
                alive: true,
                generation: 1,
            }),
            primed_with: 0,
        };

        // A run so early in the camera's life that there is nothing behind it.
        let mut fed = 0;
        analyzer.prime_with(&mut slot, PRIMING_SEGMENTS - 1, |_, _, _| fed += 1);
        assert_eq!(fed, 0, "primed with segments the buffer never held");
        assert!(!slot.primed(), "a child that saw nothing was called primed");

        // A run whose window straddles the buffer's edge: part of it is footage
        // the camera has already dropped.
        let partial = first_resident + PRIMING_SEGMENTS - 1;
        analyzer.prime_with(&mut slot, partial, |_, _, _| fed += 1);
        assert_eq!(fed, PRIMING_SEGMENTS - 1, "the resident part was not fed");
        assert!(
            !slot.primed(),
            "a child that got part of the window was called primed"
        );

        // The next run, one sequence further in, has the whole window.
        analyzer.prime_with(&mut slot, partial + 1, |_, _, _| fed += 1);
        assert_eq!(
            fed,
            2 * PRIMING_SEGMENTS - 1,
            "the run with the whole window did not prime"
        );
        assert!(slot.primed());
    }

    /// A camera whose hot buffer is shorter than the priming window never has
    /// the whole thing behind a run — not on the next run, not on any of them.
    /// Waiting for a window that cannot arrive would leave the child unprimed
    /// for the life of the camera and re-feeding overlapping scraps on every
    /// motion batch, forever: not a leak and not a stall, but a decoder
    /// permanently paying a cost that is supposed to be paid once.
    ///
    /// What the child needs is input, not particular sequences, so scraps count
    /// towards the same total and the third one settles it.
    #[test]
    fn a_buffer_too_short_for_the_priming_window_still_primes_its_child() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = test_context("cam", dir.path());
        // Two seconds of retention against a three-segment window: whatever the
        // camera does, a run never has more than one segment behind it.
        ctx.buffer = HotBuffer::new("cam".to_string(), 2);
        let buffer = Arc::clone(&ctx.buffer);
        let analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

        let mut slot = LongLived {
            decoder: Some(CountedChild {
                alive: true,
                generation: 1,
            }),
            primed_with: 0,
        };

        let mut fed = 0;
        let mut fed_by_run = Vec::new();
        for seq in 0..8 {
            buffer.write_recover().push(gop(seq));
            let before = fed;
            analyzer.prime_with(&mut slot, seq, |_, _, _| fed += 1);
            fed_by_run.push(fed - before);
        }

        assert!(
            slot.primed(),
            "a camera with a buffer this short never got its child past the probe"
        );
        assert_eq!(
            fed, PRIMING_SEGMENTS,
            "the child was fed {fed} segments to spend a {PRIMING_SEGMENTS}-segment probe"
        );
        assert_eq!(
            fed_by_run.iter().sum::<u64>(),
            PRIMING_SEGMENTS,
            "runs kept feeding a child that was already primed: {fed_by_run:?}"
        );
    }

    /// The overlay's full frame is a 1080p encode per motion run, and up to a
    /// third of what the debug store then pins per entry. It follows the view
    /// being open — not detection being enabled, which is how it was decided
    /// before and is true all night on a box nobody is looking at.
    ///
    /// Driven through the run-processing path rather than the helper it calls,
    /// so that what is pinned is the decision the job is actually built with:
    /// a caller that went back to encoding unconditionally would still pass a
    /// test of the helper alone.
    #[tokio::test]
    async fn the_debug_overlay_frame_is_encoded_only_while_the_view_is_open() {
        let dir = tempfile::tempdir().unwrap();
        let debug_store = DetectionDebugStore::new(&["cam".to_string()]);
        let (detect_tx, queue) = crate::analytics::detect_worker::detect_queue(None);
        let mut ctx = test_context("cam", dir.path());
        ctx.debug_store = Some(debug_store.clone());
        ctx.detect_tx = Some(detect_tx);
        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

        let frames = || {
            vec![(
                RgbFrame {
                    data: vec![0u8; 4 * 4 * 3],
                    width: 4,
                    height: 4,
                },
                Some(FULL_FRAME),
            )]
        };
        let segment = |seq| MotionSegment {
            seq,
            data: Arc::new(Vec::new()),
            duration_ns: 1,
        };

        analyzer.process_run_frames(vec![segment(0)], frames());
        let job = queue.recv().await.expect("a crop job per run");
        assert!(
            job.full_frame_jpeg.is_none(),
            "encoded a debug overlay for a view nobody has open"
        );
        // The crops the model needs are unaffected by the gate, and reach the
        // job as handles.
        assert_eq!(job.crop_jpegs.len(), 1);

        debug_store.list("cam");
        analyzer.process_run_frames(vec![segment(1)], frames());
        let job = queue.recv().await.expect("a crop job per run");
        assert!(
            job.full_frame_jpeg.is_some(),
            "the view is open and its overlay has no frame to draw on"
        );

        debug_store.mark_requested_ago("cam", Duration::from_secs(600));
        analyzer.process_run_frames(vec![segment(2)], frames());
        let job = queue.recv().await.expect("a crop job per run");
        assert!(
            job.full_frame_jpeg.is_none(),
            "kept encoding after the viewer left"
        );
    }

    /// Nothing else can give the memory back. Both producers stop the instant
    /// the window closes, so no insert ever comes to notice that it has —
    /// without this tick the last session's tens of megabytes per camera stay
    /// resident until the next detection, which on a quiet night is never.
    ///
    /// The pass is driven with a dead decoder and the stop already requested,
    /// which is what pins the *position* of the expiry rather than only its
    /// existence: this pass analyzes nothing and forks nothing, so an expiry
    /// anywhere below the decoder check — where it used to be — never runs at
    /// all. That is the state the position is for: a camera whose decoder will
    /// not respawn, on a box short enough of memory that a fork fails.
    #[test]
    fn the_analyzer_tick_gives_back_what_an_ended_session_was_holding() {
        let dir = tempfile::tempdir().unwrap();
        let debug_store = DetectionDebugStore::new(&["cam".to_string()]);
        let mut ctx = test_context("cam", dir.path());
        ctx.debug_store = Some(debug_store.clone());
        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        let stopping = AtomicBool::new(true);

        debug_store.list("cam");
        debug_store.insert(
            "cam",
            vec![Arc::new(vec![0xaa])],
            Vec::new(),
            "test-model".to_string(),
            0,
            Some(Arc::new(vec![0xbb])),
            Vec::new(),
            None,
            Vec::new(),
        );
        assert_eq!(debug_store.stored("cam"), 1);

        // A pass while somebody is still watching keeps them.
        analyzer.tick(&stopping);
        assert_eq!(
            debug_store.stored("cam"),
            1,
            "took the frames away from a viewer who is still there"
        );

        debug_store.mark_requested_ago("cam", Duration::from_secs(600));
        analyzer.tick(&stopping);
        assert_eq!(
            debug_store.stored("cam"),
            0,
            "the viewer left and the frames stayed resident"
        );
    }

    /// The other thing a pass gives back before it looks at its decoder, and
    /// for the same reason as the expiry above.
    ///
    /// A crop decoder's channel fills itself: its reader thread keeps handing
    /// frames over from segments the last batch fed, whatever the rest of the
    /// analyzer is doing. Four of them is 24 MB at the detection crop size, and
    /// the release used to sit inside the part of a pass that a dead primary
    /// decoder never reaches — so a camera stuck in a respawn loop, on a box
    /// too short of memory to fork, held those 24 MB for as long as the loop
    /// ran. That is the same compounding shape the debug-frame expiry was
    /// hoisted over this gate to avoid.
    ///
    /// Driven with a decoder that is already gone and the stop already
    /// requested, which is what pins the *position*: this pass returns at the
    /// gate having decoded nothing, so a release anywhere below it never runs.
    /// The gate is one expression, so above it is above both of its halves.
    #[test]
    fn a_pass_that_gives_up_on_its_decoder_still_empties_the_crop_channel() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = test_context("cam", dir.path());
        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        let (crop, frames) = CropDecoder::detached();
        analyzer.crop_decoder = LongLived {
            decoder: Some(crop),
            primed_with: PRIMING_SEGMENTS,
        };
        // What the child handed over after the last batch stopped reading.
        let fill = |frames: &std::sync::mpsc::SyncSender<Vec<u8>>| {
            for _ in 0..4 {
                frames.try_send(vec![0u8; 8]).expect("the channel has room");
            }
        };

        fill(&frames);
        assert!(
            !analyzer.tick(&AtomicBool::new(true)),
            "the pass was supposed to give up before analyzing anything"
        );
        assert_eq!(
            analyzer.release_idle_crop_frames(),
            0,
            "a pass that gave up at the decoder left the crop decoder's frames pinned"
        );

        // And the harness can hold frames the release can find, so the
        // assertion above was about the pass and not about an empty channel.
        fill(&frames);
        assert_eq!(analyzer.release_idle_crop_frames(), 4);
    }

    /// The release at the top of a pass is not the last word, because a pass
    /// that cannot replace its decoder does not end at the gate — it waits out
    /// the respawn backoff there, five seconds widening towards a minute, and
    /// for as many cycles as the fork keeps failing. The crop decoder's reader
    /// thread does not pause for any of that: it goes on handing over frames
    /// from segments fed before the outage. Frames that arrive one millisecond
    /// after the pass-top release, released only by the *next* pass, are 24 MB
    /// pinned for a backoff — the thing the hoist above the gate was for,
    /// undone by the sleep below it.
    ///
    /// So the frames here are handed over from another thread while the pass is
    /// already parked, which no assertion at either end of a pass could tell
    /// apart from frames that were there all along.
    #[test]
    fn frames_arriving_during_the_respawn_backoff_are_released_before_the_pass_returns() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = test_context("cam", dir.path());
        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        let (crop, frames) = CropDecoder::detached();
        analyzer.crop_decoder = LongLived {
            decoder: Some(crop),
            primed_with: PRIMING_SEGMENTS,
        };
        // The real schedule starts at five seconds. This one is the shape of a
        // backoff — several [`POLL_INTERVAL`] wakeups — at the length of a
        // test.
        const BACKOFF: Duration = Duration::from_millis(800);
        analyzer.decoder_retry = DecoderSpawnRetry::new(RetrySchedule {
            start: BACKOFF,
            max: BACKOFF,
        });

        // The reader thread's part: nothing at the top of the pass, and four
        // frames once it is asleep in the backoff.
        let reader = std::thread::spawn(move || {
            std::thread::sleep(POLL_INTERVAL / 2);
            for _ in 0..4 {
                frames.try_send(vec![0u8; 8]).expect("the channel has room");
            }
        });

        let stop = running();
        assert!(
            !analyzer.ensure_decoder_alive_with(&stop, || Err(std::io::Error::other("no ffmpeg"))),
            "a fork that failed must not report a live decoder"
        );
        reader.join().expect("reader thread panicked");

        assert_eq!(
            analyzer.release_idle_crop_frames(),
            0,
            "frames handed over during the backoff stayed pinned until the next pass"
        );
    }

    fn test_context(camera_id: &str, data_dir: &std::path::Path) -> AnalyzerContext {
        let ids = [camera_id.to_string()];
        AnalyzerContext {
            camera_id: camera_id.to_string(),
            buffer: HotBuffer::new(camera_id.to_string(), 30),
            motion_store: MotionStore::new(&ids),
            detection_store: None,
            debug_store: None,
            detect_tx: None,
            event_registry: None,
            config: AnalyticsConfig::default(),
            motion_settings: MotionSettingsStore::new(
                &ids,
                data_dir,
                DEFAULT_VAR_THRESHOLD,
                DEFAULT_MIN_CONTOUR_AREA,
            ),
            event_tx: None,
            mqtt_tx: None,
            pre_padding_ns: 0,
            post_padding: Duration::from_secs(10),
            max_event_duration: Duration::from_secs(120),
        }
    }

    /// Collects the `segments_abandoned` field off whatever the body logs.
    ///
    /// The figure the operator reads is the whole point of the abandonment
    /// line, and the way to get it wrong is to report from the position the
    /// gate was handed rather than from where scoring actually stopped — which
    /// no assertion on the analyzer's own state can catch, because both numbers
    /// are right about different questions. So the log itself is the assertion.
    /// (`camera::rtsp`'s tests carry a capture layer of their own; sharing one
    /// would mean a crate-level test-support module, which is more than this
    /// needs.)
    #[derive(Clone, Default)]
    struct AbandonedCounts(Arc<std::sync::Mutex<Vec<Option<u64>>>>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for AbandonedCounts {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            struct Field(Option<Option<u64>>);
            impl tracing::field::Visit for Field {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "segments_abandoned" {
                        // Logged as a `Option<u64>`, so `None` arrives as the
                        // literal "None" and anything else as the number.
                        let rendered = format!("{value:?}");
                        self.0 = Some(rendered.parse().ok());
                    }
                }
                fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                    if field.name() == "segments_abandoned" {
                        self.0 = Some(Some(value));
                    }
                }
            }
            let mut field = Field(None);
            event.record(&mut field);
            if let Some(count) = field.0 {
                self.0.lock().expect("capture poisoned").push(count);
            }
        }
    }

    fn abandonments(body: impl FnOnce()) -> Vec<Option<u64>> {
        use tracing_subscriber::layer::SubscriberExt;
        let counts = AbandonedCounts::default();
        let subscriber = tracing_subscriber::registry().with(counts.clone());
        tracing::subscriber::with_default(subscriber, body);
        let captured = counts.0.lock().expect("capture poisoned").clone();
        captured
    }

    const SEC: u64 = 1_000_000_000;

    fn gop(index: u64) -> crate::buffer::GopSegment {
        crate::buffer::GopSegment {
            start_pts: index * SEC,
            duration_ns: SEC,
            data: Arc::new(vec![0x47; 188]),
            frame_count: 1,
        }
    }

    /// An analyzer whose decoder is already gone, wired to a writer channel and
    /// carrying an open motion run over everything analyzed so far — the state
    /// a real one is in when its ffmpeg dies just as the stop begins.
    ///
    /// Built without forking anything, so the drain's most fragile path is
    /// pinned by a test that runs in the suite that gates commits rather than
    /// behind `--ignored`.
    fn analyzer_with_a_dead_decoder(
        dir: &std::path::Path,
        analyzed_through: u64,
    ) -> (
        MotionAnalyzer,
        Arc<RwLock<HotBuffer>>,
        tokio::sync::mpsc::Receiver<WriterMessage>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut ctx = test_context("cam", dir);
        ctx.event_tx = Some(tx);
        let buffer = Arc::clone(&ctx.buffer);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..=analyzed_through {
                buf.push(gop(seq));
            }
        }

        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        let now = Instant::now();
        for seq in 0..=analyzed_through {
            analyzer.run_tracker.observe(seq, true, now);
        }
        analyzer.last_processed = analyzed_through + 1;
        analyzer.observed_sequences = true;
        (analyzer, buffer, rx)
    }

    fn flushed_event(
        rx: &mut tokio::sync::mpsc::Receiver<WriterMessage>,
    ) -> crate::buffer::warm::FinishedEvent {
        match rx.try_recv().expect("no event was flushed at all") {
            WriterMessage::Event(event) => event,
            WriterMessage::Upgrade(_) => panic!("the flush sent an upgrade"),
        }
    }

    /// A decoder that dies as the stop begins used to end the drain on the
    /// spot, and the run was then closed through whatever the buffer held at
    /// that instant — which is before the camera, still being joined, pushes
    /// the GOP in its hand. The recording lost its last second in exactly the
    /// scenario this whole phase exists for, and the log said it had kept it.
    ///
    /// Nothing scores that GOP, and nothing can: the analysis ends where the
    /// decoder did. The recording does not.
    #[test]
    fn a_dead_decoder_waits_out_the_camera_and_keeps_the_tail_it_cannot_score() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, buffer, mut rx) = analyzer_with_a_dead_decoder(dir.path(), 1);

        // The camera is still stopping; its last GOP lands after the drain has
        // begun, and its watermark only once phase 1 has joined it.
        let camera = Arc::clone(&buffer);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            camera.write_recover().push(gop(2));
            camera.write_recover().seal();
        });

        let started = Instant::now();
        analyzer.drain_tail(DrainGate::starting_at(started, TAIL_DRAIN_BOUND));
        analyzer.flush_open_run();

        assert!(
            started.elapsed() < TAIL_DRAIN_BOUND,
            "the analyzer sat out its whole bound instead of following the watermark"
        );
        assert_eq!(
            flushed_event(&mut rx).segments.len(),
            3,
            "the event stopped short of the GOP the camera pushed on its way out"
        );
    }

    /// The same wait, bounded. A camera that never finishes stopping must not
    /// hold a decoder-less analyzer any longer than it holds any other stalled
    /// consumer — and what is in the buffer when the bound trips still has to
    /// reach the recording.
    ///
    /// Also what the abandonment claims it cost. To end the wait, an analyzer
    /// with no decoder tells the gate it is finished consuming; reporting from
    /// that same position would print a shortfall of zero here and read as "the
    /// analyzer kept up", when in fact scoring stopped three segments back.
    #[test]
    fn a_dead_decoder_still_stops_at_its_drain_bound_and_says_what_went_unscored() {
        const BOUND: Duration = Duration::from_millis(300);

        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, buffer, mut rx) = analyzer_with_a_dead_decoder(dir.path(), 1);
        // Phase 1 gave up on this camera: a watermark that can still move, and
        // a camera that goes on recording past it and never says it stopped.
        buffer.write_recover().push(gop(2));
        buffer.write_recover().push(gop(3));
        buffer.write_recover().push(gop(4));
        buffer.write_recover().seal_provisionally();
        buffer.write_recover().push(gop(5));

        let started = Instant::now();
        let reported = abandonments(|| {
            analyzer.drain_tail(DrainGate::starting_at(started, BOUND));
        });
        analyzer.flush_open_run();

        assert!(started.elapsed() >= BOUND, "the wait was not the bound's");
        assert!(
            started.elapsed() < BOUND * 10,
            "a camera that never finished held the analyzer past its bound"
        );
        // Scored through seq 1, watermark at 5: seqs 2, 3 and 4 went unscored.
        assert_eq!(
            reported,
            vec![Some(3)],
            "the abandonment reported from the position that ended the wait, not from where \
             scoring stopped"
        );
        assert_eq!(
            flushed_event(&mut rx).segments.len(),
            6,
            "the flush left behind footage that was already in the buffer"
        );
    }

    /// The drain's refusal through the analyzer's own path: a batch that
    /// arrives after the drain has begun finds no crop decoder and starts none,
    /// whatever ffmpeg the box has.
    ///
    /// What carries the refusal down to here is the signature rather than any
    /// state — [`MotionAnalyzer::drain_tail`] holds no stop flag, so
    /// [`ForkWindow::Closed`] is the only window its passes can run in — which
    /// is why the drain is run first and then a batch is put through the window
    /// it leaves behind.
    #[test]
    fn a_batch_arriving_inside_the_drain_forks_no_crop_decoder() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, buffer, _rx) = analyzer_with_a_dead_decoder(dir.path(), 1);

        buffer.write_recover().seal();
        analyzer.drain_tail(DrainGate::starting_at(Instant::now(), TAIL_DRAIN_BOUND));

        analyzer.process_motion_runs(
            vec![MotionSegment {
                seq: 1,
                data: Arc::new(Vec::new()),
                duration_ns: SEC,
            }],
            ForkWindow::Closed,
        );
        assert!(
            analyzer.crop_decoder.decoder.is_none(),
            "the drain forked a crop decoder"
        );
    }

    // ---- Event assembly against a verdict arriving at the worst moment ----

    /// An analyzer with a live event registry, an open run over everything
    /// analyzed so far, and a writer channel of exactly `slots`. The spare
    /// sender is handed back so a test can occupy those slots and hold the
    /// analyzer's handoff open where a slow store holds it in production.
    fn analyzer_with_a_registry(
        dir: &std::path::Path,
        analyzed_through: u64,
        slots: usize,
    ) -> (
        MotionAnalyzer,
        EventRegistry,
        Arc<RwLock<HotBuffer>>,
        tokio::sync::mpsc::Sender<WriterMessage>,
        tokio::sync::mpsc::Receiver<WriterMessage>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(slots);
        let registry = EventRegistry::new(&["cam".to_string()]);
        let mut ctx = test_context("cam", dir);
        ctx.event_tx = Some(tx.clone());
        ctx.event_registry = Some(registry.clone());
        // A real store, empty: `assemble_event` reads it, finds nothing, and
        // classifies the event as movement — so an upgrade afterwards is the
        // only way it can become an object event.
        ctx.detection_store = Some(DetectionStore::new(&["cam".to_string()]));
        let buffer = Arc::clone(&ctx.buffer);
        {
            let mut buf = buffer.write_recover();
            for seq in 0..=analyzed_through {
                buf.push(gop(seq));
            }
        }

        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        let now = Instant::now();
        for seq in 0..=analyzed_through {
            analyzer.run_tracker.observe(seq, true, now);
        }
        analyzer.last_processed = analyzed_through + 1;
        analyzer.observed_sequences = true;
        (analyzer, registry, buffer, tx, rx)
    }

    /// An analyzer wired to a motion sensor and nothing else. These tests feed
    /// the run tracker straight through [`MotionAnalyzer::observe_run`], the
    /// one place segments reach it in production, so the sensor's view of a
    /// camera is pinned without a decoder in the way.
    fn analyzer_with_a_motion_sensor(
        dir: &std::path::Path,
    ) -> (MotionAnalyzer, tokio::sync::mpsc::Receiver<MqttEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let mut ctx = test_context("cam", dir);
        ctx.mqtt_tx = Some(tx);
        (MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead()), rx)
    }

    fn sensor_events(rx: &mut tokio::sync::mpsc::Receiver<MqttEvent>) -> Vec<&'static str> {
        let mut seen = Vec::new();
        while let Ok(event) = rx.try_recv() {
            seen.push(match event {
                MqttEvent::MotionStart { .. } => "start",
                MqttEvent::MotionEnd { .. } => "end",
                _ => "other",
            });
        }
        seen
    }

    /// Shutdown inside the quiet window of a run longer than the cap: the cap
    /// suspended the run on a padding segment, so its last chunk is already
    /// written and the flush has nothing to return. The motion sensor is still
    /// on all the same, and the shutdown flush is the last chance to clear it —
    /// gate that on the flush returning an event and Home Assistant shows
    /// movement until camon comes back, which is the leak the whole drain
    /// exists to close.
    #[test]
    fn the_shutdown_flush_clears_the_sensor_for_a_run_that_was_only_pending() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, mut rx) = analyzer_with_a_motion_sensor(dir.path());
        let t0 = Instant::now();
        let cap = Duration::from_secs(120);
        analyzer.observe_run(0, true, t0);
        analyzer.observe_run(1, true, t0 + cap - Duration::from_secs(2));
        // The cap crosses on padding: the chunk closes, the run stays pending.
        assert!(analyzer.observe_run(2, false, t0 + cap).is_some());
        assert_eq!(
            sensor_events(&mut rx),
            vec!["start"],
            "the cap boundary is event bookkeeping and must not touch the sensor"
        );

        analyzer.flush_open_run();
        assert_eq!(sensor_events(&mut rx), vec!["end"]);
    }

    /// A pending run whose window elapses on a motion segment dies and is
    /// replaced inside one call. Two physically separate motion periods, with a
    /// real gap between them that the events themselves report (the new run is
    /// not a continuation) — so the sensor must go off and on again, in that
    /// order. Read off liveness alone the tracker looks open throughout and the
    /// two fuse into one uninterrupted movement.
    #[test]
    fn a_run_replaced_after_its_window_elapses_ends_the_sensor_before_starting_it_again() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, mut rx) = analyzer_with_a_motion_sensor(dir.path());
        let t0 = Instant::now();
        let cap = Duration::from_secs(120);
        let post = Duration::from_secs(10);
        let last_motion = t0 + cap - Duration::from_secs(2);
        analyzer.observe_run(0, true, t0);
        analyzer.observe_run(1, true, last_motion);
        assert!(analyzer.observe_run(2, false, t0 + cap).is_some());
        assert_eq!(sensor_events(&mut rx), vec!["start"]);

        // Motion again, past the quiet window: a new run, not a continuation.
        assert_eq!(
            analyzer.observe_run(3, true, last_motion + post + Duration::from_secs(1)),
            None
        );
        assert_eq!(sensor_events(&mut rx), vec!["end", "start"]);
    }

    /// The ordinary cap boundary — motion straight through it — closes a chunk
    /// and opens its follow-on with nothing physically changing. The sensor
    /// must not hear about it, then or when the run finally ends.
    #[test]
    fn a_cap_boundary_in_the_middle_of_motion_is_silent_on_the_sensor() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, mut rx) = analyzer_with_a_motion_sensor(dir.path());
        let t0 = Instant::now();
        let cap = Duration::from_secs(120);
        let post = Duration::from_secs(10);
        analyzer.observe_run(0, true, t0);
        assert!(analyzer.observe_run(1, true, t0 + cap).is_some());
        assert_eq!(sensor_events(&mut rx), vec!["start"]);

        analyzer.observe_run(2, false, t0 + cap + post + Duration::from_secs(1));
        assert_eq!(sensor_events(&mut rx), vec!["end"]);
    }

    fn person() -> crate::storage::Verdict {
        crate::storage::Verdict {
            object_classes: vec!["person".to_string()],
            detections: vec![crate::storage::event_index::DetectionDetail {
                class: "person".to_string(),
                confidence: 0.9,
            }],
            backend: "ollama".to_string(),
            model: "test-model".to_string(),
        }
    }

    /// Long enough that a machine under load is not the reason a test fails,
    /// short enough that a message which is never coming is reported as one
    /// rather than hanging the suite.
    const PATIENCE: Duration = Duration::from_secs(10);

    /// Spin until the analyzer thread has got somewhere observable, failing
    /// with `what_went_wrong` rather than hanging if it never does.
    fn wait_for(mut reached: impl FnMut() -> bool, what_went_wrong: &str) {
        let waited = Instant::now();
        while !reached() {
            assert!(waited.elapsed() < PATIENCE, "{what_went_wrong}");
            std::thread::yield_now();
        }
    }

    async fn next_message(
        rx: &mut tokio::sync::mpsc::Receiver<WriterMessage>,
        expected: &str,
    ) -> WriterMessage {
        match tokio::time::timeout(PATIENCE, rx.recv()).await {
            Ok(Some(message)) => message,
            Ok(None) => panic!("the writer channel closed before {expected}"),
            Err(_) => panic!("{expected} never arrived"),
        }
    }

    fn filler() -> WriterMessage {
        WriterMessage::Upgrade(EventUpgrade::for_event(
            UpgradeTarget {
                start_pts_ns: u64::MAX,
                duration_ms: 0,
                continues: false,
            },
            person(),
        ))
    }

    /// The race the whole registry exists for, and both of its windows.
    ///
    /// The analyzer reads the detection store, assembles the event, and then
    /// blocks handing it to a writer that is one slow remote store away —
    /// minutes, in production. A verdict landing anywhere from the store read
    /// to the end of that handoff used to hit nothing on either path: the read
    /// had already happened, and there was no entry to claim until after the
    /// handoff returned. The event stayed movement-classified and its footage
    /// was deleted twelve days early.
    ///
    /// Both windows are held open here rather than waited for, so neither
    /// assertion rests on a timing:
    ///
    /// * the hot buffer's write lock is held, and `assemble_event` takes a
    ///   read lock on it before it touches the detection store — so a record
    ///   appearing while the lock is held proves the record was opened
    ///   *before* the store was read;
    /// * the writer channel's only slot is taken, so the analyzer provably
    ///   cannot have got its write away when the verdict lands.
    ///
    /// The store is real and empty, so the event is assembled as a movement
    /// event: the upgrade that follows is the only thing that can save its
    /// footage.
    #[tokio::test]
    async fn a_verdict_landing_while_the_analyzer_holds_the_write_still_upgrades_the_event() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, registry, buffer, spare, mut rx) =
            analyzer_with_a_registry(dir.path(), 2, 1);
        spare.try_send(filler()).expect("the channel starts empty");

        let run = analyzer
            .run_tracker
            .flush(Some(2))
            .expect("the analyzer carries an open run");
        // Nothing can be read out of the hot buffer until this is dropped, so
        // the analyzer cannot have reached the detection store read inside
        // `assemble_event`.
        let before_the_store_read = buffer.write_recover();
        let flushing = std::thread::spawn(move || analyzer.emit_event(run, None));

        wait_for(
            || registry.held("cam") > 0,
            "no record was open while the analyzer was still short of the detection store: a \
             verdict landing between the read and the handoff would have nothing to land on",
        );

        // Window one: the record exists, nothing has been read out of the
        // store yet.
        let targets = registry.deliver_verdict("cam", &[0, 1, 2], &person());
        assert!(
            targets.is_empty(),
            "an upgrade was sent for an event whose write is not in the channel yet: it would \
             reach the writer first and find no file"
        );
        drop(before_the_store_read);

        // Window two: assembly is done and the handoff is blocked on a full
        // channel, which is where a slow store holds it in production.
        assert!(
            !flushing.is_finished(),
            "the analyzer got its write away before the verdict landed; the window this test \
             needs was never open"
        );

        // Let go of the slot: the write goes in, and the upgrade the verdict
        // earned follows it down the same channel.
        next_message(&mut rx, "the filler message").await;
        let event = match next_message(&mut rx, "the event").await {
            WriterMessage::Event(event) => event,
            WriterMessage::Upgrade(_) => panic!("the upgrade overtook the write it belongs to"),
        };
        assert!(
            !event.has_objects,
            "the assembly saw detections the test never stored; the upgrade below would prove \
             nothing"
        );
        match next_message(&mut rx, "the upgrade the verdict earned").await {
            WriterMessage::Upgrade(upgrade) => {
                assert_eq!(upgrade.start_pts_ns, event.first_pts);
                assert_eq!(upgrade.duration_ms, event.duration_ms() as u32);
                assert_eq!(upgrade.object_classes, vec!["person".to_string()]);
            }
            WriterMessage::Event(_) => panic!("a second event was written"),
        }
        flushing.join().expect("the analyzer panicked");
    }

    /// The same verdict on the other side of the handoff. Once the write is in
    /// the channel the record carries the identity the file will have, so the
    /// detection worker upgrades it itself — and the identity it gets has to
    /// be the one the write is about to create.
    #[test]
    fn a_verdict_landing_after_the_write_is_queued_upgrades_the_event_it_names() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, registry, _buffer, _spare, mut rx) =
            analyzer_with_a_registry(dir.path(), 2, 4);

        analyzer.flush_open_run();
        let event = flushed_event(&mut rx);

        assert_eq!(
            registry.deliver_verdict("cam", &[0, 1, 2], &person()),
            [UpgradeTarget {
                start_pts_ns: event.first_pts,
                duration_ms: event.duration_ms() as u32,
                continues: false,
            }]
        );
    }

    /// A write that never left the analyzer — the writer was already gone —
    /// creates no file, so its record goes with it and a verdict that arrives
    /// afterwards has nothing to rewrite. The detections are still in the
    /// detection store and the API, as they always were.
    #[test]
    fn an_event_whose_write_never_left_leaves_no_record_behind() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, registry, _buffer, spare, rx) =
            analyzer_with_a_registry(dir.path(), 2, 4);
        drop(rx);
        drop(spare);

        analyzer.flush_open_run();

        assert_eq!(
            registry.held("cam"),
            0,
            "a record outlived the event whose write was lost"
        );
        assert!(registry
            .deliver_verdict("cam", &[0, 1, 2], &person())
            .is_empty());
    }

    /// Phase 2 of the stop flushes open runs while crop jobs for them are
    /// still queued behind a model that answers one request at a time. The
    /// detection worker is not aborted until the analyzers have been joined,
    /// so those verdicts are not *yet* impossible — they are simply not
    /// coming in time, and the abort that follows the join ends them for good.
    /// The registry must not make the drain wait for one, and must not lose
    /// the half that did happen: the write is at the writer's door and the
    /// record stays, unresolved and harmless, until the process goes away
    /// with it.
    #[test]
    fn a_flush_at_shutdown_neither_waits_for_a_verdict_nor_drops_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, registry, _buffer, _spare, mut rx) =
            analyzer_with_a_registry(dir.path(), 2, 4);
        // A crop job queued behind a model that will be aborted still holding it.
        let _never_answered = registry.expect_verdict("cam", &[0, 1, 2]);

        let started = Instant::now();
        analyzer.flush_open_run();

        assert!(
            started.elapsed() < TAIL_DRAIN_BOUND,
            "the flush waited on a verdict instead of getting the recording away"
        );
        assert_eq!(
            flushed_event(&mut rx).segments.len(),
            3,
            "the event the drain exists to save never reached the writer"
        );
        assert_eq!(
            registry.held("cam"),
            1,
            "the record was dropped while its verdict was still outstanding"
        );
    }

    /// The other end of that stop: a verdict that *did* park during the
    /// handoff, and a writer that is gone by the time the analyzer comes to
    /// send the upgrade it earned. Phase 3 drains the writers after the
    /// analyzers, so this is narrow — but it is reachable, and the one thing
    /// it must not do is wedge a drain that is already on a bound.
    ///
    /// What it costs is the upgrade, with an error line saying so: the event
    /// keeps movement retention. What it must not cost is the write, which is
    /// the footage, and which is already through.
    ///
    /// Both halves are deterministic. The event lands in the channel's only
    /// slot, so the upgrade behind it cannot have been sent; closing the
    /// receiver from there fails that send whether the analyzer has reached it
    /// yet or not.
    #[tokio::test]
    async fn an_upgrade_with_no_writer_left_to_take_it_is_dropped_not_waited_on() {
        let dir = tempfile::tempdir().unwrap();
        let (mut analyzer, registry, _buffer, spare, mut rx) =
            analyzer_with_a_registry(dir.path(), 2, 1);
        spare.try_send(filler()).expect("the channel starts empty");

        let flushing = std::thread::spawn(move || analyzer.flush_open_run());
        wait_for(
            || registry.held("cam") > 0,
            "no record was opened for the event",
        );
        assert!(
            registry
                .deliver_verdict("cam", &[0, 1, 2], &person())
                .is_empty(),
            "an upgrade was sent for an event whose write is not in the channel yet"
        );

        // Let the write through and no further: with the event in the only
        // slot, the upgrade has not been sent and cannot be.
        next_message(&mut rx, "the filler message").await;
        wait_for(|| rx.len() == 1, "the write never reached the writer");

        // The writer goes away with the event in its queue and the upgrade
        // still in the analyzer's hand.
        rx.close();
        wait_for(
            || flushing.is_finished(),
            "the analyzer wedged on an upgrade no writer was left to take",
        );
        flushing.join().expect("the analyzer panicked");

        match rx
            .recv()
            .await
            .expect("the write was lost with the upgrade")
        {
            WriterMessage::Event(event) => assert_eq!(event.segments.len(), 3),
            WriterMessage::Upgrade(_) => panic!("the upgrade was queued after all"),
        }
        assert!(rx.recv().await.is_none());
    }

    /// The real spawn path, not just the retry helper it is built from: a task
    /// spawned into a drain must not construct anything and must not sit out a
    /// backoff before noticing.
    #[tokio::test]
    async fn spawn_analyzer_stops_at_once_when_shutdown_is_already_requested() {
        let dir = tempfile::tempdir().unwrap();
        let started = Instant::now();
        let handle = spawn_analyzer(
            test_context("cam", dir.path()),
            Arc::new(AtomicBool::new(true)),
        );
        handle.await.expect("analyzer task panicked");
        assert!(started.elapsed() < DECODER_SPAWN_SCHEDULE.start);
    }

    /// The other half of the real path — construction that actually spawns
    /// ffmpeg, then the run loop, then a clean stop. Ignored by default like
    /// the rest of the tests that need an `ffmpeg` binary.
    #[tokio::test]
    #[ignore]
    async fn spawn_analyzer_builds_a_real_analyzer_and_stops_on_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        let ctx = test_context("cam", dir.path());
        let buffer = Arc::clone(&ctx.buffer);
        let handle = spawn_analyzer(ctx, Arc::clone(&shutdown));

        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(!handle.is_finished(), "analyzer stopped on its own");

        // Both halves of a real stop: the flag, then the watermark phase 1
        // publishes once the camera has been joined. Without the second the
        // analyzer keeps draining, which is the point of the test below.
        shutdown.store(true, Ordering::Relaxed);
        buffer.write_recover().seal();
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("analyzer did not stop")
            .expect("analyzer task panicked");
    }

    /// The analyzer half of the lost tail: the stop flag is the start of a
    /// stop, not the end of the camera, and an analyzer that flushed its open
    /// run on the flag alone wrote an event ending several seconds before the
    /// footage did. It now keeps analyzing until the camera's watermark says
    /// there is nothing left to analyze.
    ///
    /// Needs a real decoder, so it is ignored by default like every other test
    /// here that forks ffmpeg.
    #[tokio::test]
    #[ignore]
    async fn an_analyzer_keeps_going_until_the_camera_publishes_its_watermark() {
        let dir = tempfile::tempdir().unwrap();
        let shutdown = Arc::new(AtomicBool::new(false));
        let ctx = test_context("cam", dir.path());
        let buffer = Arc::clone(&ctx.buffer);
        let handle = spawn_analyzer(ctx, Arc::clone(&shutdown));
        tokio::time::sleep(Duration::from_millis(500)).await;

        shutdown.store(true, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(
            !handle.is_finished(),
            "the analyzer exited on the flag alone, before the camera had finished"
        );

        buffer.write_recover().seal();
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("the analyzer ignored the watermark and sat out its whole drain bound")
            .expect("analyzer task panicked");
    }

    /// The fork count, against a real ffmpeg and through the real extraction
    /// path: two motion batches, one child, primed once. A crop decoder built
    /// per batch — as this one was — forked, primed and killed one for each of
    /// them, up to five times a second on a camera with continuous motion.
    ///
    /// Driven through [`MotionAnalyzer::process_motion_runs`] rather than the
    /// slot it keeps, because the slot is only half the claim: an extraction
    /// that stopped asking it for a decoder would pass a test of the policy
    /// alone. Ignored by default like every other test here that forks ffmpeg.
    #[test]
    #[ignore]
    fn two_motion_batches_are_extracted_through_one_crop_decoder() {
        const SEGMENTS: usize = 8;
        let dir = tempfile::tempdir().unwrap();
        let recorded = crate::analytics::decoder::tests::recorded_segments(SEGMENTS);
        let ctx = test_context("cam", dir.path());
        let buffer = Arc::clone(&ctx.buffer);
        {
            let mut buf = buffer.write_recover();
            for (i, data) in recorded.iter().enumerate() {
                buf.push(crate::buffer::GopSegment {
                    start_pts: i as u64 * SEC,
                    duration_ns: SEC,
                    data: Arc::clone(data),
                    frame_count: 25,
                });
            }
        }
        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        // The cheap crop size: this is about which ffmpeg decodes, not about
        // what the vision model would have been sent.
        analyzer.frame_use = FrameUse::Thumbnails;

        let batch = |seq: u64| {
            vec![MotionSegment {
                seq,
                data: Arc::clone(&recorded[seq as usize]),
                duration_ns: SEC,
            }]
        };

        let stop = running();
        analyzer.process_motion_runs(batch(3), ForkWindow::Open(&stop));
        let first = analyzer
            .crop_decoder
            .decoder
            .as_ref()
            .expect("the first batch forked no crop decoder")
            .child_id();
        assert!(first.is_some(), "the crop decoder has no child");
        assert!(
            analyzer.crop_decoder.primed(),
            "the first batch never primed the child it forked"
        );

        analyzer.process_motion_runs(batch(7), ForkWindow::Open(&stop));
        let second = analyzer
            .crop_decoder
            .decoder
            .as_ref()
            .expect("the second batch lost the crop decoder")
            .child_id();
        assert_eq!(
            first, second,
            "the second batch forked a crop decoder of its own"
        );
    }

    /// What a kept decoder holds between batches, and for how long.
    ///
    /// A batch stops reading when it has the frames it wants; ffmpeg is still
    /// working through the segments it was fed and fills the channel behind it.
    /// Four frames is 24 MB at the detection crop size, per camera, and nothing
    /// takes them until the camera moves again — which on a quiet night is the
    /// morning, on a box whose analytics memory has been the problem before.
    /// The top of the next pass is where they go.
    ///
    /// This is the half of that claim a real ffmpeg is needed for: that a batch
    /// leaves frames behind at all. That a pass is what takes them is pinned
    /// without one in
    /// [`a_pass_that_gives_up_on_its_decoder_still_empties_the_crop_channel`],
    /// which is also where the position of the release is pinned. Ignored by
    /// default: it needs an `ffmpeg` binary.
    #[test]
    #[ignore]
    fn a_batch_leaves_frames_behind_and_the_next_pass_takes_them() {
        // Enough footage for ffmpeg to be past its stream probe and emitting,
        // which is the only state in which it has anything to leave behind: a
        // run is sampled, not fed whole, so a batch hands over four segments.
        const SEGMENTS: usize = 16;
        let dir = tempfile::tempdir().unwrap();
        let recorded = crate::analytics::decoder::tests::recorded_segments(SEGMENTS);
        let ctx = test_context("cam", dir.path());
        let buffer = Arc::clone(&ctx.buffer);
        {
            let mut buf = buffer.write_recover();
            for (i, data) in recorded.iter().enumerate() {
                buf.push(crate::buffer::GopSegment {
                    start_pts: i as u64 * SEC,
                    duration_ns: SEC,
                    data: Arc::clone(data),
                    frame_count: 25,
                });
            }
        }
        let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
        analyzer.frame_use = FrameUse::Thumbnails;
        let stop = running();
        let batch = |from: u64| {
            (from..from + 4)
                .map(|seq| MotionSegment {
                    seq,
                    data: Arc::clone(&recorded[seq as usize]),
                    duration_ns: SEC,
                })
                .collect::<Vec<_>>()
        };
        // Long enough for a child that has been handed everything it is going
        // to get to finish saying so.
        //
        // It is also what this test does not pin, and the pause is here rather
        // than absent for exactly that reason. The release is best-effort per
        // pass — a drain stops at the first empty channel and the pipe's reader
        // thread can hand over one more frame immediately after — so with a
        // real child still emitting, a single pass is not guaranteed to empty
        // anything. What closes that is the next pass a poll later, which
        // is a property about repetition and not one an assertion here can hold
        // still for; see [`MotionAnalyzer::release_idle_crop_frames`]. Waiting
        // the child out first takes the race off the table so what remains is
        // the claim this test is for: that a pass with no motion is what takes
        // the frames, and that they are there to be taken.
        let settle = || std::thread::sleep(Duration::from_millis(500));

        analyzer.process_motion_runs(batch(3), ForkWindow::Open(&stop));
        analyzer.process_motion_runs(batch(7), ForkWindow::Open(&stop));
        settle();
        assert!(
            analyzer.release_idle_crop_frames() > 0,
            "the batches left nothing behind, so there is nothing here to release"
        );

        analyzer.process_motion_runs(batch(11), ForkWindow::Open(&stop));
        settle();
        // A pass, driven the way the one above the decoder gate runs: the stop
        // flag ends it immediately afterwards, which is the point — the release
        // is above everything that can stop a pass, so a pass that does nothing
        // else still does this.
        assert!(
            !analyzer.tick(&AtomicBool::new(true)),
            "the pass was supposed to give up before analyzing anything"
        );
        assert_eq!(
            analyzer.release_idle_crop_frames(),
            0,
            "the pass left the last batch's frames resident"
        );
    }

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

    /// The tagged frames are moved out of the vector, not copied out of it, so
    /// which frames survive — and that each keeps its own crop tag — has to be
    /// pinned: taking the wrong index is invisible in a filmstrip.
    #[test]
    fn subsample_tagged_keeps_four_frames_spread_over_the_run() {
        fn tagged(n: usize) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
            (0..n)
                .map(|i| {
                    let frame = RgbFrame {
                        data: vec![i as u8; 3],
                        width: 1,
                        height: 1,
                    };
                    let crop = NormalizedRect {
                        x: i as f32,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    };
                    (frame, Some(crop))
                })
                .collect()
        }
        fn picked(frames: Vec<(RgbFrame, Option<NormalizedRect>)>) -> Vec<u8> {
            frames
                .iter()
                .map(|(frame, crop)| {
                    assert_eq!(crop.unwrap().x, frame.data[0] as f32, "tag follows frame");
                    frame.data[0]
                })
                .collect()
        }

        // Four or fewer are all kept, untouched.
        assert_eq!(picked(subsample_tagged(tagged(4))), vec![0, 1, 2, 3]);
        // Five is the first length past the short-circuit, and the one where
        // the four picks pack tightest — a collision would silently return
        // three frames.
        assert_eq!(picked(subsample_tagged(tagged(5))), vec![0, 1, 3, 4]);
        assert_eq!(picked(subsample_tagged(tagged(6))), vec![0, 2, 4, 5]);
        assert_eq!(picked(subsample_tagged(tagged(12))), vec![0, 4, 8, 11]);
    }

    #[test]
    fn frames_per_segment_covers_the_final_pick_and_one_spare() {
        assert_eq!(frames_per_segment(1), FILMSTRIP_FRAMES + 1);
        assert_eq!(frames_per_segment(2), 3);
        assert_eq!(frames_per_segment(3), 3);
        assert_eq!(frames_per_segment(4), 2);
        // `sample_indices` never yields none, but a zero must not divide.
        assert_eq!(frames_per_segment(0), FILMSTRIP_FRAMES + 1);
    }

    /// The in-flight bound behind [`sample_run_frames`]: one push then one
    /// halve, so the accumulator is never more than a frame over its cap
    /// however many arrive. This is what keeps `sample_fps` — config with no
    /// ceiling — out of the memory peak.
    #[test]
    fn halve_past_never_settles_above_its_cap() {
        for cap in 1..8usize {
            let mut acc: Vec<usize> = Vec::new();
            for i in 0..1000usize {
                acc.push(i);
                assert!(acc.len() <= cap + 1, "cap {cap} peaked at {}", acc.len());
                halve_past(&mut acc, cap);
                assert!(acc.len() <= cap, "cap {cap} settled at {}", acc.len());
            }
            assert_eq!(acc[0], 0, "the first entry always survives");
        }
    }

    #[test]
    fn thin_evenly_keeps_both_ends_and_spreads_the_rest() {
        fn picked(n: usize, keep: usize) -> Vec<usize> {
            thin_evenly((0..n).collect(), keep)
        }
        // One frame is the segment's keyframe — the frame its crop tag was
        // measured on — not an arbitrary one from the middle.
        assert_eq!(picked(5, 1), vec![0]);
        assert_eq!(picked(5, 2), vec![0, 4]);
        assert_eq!(picked(5, 4), vec![0, 1, 2, 4]);
        assert_eq!(picked(10, 2), vec![0, 9]);
        assert_eq!(picked(10, 4), vec![0, 3, 6, 9]);
        // Fewer frames than asked for are kept whole, in order.
        assert_eq!(picked(3, 4), vec![0, 1, 2]);
        assert!(picked(0, 4).is_empty());
        assert!(picked(5, 0).is_empty());
    }

    /// The accumulator used to grow with every frame of every decoded segment
    /// — a run's worth of full-HD RGB, hundreds of megabytes live at once on a
    /// box with a history of OOM kills. Its size is now a property of the run
    /// length alone, and this is the ceiling.
    #[test]
    fn the_run_accumulator_cannot_outgrow_its_cap() {
        for run_len in 1..64usize {
            let sampled = sample_indices(run_len).len();
            let held = sampled * frames_per_segment(sampled);
            assert!(
                held <= RUN_FRAME_ACCUMULATOR_CAP,
                "a run of {run_len} segments holds {held} frames"
            );
        }
    }

    /// Drive the real [`sample_run_frames`] with a decoder that emits
    /// `frames(seq)` frames for each segment, every frame carrying its own
    /// segment and its position within that segment. Returns what the vision
    /// model and the event filmstrip are handed, as (run position, frame).
    fn run_selection(run_len: usize, frames: impl Fn(u64) -> usize) -> Vec<(u8, u8)> {
        let run: Vec<MotionSegment> = (0..run_len as u64)
            .map(|seq| MotionSegment {
                seq,
                // The decode closure reads the segment back out of its own
                // bytes, the way ffmpeg reads it out of the footage.
                data: Arc::new(vec![seq as u8]),
                duration_ns: 1_000_000_000,
            })
            .collect();
        let crops: HashMap<u64, NormalizedRect> = (0..run_len as u64)
            .map(|seq| {
                let crop = NormalizedRect {
                    x: seq as f32,
                    y: 0.0,
                    w: 1.0,
                    h: 1.0,
                };
                (seq, crop)
            })
            .collect();

        sample_run_frames(&run, &crops, 1, 1, |data, _duration_ns, sink| {
            let seq = data[0];
            for i in 0..frames(seq as u64) {
                sink(vec![seq, i as u8, 0]);
            }
        })
        .iter()
        .map(|(frame, crop)| {
            // The crop was measured on one segment's keyframe; a frame that
            // reached the accumulator under another segment's tag is the
            // failure this asserts against, not a cosmetic mismatch.
            assert_eq!(
                crop.expect("a sampled segment always carries its crop").x,
                frame.data[0] as f32,
                "crop tag followed the wrong segment"
            );
            (frame.data[0], frame.data[1])
        })
        .collect()
    }

    /// Which frames the vision model and the event filmstrip end up with.
    /// Pinned because nothing downstream can tell a badly chosen frame from a
    /// well chosen one — the model simply answers about whatever it is handed.
    #[test]
    fn a_run_is_reduced_to_four_frames_of_its_own_motion() {
        // A long run: one frame from each of the four sampled segments, spread
        // across it, never a frame from before it.
        assert_eq!(
            run_selection(9, |_| 5),
            vec![(0, 0), (3, 0), (6, 4), (8, 4)]
        );
        assert_eq!(
            run_selection(4, |_| 5),
            vec![(0, 0), (1, 0), (2, 4), (3, 4)]
        );
        // Too few segments to spend one frame each: the extras come from
        // within the segments.
        assert_eq!(
            run_selection(3, |_| 5),
            vec![(0, 0), (1, 0), (2, 0), (2, 4)]
        );
        assert_eq!(
            run_selection(2, |_| 5),
            vec![(0, 0), (0, 4), (1, 2), (1, 4)]
        );
        assert_eq!(
            run_selection(1, |_| 5),
            vec![(0, 0), (0, 1), (0, 3), (0, 4)]
        );
        // A segment that decoded fewer frames than asked for shortens the strip
        // rather than padding it from somewhere else.
        assert_eq!(run_selection(1, |_| 2), vec![(0, 0), (0, 1)]);
    }

    /// What the spare frame per segment buys. A crop decoder runs behind its
    /// segments whenever ffmpeg is still releasing what it swallowed while
    /// probing, so a short decode is the ordinary case rather than an exotic
    /// one — and with a single frame per segment there is nothing to fall back
    /// on when one goes missing.
    #[test]
    fn a_segment_that_decoded_short_costs_a_frame_not_its_place_in_the_strip() {
        let picked = run_selection(9, |seq| if seq == 3 { 1 } else { 5 });
        assert_eq!(picked, vec![(0, 0), (3, 0), (6, 4), (8, 4)]);
    }

    /// `sample_fps` is config with no ceiling, so a segment's decode is
    /// unbounded: at 30 fps over a 2s GOP it is 60 frames, 373 MB of raw RGB at
    /// the detection crop size. What the run keeps must not follow it.
    #[test]
    fn a_high_sample_fps_moves_the_frames_picked_but_not_how_many() {
        // Still four, still one per sampled segment — and still spanning each
        // segment rather than clustering at its start, which is what the
        // halving reservoir buys over keeping the first arrivals. The exact
        // frames pin the reservoir's size: widen it and these move.
        assert_eq!(
            run_selection(9, |_| 60),
            vec![(0, 0), (3, 0), (6, 58), (8, 58)]
        );
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
    fn skipped_segments_reports_the_dropped_range() {
        let skipped = SkippedSegments::between(10, 14).unwrap();
        assert_eq!(
            skipped,
            SkippedSegments {
                count: 4,
                from_seq: 10,
                to_seq: 13,
            }
        );
        // A single dropped segment still names itself.
        let one = SkippedSegments::between(10, 11).unwrap();
        assert_eq!(one.count, 1);
        assert_eq!((one.from_seq, one.to_seq), (10, 10));
    }

    #[test]
    fn skipped_segments_is_none_when_the_analyzer_kept_up() {
        assert_eq!(SkippedSegments::between(10, 10), None);
        // Ahead of the buffer's oldest resident segment: nothing was missed.
        assert_eq!(SkippedSegments::between(12, 10), None);
        assert_eq!(SkippedSegments::between(0, 0), None);
    }

    #[test]
    fn skipped_segments_reports_what_vanished_mid_collection() {
        // Evicted after the buffer snapshot, while later sequences survived.
        assert_eq!(
            SkippedSegments::of(&[10, 11, 12, 13, 14]),
            Some(SkippedSegments {
                count: 5,
                from_seq: 10,
                to_seq: 14,
            })
        );
        // A hole that is not contiguous still reports an exact count.
        let scattered = SkippedSegments::of(&[10, 14]).unwrap();
        assert_eq!(scattered.count, 2);
        assert_eq!((scattered.from_seq, scattered.to_seq), (10, 14));
        assert_eq!(SkippedSegments::of(&[]), None);
    }

    #[test]
    fn skip_reporter_coalesces_a_chronically_behind_analyzer() {
        let mut reporter = SkipReporter::default();
        let t0 = Instant::now();
        // The first loss is reported at once: a one-off is worth seeing.
        let first = reporter
            .record(SkippedSegments::between(0, 3).unwrap(), t0)
            .unwrap();
        assert_eq!(first.count, 3);

        // Every 200 ms poll after that skips something too, and warnings stay
        // on in release — so they accumulate instead of printing.
        let mut polls = 0;
        for tick in 1..200u64 {
            let at = t0 + Duration::from_millis(200 * tick);
            let seq = 3 + tick;
            if reporter
                .record(SkippedSegments::between(seq, seq + 1).unwrap(), at)
                .is_some()
            {
                polls += 1;
            }
        }
        // 40 s of polling: one line per interval, not one per poll.
        assert_eq!(polls, 1);
    }

    #[test]
    fn skip_reporter_totals_everything_it_held_back() {
        let mut reporter = SkipReporter::default();
        let t0 = Instant::now();
        reporter.record(SkippedSegments::between(0, 1).unwrap(), t0);
        assert!(reporter
            .record(SkippedSegments::between(1, 3).unwrap(), t0)
            .is_none());
        assert!(reporter
            .record(SkippedSegments::between(3, 6).unwrap(), t0)
            .is_none());
        let total = reporter
            .record(
                SkippedSegments::between(6, 7).unwrap(),
                t0 + SKIP_REPORT_INTERVAL,
            )
            .unwrap();
        // Nothing suppressed is lost: the count and the range cover the lot.
        assert_eq!(
            total,
            SkippedSegments {
                count: 6,
                from_seq: 1,
                to_seq: 6,
            }
        );
    }

    #[test]
    fn motion_verdict_needs_the_threshold() {
        let scored = |score| SegmentAnalysis {
            score,
            crop: None,
            motion_rects: Vec::new(),
        };
        assert!(!scored(0.0).has_motion());
        assert!(!scored(MOTION_THRESHOLD - 0.001).has_motion());
        assert!(scored(MOTION_THRESHOLD).has_motion());
    }

    /// A segment the decoder produced no frames for is not evidence of
    /// stillness, so `run_motion_analysis` feeds the tracker nothing for it.
    /// Scoring it quiet instead ends events on footage nobody looked at.
    #[test]
    fn unanalyzed_segments_do_not_end_an_event() {
        const POST: Duration = Duration::from_secs(10);
        const CAP: Duration = Duration::from_secs(300);
        let segments: Vec<_> = (0..20).map(|seq| pending(seq, SECOND_NS)).collect();
        let times = batch_instants(&segments, Instant::now());
        // Motion at each end of a stretch the decoder produced nothing for.
        let motion = |seq: u64| seq == 0 || seq == 16;

        let mut tracker = RunTracker::new(POST, CAP);
        let closed: Vec<_> = segments
            .iter()
            .zip(&times)
            .filter(|(seg, _)| motion(seg.seq))
            .filter_map(|(seg, &at)| tracker.observe(seg.seq, true, at))
            .collect();
        assert!(closed.is_empty(), "unanalyzed footage ended the event");
        let run = tracker.flush(None).unwrap();
        assert_eq!(run.first_motion_seq, 0);
        assert_eq!(run.last_seq, 16);

        // Scored quiet, the blind stretch invents the end of the event: post-
        // padding elapses inside it and the second motion becomes a separate
        // event.
        let mut as_quiet = RunTracker::new(POST, CAP);
        let closed: Vec<_> = segments
            .iter()
            .zip(&times)
            .filter_map(|(seg, &at)| as_quiet.observe(seg.seq, motion(seg.seq), at))
            .collect();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].last_seq, 10);
        assert_eq!(as_quiet.flush(None).unwrap().first_motion_seq, 16);
    }

    fn pending(seq: u64, duration_ns: u64) -> PendingSegment {
        PendingSegment {
            seq,
            data: Arc::new(Vec::new()),
            start_pts: seq * duration_ns,
            duration_ns,
        }
    }

    const SECOND_NS: u64 = 1_000_000_000;

    #[test]
    fn batch_instants_space_segments_by_their_media_duration() {
        let now = Instant::now();
        let segments = vec![
            pending(0, SECOND_NS),
            pending(1, 2 * SECOND_NS),
            pending(2, 0),
        ];
        let times = batch_instants(&segments, now);
        // The batch ends now and reaches back over its own three seconds.
        assert_eq!(times[0], now - Duration::from_secs(2));
        assert_eq!(times[1], now);
        assert_eq!(times[2], now);
    }

    #[test]
    fn batch_instants_of_a_single_segment_is_now() {
        let now = Instant::now();
        assert_eq!(batch_instants(&[pending(7, SECOND_NS)], now), vec![now]);
        assert!(batch_instants(&[], now).is_empty());
    }

    #[test]
    fn batch_instants_never_run_past_now() {
        let now = Instant::now();
        // Backlog reaching further back than the monotonic clock goes, with a
        // span no sum could hold. A capture instant in the future would freeze
        // the tracker's countdowns instead of merely mis-dating the footage.
        let segments = vec![
            pending(0, u64::MAX),
            pending(1, u64::MAX),
            pending(2, SECOND_NS),
            pending(3, SECOND_NS),
        ];
        let times = batch_instants(&segments, now);
        assert!(times.iter().all(|&at| at <= now), "instant past now");
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "capture order reversed"
        );
        assert_eq!(*times.last().unwrap(), now);
        assert_eq!(times[2], now - Duration::from_secs(1));
    }

    /// A batch of backlog must be scored as the footage it is: 90 seconds of
    /// continuous motion produces the same chunking whether it arrives one
    /// segment per poll or all at once after a stall.
    #[test]
    fn duration_cap_fires_inside_a_backlogged_batch() {
        const POST: Duration = Duration::from_secs(10);
        const CAP: Duration = Duration::from_secs(30);
        let segments: Vec<_> = (0..90).map(|seq| pending(seq, SECOND_NS)).collect();

        let mut tracker = RunTracker::new(POST, CAP);
        let closed: Vec<_> = segments
            .iter()
            .zip(batch_instants(&segments, Instant::now()))
            .filter_map(|(seg, at)| tracker.observe(seg.seq, true, at))
            .collect();

        assert_eq!(closed.len(), 2, "two chunks close, the third stays open");
        assert_eq!(closed[0].first_motion_seq, 0);
        assert_eq!(closed[0].last_seq, 29);
        assert!(!closed[0].continues);
        assert_eq!(closed[1].first_motion_seq, 30);
        assert_eq!(closed[1].last_seq, 59);
        assert!(closed[1].continues);
        assert!(tracker.is_open());

        // One shared instant for the whole batch is what this replaces: the
        // cap cannot fire at all, and the run closes late and oversized.
        let mut shared = RunTracker::new(POST, CAP);
        let now = Instant::now();
        assert!(segments
            .iter()
            .all(|seg| shared.observe(seg.seq, true, now).is_none()));
    }

    #[test]
    fn post_padding_elapses_inside_a_backlogged_batch() {
        const POST: Duration = Duration::from_secs(10);
        const CAP: Duration = Duration::from_secs(300);
        let segments: Vec<_> = (0..60).map(|seq| pending(seq, SECOND_NS)).collect();

        let mut tracker = RunTracker::new(POST, CAP);
        let closed: Vec<_> = segments
            .iter()
            .zip(batch_instants(&segments, Instant::now()))
            // Motion in the first segment only; the rest is quiet footage.
            .filter_map(|(seg, at)| tracker.observe(seg.seq, seg.seq == 0, at))
            .collect();

        assert_eq!(closed.len(), 1, "the run closes within the batch");
        assert_eq!(closed[0].first_motion_seq, 0);
        // Post-padding keeps the ten segments that followed the motion.
        assert_eq!(closed[0].last_seq, 10);
        assert!(!tracker.is_open());
    }

    #[test]
    fn detection_mask_ignores_wrong_length() {
        let mut frame = white_frame(160, 120);
        apply_detection_mask(&mut frame, FULL_FRAME, &[true, false, true]);
        assert!(frame.data.iter().all(|&b| b == 255));
    }
}
