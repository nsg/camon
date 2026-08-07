//! Decoder lifecycle machinery: spawn retry/backoff policy, the long-lived
//! decoder slot kept across batches, and the fork window a shutdown closes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::analytics::decoder::CropDecoder;
use crate::retry::{jittered, RetrySchedule, Streak};

use super::POLL_INTERVAL;

/// How long to wait before trying a dead decoder again.
pub(super) const DECODER_RESTART_BACKOFF: Duration = Duration::from_secs(5);

/// Ceiling on the decoder-spawn backoff. Spawning a decoder fails for two very
/// different reasons: a transient one (a fork that lost a race for memory, an
/// exhausted fd table) that clears on its own, and a permanent one (no ffmpeg on
/// PATH) that never does. Doubling from [`DECODER_RESTART_BACKOFF`] to a minute
/// serves both — the same shape the camera pipeline's reconnect uses — so the
/// first recovers within a minute of clearing and the second stops costing an
/// ffmpeg fork every five seconds.
pub(super) const DECODER_SPAWN_BACKOFF_MAX: Duration = Duration::from_secs(60);

pub(super) const DECODER_SPAWN_SCHEDULE: RetrySchedule = RetrySchedule {
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
pub(super) struct DecoderSpawnRetry {
    schedule: RetrySchedule,
    backoff: Duration,
    streak: Streak,
}

impl DecoderSpawnRetry {
    pub(super) fn new(schedule: RetrySchedule) -> Self {
        Self {
            schedule,
            backoff: schedule.start,
            streak: Streak::new(),
        }
    }

    /// Record a failed spawn. Returns how long to wait, and the streak length
    /// when this failure is one worth a log line.
    pub(super) fn failed(&mut self) -> (Duration, Option<u32>) {
        let delay = jittered(self.backoff);
        self.backoff = self.schedule.next(self.backoff);
        (delay, self.streak.record())
    }

    pub(super) fn succeeded(&mut self) {
        self.backoff = self.schedule.start;
        self.streak.reset();
    }
}

/// Segments fed to a freshly forked crop decoder before its frames are kept.
/// Three seconds of footage is enough to carry ffmpeg past the stream probe it
/// swallows its first input for; see [`MotionAnalyzer::prime_with`].
pub(super) const PRIMING_SEGMENTS: u64 = 3;

/// Consecutive zero-frame decodes tolerated before the decoder is declared
/// blind. A segment is one GOP and always opens on a keyframe, so a healthy
/// decode yields at least one I-frame — but a freshly spawned ffmpeg swallows
/// several seconds of input while it probes the stream, so a single empty
/// decode proves nothing. Only an unbroken streak does, and at roughly
/// one segment per second thirty of them is about half a minute of blindness:
/// long enough that no buffering hiccup explains it, short enough that little
/// motion is missed before the respawn.
pub(super) const BLIND_DECODER_STREAK: u32 = 30;

/// Sleep up to `total`, returning early once shutdown is requested, so a backoff
/// never holds the drain up. The analyzer body runs on a blocking thread and so
/// cannot select against the shutdown notify the async tasks use; polling the
/// same flag it already polls every tick is the equivalent.
pub(super) fn sleep_unless_shutdown(total: Duration, shutdown: &AtomicBool) {
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
pub(super) fn sleep_unless_shutdown_watching(
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
pub(super) struct ZeroFrameTripwire {
    streak: u32,
}

impl ZeroFrameTripwire {
    /// Record one decode's frame count. Returns `true` when the streak reaches
    /// [`BLIND_DECODER_STREAK`], which also resets it — a decoder still blind
    /// after its respawn trips again rather than going quiet.
    pub(super) fn observe(&mut self, frames: usize) -> bool {
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

    pub(super) fn reset(&mut self) {
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
pub(super) enum ForkWindow<'a> {
    Open(&'a AtomicBool),
    Closed,
}

impl ForkWindow<'_> {
    pub(super) fn is_open(self) -> bool {
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
pub(super) trait Respawnable {
    fn is_alive(&mut self) -> bool;
}

impl Respawnable for CropDecoder {
    fn is_alive(&mut self) -> bool {
        CropDecoder::is_alive(self)
    }
}

/// A decoder that outlives the batch which first needed it, together with the
/// one thing its current child has to be told before it is useful.
pub(super) struct LongLived<D> {
    pub(super) decoder: Option<D>,
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
    pub(super) primed_with: u64,
}

impl<D> LongLived<D> {
    /// Whether this child has had enough input to be past its probe, and so
    /// whether the next run may keep what it decodes.
    pub(super) fn primed(&self) -> bool {
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
pub(super) fn ensure_long_lived<D: Respawnable>(
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
pub(super) trait KeepsDecoder<D> {
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
pub(super) fn lend_for_batch<O, D>(
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
