//! Decoder lifecycle machinery: spawn retry/backoff policy and the long-lived
//! decoder slot kept across batches.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::retry::{jittered, RetrySchedule, Streak};

use super::POLL_INTERVAL;

/// How long to wait before trying a dead decoder again.
pub(super) const DECODER_RESTART_BACKOFF: Duration = Duration::from_secs(5);

/// Ceiling on the decoder-spawn backoff.
pub(super) const DECODER_SPAWN_BACKOFF_MAX: Duration = Duration::from_secs(60);

pub(super) const DECODER_SPAWN_SCHEDULE: RetrySchedule = RetrySchedule {
    start: DECODER_RESTART_BACKOFF,
    max: DECODER_SPAWN_BACKOFF_MAX,
};

/// The one policy for failing to spawn a decoder, shared by analyzer construction and decoder
/// replacement: they fail for identical reasons, so a missing ffmpeg must not log at two
/// different rates.
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
/// Three seconds of footage carries ffmpeg past the stream probe it swallows
/// its first input for; see [`MotionAnalyzer::prime_with`].
pub(super) const PRIMING_SEGMENTS: u64 = 3;

/// Consecutive zero-frame decodes tolerated before the decoder is declared blind.
pub(super) const BLIND_DECODER_STREAK: u32 = 30;

/// Sleep up to `total`, returning early once shutdown is requested, so a backoff never holds
/// the drain up.
pub(super) fn sleep_unless_shutdown(total: Duration, shutdown: &AtomicBool) {
    sleep_unless_shutdown_watching(total, shutdown, || {});
}

/// The same sleep, with something to do each time it surfaces.
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

/// Counts consecutive zero-frame decodes so an ffmpeg that consumes input but emits nothing is
/// caught.
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

/// A decoder that outlives the batch which first needed it, together with the
/// one thing its current child has to be told before it is useful.
pub(super) struct LongLived<D> {
    pub(super) decoder: Option<D>,
    /// How many segments the child in `decoder` has been fed to carry it past ffmpeg's stream
    /// probe.
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

/// Make sure `slot` holds a decoder whose child is alive, forking one only when it does not,
/// and reporting whether this batch has one to decode with.
pub(super) fn ensure_long_lived<D>(
    slot: &mut LongLived<D>,
    stop: Option<&AtomicBool>,
    camera_id: &str,
    is_alive: impl FnOnce(&mut D) -> bool,
    spawn: impl FnOnce() -> Result<D, std::io::Error>,
) -> bool {
    if slot.decoder.as_mut().is_some_and(is_alive) {
        return true;
    }
    if !stop.is_some_and(|stop| !stop.load(Ordering::Relaxed)) {
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
            // Dropping the dead one kills whatever is left of it; the next
            // batch tries the fork again.
            *slot = LongLived::default();
            tracing::error!(camera = %camera_id, error = %e, "failed to create crop decoder");
            false
        }
    }
}
