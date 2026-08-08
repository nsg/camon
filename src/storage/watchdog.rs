//! Watches for a camera that is recording nothing.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::locks::MutexExt;

/// How long a camera in event mode may record nothing before it is reported.
const EVENT_SILENCE_LIMIT: Duration = Duration::from_secs(24 * 3600);

/// Consecutive chunks a continuous recorder may fail to produce before it is reported.
const CONTINUOUS_SILENT_CHUNKS: u32 = 10;

/// Floor under the continuous limit, so a short chunk cap does not turn a camera's ordinary
/// reconnect into a warning: reconnect backoff is capped at a minute, and five of those is a
/// camera that is properly down rather than blinking.
const MIN_CONTINUOUS_SILENCE_LIMIT: Duration = Duration::from_secs(300);

/// How often the watchdog compares each camera against its limit. Coarse
/// against limits measured in tens of minutes and up.
pub const POLL_INTERVAL: Duration = Duration::from_secs(60);

/// How long a camera had already been silent when camon started, from the end of its newest
/// stored event.
pub fn silence_before_startup(newest_event_end_ns: Option<u64>, now_ns: u64) -> Duration {
    match newest_event_end_ns {
        // Saturating both ways: an event stamped in the future (a camera clock
        // ahead of ours) is silence of zero, not a panic.
        Some(end) => Duration::from_nanos(now_ns.saturating_sub(end)),
        None => Duration::ZERO,
    }
}

/// How a camera is meant to produce footage, which is what decides how long its
/// silence is allowed to last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingMode {
    /// Events are written only when the analyzer closes a motion run, so quiet
    /// is the normal state of a camera watching an empty scene.
    Event,
    /// A chunk is rolled every `chunk`, whatever the scene does.
    Continuous { chunk: Duration },
}

impl RecordingMode {
    /// Clamped at both ends.
    fn silence_limit(self) -> Duration {
        match self {
            RecordingMode::Event => EVENT_SILENCE_LIMIT,
            RecordingMode::Continuous { chunk } => chunk
                .saturating_mul(CONTINUOUS_SILENT_CHUNKS)
                .clamp(MIN_CONTINUOUS_SILENCE_LIMIT, EVENT_SILENCE_LIMIT),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            RecordingMode::Event => "event",
            RecordingMode::Continuous { .. } => "continuous",
        }
    }
}

struct CameraState {
    mode: RecordingMode,
    /// When footage last reached storage, or registration until any ever has.
    last_write: Instant,
    /// Silence that predates this process, from [`silence_before_startup`].
    carried: Duration,
    /// When the silence was last reported, which is all that a report changes.
    /// Repeats are due one full limit apart, not once per poll.
    last_report: Option<Instant>,
    events: u64,
}

/// One camera's silence, at the moment it passed its limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SilenceReport {
    pub camera_id: String,
    pub mode: RecordingMode,
    /// The whole silence, measured from the last write (or from startup), not
    /// from the previous report.
    pub silent_for: Duration,
    /// Events this camera has written since the process started.
    pub events: u64,
    /// Whether this camera has ever recorded — in this process, or in an
    /// earlier one according to what is stored. Separates a camera that stopped
    /// working from one that never has.
    pub has_recorded: bool,
}

/// Per-camera record of when footage last reached warm storage.
#[derive(Default)]
pub struct RecordingWatchdog {
    cameras: Mutex<HashMap<String, CameraState>>,
}

impl RecordingWatchdog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start watching a camera, `already_silent_for` carrying whatever silence its stored
    /// footage says predates this process (see [`silence_before_startup`]).
    pub fn register(
        &self,
        camera_id: &str,
        mode: RecordingMode,
        now: Instant,
        already_silent_for: Duration,
    ) {
        self.cameras.lock_recover().insert(
            camera_id.to_string(),
            CameraState {
                mode,
                last_write: now,
                carried: already_silent_for,
                last_report: None,
                events: 0,
            },
        );
    }

    /// Note that an event reached storage. Unregistered cameras are ignored:
    /// the writer outlives nothing, but a camera that was never registered is
    /// simply not watched.
    pub fn record(&self, camera_id: &str, now: Instant) {
        if let Some(state) = self.cameras.lock_recover().get_mut(camera_id) {
            state.last_write = now;
            state.carried = Duration::ZERO;
            // A camera that recorded and then stops again is a fresh silence,
            // due one full limit from this write rather than from whenever it
            // was last complained about.
            state.last_report = None;
            state.events += 1;
        }
    }

    pub fn check(&self, now: Instant) -> Vec<SilenceReport> {
        let mut reports = Vec::new();
        for (camera_id, state) in self.cameras.lock_recover().iter_mut() {
            let limit = state.mode.silence_limit();
            let silent_for = now
                .saturating_duration_since(state.last_write)
                .saturating_add(state.carried);
            if silent_for < limit {
                continue;
            }
            let repeat_due = match state.last_report {
                Some(reported) => now.saturating_duration_since(reported) >= limit,
                None => true,
            };
            if !repeat_due {
                continue;
            }
            state.last_report = Some(now);
            reports.push(SilenceReport {
                camera_id: camera_id.clone(),
                mode: state.mode,
                silent_for,
                events: state.events,
                has_recorded: state.events > 0 || !state.carried.is_zero(),
            });
        }
        reports
    }

    /// Aborted at shutdown rather than joined — it holds nothing that needs flushing, and a
    /// camera's silence is not news during a drain.
    pub async fn run(self: std::sync::Arc<Self>) {
        let mut interval = tokio::time::interval(POLL_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tracing::debug!("recording watchdog started");
        loop {
            interval.tick().await;
            for report in self.check(tokio::time::Instant::now().into_std()) {
                report.log();
            }
        }
    }
}

impl SilenceReport {
    fn log(&self) {
        tracing::warn!(
            camera = %self.camera_id,
            mode = self.mode.as_str(),
            silent_for_mins = self.silent_for.as_secs() / 60,
            events_this_run = self.events,
            "{}",
            self.explanation()
        );
    }

    fn explanation(&self) -> String {
        let ago = humanized(self.silent_for);
        let ever = if self.has_recorded {
            format!("has recorded nothing in the {ago} since its last event")
        } else {
            format!("has recorded nothing since camon started {ago} ago")
        };
        match self.mode {
            RecordingMode::Continuous { chunk } => format!(
                "{ever}. Continuous recording rolls a chunk every {}s whatever the scene does, \
                 so this is not a quiet camera: either the stream is not reaching camon or its \
                 writes to warm storage are failing",
                chunk.as_secs()
            ),
            RecordingMode::Event => format!(
                "{ever}. A scene with nothing happening in it looks exactly like this and is \
                 nothing to worry about — but so do an ignore mask painted over the whole frame, \
                 a sensitivity slider left at its least sensitive, a stream that is not reaching \
                 camon, and warm writes that are failing"
            ),
        }
    }
}

/// Round the silence to whatever unit reads without arithmetic. The number is the point of the
/// message, so a continuous camera's twenty minutes and an event camera's three days both have
/// to land at a glance.
fn humanized(d: Duration) -> String {
    const MINUTE: u64 = 60;
    const HOUR: u64 = 60 * MINUTE;
    const DAY: u64 = 24 * HOUR;
    let secs = d.as_secs();
    if secs < 2 * MINUTE {
        count_of(secs, "second")
    } else if secs < 2 * HOUR {
        count_of(secs / MINUTE, "minute")
    } else if secs < 2 * DAY {
        count_of(secs / HOUR, "hour")
    } else {
        count_of(secs / DAY, "day")
    }
}

fn count_of(n: u64, unit: &str) -> String {
    match n {
        1 => format!("1 {unit}"),
        n => format!("{n} {unit}s"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: Duration = Duration::from_secs(120);
    const CONTINUOUS: RecordingMode = RecordingMode::Continuous { chunk: CHUNK };

    fn watchdog(mode: RecordingMode, t0: Instant) -> RecordingWatchdog {
        let watchdog = RecordingWatchdog::new();
        watchdog.register("cam", mode, t0, Duration::ZERO);
        watchdog
    }

    #[test]
    fn reports_a_camera_that_has_written_nothing_since_startup() {
        let t0 = Instant::now();
        let watchdog = watchdog(RecordingMode::Event, t0);

        assert!(watchdog
            .check(t0 + EVENT_SILENCE_LIMIT - POLL_INTERVAL)
            .is_empty());

        let reports = watchdog.check(t0 + EVENT_SILENCE_LIMIT);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].camera_id, "cam");
        assert_eq!(reports[0].events, 0);
        assert!(!reports[0].has_recorded);
    }

    const SEC_NS: u64 = 1_000_000_000;

    #[test]
    fn silence_before_startup_is_the_age_of_the_newest_stored_event() {
        let now_ns = 10 * 24 * 3600 * SEC_NS;
        assert_eq!(
            silence_before_startup(Some(now_ns - 3 * 3600 * SEC_NS), now_ns),
            Duration::from_secs(3 * 3600)
        );
    }

    #[test]
    fn a_camera_with_nothing_stored_starts_from_now() {
        assert_eq!(silence_before_startup(None, 10 * SEC_NS), Duration::ZERO);
    }

    #[test]
    fn an_event_stamped_in_the_future_is_no_silence_at_all() {
        assert_eq!(
            silence_before_startup(Some(20 * SEC_NS), 10 * SEC_NS),
            Duration::ZERO
        );
    }

    #[test]
    fn silence_survives_a_restart() {
        let t0 = Instant::now();
        let watchdog = RecordingWatchdog::new();
        let before = Duration::from_secs(2 * 24 * 3600);
        watchdog.register("cam", RecordingMode::Event, t0, before);

        let reports = watchdog.check(t0 + POLL_INTERVAL);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].silent_for, before + POLL_INTERVAL);
        assert!(
            reports[0].has_recorded,
            "stored footage proves it worked once"
        );
        assert_eq!(reports[0].events, 0);
    }

    #[test]
    fn a_write_clears_the_silence_carried_in_from_before() {
        let t0 = Instant::now();
        let watchdog = RecordingWatchdog::new();
        watchdog.register(
            "cam",
            RecordingMode::Event,
            t0,
            Duration::from_secs(20 * 3600),
        );

        watchdog.record("cam", t0 + POLL_INTERVAL);
        assert!(watchdog
            .check(t0 + POLL_INTERVAL + EVENT_SILENCE_LIMIT / 2)
            .is_empty());
    }

    #[test]
    fn stays_quiet_while_events_keep_arriving() {
        let t0 = Instant::now();
        let watchdog = watchdog(RecordingMode::Event, t0);

        for step in 1..=56u32 {
            let now = t0 + Duration::from_secs(3 * 3600) * step;
            watchdog.record("cam", now);
            assert!(watchdog.check(now).is_empty(), "reported at step {step}");
        }
    }

    #[test]
    fn reports_a_camera_that_stops_recording_after_working() {
        let t0 = Instant::now();
        let watchdog = watchdog(RecordingMode::Event, t0);
        watchdog.record("cam", t0 + Duration::from_secs(60));

        let last_event = t0 + Duration::from_secs(60);
        assert!(watchdog
            .check(last_event + EVENT_SILENCE_LIMIT / 2)
            .is_empty());

        let reports = watchdog.check(last_event + EVENT_SILENCE_LIMIT);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].events, 1);
    }

    #[test]
    fn repeats_once_per_limit_while_the_silence_lasts() {
        let t0 = Instant::now();
        let watchdog = watchdog(RecordingMode::Event, t0);

        assert_eq!(watchdog.check(t0 + EVENT_SILENCE_LIMIT).len(), 1);
        assert!(watchdog
            .check(t0 + EVENT_SILENCE_LIMIT + POLL_INTERVAL)
            .is_empty());
        assert_eq!(watchdog.check(t0 + EVENT_SILENCE_LIMIT * 2).len(), 1);
    }

    #[test]
    fn a_lasting_silence_is_reported_as_it_accumulates() {
        let t0 = Instant::now();
        let watchdog = watchdog(CONTINUOUS, t0);
        let limit = CONTINUOUS.silence_limit();

        for repeat in 1..=5u32 {
            assert!(watchdog
                .check(t0 + limit * repeat - POLL_INTERVAL)
                .is_empty());

            let reports = watchdog.check(t0 + limit * repeat);
            assert_eq!(reports.len(), 1, "missing repeat {repeat}");
            assert_eq!(
                reports[0].silent_for,
                limit * repeat,
                "repeat {repeat} reported the gap since the last warning, not the silence"
            );
        }
    }

    #[test]
    fn a_camera_that_records_again_restarts_the_clock() {
        let t0 = Instant::now();
        let watchdog = watchdog(RecordingMode::Event, t0);
        assert_eq!(watchdog.check(t0 + EVENT_SILENCE_LIMIT).len(), 1);

        let resumed = t0 + EVENT_SILENCE_LIMIT + Duration::from_secs(60);
        watchdog.record("cam", resumed);
        assert!(watchdog.check(resumed + EVENT_SILENCE_LIMIT / 2).is_empty());

        let reports = watchdog.check(resumed + EVENT_SILENCE_LIMIT);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].silent_for, EVENT_SILENCE_LIMIT);
    }

    #[test]
    fn continuous_mode_reports_after_ten_missed_chunks() {
        let t0 = Instant::now();
        let watchdog = watchdog(CONTINUOUS, t0);

        assert!(watchdog.check(t0 + CHUNK * 9).is_empty());
        let reports = watchdog.check(t0 + CHUNK * 10);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].mode, CONTINUOUS);
    }

    #[test]
    fn continuous_mode_is_far_less_patient_than_event_mode() {
        assert!(CONTINUOUS.silence_limit() < EVENT_SILENCE_LIMIT / 10);
        assert!(RecordingMode::Event.silence_limit() > Duration::from_secs(18 * 3600));
    }

    #[test]
    fn a_short_chunk_cap_cannot_report_faster_than_a_reconnect() {
        let mode = RecordingMode::Continuous {
            chunk: Duration::from_secs(5),
        };
        assert_eq!(mode.silence_limit(), MIN_CONTINUOUS_SILENCE_LIMIT);
    }

    #[test]
    fn a_long_chunk_cap_is_capped_at_the_event_limit() {
        for chunk in [
            Duration::from_secs(3 * 3600),
            Duration::from_secs(u64::MAX / 2),
            Duration::MAX,
        ] {
            let mode = RecordingMode::Continuous { chunk };
            assert_eq!(mode.silence_limit(), EVENT_SILENCE_LIMIT);
        }
    }

    #[test]
    fn silence_is_rounded_to_a_unit_that_reads() {
        assert_eq!(humanized(Duration::from_secs(20 * 60)), "20 minutes");
        assert_eq!(humanized(Duration::from_secs(3 * 3600)), "3 hours");
        assert_eq!(humanized(Duration::from_secs(7 * 24 * 3600)), "7 days");
    }

    #[test]
    fn rounding_is_honest_and_grammatical() {
        for secs in (0..(4 * 24 * 3600)).step_by(37) {
            let text = humanized(Duration::from_secs(secs));
            let singular = text.starts_with("1 ");
            assert_eq!(
                singular,
                !text.ends_with('s'),
                "{secs}s rendered as {text:?}"
            );
            assert!(
                !text.starts_with("1 hour") && !text.starts_with("1 day"),
                "{secs}s rounded down to {text:?}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn the_poll_loop_reports_a_silent_camera() {
        let watchdog = std::sync::Arc::new(RecordingWatchdog::new());
        watchdog.register(
            "cam",
            CONTINUOUS,
            tokio::time::Instant::now().into_std(),
            Duration::ZERO,
        );
        let handle = tokio::spawn(std::sync::Arc::clone(&watchdog).run());

        tokio::time::sleep(CONTINUOUS.silence_limit() / 2).await;
        assert_eq!(
            watchdog.cameras.lock_recover()["cam"].last_report,
            None,
            "reported before the limit"
        );

        tokio::time::sleep(CONTINUOUS.silence_limit()).await;
        assert!(
            watchdog.cameras.lock_recover()["cam"].last_report.is_some(),
            "the loop never reported a camera past its limit"
        );
        handle.abort();
    }

    #[test]
    fn each_camera_is_tracked_on_its_own() {
        let t0 = Instant::now();
        let watchdog = RecordingWatchdog::new();
        watchdog.register("busy", RecordingMode::Event, t0, Duration::ZERO);
        watchdog.register("silent", RecordingMode::Event, t0, Duration::ZERO);

        watchdog.record("busy", t0 + EVENT_SILENCE_LIMIT / 2);
        watchdog.record("stranger", t0 + EVENT_SILENCE_LIMIT / 2);

        let reports = watchdog.check(t0 + EVENT_SILENCE_LIMIT);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].camera_id, "silent");
    }
}
