//! Notices the storage volume moving out from under a running camon.
//!
//! An unmounted `data_dir` is the one storage fault that looks like health from
//! every other angle. The mountpoint is still a directory, so `create_dir_all`
//! succeeds, the write lands on whatever filesystem is behind it — usually the
//! root one — and reports success; the recording-silence watchdog is fed by
//! exactly those successful writes and stays quiet; the `min_free_bytes` guard
//! calls `statvfs` on the same path and so measures the wrong device. Nothing
//! is silent and nothing errors. The fault is *footage in the wrong place*, and
//! no watchdog built on the absence of writes can see it.
//!
//! So it is checked for directly, with two facts that a single `stat` yields
//! together:
//!
//! * **A marker file**, written into `data_dir` at startup. Its absence means
//!   the directory tree camon initialised is not the tree it is writing into
//!   now. This is what catches a *bind* mount going away, which need not change
//!   the device id at all — a bind mount shares the superblock, and so the
//!   `st_dev`, of whatever it is bound from.
//! * **The device id** the marker was written to. A marker that is still there
//!   but on a different filesystem means the volume was swapped, or something
//!   was mounted over the path. This is what catches a replacement that happens
//!   to carry a marker of its own, and it is the fact that says the free-space
//!   guard is now reading a different disk.
//!
//! Both are compared only against *this process's own* startup observation. No
//! attempt is made to decide whether `data_dir` "is a mount point" — the usual
//! test for that (its device differs from its parent's) calls every bind mount
//! unmounted and would warn forever on a perfectly good one. The baseline is
//! never written down anywhere either, so an operator who moves storage to a
//! different volume between runs simply gets a new baseline on the next start.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::locks::MutexExt;

/// Marker file name, in `data_dir` itself. Dotted so it stays out of the way,
/// and outside every directory the index and the orphan sweep walk (those work
/// in `{data_dir}/{camera}/{tier}/`), so nothing can mistake it for an event.
const MARKER_NAME: &str = ".camon-volume";

/// Contents of the marker, for whoever finds it and wonders. Camon never reads
/// it back — the file's *existence* and the device under it are the whole
/// signal, and checking the bytes too would cost an open and a read to prove
/// something the device id already proves.
const MARKER_TEXT: &[u8] = b"\
camon storage marker.

Written when camon starts and stat()ed once a minute after that. Its
disappearance, or its turning up on a different filesystem than the one it was
written to, is how camon notices that the storage volume under this directory
was unmounted, replaced, or mounted over while it was running.

Recreated on every start, so it is safe to delete while camon is stopped.
Deleting it while camon is running is reported as the volume going away.
";

/// How often the anchor is checked. The same cadence as the recording watchdog,
/// against a fault that is a persistent state rather than an event: a volume
/// does not unmount for the duration of one write and come back.
const POLL_INTERVAL: Duration = Duration::from_secs(60);

/// How long between repeats of a fault that is still there. Long, because the
/// message is unactionable more than once an hour and the log it lands in is
/// otherwise empty in a release build.
const REPEAT_INTERVAL: Duration = Duration::from_secs(3600);

/// What a check found that is worth telling the operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorReport {
    /// The marker is gone: the tree being written into is not the one that was
    /// initialised.
    Vanished,
    /// The marker is on a different filesystem than at startup.
    Moved { from: u64, to: u64 },
    /// The marker could not be looked at for some reason other than its being
    /// absent — a permission change, or an I/O error from the device itself.
    Unreadable(std::io::ErrorKind),
    /// A fault reported earlier is no longer there.
    Recovered,
}

/// The fault as it is remembered between checks, so a repeat can be told from a
/// change and a recovery from continued health.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    Vanished,
    Moved(u64),
    Unreadable(std::io::ErrorKind),
}

#[derive(Default)]
struct AnchorState {
    /// Device the marker was written to, or `None` while the anchor has not
    /// been armed yet. Set exactly once per process.
    baseline: Option<u64>,
    fault: Option<Fault>,
    last_report: Option<Instant>,
}

/// Startup observation of `data_dir`, and the periodic re-check against it.
pub struct StorageAnchor {
    data_dir: PathBuf,
    marker: PathBuf,
    state: Mutex<AnchorState>,
    /// Cached verdict for callers that need it on a path where a syscall would
    /// be wrong. True until a check proves otherwise, so an anchor that has not
    /// armed yet vetoes nothing: no evidence is not evidence of a fault.
    intact: AtomicBool,
}

impl StorageAnchor {
    /// Mark `data_dir` and record what it is on. A `data_dir` that does not
    /// exist yet is not a fault — the low-space guard creates it before the
    /// first write — so the anchor stays unarmed and tries again on each check
    /// until there is a directory to mark.
    pub fn new(data_dir: PathBuf) -> Self {
        let marker = data_dir.join(MARKER_NAME);
        let anchor = Self {
            data_dir,
            marker,
            state: Mutex::new(AnchorState::default()),
            intact: AtomicBool::new(true),
        };
        anchor.arm(&mut anchor.state.lock_recover());
        anchor
    }

    /// No check has found the storage volume moved or gone. Read on the write
    /// path, so it must stay a plain atomic load.
    pub fn is_intact(&self) -> bool {
        self.intact.load(Ordering::Relaxed)
    }

    /// One `stat` of the marker, folded against the startup baseline and the
    /// last thing reported.
    pub fn check(&self, now: Instant) -> Option<AnchorReport> {
        let mut state = self.state.lock_recover();
        if state.baseline.is_none() {
            self.arm(&mut state);
            return None;
        }
        let observed = device_of(&self.marker);
        self.evaluate(&mut state, observed, now)
    }

    /// The poll loop. Aborted at shutdown: it holds nothing that has to be
    /// flushed, and where the footage went is not news during a drain.
    pub async fn run(self: std::sync::Arc<Self>) {
        let mut interval = tokio::time::interval(POLL_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tracing::debug!(data_dir = %self.data_dir.display(), "storage anchor watching");
        loop {
            interval.tick().await;
            if let Some(report) = self.check(tokio::time::Instant::now().into_std()) {
                report.log(&self.data_dir);
            }
        }
    }

    fn arm(&self, state: &mut AnchorState) {
        // Not staged and not fsynced, unlike everything else camon writes: the
        // marker is authored fresh on every start and never read across one, so
        // a power cut losing it costs nothing the next start does not redo.
        if std::fs::write(&self.marker, MARKER_TEXT).is_err() {
            return;
        }
        match device_of(&self.marker) {
            Ok(dev) => {
                state.baseline = Some(dev);
                tracing::debug!(
                    data_dir = %self.data_dir.display(),
                    device = dev,
                    "storage anchor armed"
                );
            }
            Err(e) => tracing::debug!(error = %e, "storage anchor could not stat its own marker"),
        }
    }

    /// The decision, split from the syscall so the device-change branch is
    /// reachable from a test: staging a real remount is not.
    fn evaluate(
        &self,
        state: &mut AnchorState,
        observed: std::io::Result<u64>,
        now: Instant,
    ) -> Option<AnchorReport> {
        let baseline = state.baseline?;
        let fault = match observed {
            Ok(dev) if dev == baseline => None,
            Ok(dev) => Some(Fault::Moved(dev)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(Fault::Vanished),
            Err(e) => Some(Fault::Unreadable(e.kind())),
        };
        self.intact.store(fault.is_none(), Ordering::Relaxed);

        let changed = fault != state.fault;
        let was_faulty = state.fault.is_some();
        state.fault = fault;
        let report = match fault {
            None if was_faulty => Some(AnchorReport::Recovered),
            None => None,
            Some(f) => {
                let due = changed
                    || state
                        .last_report
                        .is_none_or(|at| now.saturating_duration_since(at) >= REPEAT_INTERVAL);
                due.then_some(match f {
                    Fault::Vanished => AnchorReport::Vanished,
                    Fault::Moved(to) => AnchorReport::Moved { from: baseline, to },
                    Fault::Unreadable(kind) => AnchorReport::Unreadable(kind),
                })
            }
        };
        if report.is_some() {
            state.last_report = Some(now);
        }
        report
    }
}

/// The filesystem the path is on, in one `stat`. Follows symlinks, because that
/// is what every write to `data_dir` does too: the question is where the bytes
/// land, not what the entry nominally is.
fn device_of(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).map(|m| m.dev())
}

impl AnchorReport {
    /// Loud enough to survive the release filter, which passes warn and up.
    /// Deliberately not fatal: refusing to record would turn footage in the
    /// wrong place into no footage at all, and footage in the wrong place can
    /// still be moved back afterwards. What camon stops doing is *deleting* —
    /// see the low-space guard.
    fn log(&self, data_dir: &Path) {
        let dir = data_dir.display();
        match *self {
            AnchorReport::Vanished => tracing::error!(
                data_dir = %dir,
                marker = MARKER_NAME,
                "the marker camon wrote into the storage directory at startup is gone: the \
                 volume holding it has been unmounted or replaced. Recording continues, but it \
                 is landing on whatever filesystem is behind this path now — footage written \
                 from here is not going into the archive, the free-space guard is measuring \
                 another device, and the disk it is filling is most likely the root one. \
                 Remount the storage volume and restart camon"
            ),
            AnchorReport::Moved { from, to } => tracing::error!(
                data_dir = %dir,
                device_at_startup = from,
                device_now = to,
                "the storage directory is on a different filesystem than it was at startup: \
                 something has been mounted over it, or the volume under it was swapped. \
                 Recording continues into the new one, but it is not the archive camon scanned \
                 at startup and it is not the volume the retention and free-space limits were \
                 sized for. Restart camon once storage is where it should be"
            ),
            AnchorReport::Unreadable(kind) => tracing::error!(
                data_dir = %dir,
                error = ?kind,
                "the storage directory cannot be examined; camon can no longer tell whether \
                 footage is reaching the volume it is supposed to"
            ),
            AnchorReport::Recovered => tracing::warn!(
                data_dir = %dir,
                "the storage directory is back on the filesystem camon started with. Anything \
                 written while it was away went somewhere else and is not in the archive; \
                 restart camon so the index matches what is actually stored"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn armed(dir: &Path) -> StorageAnchor {
        let anchor = StorageAnchor::new(dir.to_path_buf());
        assert!(
            anchor.state.lock_recover().baseline.is_some(),
            "anchor did not arm on an existing directory"
        );
        anchor
    }

    #[test]
    fn marks_the_storage_directory_and_stays_quiet_while_it_is_there() {
        let dir = tempfile::tempdir().unwrap();
        let anchor = armed(dir.path());

        assert!(dir.path().join(MARKER_NAME).is_file());
        let t0 = Instant::now();
        assert_eq!(anchor.check(t0), None);
        assert_eq!(anchor.check(t0 + REPEAT_INTERVAL * 3), None);
        assert!(anchor.is_intact());
    }

    /// The unmount case, as far as a test without root can stage it: the tree
    /// camon marked is not the tree it is looking at now.
    #[test]
    fn reports_a_storage_directory_whose_marker_has_gone() {
        let dir = tempfile::tempdir().unwrap();
        let anchor = armed(dir.path());
        let t0 = Instant::now();

        std::fs::remove_file(dir.path().join(MARKER_NAME)).unwrap();
        assert_eq!(anchor.check(t0), Some(AnchorReport::Vanished));
        assert!(!anchor.is_intact());
    }

    /// A fault that persists is repeated on the hour, not on every poll — the
    /// release log is otherwise empty and a minute-by-minute repeat would bury
    /// everything else in it.
    #[test]
    fn a_lasting_fault_repeats_hourly_rather_than_every_poll() {
        let dir = tempfile::tempdir().unwrap();
        let anchor = armed(dir.path());
        let t0 = Instant::now();

        std::fs::remove_file(dir.path().join(MARKER_NAME)).unwrap();
        assert_eq!(anchor.check(t0), Some(AnchorReport::Vanished));
        assert_eq!(anchor.check(t0 + POLL_INTERVAL), None);
        assert_eq!(anchor.check(t0 + REPEAT_INTERVAL - POLL_INTERVAL), None);
        assert_eq!(
            anchor.check(t0 + REPEAT_INTERVAL),
            Some(AnchorReport::Vanished)
        );
    }

    #[test]
    fn a_volume_that_comes_back_is_reported_once_and_clears_the_veto() {
        let dir = tempfile::tempdir().unwrap();
        let anchor = armed(dir.path());
        let t0 = Instant::now();

        let marker = dir.path().join(MARKER_NAME);
        std::fs::remove_file(&marker).unwrap();
        assert_eq!(anchor.check(t0), Some(AnchorReport::Vanished));

        std::fs::write(&marker, MARKER_TEXT).unwrap();
        assert_eq!(
            anchor.check(t0 + POLL_INTERVAL),
            Some(AnchorReport::Recovered)
        );
        assert!(anchor.is_intact());
        assert_eq!(anchor.check(t0 + POLL_INTERVAL * 2), None);
    }

    /// A change of device is the branch no test can stage — mounting anything
    /// needs root — so the comparison is exercised at the seam instead, with
    /// the `stat` result supplied.
    #[test]
    fn a_marker_on_a_different_filesystem_is_a_moved_volume() {
        let dir = tempfile::tempdir().unwrap();
        let anchor = armed(dir.path());
        let baseline = anchor.state.lock_recover().baseline.unwrap();
        let t0 = Instant::now();

        let mut state = anchor.state.lock_recover();
        assert_eq!(
            anchor.evaluate(&mut state, Ok(baseline ^ 1), t0),
            Some(AnchorReport::Moved {
                from: baseline,
                to: baseline ^ 1
            })
        );
        assert!(!anchor.is_intact());
        // Back on the original device: one recovery, then quiet.
        assert_eq!(
            anchor.evaluate(&mut state, Ok(baseline), t0 + POLL_INTERVAL),
            Some(AnchorReport::Recovered)
        );
        assert!(anchor.is_intact());
    }

    /// An unreadable directory is a third state: it is not proof the volume
    /// moved, but it is proof camon can no longer tell.
    #[test]
    fn a_stat_that_fails_for_another_reason_is_reported_as_its_own_fault() {
        let dir = tempfile::tempdir().unwrap();
        let anchor = armed(dir.path());
        let t0 = Instant::now();
        let mut state = anchor.state.lock_recover();

        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            anchor.evaluate(&mut state, Err(denied), t0),
            Some(AnchorReport::Unreadable(
                std::io::ErrorKind::PermissionDenied
            ))
        );
        assert!(!anchor.is_intact());

        // A different fault reports at once rather than waiting out the repeat
        // interval — it is different news.
        assert_eq!(
            anchor.evaluate(
                &mut state,
                Err(std::io::Error::from(std::io::ErrorKind::NotFound)),
                t0 + POLL_INTERVAL
            ),
            Some(AnchorReport::Vanished)
        );
    }

    /// First run: `data_dir` does not exist until the low-space guard creates
    /// it before the first write. That is not a fault, and it must not veto
    /// anything either — it arms on the next check instead.
    #[test]
    fn a_data_dir_that_does_not_exist_yet_arms_when_it_appears() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("not").join("created").join("yet");
        let anchor = StorageAnchor::new(data_dir.clone());

        assert!(anchor.state.lock_recover().baseline.is_none());
        assert!(anchor.is_intact(), "an unarmed anchor must veto nothing");
        let t0 = Instant::now();
        assert_eq!(anchor.check(t0), None);

        std::fs::create_dir_all(&data_dir).unwrap();
        assert_eq!(anchor.check(t0 + POLL_INTERVAL), None);
        assert!(anchor.state.lock_recover().baseline.is_some());
        assert!(data_dir.join(MARKER_NAME).is_file());
        assert_eq!(anchor.check(t0 + POLL_INTERVAL * 2), None);
    }

    /// An operator who moves storage to another volume between runs gets a new
    /// baseline, not a warning: the baseline is taken at startup and never
    /// written down, so a marker left by a previous run proves nothing and is
    /// simply overwritten.
    #[test]
    fn a_marker_left_by_an_earlier_run_is_not_evidence_of_anything() {
        let dir = tempfile::tempdir().unwrap();
        drop(armed(dir.path()));

        let next_run = armed(dir.path());
        assert_eq!(next_run.check(Instant::now()), None);
        assert!(next_run.is_intact());
    }

    /// A relative `data_dir` is permitted config. The anchor re-resolves the
    /// configured path on every check, exactly as the writes do, so it follows
    /// them wherever they go — no canonicalisation, which would pin an absolute
    /// path the writer never uses.
    #[test]
    fn a_relative_data_dir_is_marked_and_checked_like_any_other() {
        let data_dir = PathBuf::from(format!("anchor-relative-{}", std::process::id()));
        std::fs::create_dir_all(&data_dir).unwrap();
        let anchor = StorageAnchor::new(data_dir.clone());
        let armed = anchor.state.lock_recover().baseline.is_some();
        let quiet = anchor.check(Instant::now());
        let _ = std::fs::remove_dir_all(&data_dir);

        assert!(armed, "a relative data_dir must arm like an absolute one");
        assert_eq!(quiet, None);
    }
}
