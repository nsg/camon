//! Process orchestration: startup, the camera/analyzer/writer task graph, and
//! the graceful drain both shutdown paths end in.
//!
//! The drain is phased — producers, then consumers, then writers — and
//! [`crate::shutdown`] is where that contract and its bounds are written down.
//! What lives here is the sequencing itself.
//!
//! Lives in the library rather than in `main.rs` so it is compiled and tested
//! once. The binary is the shim around it, and keeps what only a binary has:
//! argument dispatch and the self-updater.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tracing_subscriber::EnvFilter;

use crate::analytics::{
    self, detect_queue, AnalyzerContext, DetectQueueSender, DetectionWorker, OllamaClient,
};
use crate::api::{self, AppState};
use crate::buffer::warm::{run_continuous_recorder, RetentionTask, WarmWriter, WriterMessage};
use crate::buffer::HotBuffer;
use crate::camera::{self, FfmpegPipeline};
use crate::config::{self, Config};
use crate::locks::LockExt;
use crate::mqtt::{self, BridgeContext, MqttEvent, MQTT_EVENT_CAPACITY};
use crate::retry::{apply_jitter, jitter_source};
use crate::storage::{
    self, DetectionDebugStore, DetectionStore, EventRegistry, LocalDiskBackend, MotionStore,
    RecordingMode, RecordingWatchdog, StathostBackend, WarmStorageBackend,
};

/// Parsed command-line arguments. Kept deliberately tiny (no clap): camon takes
/// at most `--config <path>` and any number of `--set <dotted.path>=<value>`
/// overrides.
struct CliArgs {
    /// Explicit config path from `--config`; `None` falls back to `config.toml`.
    config: Option<String>,
    /// `--set` overrides, in the order given (later wins on a repeated key).
    overrides: Vec<config::Override>,
}

/// Hand-rolled `--config` / `--set` parser (both `--flag value` and
/// `--flag=value` forms). Unknown arguments are ignored here so subcommands and
/// future flags keep working; a malformed `--set` is a hard startup error.
fn parse_cli_args() -> CliArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = None;
    let mut overrides = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if let Some(path) = arg.strip_prefix("--config=") {
            config = Some(path.to_string());
        } else if arg == "--config" {
            match args.get(i + 1) {
                Some(path) => {
                    config = Some(path.clone());
                    i += 1;
                }
                None => {
                    eprintln!("error: --config requires a path argument");
                    std::process::exit(1);
                }
            }
        } else if let Some(spec) = arg.strip_prefix("--set=") {
            overrides.push(parse_override_or_exit(spec));
        } else if arg == "--set" {
            match args.get(i + 1) {
                Some(spec) => {
                    overrides.push(parse_override_or_exit(spec));
                    i += 1;
                }
                None => {
                    eprintln!("error: --set requires a <dotted.path>=<value> argument");
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }
    CliArgs { config, overrides }
}

fn parse_override_or_exit(spec: &str) -> config::Override {
    match config::Override::parse(spec) {
        Ok(ov) => ov,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// The one way anything inside the process asks for shutdown: the flag every
/// worker already polls, plus a wakeup for the main task so the drain starts at
/// once instead of on the next poll. Signals raise the same flag.
#[derive(Clone)]
struct ShutdownSignal {
    flag: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    /// Broadcast wakeup for workers that would otherwise only notice the flag
    /// on their next poll — a camera task parked in its reconnect backoff can
    /// be a minute away from one. Separate from `wake`, which carries a single
    /// permit meant for the main task.
    drain: Arc<tokio::sync::Notify>,
    /// Set once an update has replaced the binary on disk. Tracked separately
    /// from the flag because it can become true at any point up to the end of
    /// the drain — including after a signal already started one, from a check
    /// that was in flight — and the restart watchdog keys off it.
    update_installed: Arc<AtomicBool>,
}

impl ShutdownSignal {
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            wake: Arc::new(tokio::sync::Notify::new()),
            drain: Arc::new(tokio::sync::Notify::new()),
            update_installed: Arc::new(AtomicBool::new(false)),
        }
    }

    fn requested(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Raise the flag and wake every worker waiting on it. `notify_waiters`
    /// reaches all of them at once but leaves no permit behind for one that
    /// arrives afterwards, hence the flag re-check in `sleep_or_shutdown`.
    ///
    /// It does not wake the main task parked in `wait_for_shutdown`, so it is
    /// only for callers that reach that themselves — anyone else has to notify
    /// `wake` too, as `request_restart` does, or the drain never starts.
    fn request(&self) {
        self.flag.store(true, Ordering::Relaxed);
        self.drain.notify_waiters();
    }

    /// `notify_one` leaves a permit behind when nobody is waiting yet, so the
    /// wakeup cannot be lost to a race with the main task reaching it.
    fn request_restart(&self) {
        self.update_installed.store(true, Ordering::Relaxed);
        self.request();
        self.wake.notify_one();
    }

    /// Sleep for `duration`, cut short the moment shutdown is requested, so a
    /// pending delay never holds the drain up.
    ///
    /// The `Notified` is created before the flag is read, and tokio guarantees
    /// it receives every `notify_waiters` that happens after its creation, even
    /// one landing before it is first polled. A request is therefore either
    /// already visible in the flag or still to come and caught by the notify —
    /// there is no ordering in which this sleeps the full duration.
    async fn sleep_or_shutdown(&self, duration: std::time::Duration) {
        let notified = self.drain.notified();
        if self.requested() {
            return;
        }
        tokio::select! {
            _ = tokio::time::sleep(duration) => {}
            _ = notified => {}
        }
    }
}

const UPDATE_CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(12 * 60 * 60);

/// An update installed while camon is running must not cut the recording short:
/// it asks for the same graceful shutdown a signal does, so the analyzers,
/// continuous recorders and warm writers drain before the process goes away.
/// The loop is one-shot in that sense — after an install there is nothing left
/// to check, the process is on its way out.
async fn run_update_check_loop_with<F, Fut, E>(
    interval_period: std::time::Duration,
    shutdown: ShutdownSignal,
    check: F,
) where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<bool, E>>,
    E: std::fmt::Display,
{
    let mut interval = tokio::time::interval(interval_period);
    interval.tick().await; // skip immediate tick
    loop {
        interval.tick().await;
        // Never start an install into a process that is already shutting down.
        if shutdown.requested() {
            return;
        }
        match check().await {
            Ok(true) => {
                tracing::info!("update applied, shutting down for restart");
                shutdown.request_restart();
                return;
            }
            Ok(false) => {}
            Err(e) => {
                tracing::warn!(error = %e, "periodic update check failed");
            }
        }
    }
}

/// The self-updater lives in the binary, so it reaches this as a plain
/// argument: the library orchestrates the check (and what an installed update
/// does to the running process) without depending on the updater itself.
async fn check_for_updates<F, Fut, E>(config: &Config, shutdown: &ShutdownSignal, check: F)
where
    F: Fn() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<bool, E>> + Send + 'static,
    E: std::fmt::Display + 'static,
{
    if !config.update.enabled {
        return;
    }
    match check().await {
        // Nothing is recording yet at this point in startup, so there is
        // nothing to drain and exiting inline is safe.
        Ok(true) => {
            tracing::info!("update applied, exiting for restart");
            std::process::exit(0);
        }
        Ok(false) => {}
        Err(e) => {
            tracing::warn!(error = %e, "update check failed, continuing startup");
        }
    }
    tokio::spawn(run_update_check_loop_with(
        UPDATE_CHECK_INTERVAL,
        shutdown.clone(),
        check,
    ));
}

fn log_object_detection_config(config: &Config) -> bool {
    if !config.analytics.enabled || !config.analytics.object_detection.enabled {
        return false;
    }
    let od = &config.analytics.object_detection;
    tracing::info!(
        url = %od.ollama.url,
        model = %od.ollama.model,
        "object detection configured (ollama)"
    );
    if let Some(ref fb) = od.ollama.fallback {
        tracing::info!(
            url = %fb.url,
            model = %fb.model,
            "ollama fallback server configured"
        );
    }
    true
}

/// The warm backend and whether its index describes the store.
///
/// The second half exists for the retention task: a backend whose scan failed
/// is refusing to prune or enforce its budget until a sweep rebuilds the index,
/// and the sweep it is waiting for is the one this schedules — so that first
/// sweep comes early instead of an hour later. `true` when there is no warm
/// storage at all, which has nothing to heal.
struct Storage {
    backend: Option<Arc<dyn WarmStorageBackend>>,
    scanned: bool,
}

async fn init_storage(config: &Config, camera_ids: &[String]) -> Storage {
    if !config.storage.enabled {
        return Storage {
            backend: None,
            scanned: true,
        };
    }

    // A configured (and enabled) [storage.stathost] section switches the warm
    // backend from local disk to the remote host. Local disk is the default.
    if let Some(stathost) = config.storage.stathost.as_ref().filter(|s| s.enabled) {
        tracing::info!(
            url = %stathost.url,
            bucket = %stathost.bucket,
            "using remote stathost warm-storage backend"
        );
        let backend = StathostBackend::new(stathost, camera_ids);
        backend.recover_orphans();
        // A store that cannot be listed does not stop camon: the cameras record
        // and upload either way, and holding startup for a host that may be
        // down for hours would cost footage that nothing else would have
        // recorded. What it does cost is retention, and that is worth a line of
        // its own — the backend refuses to prune or evict until a later scan
        // succeeds, saying so on every retention sweep and, for the budget it
        // is asked about before every write, on a widening schedule.
        let scanned = match backend.scan().await {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "starting on a warm index that was never rebuilt from stathost: retention, \
                     the storage budget and the orphan sweep stay paused, and uploads are \
                     unbounded, until a retention sweep lists the store"
                );
                false
            }
        };
        return Storage {
            backend: Some(Arc::new(backend)),
            scanned,
        };
    }

    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    let backend = LocalDiskBackend::new(data_dir, camera_ids);
    // Salvage any event files orphaned mid-write by a crash or power cut
    // BEFORE the scan, so recovered events are indexed like any other.
    backend.recover_orphans();
    // Infallible for local disk — see its `scan` — but handled rather than
    // discarded, so that a backend which grows a failure mode here is not able
    // to acquire a silent one. Nothing here gates on the scan the way the
    // remote backend does: a directory that could not be read is one directory,
    // and retention sweeps what it did read.
    if let Err(e) = backend.scan().await {
        tracing::warn!(error = %e, "warm index scan failed; some events may not be indexed");
    }
    Storage {
        backend: Some(Arc::new(backend)),
        scanned: true,
    }
}

fn init_motion_settings(
    config: &Config,
    camera_ids: &[String],
) -> Option<analytics::MotionSettingsStore> {
    if !config.analytics.enabled {
        return None;
    }
    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    Some(analytics::MotionSettingsStore::new(
        camera_ids,
        &data_dir,
        config.analytics.motion.var_threshold,
        config.analytics.motion.min_contour_area,
    ))
}

/// The classes the MQTT bridge publishes occupancy entities for: read off the
/// detector itself, so an entity can never exist for a class nothing looks for
/// or the other way round. No client means no detections at all — object
/// detection is off, or its client could not be built — and so no entities.
fn mqtt_object_classes(client: Option<&OllamaClient>) -> Vec<String> {
    client
        .map(|c| c.allowed_classes().to_vec())
        .unwrap_or_default()
}

/// Where the bridge remembers the entity set it announced to Home Assistant.
/// Beside the per-camera state in the data dir, and written even when warm
/// storage is off: the file is a few hundred bytes and the entities exist
/// either way.
fn mqtt_entities_path(config: &Config) -> std::path::PathBuf {
    std::path::PathBuf::from(&config.storage.data_dir).join("mqtt_entities.json")
}

fn create_ollama_client(config: &Config) -> Option<OllamaClient> {
    let od = &config.analytics.object_detection;
    let fallback = od
        .ollama
        .fallback
        .as_ref()
        .map(|fb| (fb.url.as_str(), fb.model.as_str()));
    match OllamaClient::new(
        &od.ollama.url,
        &od.ollama.model,
        od.ollama.timeout_secs,
        od.confidence_threshold,
        od.classes.clone(),
        fallback,
    ) {
        Ok(client) => Some(client),
        Err(e) => {
            tracing::error!(error = %e, "failed to create ollama client, object detection disabled");
            None
        }
    }
}

/// Small bounded queue between each analyzer and its warm writer. Events are
/// rare and written quickly; the analyzer blocks briefly if the writer falls
/// behind rather than ever dropping an event.
const EVENT_CHANNEL_CAPACITY: usize = 8;

struct CameraHandles {
    /// The producers, with the buffer each one fills. Phase 1 of the drain
    /// joins these first and then publishes each buffer's watermark, which is
    /// why the buffer travels with the handle rather than only in
    /// `buffers_map`.
    pipeline_handles: Vec<(String, tokio::task::JoinHandle<()>, Arc<RwLock<HotBuffer>>)>,
    /// Every worker below is per-camera and keeps its camera's id beside it, so
    /// that a drain which has to abandon one can say which recording to
    /// distrust rather than only which kind of task stopped answering.
    analyzer_handles: Vec<(String, tokio::task::JoinHandle<()>)>,
    /// Per-camera continuous-recording drivers (storage on + analytics off).
    /// Empty in event mode. Drained in phase 2, before the writers' senders
    /// drop, so their final chunk is still accepted.
    continuous_handles: Vec<(String, tokio::task::JoinHandle<()>)>,
    warm_handles: Vec<(String, tokio::task::JoinHandle<()>)>,
    /// Kept alive so the writer channels stay open for as long as the process
    /// runs; dropped during shutdown to let the writers drain and exit.
    event_senders: Vec<tokio::sync::mpsc::Sender<WriterMessage>>,
    /// The same senders keyed by camera, for the detection worker's post-hoc
    /// event upgrades.
    event_sender_map: HashMap<String, tokio::sync::mpsc::Sender<WriterMessage>>,
    buffers_map: HashMap<String, Arc<RwLock<HotBuffer>>>,
}

struct SpawnContext<'a> {
    config: &'a Config,
    motion_store: &'a MotionStore,
    detection_store: &'a DetectionStore,
    storage: &'a Option<Arc<dyn WarmStorageBackend>>,
    motion_settings: &'a Option<analytics::MotionSettingsStore>,
    /// Crop-job queue into the global detection worker; `None` when object
    /// detection is off.
    detect_tx: &'a Option<DetectQueueSender>,
    event_registry: &'a Option<EventRegistry>,
    /// Motion lifecycle events for the Home Assistant bridge; `None` when
    /// `[mqtt].enabled` is false.
    mqtt_tx: &'a Option<tokio::sync::mpsc::Sender<MqttEvent>>,
    /// Notices a camera that is recording nothing, whatever the reason.
    recording_watchdog: &'a Arc<RecordingWatchdog>,
    shutdown: &'a ShutdownSignal,
}

/// Say what this process is going to do with its footage, once, at startup.
///
/// The two states where nothing is written are the reason this is not just an
/// info line: neither can be inferred later from an empty archive, and the
/// recording watchdog deliberately stays quiet about both, so this is the only
/// place they are ever said. Analytics without storage warns — events are
/// detected, published to MQTT, and then discarded, which is a real setup for a
/// motion sensor but an expensive mistake if recording was the point. Neither
/// disabled is only an info line: nothing is being computed and thrown away,
/// camon is a live-view proxy, and the operator asked for exactly that.
fn log_recording_mode(config: &Config) {
    match (config.storage.enabled, config.analytics.enabled) {
        (true, true) => {
            tracing::info!("event recording mode: motion and object events saved to disk")
        }
        (true, false) => tracing::info!(
            retention_days = config.storage.continuous_retention_days,
            "continuous recording mode (analytics disabled): every segment saved to \
             continuous/, roughly 43 GB/day/camera at 4 Mbps"
        ),
        (false, true) => tracing::warn!(
            "[storage] enabled = false: motion is detected and published, but no event is ever \
             written and nothing is kept beyond the in-memory buffer. Set [storage] enabled = \
             true if this camon is meant to be recording"
        ),
        (false, false) => tracing::info!(
            "live-view only: recording and analytics are disabled, so nothing is written to \
             disk and no camera is watched for silence"
        ),
    }
}

/// What a camera is expected to produce, which is what makes its silence
/// suspicious or not — or `None` when nothing is expected of it.
///
/// With storage off no write can ever succeed, so a silence timer is guaranteed
/// to fire and says nothing the configuration did not already say. Those states
/// are reported once at startup by [`log_recording_mode`] and then left alone:
/// a daily warning about a camera that was never asked to record only teaches
/// the operator to ignore the ones about cameras that were.
fn recording_mode(config: &Config) -> Option<RecordingMode> {
    if !config.storage.enabled {
        return None;
    }
    if config.analytics.enabled {
        Some(RecordingMode::Event)
    } else {
        Some(RecordingMode::Continuous {
            chunk: std::time::Duration::from_secs(config.storage.max_event_duration_secs),
        })
    }
}

fn spawn_cameras(ctx: &SpawnContext, cameras: Vec<config::CameraConfig>) -> CameraHandles {
    let mut handles = CameraHandles {
        pipeline_handles: Vec::new(),
        analyzer_handles: Vec::new(),
        continuous_handles: Vec::new(),
        warm_handles: Vec::new(),
        event_senders: Vec::new(),
        event_sender_map: HashMap::new(),
        buffers_map: HashMap::new(),
    };

    let mode = recording_mode(ctx.config);

    for cam_config in cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), ctx.config.buffer.hot_duration_secs);
        let camera_id = cam_config.id.clone();

        if let Some(mode) = mode {
            // Seeded from what is already on disk, not from now: this process
            // may be minutes old on a box that restarts nightly, and a silence
            // that resets with it is never long enough to notice.
            let already_silent_for = storage::watchdog::silence_before_startup(
                ctx.storage
                    .as_ref()
                    .and_then(|backend| backend.newest_event_end_ns(&camera_id)),
                storage::warm_index::wall_clock_ns(),
            );
            ctx.recording_watchdog.register(
                &camera_id,
                mode,
                std::time::Instant::now(),
                already_silent_for,
            );
        }

        let event_tx = if ctx.config.storage.enabled {
            let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
            let backend = ctx
                .storage
                .clone()
                .expect("storage backend present when storage enabled");
            let writer = WarmWriter::new(
                rx,
                camera_id.clone(),
                &ctx.config.storage,
                backend,
                Arc::clone(ctx.recording_watchdog),
            );
            handles
                .warm_handles
                .push((camera_id.clone(), tokio::spawn(writer.run())));
            handles.event_senders.push(tx.clone());
            handles
                .event_sender_map
                .insert(camera_id.clone(), tx.clone());
            Some(tx)
        } else {
            None
        };

        handles
            .buffers_map
            .insert(camera_id.clone(), Arc::clone(&buffer));

        let buffer_clone = Arc::clone(&buffer);
        let shutdown_clone = ctx.shutdown.clone();
        let handle = tokio::spawn(async move {
            run_camera(cam_config, buffer_clone, shutdown_clone).await;
        });
        handles
            .pipeline_handles
            .push((camera_id.clone(), handle, Arc::clone(&buffer)));

        // Continuous recording: storage on, analytics off. With no analyzer to
        // close motion runs, a dedicated task rolls fixed-length chunks straight
        // from the hot buffer into the same warm writer. Gated on the mode that
        // the watchdog is told about, so the two cannot disagree about which
        // combination is continuous.
        if let Some(RecordingMode::Continuous { chunk }) = mode {
            if let Some(tx) = event_tx.clone() {
                let recorder = run_continuous_recorder(
                    camera_id.clone(),
                    Arc::clone(&buffer),
                    tx,
                    chunk,
                    Arc::clone(&ctx.shutdown.flag),
                );
                handles
                    .continuous_handles
                    .push((camera_id.clone(), tokio::spawn(recorder)));
            }
        }

        if ctx.config.analytics.enabled {
            let analyzer_handle = analytics::spawn_analyzer(
                AnalyzerContext {
                    camera_id: camera_id.clone(),
                    buffer,
                    motion_store: ctx.motion_store.clone(),
                    detection_store: Some(ctx.detection_store.clone()),
                    detect_tx: ctx.detect_tx.clone(),
                    event_registry: ctx.event_registry.clone(),
                    config: ctx.config.analytics.clone(),
                    motion_settings: ctx
                        .motion_settings
                        .clone()
                        .expect("motion settings initialized when analytics enabled"),
                    event_tx,
                    mqtt_tx: ctx.mqtt_tx.clone(),
                    pre_padding_ns: ctx.config.storage.pre_padding_secs * 1_000_000_000,
                    post_padding: std::time::Duration::from_secs(
                        ctx.config.storage.post_padding_secs,
                    ),
                    max_event_duration: std::time::Duration::from_secs(
                        ctx.config.storage.max_event_duration_secs,
                    ),
                },
                Arc::clone(&ctx.shutdown.flag),
            );
            handles.analyzer_handles.push((camera_id, analyzer_handle));
        }
    }

    handles
}

enum ShutdownReason {
    Signal,
    UpdateInstalled,
}

async fn wait_for_shutdown(shutdown: &ShutdownSignal) -> ShutdownReason {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
    let reason = tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("SIGINT received, shutting down");
            ShutdownReason::Signal
        }
        _ = sigterm.recv() => {
            tracing::info!("SIGTERM received, shutting down");
            ShutdownReason::Signal
        }
        _ = shutdown.wake.notified() => ShutdownReason::UpdateInstalled,
    };
    shutdown.request();
    reason
}

/// Last-resort liveness backstop for the drain that follows an installed
/// update. The new binary is already on disk by then, so a drain that never
/// finishes would leave the old process running the old code forever with
/// nothing outside to notice: a signal shutdown is bounded by the service
/// manager (`TimeoutStopSec` / OpenRC's `retry`, both set by the unit
/// `camon install service` writes), an internally requested one is not.
///
/// It guarantees the restart *eventually* happens; it does not guarantee that
/// what it terminates was stuck. No honest value could: against a black-holing
/// stathost server a single event legitimately takes longer than any deadline
/// worth having — the video is put with one retry (2 x `UPLOAD_TIMEOUT`),
/// then the sidecar, then one put per filmstrip frame, all serial — and a
/// writer queue can hold several events. So this can abandon a drain that is
/// still making progress, losing the remainder. That is the accepted trade:
/// the alternative is an NVR that silently never restarts.
///
/// It is also the budget the drain's own phases are sized against, and
/// [`graceful_shutdown`] measures phase 3's deadline from it, so the drain
/// finishes and says what it lost a moment before this thread would have taken
/// the process out from under it without saying anything. See
/// [`crate::shutdown`] for the arithmetic.
pub const RESTART_DRAIN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(360);

/// How often the watchdog looks for an installed update while the drain runs.
const WATCHDOG_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Spawned at the start of the drain on *both* shutdown paths, then armed by
/// the `update_installed` state rather than by whichever `select!` arm woke the
/// wait: when a signal and an update land together the arm that wins is
/// pseudo-random, and an in-flight check can install the binary after a signal
/// drain has already begun. Either way the process must still end.
///
/// A wedged `spawn_blocking` thread and a blocked `sync_all` are both
/// uncancellable from async code, so this is a plain thread. Everything the
/// operator needs is logged when it arms: past that point the thread only
/// sleeps and terminates, because whatever wedged the drain could just as
/// easily have wedged stderr. `_exit` runs no atexit handlers and no
/// destructors — nothing that could block on the same wedge.
fn spawn_restart_watchdog(update_installed: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        while !update_installed.load(Ordering::Relaxed) {
            std::thread::sleep(WATCHDOG_POLL_INTERVAL);
        }
        tracing::warn!(
            deadline_secs = RESTART_DRAIN_DEADLINE.as_secs(),
            "update installed: the process is terminated at this deadline whether or not the \
             shutdown drain has finished, abandoning any recording still being flushed and \
             every queued event, so the new binary can start"
        );
        std::thread::sleep(RESTART_DRAIN_DEADLINE);
        unsafe { libc::_exit(0) };
    });
}

/// How long shutdown waits for the MQTT bridge to publish its retained
/// `offline` marker and disconnect before giving up on it.
pub const MQTT_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Join a set of per-camera tasks under one shared deadline, so a rack of
/// cameras cannot multiply a per-task bound into a stop that outlives its
/// budget. See [`crate::shutdown`] for why every phase of the drain is bounded.
///
/// What is abandoned is named — the task and the camera it belongs to — because
/// an operator reading this line afterwards is trying to work out which
/// camera's recording to distrust, and "a motion analyzer" does not answer
/// that. The consumer's own bound has usually already logged how much it left
/// behind; this says which one stopped answering at all.
async fn join_all_before(
    handles: Vec<(String, tokio::task::JoinHandle<()>)>,
    deadline: tokio::time::Instant,
    what: &str,
) {
    for (camera_id, handle) in handles {
        if tokio::time::timeout_at(deadline, handle).await.is_err() {
            tracing::warn!(
                camera = %camera_id,
                "{what} did not stop within the shutdown drain bound, abandoning it where it stands"
            );
        }
    }
}

/// The drain both stop paths end in, in the three phases [`crate::shutdown`]
/// describes: the producers stop and publish their watermarks, the consumers
/// drain through them, and the writers write out what they were handed.
async fn graceful_shutdown(
    handles: CameraHandles,
    retention_handle: Option<tokio::task::JoinHandle<()>>,
    detect_worker_handle: Option<tokio::task::JoinHandle<()>>,
    mqtt_handle: Option<tokio::task::JoinHandle<()>>,
) {
    // Phase 3's deadline, taken here rather than when phase 3 begins: the
    // budget is one budget, and a phase that overran has to come out of the
    // writers' share and not out of the service manager's patience.
    let budget_ends =
        tokio::time::Instant::now() + RESTART_DRAIN_DEADLINE - crate::shutdown::TEARDOWN_MARGIN;

    // PHASE 1 — the producers stop, completely, before anything downstream is
    // asked to finish. Each camera thread is at most one 500ms poll from
    // noticing the flag, after which it flushes the GOP it was filling, kills
    // its ffmpeg and returns. A camera that does not come back inside the
    // shared bound is left running: a provisional watermark is published from
    // wherever its buffer had reached, and everything behind it carries on.
    // Waiting for it instead would mean a stuck network read could stop the
    // process from ever restarting.
    //
    // This is the first thing the drain does. It used to be the second, behind
    // the retention join below, and that cost nothing back when every worker
    // exited on the flag — but a consumer's phase-2 bound runs from the flag,
    // so a remote sweep parked on a request timeout would spend the consumers'
    // whole gate before the cameras were even joined, and every one of them
    // would then report a lost tail it never actually had a chance to drain.
    let mut buffers_with_ids = Vec::new();
    let cameras_joined_by = tokio::time::Instant::now() + crate::shutdown::CAMERA_JOIN_BOUND;
    for (camera_id, handle, buffer) in handles.pipeline_handles {
        let stopped = tokio::time::timeout_at(cameras_joined_by, handle)
            .await
            .is_ok();
        if !stopped {
            tracing::warn!(
                camera = %camera_id,
                bound_secs = crate::shutdown::CAMERA_JOIN_BOUND.as_secs(),
                "camera pipeline did not stop in time; its watermark goes out provisional and its \
                 consumers drain to their own bound instead"
            );
        } else {
            tracing::info!(camera = %camera_id, "camera pipeline stopped");
        }
        // PHASE 2 opens here: the watermark is published by the drain, after
        // the join, so that no push can land behind it. Unconditional — a
        // consumer waiting on a watermark that is never published is a
        // consumer waiting out its whole bound for nothing.
        let watermark = {
            let mut buf = buffer.write_recover();
            if stopped {
                buf.seal()
            } else {
                buf.seal_provisionally()
            }
        };
        tracing::debug!(
            camera = %camera_id,
            terminal_sequence = watermark.sequence,
            provisional = watermark.provisional,
            "camera watermark published"
        );
        buffers_with_ids.push((camera_id, buffer));
    }

    // PHASE 2 — analyzers and continuous recorders keep consuming until they
    // have drained through their camera's watermark, so the tail the camera
    // pushed on its way out is part of the event or chunk they flush rather
    // than footage that arrived one poll too late. Each bounds its own drain;
    // this bound is only for the tick it is in when that one trips. Recorders
    // are joined before the senders are dropped below, so their final chunk is
    // guaranteed accepted.
    let consumers_joined_by = tokio::time::Instant::now() + crate::shutdown::CONSUMER_JOIN_BOUND;
    join_all_before(
        handles.analyzer_handles,
        consumers_joined_by,
        "motion analyzer",
    )
    .await;
    join_all_before(
        handles.continuous_handles,
        consumers_joined_by,
        "continuous recorder",
    )
    .await;

    // The detection worker is aborted, not drained: queued jobs and even an
    // in-flight Ollama request (up to 90s) are droppable by design — losing
    // one costs only an object upgrade, never footage. Aborting also releases
    // the worker's warm-writer senders so the writers below can drain.
    if let Some(handle) = detect_worker_handle {
        handle.abort();
        let _ = handle.await;
    }

    // Joined here — after the analyzers flushed their final MotionEnd, before
    // the buffers and writers go away. What preserves that last transition is
    // not the position of this join but how long the bridge goes on receiving:
    // it stops only once the producers have dropped their senders (see
    // `mqtt::bridge_is_done`), which for a phase-2 analyzer can be half a
    // minute after the stop flag. Joining here is what gives it somewhere to
    // publish the transition to, and the retained `offline` marker after it. A
    // broker that has become unreachable must not hold shutdown up, hence the
    // timeout.
    if let Some(handle) = mqtt_handle {
        let abort = handle.abort_handle();
        if tokio::time::timeout(MQTT_SHUTDOWN_TIMEOUT, handle)
            .await
            .is_err()
        {
            tracing::warn!("mqtt bridge did not stop in time, aborting it");
            abort.abort();
        }
    }

    // PHASE 3 — with all senders gone the warm writers drain their queues and
    // exit; awaiting them is what puts every accepted event on disk. BOTH
    // sender holders must drop — the map's clones alone keep the channels open,
    // which deadlocked shutdown here until 2026-07-24. Bounded by what the
    // earlier phases left of the budget, which is nearly all of it in a healthy
    // stop: a consumer abandoned in phase 2 still holds a sender, so the
    // channel it holds open would otherwise make this wait for ever.
    drop(handles.event_senders);
    drop(handles.event_sender_map);
    join_all_before(handles.warm_handles, budget_ends, "warm writer").await;

    // The retention sweep, joined last and inside phase 3's deadline. Waited
    // for rather than aborted, because a sweep that is allowed to reach the end
    // of the event it is deleting leaves no work behind, and asking it to stop
    // is how that is arranged: it polls the shutdown flag between events, so it
    // ends by itself within one event's deletes — or, on a remote backend still
    // retrying the index scan its startup could not do, within the one listing
    // or sidecar read already in flight, since that retry polls the same flag
    // between attempts and between reads.
    //
    // What waiting does NOT buy is protection from being cut mid-delete. If the
    // bound below trips, this drain returns, the runtime is torn down and the
    // sweep's future is dropped wherever it was awaiting — and on the updater's
    // path the process `_exit`s under it. Detaching a task is not letting it
    // finish. The store survives that for the same reason it survives the
    // service manager's SIGKILL, which no arrangement here has ever been able
    // to prevent: one event's delete is recoverable wherever it is interrupted.
    // How, and by whom, differs by backend, because the two delete in different
    // orders and for good reasons of their own:
    //
    // * Local disk (`warm_index::remove_event_files`) unlinks the sidecar and
    //   thumbnails first and the `.ts` last, so the survivor is a bare `.ts` —
    //   which is what the startup scan indexes and what any later retention
    //   sweep expires again, finishing the job. Pinned by
    //   `a_delete_that_cannot_finish_leaves_the_video_rather_than_its_metadata`
    //   (the interrupted state is the recoverable one),
    //   `prune_keeps_events_it_could_not_delete_and_retries_them` (a later
    //   sweep retries it) and `prune_unindexes_events_whose_files_already_
    //   vanished` (an interruption past the last unlink leaves no phantom
    //   entry).
    // * Stathost (`delete_event_objects`) goes thumbnails, video, sidecar —
    //   deliberately the other way round, so that a refused video delete never
    //   leaves a `.ts` whose type the next scan has to guess. Its survivor is
    //   therefore an orphan `.json` (or an orphan thumbnail), which indexes
    //   nothing, and its healer is `sweep_orphaned_metadata`. That runs on a
    //   `ScanKind::Startup` pass ONLY — a healing rescan skips orphan
    //   collection by design, pinned by
    //   `a_healing_rescan_leaves_orphaned_metadata_for_the_next_startup` — so
    //   this survivor waits for a restart rather than for the next sweep. It
    //   costs a few hundred bytes until then and nothing else: nothing indexes
    //   it, counts it against the budget or reads it. No test reaches that
    //   state by interrupting a delete, but `an_orphan_sidecar_indexes_nothing_
    //   and_is_collected` reaches the identical on-disk state from the other
    //   direction — an upload whose video failed — and pins the collection.
    //
    // Moving the wait here from the head of the drain changes no interleaving:
    // the sweep has been running concurrently with everything since the flag
    // went up and still is, and it touches the store while the cameras touch
    // their hot buffers — only the waiting moved. What it buys is that a sweep
    // parked on a remote request timeout now spends the writers' remainder of
    // the budget, which it shares with them, instead of spending phase 2's gate
    // before phase 2 had begun.
    if let Some(handle) = retention_handle {
        if tokio::time::timeout_at(budget_ends, handle).await.is_err() {
            tracing::warn!(
                "retention sweep did not finish within the shutdown budget; it is abandoned where \
                 it stands and may be cut part-way through deleting an event, which a later sweep \
                 (local disk) or the next startup (stathost) finishes"
            );
        }
    }

    for (camera_id, buffer) in buffers_with_ids {
        let buf = buffer.read_recover();
        tracing::info!(
            camera = %camera_id,
            segments = buf.segment_count(),
            duration_secs = format!("{:.1}", buf.current_duration_secs()),
            "final buffer stats"
        );
    }
}

/// Run camon: load the configuration, bring the cameras up, serve the API, and
/// drain everything again when a signal or an installed update asks for it.
///
/// `check_update` is the binary's self-updater, called once at startup and then
/// every [`UPDATE_CHECK_INTERVAL`] until it installs something.
pub async fn run<F, Fut, E>(check_update: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<bool, E>> + Send + 'static,
    E: std::fmt::Display + 'static,
{
    // A healthy production log is empty: only things that need attention
    // (warn and up) are logged by default. Dev builds keep the full debug
    // stream, and RUST_LOG (e.g. `RUST_LOG=camon=debug`) overrides both when
    // an incident needs more detail.
    let default_filter = if cfg!(debug_assertions) {
        "camon=debug"
    } else {
        "camon=warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .init();

    let args = parse_cli_args();
    let loaded = match args.config {
        Some(path) => Config::load_from_with_overrides(path, &args.overrides),
        None => Config::load(&args.overrides),
    };
    // Display, not the `?` operator's Debug: a config error is for the operator
    // to read and fix.
    let config = match loaded {
        Ok(config) => config,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    // Created before the update check so the periodic checker can ask for the
    // same graceful shutdown a signal does.
    let shutdown = ShutdownSignal::new();
    check_for_updates(&config, &shutdown, check_update).await;

    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let camera_ids: Vec<String> = config.cameras.iter().map(|c| c.id.clone()).collect();
    let motion_store = MotionStore::new(&camera_ids);
    let detection_store = DetectionStore::new(&camera_ids);
    let debug_store = DetectionDebugStore::new(&camera_ids);
    let object_detection_ready = log_object_detection_config(&config);
    let Storage {
        backend: storage,
        scanned: warm_index_scanned,
    } = init_storage(&config, &camera_ids).await;
    // The recording-silence watchdog cannot see the storage volume being
    // unmounted: writes to the bare mountpoint succeed and keep resetting it.
    // So the volume is watched directly — by the backends that have one to
    // lose, which is the local-disk one only.
    let volume_anchor = storage
        .as_ref()
        .and_then(|backend| backend.volume_anchor().cloned());
    let motion_settings = init_motion_settings(&config, &camera_ids);

    log_recording_mode(&config);

    // Object detection runs on ONE global worker task with a small bounded
    // job queue — strictly serial, at most one in-flight Ollama request
    // across all cameras (the GPU degrades badly under parallel load).
    let ollama_client = if object_detection_ready {
        create_ollama_client(&config)
    } else {
        None
    };
    let object_classes = mqtt_object_classes(ollama_client.as_ref());
    let event_registry = if ollama_client.is_some() && config.storage.enabled {
        Some(EventRegistry::new(&camera_ids))
    } else {
        None
    };
    let (detect_tx, detect_rx) = match ollama_client {
        Some(_) => {
            let (tx, queue) = detect_queue();
            (Some(tx), Some(queue))
        }
        None => (None, None),
    };

    // The MQTT bridge is fed by the analyzers (motion) and the detection worker
    // (verdicts), so its channel must exist before either is spawned.
    let (mqtt_tx, mqtt_rx) = if config.mqtt.enabled {
        if !config.analytics.enabled {
            tracing::warn!(
                "mqtt enabled without analytics: snapshots and sensors are motion-gated, so \
                 the entities will stay idle (availability and discovery still published)"
            );
        }
        let (tx, rx) = tokio::sync::mpsc::channel(MQTT_EVENT_CAPACITY);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    // "This camera has recorded nothing" is the one symptom that every way of
    // failing to record shares, so one watchdog covers them all. It watches
    // only the cameras that are expected to record — see `recording_mode`.
    let recording_watchdog = Arc::new(RecordingWatchdog::new());

    let spawn_ctx = SpawnContext {
        config: &config,
        motion_store: &motion_store,
        detection_store: &detection_store,
        storage: &storage,
        motion_settings: &motion_settings,
        detect_tx: &detect_tx,
        event_registry: &event_registry,
        mqtt_tx: &mqtt_tx,
        recording_watchdog: &recording_watchdog,
        shutdown: &shutdown,
    };
    let camera_handles = spawn_cameras(&spawn_ctx, config.cameras.clone());

    // Retention is a property of the store, not of a camera: one task sweeps
    // every camera on a schedule, however many writers there are.
    let retention_handle = storage.as_ref().map(|backend| {
        let task = RetentionTask::new(
            Arc::clone(backend),
            &config.storage,
            Arc::clone(&shutdown.flag),
        );
        // A backend that never got its index is waiting for a sweep to retry
        // the scan, so it does not wait the usual hour for one.
        let task = if warm_index_scanned {
            task
        } else {
            task.after_a_failed_scan()
        };
        tokio::spawn(task.run())
    });

    let detect_worker_handle = match (ollama_client, detect_rx) {
        (Some(client), Some(rx)) => {
            let worker = DetectionWorker::new(
                client,
                detection_store.clone(),
                Some(debug_store.clone()),
                event_registry.clone(),
                camera_handles.event_sender_map.clone(),
                mqtt_tx.clone(),
            );
            Some(tokio::spawn(worker.run(rx)))
        }
        _ => None,
    };

    let mqtt_handle = mqtt_rx.map(|rx| {
        mqtt::spawn_bridge(
            BridgeContext {
                config: config.mqtt.clone(),
                buffers: Arc::new(camera_handles.buffers_map.clone()),
                camera_ids: camera_ids.clone(),
                classes: object_classes,
                entities_path: Some(mqtt_entities_path(&config)),
                shutdown: Arc::clone(&shutdown.flag),
            },
            rx,
        )
    });
    // The analyzers and detection worker hold their own clones; dropping this
    // one lets the bridge see the channel close once they are gone.
    drop(mqtt_tx);
    // Analyzers hold their own clones; dropping the original closes the job
    // channel once they exit, letting the worker finish in normal operation.
    drop(detect_tx);

    let app_state = AppState::new(
        camera_handles.buffers_map.clone(),
        motion_store,
        detection_store,
        debug_store,
        storage,
        motion_settings,
    );
    let http_addr = std::net::SocketAddr::new(config.http.bind_addr(), config.http.port);
    let http_token = config.http.token.clone();
    api::warn_if_open(
        http_addr.ip(),
        http_token.as_deref(),
        config.http.allow_open,
    );
    let server_handle = tokio::spawn(async move {
        if let Err(e) = api::start_server(app_state, http_addr, http_token).await {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    // No camera is registered when nothing is expected to record, and an empty
    // watchdog has nothing to poll.
    let watchdog_handle =
        recording_mode(&config).map(|_| tokio::spawn(Arc::clone(&recording_watchdog).run()));

    let anchor_handle = volume_anchor.map(|anchor| tokio::spawn(anchor.run()));

    let reason = wait_for_shutdown(&shutdown).await;
    server_handle.abort();
    // Aborted rather than drained: it holds nothing that has to reach disk, and
    // a camera being quiet is not news while the process is stopping.
    if let Some(handle) = watchdog_handle {
        handle.abort();
    }
    if let Some(handle) = anchor_handle {
        handle.abort();
    }

    spawn_restart_watchdog(Arc::clone(&shutdown.update_installed));
    graceful_shutdown(
        camera_handles,
        retention_handle,
        detect_worker_handle,
        mqtt_handle,
    )
    .await;

    match reason {
        ShutdownReason::Signal => tracing::info!("shutdown complete"),
        ShutdownReason::UpdateInstalled => {
            tracing::info!("shutdown complete, restarting into the updated binary")
        }
    }
    Ok(())
}

/// Base reconnect delay, and the value backoff resets to after a healthy run.
const RECONNECT_BASE_SECS: u64 = 5;
/// Upper bound on the reconnect backoff delay.
const RECONNECT_MAX_SECS: u64 = 60;
/// A pipeline run lasting at least this long is considered healthy and resets backoff.
const HEALTHY_RUN_SECS: u64 = 60;

/// Next delay in the exponential backoff progression (5 -> 10 -> 20 -> 40 -> cap).
fn next_backoff_secs(current: u64) -> u64 {
    (current * 2).min(RECONNECT_MAX_SECS)
}

async fn run_camera(
    config: config::CameraConfig,
    buffer: Arc<RwLock<HotBuffer>>,
    shutdown: ShutdownSignal,
) {
    let camera_id = config.id.clone();

    let buffer_ref = Arc::clone(&buffer);
    let camera_id_clone = camera_id.clone();
    let shutdown_clone = Arc::clone(&shutdown.flag);

    let stats_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        while !shutdown_clone.load(Ordering::Relaxed) {
            interval.tick().await;
            let buf = buffer_ref.read_recover();
            tracing::info!(
                camera = %camera_id_clone,
                segments = buf.segment_count(),
                duration_secs = format!("{:.1}", buf.current_duration_secs()),
                "buffer stats"
            );
        }
    });

    let mut backoff_secs = RECONNECT_BASE_SECS;
    let mut no_recording = camera::NoRecordingTracker::default();

    while !shutdown.requested() {
        tracing::info!(camera = %camera_id, url = %config.redacted_url(), "connecting to camera");

        let pipeline = FfmpegPipeline::new(&config, Arc::clone(&buffer));
        let shutdown_ref = Arc::clone(&shutdown.flag);

        let started = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || pipeline.run(&shutdown_ref)).await;
        let ran_for = started.elapsed();

        // Only the tracker's own outcomes move its streak: the other arms end a
        // run without showing that anything was recorded, so clearing the count
        // on them would let unrelated failures defer the diagnosis forever.
        match result {
            Ok(Ok(())) => {
                tracing::info!(camera = %camera_id, "pipeline stopped normally");
            }
            // Reported by the tracker instead of here: a stream that records
            // nothing fails on every reconnect, and the same error line once a
            // minute forever tells an operator less than one that escalates.
            Ok(Err(camera::RtspError::NoRecording(failure))) => {
                no_recording.report(&camera_id, &failure);
            }
            Ok(Err(e)) => {
                tracing::error!(camera = %camera_id, "pipeline error: {}", e);
            }
            Err(e) => {
                tracing::error!(camera = %camera_id, "pipeline task panicked: {}", e);
            }
        }

        if shutdown.requested() {
            break;
        }

        // A long healthy run resets the backoff to the base delay.
        if ran_for >= std::time::Duration::from_secs(HEALTHY_RUN_SECS) {
            backoff_secs = RECONNECT_BASE_SECS;
        }

        let delay_ms = apply_jitter(backoff_secs * 1000, jitter_source());
        tracing::info!(
            camera = %camera_id,
            "reconnecting in {:.1} seconds",
            delay_ms as f64 / 1000.0
        );
        shutdown
            .sleep_or_shutdown(std::time::Duration::from_millis(delay_ms))
            .await;

        backoff_secs = next_backoff_secs(backoff_secs);
    }

    stats_handle.abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_progression_doubles_then_caps() {
        assert_eq!(next_backoff_secs(5), 10);
        assert_eq!(next_backoff_secs(10), 20);
        assert_eq!(next_backoff_secs(20), 40);
        assert_eq!(next_backoff_secs(40), 60);
        assert_eq!(next_backoff_secs(60), 60);
    }

    /// A camera parked in its reconnect backoff used to hold the drain up for
    /// the whole delay — up to a minute plus jitter — because the sleep only
    /// ended on its own. Paused time: the assertion is on the virtual clock, so
    /// a regression reads as a full 60s rather than as a slow test.
    #[tokio::test(start_paused = true)]
    async fn reconnect_backoff_ends_as_soon_as_shutdown_is_requested() {
        let shutdown = ShutdownSignal::new();
        let signaller = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            signaller.request();
        });

        let started = tokio::time::Instant::now();
        shutdown
            .sleep_or_shutdown(Duration::from_secs(RECONNECT_MAX_SECS))
            .await;
        assert_eq!(
            started.elapsed(),
            Duration::from_millis(20),
            "backoff outlived the shutdown request"
        );
    }

    /// The flag can already be up when the sleep is reached — the pipeline it
    /// follows may have exited because of it.
    #[tokio::test(start_paused = true)]
    async fn reconnect_backoff_is_skipped_when_shutdown_is_already_requested() {
        let shutdown = ShutdownSignal::new();
        shutdown.request();

        let started = tokio::time::Instant::now();
        shutdown
            .sleep_or_shutdown(Duration::from_secs(RECONNECT_MAX_SECS))
            .await;
        assert_eq!(started.elapsed(), Duration::ZERO);
    }

    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn config_with(storage: bool, analytics: bool) -> Config {
        let mut config: Config = toml::from_str("").expect("every config section has a default");
        config.storage.enabled = storage;
        config.analytics.enabled = analytics;
        config
    }

    /// Only a camera that is expected to record is watched for silence. The two
    /// storage-off states are said once at startup instead: a daily warning
    /// about a camera nobody asked to record is noise that teaches the operator
    /// to skip the warnings about the ones that were.
    #[test]
    fn only_cameras_expected_to_record_are_watched() {
        assert_eq!(
            recording_mode(&config_with(true, true)),
            Some(RecordingMode::Event)
        );
        assert_eq!(
            recording_mode(&config_with(true, false)),
            Some(RecordingMode::Continuous {
                chunk: Duration::from_secs(
                    config_with(true, false).storage.max_event_duration_secs
                )
            })
        );
        assert_eq!(recording_mode(&config_with(false, true)), None);
        assert_eq!(recording_mode(&config_with(false, false)), None);
    }

    /// The continuous limit is derived from the configured cap, not from the
    /// default, so an operator who changes one changes the other.
    #[test]
    fn the_continuous_chunk_cap_comes_from_the_config() {
        let mut config = config_with(true, false);
        config.storage.max_event_duration_secs = 30;
        assert_eq!(
            recording_mode(&config),
            Some(RecordingMode::Continuous {
                chunk: Duration::from_secs(30)
            })
        );
    }

    /// A stand-in for the binary's update check that counts its calls and
    /// replays a scripted sequence of outcomes.
    fn scripted_checker(
        outcomes: Vec<Result<bool, std::io::Error>>,
    ) -> (
        impl Fn() -> std::future::Ready<Result<bool, std::io::Error>>,
        Arc<AtomicUsize>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let outcomes = Arc::new(std::sync::Mutex::new(outcomes.into_iter()));
        let check = move || {
            counter.fetch_add(1, Ordering::Relaxed);
            let next = outcomes
                .lock()
                .expect("scripted checker poisoned")
                .next()
                .expect("checker called more times than scripted");
            std::future::ready(next)
        };
        (check, calls)
    }

    /// Failed and empty checks keep the loop alive and must never look like an
    /// installed update. Terminated by a stand-in signal rather than an
    /// install, so that the install branch — the one that would take the whole
    /// test binary down if it regressed — stays confined to the child process
    /// in `applied_update_returns_instead_of_exiting`.
    #[tokio::test]
    async fn failed_update_checks_do_not_request_shutdown() {
        let shutdown = ShutdownSignal::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let signalled = shutdown.clone();
        let check = move || {
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 3 {
                signalled.flag.store(true, Ordering::Relaxed);
            }
            std::future::ready(match n {
                1 => Err(std::io::Error::other("network down")),
                _ => Ok(false),
            })
        };
        run_update_check_loop_with(Duration::from_millis(1), shutdown.clone(), check).await;

        assert_eq!(calls.load(Ordering::Relaxed), 3, "loop stopped checking");
        assert!(
            !shutdown.update_installed.load(Ordering::Relaxed),
            "a failed check asked for a restart"
        );
    }

    /// A shutdown already under way must not have an install started
    /// underneath it — the binary would be swapped while the drain runs.
    #[tokio::test]
    async fn update_check_is_skipped_once_shutdown_is_requested() {
        let shutdown = ShutdownSignal::new();
        shutdown.flag.store(true, Ordering::Relaxed); // as a signal would
        let (check, calls) = scripted_checker(vec![Ok(false)]);
        run_update_check_loop_with(Duration::from_millis(1), shutdown.clone(), check).await;

        assert_eq!(calls.load(Ordering::Relaxed), 0, "checked during shutdown");
    }

    /// Both shutdown paths end in this same drain, and phase 3's contract is
    /// that an event already accepted by a warm writer reaches disk before the
    /// process goes away — the footage the update path used to lose.
    #[tokio::test]
    async fn graceful_shutdown_drains_queued_events_to_disk() {
        use crate::buffer::warm::FinishedEvent;
        use crate::buffer::GopSegment;

        let dir = tempfile::tempdir().unwrap();
        let backend: Arc<dyn WarmStorageBackend> = Arc::new(LocalDiskBackend::new(
            dir.path().to_path_buf(),
            &["cam".to_string()],
        ));
        let warm_config = config::WarmConfig::default();
        let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let writer = WarmWriter::new(
            rx,
            "cam".to_string(),
            &warm_config,
            backend,
            Arc::new(RecordingWatchdog::new()),
        );

        let event = FinishedEvent {
            segments: vec![GopSegment {
                start_pts: 0,
                duration_ns: 1_000_000_000,
                data: Arc::new(vec![0x47; 188]),
                frame_count: 1,
            }],
            first_pts: 0,
            total_bytes: 188,
            has_objects: false,
            object_classes: Vec::new(),
            filmstrip_frames: None,
            backend: None,
            model: None,
            detection_details: Vec::new(),
            continues: false,
            is_continuous: false,
        };
        tx.send(WriterMessage::Event(event)).await.unwrap();

        let handles = CameraHandles {
            pipeline_handles: Vec::new(),
            analyzer_handles: Vec::new(),
            continuous_handles: Vec::new(),
            warm_handles: vec![("cam".to_string(), tokio::spawn(writer.run()))],
            event_senders: vec![tx.clone()],
            event_sender_map: HashMap::from([("cam".to_string(), tx)]),
            buffers_map: HashMap::new(),
        };
        // Bounded: the failure this guards against — a sender that outlives the
        // drain, as in the 2026-07-24 deadlock — hangs rather than returns, and
        // an unexplained CI timeout is a bad way to find that out.
        tokio::time::timeout(
            Duration::from_secs(30),
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("graceful_shutdown deadlocked instead of draining");

        let written = dir.path().join("cam").join("movements").join("0_1000.ts");
        assert!(written.exists(), "queued event was not flushed to disk");
    }

    // ---- The phased stop, end to end (see `crate::shutdown`) ----

    const SEC: u64 = 1_000_000_000;

    fn gop(index: u64) -> crate::buffer::GopSegment {
        crate::buffer::GopSegment {
            start_pts: index * SEC,
            duration_ns: SEC,
            data: Arc::new(vec![0x47; 188]),
            frame_count: 1,
        }
    }

    fn empty_handles() -> CameraHandles {
        CameraHandles {
            pipeline_handles: Vec::new(),
            analyzer_handles: Vec::new(),
            continuous_handles: Vec::new(),
            warm_handles: Vec::new(),
            event_senders: Vec::new(),
            event_sender_map: HashMap::new(),
            buffers_map: HashMap::new(),
        }
    }

    /// One camera wired the way [`spawn_cameras`] wires a continuous-recording
    /// one — hot buffer, continuous recorder, warm writer, local disk — with
    /// `camera` standing in for the camera task and handed the buffer to push
    /// into. The chunk cap is an hour, so the only chunk that ever rolls is the
    /// final one and its name says exactly how much of the recording survived.
    fn add_continuous_camera<F, Fut>(
        handles: &mut CameraHandles,
        dir: &std::path::Path,
        camera_id: &str,
        shutdown: &ShutdownSignal,
        camera: F,
    ) where
        F: FnOnce(Arc<RwLock<HotBuffer>>) -> Fut,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let backend: Arc<dyn WarmStorageBackend> = Arc::new(LocalDiskBackend::new(
            dir.to_path_buf(),
            &[camera_id.to_string()],
        ));
        let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let writer = WarmWriter::new(
            rx,
            camera_id.to_string(),
            &config::WarmConfig::default(),
            backend,
            Arc::new(RecordingWatchdog::new()),
        );
        let buffer = HotBuffer::new(camera_id.to_string(), 3600);

        handles
            .warm_handles
            .push((camera_id.to_string(), tokio::spawn(writer.run())));
        handles.continuous_handles.push((
            camera_id.to_string(),
            tokio::spawn(crate::buffer::warm::run_continuous_recorder(
                camera_id.to_string(),
                Arc::clone(&buffer),
                tx.clone(),
                Duration::from_secs(3600),
                Arc::clone(&shutdown.flag),
            )),
        ));
        handles.event_senders.push(tx.clone());
        handles
            .event_sender_map
            .insert(camera_id.to_string(), tx.clone());
        handles.pipeline_handles.push((
            camera_id.to_string(),
            tokio::spawn(camera(Arc::clone(&buffer))),
            buffer,
        ));
    }

    /// A camera that notices the stop flag one poll late and then flushes the
    /// GOP it was still filling, which is what every real one does.
    async fn records_two_seconds_then_a_tail(buffer: Arc<RwLock<HotBuffer>>) {
        buffer.write_recover().push(gop(0));
        buffer.write_recover().push(gop(1));
        tokio::time::sleep(Duration::from_millis(500)).await;
        buffer.write_recover().push(gop(2));
    }

    fn continuous_file(dir: &std::path::Path, camera_id: &str, stem: &str) -> std::path::PathBuf {
        dir.join(camera_id)
            .join("continuous")
            .join(format!("{stem}.ts"))
    }

    /// The bug the phases exist for, through the whole graph. One stop flag
    /// used to halt the producer and its consumer at the same instant, so the
    /// GOP a camera pushed on its way out was written by nobody — every stop,
    /// every camera, every time.
    ///
    /// The filename is the proof: `{first_pts}_{duration_ms}.ts`, so a
    /// three-second recording that lost its tail is not a subtly shorter file,
    /// it is `0_2000.ts` where `0_3000.ts` should be. Paused time throughout —
    /// the bounds involved are tens of seconds and none of them should be
    /// reached.
    #[tokio::test(start_paused = true)]
    async fn the_tail_a_camera_pushes_on_its_way_out_reaches_disk() {
        let dir = tempfile::tempdir().unwrap();
        let shutdown = ShutdownSignal::new();
        shutdown.request();
        let mut handles = empty_handles();
        add_continuous_camera(
            &mut handles,
            dir.path(),
            "cam",
            &shutdown,
            records_two_seconds_then_a_tail,
        );

        let started = tokio::time::Instant::now();
        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("the drain never finished");

        assert!(
            continuous_file(dir.path(), "cam", "0_3000").exists(),
            "the recording is missing the tail the camera pushed while stopping"
        );
        // Following the watermark, not waiting out a bound: a drain that only
        // finished because something timed out is not the fix.
        assert!(
            started.elapsed() < crate::shutdown::CAMERA_JOIN_BOUND,
            "the drain waited out a bound instead of following the watermark"
        );
    }

    /// The second time this goes wrong: a camera thread that never comes back —
    /// an ffmpeg unkillable in D state, a read that returns to nobody — must
    /// cost its own tail and not the restart. Its watermark is published on its
    /// behalf so the consumers behind it are not left waiting for one.
    #[tokio::test(start_paused = true)]
    async fn a_camera_that_never_stops_is_abandoned_with_a_provisional_watermark() {
        let buffer = HotBuffer::new("cam".to_string(), 3600);
        buffer.write_recover().push(gop(0));
        let mut handles = empty_handles();
        handles.pipeline_handles.push((
            "cam".to_string(),
            tokio::spawn(std::future::pending::<()>()),
            Arc::clone(&buffer),
        ));

        let started = tokio::time::Instant::now();
        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("a wedged camera thread held the whole drain open");

        assert_eq!(
            started.elapsed(),
            crate::shutdown::CAMERA_JOIN_BOUND,
            "the camera join was not bounded"
        );
        // Published, so nothing downstream waits on a watermark that is never
        // coming — and flagged provisional, because the thread it was published
        // for is still running and its consumers must not read the sequence as
        // the end of the recording.
        assert_eq!(
            buffer.read_recover().terminal_watermark(),
            Some(crate::shutdown::Watermark {
                sequence: 1,
                provisional: true,
            }),
            "the camera that hung got no watermark, or one claiming it had stopped"
        );
    }

    /// Watermarks are per camera, so one camera's failure is one camera's loss.
    /// A shared one — a single "the cameras have stopped" flag, say — would let
    /// the slowest camera in the rack decide how much of everyone else's
    /// footage was kept.
    #[tokio::test(start_paused = true)]
    async fn a_camera_that_hangs_does_not_cost_its_neighbour_its_tail() {
        let dir = tempfile::tempdir().unwrap();
        let shutdown = ShutdownSignal::new();
        shutdown.request();
        let mut handles = empty_handles();
        add_continuous_camera(
            &mut handles,
            dir.path(),
            "clean",
            &shutdown,
            records_two_seconds_then_a_tail,
        );
        add_continuous_camera(
            &mut handles,
            dir.path(),
            "wedged",
            &shutdown,
            |buffer| async move {
                buffer.write_recover().push(gop(0));
                buffer.write_recover().push(gop(1));
                std::future::pending::<()>().await;
            },
        );

        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("the drain never finished");

        assert!(
            continuous_file(dir.path(), "clean", "0_3000").exists(),
            "the camera that stopped cleanly lost its tail to the one that hung"
        );
        assert!(
            continuous_file(dir.path(), "wedged", "0_2000").exists(),
            "the wedged camera's recorder wrote nothing at all"
        );
    }

    /// The retention sweep is joined at the end of the drain, not the start.
    ///
    /// Waiting for it first cost nothing back when every worker exited on the
    /// stop flag, but a phase-2 consumer's bound runs from that flag: a remote
    /// sweep parked on a request timeout would spend the consumers' whole gate
    /// before the cameras had even been joined, and every consumer would then
    /// report a tail it was never given a chance to drain. So the assertion is
    /// on the ordering itself — the sweep observes, from inside itself, that
    /// the cameras were joined and their watermarks published long before it
    /// finished.
    #[tokio::test(start_paused = true)]
    async fn a_slow_retention_sweep_does_not_hold_up_the_camera_joins() {
        let buffer = HotBuffer::new("cam".to_string(), 3600);
        buffer.write_recover().push(gop(0));

        let watermark_was_out = Arc::new(AtomicBool::new(false));
        let probe = Arc::clone(&watermark_was_out);
        let watched = Arc::clone(&buffer);
        let retention = tokio::spawn(async move {
            // Longer than every phase-2 bound there is, which is exactly the
            // sweep this ordering exists for.
            tokio::time::sleep(crate::shutdown::TAIL_DRAIN_BOUND * 2).await;
            probe.store(
                watched.read_recover().terminal_watermark().is_some(),
                Ordering::Relaxed,
            );
        });

        let mut handles = empty_handles();
        handles.pipeline_handles.push((
            "cam".to_string(),
            tokio::spawn(std::future::ready(())),
            Arc::clone(&buffer),
        ));

        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, Some(retention), None, None),
        )
        .await
        .expect("the drain never finished");

        assert!(
            watermark_was_out.load(Ordering::Relaxed),
            "the drain waited out the retention sweep before joining its cameras, spending every \
             consumer's phase-2 gate before phase 2 had begun"
        );
    }

    /// A consumer that stalls past its own bound is abandoned where it stands
    /// and the phases behind it run anyway. It is still holding a warm-writer
    /// sender when that happens, which is why phase 3 is bounded too: a channel
    /// nobody will ever close is a writer that never returns, and a stop that
    /// never ends is worse than the tail it was trying to save.
    #[tokio::test(start_paused = true)]
    async fn a_stalled_consumer_does_not_hold_the_stop_open() {
        use crate::buffer::warm::FinishedEvent;

        let dir = tempfile::tempdir().unwrap();
        let backend: Arc<dyn WarmStorageBackend> = Arc::new(LocalDiskBackend::new(
            dir.path().to_path_buf(),
            &["cam".to_string()],
        ));
        let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let writer = WarmWriter::new(
            rx,
            "cam".to_string(),
            &config::WarmConfig::default(),
            backend,
            Arc::new(RecordingWatchdog::new()),
        );
        tx.send(WriterMessage::Event(FinishedEvent {
            segments: vec![gop(0)],
            first_pts: 0,
            total_bytes: 188,
            has_objects: false,
            object_classes: Vec::new(),
            filmstrip_frames: None,
            backend: None,
            model: None,
            detection_details: Vec::new(),
            continues: false,
            is_continuous: false,
        }))
        .await
        .unwrap();

        let mut handles = empty_handles();
        handles
            .warm_handles
            .push(("cam".to_string(), tokio::spawn(writer.run())));
        handles.event_senders.push(tx.clone());
        let held = tx.clone();
        handles.analyzer_handles.push((
            "cam".to_string(),
            tokio::spawn(async move {
                let _sender_the_stall_keeps_open = held;
                std::future::pending::<()>().await;
            }),
        ));
        drop(tx);

        let started = tokio::time::Instant::now();
        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("a stalled analyzer held the stop open past its whole budget");

        assert!(
            started.elapsed() >= crate::shutdown::CONSUMER_JOIN_BOUND,
            "the analyzer was abandoned before its bound"
        );
        assert!(
            dir.path()
                .join("cam")
                .join("movements")
                .join("0_1000.ts")
                .exists(),
            "the drain gave up on the events it could still write"
        );
    }

    /// A stop can be asked for twice — a SIGTERM landing while an update check
    /// already in flight installs a binary and asks for a restart, the ordering
    /// [`spawn_restart_watchdog`] exists for. The second request finds a flag
    /// that is already up and a drain already running, and must change nothing
    /// about what reaches disk.
    #[tokio::test(start_paused = true)]
    async fn a_stop_asked_for_twice_still_lands_the_whole_tail() {
        let dir = tempfile::tempdir().unwrap();
        let shutdown = ShutdownSignal::new();
        shutdown.request();
        let mut handles = empty_handles();
        add_continuous_camera(
            &mut handles,
            dir.path(),
            "cam",
            &shutdown,
            records_two_seconds_then_a_tail,
        );

        let second = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            second.request_restart();
        });
        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("the second request deadlocked the drain");

        assert!(
            continuous_file(dir.path(), "cam", "0_3000").exists(),
            "a second stop request truncated the recording"
        );
        assert!(shutdown.requested(), "the second request cleared the flag");
        assert!(
            shutdown.update_installed.load(Ordering::Relaxed),
            "the restart watchdog would never arm"
        );
    }

    /// Set on the child process spawned by
    /// [`applied_update_returns_instead_of_exiting`]; holds the path of the
    /// marker file that only a normal return can produce.
    const UPDATE_LOOP_MARKER_ENV: &str = "CAMON_TEST_UPDATE_LOOP_MARKER";

    /// The regression itself: this branch used to call `process::exit(0)`,
    /// skipping the drain. libtest cannot catch that in-process — all tests
    /// share one process, so an `exit(0)` mid-run ends the binary with a
    /// success status and the whole run is reported green — so the install
    /// branch is exercised *only* here, in a child process that can write its
    /// marker only by returning from the loop. No other test may install an
    /// update, or it would take the binary down before this one runs.
    #[test]
    fn applied_update_returns_instead_of_exiting() {
        if let Ok(marker) = std::env::var(UPDATE_LOOP_MARKER_ENV) {
            let shutdown = ShutdownSignal::new();
            let (check, calls) = scripted_checker(vec![Ok(true)]);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(run_update_check_loop_with(
                Duration::from_millis(1),
                shutdown.clone(),
                check,
            ));

            assert_eq!(calls.load(Ordering::Relaxed), 1, "update installed twice");
            assert!(shutdown.requested(), "drain flag not raised");
            assert!(
                shutdown.update_installed.load(Ordering::Relaxed),
                "watchdog would never arm"
            );
            // A stored permit, so the main task starts the drain even if it
            // only reaches `notified()` after the update landed.
            runtime.block_on(async {
                tokio::time::timeout(Duration::from_secs(5), shutdown.wake.notified())
                    .await
                    .expect("main task would never have been woken");
            });

            std::fs::write(marker, "returned").unwrap();
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("returned");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "--test-threads=1",
                "app::tests::applied_update_returns_instead_of_exiting",
            ])
            .env(UPDATE_LOOP_MARKER_ENV, &marker)
            // The child's own libtest chatter is noise here; its stderr is kept
            // so a panic inside it is still readable.
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("failed to re-run this test as a child process");

        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if std::time::Instant::now() > deadline {
                let _ = child.kill();
                panic!("child process did not finish");
            }
            std::thread::sleep(Duration::from_millis(20));
        };

        assert!(status.success(), "child test failed: {status}");
        assert!(
            marker.exists(),
            "the update branch took the process down instead of returning"
        );
    }

    const ONE_CAMERA: &str = "\n[[cameras]]\nid = \"yard\"\nurl = \"rtsp://10.0.0.5:554/s0\"\n";

    fn config_from(toml: &str) -> Config {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        Config::load_from_with_overrides(&path, &[]).unwrap()
    }

    /// The classes the model is asked about and the classes Home Assistant
    /// gets occupancy entities for are one list. Two gates that drifted apart
    /// is how `classes = []` came to mean opposite things.
    #[test]
    fn both_consumers_are_handed_the_same_object_classes() {
        let config = config_from(&format!(
            "[analytics]\nenabled = true\n\n[analytics.object_detection]\nenabled = true\n\
             classes = [\"Person\", \"cat\"]\n{ONE_CAMERA}"
        ));
        let client = create_ollama_client(&config).expect("client");
        let expected = vec!["person".to_string(), "cat".to_string()];
        assert_eq!(client.allowed_classes(), expected);
        assert_eq!(mqtt_object_classes(Some(&client)), expected);
    }

    /// Either half of the gate off means no detector runs, so the bridge must
    /// publish no occupancy entities either.
    #[test]
    fn no_object_classes_reach_the_bridge_when_nothing_detects() {
        for flags in [
            "[analytics]\nenabled = true\n\n[analytics.object_detection]\nenabled = false",
            "[analytics]\nenabled = false\n\n[analytics.object_detection]\nenabled = true",
        ] {
            let config = config_from(&format!("{flags}\n{ONE_CAMERA}"));
            assert!(!log_object_detection_config(&config));
            // Startup builds no client in this state, and without one the
            // bridge is handed nothing.
            assert!(mqtt_object_classes(None).is_empty(), "with {flags:?}");
        }
    }
}
