//! Process orchestration: startup, the camera/analyzer/writer task graph, and the graceful
//! drain both shutdown paths end in.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tracing_subscriber::EnvFilter;

use crate::analytics::{
    self, detect_queue, AnalyzerContext, DetectQueueSender, DetectionWorker, OllamaClient,
};
use crate::api::{self, AppState};
use crate::buffer::warm::{run_continuous_recorder, RetentionTask, WarmWriter, WriterMessage};
use crate::buffer::{wall_clock_ns, HotBuffer};
use crate::camera::{self, FfmpegPipeline};
use crate::config::{self, Config};
use crate::locks::LockExt;
use crate::mqtt::{self, BridgeContext, MqttEvent, MQTT_EVENT_CAPACITY};
use crate::retry::{apply_jitter, jitter_source};
use crate::storage::{
    self, DetectionDebugStore, DetectionStore, EventRegistry, LocalDiskBackend, MotionStore,
    RecordingMode, RecordingWatchdog, StathostBackend, WarmStorageBackend,
};
use crate::supervise::{RestartLimit, Supervisor};

/// Why camon stopped being able to run. Both are reported to the operator and
/// both exit nonzero, which is what asks systemd (`Restart=always`) or the Home
/// Assistant Supervisor to start the process again.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// Startup could not take the API's listening socket. Almost always
    /// another process on the port, or a `[http] bind` address this host does
    /// not have.
    #[error("cannot serve the API on {addr}: {source}")]
    Bind {
        addr: std::net::SocketAddr,
        source: std::io::Error,
    },
    /// Camon needed an API token of its own — a network-reachable bind with no
    /// `[http] token` and no `allow_open` — and could not get the randomness to
    /// make one. Startup ends rather than serving the writes unguarded.
    #[error("cannot generate an API token: {source}")]
    ApiToken { source: std::io::Error },
    /// Warm storage is configured and could not be built — in practice the remote backend's
    /// HTTP client (TLS stack, root certificates, proxy environment).
    #[error("cannot start warm storage: {source}")]
    WarmStorage { source: std::io::Error },
    /// A supervised task the process cannot run without died; the drain has already run by now
    /// (see [`crate::supervise`]).
    #[error("supervised tasks died and camon cannot run without them: {}", .tasks.join(", "))]
    TaskDied { tasks: Vec<String> },
}

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
    /// Broadcast wakeup for workers parked far from their next flag poll (a
    /// reconnect backoff can be a minute long). Separate from `wake`, whose
    /// single permit is meant for the main task.
    drain: Arc<tokio::sync::Notify>,
    /// Set once an update has replaced the binary on disk — possibly after a
    /// signal already started a drain, from a check that was in flight. The
    /// restart watchdog keys off it.
    update_installed: Arc<AtomicBool>,
}

/// The switch that tells the restart enforcement a new binary is on disk.
#[derive(Clone, Default)]
pub struct InstalledMarker {
    installed: Arc<AtomicBool>,
}

impl InstalledMarker {
    /// A marker attached to nothing, for a caller that wants the shape without
    /// a whole [`ShutdownSignal`] behind it.
    pub fn new() -> Self {
        Self::default()
    }

    /// A new binary is in place under the name this process was started from.
    pub fn record(&self) {
        self.installed.store(true, Ordering::Relaxed);
    }

    pub fn recorded(&self) -> bool {
        self.installed.load(Ordering::Relaxed)
    }
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

    /// Raise the flag and wake every worker waiting on it.
    fn request(&self) {
        self.flag.store(true, Ordering::Relaxed);
        self.drain.notify_waiters();
    }

    /// Ask for the drain from inside the process, where no signal is coming.
    fn request_now(&self) {
        self.request();
        self.wake.notify_one();
    }

    fn request_restart(&self) {
        self.installed_marker().record();
        self.request_now();
    }

    /// The half of a restart request the updater cannot leave until it returns.
    fn installed_marker(&self) -> InstalledMarker {
        InstalledMarker {
            installed: Arc::clone(&self.update_installed),
        }
    }

    /// The supervisor that watches every long-lived task, wired so that a task
    /// whose death is fatal asks for exactly the drain a signal would.
    fn supervisor(&self) -> Supervisor {
        let signal = self.clone();
        Supervisor::new(Arc::clone(&self.flag), Arc::clone(&self.drain), move || {
            signal.request_now()
        })
    }

    /// Sleep for `duration`, cut short the moment shutdown is requested.
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

/// When the loop makes its first check.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FirstCheck {
    Immediately,
    AfterOneInterval,
}

/// An installed update asks for the same graceful shutdown a signal does, so
/// the recording drains before the process goes away; after an install there
/// is nothing left to check and the loop returns.
async fn run_update_check_loop_with<F, Fut, E>(
    interval_period: std::time::Duration,
    first: FirstCheck,
    shutdown: ShutdownSignal,
    check: F,
) where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<bool, E>>,
    E: std::fmt::Display,
{
    // The first tick of a tokio interval completes immediately, so delaying the
    // first check means spending that tick rather than waiting for one.
    let mut interval = tokio::time::interval(interval_period);
    if first == FirstCheck::AfterOneInterval {
        interval.tick().await;
    }
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

/// The self-updater lives in the binary and reaches this as a plain argument.
fn check_for_updates<F, Fut, E>(
    config: &Config,
    shutdown: &ShutdownSignal,
    supervisor: &Supervisor,
    _enforcement: &RestartEnforcement,
    check: F,
) where
    F: Fn(InstalledMarker) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<bool, E>> + Send + 'static,
    E: std::fmt::Display + 'static,
{
    if !config.update.enabled {
        return;
    }
    let check = Arc::new(check);
    let marker = shutdown.installed_marker();
    let shutdown = shutdown.clone();
    let limit = RestartLimit::cycling_every(UPDATE_CHECK_INTERVAL);
    let startup = Arc::new(AtomicBool::new(true));
    supervisor.restartable("update-check", limit, move || {
        let check = Arc::clone(&check);
        let marker = marker.clone();
        let first = if startup.swap(false, Ordering::Relaxed) {
            FirstCheck::Immediately
        } else {
            FirstCheck::AfterOneInterval
        };
        run_update_check_loop_with(UPDATE_CHECK_INTERVAL, first, shutdown.clone(), move || {
            check(marker.clone())
        })
    });
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
struct Storage {
    backend: Option<Arc<dyn WarmStorageBackend>>,
    scanned: bool,
}

/// `stop` is the process shutdown flag.
async fn init_storage(
    config: &Config,
    camera_ids: &[String],
    stop: crate::storage::StopFlag,
) -> Result<Storage, RunError> {
    if !config.storage.enabled {
        return Ok(Storage {
            backend: None,
            scanned: true,
        });
    }

    if let Some(stathost) = config.storage.stathost.as_ref().filter(|s| s.enabled) {
        tracing::info!(
            url = %stathost.url,
            bucket = %stathost.bucket,
            "using remote stathost warm-storage backend"
        );
        // A client that cannot be built is a permanent local fault and ends
        // startup — see [`RunError::WarmStorage`].
        let backend = StathostBackend::new(stathost, camera_ids, stop)
            .map_err(|source| RunError::WarmStorage { source })?;
        backend.recover_orphans();
        // A store that cannot be listed does not stop camon — holding startup for a host that
        // may be down for hours would cost footage. What it costs is retention: the backend
        // refuses to prune or evict until a later scan succeeds.
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
        return Ok(Storage {
            backend: Some(Arc::new(backend)),
            scanned,
        });
    }

    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    let backend = LocalDiskBackend::new(data_dir, camera_ids);
    // Orphans are recovered BEFORE the scan, so recovered events are indexed
    // like any other.
    backend.recover_orphans();
    // Infallible for local disk today, but handled so a backend that grows a
    // failure mode here cannot acquire a silent one.
    if let Err(e) = backend.scan().await {
        tracing::warn!(error = %e, "warm index scan failed; some events may not be indexed");
    }
    Ok(Storage {
        backend: Some(Arc::new(backend)),
        scanned: true,
    })
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
/// detector itself, so an entity can never exist for a class nothing looks
/// for or the other way round. No client, no detections, no entities.
fn mqtt_object_classes(client: Option<&OllamaClient>) -> Vec<String> {
    client
        .map(|c| c.allowed_classes().to_vec())
        .unwrap_or_default()
}

/// Where the bridge remembers the entity set it announced to Home Assistant.
/// Written even when warm storage is off: the entities exist either way.
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

/// The grid serves only a roughly six-segment live tail, so 60 seconds covers reconnects
/// without duplicating the long low-resolution history already provided by the main buffer.
const SUB_HOT_DURATION_SECS: u64 = 60;

struct CameraHandles {
    /// The producers, with the buffer each one fills: phase 1 joins these and
    /// then publishes each buffer's watermark.
    pipeline_handles: Vec<(String, tokio::task::JoinHandle<()>, Arc<RwLock<HotBuffer>>)>,
    /// Every worker below keeps its camera's id beside it, so a drain that
    /// abandons one can say which recording to distrust.
    analyzer_handles: Vec<(String, tokio::task::JoinHandle<()>)>,
    /// Per-camera continuous-recording drivers (storage on + analytics off);
    /// empty in event mode. Drained in phase 2, before the writers' senders
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
    sub_buffers_map: HashMap<String, Arc<RwLock<HotBuffer>>>,
}

struct SpawnContext<'a> {
    config: &'a Config,
    motion_store: &'a MotionStore,
    detection_store: &'a DetectionStore,
    /// The detector's debug view, for the analyzers to ask whether anybody is
    /// watching one before encoding frames for it.
    debug_store: &'a DetectionDebugStore,
    storage: &'a Option<Arc<dyn WarmStorageBackend>>,
    motion_settings: &'a Option<analytics::MotionSettingsStore>,
    tuner_store: &'a Option<analytics::TunerStore>,
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
    /// Every task below is spawned through this; all of them are fatal — see
    /// the policy table in [`run_with_config`].
    supervisor: &'a Supervisor,
}

/// Say what this process is going to do with its footage, once, at startup.
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

/// What a camera is expected to produce — `None` when nothing is expected of it.
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
        sub_buffers_map: HashMap::new(),
    };

    let mode = recording_mode(ctx.config);

    for cam_config in cameras {
        let sub_url = cam_config.sub_url.clone();
        let buffer = HotBuffer::new(cam_config.id.clone(), ctx.config.buffer.hot_duration_secs);
        let camera_id = cam_config.id.clone();

        if let Some(mode) = mode {
            // Seeded from what is already on disk, not from now: on a box that
            // restarts nightly, a silence that resets with the process is
            // never long enough to notice.
            let already_silent_for = storage::watchdog::silence_before_startup(
                ctx.storage
                    .as_ref()
                    .and_then(|backend| backend.newest_event_end_ns(&camera_id)),
                wall_clock_ns(),
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
                .expect("storage backend present when storage enabled; see init_storage");
            let writer = WarmWriter::new(
                rx,
                camera_id.clone(),
                &ctx.config.storage,
                backend,
                Arc::clone(ctx.recording_watchdog),
            );
            handles.warm_handles.push((
                camera_id.clone(),
                ctx.supervisor
                    .critical(format!("warm-writer:{camera_id}"), writer.run()),
            ));
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
        let handle = ctx
            .supervisor
            .critical(format!("camera:{camera_id}"), async move {
                run_camera(cam_config, buffer_clone, shutdown_clone).await;
            });
        handles
            .pipeline_handles
            .push((camera_id.clone(), handle, Arc::clone(&buffer)));

        if let Some(sub_url) = sub_url {
            let sub_id = format!("{camera_id}:sub");
            let sub_buffer = HotBuffer::new(sub_id.clone(), SUB_HOT_DURATION_SECS);
            handles
                .sub_buffers_map
                .insert(camera_id.clone(), Arc::clone(&sub_buffer));

            let sub_config = config::CameraConfig {
                id: sub_id.clone(),
                url: sub_url,
                sub_url: None,
            };
            let buffer_clone = Arc::clone(&sub_buffer);
            let shutdown_clone = ctx.shutdown.clone();
            let handle = ctx
                .supervisor
                .critical(format!("camera:{camera_id}:sub"), async move {
                    run_camera(sub_config, buffer_clone, shutdown_clone).await;
                });
            handles.pipeline_handles.push((sub_id, handle, sub_buffer));
        }

        // Continuous mode rolls fixed-length chunks straight from the hot
        // buffer into the warm writer. Gated on the same mode the watchdog is
        // told about, so the two cannot disagree.
        if let Some(RecordingMode::Continuous { chunk }) = mode {
            if let Some(tx) = event_tx.clone() {
                let recorder = run_continuous_recorder(
                    camera_id.clone(),
                    Arc::clone(&buffer),
                    tx,
                    chunk,
                    Arc::clone(&ctx.shutdown.flag),
                );
                handles.continuous_handles.push((
                    camera_id.clone(),
                    ctx.supervisor
                        .critical(format!("continuous-recorder:{camera_id}"), recorder),
                ));
            }
        }

        if ctx.config.analytics.enabled {
            let analyzer = analytics::analyzer_body(
                AnalyzerContext {
                    camera_id: camera_id.clone(),
                    buffer,
                    motion_store: ctx.motion_store.clone(),
                    detection_store: Some(ctx.detection_store.clone()),
                    debug_store: Some(ctx.debug_store.clone()),
                    detect_tx: ctx.detect_tx.clone(),
                    event_registry: ctx.event_registry.clone(),
                    config: ctx.config.analytics.clone(),
                    motion_settings: ctx
                        .motion_settings
                        .clone()
                        .expect("motion settings initialized when analytics enabled"),
                    tuner_store: ctx
                        .tuner_store
                        .clone()
                        .expect("tuner store initialized when analytics enabled"),
                    data_dir: std::path::PathBuf::from(&ctx.config.storage.data_dir),
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
            let analyzer_handle = ctx
                .supervisor
                .critical_blocking(format!("analyzer:{camera_id}"), analyzer);
            handles.analyzer_handles.push((camera_id, analyzer_handle));
        }
    }

    handles
}

/// Which of the two kinds of stop this was. `Internal` covers both an
/// installed update and a supervised task's death — which of them it was is
/// [`Supervisor::first_failure`]'s answer, not this one's.
enum ShutdownReason {
    Signal,
    Internal,
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
        _ = shutdown.wake.notified() => ShutdownReason::Internal,
    };
    shutdown.request();
    reason
}

/// Last-resort liveness backstop for a drain nobody outside the process is bounding — after
/// an installed update or a supervised task's death.
pub const RESTART_DRAIN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(360);

/// How often the watchdog looks for a reason to arm. It polls from startup
/// onwards, so this is also how long it can lag the flag going up.
const WATCHDOG_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Passed to update checks so they cannot run before the restart watchdog is armed.
struct RestartEnforcement;

/// Block until something has asked for a restart that nothing outside this
/// process is bounding. Split out from the thread because the sleep and
/// `_exit` after it are not survivable in a test.
fn wait_for_a_restart_reason(update_installed: &AtomicBool, task_died: &AtomicBool) {
    while !update_installed.load(Ordering::Relaxed) && !task_died.load(Ordering::Relaxed) {
        std::thread::sleep(WATCHDOG_POLL_INTERVAL);
    }
}

/// Starts before any restart request and watches state so requests arriving during startup or
/// drain still arm it. A plain thread and `_exit` remain effective when async work or destructors
/// wedge.
fn spawn_restart_watchdog(
    update_installed: Arc<AtomicBool>,
    task_died: Arc<AtomicBool>,
) -> RestartEnforcement {
    std::thread::spawn(move || {
        wait_for_a_restart_reason(&update_installed, &task_died);
        tracing::warn!(
            deadline_secs = RESTART_DRAIN_DEADLINE.as_secs(),
            "camon is restarting itself: the process is terminated at this deadline whether or not \
             it has finished starting up and draining by then, abandoning any recording still \
             being flushed and every queued event, so the replacement can start"
        );
        std::thread::sleep(RESTART_DRAIN_DEADLINE);
        let code = i32::from(task_died.load(Ordering::Relaxed));
        unsafe { libc::_exit(code) };
    });
    RestartEnforcement
}

/// How long shutdown waits for the MQTT bridge to publish its retained
/// `offline` marker and disconnect before giving up on it.
pub const MQTT_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Join a set of per-camera tasks under one shared deadline, so a rack of cameras cannot
/// multiply a per-task bound into a stop that outlives its budget (see [`crate::shutdown`]).
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
    // Phase 3's deadline, taken here rather than when phase 3 begins: a phase
    // that overran comes out of the writers' share, not the service manager's
    // patience.
    let budget_ends =
        tokio::time::Instant::now() + RESTART_DRAIN_DEADLINE - crate::shutdown::TEARDOWN_MARGIN;

    // PHASE 1 — the producers stop, completely, before anything downstream is asked to
    // finish.
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
        // PHASE 2 opens here: the watermark is published after the join, so
        // no push can land behind it. Unconditional — a consumer waiting on a
        // watermark that never comes waits out its whole bound for nothing.
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

    // PHASE 2 — analyzers and continuous recorders drain through their camera's watermark, so
    // the tail the camera pushed on its way out is part of what they flush.
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

    // The detection worker is aborted, not drained: losing a queued job or an
    // in-flight Ollama request costs only an object upgrade, never footage.
    // Aborting also releases its warm-writer senders for the drain below.
    if let Some(handle) = detect_worker_handle {
        handle.abort();
        let _ = handle.await;
    }

    // Joined after the analyzers flushed their final MotionEnd: the bridge stops only once the
    // producers drop their senders (see `mqtt::bridge_is_done`), and joining here lets it
    // publish that last transition and the retained `offline` marker.
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

    // PHASE 3 — with all senders gone the warm writers drain their queues and exit; awaiting
    // them puts every accepted event on disk.
    drop(handles.event_senders);
    drop(handles.event_sender_map);
    join_all_before(handles.warm_handles, budget_ends, "warm writer").await;

    // The retention sweep, joined last and inside phase 3's deadline — so a sweep parked on a
    // remote request timeout spends the writers' remainder of the budget instead of phase 2's
    // gate.
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

/// Run camon: load the configuration, bring the cameras up, serve the API, and drain everything
/// again when a signal or an installed update asks for it.
pub async fn run<F, Fut, E>(check_update: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(InstalledMarker) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<bool, E>> + Send + 'static,
    E: std::fmt::Display + 'static,
{
    // A healthy production log is empty (warn and up); dev builds keep the
    // debug stream, and RUST_LOG overrides both.
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

    // One line the operator can act on and a nonzero status for the service
    // manager; exiting here keeps `main` from printing the `Debug` form on
    // top of it. By this point the drain, if there was one, is over.
    if let Err(e) = run_with_config(config, check_update).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    Ok(())
}

/// Everything after the configuration has been read, which is everything that can be tested:
/// the same startup, task graph and drain, minus the logging setup and argument parsing a
/// process only does once.
async fn run_with_config<F, Fut, E>(config: Config, check_update: F) -> Result<(), RunError>
where
    F: Fn(InstalledMarker) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<bool, E>> + Send + 'static,
    E: std::fmt::Display + 'static,
{
    // Created before the update check so the periodic checker can ask for the
    // same graceful shutdown a signal does.
    let shutdown = ShutdownSignal::new();
    let supervisor = shutdown.supervisor();

    // The listening socket is taken here, synchronously, before anything is
    // brought up behind it — see [`api::bind`].
    let http_addr = std::net::SocketAddr::new(config.http.bind_addr(), config.http.port);
    let listener = api::bind(http_addr)
        .await
        .map_err(|source| RunError::Bind {
            addr: http_addr,
            source,
        })?;
    // Decided once, before anything is served: what the API asks of a
    // request, generating and persisting a token if the deployment would
    // otherwise be open.
    let api_auth = api::ApiAuth::resolve(
        http_addr.ip(),
        config.http.token.as_deref(),
        config.http.allow_open,
        config.token_file_path().as_deref(),
    )
    .map_err(|source| RunError::ApiToken { source })?;

    // Must precede the update check: from the moment the check runs, an
    // update can install into a process still starting up, and the restart
    // must be guaranteed even if startup never finishes.
    let enforcement =
        spawn_restart_watchdog(Arc::clone(&shutdown.update_installed), supervisor.died());
    // Spawned, not awaited: nothing below waits for a version check. See
    // [`check_for_updates`].
    check_for_updates(&config, &shutdown, &supervisor, &enforcement, check_update);

    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let camera_ids: Vec<String> = config.cameras.iter().map(|c| c.id.clone()).collect();
    let motion_store = MotionStore::new(&camera_ids);
    let detection_store = DetectionStore::new(&camera_ids);
    let debug_store = DetectionDebugStore::new(&camera_ids);
    let object_detection_ready = log_object_detection_config(&config);
    let Storage {
        backend: storage,
        scanned: warm_index_scanned,
    } = init_storage(
        &config,
        &camera_ids,
        crate::storage::StopFlag::shared(Arc::clone(&shutdown.flag)),
    )
    .await?;
    // The recording-silence watchdog cannot see the storage volume being
    // unmounted — writes to the bare mountpoint succeed and keep resetting it
    // — so the local-disk backend watches its volume directly.
    let volume_anchor = storage
        .as_ref()
        .and_then(|backend| backend.volume_anchor().cloned());
    let motion_settings = init_motion_settings(&config, &camera_ids);
    let tuner_store = config.analytics.enabled.then(|| {
        analytics::TunerStore::with_params(
            &camera_ids,
            analytics::TunerParams::from(&config.analytics.motion),
        )
    });

    log_recording_mode(&config);

    // Object detection runs on ONE global worker: at most one in-flight
    // Ollama request across all cameras (the GPU degrades badly under
    // parallel load).
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
            let (tx, queue) = detect_queue(event_registry.clone());
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
        debug_store: &debug_store,
        storage: &storage,
        motion_settings: &motion_settings,
        tuner_store: &tuner_store,
        detect_tx: &detect_tx,
        event_registry: &event_registry,
        mqtt_tx: &mqtt_tx,
        recording_watchdog: &recording_watchdog,
        shutdown: &shutdown,
        supervisor: &supervisor,
    };
    let camera_handles = spawn_cameras(&spawn_ctx, config.cameras.clone());

    // Retention is a property of the store, not of a camera: one task sweeps
    // every camera on a schedule, however many writers there are.
    let retention_handle = storage.as_ref().map(|backend| {
        let backend = Arc::clone(backend);
        let warm_config = config.storage.clone();
        let flag = Arc::clone(&shutdown.flag);
        let limit = RestartLimit::cycling_every(crate::buffer::warm::PRUNE_INTERVAL);
        supervisor.restartable("retention", limit, move || {
            let task = RetentionTask::new(Arc::clone(&backend), &warm_config, Arc::clone(&flag));
            // A backend that never got its index is waiting for a sweep to retry the scan, so
            // it does not wait the usual hour for one. A restarted sweep asks for the same head
            // start: whatever killed the last one, the index is no fresher than it was.
            let task = if warm_index_scanned {
                task
            } else {
                task.after_a_failed_scan()
            };
            task.run()
        })
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
            Some(supervisor.critical("detection-worker", worker.run(rx)))
        }
        _ => None,
    };

    let mqtt_handle = mqtt_rx.map(|rx| {
        let ctx = BridgeContext {
            config: config.mqtt.clone(),
            buffers: Arc::new(camera_handles.buffers_map.clone()),
            camera_ids: camera_ids.clone(),
            classes: object_classes,
            entities_path: Some(mqtt_entities_path(&config)),
            shutdown: Arc::clone(&shutdown.flag),
        };
        supervisor.critical("mqtt-bridge", mqtt::run_bridge(ctx, rx))
    });
    // The analyzers and detection worker hold their own clones; dropping this
    // one lets the bridge see the channel close once they are gone.
    drop(mqtt_tx);
    // Analyzers hold their own clones; dropping the original closes the job
    // channel once they exit, letting the worker finish in normal operation.
    drop(detect_tx);

    let app_state = AppState::new(
        camera_handles.buffers_map.clone(),
        camera_handles.sub_buffers_map.clone(),
        motion_store,
        detection_store,
        debug_store,
        storage,
        motion_settings,
    )
    .with_tuner_store(tuner_store);
    // Serving on the socket startup already took.
    let server_handle = supervisor.critical("http-server", async move {
        if let Err(e) = api::serve(listener, app_state, api_auth).await {
            tracing::error!(error = %e, "the API server stopped serving");
        }
    });

    // No camera is registered when nothing is expected to record, and an empty
    // watchdog has nothing to poll.
    let watchdog_handle = recording_mode(&config).map(|_| {
        let watchdog = Arc::clone(&recording_watchdog);
        let limit = RestartLimit::cycling_every(storage::watchdog::POLL_INTERVAL);
        supervisor.restartable("recording-watchdog", limit, move || {
            Arc::clone(&watchdog).run()
        })
    });

    let anchor_handle = volume_anchor.map(|anchor| {
        let limit = RestartLimit::cycling_every(storage::anchor::POLL_INTERVAL);
        supervisor.restartable("storage-anchor", limit, move || Arc::clone(&anchor).run())
    });

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

    let RestartEnforcement = enforcement;
    graceful_shutdown(
        camera_handles,
        retention_handle,
        detect_worker_handle,
        mqtt_handle,
    )
    .await;

    stop_outcome(reason, supervisor.deaths(), supervisor.first_failure())
}

/// What the process leaves behind once the drain is over.
fn stop_outcome(
    reason: ShutdownReason,
    deaths: Vec<String>,
    decided_by: Option<crate::supervise::Death>,
) -> Result<(), RunError> {
    if !deaths.is_empty() {
        tracing::error!(
            tasks = %deaths.join(", "),
            decided_by = %decided_by.map(|d| d.task).unwrap_or_default(),
            "drain complete after a supervised task died; exiting nonzero"
        );
        return Err(RunError::TaskDied { tasks: deaths });
    }
    match reason {
        ShutdownReason::Signal => tracing::info!("shutdown complete"),
        ShutdownReason::Internal => {
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

    #[tokio::test]
    async fn a_warm_backend_that_cannot_be_built_ends_startup_instead_of_running_storageless() {
        let remote: Config = toml::from_str(
            r#"
            [storage.stathost]
            url = "https://files.example.com"
            bucket = "cams"
            token = "secret"
            "#,
        )
        .unwrap();
        let cameras = ["cam".to_string()];

        crate::storage::stathost::force_client_build_failure(true);
        let refused = init_storage(&remote, &cameras, crate::storage::StopFlag::never()).await;
        crate::storage::stathost::force_client_build_failure(false);

        match refused {
            Err(RunError::WarmStorage { source }) => {
                let message = format!("{}", RunError::WarmStorage { source });
                assert!(message.contains("warm storage"), "{message}");
                assert!(message.contains("HTTP client"), "{message}");
                assert!(message.contains("root certificate"), "{message}");
                assert!(message.contains("proxy"), "{message}");
            }
            Err(other) => panic!("the wrong startup failure: {other}"),
            Ok(storage) => panic!(
                "startup carried on with backend present = {}, and every camera's writer \
                 will unwrap it",
                storage.backend.is_some()
            ),
        }

        let local = tempfile::tempdir().unwrap();
        let on: Config = toml::from_str(&format!(
            "[storage]\ndata_dir = {:?}\n",
            local.path().display()
        ))
        .unwrap();
        let started = init_storage(&on, &cameras, crate::storage::StopFlag::never())
            .await
            .expect("local disk storage failed to start");
        assert!(
            started.backend.is_some(),
            "enabled storage without a backend"
        );

        let off: Config = toml::from_str("[storage]\nenabled = false\n").unwrap();
        let disabled = init_storage(&off, &cameras, crate::storage::StopFlag::never())
            .await
            .expect("disabled storage is not a failure");
        assert!(disabled.backend.is_none());
    }

    #[test]
    fn backoff_progression_doubles_then_caps() {
        assert_eq!(next_backoff_secs(5), 10);
        assert_eq!(next_backoff_secs(10), 20);
        assert_eq!(next_backoff_secs(20), 40);
        assert_eq!(next_backoff_secs(40), 60);
        assert_eq!(next_backoff_secs(60), 60);
    }

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
        run_update_check_loop_with(
            Duration::from_millis(1),
            FirstCheck::Immediately,
            shutdown.clone(),
            check,
        )
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 3, "loop stopped checking");
        assert!(
            !shutdown.update_installed.load(Ordering::Relaxed),
            "a failed check asked for a restart"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn check_for_updates_returns_without_waiting_for_an_answer() {
        let config = config_from(&format!("[update]\nenabled = true\n{ONE_CAMERA}"));
        let shutdown = ShutdownSignal::new();
        let supervisor = shutdown.supervisor();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let check = move |_: InstalledMarker| {
            counter.fetch_add(1, Ordering::Relaxed);
            std::future::pending::<Result<bool, std::io::Error>>()
        };

        let past_the_check = tokio::spawn(async move {
            let enforcement =
                spawn_restart_watchdog(Arc::clone(&shutdown.update_installed), supervisor.died());
            check_for_updates(&config, &shutdown, &supervisor, &enforcement, check)
        });
        let () = tokio::time::timeout(Duration::from_secs(600), past_the_check)
            .await
            .expect("the call waited for a version check that never came back")
            .expect("it panicked instead of returning");

        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "the call returned without ever starting the check"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_first_check_of_the_startup_loop_is_immediate() {
        let shutdown = ShutdownSignal::new();
        let started = tokio::time::Instant::now();
        let when = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&when);
        let signalled = shutdown.clone();
        let check = move || {
            let mut seen = observed.lock().expect("checker poisoned");
            seen.push(started.elapsed());
            if seen.len() == 2 {
                signalled.flag.store(true, Ordering::Relaxed);
            }
            std::future::ready(Ok::<bool, std::io::Error>(false))
        };

        run_update_check_loop_with(
            UPDATE_CHECK_INTERVAL,
            FirstCheck::Immediately,
            shutdown.clone(),
            check,
        )
        .await;

        assert_eq!(
            *when.lock().expect("checker poisoned"),
            vec![Duration::ZERO, UPDATE_CHECK_INTERVAL],
            "the first check was not immediate, or the second did not wait its turn"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_restarted_update_check_waits_a_cadence_before_checking() {
        let config = config_from(&format!("[update]\nenabled = true\n{ONE_CAMERA}"));
        let shutdown = ShutdownSignal::new();
        let supervisor = shutdown.supervisor();
        let started = tokio::time::Instant::now();
        let when = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&when);
        let check = move |_: InstalledMarker| {
            let mut seen = observed.lock().expect("checker poisoned");
            seen.push(started.elapsed());
            let first = seen.len() == 1;
            drop(seen);
            async move {
                assert!(!first, "the first attempt died, as this test intends");
                Ok::<bool, std::io::Error>(false)
            }
        };
        let enforcement =
            spawn_restart_watchdog(Arc::clone(&shutdown.update_installed), supervisor.died());
        check_for_updates(&config, &shutdown, &supervisor, &enforcement, check);

        tokio::time::sleep(UPDATE_CHECK_INTERVAL * 2).await;

        let seen = when.lock().expect("checker poisoned");
        assert_eq!(seen[0], Duration::ZERO, "the startup check waited");
        assert!(
            seen[1] >= UPDATE_CHECK_INTERVAL,
            "a restarted attempt checked after only {:?}, so its next death would read as an \
             attempt that never worked",
            seen[1]
        );
    }

    #[test]
    fn the_restart_enforcement_arms_without_a_drain_to_start_it() {
        let update_installed = Arc::new(AtomicBool::new(false));
        let task_died = Arc::new(AtomicBool::new(false));
        let armed = Arc::new(AtomicBool::new(false));

        let (installed, died, reached) = (
            Arc::clone(&update_installed),
            Arc::clone(&task_died),
            Arc::clone(&armed),
        );
        std::thread::spawn(move || {
            wait_for_a_restart_reason(&installed, &died);
            reached.store(true, Ordering::Relaxed);
        });

        std::thread::sleep(WATCHDOG_POLL_INTERVAL * 2);
        assert!(
            !armed.load(Ordering::Relaxed),
            "the enforcement armed with nothing to enforce"
        );

        update_installed.store(true, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !armed.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            armed.load(Ordering::Relaxed),
            "an installed update did not arm the enforcement on its own"
        );
    }

    #[tokio::test]
    async fn update_check_is_skipped_once_shutdown_is_requested() {
        let shutdown = ShutdownSignal::new();
        shutdown.flag.store(true, Ordering::Relaxed); // as a signal would
        let (check, calls) = scripted_checker(vec![Ok(false)]);
        run_update_check_loop_with(
            Duration::from_millis(1),
            FirstCheck::Immediately,
            shutdown.clone(),
            check,
        )
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 0, "checked during shutdown");
    }

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
            sub_buffers_map: HashMap::new(),
        };
        tokio::time::timeout(
            Duration::from_secs(30),
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("graceful_shutdown deadlocked instead of draining");

        let written = dir.path().join("cam").join("movements").join("0_1000.ts");
        assert!(written.exists(), "queued event was not flushed to disk");
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

    fn empty_handles() -> CameraHandles {
        CameraHandles {
            pipeline_handles: Vec::new(),
            analyzer_handles: Vec::new(),
            continuous_handles: Vec::new(),
            warm_handles: Vec::new(),
            event_senders: Vec::new(),
            event_sender_map: HashMap::new(),
            buffers_map: HashMap::new(),
            sub_buffers_map: HashMap::new(),
        }
    }

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
        assert!(
            started.elapsed() < crate::shutdown::CAMERA_JOIN_BOUND,
            "the drain waited out a bound instead of following the watermark"
        );
    }

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
        assert_eq!(
            buffer.read_recover().terminal_watermark(),
            Some(crate::shutdown::Watermark {
                sequence: 1,
                provisional: true,
            }),
            "the camera that hung got no watermark, or one claiming it had stopped"
        );
    }

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

    #[tokio::test(start_paused = true)]
    async fn a_slow_retention_sweep_does_not_hold_up_the_camera_joins() {
        let buffer = HotBuffer::new("cam".to_string(), 3600);
        buffer.write_recover().push(gop(0));

        let watermark_was_out = Arc::new(AtomicBool::new(false));
        let probe = Arc::clone(&watermark_was_out);
        let watched = Arc::clone(&buffer);
        let retention = tokio::spawn(async move {
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

    #[tokio::test]
    async fn a_bind_that_fails_ends_startup_before_anything_is_brought_up() {
        let held = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let taken = held.local_addr().unwrap().port();
        let data_dir = tempfile::tempdir().unwrap();
        let config = config_from(&format!(
            "[http]\nbind = \"127.0.0.1\"\nport = {taken}\n\n\
             [update]\nenabled = true\n\n\
             [storage]\nenabled = true\ndata_dir = {:?}\n{ONE_CAMERA}",
            data_dir.path()
        ));
        let (check, calls) = scripted_checker(vec![Ok(false)]);

        let error = run_with_config(config, move |_: InstalledMarker| check())
            .await
            .expect_err("startup carried on without the socket it is there to serve");

        assert!(
            matches!(error, RunError::Bind { addr, .. } if addr.port() == taken),
            "{error}"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "startup got past the bind and ran the update check"
        );
        assert_eq!(
            std::fs::read_dir(data_dir.path()).unwrap().count(),
            0,
            "storage was brought up behind a socket camon never took"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_supervised_death_drains_the_footage_and_exits_nonzero() {
        let dir = tempfile::tempdir().unwrap();
        let shutdown = ShutdownSignal::new();
        let supervisor = shutdown.supervisor();
        let mut handles = empty_handles();
        add_continuous_camera(
            &mut handles,
            dir.path(),
            "cam",
            &shutdown,
            records_two_seconds_then_a_tail,
        );

        supervisor.critical("warm-writer:cam", async { panic!("nothing to write with") });

        let reason = tokio::time::timeout(RESTART_DRAIN_DEADLINE, wait_for_shutdown(&shutdown))
            .await
            .expect("a supervised death never woke the drain");
        spawn_restart_watchdog(Arc::clone(&shutdown.update_installed), supervisor.died());
        tokio::time::timeout(
            RESTART_DRAIN_DEADLINE,
            graceful_shutdown(handles, None, None, None),
        )
        .await
        .expect("the drain never finished");

        assert!(
            continuous_file(dir.path(), "cam", "0_3000").exists(),
            "a supervised death cost the recording its tail"
        );
        assert!(
            supervisor.died().load(Ordering::Relaxed),
            "nothing would have bounded a drain no service manager is watching"
        );
        let outcome = stop_outcome(reason, supervisor.deaths(), supervisor.first_failure());
        assert!(
            matches!(
                outcome,
                Err(RunError::TaskDied { ref tasks }) if tasks == &["warm-writer:cam (panicked)"]
            ),
            "a death exited as if camon had been asked to stop"
        );
    }

    #[test]
    fn a_stop_that_was_asked_for_exits_zero() {
        assert!(stop_outcome(ShutdownReason::Signal, Vec::new(), None).is_ok());
        assert!(stop_outcome(ShutdownReason::Internal, Vec::new(), None).is_ok());
    }

    #[test]
    fn every_restartable_task_outlives_its_own_cadence_before_it_counts_as_healthy() {
        for (task, cadence) in [
            ("retention", crate::buffer::warm::PRUNE_INTERVAL),
            ("recording-watchdog", storage::watchdog::POLL_INTERVAL),
            ("storage-anchor", storage::anchor::POLL_INTERVAL),
            ("update-check", UPDATE_CHECK_INTERVAL),
        ] {
            let limit = RestartLimit::cycling_every(cadence);
            assert!(
                limit.healthy_after > cadence,
                "{task} can clear its streak by surviving a single {cadence:?} cycle"
            );
            assert_eq!(limit.max, crate::supervise::PERIODIC_RESTARTS);
        }
    }

    #[test]
    fn the_exit_message_names_every_task_that_died() {
        let deaths = vec![
            "detection-worker (returned)".to_string(),
            "analyzer:yard (panicked)".to_string(),
        ];
        let error = stop_outcome(ShutdownReason::Internal, deaths, None)
            .expect_err("a cascade of deaths exited clean");

        let message = error.to_string();
        assert!(message.contains("analyzer:yard"), "{message}");
        assert!(message.contains("detection-worker"), "{message}");
    }

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

    const UPDATE_LOOP_MARKER_ENV: &str = "CAMON_TEST_UPDATE_LOOP_MARKER";

    #[test]
    fn applied_update_returns_instead_of_exiting() {
        if let Ok(marker) = std::env::var(UPDATE_LOOP_MARKER_ENV) {
            let shutdown = ShutdownSignal::new();
            let supervisor = shutdown.supervisor();
            let (check, calls) = scripted_checker(vec![Ok(true)]);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let check = Arc::new(check);
                let signal = shutdown.clone();
                supervisor
                    .restartable(
                        "update-check",
                        RestartLimit::cycling_every(Duration::from_millis(1)),
                        move || {
                            let check = Arc::clone(&check);
                            run_update_check_loop_with(
                                Duration::from_millis(1),
                                FirstCheck::Immediately,
                                signal.clone(),
                                move || check(),
                            )
                        },
                    )
                    .await
                    .expect("the supervised update loop panicked");
            });

            assert_eq!(calls.load(Ordering::Relaxed), 1, "update installed twice");
            assert_eq!(
                supervisor.deaths(),
                Vec::<String>::new(),
                "an installed update looked like a supervised task death"
            );
            assert!(shutdown.requested(), "drain flag not raised");
            assert!(
                shutdown.update_installed.load(Ordering::Relaxed),
                "watchdog would never arm"
            );
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

    #[test]
    fn no_object_classes_reach_the_bridge_when_nothing_detects() {
        for flags in [
            "[analytics]\nenabled = true\n\n[analytics.object_detection]\nenabled = false",
            "[analytics]\nenabled = false\n\n[analytics.object_detection]\nenabled = true",
        ] {
            let config = config_from(&format!("{flags}\n{ONE_CAMERA}"));
            assert!(!log_object_detection_config(&config));
            assert!(mqtt_object_classes(None).is_empty(), "with {flags:?}");
        }
    }
}
