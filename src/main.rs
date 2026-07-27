use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tracing_subscriber::EnvFilter;

mod analytics;
mod api;
mod buffer;
mod camera;
mod config;
mod install;
mod locks;
mod mpegts;
mod mqtt;
mod storage;
mod update;

use analytics::{detect_queue, AnalyzerContext, DetectQueueSender, DetectionWorker, OllamaClient};
use api::AppState;
use buffer::warm::{run_continuous_recorder, RetentionTask, WarmWriter, WriterMessage};
use buffer::HotBuffer;
use camera::FfmpegPipeline;
use config::Config;
use locks::LockExt;
use mqtt::{BridgeContext, MqttEvent, MQTT_EVENT_CAPACITY};
use storage::{
    DetectionDebugStore, DetectionStore, EventRegistry, LocalDiskBackend, MotionStore,
    StathostBackend, WarmStorageBackend,
};

fn dispatch_subcommand() -> bool {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Nothing to dispatch, or the first argument is a flag (e.g. `--config`) —
    // leave it to normal startup / `parse_cli_args`.
    match args.first() {
        None => return false,
        Some(first) if first.starts_with('-') => return false,
        _ => {}
    }

    match args[0].as_str() {
        "install" => {
            if args.get(1).map(|s| s.as_str()) == Some("service") {
                if let Err(e) = install::install_service() {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
                std::process::exit(0);
            }
            eprintln!("usage: camon install service");
            std::process::exit(1);
        }
        other => {
            eprintln!("unknown command: {other}");
            eprintln!(
                "usage: camon [--config <path>] [--set <dotted.path>=<value>]... [install service]"
            );
            std::process::exit(1);
        }
    }
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

async fn run_update_check_loop(shutdown: ShutdownSignal) {
    run_update_check_loop_with(UPDATE_CHECK_INTERVAL, shutdown, update::check_and_update).await;
}

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

async fn check_for_updates(config: &Config, shutdown: &ShutdownSignal) {
    if !config.update.enabled {
        return;
    }
    match update::check_and_update().await {
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
    tokio::spawn(run_update_check_loop(shutdown.clone()));
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

async fn init_storage(
    config: &Config,
    camera_ids: &[String],
) -> Option<Arc<dyn WarmStorageBackend>> {
    if !config.storage.enabled {
        return None;
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
        backend.scan().await;
        return Some(Arc::new(backend));
    }

    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    let backend = LocalDiskBackend::new(data_dir, camera_ids);
    // Salvage any event files orphaned mid-write by a crash or power cut
    // BEFORE the scan, so recovered events are indexed like any other.
    backend.recover_orphans();
    backend.scan().await;
    Some(Arc::new(backend))
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
    pipeline_handles: Vec<(String, tokio::task::JoinHandle<()>, Arc<RwLock<HotBuffer>>)>,
    analyzer_handles: Vec<tokio::task::JoinHandle<()>>,
    /// Per-camera continuous-recording drivers (storage on + analytics off).
    /// Empty in event mode. Flushed at shutdown before the writers' senders drop.
    continuous_handles: Vec<tokio::task::JoinHandle<()>>,
    warm_handles: Vec<tokio::task::JoinHandle<()>>,
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
    shutdown: &'a ShutdownSignal,
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

    for cam_config in cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), ctx.config.buffer.hot_duration_secs);
        let camera_id = cam_config.id.clone();

        let event_tx = if ctx.config.storage.enabled {
            let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
            let backend = ctx
                .storage
                .clone()
                .expect("storage backend present when storage enabled");
            let writer = WarmWriter::new(rx, camera_id.clone(), &ctx.config.storage, backend);
            handles.warm_handles.push(tokio::spawn(writer.run()));
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
        // from the hot buffer into the same warm writer.
        if ctx.config.storage.enabled && !ctx.config.analytics.enabled {
            if let Some(tx) = event_tx.clone() {
                let recorder = run_continuous_recorder(
                    camera_id.clone(),
                    Arc::clone(&buffer),
                    tx,
                    std::time::Duration::from_secs(ctx.config.storage.max_event_duration_secs),
                    Arc::clone(&ctx.shutdown.flag),
                );
                handles.continuous_handles.push(tokio::spawn(recorder));
            }
        }

        if ctx.config.analytics.enabled {
            let analyzer_handle = analytics::spawn_analyzer(
                AnalyzerContext {
                    camera_id,
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
            handles.analyzer_handles.push(analyzer_handle);
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
/// manager (`TimeoutStopSec` / OpenRC's `retry`, both set by
/// `install::install_service`), an internally requested one is not.
///
/// It guarantees the restart *eventually* happens; it does not guarantee that
/// what it terminates was stuck. No honest value could: against a black-holing
/// stathost server a single event legitimately takes longer than any deadline
/// worth having — the video is put with one retry (2 x `UPLOAD_TIMEOUT`),
/// then the sidecar, then one put per filmstrip frame, all serial — and a
/// writer queue can hold several events. So this can abandon a drain that is
/// still making progress, losing the remainder. That is the accepted trade:
/// the alternative is an NVR that silently never restarts.
const RESTART_DRAIN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(360);

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
const MQTT_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

async fn graceful_shutdown(
    handles: CameraHandles,
    retention_handle: Option<tokio::task::JoinHandle<()>>,
    detect_worker_handle: Option<tokio::task::JoinHandle<()>>,
    mqtt_handle: Option<tokio::task::JoinHandle<()>>,
) {
    // Joined, not aborted: an abort mid-delete could strip an event's .ts and
    // leave its sidecar and thumbnails behind, where the startup scan — which
    // only looks at .ts files — would never see them again. That is the exact
    // silent leak this drain is not allowed to create, so the sweep is asked
    // to stop instead: it polls the shutdown flag between events, and the wait
    // is bounded by one event's deletes. Waiting here first costs nothing —
    // the flag was raised before the drain began, so every writer, analyzer
    // and recorder is already winding down concurrently with this await.
    if let Some(handle) = retention_handle {
        let _ = handle.await;
    }

    // Analyzers poll the shutdown flag every ~200ms and flush any open motion
    // run as a complete event before exiting — join them, never abort.
    for handle in handles.analyzer_handles {
        let _ = handle.await;
    }

    // Continuous recorders watch the same shutdown flag; each flushes its
    // partial chunk to the writer and exits. Awaited here — before the senders
    // are dropped below — so the final chunk is guaranteed accepted.
    for handle in handles.continuous_handles {
        let _ = handle.await;
    }

    // The detection worker is aborted, not drained: queued jobs and even an
    // in-flight Ollama request (up to 90s) are droppable by design — losing
    // one costs only an object upgrade, never footage. Aborting also releases
    // the worker's warm-writer senders so the writers below can drain.
    if let Some(handle) = detect_worker_handle {
        handle.abort();
        let _ = handle.await;
    }

    // Joined here — after the analyzers flushed their final MotionEnd, before
    // the buffers and writers go away — so the bridge can still reflect that
    // last transition and publish the retained `offline` marker. A broker that
    // has become unreachable must not hold shutdown up, hence the timeout.
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

    let mut buffers_with_ids = Vec::new();
    for (camera_id, handle, buffer) in handles.pipeline_handles {
        let _ = handle.await;
        tracing::info!(camera = %camera_id, "camera pipeline stopped");
        buffers_with_ids.push((camera_id, buffer));
    }

    // With all senders gone the warm writers drain their queues and exit;
    // awaiting them guarantees every accepted event reached disk. BOTH sender
    // holders must drop — the map's clones alone keep the channels open, which
    // deadlocked shutdown here until 2026-07-24.
    drop(handles.event_senders);
    drop(handles.event_sender_map);
    for handle in handles.warm_handles {
        let _ = handle.await;
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dispatch_subcommand();

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
    check_for_updates(&config, &shutdown).await;

    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let camera_ids: Vec<String> = config.cameras.iter().map(|c| c.id.clone()).collect();
    let motion_store = MotionStore::new(&camera_ids);
    let detection_store = DetectionStore::new(&camera_ids);
    let debug_store = DetectionDebugStore::new(&camera_ids);
    let object_detection_ready = log_object_detection_config(&config);
    let storage = init_storage(&config, &camera_ids).await;
    let motion_settings = init_motion_settings(&config, &camera_ids);

    if config.storage.enabled {
        if config.analytics.enabled {
            tracing::info!("event recording mode: motion and object events saved to disk");
        } else {
            tracing::info!(
                retention_days = config.storage.continuous_retention_days,
                "continuous recording mode (analytics disabled): every segment saved to \
                 continuous/, roughly 43 GB/day/camera at 4 Mbps"
            );
        }
    }

    // Object detection runs on ONE global worker task with a small bounded
    // job queue — strictly serial, at most one in-flight Ollama request
    // across all cameras (the GPU degrades badly under parallel load).
    let ollama_client = if object_detection_ready {
        create_ollama_client(&config)
    } else {
        None
    };
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

    let spawn_ctx = SpawnContext {
        config: &config,
        motion_store: &motion_store,
        detection_store: &detection_store,
        storage: &storage,
        motion_settings: &motion_settings,
        detect_tx: &detect_tx,
        event_registry: &event_registry,
        mqtt_tx: &mqtt_tx,
        shutdown: &shutdown,
    };
    let camera_handles = spawn_cameras(&spawn_ctx, config.cameras.clone());

    // Retention is a property of the store, not of a camera: one task sweeps
    // every camera on a schedule, however many writers there are.
    let retention_handle = storage.as_ref().map(|backend| {
        tokio::spawn(
            RetentionTask::new(
                Arc::clone(backend),
                &config.storage,
                Arc::clone(&shutdown.flag),
            )
            .run(),
        )
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
        // Occupancy sensors only exist for classes the model is actually asked
        // about; with object detection off there are none.
        let classes = if config.analytics.enabled && config.analytics.object_detection.enabled {
            config.analytics.object_detection.classes.clone()
        } else {
            Vec::new()
        };
        mqtt::spawn_bridge(
            BridgeContext {
                config: config.mqtt.clone(),
                buffers: Arc::new(camera_handles.buffers_map.clone()),
                camera_ids: camera_ids.clone(),
                classes,
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

    let reason = wait_for_shutdown(&shutdown).await;
    server_handle.abort();

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

/// Apply +/-20% jitter to a delay (in ms) using an externally supplied random value.
fn apply_jitter(base_ms: u64, rand: u64) -> u64 {
    if base_ms == 0 {
        return 0;
    }
    let span = base_ms / 5; // 20%
    let offset = (rand % (2 * span + 1)) as i64 - span as i64;
    (base_ms as i64 + offset).max(0) as u64
}

/// A random-ish u64 from std only (RandomState is seeded per construction).
fn jitter_source() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish()
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

    while !shutdown.requested() {
        tracing::info!(camera = %camera_id, url = %config.redacted_url(), "connecting to camera");

        let pipeline = FfmpegPipeline::new(&config, Arc::clone(&buffer));
        let shutdown_ref = Arc::clone(&shutdown.flag);

        let started = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || pipeline.run(&shutdown_ref)).await;
        let ran_for = started.elapsed();

        match result {
            Ok(Ok(())) => {
                tracing::info!(camera = %camera_id, "pipeline stopped normally");
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

    #[test]
    fn jitter_stays_within_twenty_percent() {
        let base_ms = 20_000;
        let span = base_ms / 5;
        for rand in [0u64, 1, 7, 12345, u64::MAX] {
            let out = apply_jitter(base_ms, rand);
            assert!(out >= base_ms - span, "{out} below lower bound");
            assert!(out <= base_ms + span, "{out} above upper bound");
        }
    }

    #[test]
    fn jitter_zero_base_is_zero() {
        assert_eq!(apply_jitter(0, 999), 0);
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

    /// A stand-in for `update::check_and_update` that counts its calls and
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

    /// Both shutdown paths end in this same drain, and its contract is that an
    /// event already accepted by a warm writer reaches disk before the process
    /// goes away — the footage the update path used to lose.
    #[tokio::test]
    async fn graceful_shutdown_drains_queued_events_to_disk() {
        use buffer::warm::FinishedEvent;
        use buffer::GopSegment;

        let dir = tempfile::tempdir().unwrap();
        let backend: Arc<dyn WarmStorageBackend> = Arc::new(LocalDiskBackend::new(
            dir.path().to_path_buf(),
            &["cam".to_string()],
        ));
        let warm_config = config::WarmConfig::default();
        let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let writer = WarmWriter::new(rx, "cam".to_string(), &warm_config, backend);

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
            warm_handles: vec![tokio::spawn(writer.run())],
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
                "tests::applied_update_returns_instead_of_exiting",
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
}
