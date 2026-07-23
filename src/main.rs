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
mod storage;
mod update;

use analytics::{AnalyzerContext, OllamaDetector};
use api::AppState;
use buffer::warm::{run_continuous_recorder, FinishedEvent, WarmWriter};
use buffer::HotBuffer;
use camera::FfmpegPipeline;
use config::Config;
use locks::LockExt;
use storage::{DetectionDebugStore, DetectionStore, MotionStore, WarmEventIndex};

fn dispatch_subcommand() -> bool {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return false;
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
            eprintln!("usage: camon [install service]");
            std::process::exit(1);
        }
    }
}

async fn run_update_check_loop() {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(12 * 60 * 60));
    interval.tick().await; // skip immediate tick
    loop {
        interval.tick().await;
        match update::check_and_update().await {
            Ok(true) => {
                tracing::info!("update applied, exiting for restart");
                std::process::exit(0);
            }
            Ok(false) => {}
            Err(e) => {
                tracing::warn!(error = %e, "periodic update check failed");
            }
        }
    }
}

async fn check_for_updates(config: &Config) {
    if !config.update.enabled {
        return;
    }
    match update::check_and_update().await {
        Ok(true) => {
            tracing::info!("update applied, exiting for restart");
            std::process::exit(0);
        }
        Ok(false) => {}
        Err(e) => {
            tracing::warn!(error = %e, "update check failed, continuing startup");
        }
    }
    tokio::spawn(run_update_check_loop());
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

fn init_warm_index(config: &Config, camera_ids: &[String]) -> Option<WarmEventIndex> {
    if !config.storage.enabled {
        return None;
    }
    let data_dir = std::path::PathBuf::from(&config.storage.data_dir);
    // Salvage any event files orphaned mid-write by a crash or power cut
    // BEFORE the scan, so recovered events are indexed like any other.
    storage::recover_orphans(&data_dir, camera_ids);
    let index = WarmEventIndex::new(camera_ids, data_dir);
    index.scan();
    Some(index)
}

fn init_detection_grid(
    config: &Config,
    camera_ids: &[String],
) -> Option<analytics::detection_grid::DetectionGrid> {
    if !config.analytics.enabled || !config.analytics.object_detection.enabled {
        return None;
    }
    Some(analytics::detection_grid::DetectionGrid::new(
        camera_ids,
        std::path::PathBuf::from(&config.storage.data_dir),
    ))
}

fn create_object_detector(config: &Config) -> Option<OllamaDetector> {
    let od = &config.analytics.object_detection;
    let fallback = od
        .ollama
        .fallback
        .as_ref()
        .map(|fb| (fb.url.as_str(), fb.model.as_str()));
    OllamaDetector::new(
        &od.ollama.url,
        &od.ollama.model,
        od.confidence_threshold,
        od.classes.clone(),
        fallback,
    )
    .ok()
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
    /// Kept alive so warm writers keep running (prune tick) even without
    /// analyzers; dropped during shutdown to let the writers drain and exit.
    event_senders: Vec<tokio::sync::mpsc::Sender<FinishedEvent>>,
    buffers_map: HashMap<String, Arc<RwLock<HotBuffer>>>,
}

struct SpawnContext<'a> {
    config: &'a Config,
    motion_store: &'a MotionStore,
    detection_store: &'a DetectionStore,
    debug_store: &'a DetectionDebugStore,
    warm_index: &'a Option<WarmEventIndex>,
    detection_grid: &'a Option<analytics::detection_grid::DetectionGrid>,
    object_detection_ready: bool,
    shutdown: &'a Arc<AtomicBool>,
}

fn spawn_cameras(ctx: &SpawnContext, cameras: Vec<config::CameraConfig>) -> CameraHandles {
    let mut handles = CameraHandles {
        pipeline_handles: Vec::new(),
        analyzer_handles: Vec::new(),
        continuous_handles: Vec::new(),
        warm_handles: Vec::new(),
        event_senders: Vec::new(),
        buffers_map: HashMap::new(),
    };

    for cam_config in cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), ctx.config.buffer.hot_duration_secs);
        let camera_id = cam_config.id.clone();

        let event_tx = if ctx.config.storage.enabled {
            let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
            let writer = WarmWriter::new(
                rx,
                camera_id.clone(),
                &ctx.config.storage,
                ctx.warm_index.clone(),
            );
            handles.warm_handles.push(tokio::spawn(writer.run()));
            handles.event_senders.push(tx.clone());
            Some(tx)
        } else {
            None
        };

        handles
            .buffers_map
            .insert(camera_id.clone(), Arc::clone(&buffer));

        let buffer_clone = Arc::clone(&buffer);
        let shutdown_clone = Arc::clone(ctx.shutdown);
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
                    Arc::clone(ctx.shutdown),
                );
                handles.continuous_handles.push(tokio::spawn(recorder));
            }
        }

        if ctx.config.analytics.enabled {
            let det_store = Some(ctx.detection_store.clone());
            let dbg_store = if ctx.object_detection_ready {
                Some(ctx.debug_store.clone())
            } else {
                None
            };
            let obj_det = if ctx.object_detection_ready {
                create_object_detector(ctx.config)
            } else {
                None
            };

            let analyzer_handle = analytics::spawn_analyzer(
                AnalyzerContext {
                    camera_id,
                    buffer,
                    motion_store: ctx.motion_store.clone(),
                    detection_store: det_store,
                    debug_store: dbg_store,
                    object_detector: obj_det,
                    config: ctx.config.analytics.clone(),
                    detection_grid: ctx.detection_grid.clone(),
                    data_dir: std::path::PathBuf::from(&ctx.config.storage.data_dir),
                    event_tx,
                    pre_padding_ns: ctx.config.storage.pre_padding_secs * 1_000_000_000,
                    post_padding: std::time::Duration::from_secs(
                        ctx.config.storage.post_padding_secs,
                    ),
                    max_event_duration: std::time::Duration::from_secs(
                        ctx.config.storage.max_event_duration_secs,
                    ),
                },
                Arc::clone(ctx.shutdown),
            );
            handles.analyzer_handles.push(analyzer_handle);
        }
    }

    handles
}

async fn wait_for_signal(shutdown: &AtomicBool) {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("SIGINT received, shutting down");
        }
        _ = sigterm.recv() => {
            tracing::info!("SIGTERM received, shutting down");
        }
    }
    shutdown.store(true, Ordering::Relaxed);
}

async fn graceful_shutdown(handles: CameraHandles) {
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

    let mut buffers_with_ids = Vec::new();
    for (camera_id, handle, buffer) in handles.pipeline_handles {
        let _ = handle.await;
        tracing::info!(camera = %camera_id, "camera pipeline stopped");
        buffers_with_ids.push((camera_id, buffer));
    }

    // With all senders gone the warm writers drain their queues and exit;
    // awaiting them guarantees every accepted event reached disk.
    drop(handles.event_senders);
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

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("camon=debug".parse()?))
        .init();

    let config = Config::load()?;
    check_for_updates(&config).await;

    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let camera_ids: Vec<String> = config.cameras.iter().map(|c| c.id.clone()).collect();
    let motion_store = MotionStore::new(&camera_ids);
    let detection_store = DetectionStore::new(&camera_ids);
    let debug_store = DetectionDebugStore::new(&camera_ids);
    let object_detection_ready = log_object_detection_config(&config);
    let warm_index = init_warm_index(&config, &camera_ids);
    let detection_grid = init_detection_grid(&config, &camera_ids);

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

    let shutdown = Arc::new(AtomicBool::new(false));
    let spawn_ctx = SpawnContext {
        config: &config,
        motion_store: &motion_store,
        detection_store: &detection_store,
        debug_store: &debug_store,
        warm_index: &warm_index,
        detection_grid: &detection_grid,
        object_detection_ready,
        shutdown: &shutdown,
    };
    let camera_handles = spawn_cameras(&spawn_ctx, config.cameras.clone());

    let app_state = AppState::new(
        camera_handles.buffers_map.clone(),
        motion_store,
        detection_store,
        debug_store,
        warm_index,
        detection_grid,
    );
    let http_port = config.http.port;
    let server_handle = tokio::spawn(async move {
        if let Err(e) = api::start_server(app_state, http_port).await {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    wait_for_signal(&shutdown).await;
    server_handle.abort();

    graceful_shutdown(camera_handles).await;

    tracing::info!("shutdown complete");
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
    shutdown: Arc<AtomicBool>,
) {
    let camera_id = config.id.clone();

    let buffer_ref = Arc::clone(&buffer);
    let camera_id_clone = camera_id.clone();
    let shutdown_clone = Arc::clone(&shutdown);

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

    while !shutdown.load(Ordering::Relaxed) {
        tracing::info!(camera = %camera_id, url = %config.url, "connecting to camera");

        let pipeline = match FfmpegPipeline::new(&config, Arc::clone(&buffer)) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(camera = %camera_id, "failed to create pipeline: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(RECONNECT_BASE_SECS)).await;
                continue;
            }
        };

        let shutdown_ref = Arc::clone(&shutdown);

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

        if shutdown.load(Ordering::Relaxed) {
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
        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;

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
}
