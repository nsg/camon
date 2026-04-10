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
mod storage;
mod update;

use analytics::{AnalyzerContext, OllamaDetector};
use api::AppState;
use buffer::warm::WarmWriter;
use buffer::HotBuffer;
use camera::FfmpegPipeline;
use config::Config;
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
    let index = WarmEventIndex::new(
        camera_ids,
        std::path::PathBuf::from(&config.storage.data_dir),
    );
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

struct CameraHandles {
    pipeline_handles: Vec<(String, tokio::task::JoinHandle<()>, Arc<RwLock<HotBuffer>>)>,
    analyzer_handles: Vec<tokio::task::JoinHandle<()>>,
    warm_handles: Vec<tokio::task::JoinHandle<()>>,
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
        warm_handles: Vec::new(),
        buffers_map: HashMap::new(),
    };

    for cam_config in cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), ctx.config.buffer.hot_duration_secs);
        let camera_id = cam_config.id.clone();

        if ctx.config.storage.enabled {
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            buffer.write().unwrap().set_eviction_sender(tx);
            let writer = WarmWriter::new(
                rx,
                ctx.motion_store.clone(),
                ctx.detection_store.clone(),
                camera_id.clone(),
                &ctx.config.storage,
                ctx.warm_index.clone(),
            );
            handles.warm_handles.push(tokio::spawn(writer.run()));
        }

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

        if ctx.config.analytics.enabled {
            let det_store = if ctx.object_detection_ready {
                Some(ctx.detection_store.clone())
            } else {
                None
            };
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

async fn graceful_shutdown(handles: CameraHandles, warm_handles: Vec<tokio::task::JoinHandle<()>>) {
    for handle in handles.analyzer_handles {
        handle.abort();
    }

    let mut buffers_with_ids = Vec::new();
    for (camera_id, handle, buffer) in handles.pipeline_handles {
        let _ = handle.await;
        tracing::info!(camera = %camera_id, "camera pipeline stopped");
        buffers_with_ids.push((camera_id, buffer));
    }

    for (_, buffer) in &buffers_with_ids {
        if let Ok(mut buf) = buffer.write() {
            buf.close_eviction_channel();
        }
    }

    for handle in warm_handles {
        let _ = handle.await;
    }

    for (camera_id, buffer) in buffers_with_ids {
        if let Ok(buf) = buffer.read() {
            tracing::info!(
                camera = %camera_id,
                segments = buf.segment_count(),
                duration_secs = format!("{:.1}", buf.current_duration_secs()),
                "final buffer stats"
            );
        }
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

    let warm_handles = camera_handles.warm_handles;
    graceful_shutdown(
        CameraHandles {
            pipeline_handles: camera_handles.pipeline_handles,
            analyzer_handles: camera_handles.analyzer_handles,
            warm_handles: Vec::new(),
            buffers_map: HashMap::new(),
        },
        warm_handles,
    )
    .await;

    tracing::info!("shutdown complete");
    Ok(())
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
            if let Ok(buf) = buffer_ref.read() {
                tracing::info!(
                    camera = %camera_id_clone,
                    segments = buf.segment_count(),
                    duration_secs = format!("{:.1}", buf.current_duration_secs()),
                    "buffer stats"
                );
            }
        }
    });

    while !shutdown.load(Ordering::Relaxed) {
        tracing::info!(camera = %camera_id, url = %config.url, "connecting to camera");

        let pipeline = match FfmpegPipeline::new(&config, Arc::clone(&buffer)) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(camera = %camera_id, "failed to create pipeline: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        let shutdown_ref = Arc::clone(&shutdown);
        let camera_id_ref = camera_id.clone();

        let result = tokio::task::spawn_blocking(move || pipeline.run(&shutdown_ref)).await;

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

        tracing::info!(camera = %camera_id_ref, "reconnecting in 5 seconds");
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }

    stats_handle.abort();
}
