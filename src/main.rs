use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tracing_subscriber::EnvFilter;

mod analytics;
mod api;
mod buffer;
mod camera;
mod config;
mod storage;

use analytics::ObjectDetector;
use api::AppState;
use buffer::warm::WarmWriter;
use buffer::HotBuffer;
use camera::FfmpegPipeline;
use config::Config;
use storage::{DetectionStore, MotionStore, WarmEventIndex};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("camon=debug".parse()?))
        .init();

    let config = Config::load()?;
    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let http_port = config.http.port;
    let camera_ids: Vec<String> = config.cameras.iter().map(|c| c.id.clone()).collect();
    let motion_store = MotionStore::new(&camera_ids);
    let detection_store = DetectionStore::new(&camera_ids);

    let object_detector = if config.analytics.enabled && config.analytics.object_detection.enabled {
        match ObjectDetector::new(
            &config.analytics.object_detection.model_path,
            config.analytics.object_detection.confidence_threshold,
            config.analytics.object_detection.classes.clone(),
        ) {
            Ok(detector) => {
                tracing::info!(
                    model = %config.analytics.object_detection.model_path,
                    "object detector loaded"
                );
                Some(Arc::new(detector))
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to load object detector, continuing without it");
                None
            }
        }
    } else {
        None
    };

    let warm_index = if config.storage.enabled {
        let index = WarmEventIndex::new(
            &camera_ids,
            std::path::PathBuf::from(&config.storage.data_dir),
        );
        index.scan();
        Some(index)
    } else {
        None
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    let mut analyzer_handles = Vec::new();
    let mut warm_handles = Vec::new();
    let mut buffers_map: HashMap<String, Arc<RwLock<HotBuffer>>> = HashMap::new();

    for cam_config in config.cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), config.buffer.hot_duration_secs);
        let buffer_clone = Arc::clone(&buffer);
        let camera_id = cam_config.id.clone();
        let shutdown_clone = Arc::clone(&shutdown);

        if config.storage.enabled {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            buffer.write().unwrap().set_eviction_sender(tx);
            let writer = WarmWriter::new(
                rx,
                motion_store.clone(),
                detection_store.clone(),
                std::path::PathBuf::from(&config.storage.data_dir),
                camera_id.clone(),
                config.storage.pre_padding_secs,
                config.storage.post_padding_secs,
                warm_index.clone(),
            );
            let warm_handle = tokio::spawn(writer.run());
            warm_handles.push(warm_handle);
        }

        buffers_map.insert(camera_id.clone(), Arc::clone(&buffer));

        let handle = tokio::spawn(async move {
            run_camera(cam_config, buffer_clone, shutdown_clone).await;
        });

        handles.push((camera_id.clone(), handle, Arc::clone(&buffer)));

        if config.analytics.enabled {
            let det_store = if object_detector.is_some() {
                Some(detection_store.clone())
            } else {
                None
            };

            let obj_det = object_detector.as_ref().and_then(|_| {
                ObjectDetector::new(
                    &config.analytics.object_detection.model_path,
                    config.analytics.object_detection.confidence_threshold,
                    config.analytics.object_detection.classes.clone(),
                )
                .ok()
            });

            let analyzer_handle = analytics::spawn_analyzer(
                camera_id,
                buffer,
                motion_store.clone(),
                det_store,
                obj_det,
                config.analytics.clone(),
                Arc::clone(&shutdown),
            );
            analyzer_handles.push(analyzer_handle);
        }
    }

    let app_state = AppState::new(buffers_map, motion_store, detection_store, warm_index);
    let server_handle = tokio::spawn(async move {
        if let Err(e) = api::start_server(app_state, http_port).await {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    tokio::select! {
        _ = async {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        } => {}
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown signal received");
            shutdown.store(true, Ordering::Relaxed);
        }
    }

    server_handle.abort();

    for handle in analyzer_handles {
        handle.abort();
    }

    // Warm writers drain remaining segments and exit when senders are dropped
    for handle in warm_handles {
        let _ = handle.await;
    }

    for (camera_id, handle, buffer) in handles {
        let _ = handle.await;
        if let Ok(buf) = buffer.read() {
            tracing::info!(
                camera = %camera_id,
                segments = buf.segment_count(),
                duration_secs = format!("{:.1}", buf.current_duration_secs()),
                "final buffer stats"
            );
        }
    }

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
