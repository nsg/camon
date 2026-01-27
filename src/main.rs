use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use tracing_subscriber::EnvFilter;

mod buffer;
mod camera;
mod config;

use buffer::HotBuffer;
use camera::FfmpegPipeline;
use config::Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("camon=debug".parse()?))
        .init();

    let config = Config::load()?;
    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let shutdown = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    for cam_config in config.cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), config.buffer.hot_duration_secs);
        let buffer_clone = Arc::clone(&buffer);
        let camera_id = cam_config.id.clone();
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = tokio::spawn(async move {
            run_camera(cam_config, buffer_clone, shutdown_clone).await;
        });

        handles.push((camera_id, handle, buffer));
    }

    // Wait for Ctrl+C
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

    // Wait for camera tasks to finish
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

    // Stats logging task
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

    // Main camera loop with reconnection
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

        // Run ffmpeg in a blocking task
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
