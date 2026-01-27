use std::sync::{Arc, RwLock};

use gstreamer::prelude::*;
use tracing_subscriber::EnvFilter;

mod buffer;
mod camera;
mod config;

use buffer::HotBuffer;
use camera::RtspPipeline;
use config::Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("camon=debug".parse()?))
        .init();

    gstreamer::init()?;

    let config = Config::load()?;
    tracing::info!("loaded {} camera(s)", config.cameras.len());

    let mut handles = Vec::new();

    for cam_config in config.cameras {
        let buffer = HotBuffer::new(cam_config.id.clone(), config.buffer.hot_duration_secs);
        let buffer_clone = Arc::clone(&buffer);
        let camera_id = cam_config.id.clone();

        let handle = tokio::spawn(async move {
            run_camera(cam_config, buffer_clone).await;
        });

        handles.push((camera_id, handle, buffer));
    }

    let status_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
        }
    });

    for (camera_id, handle, buffer) in handles {
        tokio::select! {
            result = handle => {
                if let Err(e) = result {
                    tracing::error!(camera = %camera_id, "camera task failed: {}", e);
                }
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("shutdown signal received");
                if let Ok(buf) = buffer.read() {
                    tracing::info!(
                        camera = %buf.camera_id(),
                        segments = buf.segment_count(),
                        duration_secs = buf.current_duration_secs(),
                        "final buffer stats"
                    );
                }
                break;
            }
        }
    }

    status_handle.abort();
    tracing::info!("shutdown complete");

    Ok(())
}

async fn run_camera(config: config::CameraConfig, buffer: Arc<RwLock<HotBuffer>>) {
    let camera_id = config.id.clone();

    loop {
        tracing::info!(camera = %camera_id, url = %config.url, "connecting to camera");

        let pipeline = match RtspPipeline::new(&config, Arc::clone(&buffer)) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(camera = %camera_id, "failed to create pipeline: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        if let Err(e) = pipeline.start() {
            tracing::error!(camera = %camera_id, "failed to start pipeline: {}", e);
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            continue;
        }

        let Some(bus) = pipeline.bus() else {
            tracing::error!(camera = %camera_id, "failed to get pipeline bus");
            continue;
        };

        let buffer_ref = Arc::clone(&buffer);
        let camera_id_clone = camera_id.clone();

        let stats_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
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

        loop {
            let msg = bus.timed_pop(gstreamer::ClockTime::from_mseconds(100));

            if let Some(msg) = msg {
                use gstreamer::MessageView;

                match msg.view() {
                    MessageView::Eos(_) => {
                        tracing::warn!(camera = %camera_id, "end of stream");
                        break;
                    }
                    MessageView::Error(err) => {
                        tracing::error!(
                            camera = %camera_id,
                            error = %err.error(),
                            debug = ?err.debug(),
                            "pipeline error"
                        );
                        break;
                    }
                    MessageView::StateChanged(state) => {
                        if state
                            .src()
                            .map(|s| s.name().as_str() == format!("camera-{}", camera_id))
                            .unwrap_or(false)
                        {
                            tracing::debug!(
                                camera = %camera_id,
                                old = ?state.old(),
                                new = ?state.current(),
                                "state changed"
                            );
                        }
                    }
                    _ => {}
                }
            }

            tokio::task::yield_now().await;
        }

        stats_handle.abort();
        let _ = pipeline.stop();

        tracing::info!(camera = %camera_id, "reconnecting in 5 seconds");
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}
