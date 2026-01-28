use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;
use serde::Serialize;

use crate::buffer::HotBuffer;
use crate::storage::{DetectionStore, MotionStore};

use super::hls;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

#[derive(Clone)]
pub struct AppState {
    pub buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
    pub motion_store: MotionStore,
    pub detection_store: DetectionStore,
    pub motion_threshold: f32,
}

impl AppState {
    pub fn new(
        buffers: HashMap<String, Arc<RwLock<HotBuffer>>>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        motion_threshold: f32,
    ) -> Self {
        Self {
            buffers: Arc::new(buffers),
            motion_store,
            detection_store,
            motion_threshold,
        }
    }
}

#[derive(Serialize)]
struct MotionSegmentResponse {
    start: f64,
    end: f64,
    intensity: f32,
}

#[derive(Serialize)]
struct MotionResponse {
    segments: Vec<MotionSegmentResponse>,
}

#[derive(Serialize)]
struct DetectionItem {
    id: u64,
    timestamp: f64,
    object_class: String,
    confidence: f32,
}

#[derive(Serialize)]
struct DetectionResponse {
    detections: Vec<DetectionItem>,
}

pub async fn start_server(state: AppState, port: u16) -> Result<(), std::io::Error> {
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/assets/{*path}", get(static_handler))
        .route("/api/cameras", get(cameras_handler))
        .route("/api/cameras/{id}/motion", get(motion_handler))
        .route("/api/cameras/{id}/detections", get(detections_handler))
        .route(
            "/api/cameras/{id}/detections/{detection_id}/frame",
            get(detection_frame_handler),
        )
        .route("/api/stream/{id}/playlist.m3u8", get(playlist_handler))
        .route("/api/stream/{id}/segment/{n}", get(segment_handler))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("starting HTTP server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await
}

async fn index_handler() -> impl IntoResponse {
    match Assets::get("index.html") {
        Some(content) => Html(content.data.to_vec()).into_response(),
        None => (StatusCode::NOT_FOUND, "index.html not found").into_response(),
    }
}

async fn static_handler(Path(path): Path<String>) -> impl IntoResponse {
    match Assets::get(&path) {
        Some(content) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream();
            (
                [(header::CONTENT_TYPE, mime.as_ref())],
                content.data.to_vec(),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

async fn cameras_handler(State(state): State<AppState>) -> impl IntoResponse {
    let cameras: Vec<String> = state.buffers.keys().cloned().collect();
    axum::Json(cameras)
}

async fn playlist_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.buffers.get(&id) {
        Some(buffer) => match buffer.read() {
            Ok(buf) => {
                let playlist = hls::generate_playlist(&buf);
                (
                    [(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")],
                    playlist,
                )
                    .into_response()
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "buffer lock error").into_response(),
        },
        None => (StatusCode::NOT_FOUND, "camera not found").into_response(),
    }
}

async fn segment_handler(
    State(state): State<AppState>,
    Path((id, n)): Path<(String, u64)>,
) -> Response {
    match state.buffers.get(&id) {
        Some(buffer) => match buffer.read() {
            Ok(buf) => match hls::generate_segment(&buf, n) {
                Some(data) => ([(header::CONTENT_TYPE, "video/mp2t")], data).into_response(),
                None => (StatusCode::NOT_FOUND, "segment not found").into_response(),
            },
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "buffer lock error").into_response(),
        },
        None => (StatusCode::NOT_FOUND, "camera not found").into_response(),
    }
}

async fn motion_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let buffer = match state.buffers.get(&id) {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "camera not found").into_response(),
    };

    let base_time = buffer
        .read()
        .ok()
        .and_then(|b| b.first_pts())
        .unwrap_or(0);

    let segments = state.motion_store.get_motion(&id, state.motion_threshold);

    let response = MotionResponse {
        segments: segments
            .iter()
            .filter(|s| s.start_time_ns >= base_time)
            .map(|s| MotionSegmentResponse {
                start: (s.start_time_ns - base_time) as f64 / 1_000_000_000.0,
                end: (s.end_time_ns - base_time) as f64 / 1_000_000_000.0,
                intensity: s.motion_score,
            })
            .collect(),
    };

    axum::Json(response).into_response()
}

async fn detections_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let buffer = match state.buffers.get(&id) {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "camera not found").into_response(),
    };

    let base_time = buffer
        .read()
        .ok()
        .and_then(|b| b.first_pts())
        .unwrap_or(0);

    let detections = state.detection_store.get_detections(&id);

    let response = DetectionResponse {
        detections: detections
            .iter()
            .filter(|d| d.timestamp_ns >= base_time)
            .map(|d| DetectionItem {
                id: d.id,
                timestamp: (d.timestamp_ns - base_time) as f64 / 1_000_000_000.0,
                object_class: d.object_class.clone(),
                confidence: d.confidence,
            })
            .collect(),
    };

    axum::Json(response).into_response()
}

async fn detection_frame_handler(
    State(state): State<AppState>,
    Path((id, detection_id)): Path<(String, u64)>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.detection_store.get_frame(&id, detection_id) {
        Some(frame) => ([(header::CONTENT_TYPE, "image/jpeg")], frame).into_response(),
        None => (StatusCode::NOT_FOUND, "detection not found").into_response(),
    }
}
