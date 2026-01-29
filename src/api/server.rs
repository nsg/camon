use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;
use serde::{Deserialize, Serialize};

use crate::buffer::HotBuffer;
use crate::storage::{DetectionStore, MotionStore, WarmEventIndex};

use super::hls;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

#[derive(Clone)]
pub struct AppState {
    pub buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
    pub motion_store: MotionStore,
    pub detection_store: DetectionStore,
    pub warm_index: Option<WarmEventIndex>,
}

impl AppState {
    pub fn new(
        buffers: HashMap<String, Arc<RwLock<HotBuffer>>>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        warm_index: Option<WarmEventIndex>,
    ) -> Self {
        Self {
            buffers: Arc::new(buffers),
            motion_store,
            detection_store,
            warm_index,
        }
    }
}

#[derive(Serialize)]
struct MotionSegmentResponse {
    sequence: u64,
    start: f64,
    end: f64,
    intensity: f32,
}

#[derive(Serialize)]
struct MotionResponse {
    total_duration: f64,
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
    total_duration: f64,
    detections: Vec<DetectionItem>,
}

pub async fn start_server(state: AppState, port: u16) -> Result<(), std::io::Error> {
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/assets/{*path}", get(static_handler))
        .route("/api/cameras", get(cameras_handler))
        .route("/api/cameras/{id}/motion", get(motion_handler))
        .route(
            "/api/cameras/{id}/motion/{seq}/mask",
            get(motion_mask_handler),
        )
        .route("/api/cameras/{id}/detections", get(detections_handler))
        .route(
            "/api/cameras/{id}/detections/{detection_id}/frame",
            get(detection_frame_handler),
        )
        .route("/api/cameras/{id}/events", get(warm_events_handler))
        .route(
            "/api/cameras/{id}/events/{start_pts}/playlist.m3u8",
            get(warm_playlist_handler),
        )
        .route(
            "/api/cameras/{id}/events/{start_pts}/segment",
            get(warm_segment_handler),
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

    let buf = match buffer.read() {
        Ok(b) => b,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "buffer lock error").into_response(),
    };

    let first_sequence = buf.first_sequence();
    let total_duration = buf.total_duration_ns() as f64 / 1_000_000_000.0;

    let segments = state.motion_store.get_motion(&id);

    let response = MotionResponse {
        total_duration,
        segments: segments
            .iter()
            .filter(|s| s.segment_sequence >= first_sequence)
            .filter_map(|s| {
                let start_ns = buf.sequence_to_offset_ns(s.segment_sequence)?;
                let start = start_ns as f64 / 1_000_000_000.0;
                let end = start + s.duration_ns as f64 / 1_000_000_000.0;
                Some(MotionSegmentResponse {
                    sequence: s.segment_sequence,
                    start,
                    end,
                    intensity: s.motion_score,
                })
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

    let buf = match buffer.read() {
        Ok(b) => b,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "buffer lock error").into_response(),
    };

    let first_sequence = buf.first_sequence();
    let total_duration = buf.total_duration_ns() as f64 / 1_000_000_000.0;

    let detections = state.detection_store.get_detections(&id);

    let response = DetectionResponse {
        total_duration,
        detections: detections
            .iter()
            .filter(|d| d.segment_sequence >= first_sequence)
            .filter_map(|d| {
                let offset_ns = buf.sequence_to_offset_ns(d.segment_sequence)?;
                Some(DetectionItem {
                    id: d.id,
                    timestamp: offset_ns as f64 / 1_000_000_000.0,
                    object_class: d.object_class.clone(),
                    confidence: d.confidence,
                })
            })
            .collect(),
    };

    axum::Json(response).into_response()
}

async fn motion_mask_handler(
    State(state): State<AppState>,
    Path((id, seq)): Path<(String, u64)>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.motion_store.get_mask(&id, seq) {
        Some(mask) => ([(header::CONTENT_TYPE, "image/jpeg")], mask).into_response(),
        None => (StatusCode::NOT_FOUND, "mask not found").into_response(),
    }
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

// Warm event types and handlers

#[derive(Deserialize)]
struct EventsQuery {
    from: Option<u64>,
    to: Option<u64>,
}

#[derive(Serialize)]
struct WarmEventResponse {
    start_pts_ns: String,
    duration_ms: u32,
    event_type: String,
}

async fn warm_events_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<EventsQuery>,
) -> Response {
    let index = match &state.warm_index {
        Some(idx) => idx,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    let from = query.from.unwrap_or(0);
    let to = query.to.unwrap_or(u64::MAX);
    let events = index.query(&id, from, to);

    let response: Vec<WarmEventResponse> = events
        .iter()
        .map(|e| WarmEventResponse {
            start_pts_ns: e.start_pts_ns.to_string(),
            duration_ms: e.duration_ms,
            event_type: match e.event_type {
                crate::storage::EventType::Movement => "movement".to_string(),
                crate::storage::EventType::Object => "object".to_string(),
            },
        })
        .collect();

    axum::Json(response).into_response()
}

async fn warm_playlist_handler(
    State(state): State<AppState>,
    Path((id, start_pts_str)): Path<(String, String)>,
) -> Response {
    let index = match &state.warm_index {
        Some(idx) => idx,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    let start_pts: u64 = match start_pts_str.parse() {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid start_pts").into_response(),
    };

    let entry = match index.find_event(&id, start_pts) {
        Some(e) => e,
        None => return (StatusCode::NOT_FOUND, "event not found").into_response(),
    };

    let duration_secs = entry.duration_ms as f64 / 1000.0;
    let target_duration = duration_secs.ceil() as u64;

    let playlist = format!(
        "#EXTM3U\n\
         #EXT-X-VERSION:3\n\
         #EXT-X-TARGETDURATION:{target_duration}\n\
         #EXT-X-MEDIA-SEQUENCE:0\n\
         #EXT-X-PLAYLIST-TYPE:VOD\n\
         #EXTINF:{duration_secs:.3},\n\
         segment\n\
         #EXT-X-ENDLIST\n"
    );

    (
        [(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")],
        playlist,
    )
        .into_response()
}

async fn warm_segment_handler(
    State(state): State<AppState>,
    Path((id, start_pts_str)): Path<(String, String)>,
) -> Response {
    let index = match &state.warm_index {
        Some(idx) => idx,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    let start_pts: u64 = match start_pts_str.parse() {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid start_pts").into_response(),
    };

    let entry = match index.find_event(&id, start_pts) {
        Some(e) => e,
        None => return (StatusCode::NOT_FOUND, "event not found").into_response(),
    };

    let file_path = index.resolve_file_path(&id, &entry);

    match tokio::fs::read(&file_path).await {
        Ok(data) => ([(header::CONTENT_TYPE, "video/mp2t")], data).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "event file not found").into_response(),
    }
}
