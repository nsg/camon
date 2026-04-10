use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;
use serde::{Deserialize, Serialize};

use crate::analytics::detection_grid::DetectionGrid;
use crate::buffer::HotBuffer;
use crate::storage::{DetectionDebugStore, DetectionStore, MotionStore, WarmEventIndex};

use super::hls;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

#[derive(Clone)]
pub struct AppState {
    pub buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
    pub motion_store: MotionStore,
    pub detection_store: DetectionStore,
    pub debug_store: DetectionDebugStore,
    pub warm_index: Option<WarmEventIndex>,
    pub detection_grid: Option<DetectionGrid>,
}

impl AppState {
    pub fn new(
        buffers: HashMap<String, Arc<RwLock<HotBuffer>>>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        debug_store: DetectionDebugStore,
        warm_index: Option<WarmEventIndex>,
        detection_grid: Option<DetectionGrid>,
    ) -> Self {
        Self {
            buffers: Arc::new(buffers),
            motion_store,
            detection_store,
            debug_store,
            warm_index,
            detection_grid,
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

#[derive(Deserialize)]
struct PlaylistQuery {
    live: Option<bool>,
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
        .route(
            "/api/cameras/{id}/motion/stability",
            get(stability_map_handler),
        )
        .route(
            "/api/cameras/{id}/motion/background",
            get(background_map_handler),
        )
        .route("/api/cameras/{id}/motion/tuner", get(tuner_stats_handler))
        .route(
            "/api/cameras/{id}/detection/grid",
            get(detection_grid_handler),
        )
        .route("/api/cameras/{id}/detections", get(detections_handler))
        .route(
            "/api/cameras/{id}/detections/{detection_id}/frame",
            get(detection_frame_handler),
        )
        .route("/api/cameras/{id}/hot-events", get(hot_events_handler))
        .route("/api/cameras/{id}/events", get(warm_events_handler))
        .route(
            "/api/cameras/{id}/events/{start_pts}/playlist.m3u8",
            get(warm_playlist_handler),
        )
        .route(
            "/api/cameras/{id}/events/{start_pts}/segment",
            get(warm_segment_handler),
        )
        .route(
            "/api/cameras/{id}/events/{start_pts}/thumbnail",
            get(warm_thumbnail_handler),
        )
        .route(
            "/api/cameras/{id}/events/{start_pts}/filmstrip/{index}",
            get(warm_filmstrip_handler),
        )
        .route(
            "/api/cameras/{id}/detection-debug",
            get(detection_debug_handler),
        )
        .route(
            "/api/cameras/{id}/detection-debug/{debug_id}/grid",
            get(detection_debug_grid_handler),
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
        Some(content) => {
            let html = String::from_utf8_lossy(&content.data)
                .replace("__VERSION__", env!("CAMON_VERSION"));
            Html(html).into_response()
        }
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
    Query(query): Query<PlaylistQuery>,
) -> impl IntoResponse {
    let tail_count = if query.live.unwrap_or(false) {
        Some(6)
    } else {
        None
    };
    match state.buffers.get(&id) {
        Some(buffer) => match buffer.read() {
            Ok(buf) => {
                let playlist = hls::generate_playlist(&buf, tail_count);
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

struct BufferContext {
    first_sequence: u64,
    total_duration: f64,
}

#[allow(clippy::result_large_err)]
fn read_buffer_context<'a>(
    state: &'a AppState,
    id: &str,
) -> Result<
    (
        std::sync::RwLockReadGuard<'a, crate::buffer::HotBuffer>,
        BufferContext,
    ),
    Response,
> {
    let buffer = state
        .buffers
        .get(id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "camera not found").into_response())?;
    let buf = buffer
        .read()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "buffer lock error").into_response())?;
    let ctx = BufferContext {
        first_sequence: buf.first_sequence(),
        total_duration: buf.total_duration_ns() as f64 / 1_000_000_000.0,
    };
    Ok((buf, ctx))
}

async fn motion_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let (buf, ctx) = match read_buffer_context(&state, &id) {
        Ok(v) => v,
        Err(r) => return r,
    };

    let segments = state.motion_store.get_motion(&id);

    let response = MotionResponse {
        total_duration: ctx.total_duration,
        segments: segments
            .iter()
            .filter(|s| s.segment_sequence >= ctx.first_sequence)
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
    let (buf, ctx) = match read_buffer_context(&state, &id) {
        Ok(v) => v,
        Err(r) => return r,
    };

    let detections = state.detection_store.get_detections(&id);

    let response = DetectionResponse {
        total_duration: ctx.total_duration,
        detections: detections
            .iter()
            .filter(|d| d.segment_sequence >= ctx.first_sequence)
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

async fn stability_map_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.motion_store.get_stability_map(&id) {
        Some(jpeg) => ([(header::CONTENT_TYPE, "image/jpeg")], jpeg).into_response(),
        None => (StatusCode::NOT_FOUND, "stability map not available yet").into_response(),
    }
}

async fn background_map_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.motion_store.get_background_map(&id) {
        Some(jpeg) => ([(header::CONTENT_TYPE, "image/jpeg")], jpeg).into_response(),
        None => (StatusCode::NOT_FOUND, "background not available yet").into_response(),
    }
}

async fn tuner_stats_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.motion_store.get_tuner_stats(&id) {
        Some(stats) => axum::Json(stats).into_response(),
        None => (StatusCode::NOT_FOUND, "tuner stats not available yet").into_response(),
    }
}

async fn detection_grid_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match &state.detection_grid {
        Some(grid) => match grid.get_grid(&id) {
            Some(data) => axum::Json(data).into_response(),
            None => (StatusCode::NOT_FOUND, "no grid data").into_response(),
        },
        None => (StatusCode::NOT_FOUND, "detection grid not enabled").into_response(),
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

// Hot event types and handler

/// Gap threshold for grouping motion segments into hot events (nanoseconds).
/// Segments with a gap larger than this are treated as separate events.
const HOT_EVENT_GAP_NS: u64 = 10 * 1_000_000_000;

#[derive(Serialize)]
struct HotEventResponse {
    offset_secs: f64,
    duration_secs: f64,
    ago_secs: f64,
}

async fn hot_events_handler(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let (buf, ctx) = match read_buffer_context(&state, &id) {
        Ok(v) => v,
        Err(r) => return r,
    };

    let motion = state.motion_store.get_motion(&id);

    // Collect motion segments that are within the hot buffer, with their timeline offsets
    let mut motion_spans: Vec<(f64, f64)> = motion
        .iter()
        .filter(|s| s.segment_sequence >= ctx.first_sequence)
        .filter_map(|s| {
            let start_ns = buf.sequence_to_offset_ns(s.segment_sequence)?;
            let start = start_ns as f64 / 1_000_000_000.0;
            let end = start + s.duration_ns as f64 / 1_000_000_000.0;
            Some((start, end))
        })
        .collect();

    motion_spans.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Group into events using gap threshold
    let gap_secs = HOT_EVENT_GAP_NS as f64 / 1_000_000_000.0;
    let mut events: Vec<HotEventResponse> = Vec::new();

    let mut event_start: Option<f64> = None;
    let mut event_end: f64 = 0.0;

    for (start, end) in &motion_spans {
        match event_start {
            None => {
                event_start = Some(*start);
                event_end = *end;
            }
            Some(_) => {
                if *start - event_end <= gap_secs {
                    event_end = end.max(event_end);
                } else {
                    let es = event_start.unwrap();
                    events.push(HotEventResponse {
                        offset_secs: es,
                        duration_secs: event_end - es,
                        ago_secs: ctx.total_duration - es,
                    });
                    event_start = Some(*start);
                    event_end = *end;
                }
            }
        }
    }

    if let Some(es) = event_start {
        events.push(HotEventResponse {
            offset_secs: es,
            duration_secs: event_end - es,
            ago_secs: ctx.total_duration - es,
        });
    }

    axum::Json(events).into_response()
}

// Warm event types and handlers

#[derive(Deserialize)]
struct EventsQuery {
    from: Option<u64>,
    to: Option<u64>,
}

#[derive(Serialize)]
struct ObjectClassResponse {
    class: String,
    confidence: f32,
}

#[derive(Serialize)]
struct WarmEventResponse {
    start_pts_ns: String,
    duration_ms: u32,
    event_type: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    object_classes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    detections: Vec<ObjectClassResponse>,
    has_filmstrip: bool,
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
            object_classes: e.object_classes.clone(),
            backend: e.backend.clone(),
            model: e.model.clone(),
            detections: e
                .detections
                .iter()
                .map(|d| ObjectClassResponse {
                    class: d.class.clone(),
                    confidence: d.confidence,
                })
                .collect(),
            has_filmstrip: e.has_filmstrip,
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

fn jpeg_response(data: Vec<u8>) -> Response {
    (
        [
            (header::CONTENT_TYPE, "image/jpeg"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        data,
    )
        .into_response()
}

async fn generate_thumbnail(
    ts_path: &std::path::Path,
    thumb_path: &std::path::Path,
) -> Result<(), (StatusCode, &'static str)> {
    let mut child = tokio::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(ts_path)
        .args(["-frames:v", "1", "-vf", "scale=320:-1", "-q:v", "5", "-y"])
        .arg(thumb_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "failed to spawn ffmpeg"))?;

    let status = child
        .wait()
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "ffmpeg process error"))?;

    if !status.success() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "thumbnail generation failed",
        ));
    }
    Ok(())
}

async fn warm_thumbnail_handler(
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

    let ts_path = index.resolve_file_path(&id, &entry);
    let thumb_path = ts_path.with_extension("jpg");

    if let Ok(data) = tokio::fs::read(&thumb_path).await {
        return jpeg_response(data);
    }

    if let Err((code, msg)) = generate_thumbnail(&ts_path, &thumb_path).await {
        return (code, msg).into_response();
    }

    match tokio::fs::read(&thumb_path).await {
        Ok(data) => jpeg_response(data),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to read thumbnail",
        )
            .into_response(),
    }
}

// Detection debug handlers

#[derive(Serialize)]
struct DebugEntryResponse {
    id: u64,
    timestamp: u64,
    raw_response: String,
    model: String,
    detection_count: usize,
}

async fn detection_debug_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    let entries: Vec<DebugEntryResponse> = state
        .debug_store
        .list(&id)
        .into_iter()
        .map(|e| DebugEntryResponse {
            id: e.id,
            timestamp: e.timestamp,
            raw_response: e.raw_response,
            model: e.model,
            detection_count: e.detection_count,
        })
        .collect();

    axum::Json(entries).into_response()
}

async fn detection_debug_grid_handler(
    State(state): State<AppState>,
    Path((id, debug_id)): Path<(String, u64)>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.debug_store.get_grid_jpeg(&id, debug_id) {
        Some(jpeg) => ([(header::CONTENT_TYPE, "image/jpeg")], jpeg).into_response(),
        None => (StatusCode::NOT_FOUND, "debug entry not found").into_response(),
    }
}

async fn warm_filmstrip_handler(
    State(state): State<AppState>,
    Path((id, start_pts_str, index)): Path<(String, String, u8)>,
) -> Response {
    let index_val = match &state.warm_index {
        Some(idx) => idx,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    if index > 3 {
        return (StatusCode::BAD_REQUEST, "index must be 0-3").into_response();
    }

    let start_pts_ns: u64 = match start_pts_str.parse() {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid start_pts").into_response(),
    };

    let entry = match index_val.find_event(&id, start_pts_ns) {
        Some(e) => e,
        None => return (StatusCode::NOT_FOUND, "event not found").into_response(),
    };

    let ts_path = index_val.resolve_file_path(&id, &entry);
    let stem = format!("{}_{}", entry.start_pts_ns, entry.duration_ms);
    let thumb_path = ts_path
        .parent()
        .unwrap()
        .join(format!("{}_thumb_{}.jpg", stem, index));

    match tokio::fs::read(&thumb_path).await {
        Ok(data) => (
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, "public, max-age=86400"),
            ],
            data,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "filmstrip frame not found").into_response(),
    }
}
