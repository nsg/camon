use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use bytes::Bytes;
use rust_embed::Embed;
use serde::{Deserialize, Serialize};

use crate::analytics::motion_settings::{
    MotionSettings, MASK_COLS, MASK_ROWS, MIN_CONTOUR_AREA_MAX, MIN_CONTOUR_AREA_MIN,
    VAR_THRESHOLD_MAX, VAR_THRESHOLD_MIN,
};
use crate::analytics::{MotionSettingsStore, SettingsUpdate, UpdateError};
use crate::buffer::HotBuffer;
use crate::locks::LockExt;
use crate::storage::event_index::MAX_FILMSTRIP_FRAMES;
use crate::storage::{
    DetectionDebugStore, DetectionStore, EventCursor, EventPage, EventRef, MapKind, MotionStore,
    RangeRequest, ServedRange, ThumbnailError, VideoStream, WarmEventEntry, WarmStorageBackend,
};

use super::auth::{require_token, ApiAuth};
use super::hls;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

#[derive(Clone)]
pub struct AppState {
    pub buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
    pub sub_buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
    pub motion_store: MotionStore,
    pub detection_store: DetectionStore,
    pub debug_store: DetectionDebugStore,
    /// Warm storage backend (local disk today). `None` when storage is disabled.
    pub storage: Option<Arc<dyn WarmStorageBackend>>,
    /// Per-camera deterministic motion settings. `None` when analytics is off.
    pub motion_settings: Option<MotionSettingsStore>,
}

impl AppState {
    pub fn new(
        buffers: HashMap<String, Arc<RwLock<HotBuffer>>>,
        sub_buffers: HashMap<String, Arc<RwLock<HotBuffer>>>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        debug_store: DetectionDebugStore,
        storage: Option<Arc<dyn WarmStorageBackend>>,
        motion_settings: Option<MotionSettingsStore>,
    ) -> Self {
        Self {
            buffers: Arc::new(buffers),
            sub_buffers: Arc::new(sub_buffers),
            motion_store,
            detection_store,
            debug_store,
            storage,
            motion_settings,
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
    stream: Option<String>,
}

#[derive(Deserialize)]
struct SegmentQuery {
    stream: Option<String>,
}

/// The UI shell (`/` and `/assets/*`) stays unauthenticated so the token prompt
/// can load; what the routes under `/api` ask for is [`ApiAuth`]'s to say.
pub fn build_router(state: AppState, auth: &ApiAuth) -> Router {
    let mut api = api_routes().with_state(state);
    if let Some(token_auth) = auth.layer() {
        api = api.route_layer(middleware::from_fn_with_state(token_auth, require_token));
    }

    Router::new()
        .route("/", get(index_handler))
        .route("/assets/{*path}", get(static_handler))
        .merge(api)
}

/// Take the listening socket, and fail startup if it cannot be had.
pub async fn bind(addr: SocketAddr) -> Result<tokio::net::TcpListener, std::io::Error> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("serving the API on http://{}", addr);
    Ok(listener)
}

/// Serve the API on an already-bound listener until something goes wrong. The
/// return is a failure however it reads: the process is meant to be serving.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: AppState,
    auth: ApiAuth,
) -> Result<(), std::io::Error> {
    axum::serve(listener, build_router(state, &auth)).await
}

fn api_routes() -> Router<AppState> {
    Router::new()
        .route("/api/cameras", get(cameras_handler))
        .route("/api/cameras/{id}/motion", get(motion_handler))
        .route(
            "/api/cameras/{id}/motion/{seq}/mask",
            get(motion_mask_handler),
        )
        .route(
            "/api/cameras/{id}/motion/maps/{stage}",
            get(motion_map_handler),
        )
        .route(
            "/api/cameras/{id}/motion/settings",
            get(motion_settings_get_handler).put(motion_settings_put_handler),
        )
        .route("/api/cameras/{id}/detections", get(detections_handler))
        .route(
            "/api/cameras/{id}/detections/{detection_id}/frame",
            get(detection_frame_handler),
        )
        .route("/api/cameras/{id}/hot-events", get(hot_events_handler))
        .route("/api/cameras/{id}/events", get(warm_events_handler))
        .route(
            "/api/cameras/{id}/events/{event}/playlist.m3u8",
            get(warm_playlist_handler),
        )
        .route(
            "/api/cameras/{id}/events/{event}/segment",
            get(warm_segment_handler),
        )
        .route(
            "/api/cameras/{id}/events/{event}/thumbnail",
            get(warm_thumbnail_handler),
        )
        .route(
            "/api/cameras/{id}/events/{event}/filmstrip/{index}",
            get(warm_filmstrip_handler),
        )
        .route(
            "/api/cameras/{id}/detection-debug",
            get(detection_debug_handler),
        )
        .route(
            "/api/cameras/{id}/detection-debug/{debug_id}/frame/{frame_index}",
            get(detection_debug_frame_handler),
        )
        .route(
            "/api/cameras/{id}/detection-debug/{debug_id}/full-frame",
            get(detection_debug_full_frame_handler),
        )
        .route("/api/stream/{id}/playlist.m3u8", get(playlist_handler))
        .route("/api/stream/{id}/segment/{n}", get(segment_handler))
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
                Bytes::from_owner(content.data),
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
    let (buffer, segment_uri_suffix) = match query.stream.as_deref() {
        None => (state.buffers.get(&id), ""),
        Some("sub") => match state.sub_buffers.get(&id) {
            Some(buffer) => (Some(buffer), "?stream=sub"),
            // Main and sub buffers have unrelated media-sequence namespaces. A live player
            // refreshes on every target duration, so content-dependent fallback could switch
            // namespaces mid-watch; config membership is stable for the process lifetime.
            None => (state.buffers.get(&id), ""),
        },
        Some(_) => return (StatusCode::BAD_REQUEST, "invalid stream").into_response(),
    };
    match buffer {
        Some(buffer) => {
            let buf = buffer.read_recover();
            let playlist = hls::generate_playlist(&buf, tail_count, segment_uri_suffix);
            (
                [(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")],
                playlist,
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "camera not found").into_response(),
    }
}

async fn segment_handler(
    State(state): State<AppState>,
    Path((id, n)): Path<(String, u64)>,
    Query(query): Query<SegmentQuery>,
) -> Response {
    let buffer = match query.stream.as_deref() {
        None => state.buffers.get(&id),
        Some("sub") => state.sub_buffers.get(&id),
        Some(_) => return (StatusCode::BAD_REQUEST, "invalid stream").into_response(),
    };
    match buffer {
        Some(buffer) => {
            // Take only the segment's Arc under the lock so the ingest thread
            // never waits on a response body; hls::segment_body serves that
            // same allocation without copying.
            let data = {
                let buf = buffer.read_recover();
                hls::generate_segment(&buf, n)
            };
            match data {
                Some(data) => (
                    [(header::CONTENT_TYPE, "video/mp2t")],
                    hls::segment_body(data),
                )
                    .into_response(),
                None => (StatusCode::NOT_FOUND, "segment not found").into_response(),
            }
        }
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
    let buf = buffer.read_recover();
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

async fn motion_map_handler(
    State(state): State<AppState>,
    Path((id, stage)): Path<(String, String)>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    let Some(kind) = MapKind::parse(&stage) else {
        return (StatusCode::NOT_FOUND, "unknown motion map").into_response();
    };

    match state.motion_store.get_map(&id, kind) {
        Some(jpeg) => ([(header::CONTENT_TYPE, "image/jpeg")], jpeg).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            format!("{stage} map not available yet"),
        )
            .into_response(),
    }
}

/// JSON shape for the motion-settings endpoints: current values, mask grid
/// geometry, and slider bounds so the UI need not hard-code them.
#[derive(Serialize)]
struct MotionSettingsResponse {
    var_threshold: f64,
    min_contour_area: f64,
    mask_cols: usize,
    mask_rows: usize,
    mask: Vec<bool>,
    detection_mask: Vec<bool>,
    var_threshold_min: f64,
    var_threshold_max: f64,
    min_contour_area_min: f64,
    min_contour_area_max: f64,
}

impl From<MotionSettings> for MotionSettingsResponse {
    fn from(s: MotionSettings) -> Self {
        Self {
            var_threshold: s.var_threshold,
            min_contour_area: s.min_contour_area,
            mask_cols: MASK_COLS,
            mask_rows: MASK_ROWS,
            mask: s.mask,
            detection_mask: s.detection_mask,
            var_threshold_min: VAR_THRESHOLD_MIN,
            var_threshold_max: VAR_THRESHOLD_MAX,
            min_contour_area_min: MIN_CONTOUR_AREA_MIN,
            min_contour_area_max: MIN_CONTOUR_AREA_MAX,
        }
    }
}

async fn motion_settings_get_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    let store = match &state.motion_settings {
        Some(s) => s,
        None => return (StatusCode::NOT_FOUND, "motion settings not enabled").into_response(),
    };
    match store.get(&id) {
        Some(s) => axum::Json(MotionSettingsResponse::from(s)).into_response(),
        None => (StatusCode::NOT_FOUND, "camera not found").into_response(),
    }
}

async fn motion_settings_put_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(update): Json<SettingsUpdate>,
) -> Response {
    let store = match &state.motion_settings {
        Some(s) => s.clone(),
        None => return (StatusCode::NOT_FOUND, "motion settings not enabled").into_response(),
    };
    // The update fsyncs and holds the camera's persistence lock while it does,
    // so it runs off the async workers.
    let result = tokio::task::spawn_blocking(move || store.update(&id, update)).await;
    match result {
        Ok(Ok(s)) => axum::Json(MotionSettingsResponse::from(s)).into_response(),
        Ok(Err(UpdateError::UnknownCamera)) => {
            (StatusCode::NOT_FOUND, "camera not found").into_response()
        }
        Ok(Err(e @ UpdateError::NotPersisted(_))) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
        // The request is the problem; the live settings are untouched.
        Ok(Err(e @ UpdateError::NotANumber { .. })) => {
            (StatusCode::BAD_REQUEST, e.to_string()).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "motion settings update panicked");
            (StatusCode::INTERNAL_SERVER_ERROR, "settings update failed").into_response()
        }
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
        Some(frame) => ([(header::CONTENT_TYPE, "image/jpeg")], (*frame).clone()).into_response(),
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

/// How many events one listing answers with, and the most a caller may ask for.
const MAX_EVENTS_PER_PAGE: usize = 1000;

#[derive(Deserialize)]
struct EventsQuery {
    from: Option<u64>,
    to: Option<u64>,
    /// Where to resume: a start PTS, or a whole event key. See
    /// [`parse_cursor`].
    before: Option<String>,
    limit: Option<usize>,
}

/// Parse a bare start PTS or full event key cursor. Only a full key can resume inside events
/// sharing a start; malformed cursors must fail rather than resume elsewhere.
fn parse_cursor(before: &str) -> Option<EventCursor> {
    match before.parse::<u64>() {
        Ok(start_pts_ns) => Some(EventCursor::Start(start_pts_ns)),
        Err(_) => EventRef::parse(before).map(EventCursor::Event),
    }
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
    /// Number of filmstrip thumbnail frames available for this event (0..=4).
    /// The UI requests `filmstrip/{i}` for `i in 0..filmstrip_frames`; older
    /// events with no filmstrip report 0 and fall back to the single thumbnail.
    filmstrip_frames: usize,
    /// True when this event continues a previous chunk of the same motion run
    /// (the run was split at the duration cap). Lets the UI stitch the chain.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    continues: bool,
    /// True when this event was salvaged from an interrupted write at startup
    /// (crash/power-cut recovery); its tail may be truncated.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    recovered: bool,
}

/// One page of a camera's stored events, oldest first: the *newest* [`MAX_EVENTS_PER_PAGE`]
/// events of the window asked about.
async fn warm_events_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<EventsQuery>,
) -> Response {
    let backend = match &state.storage {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    let from = query.from.unwrap_or(0);
    let to = query.to.unwrap_or(u64::MAX);
    if from > to {
        return (StatusCode::BAD_REQUEST, "from must not be greater than to").into_response();
    }
    // Clamped rather than refused: a limit of zero would read as an empty
    // page, i.e. end-of-walk. Non-numeric limits fail in the extractor (400).
    let limit = query
        .limit
        .unwrap_or(MAX_EVENTS_PER_PAGE)
        .clamp(1, MAX_EVENTS_PER_PAGE);
    let mut page = EventPage::new(from, to, limit);
    if let Some(before) = &query.before {
        match parse_cursor(before) {
            Some(cursor) => page = page.before(cursor),
            None => return (StatusCode::BAD_REQUEST, "invalid before cursor").into_response(),
        }
    }
    let events = backend.query(&id, page);

    let response: Vec<WarmEventResponse> = events
        .iter()
        .map(|e| WarmEventResponse {
            start_pts_ns: e.start_pts_ns.to_string(),
            duration_ms: e.duration_ms,
            // Same `as_str` spelling event keys are built from, so a listing
            // and its keys cannot disagree.
            event_type: e.event_type.as_str().to_string(),
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
            filmstrip_frames: e.filmstrip_frames,
            continues: e.continues,
            recovered: e.recovered,
        })
        .collect();

    axum::Json(response).into_response()
}

/// Resolve an `{event}` path segment to the one stored event it names.
fn resolve_event(
    backend: &Arc<dyn WarmStorageBackend>,
    camera_id: &str,
    segment: &str,
) -> Result<WarmEventEntry, (StatusCode, &'static str)> {
    let key = EventRef::parse(segment).ok_or((StatusCode::BAD_REQUEST, "invalid event key"))?;
    backend
        .find_event(camera_id, key)
        .ok_or((StatusCode::NOT_FOUND, "event not found"))
}

async fn warm_playlist_handler(
    State(state): State<AppState>,
    Path((id, event)): Path<(String, String)>,
) -> Response {
    let backend = match &state.storage {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    let entry = match resolve_event(backend, &id, &event) {
        Ok(e) => e,
        Err(response) => return response.into_response(),
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

/// Parse an incoming single-range HTTP `Range` header into a [`RangeRequest`].
fn parse_range_header(value: &str) -> Option<RangeRequest> {
    let spec = value.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (start, end) = spec.split_once('-')?;
    let (start, end) = (start.trim(), end.trim());
    if start.is_empty() {
        // bytes=-n (suffix): the final n bytes.
        return Some(RangeRequest::Suffix(end.parse().ok()?));
    }
    let start: u64 = start.parse().ok()?;
    if end.is_empty() {
        // bytes=a-
        return Some(RangeRequest::FromTo { start, end: None });
    }
    let end: u64 = end.parse().ok()?;
    if end < start {
        return None;
    }
    Some(RangeRequest::FromTo {
        start,
        end: Some(end),
    })
}

async fn warm_segment_handler(
    State(state): State<AppState>,
    Path((id, event)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let backend = match &state.storage {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    let entry = match resolve_event(backend, &id, &event) {
        Ok(e) => e,
        Err(response) => return response.into_response(),
    };

    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_range_header);

    match backend.read_video(&id, &entry, range).await {
        Ok(video) => video_stream_response(video),
        Err(e) => video_error_response(&e),
    }
}

/// What a failed read of an indexed event answers.
fn video_error_response(error: &std::io::Error) -> Response {
    if error.kind() == std::io::ErrorKind::InvalidData {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage could not serve the event",
        )
            .into_response();
    }
    (StatusCode::NOT_FOUND, "event file not found").into_response()
}

/// Turn a streamed [`VideoStream`] into an HTTP response: `206` for a satisfied
/// range, `200` + `Accept-Ranges` for a full body, `416` for an unsatisfiable
/// range. The body streams straight from the backend — never buffered whole.
fn video_stream_response(video: VideoStream) -> Response {
    let VideoStream {
        stream,
        total_size,
        range,
    } = video;
    match range {
        ServedRange::Unsatisfiable => (
            StatusCode::RANGE_NOT_SATISFIABLE,
            [(header::CONTENT_RANGE, format!("bytes */{total_size}"))],
            "range not satisfiable",
        )
            .into_response(),
        ServedRange::Full => (
            [
                (header::CONTENT_TYPE, "video/mp2t".to_string()),
                (header::ACCEPT_RANGES, "bytes".to_string()),
                (header::CONTENT_LENGTH, total_size.to_string()),
            ],
            Body::from_stream(stream),
        )
            .into_response(),
        ServedRange::Partial { start, end } => {
            // Reject invalid 206 bounds before subtraction or playback.
            let Some(body_len) = range.body_len(total_size) else {
                tracing::warn!(
                    start,
                    end,
                    total_size,
                    "storage returned a partial range that is not a range of the event; \
                     refusing to serve it"
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "storage returned an invalid partial range",
                )
                    .into_response();
            };
            (
                StatusCode::PARTIAL_CONTENT,
                [
                    (header::CONTENT_TYPE, "video/mp2t".to_string()),
                    (header::ACCEPT_RANGES, "bytes".to_string()),
                    (
                        header::CONTENT_RANGE,
                        format!("bytes {start}-{end}/{total_size}"),
                    ),
                    (header::CONTENT_LENGTH, body_len.to_string()),
                ],
                Body::from_stream(stream),
            )
                .into_response()
        }
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

async fn warm_thumbnail_handler(
    State(state): State<AppState>,
    Path((id, event)): Path<(String, String)>,
) -> Response {
    let backend = match &state.storage {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    let entry = match resolve_event(backend, &id, &event) {
        Ok(e) => e,
        Err(response) => return response.into_response(),
    };

    match backend.read_thumbnail(&id, &entry).await {
        Ok(data) => jpeg_response(data),
        Err(e) => thumbnail_error_response(e),
    }
}

fn thumbnail_error_response(e: ThumbnailError) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, e.message()).into_response()
}

// Detection debug handlers

#[derive(Serialize)]
struct DebugEntryResponse {
    id: u64,
    timestamp: u64,
    raw_responses: Vec<String>,
    model: String,
    detection_count: usize,
    frame_count: usize,
    has_full_frame: bool,
    motion_rects: Vec<(f32, f32, f32, f32)>,
    crop_rect: Option<(f32, f32, f32, f32)>,
    ollama_rects: Vec<(String, f32, f32, f32, f32)>,
}

/// The debug view's poll, and the one thing that opens the store's demand window: frames are
/// produced and kept only while it is open, so a route that stops reaching the store leaves the
/// view permanently empty.
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
            raw_responses: e.raw_responses,
            model: e.model,
            detection_count: e.detection_count,
            frame_count: e.frame_count,
            has_full_frame: e.has_full_frame,
            motion_rects: e.motion_rects,
            crop_rect: e.crop_rect,
            ollama_rects: e.ollama_rects,
        })
        .collect();

    axum::Json(entries).into_response()
}

async fn detection_debug_frame_handler(
    State(state): State<AppState>,
    Path((id, debug_id, frame_index)): Path<(String, u64, usize)>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.debug_store.get_frame_jpeg(&id, debug_id, frame_index) {
        Some(jpeg) => ([(header::CONTENT_TYPE, "image/jpeg")], (*jpeg).clone()).into_response(),
        None => (StatusCode::NOT_FOUND, "debug frame not found").into_response(),
    }
}

async fn detection_debug_full_frame_handler(
    State(state): State<AppState>,
    Path((id, debug_id)): Path<(String, u64)>,
) -> Response {
    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    match state.debug_store.get_full_frame_jpeg(&id, debug_id) {
        Some(jpeg) => ([(header::CONTENT_TYPE, "image/jpeg")], (*jpeg).clone()).into_response(),
        None => (StatusCode::NOT_FOUND, "full frame not found").into_response(),
    }
}

async fn warm_filmstrip_handler(
    State(state): State<AppState>,
    Path((id, event, index)): Path<(String, String, u8)>,
) -> Response {
    let backend = match &state.storage {
        Some(b) => b,
        None => return (StatusCode::NOT_FOUND, "warm storage not enabled").into_response(),
    };

    if !state.buffers.contains_key(&id) {
        return (StatusCode::NOT_FOUND, "camera not found").into_response();
    }

    // Bounded by the same constant the analyzer writes and the store deletes
    // by, not a hand-written copy of it.
    if usize::from(index) >= MAX_FILMSTRIP_FRAMES {
        let last = MAX_FILMSTRIP_FRAMES - 1;
        return (StatusCode::BAD_REQUEST, format!("index must be 0-{last}")).into_response();
    }

    let entry = match resolve_event(backend, &id, &event) {
        Ok(e) => e,
        Err(response) => return response.into_response(),
    };

    match backend.read_filmstrip(&id, &entry, index).await {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::GopSegment;
    use crate::storage::EventType;

    const TOKEN: &str = "s3cr3t token";

    async fn serve(auth: &ApiAuth) -> String {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let state = AppState::new(
            buffers,
            HashMap::new(),
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            None,
        );
        let app = build_router(state, auth);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn stream_buffer(id: &str, bytes: &[u8]) -> Arc<RwLock<HotBuffer>> {
        let buffer = HotBuffer::new(id.to_string(), 60);
        let mut segment = GopSegment::new(0);
        segment.duration_ns = 1_000_000_000;
        segment.frame_count = 1;
        segment.data = Arc::new(bytes.to_vec());
        buffer.write_recover().push(segment);
        buffer
    }

    async fn serve_stream_buffers(
        main: Arc<RwLock<HotBuffer>>,
        sub: Option<Arc<RwLock<HotBuffer>>>,
    ) -> String {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), main)]);
        let sub_buffers = sub
            .map(|buffer| HashMap::from([("cam".to_string(), buffer)]))
            .unwrap_or_default();
        let state = AppState::new(
            buffers,
            sub_buffers,
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            None,
        );
        let app = build_router(state, &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn the_substream_playlist_uses_substream_segment_uris() {
        let base = serve_stream_buffers(
            stream_buffer("cam", b"main"),
            Some(stream_buffer("cam:sub", b"sub")),
        )
        .await;

        let playlist = reqwest::get(format!(
            "{base}/api/stream/cam/playlist.m3u8?live=true&stream=sub"
        ))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

        assert!(playlist.contains("segment/0?stream=sub\n"), "{playlist}");
        assert!(
            playlist
                .lines()
                .filter(|line| line.starts_with("segment/"))
                .all(|line| line.ends_with("?stream=sub")),
            "{playlist}"
        );
    }

    #[tokio::test]
    async fn requesting_a_substream_without_a_sub_buffer_serves_the_main_playlist() {
        let base = serve_stream_buffers(stream_buffer("cam", b"main"), None).await;
        let main = reqwest::get(format!("{base}/api/stream/cam/playlist.m3u8"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let fallback = reqwest::get(format!("{base}/api/stream/cam/playlist.m3u8?stream=sub"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();

        assert_eq!(fallback, main);
        assert!(fallback.contains("segment/0\n"), "{fallback}");
        assert!(!fallback.contains("?stream=sub"), "{fallback}");

        let segment = reqwest::get(format!("{base}/api/stream/cam/segment/0?stream=sub"))
            .await
            .unwrap();
        assert_eq!(segment.status(), reqwest::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn an_empty_sub_buffer_does_not_fall_back_to_the_main_playlist() {
        let empty_sub = HotBuffer::new("cam:sub".to_string(), 60);
        let base = serve_stream_buffers(stream_buffer("cam", b"main"), Some(empty_sub)).await;
        let playlist = reqwest::get(format!("{base}/api/stream/cam/playlist.m3u8?stream=sub"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();

        assert_eq!(
            playlist,
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n\
             #EXT-X-MEDIA-SEQUENCE:0\n"
        );
        assert!(!playlist.contains("segment/"), "{playlist}");
    }

    #[tokio::test]
    async fn the_same_sequence_number_selects_independent_main_and_substream_segments() {
        let base = serve_stream_buffers(
            stream_buffer("cam", b"main bytes"),
            Some(stream_buffer("cam:sub", b"sub bytes")),
        )
        .await;

        let main = reqwest::get(format!("{base}/api/stream/cam/segment/0"))
            .await
            .unwrap();
        assert_eq!(main.status(), reqwest::StatusCode::OK);
        assert_eq!(main.bytes().await.unwrap().as_ref(), b"main bytes");

        let sub = reqwest::get(format!("{base}/api/stream/cam/segment/0?stream=sub"))
            .await
            .unwrap();
        assert_eq!(sub.status(), reqwest::StatusCode::OK);
        assert_eq!(sub.bytes().await.unwrap().as_ref(), b"sub bytes");
    }

    #[tokio::test]
    async fn an_unknown_stream_selector_is_rejected_for_playlists_and_segments() {
        let base = serve_stream_buffers(stream_buffer("cam", b"main"), None).await;
        for path in ["playlist.m3u8?stream=main", "segment/0?stream=main"] {
            let response = reqwest::get(format!("{base}/api/stream/cam/{path}"))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::BAD_REQUEST,
                "{path}"
            );
        }
    }

    async fn serve_with_motion_store() -> (String, MotionStore) {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let motion_store = MotionStore::new(&ids);
        let state = AppState::new(
            buffers,
            HashMap::new(),
            motion_store.clone(),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            None,
        );
        let app = build_router(state, &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), motion_store)
    }

    async fn serve_with_debug_store() -> (String, DetectionDebugStore) {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let debug_store = DetectionDebugStore::new(&ids);
        let state = AppState::new(
            buffers,
            HashMap::new(),
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            debug_store.clone(),
            None,
            None,
        );
        let app = build_router(state, &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), debug_store)
    }

    #[tokio::test]
    async fn asking_for_the_debug_entries_is_what_says_somebody_is_watching() {
        let (base, debug_store) = serve_with_debug_store().await;
        assert!(
            !debug_store.wanted("cam"),
            "frames were being kept before anyone opened the view"
        );

        let response = reqwest::get(format!("{base}/api/cameras/cam/detection-debug"))
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(
            debug_store.wanted("cam"),
            "the view was polled and the detector was still told nobody wants its frames"
        );

        debug_store.insert(
            "cam",
            vec![Arc::new(vec![0xaa])],
            vec!["{}".to_string()],
            "test-model".to_string(),
            0,
            Some(Arc::new(vec![0xbb])),
            Vec::new(),
            None,
            Vec::new(),
        );
        let entries = reqwest::get(format!("{base}/api/cameras/cam/detection-debug"))
            .await
            .unwrap()
            .json::<Vec<serde_json::Value>>()
            .await
            .unwrap();
        assert_eq!(entries.len(), 1);
        let id = entries[0]["id"].as_u64().unwrap();

        let frame = reqwest::get(format!(
            "{base}/api/cameras/cam/detection-debug/{id}/frame/0"
        ))
        .await
        .unwrap();
        assert_eq!(frame.status(), reqwest::StatusCode::OK);
        assert_eq!(frame.bytes().await.unwrap().as_ref(), [0xaa]);

        let full = reqwest::get(format!(
            "{base}/api/cameras/cam/detection-debug/{id}/full-frame"
        ))
        .await
        .unwrap();
        assert_eq!(full.status(), reqwest::StatusCode::OK);
        assert_eq!(full.bytes().await.unwrap().as_ref(), [0xbb]);

        debug_store.mark_requested_ago("cam", std::time::Duration::from_secs(600));
        for url in [
            format!("{base}/api/cameras/cam/detection-debug/{id}/frame/0"),
            format!("{base}/api/cameras/cam/detection-debug/{id}/full-frame"),
        ] {
            assert_eq!(
                reqwest::get(&url).await.unwrap().status(),
                reqwest::StatusCode::OK
            );
            assert!(
                !debug_store.wanted("cam"),
                "{url} re-armed the window on its own"
            );
        }
    }

    #[tokio::test]
    async fn each_motion_map_stage_serves_its_own_jpeg() {
        let (base, motion_store) = serve_with_motion_store().await;
        for kind in MapKind::ALL {
            motion_store.set_map("cam", kind, kind.as_str().as_bytes().to_vec());
        }

        for kind in MapKind::ALL {
            let response = reqwest::get(format!(
                "{base}/api/cameras/cam/motion/maps/{}",
                kind.as_str()
            ))
            .await
            .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK, "{kind:?}");
            assert_eq!(
                response.headers()[reqwest::header::CONTENT_TYPE],
                "image/jpeg"
            );
            assert_eq!(response.text().await.unwrap(), kind.as_str(), "{kind:?}");
        }
    }

    #[tokio::test]
    async fn unpublished_stages_unknown_stages_and_unknown_cameras_are_all_not_found() {
        let (base, motion_store) = serve_with_motion_store().await;
        motion_store.set_map("cam", MapKind::Morph, vec![0xFF]);

        for url in [
            format!("{base}/api/cameras/cam/motion/maps/stability"),
            format!("{base}/api/cameras/cam/motion/maps/nonsense"),
            format!("{base}/api/cameras/nope/motion/maps/morph"),
        ] {
            let status = reqwest::get(&url).await.unwrap().status();
            assert_eq!(status, reqwest::StatusCode::NOT_FOUND, "{url}");
        }
    }

    async fn serve_with_same_start_events() -> (String, tempfile::TempDir, Vec<WarmEventEntry>) {
        use crate::buffer::warm::{assemble_continuous_chunk, assemble_event};
        use crate::buffer::GopSegment;

        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["cam".to_string()];
        let source = HotBuffer::new("cam".to_string(), 3600);
        {
            let mut buf = source.write_recover();
            for seq in 0..10u64 {
                buf.push(GopSegment {
                    start_pts: seq * 1_000_000_000,
                    duration_ns: 1_000_000_000,
                    data: Arc::new(vec![seq as u8; 4]),
                    frame_count: 1,
                });
            }
        }
        let (mut movement, chunk) = {
            let buf = source.read_recover();
            (
                assemble_event(&buf, None, "cam", 5, 6, 5, 0, false, None).unwrap(),
                assemble_continuous_chunk(&buf, "cam", 5, 8, false).unwrap(),
            )
        };
        movement.filmstrip_frames = Some(Arc::new(vec![vec![0xaa]]));

        let storage = Arc::new(crate::storage::LocalDiskBackend::new(
            dir.path().to_path_buf(),
            &ids,
        ));
        storage.write_event("cam", &movement).await;
        storage.write_event("cam", &chunk).await;
        let events = storage.query("cam", EventPage::unbounded(0, u64::MAX));
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].start_pts_ns, events[1].start_pts_ns);

        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let state = AppState::new(
            buffers,
            HashMap::new(),
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            Some(storage),
            None,
        );
        let app = build_router(state, &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), dir, events)
    }

    #[tokio::test]
    async fn each_same_start_event_is_served_under_its_own_key() {
        let (base, _dir, events) = serve_with_same_start_events().await;
        let start = events[0].start_pts_ns;

        for (event_type, duration_ms, bytes) in [
            (EventType::Movement, 2000u32, 8usize),
            (EventType::Continuous, 4000, 16),
        ] {
            let key = EventRef::new(start, duration_ms, event_type);
            let events = format!("{base}/api/cameras/cam/events/{key}");

            let playlist = reqwest::get(format!("{events}/playlist.m3u8"))
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            let secs = duration_ms as f64 / 1000.0;
            assert!(
                playlist.contains(&format!("#EXTINF:{secs:.3},")),
                "{key} was served another event's playlist: {playlist}"
            );

            let segment = reqwest::get(format!("{events}/segment")).await.unwrap();
            assert_eq!(segment.status(), reqwest::StatusCode::OK, "{key}");
            let body = segment.bytes().await.unwrap();
            assert_eq!(body.len(), bytes, "{key} was served another event's video");

            let frame = reqwest::get(format!("{events}/filmstrip/0")).await.unwrap();
            match event_type {
                EventType::Movement => {
                    assert_eq!(frame.status(), reqwest::StatusCode::OK, "{key}");
                    assert_eq!(frame.bytes().await.unwrap().as_ref(), [0xaa]);
                }
                _ => assert_eq!(frame.status(), reqwest::StatusCode::NOT_FOUND, "{key}"),
            }
        }

        let missing = EventRef::new(start, 2000, EventType::Object);
        let status = reqwest::get(format!(
            "{base}/api/cameras/cam/events/{missing}/playlist.m3u8"
        ))
        .await
        .unwrap()
        .status();
        assert_eq!(status, reqwest::StatusCode::NOT_FOUND);

        for bad in [
            start.to_string(),
            format!("{start}_2000"),
            format!("{start}_2000_movements"),
            "nonsense".to_string(),
        ] {
            let status = reqwest::get(format!("{base}/api/cameras/cam/events/{bad}/playlist.m3u8"))
                .await
                .unwrap()
                .status();
            assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "{bad}");
        }
    }

    async fn serve_with_storage() -> (String, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let storage = Arc::new(crate::storage::LocalDiskBackend::new(
            dir.path().to_path_buf(),
            &ids,
        ));
        let state = AppState::new(
            buffers,
            HashMap::new(),
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            Some(storage),
            None,
        );
        let app = build_router(state, &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), dir)
    }

    fn motion_settings_state(data_dir: &std::path::Path) -> AppState {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        AppState::new(
            buffers,
            HashMap::new(),
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            Some(MotionSettingsStore::new(&ids, data_dir, 16.0, 200.0)),
        )
    }

    async fn serve_with_motion_settings(data_dir: &std::path::Path) -> String {
        let app = build_router(motion_settings_state(data_dir), &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_settings_put_that_cannot_be_persisted_is_not_a_success() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("cam"), b"not a directory").unwrap();
        let base = serve_with_motion_settings(dir.path()).await;

        let response = reqwest::Client::new()
            .put(format!("{base}/api/cameras/cam/motion/settings"))
            .json(&serde_json::json!({ "var_threshold": 32.0 }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(response.text().await.unwrap().contains("not saved"));

        let live: serde_json::Value =
            reqwest::get(format!("{base}/api/cameras/cam/motion/settings"))
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
        assert_eq!(live["var_threshold"], 32.0);
    }

    #[tokio::test]
    async fn a_settings_put_that_is_not_a_number_never_reaches_the_detector() {
        let dir = tempfile::tempdir().unwrap();
        let base = serve_with_motion_settings(dir.path()).await;
        let url = format!("{base}/api/cameras/cam/motion/settings");
        let client = reqwest::Client::new();

        client
            .put(&url)
            .json(&serde_json::json!({ "var_threshold": 32.0 }))
            .send()
            .await
            .unwrap();

        let response = client
            .put(&url)
            .header("content-type", "application/json")
            .body(r#"{"var_threshold": 1e999}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);

        let live: serde_json::Value = reqwest::get(&url).await.unwrap().json().await.unwrap();
        assert_eq!(live["var_threshold"], 32.0, "the refused value took hold");
    }

    #[tokio::test]
    async fn a_slider_value_the_store_refuses_is_a_400_not_a_500() {
        let dir = tempfile::tempdir().unwrap();
        let state = motion_settings_state(dir.path());
        let store = state.motion_settings.clone().unwrap();

        let response = motion_settings_put_handler(
            State(state),
            Path("cam".to_string()),
            Json(SettingsUpdate {
                var_threshold: Some(f64::NAN),
                ..Default::default()
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(store.get("cam").unwrap().var_threshold, 16.0);
    }

    #[tokio::test]
    async fn settings_for_an_unknown_camera_are_a_404_not_a_500() {
        let dir = tempfile::tempdir().unwrap();
        let base = serve_with_motion_settings(dir.path()).await;

        let response = reqwest::Client::new()
            .put(format!("{base}/api/cameras/nope/motion/settings"))
            .json(&serde_json::json!({ "var_threshold": 32.0 }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

        let response = reqwest::Client::new()
            .put(format!("{base}/api/cameras/cam/motion/settings"))
            .json(&serde_json::json!({ "var_threshold": 32.0 }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }

    fn indexed_entry(start_pts_ns: u64, duration_ms: u32) -> WarmEventEntry {
        WarmEventEntry {
            start_pts_ns,
            duration_ms,
            event_type: crate::storage::EventType::Movement,
            file_size: 0,
            sidecar_bytes: 0,
            thumbnail_bytes: 0,
            object_classes: Vec::new(),
            backend: None,
            model: None,
            detections: Vec::new(),
            filmstrip_frames: 0,
            continues: false,
            recovered: false,
            delete_failed: false,
        }
    }

    async fn serve_with_deep_archive(count: u64) -> (String, tempfile::TempDir) {
        serve_with_archive((1..=count).map(|i| indexed_entry(i * 1_000_000_000, 1000))).await
    }

    async fn serve_with_zero_stamped_archive(count: u32) -> (String, tempfile::TempDir) {
        serve_with_archive((1..=count).map(|duration_ms| indexed_entry(0, duration_ms))).await
    }

    async fn serve_with_archive(
        entries: impl Iterator<Item = WarmEventEntry>,
    ) -> (String, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let storage = Arc::new(crate::storage::LocalDiskBackend::new(
            dir.path().to_path_buf(),
            &ids,
        ));
        for entry in entries {
            storage.index_for_tests().insert("cam", entry);
        }
        let state = AppState::new(
            buffers,
            HashMap::new(),
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            Some(storage),
            None,
        );
        let app = build_router(state, &ApiAuth::Open);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), dir)
    }

    fn listed_starts(events: &[serde_json::Value]) -> Vec<u64> {
        events
            .iter()
            .map(|e| e["start_pts_ns"].as_str().unwrap().parse().unwrap())
            .collect()
    }

    async fn list_events(url: String) -> Vec<serde_json::Value> {
        let response = reqwest::get(url).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        response.json().await.unwrap()
    }

    #[tokio::test]
    async fn a_listing_answers_with_one_page_of_the_newest_events() {
        let (base, _dir) = serve_with_deep_archive(2500).await;

        let events = list_events(format!("{base}/api/cameras/cam/events")).await;
        assert_eq!(events.len(), MAX_EVENTS_PER_PAGE);

        let starts = listed_starts(&events);
        assert!(starts.windows(2).all(|w| w[0] < w[1]), "not in start order");
        assert_eq!(*starts.last().unwrap(), 2500 * 1_000_000_000);
        assert_eq!(starts[0], 1501 * 1_000_000_000);

        let smaller = list_events(format!("{base}/api/cameras/cam/events?limit=7")).await;
        assert_eq!(smaller.len(), 7);
        let capped = list_events(format!("{base}/api/cameras/cam/events?limit=99999")).await;
        assert_eq!(capped.len(), MAX_EVENTS_PER_PAGE);
    }

    fn listed_key(event: &serde_json::Value) -> String {
        format!(
            "{}_{}_{}",
            event["start_pts_ns"].as_str().unwrap(),
            event["duration_ms"].as_u64().unwrap(),
            event["event_type"].as_str().unwrap()
        )
    }

    #[tokio::test]
    async fn a_page_walk_reaches_every_older_event_exactly_once() {
        let (base, _dir) = serve_with_deep_archive(1000).await;

        let mut seen: Vec<u64> = Vec::new();
        let mut cursor: Option<String> = None;
        let mut requests = 0;
        for _ in 0..10 {
            let url = match &cursor {
                None => format!("{base}/api/cameras/cam/events?limit=300"),
                Some(c) => format!("{base}/api/cameras/cam/events?limit=300&before={c}"),
            };
            requests += 1;
            let page = list_events(url).await;
            if page.is_empty() {
                break;
            }
            cursor = Some(listed_key(&page[0]));
            seen.splice(0..0, listed_starts(&page));
        }

        let every: Vec<u64> = (1..=1000).map(|i| i * 1_000_000_000).collect();
        assert_eq!(seen, every);
        assert_eq!(requests, 5);
    }

    #[tokio::test]
    async fn a_single_start_archive_pages_at_exactly_its_limit() {
        let (base, _dir) = serve_with_zero_stamped_archive(3000).await;

        let mut seen: Vec<u64> = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..8 {
            let url = match &cursor {
                None => format!("{base}/api/cameras/cam/events"),
                Some(c) => format!("{base}/api/cameras/cam/events?before={c}"),
            };
            let page = list_events(url).await;
            if page.is_empty() {
                break;
            }
            assert!(
                page.len() <= MAX_EVENTS_PER_PAGE,
                "a page of {} events",
                page.len()
            );
            cursor = Some(listed_key(&page[0]));
            seen.splice(
                0..0,
                page.iter().map(|e| e["duration_ms"].as_u64().unwrap()),
            );
        }

        assert_eq!(seen, (1..=3000u64).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_listing_keeps_the_shape_the_ui_reads() {
        let (base, _dir, _events) = serve_with_same_start_events().await;

        let events = list_events(format!("{base}/api/cameras/cam/events?limit=1")).await;
        assert_eq!(events.len(), 1);
        let next = list_events(format!(
            "{base}/api/cameras/cam/events?limit=1&before={}",
            listed_key(&events[0])
        ))
        .await;
        assert_eq!(next.len(), 1);
        assert_eq!(
            next[0]["start_pts_ns"], events[0]["start_pts_ns"],
            "the run's other member"
        );

        let event = &events[0];
        assert!(event["start_pts_ns"].is_string());
        assert!(event["duration_ms"].is_u64());
        assert!(event["event_type"].is_string());
        assert!(event["filmstrip_frames"].is_u64());
        let key = listed_key(event);
        let playlist = reqwest::get(format!("{base}/api/cameras/cam/events/{key}/playlist.m3u8"))
            .await
            .unwrap();
        assert_eq!(playlist.status(), reqwest::StatusCode::OK);
    }

    #[tokio::test]
    async fn a_cursor_that_is_neither_form_is_rejected() {
        let (base, _dir) = serve_with_deep_archive(3).await;

        for bad in ["nonsense", "12_34", "12_34_banana", "12_34_movement_5"] {
            let response = reqwest::get(format!("{base}/api/cameras/cam/events?before={bad}"))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::BAD_REQUEST,
                "before={bad}"
            );
        }

        for good in ["3000000000", "3000000000_1000_movement"] {
            let response = reqwest::get(format!("{base}/api/cameras/cam/events?before={good}"))
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK, "before={good}");
        }
    }

    #[tokio::test]
    async fn an_inverted_event_range_is_rejected() {
        let (base, _dir) = serve_with_storage().await;

        let response = reqwest::get(format!(
            "{base}/api/cameras/cam/events?from=9999999999&to=0"
        ))
        .await
        .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);

        let response = reqwest::get(format!(
            "{base}/api/cameras/cam/events?from=0&to=9999999999"
        ))
        .await
        .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(response.json::<Vec<serde_json::Value>>().await.is_ok());
    }

    #[tokio::test]
    async fn api_rejects_requests_without_the_token() {
        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let client = reqwest::Client::new();

        let status = client
            .get(format!("{base}/api/cameras"))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);

        let status = client
            .get(format!("{base}/api/cameras"))
            .bearer_auth("wrong")
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn bearer_header_carries_the_token() {
        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let response = reqwest::Client::new()
            .get(format!("{base}/api/cameras"))
            .bearer_auth(TOKEN)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.json::<Vec<String>>().await.unwrap(), ["cam"]);
    }

    #[tokio::test]
    async fn query_token_carries_it_where_headers_cannot() {
        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let response = reqwest::get(format!(
            "{base}/api/stream/cam/playlist.m3u8?live=true&token=s3cr3t%20token"
        ))
        .await
        .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(response.text().await.unwrap().starts_with("#EXTM3U"));
    }

    #[tokio::test]
    async fn a_wrong_query_token_is_rejected() {
        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let status = reqwest::get(format!("{base}/api/cameras?token=wrong"))
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn writes_require_the_header_and_never_the_query_token() {
        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let url = format!("{base}/api/cameras/cam/motion/settings");
        let client = reqwest::Client::new();

        let status = client
            .put(&url)
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);

        let status = client
            .put(format!("{url}?token=s3cr3t%20token"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);

        let status = client
            .put(&url)
            .bearer_auth(TOKEN)
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn ui_shell_loads_without_a_token() {
        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let client = reqwest::Client::new();

        let response = client.get(&base).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(response.text().await.unwrap().contains("<title>"));

        let status = client
            .get(format!("{base}/assets/app.js"))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::OK);
    }

    #[tokio::test]
    async fn an_open_policy_asks_for_nothing_at_all() {
        let base = serve(&ApiAuth::Open).await;
        let status = reqwest::get(format!("{base}/api/cameras"))
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::OK);

        let status = reqwest::Client::new()
            .put(format!("{base}/api/cameras/cam/motion/settings"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_generated_token_refuses_a_write_that_does_not_carry_it() {
        let base = serve(&ApiAuth::Writes(TOKEN.to_string())).await;
        let url = format!("{base}/api/cameras/cam/motion/settings");
        let client = reqwest::Client::new();

        let response = client
            .put(&url)
            .json(&serde_json::json!({ "mask": [true, true] }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()[reqwest::header::WWW_AUTHENTICATE],
            "Bearer"
        );

        let status = client
            .put(format!("{url}?token=s3cr3t%20token"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);

        let status = client
            .put(&url)
            .bearer_auth("wrong")
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_generated_token_lets_the_write_through() {
        let base = serve(&ApiAuth::Writes(TOKEN.to_string())).await;
        let status = reqwest::Client::new()
            .put(format!("{base}/api/cameras/cam/motion/settings"))
            .bearer_auth(TOKEN)
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_generated_token_deliberately_leaves_reading_open() {
        let base = serve(&ApiAuth::Writes(TOKEN.to_string())).await;
        for url in [
            format!("{base}/api/cameras"),
            format!("{base}/api/stream/cam/playlist.m3u8?live=true"),
        ] {
            let status = reqwest::get(&url).await.unwrap().status();
            assert_eq!(status, reqwest::StatusCode::OK, "{url}");
        }

        let base = serve(&ApiAuth::Everything(TOKEN.to_string())).await;
        let status = reqwest::get(format!("{base}/api/cameras"))
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_config_from_before_this_existed_keeps_its_ui_and_gains_a_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[http]\nport = 8080\n\n[[cameras]]\nid = \"cam\"\nurl = \"rtsp://10.0.0.5:554/s0\"\n",
        )
        .unwrap();
        let config = crate::config::Config::load_from_with_overrides(&path, &[]).unwrap();
        assert_eq!(config.http.bind, "0.0.0.0");

        let auth = ApiAuth::resolve(
            config.http.bind_addr(),
            config.http.token.as_deref(),
            config.http.allow_open,
            config.token_file_path().as_deref(),
        )
        .unwrap();
        let ApiAuth::Writes(token) = &auth else {
            panic!("an old config did not gain a generated token: {auth:?}");
        };
        assert_eq!(
            std::fs::read_to_string(dir.path().join("api-token"))
                .unwrap()
                .trim(),
            token
        );

        let base = serve(&auth).await;
        assert_eq!(
            reqwest::get(&base).await.unwrap().status(),
            reqwest::StatusCode::OK
        );
        assert_eq!(
            reqwest::get(format!("{base}/api/cameras"))
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        let status = reqwest::Client::new()
            .put(format!("{base}/api/cameras/cam/motion/settings"))
            .bearer_auth(token)
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .status();
        assert_ne!(status, reqwest::StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn parse_range_header_accepts_every_single_range_form() {
        assert_eq!(
            parse_range_header("bytes=0-99"),
            Some(RangeRequest::FromTo {
                start: 0,
                end: Some(99)
            })
        );
        assert_eq!(
            parse_range_header("bytes=500-"),
            Some(RangeRequest::FromTo {
                start: 500,
                end: None
            })
        );
        assert_eq!(
            parse_range_header("bytes=-200"),
            Some(RangeRequest::Suffix(200))
        );
        assert_eq!(
            parse_range_header("bytes= 10-20 "),
            Some(RangeRequest::FromTo {
                start: 10,
                end: Some(20)
            })
        );
    }

    #[test]
    fn parse_range_header_declines_garbage_and_multi_range() {
        assert_eq!(parse_range_header("bytes=abc"), None);
        assert_eq!(parse_range_header("items=0-1"), None);
        assert_eq!(parse_range_header("bytes=10"), None);
        assert_eq!(parse_range_header("bytes="), None);
        assert_eq!(parse_range_header("bytes=-"), None);
        assert_eq!(parse_range_header("garbage"), None);
        assert_eq!(parse_range_header("bytes=20-10"), None);
        assert_eq!(parse_range_header("bytes=0-10,20-30"), None);
    }

    fn streamed(range: ServedRange, total_size: u64) -> Response {
        video_stream_response(VideoStream {
            stream: Box::pin(futures_util::stream::empty()),
            total_size,
            range,
        })
    }

    #[test]
    fn a_partial_range_that_is_not_a_range_of_the_event_is_refused_not_subtracted() {
        for (start, end) in [
            (19u64, 10u64), // reversed: the underflow
            (1, 0),         // reversed by one, at zero
            (10, 40),       // end is one past the last byte
            (40, 45),       // wholly past the object
            (0, u64::MAX),  // an end nothing can be sliced to
        ] {
            let response = streamed(ServedRange::Partial { start, end }, 40);
            assert_eq!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "bytes {start}-{end}/40 was served"
            );
            assert!(response.headers().get(header::CONTENT_RANGE).is_none());
        }
    }

    #[test]
    fn a_store_that_answered_with_something_else_is_not_a_missing_event() {
        let refused = video_error_response(&std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "206 without a usable Content-Range",
        ));
        assert_eq!(refused.status(), StatusCode::INTERNAL_SERVER_ERROR);

        for kind in [
            std::io::ErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::ConnectionReset,
        ] {
            let missing = video_error_response(&std::io::Error::from(kind));
            assert_eq!(missing.status(), StatusCode::NOT_FOUND, "{kind:?}");
        }
    }

    #[test]
    fn a_satisfied_range_is_served_206_with_its_own_length() {
        let response = streamed(ServedRange::Partial { start: 10, end: 19 }, 40);
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "10");
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes 10-19/40");

        let response = streamed(ServedRange::Partial { start: 0, end: 39 }, 40);
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "40");
        let response = streamed(ServedRange::Partial { start: 39, end: 39 }, 40);
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "1");

        let response = streamed(ServedRange::Full, 40);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "40");
        let response = streamed(ServedRange::Unsatisfiable, 40);
        assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes */40");
    }
}
