use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, RwLock};

use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::analytics::motion_settings::{
    MotionSettings, MASK_COLS, MASK_ROWS, MIN_CONTOUR_AREA_MAX, MIN_CONTOUR_AREA_MIN,
    VAR_THRESHOLD_MAX, VAR_THRESHOLD_MIN,
};
use crate::analytics::{MotionSettingsStore, SettingsUpdate, UpdateError};
use crate::buffer::HotBuffer;
use crate::locks::LockExt;
use crate::storage::{
    DetectionDebugStore, DetectionStore, EventRef, MapKind, MotionStore, RangeRequest, ServedRange,
    ThumbnailError, VideoStream, WarmEventEntry, WarmStorageBackend,
};

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
    /// Warm storage backend (local disk today). `None` when storage is disabled.
    pub storage: Option<Arc<dyn WarmStorageBackend>>,
    /// Per-camera deterministic motion settings. `None` when analytics is off.
    pub motion_settings: Option<MotionSettingsStore>,
}

impl AppState {
    pub fn new(
        buffers: HashMap<String, Arc<RwLock<HotBuffer>>>,
        motion_store: MotionStore,
        detection_store: DetectionStore,
        debug_store: DetectionDebugStore,
        storage: Option<Arc<dyn WarmStorageBackend>>,
        motion_settings: Option<MotionSettingsStore>,
    ) -> Self {
        Self {
            buffers: Arc::new(buffers),
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
}

/// SHA-256 of the configured `[http] token`. The presented token is hashed the
/// same way before comparison: `==` on `[u8; 32]` is not guaranteed to be
/// constant-time, but it runs over two fixed-width digests, so how far it gets
/// says nothing usable about the secret's length or content.
#[derive(Clone)]
struct TokenAuth(Arc<[u8; 32]>);

fn token_digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// The `?token=` fallback, for requests that cannot carry headers: `<img>`
/// sources (thumbnails, filmstrips, debug maps) and native video elements.
/// Those are reads, so the fallback is confined to GET and HEAD — anything
/// that changes state must present the header.
#[derive(Deserialize)]
struct TokenQuery {
    token: Option<String>,
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

async fn require_token(State(auth): State<TokenAuth>, request: Request, next: Next) -> Response {
    let query_fallback_allowed = matches!(*request.method(), Method::GET | Method::HEAD);
    let presented = match bearer_token(request.headers()) {
        Some(token) => Some(token.to_string()),
        None if query_fallback_allowed => Query::<TokenQuery>::try_from_uri(request.uri())
            .ok()
            .and_then(|q| q.0.token),
        None => None,
    };

    match presented {
        Some(token) if token_digest(&token) == *auth.0 => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "unauthorized",
        )
            .into_response(),
    }
}

/// True when the API can be reached from another machine with no token — the
/// configuration [`warn_if_open`] shouts about.
fn is_open_to_network(bind: IpAddr, token: Option<&str>) -> bool {
    token.is_none() && !bind.is_loopback()
}

/// Warn loudly at startup when the API is reachable off-box without a token.
/// `allow_open` silences it for deployments that authenticate one layer out.
pub fn warn_if_open(bind: IpAddr, token: Option<&str>, allow_open: bool) {
    if allow_open || !is_open_to_network(bind, token) {
        return;
    }
    tracing::warn!(
        %bind,
        "THE API IS OPEN: anyone who can reach this address can watch all footage and \
         change motion settings. Set [http] token to require a token, or [http] bind to \
         \"127.0.0.1\" to keep it on this machine. Set [http] allow_open = true to silence \
         this if something in front of camon already authenticates."
    );
}

/// The UI shell (`/` and `/assets/*`) stays unauthenticated so the token prompt
/// can load; everything under `/api` needs the token once one is configured.
pub fn build_router(state: AppState, token: Option<&str>) -> Router {
    let mut api = api_routes().with_state(state);
    if let Some(token) = token {
        api = api.route_layer(middleware::from_fn_with_state(
            TokenAuth(Arc::new(token_digest(token))),
            require_token,
        ));
    }

    Router::new()
        .route("/", get(index_handler))
        .route("/assets/{*path}", get(static_handler))
        .merge(api)
}

/// Take the listening socket, and fail startup if it cannot be had.
///
/// Separate from [`serve`] because of *when* it has to happen. Binding inside
/// the server task meant an address already in use, or a `[http] bind` this
/// host does not have, was one `tracing::error!` line inside a detached task:
/// camon went on recording with no UI, no ingress and no API, and — being
/// alive — was never restarted by systemd or the Home Assistant Supervisor,
/// which are the only things that could have fixed it. Bound here, before a
/// single camera is spawned, the same failure ends startup with a nonzero exit
/// and the operator gets the address in the message.
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
    token: Option<String>,
) -> Result<(), std::io::Error> {
    axum::serve(listener, build_router(state, token.as_deref())).await
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
        Some(buffer) => {
            let buf = buffer.read_recover();
            let playlist = hls::generate_playlist(&buf, tail_count);
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
) -> Response {
    match state.buffers.get(&id) {
        Some(buffer) => {
            // Clone the Arc under the lock; copy bytes for the response only
            // after the guard is dropped so the ingest thread isn't blocked.
            let data = {
                let buf = buffer.read_recover();
                hls::generate_segment(&buf, n)
            };
            match data {
                Some(data) => {
                    ([(header::CONTENT_TYPE, "video/mp2t")], (*data).clone()).into_response()
                }
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

/// JSON shape for the motion-settings endpoints. Carries the current values,
/// the mask grid geometry (shared by the movement and detection masks), and the
/// slider bounds so the UI can build its controls without hard-coding them.
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
        // The request is the problem, and the settings the detector is using
        // are untouched — the client's, not camon's, and not a partial apply.
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
    let events = backend.query(&id, from, to);

    let response: Vec<WarmEventResponse> = events
        .iter()
        .map(|e| WarmEventResponse {
            start_pts_ns: e.start_pts_ns.to_string(),
            duration_ms: e.duration_ms,
            // The one spelling of a type: the same `as_str` an event key in a
            // URL is built from, so a listing and the keys made from it cannot
            // disagree about what a type is called.
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
///
/// The segment is a whole [`EventRef`] — `{start_pts_ns}_{duration_ms}_{type}` —
/// because a start PTS alone identifies nothing: events sharing one are two
/// recordings, and these routes used to serve whichever of them a search on the
/// start returned. So a segment that is not a key is a `400`, exactly as an
/// unparseable start PTS was, and only a key with an event behind it reads
/// anything.
///
/// `Err` carries the status and message to answer with.
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
///
/// Returns `None` — meaning "serve the whole object (200)" — for an absent
/// header, a non-`bytes` unit, syntactic garbage, a reversed range, or a
/// multi-range request (`a-b,c-d`): we do not implement `multipart/byteranges`,
/// so we deliberately fall back to the full body rather than erroring.
fn parse_range_header(value: &str) -> Option<RangeRequest> {
    let spec = value.trim().strip_prefix("bytes=")?;
    // Multi-range → decline; the caller serves the full body.
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
        // Reversed range is invalid syntax → ignore, serve full.
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
        Err(_) => (StatusCode::NOT_FOUND, "event file not found").into_response(),
    }
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
        ServedRange::Partial { start, end } => (
            StatusCode::PARTIAL_CONTENT,
            [
                (header::CONTENT_TYPE, "video/mp2t".to_string()),
                (header::ACCEPT_RANGES, "bytes".to_string()),
                (
                    header::CONTENT_RANGE,
                    format!("bytes {start}-{end}/{total_size}"),
                ),
                (header::CONTENT_LENGTH, (end - start + 1).to_string()),
            ],
            Body::from_stream(stream),
        )
            .into_response(),
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

    // The backend acquires the poster frame (LocalDisk lazily renders + caches
    // it via ffmpeg); every failure here is an internal error.
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

/// The debug view's poll — and, by being it, the one thing that tells the
/// detector somebody is watching. The store opens a demand window on this
/// request; the analyzer and the detection worker produce and keep frames only
/// while it is open, so a route that stops reaching the store here leaves the
/// view permanently empty. See [`DetectionDebugStore::wanted`].
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

    if index > 3 {
        return (StatusCode::BAD_REQUEST, "index must be 0-3").into_response();
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
    use crate::storage::EventType;

    const TOKEN: &str = "s3cr3t token";

    /// Bind the router to an ephemeral loopback port and return its base URL.
    /// Requests go over real HTTP so the middleware is exercised exactly as it
    /// is in production, headers and query string included.
    async fn serve(token: Option<&str>) -> String {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let state = AppState::new(
            buffers,
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            None,
        );
        let app = build_router(state, token);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// Like [`serve`], but hands back the motion store so a test can publish
    /// what the debug endpoints read.
    async fn serve_with_motion_store() -> (String, MotionStore) {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let motion_store = MotionStore::new(&ids);
        let state = AppState::new(
            buffers,
            motion_store.clone(),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            None,
        );
        let app = build_router(state, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), motion_store)
    }

    /// Like [`serve`], but hands back the detection debug store, whose demand
    /// window this API is the only opener of.
    async fn serve_with_debug_store() -> (String, DetectionDebugStore) {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let debug_store = DetectionDebugStore::new(&ids);
        let state = AppState::new(
            buffers,
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            debug_store.clone(),
            None,
            None,
        );
        let app = build_router(state, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), debug_store)
    }

    /// The detector's debug frames are a megabyte a run and are kept only while
    /// somebody is watching. This route is what says somebody is: the analyzer
    /// and the detection worker both ask the store, and nothing else ever
    /// answers yes. If the entry list stops registering the request, the view
    /// goes permanently blank.
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

        // And with the window open, what the worker stores is what the view
        // serves — all the way to the JPEG bytes.
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

        // The images are the seam: they are fetched as a consequence of a list
        // the viewer already has, so they must not be able to hold the window
        // open by themselves. A cached page, or a URL somebody bookmarked,
        // would otherwise keep the detector encoding frames for nobody.
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

    /// Every stage the UI asks for has to resolve to its own map — one route
    /// serving them all makes a mixed-up stage a silent wrong picture rather
    /// than a compile error.
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

    /// Like [`serve_with_storage`], but holding two events that start on the
    /// same keyframe: a 2-second movement event with one filmstrip frame, and
    /// the 4-second continuous chunk covering it. Returns the base URL, the temp
    /// dir (which must outlive the server) and the two indexed entries.
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
        let events = storage.query("cam", 0, u64::MAX);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].start_pts_ns, events[1].start_pts_ns);

        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        let state = AppState::new(
            buffers,
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            Some(storage),
            None,
        );
        let app = build_router(state, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), dir, events)
    }

    /// Two recordings begin on the same keyframe — a motion event and the
    /// continuous chunk covering it — and each of these routes has to answer
    /// with the one its URL names.
    ///
    /// The URL carries the whole key, because it used to carry the start PTS
    /// alone and that names both events: the lookup binary-searched it and
    /// answered every one of these requests with whichever of the two it landed
    /// on. A viewer asking for the movement got the continuous chunk's video,
    /// under the movement's duration in the manifest, with its filmstrip
    /// missing — and nothing in the exchange said so.
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

            // Only the movement event has a filmstrip; the continuous chunk's
            // frame 0 does not exist, and must not resolve to the movement's.
            let frame = reqwest::get(format!("{events}/filmstrip/0")).await.unwrap();
            match event_type {
                EventType::Movement => {
                    assert_eq!(frame.status(), reqwest::StatusCode::OK, "{key}");
                    assert_eq!(frame.bytes().await.unwrap().as_ref(), [0xaa]);
                }
                _ => assert_eq!(frame.status(), reqwest::StatusCode::NOT_FOUND, "{key}"),
            }
        }

        // A key nothing is stored under is a 404 — including the third type at
        // a start two other events do hold.
        let missing = EventRef::new(start, 2000, EventType::Object);
        let status = reqwest::get(format!(
            "{base}/api/cameras/cam/events/{missing}/playlist.m3u8"
        ))
        .await
        .unwrap()
        .status();
        assert_eq!(status, reqwest::StatusCode::NOT_FOUND);

        // A segment that is not a key at all is a 400, as an unparseable start
        // PTS was — the bare start PTS these routes used to take included.
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

    /// Like [`serve`], but with warm storage enabled over an empty data dir.
    /// The returned directory must outlive the server.
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
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            Some(storage),
            None,
        );
        let app = build_router(state, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), dir)
    }

    /// State with one camera whose motion settings live under `data_dir`.
    fn motion_settings_state(data_dir: &std::path::Path) -> AppState {
        let ids = vec!["cam".to_string()];
        let buffers = HashMap::from([("cam".to_string(), HotBuffer::new("cam".to_string(), 60))]);
        AppState::new(
            buffers,
            MotionStore::new(&ids),
            DetectionStore::new(&ids),
            DetectionDebugStore::new(&ids),
            None,
            Some(MotionSettingsStore::new(&ids, data_dir, 16.0, 200.0)),
        )
    }

    /// Serve with motion settings backed by `data_dir`.
    async fn serve_with_motion_settings(data_dir: &std::path::Path) -> String {
        let app = build_router(motion_settings_state(data_dir), None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// A settings PUT that cannot be made durable must not answer 200 — the
    /// operator would take a painted mask as saved and lose it on restart.
    #[tokio::test]
    async fn a_settings_put_that_cannot_be_persisted_is_not_a_success() {
        let dir = tempfile::tempdir().unwrap();
        // A regular file where the camera's directory belongs: nothing can be
        // written underneath it, whatever user the test runs as.
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

        // The value is live regardless, and the GET says so.
        let live: serde_json::Value =
            reqwest::get(format!("{base}/api/cameras/cam/motion/settings"))
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
        assert_eq!(live["var_threshold"], 32.0);
    }

    /// A slider value that is not a real number would switch motion detection
    /// off for that camera — `area >= NaN` is false for every blob — so no
    /// route into the store accepts one and the live settings are left alone.
    /// Over HTTP the body never gets that far: JSON has no way to spell a
    /// non-finite number, so serde refuses it first. The store's own refusal,
    /// which is what catches any other writer, is pinned in `motion_settings`.
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

        // Refused by serde as it is deserialized, before any handler runs —
        // which is the point being pinned here, not the store's own guard: no
        // spelling of a non-finite number survives the JSON body. The guard
        // itself is covered by the test below, which calls the handler.
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

    /// What the store's own refusal answers, reached by calling the handler
    /// directly: no JSON body can carry a non-finite number today, so this is
    /// the only way to see the arm that will answer for the writer that
    /// eventually can. A bad request, not a server fault — camon's state is
    /// fine, and untouched.
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

        // The writable camera still answers 200 on the same route.
        let response = reqwest::Client::new()
            .put(format!("{base}/api/cameras/cam/motion/settings"))
            .json(&serde_json::json!({ "var_threshold": 32.0 }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
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

        // The same route with the bounds the right way round still answers.
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
        let base = serve(Some(TOKEN)).await;
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
        let base = serve(Some(TOKEN)).await;
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
        let base = serve(Some(TOKEN)).await;
        // Media route with a query parameter of its own: the token must coexist
        // with it, and arrive percent-decoded.
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
        let base = serve(Some(TOKEN)).await;
        let status = reqwest::get(format!("{base}/api/cameras?token=wrong"))
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
    }

    /// State-changing routes are behind the same layer, and the `?token=`
    /// fallback — meant for `<img>` and native video, which only ever GET —
    /// does not reach them: a write must present the header.
    #[tokio::test]
    async fn writes_require_the_header_and_never_the_query_token() {
        let base = serve(Some(TOKEN)).await;
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

        // The header still works for writes; 404 is this build's answer once
        // authorized (motion settings are disabled in the test state), and
        // crucially it is not a 401.
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
        let base = serve(Some(TOKEN)).await;
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
    async fn no_configured_token_leaves_the_api_open() {
        let base = serve(None).await;
        let status = reqwest::get(format!("{base}/api/cameras"))
            .await
            .unwrap()
            .status();
        assert_eq!(status, reqwest::StatusCode::OK);
    }

    #[test]
    fn open_api_warning_covers_exactly_the_unprotected_network_case() {
        let all = IpAddr::from([0, 0, 0, 0]);
        let loopback = IpAddr::from([127, 0, 0, 1]);
        assert!(is_open_to_network(all, None));
        assert!(!is_open_to_network(all, Some(TOKEN)));
        assert!(!is_open_to_network(loopback, None));
        assert!(!is_open_to_network("::1".parse().unwrap(), None));
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
        // Whitespace around the value is tolerated.
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
        // Not a byte range at all.
        assert_eq!(parse_range_header("bytes=abc"), None);
        assert_eq!(parse_range_header("items=0-1"), None);
        assert_eq!(parse_range_header("bytes=10"), None);
        assert_eq!(parse_range_header("bytes="), None);
        assert_eq!(parse_range_header("bytes=-"), None);
        assert_eq!(parse_range_header("garbage"), None);
        // Reversed range is invalid → decline (serve full).
        assert_eq!(parse_range_header("bytes=20-10"), None);
        // Multi-range is unsupported → decline (serve full).
        assert_eq!(parse_range_header("bytes=0-10,20-30"), None);
    }
}
