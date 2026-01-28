use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;

use crate::buffer::HotBuffer;

use super::hls;

#[derive(Embed)]
#[folder = "src/assets/"]
struct Assets;

#[derive(Clone)]
pub struct AppState {
    pub buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
}

impl AppState {
    pub fn new(buffers: HashMap<String, Arc<RwLock<HotBuffer>>>) -> Self {
        Self {
            buffers: Arc::new(buffers),
        }
    }
}

pub async fn start_server(state: AppState, port: u16) -> Result<(), std::io::Error> {
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/assets/{*path}", get(static_handler))
        .route("/api/cameras", get(cameras_handler))
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
