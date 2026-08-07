use super::*;
use crate::storage::event_index::DetectionDetail;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, RawQuery, State},
    http::{HeaderMap, StatusCode},
    routing::any,
    Json, Router,
};

// ---- in-process stathost stub -----------------------------------------

#[derive(Clone)]
struct Stub {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    token: String,
    fail_writes: Arc<AtomicBool>,
    /// Breaks one object of an event while the others go through.
    put_fault: Arc<Mutex<Option<PutFault>>>,
    /// When set, GETs whose path ends with this suffix answer `500` — an
    /// unreadable object, as distinct from an absent one (`404`).
    fail_get_suffix: Arc<Mutex<Option<String>>>,
    /// DELETEs of exactly these paths answer `500`: an object the store
    /// refuses to drop, as distinct from one already gone (`404`).
    fail_delete_paths: Arc<Mutex<HashSet<String>>>,
    /// When set, GET ignores an incoming `Range` and answers a full `200` —
    /// a legal HTTP response the client must handle by replaying in full.
    ignore_range: Arc<AtomicBool>,
    /// When set, a ranged GET answers `206` with this `Content-Range`
    /// instead of the true one; an empty string omits the header
    /// altogether. A broken or hostile store, in other words.
    bad_content_range: Arc<Mutex<Option<String>>>,
    /// Paths that appear the instant after a listing is served: an upload
    /// committing while a scan walks the snapshot it took.
    commit_after_list: Arc<Mutex<Vec<String>>>,
    /// Listings still to be refused with a `500` before the store starts
    /// answering — a host that is not up yet.
    list_failures: Arc<AtomicUsize>,
    /// Latency added to every listing. A value larger than any timeout is a
    /// host that accepted the connection and then said nothing, which is
    /// the failure that costs a whole request timeout rather than a
    /// millisecond.
    list_delay_ms: Arc<AtomicU64>,
    /// Listings asked for, refusals included: how many times the client
    /// came back.
    lists: Arc<AtomicUsize>,
    /// Every GET path served, in arrival order — what a caller asked for,
    /// and how many times.
    gets: Arc<Mutex<Vec<String>>>,
    /// GETs currently being served, and the high-water mark: the client's
    /// fan-out width as the server actually saw it.
    in_flight: Arc<AtomicUsize>,
    peak_gets: Arc<AtomicUsize>,
    /// Latency added to every GET, so a serial caller is distinguishable
    /// from a concurrent one by the clock.
    get_delay_ms: Arc<AtomicU64>,
    /// Every PUT path served, in arrival order — what was uploaded, in what
    /// order, and how many times each was attempted.
    puts: Arc<Mutex<Vec<String>>>,
    /// Latency added to every PUT: a slow uplink, and the window a test
    /// needs to raise the shutdown flag *inside* an upload.
    put_delay_ms: Arc<AtomicU64>,
    /// Every DELETE path served, in arrival order. What a pass reclaimed —
    /// and, for a pass that should have reclaimed nothing, proof that it
    /// did not.
    deletes: Arc<Mutex<Vec<String>>>,
    /// Latency added to every DELETE, for the tests that need to look at
    /// the backend's accounting from inside a deletion.
    delete_delay_ms: Arc<AtomicU64>,
    /// Requests of this method are held open until the test lets them go.
    ///
    /// This is how a test gets *inside* a request — to raise the shutdown
    /// flag, run a sweep, or read the backend's accounting while an upload
    /// is in flight — without picking a millisecond figure and hoping the
    /// machine is quick enough to beat it. A gate the test opens is an
    /// instrument; a sleep it races is a coin toss on a loaded box.
    hold_puts: Arc<AtomicBool>,
    hold_deletes: Arc<AtomicBool>,
    hold_gets: Arc<AtomicBool>,
}

/// A PUT failure injected by path suffix. `stored` decides whether the
/// object lands anyway before the error is returned — the shape of an
/// upload timeout or a proxy 5xx over a body the origin already committed,
/// which a client cannot tell from an upload that never happened.
#[derive(Clone)]
struct PutFault {
    suffix: String,
    stored: bool,
    /// What the client is told. A `5xx` is a store having a moment; a `4xx`
    /// is the store refusing this request and every identical one after it.
    status: StatusCode,
}

impl Stub {
    fn fail_puts(&self, suffix: &str, stored: bool) {
        self.refuse_puts(suffix, stored, StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// Fail PUTs with a chosen status, so a test can tell a store that is
    /// having a moment from one that is refusing the request itself.
    fn refuse_puts(&self, suffix: &str, stored: bool, status: StatusCode) {
        *self.put_fault.lock().unwrap() = Some(PutFault {
            suffix: suffix.to_string(),
            stored,
            status,
        });
    }

    /// How many times this exact path was uploaded.
    fn put_count(&self, path: &str) -> usize {
        self.puts
            .lock()
            .unwrap()
            .iter()
            .filter(|p| *p == path)
            .count()
    }

    fn fail_gets(&self, suffix: &str) {
        *self.fail_get_suffix.lock().unwrap() = Some(suffix.to_string());
    }

    fn clear_faults(&self) {
        *self.put_fault.lock().unwrap() = None;
        *self.fail_get_suffix.lock().unwrap() = None;
    }

    /// Refuse the next `n` listings. A whole scan's worth of them is the
    /// boot-time race the un-scanned state exists for: the store is not
    /// answering yet, and everything else the client does still works.
    fn fail_next_lists(&self, n: usize) {
        self.list_failures.store(n, Ordering::SeqCst);
    }

    /// The store comes back.
    fn serve_lists_again(&self) {
        self.list_failures.store(0, Ordering::SeqCst);
    }

    /// Answer listings this slowly. With a delay longer than any timeout
    /// the client is willing to wait, this is the host that accepts a
    /// connection and then goes quiet — the failure that costs a whole
    /// request timeout instead of a millisecond.
    fn hang_lists(&self, delay: Duration) {
        self.list_delay_ms
            .store(delay.as_millis() as u64, Ordering::SeqCst);
    }

    fn lists(&self) -> usize {
        self.lists.load(Ordering::SeqCst)
    }

    fn has(&self, path: &str) -> bool {
        self.files.lock().unwrap().contains_key(path)
    }

    /// Every byte the host is holding. The client-side budget is a claim
    /// about exactly this number, so a test that has written its way to a
    /// known store checks the claim against the thing it is about rather
    /// than against arithmetic copied out of the code under test — which
    /// is how the budget came to count videos and nothing else.
    fn stored_bytes(&self) -> u64 {
        self.files
            .lock()
            .unwrap()
            .values()
            .map(|v| v.len() as u64)
            .sum()
    }

    /// Every GET served so far, clearing the record — so a test can count
    /// what one phase asked for without the setup's requests in the total.
    fn take_gets(&self) -> Vec<String> {
        std::mem::take(&mut self.gets.lock().unwrap())
    }

    /// Every PUT served so far, clearing the record.
    fn take_puts(&self) -> Vec<String> {
        std::mem::take(&mut self.puts.lock().unwrap())
    }

    /// Every DELETE served so far, clearing the record.
    fn take_deletes(&self) -> Vec<String> {
        std::mem::take(&mut self.deletes.lock().unwrap())
    }

    /// Hold every request of this kind open — they arrive, are recorded,
    /// and then wait — until [`Stub::release`] lets them through.
    fn hold(&self, gate: &Arc<AtomicBool>) {
        gate.store(true, Ordering::SeqCst);
    }

    fn release(&self, gate: &Arc<AtomicBool>) {
        gate.store(false, Ordering::SeqCst);
    }

    /// How many times this exact path was fetched.
    fn get_count(&self, path: &str) -> usize {
        self.gets
            .lock()
            .unwrap()
            .iter()
            .filter(|p| *p == path)
            .count()
    }
}

/// Drain a [`VideoStream`] body to bytes (test-only).
async fn drain(vs: VideoStream) -> Vec<u8> {
    use futures_util::StreamExt;
    let mut buf = Vec::new();
    let mut stream = vs.stream;
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.unwrap());
    }
    buf
}

/// Resolve a single-range `Range` header against `total`, mirroring real
/// stathost semantics: `Some((start, end))` inclusive, or `None` for a
/// `416`-worthy unsatisfiable range.
fn parse_stub_range(header: &str, total: u64) -> Option<(u64, u64)> {
    let spec = header.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (s, e) = spec.split_once('-')?;
    if s.is_empty() {
        let n: u64 = e.trim().parse().ok()?;
        if n == 0 || total == 0 {
            return None;
        }
        let n = n.min(total);
        return Some((total - n, total - 1));
    }
    let start: u64 = s.trim().parse().ok()?;
    if start >= total {
        return None;
    }
    let end = if e.trim().is_empty() {
        total - 1
    } else {
        e.trim().parse::<u64>().ok()?.min(total - 1)
    };
    if end < start {
        return None;
    }
    Some((start, end))
}

fn authorized(headers: &HeaderMap, token: &str) -> bool {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == format!("Bearer {token}"))
        .unwrap_or(false)
}

async fn handler(
    State(stub): State<Stub>,
    Path((_bucket, path)): Path<(String, String)>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    if path == "_meta/list" {
        if !authorized(&headers, &stub.token) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        stub.lists.fetch_add(1, Ordering::SeqCst);
        let delay = stub.list_delay_ms.load(Ordering::Relaxed);
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        if stub.list_failures.load(Ordering::SeqCst) > 0 {
            stub.list_failures.fetch_sub(1, Ordering::SeqCst);
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
        // Mirrors stathost >= 0.2.0: plain array without ?detail=true,
        // [{"path","size","mtime"}] with it.
        let detail = query.as_deref() == Some("detail=true");
        let files = stub.files.lock().unwrap();
        let mut paths: Vec<String> = files.keys().cloned().collect();
        paths.sort();
        let response = if detail {
            let arr: Vec<serde_json::Value> = paths
                .iter()
                .map(|p| serde_json::json!({"path": p, "size": files[p].len(), "mtime": 0}))
                .collect();
            Json(arr).into_response()
        } else {
            Json(paths).into_response()
        };
        drop(files);
        // Whatever was landing while the snapshot was taken lands now.
        for path in stub.commit_after_list.lock().unwrap().drain(..) {
            stub.files.lock().unwrap().insert(path, vec![0u8; 10]);
        }
        return response;
    }

    match method {
        axum::http::Method::PUT => {
            if !authorized(&headers, &stub.token) {
                return StatusCode::UNAUTHORIZED.into_response();
            }
            if stub.fail_writes.load(Ordering::Relaxed) {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            stub.puts.lock().unwrap().push(path.clone());
            wait_on_gate(&stub.hold_puts).await;
            let delay = stub.put_delay_ms.load(Ordering::Relaxed);
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            let fault = stub.put_fault.lock().unwrap().clone();
            if let Some(fault) = fault.filter(|f| path.ends_with(&f.suffix)) {
                if fault.stored {
                    stub.files.lock().unwrap().insert(path, body.to_vec());
                }
                return fault.status.into_response();
            }
            stub.files.lock().unwrap().insert(path, body.to_vec());
            StatusCode::OK.into_response()
        }
        axum::http::Method::DELETE => {
            if !authorized(&headers, &stub.token) {
                return StatusCode::UNAUTHORIZED.into_response();
            }
            stub.deletes.lock().unwrap().push(path.clone());
            wait_on_gate(&stub.hold_deletes).await;
            let delay = stub.delete_delay_ms.load(Ordering::Relaxed);
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            if stub.fail_delete_paths.lock().unwrap().contains(&path) {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            if stub.files.lock().unwrap().remove(&path).is_some() {
                StatusCode::OK.into_response()
            } else {
                StatusCode::NOT_FOUND.into_response()
            }
        }
        // GET is public.
        _ => {
            stub.gets.lock().unwrap().push(path.clone());
            let now = stub.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            stub.peak_gets.fetch_max(now, Ordering::SeqCst);
            // Read first, then take the latency: a slow response carries
            // what the object was when the request arrived. That is what a
            // real one does, and it is the only way a test can put a write
            // *inside* the window of a read that is already under way.
            let resp = get_response(&stub, &path, &headers);
            wait_on_gate(&stub.hold_gets).await;
            let delay = stub.get_delay_ms.load(Ordering::Relaxed);
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            stub.in_flight.fetch_sub(1, Ordering::SeqCst);
            resp
        }
    }
}

fn get_response(stub: &Stub, path: &str, headers: &HeaderMap) -> axum::response::Response {
    use axum::response::IntoResponse;

    let fail_get = stub.fail_get_suffix.lock().unwrap().clone();
    if fail_get.is_some_and(|s| path.ends_with(&s)) {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    let bytes = match stub.files.lock().unwrap().get(path) {
        Some(bytes) => bytes.clone(),
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    let total = bytes.len() as u64;
    let range = headers.get("range").and_then(|v| v.to_str().ok());
    match range {
        // A server may legally answer a range request with a full 200.
        Some(_) if stub.ignore_range.load(Ordering::Relaxed) => full_200(bytes),
        Some(r) => match parse_stub_range(r, total) {
            Some((start, end)) => {
                let slice = bytes[start as usize..=end as usize].to_vec();
                let mut resp = (StatusCode::PARTIAL_CONTENT, slice).into_response();
                let content_range = match stub.bad_content_range.lock().unwrap().clone() {
                    // The header omitted entirely.
                    Some(header) if header.is_empty() => return resp,
                    Some(header) => header,
                    None => format!("bytes {start}-{end}/{total}"),
                };
                resp.headers_mut()
                    .insert("content-range", content_range.parse().unwrap());
                resp.headers_mut()
                    .insert("accept-ranges", "bytes".parse().unwrap());
                resp
            }
            None => {
                let mut resp = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
                resp.headers_mut()
                    .insert("content-range", format!("bytes */{total}").parse().unwrap());
                resp
            }
        },
        None => full_200(bytes),
    }
}

/// A `200 OK` full body advertising `Accept-Ranges: bytes`.
fn full_200(bytes: Vec<u8>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let mut resp = bytes.into_response();
    resp.headers_mut()
        .insert("accept-ranges", "bytes".parse().unwrap());
    resp
}

async fn spawn_stub(token: &str) -> (String, Stub) {
    let stub = Stub {
        files: Arc::new(Mutex::new(HashMap::new())),
        token: token.to_string(),
        fail_writes: Arc::new(AtomicBool::new(false)),
        put_fault: Arc::new(Mutex::new(None)),
        fail_get_suffix: Arc::new(Mutex::new(None)),
        fail_delete_paths: Arc::new(Mutex::new(HashSet::new())),
        ignore_range: Arc::new(AtomicBool::new(false)),
        bad_content_range: Arc::new(Mutex::new(None)),
        commit_after_list: Arc::new(Mutex::new(Vec::new())),
        list_failures: Arc::new(AtomicUsize::new(0)),
        list_delay_ms: Arc::new(AtomicU64::new(0)),
        lists: Arc::new(AtomicUsize::new(0)),
        gets: Arc::new(Mutex::new(Vec::new())),
        in_flight: Arc::new(AtomicUsize::new(0)),
        peak_gets: Arc::new(AtomicUsize::new(0)),
        get_delay_ms: Arc::new(AtomicU64::new(0)),
        puts: Arc::new(Mutex::new(Vec::new())),
        put_delay_ms: Arc::new(AtomicU64::new(0)),
        deletes: Arc::new(Mutex::new(Vec::new())),
        delete_delay_ms: Arc::new(AtomicU64::new(0)),
        hold_puts: Arc::new(AtomicBool::new(false)),
        hold_deletes: Arc::new(AtomicBool::new(false)),
        hold_gets: Arc::new(AtomicBool::new(false)),
    };
    let app = Router::new()
        .route("/{bucket}/{*path}", any(handler))
        .with_state(stub.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), stub)
}

fn backend_for(url: &str, token: &str, max_stored_bytes: u64) -> StathostBackend {
    backend_stopped_by(url, token, max_stored_bytes, StopFlag::never())
}

/// A backend that shares `stop` with whoever raises it — the drain, in
/// production; a test, here.
fn backend_stopped_by(
    url: &str,
    token: &str,
    max_stored_bytes: u64,
    stop: StopFlag,
) -> StathostBackend {
    let config = StathostConfig {
        url: url.to_string(),
        bucket: "cams".to_string(),
        token: token.to_string(),
        max_stored_bytes,
        enabled: true,
    };
    StathostBackend::new(&config, &["cam".to_string()], stop).expect("http clients build")
}

/// A scanned backend owning more than one camera — the ordinary
/// installation, and the one a per-backend cursor gets wrong.
async fn scanned_backend_with_cameras(url: &str, cameras: &[&str]) -> StathostBackend {
    let config = StathostConfig {
        url: url.to_string(),
        bucket: "cams".to_string(),
        token: "secret".to_string(),
        max_stored_bytes: 0,
        enabled: true,
    };
    let ids: Vec<String> = cameras.iter().map(|c| c.to_string()).collect();
    let backend =
        StathostBackend::new(&config, &ids, StopFlag::never()).expect("http clients build");
    backend.scan().await.unwrap();
    backend
}

/// Block while `gate` is raised, so a test can act from inside a request
/// that has arrived and not yet been answered. Bounded so a test that
/// forgets to release fails rather than hanging the suite.
async fn wait_on_gate(gate: &Arc<AtomicBool>) {
    for _ in 0..10_000 {
        if !gate.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// Wait for something the stub is about to be told, polling because what is
/// being waited for is a real request arriving at a real socket. Panics
/// rather than hanging the suite if it never happens.
async fn wait_until(mut condition: impl FnMut() -> bool) {
    for _ in 0..5_000 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the stub never reached the state this test needs");
}

/// A backend whose store already holds `events` and whose budget does not
/// fit them.
///
/// Seeded through a second, unlimited backend and then scanned in, rather
/// than written through this one: a write now makes room for itself before
/// it uploads (see [`StathostBackend::make_room`]), so writing an
/// over-budget store into existence through the very guard under test would
/// evict it on the way in. What this builds is the state an operator
/// actually arrives at — a budget lowered, or a restart onto a store that
/// has grown past one.
async fn over_budget_backend(
    url: &str,
    events: &[FinishedEvent],
    max_stored_bytes: u64,
) -> StathostBackend {
    let seeder = backend_for(url, "secret", 0);
    for event in events {
        assert_eq!(
            seeder.write_event("cam", event).await,
            WriteOutcome::Written
        );
    }
    let backend = backend_for(url, "secret", max_stored_bytes);
    backend.scan().await.unwrap();
    backend
}

/// A backend that has already been through a startup scan, which is the
/// only state `init_storage` ever hands on: retention and budget eviction
/// refuse to act on an index no scan has filled, so a test that prunes or
/// evicts has to start from one. The store is empty at this point, so the
/// scan costs one listing and indexes nothing.
async fn scanned_backend_for(url: &str, token: &str, max_stored_bytes: u64) -> StathostBackend {
    let backend = backend_for(url, token, max_stored_bytes);
    backend.scan().await.unwrap();
    backend
}

// ---- event fixtures ---------------------------------------------------

fn segment(start_pts: u64, byte: u8, len: usize) -> GopSegment {
    GopSegment {
        start_pts,
        duration_ns: 1_000_000_000,
        data: Arc::new(vec![byte; len]),
        frame_count: 1,
    }
}

/// A movement event at `first_pts` (1s long), `size` bytes of video, with
/// two filmstrip frames.
fn movement_event(first_pts: u64, size: usize) -> FinishedEvent {
    FinishedEvent {
        segments: vec![segment(first_pts, 0xab, size)],
        first_pts,
        total_bytes: size,
        has_objects: false,
        object_classes: Vec::new(),
        filmstrip_frames: Some(Arc::new(vec![vec![0x01, 0x02], vec![0x03, 0x04]])),
        backend: None,
        model: None,
        detection_details: Vec::new(),
        continues: false,
        is_continuous: false,
    }
}

/// A movement event carrying a detection — the type only the sidecar can
/// record on a store without directories.
fn object_event(first_pts: u64, size: usize) -> FinishedEvent {
    let mut e = movement_event(first_pts, size);
    e.has_objects = true;
    e.object_classes = vec!["car".to_string()];
    e.detection_details = vec![DetectionDetail {
        class: "car".to_string(),
        confidence: 0.8,
    }];
    e
}

/// A second event at the same start PTS, twice as long: same start,
/// different stem, and so a different set of objects on the host. Nothing
/// enforces the uniqueness of a start PTS, which is why an event is
/// identified by its stem and not by where a binary search on the start
/// happens to land.
fn longer_movement_event(first_pts: u64, size: usize) -> FinishedEvent {
    let mut e = movement_event(first_pts, size);
    e.segments.push(segment(first_pts + SEC, 0xcd, size));
    e.total_bytes = size * 2;
    e
}

/// The key an API request carries for one stem. This backend resolves by
/// stem alone and ignores the type in the key (see its `find_event`), so
/// these lookups name `Movement` whatever the event turns out to be — a
/// deliberate choice, pinned by
/// `find_event_resolves_by_stem_across_an_upgrade`.
fn url_key(start_pts_ns: u64, duration_ms: u32) -> EventRef {
    EventRef::new(start_pts_ns, duration_ms, EventType::Movement)
}

fn sibling(backend: &StathostBackend, duration_ms: u32) -> Option<WarmEventEntry> {
    backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .into_iter()
        .find(|e| e.duration_ms == duration_ms)
}

fn continuous_event(first_pts: u64, size: usize) -> FinishedEvent {
    let mut e = movement_event(first_pts, size);
    e.is_continuous = true;
    e.filmstrip_frames = None;
    e
}

/// What one event will cost the store: its video, the sidecar that types
/// it, and its filmstrip frames. Derived from the event rather than
/// hardcoded, because the accounting fix is precisely that the metadata
/// counts — a budget expressed as a multiple of the video alone would pin
/// the bug back in place.
fn cost_of(event: &FinishedEvent) -> u64 {
    let sidecar = sidecar_json(
        Some(event.event_type()),
        event.backend.as_deref(),
        event.model.as_deref(),
        &event.detection_details,
        event.continues,
    )
    .len() as u64;
    let frames: u64 = event
        .filmstrip_frames
        .iter()
        .flat_map(|f| f.iter())
        .map(|f| f.len() as u64)
        .sum();
    event.total_bytes as u64 + sidecar + frames
}

fn upgrade_for(first_pts: u64) -> EventUpgrade {
    EventUpgrade {
        start_pts_ns: first_pts,
        duration_ms: 1000,
        object_classes: vec!["person".to_string()],
        detections: vec![DetectionDetail {
            class: "person".to_string(),
            confidence: 0.9,
        }],
        backend: "ollama".to_string(),
        model: "m".to_string(),
        continues: false,
    }
}

// ---- tests ------------------------------------------------------------

#[tokio::test]
async fn write_then_scan_round_trip_detailed_list() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);

    let event = movement_event(1_000, 40);
    assert_eq!(
        backend.write_event("cam", &event).await,
        WriteOutcome::Written
    );

    // Indexed and queryable in the writer's own index.
    let entry = backend.find_event("cam", url_key(1_000, 1000)).unwrap();
    assert_eq!(entry.event_type, EventType::Movement);
    assert_eq!(entry.file_size, 40);
    assert_eq!(entry.filmstrip_frames, 2);

    // Video and thumbnails come back through the trait (streamed).
    let vs = backend.read_video("cam", &entry, None).await.unwrap();
    assert!(matches!(vs.range, ServedRange::Full));
    assert_eq!(vs.total_size, 40);
    assert_eq!(drain(vs).await.len(), 40);
    assert_eq!(
        backend.read_thumbnail("cam", &entry).await.unwrap(),
        vec![0x01, 0x02]
    );
    assert_eq!(
        backend.read_filmstrip("cam", &entry, 1).await.unwrap(),
        vec![0x03, 0x04]
    );

    // A fresh backend rebuilding from the same host recovers the event,
    // its type, size (detailed list), and filmstrip count.
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(1_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Movement);
    assert_eq!(e.file_size, 40);
    assert_eq!(e.filmstrip_frames, 2);
    assert_eq!(scanned.free_space().unwrap(), u64::MAX); // unlimited budget
}

/// A `PUT` of a key that exists is an update, so writing a stem twice
/// rewrites one event. The index used to gain a second entry for it and the
/// budget was charged twice — an in-RAM store of two events where the host
/// holds one, drifting the client-side budget away from real usage.
#[tokio::test]
async fn a_rewritten_stem_replaces_its_entry_rather_than_adding_one() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);

    backend.write_event("cam", &movement_event(1_000, 40)).await;
    // Same start and duration — the same stem, and so the same objects.
    backend.write_event("cam", &movement_event(1_000, 25)).await;

    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        1
    );
    assert_eq!(
        backend
            .find_event("cam", url_key(1_000, 1000))
            .unwrap()
            .file_size,
        25
    );
    // The budget counts what the host actually holds — all four objects,
    // not the video alone.
    assert_eq!(backend.used(), stub.stored_bytes());
    // ts + json + 2 thumbs, overwritten in place.
    assert_eq!(stub.files.lock().unwrap().len(), 4);
}

/// The scan counts filmstrip frames contiguously from 0, so a thumbnail the
/// rewrite has no frame for would be served as part of this event and would
/// outlive it — the delete only removes the frames the entry knows about.
#[tokio::test]
async fn a_shorter_rewrite_deletes_the_thumbnails_it_no_longer_has() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);

    let mut event = movement_event(2_000, 30);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;
    assert!(stub.has("cam/2000_1000_thumb_2.jpg"));

    let mut shorter = movement_event(2_000, 30);
    shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
    backend.write_event("cam", &shorter).await;

    assert_eq!(
        backend
            .find_event("cam", url_key(2_000, 1000))
            .unwrap()
            .filmstrip_frames,
        1
    );
    assert!(stub.has("cam/2000_1000_thumb_0.jpg"));
    assert!(!stub.has("cam/2000_1000_thumb_1.jpg"));
    assert!(!stub.has("cam/2000_1000_thumb_2.jpg"));

    // What a restart rebuilds agrees with the index in RAM.
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert_eq!(
        scanned
            .find_event("cam", url_key(2_000, 1000))
            .unwrap()
            .filmstrip_frames,
        1
    );
}

/// Two events sharing a start PTS are two events. Everything that reaches
/// into the index by key — the upgrade's in-place rewrite and the sweep's
/// removal — has to find the one it named, not whichever of the pair a
/// binary search on the start returns.
#[tokio::test]
async fn siblings_sharing_a_start_pts_are_upgraded_and_removed_by_stem() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    backend
        .write_event("cam", &movement_event(OLD_PTS, 40))
        .await;
    backend
        .write_event("cam", &longer_movement_event(OLD_PTS, 40))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        2
    );
    assert_eq!(backend.used(), stub.stored_bytes());

    // Upgrade only the longer one.
    let mut upgrade = upgrade_for(OLD_PTS);
    upgrade.duration_ms = 2000;
    backend.upgrade_event("cam", &upgrade).await;
    assert_eq!(
        sibling(&backend, 2000).unwrap().event_type,
        EventType::Object
    );
    assert_eq!(
        sibling(&backend, 1000).unwrap().event_type,
        EventType::Movement
    );
    let sidecar = stub
        .files
        .lock()
        .unwrap()
        .get(&format!("cam/{OLD_PTS}_1000.json"))
        .cloned()
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&sidecar).unwrap()["event_type"],
        serde_json::json!("movement"),
        "the upgrade rewrote its sibling's sidecar"
    );

    // The movement sibling expires; the object one does not.
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(sibling(&backend, 1000).is_none());
    assert!(sibling(&backend, 2000).is_some());
    assert!(!stub.has(&format!("cam/{OLD_PTS}_1000.ts")));
    assert!(stub.has(&format!("cam/{OLD_PTS}_2000.ts")));
    assert_eq!(
        backend.used(),
        stub.stored_bytes(),
        "the wrong sibling's bytes were refunded"
    );
}

/// The read path, on the same pair: each sibling is served as itself.
///
/// An API request names a stem, and the two events under this start hold
/// different objects on the host — different videos, different lengths. The
/// lookup this replaced binary-searched the start alone, so one of the two
/// URLs was always answered with the other event's recording: the wrong
/// video streamed under the right link, at the wrong duration.
#[tokio::test]
async fn same_start_siblings_are_each_served_by_their_own_key() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(30_000, 40))
        .await;
    backend
        .write_event("cam", &longer_movement_event(30_000, 40))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        2
    );

    for (duration_ms, file_size) in [(1000u32, 40u64), (2000, 80)] {
        let entry = backend
            .find_event("cam", url_key(30_000, duration_ms))
            .unwrap_or_else(|| panic!("{duration_ms}ms sibling is not indexed"));
        assert_eq!(entry.duration_ms, duration_ms);
        assert_eq!(entry.file_size, file_size);
        // And the bytes behind it are that event's own.
        let vs = backend.read_video("cam", &entry, None).await.unwrap();
        assert_eq!(vs.total_size, file_size);
        assert_eq!(drain(vs).await.len(), file_size as usize);
    }

    // A stem nothing is stored under: this start, no such duration.
    assert!(backend.find_event("cam", url_key(30_000, 3000)).is_none());
}

/// The one asymmetry between the backends: the event type in a request's key
/// is ignored here, because the objects it names do not depend on it. An
/// upgrade rewrites the sidecar in place and moves nothing, so honoring the
/// type would 404 a link a client already holds — the event list it came
/// from, or the playlist a player is part way through — while the footage
/// sits right where it was. Local disk cannot do this: the type is a
/// directory there, so it is part of the path.
#[tokio::test]
async fn find_event_resolves_by_stem_across_an_upgrade() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(31_000, 40))
        .await;
    backend.upgrade_event("cam", &upgrade_for(31_000)).await;

    // The key a client took from the listing before the upgrade still
    // resolves, and to the event as it is now.
    for event_type in [
        EventType::Movement,
        EventType::Object,
        EventType::Continuous,
    ] {
        let entry = backend
            .find_event("cam", EventRef::new(31_000, 1000, event_type))
            .unwrap_or_else(|| panic!("{event_type:?} key stopped resolving"));
        assert_eq!(entry.event_type, EventType::Object);
    }
    // The stem is still the whole identity: the duration has to be right.
    assert!(backend
        .find_event("cam", EventRef::new(31_000, 2000, EventType::Object))
        .is_none());
}

/// The same for the two flags an entry carries: a failed delete and a type
/// the scan could not read both belong to one stem, not to a start PTS.
#[tokio::test]
async fn flags_and_type_holds_follow_the_stem_not_the_start_pts() {
    let (url, stub) = spawn_stub("secret").await;
    let writer = backend_for(&url, "secret", 0);
    writer
        .write_event("cam", &movement_event(OLD_PTS, 40))
        .await;
    writer
        .write_event("cam", &longer_movement_event(OLD_PTS, 40))
        .await;

    // Only the longer sibling's sidecar is unreadable on the next start.
    stub.fail_gets("_2000.json");
    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 2000)));
    assert!(!backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    // The typed sibling expires as a movement; the held one is measured
    // against the longest configured retention and stays.
    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert(format!("cam/{OLD_PTS}_2000.ts"));
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(sibling(&backend, 1000).is_none());
    assert!(sibling(&backend, 2000).is_some());

    // Now expire everything: the held sibling is tried, refuses, and is the
    // one flagged.
    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
    let held = sibling(&backend, 2000).unwrap();
    assert!(held.delete_failed);
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 2000)));
}

#[tokio::test]
async fn object_event_sidecar_carries_type_and_scans_back() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);

    backend.write_event("cam", &object_event(4_000, 20)).await;

    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(4_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.object_classes, vec!["car".to_string()]);
    assert_eq!(e.detections.len(), 1);
}

#[tokio::test]
async fn upgrade_rewrites_sidecar_without_reuploading_video() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend.write_event("cam", &movement_event(5_000, 25)).await;

    // Capture the video bytes and count writes before the upgrade.
    let ts_key = "cam/5000_1000.ts";
    let before = stub.files.lock().unwrap().get(ts_key).cloned().unwrap();

    // The upgrade carries the original event's chain flag into the sidecar
    // it rewrites, so the index has to take it too — LocalDisk rebuilds the
    // whole entry here and cannot drift, this one mutates in place.
    let mut upgrade = upgrade_for(5_000);
    upgrade.continues = true;
    backend.upgrade_event("cam", &upgrade).await;

    // The index flipped to Object...
    let e = backend.find_event("cam", url_key(5_000, 1000)).unwrap();
    assert!(e.continues);
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.object_classes, vec!["person".to_string()]);
    // ...the video object is byte-for-byte unchanged (no re-upload)...
    assert_eq!(
        stub.files.lock().unwrap().get(ts_key).cloned().unwrap(),
        before
    );
    // ...and the sidecar now declares the object type.
    let sidecar = stub
        .files
        .lock()
        .unwrap()
        .get("cam/5000_1000.json")
        .cloned()
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&sidecar).unwrap();
    assert_eq!(json["event_type"], serde_json::json!("object"));
    assert_eq!(json["detections"][0]["class"], serde_json::json!("person"));
    assert_eq!(json["continues"], serde_json::json!(true));
}

#[tokio::test]
async fn delete_via_prune_removes_all_objects() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    // A movement event far enough in the past to expire.
    let old_pts = 1_000_000_000; // 1s after epoch
    backend
        .write_event("cam", &movement_event(old_pts, 30))
        .await;
    assert_eq!(stub.files.lock().unwrap().len(), 4); // ts + json + 2 thumbs

    // Prune with a tiny movement retention → the old event goes.
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    assert!(backend.find_event("cam", url_key(old_pts, 1000)).is_none());
    assert!(stub.files.lock().unwrap().is_empty());
    assert_eq!(backend.used(), 0);
}

/// The per-sweep deletion cap is the remote store's protection against a
/// forward clock jump too: the whole index expiring at once must not empty
/// the bucket in one sweep.
#[tokio::test]
async fn prune_caps_how_much_one_sweep_deletes() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    for i in 0..40u64 {
        backend
            .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
            .await;
    }

    // Every event is expired; a quarter of the 40 indexed may go.
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        30
    );

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        22
    );
}

/// An event the store refuses to delete sits at the head of the sweep, so
/// without the cap exempting known failures it would spend the whole budget
/// on the same objects every hour and never reach the ones behind them.
#[tokio::test]
async fn an_undeletable_event_does_not_block_the_sweep_forever() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    for i in 0..12u64 {
        backend
            .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
            .await;
    }
    // The four oldest videos — the whole cap for a 12-event index.
    {
        let mut refused = stub.fail_delete_paths.lock().unwrap();
        for i in 0..4u64 {
            refused.insert(format!("cam/{}_1000.ts", 1_000_000_000 + i * 1_000_000));
        }
    }

    // First sweep: the entire budget goes on the four that refuse.
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        12
    );

    // Second: retrying those is free, so it reaches four behind them.
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        8,
        "a stuck head of the queue blocked the whole sweep"
    );
}

/// Shutdown reaches this backend as a raised flag, and one event here is
/// several sequential HTTP deletes: a cancelled sweep must issue none.
#[tokio::test]
async fn a_cancelled_prune_deletes_nothing() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    let old_pts = 1_000_000_000;
    backend
        .write_event("cam", &movement_event(old_pts, 30))
        .await;
    let before = stub.files.lock().unwrap().len();

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(true))
        .await;

    assert!(backend.find_event("cam", url_key(old_pts, 1000)).is_some());
    assert_eq!(stub.files.lock().unwrap().len(), before);
}

// ---- the un-scanned state ---------------------------------------------

/// The listing is one request, made at the moment the network is least
/// likely to be up. When every attempt at it fails the index holds only
/// what this process has written since, which is indistinguishable from a
/// store that is nearly empty — and pruning on that reads a full archive as
/// nothing to do, forever, until someone restarts camon.
#[tokio::test]
async fn a_scan_that_never_listed_the_store_refuses_to_prune() {
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
    let backend = backend_for(&url, "secret", 0);

    assert!(backend.scan().await.is_err(), "an unlisted store scanned");
    assert_eq!(
        stub.lists(),
        SCAN_ATTEMPTS as usize,
        "the startup scan did not make its attempts"
    );

    // Recording is unaffected by any of this — that is the point of not
    // failing startup — so events pile up in an index of this session only.
    backend
        .write_event("cam", &movement_event(1_000_000_000, 30))
        .await;
    let stored = stub.files.lock().unwrap().len();

    // Long expired, and pruned anyway if the empty index is believed.
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    assert!(
        backend
            .find_event("cam", url_key(1_000_000_000, 1000))
            .is_some(),
        "pruned against an index that has never seen the store"
    );
    assert_eq!(stub.files.lock().unwrap().len(), stored, "deleted objects");
    assert_eq!(
        stub.lists(),
        2 * SCAN_ATTEMPTS as usize,
        "the tick did not retry the scan it is the only retry for"
    );
}

/// The budget's two answers both need an index that has seen the store:
/// "under budget" is measured against a sum of what is indexed, and the
/// eviction that follows deletes what is indexed. Un-scanned, that is this
/// session's writes — the newest footage there is — while everything older
/// sits on the host uncounted and unevicted.
#[tokio::test]
async fn a_scan_that_never_listed_the_store_refuses_to_enforce_the_budget() {
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
    // 60 bytes of budget against 120 written: enough to evict twice.
    let backend = backend_for(&url, "secret", 60);
    assert!(backend.scan().await.is_err());

    for pts in [1_000u64, 2_000, 3_000] {
        backend.write_event("cam", &movement_event(pts, 40)).await;
    }
    let stored = stub.files.lock().unwrap().len();
    let listed = stub.lists();

    backend.guard_free_space("cam", 0).await;

    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        3,
        "evicted the only events it could see"
    );
    assert_eq!(stub.files.lock().unwrap().len(), stored);
    assert_eq!(
        stub.lists(),
        listed,
        "the write path waited on a scan; that stalls the camera it guards"
    );
}

/// The failure this exists for is a race with the network coming up, so the
/// attempt that succeeds is usually the second or the third.
#[tokio::test]
async fn a_listing_that_comes_back_before_the_attempts_run_out_scans_normally() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, 3, 1000, "object");
    stub.fail_next_lists(SCAN_ATTEMPTS as usize - 1);

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert_eq!(stub.lists(), SCAN_ATTEMPTS as usize);
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        3
    );
    // Retention runs on it like any other scanned index.
    backend
        .prune(u64::MAX, 1, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .is_empty());
}

/// The startup attempts are bounded, so a store that comes back a minute
/// after boot would otherwise leave retention off until the next restart.
/// The retention tick retries the scan while — and only while — the index
/// has never been built.
#[tokio::test]
async fn the_retention_tick_heals_an_index_the_startup_scan_never_built() {
    let (url, stub) = spawn_stub("secret").await;
    // Footage from before this process started: only a listing reveals it,
    // and it is long expired.
    seed_events(&stub, 1_000_000_000, 2, 1000, "movement");
    stub.fail_next_lists(usize::MAX);
    let backend = backend_for(&url, "secret", 0);
    assert!(backend.scan().await.is_err());
    assert!(backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .is_empty());

    stub.serve_lists_again();

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    assert!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty()
            && stub.files.lock().unwrap().is_empty(),
        "the tick did not rebuild the index and prune what it found"
    );

    // And having healed, it stops asking: re-listing a whole bucket every
    // hour is the request the scan exists to make once.
    let listed = stub.lists();
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(stub.lists(), listed, "kept re-scanning a scanned index");
}

/// The state means "never rebuilt", not "the last request failed". A store
/// that goes away after a good scan must not throw the index back to
/// refusing: what it learned is still the best account of the archive there
/// is, and retention is exactly what an unreachable store needs to keep.
#[tokio::test]
async fn a_scan_that_succeeded_stays_scanned_when_a_later_one_fails() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    backend
        .write_event("cam", &movement_event(1_000_000_000, 30))
        .await;

    stub.fail_next_lists(usize::MAX);
    assert!(backend.scan().await.is_err());

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(
        backend
            .find_event("cam", url_key(1_000_000_000, 1000))
            .is_none(),
        "a failed listing un-scanned an index that had been built"
    );
    assert!(stub.files.lock().unwrap().is_empty());
}

/// A healing scan runs while cameras record, and an upload in progress has
/// its sidecar on the host before its video — for as long as the video
/// takes. The sweep cannot tell that from an orphan, so it does not run
/// here at all; the next startup, where nothing of this process's can be in
/// flight, collects what accumulated. See [`ScanKind`].
#[tokio::test]
async fn a_healing_rescan_leaves_orphaned_metadata_for_the_next_startup() {
    let (url, stub) = spawn_stub("secret").await;
    // An expired event, which is how this test knows the heal happened at
    // all, and a sidecar whose video is not there.
    seed_events(&stub, 1_000_000_000, 1, 1000, "movement");
    stub.files.lock().unwrap().insert(
        "cam/5000_1000.json".to_string(),
        br#"{"event_type":"object"}"#.to_vec(),
    );
    stub.fail_next_lists(usize::MAX);
    let backend = backend_for(&url, "secret", 0);
    assert!(backend.scan().await.is_err());

    stub.serve_lists_again();
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    assert!(
        !stub.has("cam/1000000000_1000.ts")
            && backend
                .query("cam", EventPage::unbounded(0, u64::MAX))
                .is_empty(),
        "the heal did not rebuild the index and prune what it found"
    );
    assert!(
        stub.has("cam/5000_1000.json"),
        "a heal swept metadata a camera may have been uploading"
    );
    // The next startup is the one that may, and does.
    let restarted = backend_for(&url, "secret", 0);
    restarted.scan().await.unwrap();
    assert!(!stub.has("cam/5000_1000.json"), "orphan sidecar kept");
}

/// Five attempts is a bound on a host that refuses connections in a
/// millisecond. A host that accepts and then says nothing costs a whole
/// request timeout each time, and startup awaits this series before any
/// camera is spawned: five of those is five minutes of a camera system
/// recording nothing, which is worse than starting un-scanned.
#[tokio::test]
async fn a_listing_that_never_answers_gives_up_on_the_clock_not_the_attempt_count() {
    let (url, stub) = spawn_stub("secret").await;
    // Far longer than the budget, and longer than `REQUEST_TIMEOUT` would
    // allow too: without a deadline of its own the series waits for this.
    stub.hang_lists(Duration::from_secs(30));
    let backend = backend_for(&url, "secret", 0);

    let started = std::time::Instant::now();
    assert!(backend.scan().await.is_err());
    let elapsed = started.elapsed();

    assert!(
        elapsed < SCAN_LISTING_BUDGET * 4,
        "the startup scan held the cameras for {elapsed:?}"
    );
    assert!(
        stub.lists() < SCAN_ATTEMPTS as usize,
        "spent the attempt count on a host that answers nothing"
    );
}

/// The retention task is joined by the shutdown drain on a bound of one
/// event's deletes. A heal on that task must respect the same flag: its
/// waits between attempts are the drain's waits.
#[tokio::test]
async fn a_shutdown_stops_the_scan_from_retrying() {
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
    // Long enough to raise the flag while the first attempt is in flight.
    stub.hang_lists(Duration::from_millis(30));
    let backend = Arc::new(backend_for(&url, "secret", 0));

    let cancel = Arc::new(AtomicBool::new(false));
    let healing = tokio::spawn({
        let (backend, cancel) = (Arc::clone(&backend), Arc::clone(&cancel));
        async move { backend.prune(1, u64::MAX, u64::MAX, &cancel).await }
    });

    wait_until(|| stub.lists() >= 1).await;
    cancel.store(true, Ordering::SeqCst);
    healing.await.unwrap();

    assert_eq!(
        stub.lists(),
        1,
        "kept retrying the scan after shutdown was asked for"
    );
}

/// The wait between attempts is the drain's wait too, and the flag it must
/// notice is raised while it is already sleeping — so the wait polls rather
/// than sleeping through the whole delay it was given.
#[tokio::test]
async fn a_wait_between_scan_attempts_ends_when_shutdown_arrives_during_it() {
    let flag = Arc::new(AtomicBool::new(false));
    let raiser = {
        let flag = Arc::clone(&flag);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            flag.store(true, Ordering::SeqCst);
        })
    };

    let started = std::time::Instant::now();
    // A backoff far longer than any drain is allowed to take.
    sleep_unless(Duration::from_secs(30), &|| flag.load(Ordering::SeqCst)).await;
    let elapsed = started.elapsed();
    raiser.await.unwrap();

    assert!(
        elapsed < crate::storage::contract::SHUTDOWN_POLL * 4,
        "the wait sat through {elapsed:?} of a shutdown"
    );
    assert!(elapsed >= Duration::from_millis(20), "did not wait at all");
}

/// And the pass itself stops, rather than walking an archive's worth of
/// sidecars inside a drain measured in one event's deletes. What it leaves
/// is not a rebuilt index: it never reached the end of the listing, so it
/// knows nothing about what it did not read.
#[tokio::test]
async fn a_shutdown_part_way_through_a_scan_leaves_it_unscanned() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000_000_000, 4, 1000, "movement");
    stub.fail_next_lists(usize::MAX);
    let backend = Arc::new(backend_for(&url, "secret", 0));
    assert!(backend.scan().await.is_err());

    stub.serve_lists_again();
    // Every sidecar read is slow enough to raise the flag inside one.
    stub.get_delay_ms.store(100, Ordering::Relaxed);
    let cancel = Arc::new(AtomicBool::new(false));
    let healing = tokio::spawn({
        let (backend, cancel) = (Arc::clone(&backend), Arc::clone(&cancel));
        async move { backend.prune(1, u64::MAX, u64::MAX, &cancel).await }
    });

    wait_until(|| !stub.gets.lock().unwrap().is_empty()).await;
    cancel.store(true, Ordering::SeqCst);
    healing.await.unwrap();

    // Nothing was pruned on what it did manage to read...
    assert!(stub.has("cam/1000000000_1000.ts"));
    // ...and the next tick still finds an index that must be rebuilt,
    // which it would not if an interrupted pass counted as a rebuild.
    stub.get_delay_ms.store(0, Ordering::Relaxed);
    let listed = stub.lists();
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(
        stub.lists() > listed,
        "an interrupted pass was taken for a rebuilt index"
    );
    assert!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty(),
        "the completed heal did not prune what it found"
    );
}

/// The one scan that is not the only writer of the index. A heal reads a
/// sidecar over the network and inserts what it says some round trips
/// later, and in between the live write path can settle the same event's
/// type, size or filmstrip — all of which the listing predates. Writing the
/// listing's version over that would expire object footage on movement
/// retention twelve days early, with no unknown-type marker to repair it
/// by, so the heal yields to whatever the index already holds.
#[tokio::test]
async fn a_heal_yields_to_an_upgrade_that_landed_while_it_was_reading() {
    const LIVE_PTS: u64 = 5_000_000_000;
    let (url, stub) = spawn_stub("secret").await;
    // Footage from before this process: only a listing reveals it.
    seed_events(&stub, 1_000_000_000, 2, 1000, "movement");
    stub.fail_next_lists(usize::MAX);
    let backend = Arc::new(backend_for(&url, "secret", 0));
    assert!(backend.scan().await.is_err());

    // A live event of this session: uploaded and indexed as a movement,
    // which is what its sidecar on the host says too.
    backend
        .write_event("cam", &movement_event(LIVE_PTS, 40))
        .await;

    stub.serve_lists_again();
    // A sidecar read that takes long enough for a detection to come back
    // while it is in flight.
    stub.get_delay_ms.store(100, Ordering::Relaxed);
    let healing = tokio::spawn({
        let backend = Arc::clone(&backend);
        // Retention long enough that this sweep only heals.
        async move {
            backend
                .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
                .await
        }
    });

    let sidecar = format!("cam/{LIVE_PTS}_1000.json");
    wait_until(|| stub.get_count(&sidecar) >= 1).await;
    // The heal is holding a movement sidecar it has already read.
    backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;
    healing.await.unwrap();

    let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
    assert_eq!(
        entry.event_type,
        EventType::Object,
        "the heal wrote a stale movement over an upgraded event"
    );
    assert_eq!(entry.object_classes, vec!["person".to_string()]);
    // And it still did what it was for: the archive it could not see.
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        3
    );
}

/// The other half of that yield. An upgrade whose sidecar `PUT` reported
/// failure may have committed anyway — the client cannot tell — and
/// `upgrade_event` gives up before its index update when it does, leaving
/// the store saying object and RAM saying movement with nothing in the
/// process that ever looks again. The event then expires twelve days early,
/// and budget eviction, which takes movements before objects, reaches it
/// first. A heal reads that sidecar, so it can put the type right; it is
/// the only pass that ever will before a restart.
#[tokio::test]
async fn a_heal_types_an_upgrade_whose_sidecar_landed_despite_reporting_failure() {
    const LIVE_PTS: u64 = 5_000_000_000;
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
    let backend = backend_for(&url, "secret", 0);
    assert!(backend.scan().await.is_err());

    backend
        .write_event("cam", &movement_event(LIVE_PTS, 40))
        .await;
    // The upgraded sidecar lands at the origin and the client is told it
    // did not, which is a timeout or a proxy error over a committed body.
    stub.fail_puts(".json", true);
    backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;
    stub.clear_faults();

    let stale = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
    assert_eq!(
        stale.event_type,
        EventType::Movement,
        "not the state to fix"
    );

    stub.serve_lists_again();
    backend
        .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
    assert_eq!(
        entry.event_type,
        EventType::Object,
        "the heal read an object sidecar and left the index on movement"
    );
    assert_eq!(entry.object_classes, vec!["person".to_string()]);
    assert_eq!(entry.detections.len(), 1);
    assert_eq!(entry.backend.as_deref(), Some("ollama"));
    // What the sidecar says nothing newer about is untouched: the video and
    // its thumbnails are the ones the index already had.
    assert_eq!(entry.file_size, 40);
    assert_eq!(entry.filmstrip_frames, 2);
}

/// The join only ever moves an entry forward. An event the index already
/// holds as an object is the *later* account of it — the join exists for an
/// index that is behind the store, not for one that is ahead of it — so a
/// sidecar naming different detections leaves them alone. That is what
/// makes the join safe against a live upgrade landing while the heal reads:
/// whichever order they fall in, the detections that survive are the ones
/// the write path put there.
#[tokio::test]
async fn a_heal_leaves_the_detections_of_an_entry_it_already_has_as_an_object() {
    const LIVE_PTS: u64 = 6_000_000_000;
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
    let backend = backend_for(&url, "secret", 0);
    assert!(backend.scan().await.is_err());

    backend
        .write_event("cam", &movement_event(LIVE_PTS, 40))
        .await;
    backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;

    // An object sidecar on the store that says something else — a second
    // writer, or an upgrade this one has already superseded.
    stub.files.lock().unwrap().insert(
        format!("cam/{LIVE_PTS}_1000.json"),
        br#"{"event_type":"object","detections":[{"class":"car","confidence":0.5}]}"#.to_vec(),
    );

    stub.serve_lists_again();
    backend
        .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
    assert_eq!(entry.event_type, EventType::Object);
    assert_eq!(
        entry.object_classes,
        vec!["person".to_string()],
        "the heal wrote the store's detections over the write path's"
    );
}

#[tokio::test]
async fn budget_prune_evicts_cheapest_and_oldest_first() {
    let (url, _stub) = spawn_stub("secret").await;
    let continuous = continuous_event(1_000, 40);
    let movement = movement_event(2_000, 40);
    let mut obj = movement_event(3_000, 40);
    obj.has_objects = true;
    obj.object_classes = vec!["person".to_string()];
    let budget = cost_of(&obj) + cost_of(&movement) / 2;

    // The budget above fits the object event and not the movement beside
    // it, priced off what the three events actually cost the store —
    // sidecars and filmstrips included, which is the whole of what the
    // budget is about.
    let backend = over_budget_backend(&url, &[continuous, obj, movement], budget).await;
    assert!(backend.used() > budget, "the store is not over its budget");

    // Enforce the budget (as the pre-write guard would).
    backend.guard_free_space("cam", 0).await;

    // Cheapest tier first: the continuous chunk goes, and the store is
    // still over, so the movement follows. The object is the tier kept
    // longest and survives both.
    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none()); // continuous evicted
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none()); // movement evicted
    assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some()); // object survives
    assert!(backend.used() <= budget);
}

/// Budget eviction runs ahead of every write, so an object the store
/// refuses must not be re-attempted by every pass: it would spend each one
/// on the same doomed delete and never reach the events that would free
/// space. Local disk's emergency prune skips its own failures for exactly
/// this reason.
#[tokio::test]
async fn budget_eviction_skips_an_event_it_already_failed_to_delete() {
    let (url, stub) = spawn_stub("secret").await;
    let budget = cost_of(&movement_event(0, 40)) * 3 / 2;
    let backend = over_budget_backend(
        &url,
        &[
            movement_event(1_000, 40),
            movement_event(2_000, 40),
            movement_event(3_000, 40),
        ],
        budget,
    )
    .await;
    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert("cam/1000_1000.ts".to_string());

    // First pass: the oldest refuses, and the pass stops there.
    backend.guard_free_space("cam", 0).await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        3
    );
    assert!(
        backend
            .find_event("cam", url_key(1_000, 1000))
            .unwrap()
            .delete_failed
    );

    // Second: it is skipped, and the budget is enforced around it.
    backend.guard_free_space("cam", 0).await;
    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_some());
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(3_000, 1000)).is_none());
    assert!(stub.has("cam/1000_1000.ts"));
}

/// An object that was already gone reclaimed nothing on the host — it only
/// corrected an index entry describing nothing. Its entry still has to go,
/// and the pass still has to go on to something that does free bytes.
#[tokio::test]
async fn budget_eviction_unindexes_an_object_that_is_already_gone() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let backend = over_budget_backend(
        &url,
        &[
            movement_event(1_000, 40),
            movement_event(2_000, 40),
            movement_event(3_000, 40),
        ],
        cost * 3 / 2,
    )
    .await;
    // Someone else removed the oldest video behind camon's back.
    stub.files.lock().unwrap().remove("cam/1000_1000.ts");

    backend.guard_free_space("cam", 0).await;

    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some());
    assert_eq!(backend.used(), cost);
}

/// An outage flags one candidate per pass (the pass stops at the first
/// failure). If flagging *excluded* an event from eviction, the store
/// coming back would leave the budget permanently over its limit: nothing
/// already written would ever be reconsidered, and the hourly sweep only
/// retries events that are age-expired. Flagging demotes instead.
#[tokio::test]
async fn budget_eviction_recovers_after_an_outage_flagged_every_candidate() {
    let (url, stub) = spawn_stub("secret").await;
    let budget = cost_of(&movement_event(0, 40)) * 3 / 2;
    let backend = over_budget_backend(
        &url,
        &[
            movement_event(1_000, 40),
            movement_event(2_000, 40),
            movement_event(3_000, 40),
            movement_event(4_000, 40),
        ],
        budget,
    )
    .await;
    {
        let mut refused = stub.fail_delete_paths.lock().unwrap();
        for pts in [1_000u64, 2_000, 3_000, 4_000] {
            refused.insert(format!("cam/{pts}_1000.ts"));
        }
    }

    // The store is unreachable: every pass flags one more candidate.
    for _ in 0..4 {
        backend.guard_free_space("cam", 0).await;
    }
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        4
    );
    assert!(backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .iter()
        .all(|e| e.delete_failed));

    // It comes back. Eviction has to reconsider what it flagged, or the
    // budget stays four events over its limit for the life of the process.
    stub.fail_delete_paths.lock().unwrap().clear();
    backend.guard_free_space("cam", 0).await;
    assert!(backend.used() <= budget);
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        1
    );
}

/// The video is deleted before the sidecar, so a refused video delete keeps
/// the event's type. The old order lost it, and a type-less survivor is not
/// simply "expired sooner": `continuous_retention_days` defaults to 1 day
/// against movement's 2, and all three retentions are freely configurable,
/// so reading a continuous chunk back as a movement can keep it a day
/// longer than its own class allows.
#[tokio::test]
async fn a_video_that_refuses_to_delete_keeps_its_sidecar_and_its_type() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    let mut event = continuous_event(OLD_PTS, 30);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01]]));
    backend.write_event("cam", &event).await;
    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert(format!("cam/{OLD_PTS}_1000.ts"));

    // Expire it as a continuous chunk (1 day) while movements keep 2.
    backend
        .prune(u64::MAX, u64::MAX, 1, &AtomicBool::new(false))
        .await;

    let entry = backend.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
    assert!(entry.delete_failed);
    assert!(stub.has(&format!("cam/{OLD_PTS}_1000.ts")));
    assert!(stub.has(&format!("cam/{OLD_PTS}_1000.json")), "type lost");
    // Thumbnails are decoration and carry no type; they go first.
    assert!(!stub.has(&format!("cam/{OLD_PTS}_1000_thumb_0.jpg")));

    // A restart still knows what it is, so the retry measures it against
    // the retention it actually belongs to.
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert_eq!(
        scanned
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .event_type,
        EventType::Continuous
    );
    // ...and once the store lets go, the whole event goes.
    stub.fail_delete_paths.lock().unwrap().clear();
    scanned
        .prune(u64::MAX, u64::MAX, 1, &AtomicBool::new(false))
        .await;
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
    assert!(stub.files.lock().unwrap().is_empty());
}

/// A thumbnail the store refuses to delete stays part of the event rather
/// than leaking: frames are trimmed top-down, so what survives is still
/// contiguous from 0 and the entry can say so — which is what the next scan
/// counts, and what the event's own delete removes.
#[tokio::test]
async fn a_thumbnail_that_refuses_to_delete_stays_part_of_the_event() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    let mut event = movement_event(24_000, 30);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;
    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert("cam/24000_1000_thumb_1.jpg".to_string());

    let mut shorter = movement_event(24_000, 30);
    shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
    backend.write_event("cam", &shorter).await;

    // Frame 2 went; frame 1 refused, so the event still has 0 and 1.
    assert!(!stub.has("cam/24000_1000_thumb_2.jpg"));
    assert!(stub.has("cam/24000_1000_thumb_1.jpg"));
    assert_eq!(
        backend
            .find_event("cam", url_key(24_000, 1000))
            .unwrap()
            .filmstrip_frames,
        2,
        "index disagrees with the host about what exists"
    );

    // The scan counts the same thing, and the event's delete takes it all.
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert_eq!(
        scanned
            .find_event("cam", url_key(24_000, 1000))
            .unwrap()
            .filmstrip_frames,
        2
    );
    stub.fail_delete_paths.lock().unwrap().clear();
    scanned
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(stub.files.lock().unwrap().is_empty(), "leaked a thumbnail");
}

#[tokio::test]
async fn scan_tolerates_ts_without_sidecar() {
    let (url, stub) = spawn_stub("secret").await;
    // A lone .ts, as a plain movement event whose sidecar upload failed
    // leaves (an interruption cannot: the sidecar precedes the video).
    stub.files
        .lock()
        .unwrap()
        .insert("cam/9000_1000.ts".to_string(), vec![0u8; 10]);

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    let e = backend.find_event("cam", url_key(9_000, 1000)).unwrap();
    // A confirmed-absent sidecar is what a plain movement event is written
    // with, so the default is a fact about the write path, not a fallback.
    assert_eq!(e.event_type, EventType::Movement);
    assert_eq!(e.filmstrip_frames, 0);
    assert!(e.object_classes.is_empty());
}

/// Seed `count` stored events, one `.ts` plus one sidecar each, at 1s
/// intervals from `first_pts`. `event_type` is written into every sidecar.
fn seed_events(stub: &Stub, first_pts: u64, count: u64, duration_ms: u32, event_type: &str) {
    seed_events_for(stub, "cam", first_pts, count, duration_ms, event_type);
}

fn seed_events_for(
    stub: &Stub,
    camera_id: &str,
    first_pts: u64,
    count: u64,
    duration_ms: u32,
    event_type: &str,
) {
    let mut files = stub.files.lock().unwrap();
    for i in 0..count {
        let stem = format!("{}_{duration_ms}", first_pts + i * SEC);
        files.insert(format!("{camera_id}/{stem}.ts"), vec![0u8; 10]);
        files.insert(
            format!("{camera_id}/{stem}.json"),
            format!(r#"{{"event_type":"{event_type}"}}"#).into_bytes(),
        );
    }
}

/// The scan is awaited before the first camera is spawned, so its cost is
/// startup latency with nothing recording. One awaited round trip per
/// stored event made that a function of the archive's size; the reads now
/// overlap [`SCAN_CONCURRENCY`]-wide.
#[tokio::test]
async fn the_scan_reads_sidecars_concurrently() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, 64, 1000, "object");
    // Enough per-request latency that a serial scan cannot hide in the
    // noise: 64 × 50ms = 3.2s of it.
    stub.get_delay_ms.store(50, Ordering::Relaxed);

    let backend = backend_for(&url, "secret", 0);
    let started = std::time::Instant::now();
    backend.scan().await.unwrap();
    let elapsed = started.elapsed();

    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        64
    );
    // Serial is 3.2s of injected latency alone and measures at ~3.4s;
    // sixteen at a time is four waves, ~0.24s. The bound sits between them
    // with room for a loaded machine on either count.
    assert!(
        elapsed < Duration::from_millis(2_000),
        "sidecar reads were serial: {elapsed:?}"
    );
    let peak = stub.peak_gets.load(Ordering::SeqCst);
    assert!(peak > 1, "no reads overlapped");
    assert!(peak <= SCAN_CONCURRENCY, "fan-out ran unbounded: {peak}");
}

/// The startup scan is the one pass that walks a whole archive, and it used
/// to build the index one insertion at a time. A store lists in no useful
/// order, so nearly every one of those landed in the middle of the list and
/// shifted the rest along — quadratic memory traffic in the number of
/// stored events, paid at startup, on the box with the least memory
/// bandwidth to pay it with. Building each camera's list and sorting it
/// once is O(n log n) and shifts nothing.
///
/// Counted rather than timed: a clock-based bound on a quadratic term
/// either fails on a loaded machine or passes on a fast one.
#[tokio::test]
async fn the_startup_scan_builds_the_index_without_shifting_it_into_shape() {
    const EVENTS: u64 = 1_000;
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, EVENTS, 1000, "object");

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    // The index is right: every event, in order, charged once.
    let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
    assert_eq!(entries.len() as u64, EVENTS);
    assert!(entries
        .windows(2)
        .all(|w| w[0].start_pts_ns <= w[1].start_pts_ns));
    assert_eq!(
        backend.used(),
        EVENTS * (10 + r#"{"event_type":"object"}"#.len() as u64)
    );

    // And it was not shifted into that shape. One insertion per event
    // averages half a list of shifts each — a quarter of a million here,
    // and 25 million on a box holding ten thousand events.
    let shifted = backend.events.shifted_entries();
    assert!(
        shifted <= 4 * EVENTS,
        "the scan shifted {shifted} entries to index {EVENTS} events"
    );
}

/// A startup pass that stops part way through: what it had already read
/// came from the store and is true, so it stays indexed — collecting the
/// entries before handing them over must not turn an interrupted pass into
/// a lost one. What does not happen is the index being marked as describing
/// the store.
///
/// Driven through `scan_once` because the production startup scan passes a
/// stop that never fires (there is no drain that early); this is the
/// invariant that keeps the collecting honest if that ever changes.
#[tokio::test]
async fn a_startup_pass_that_stops_part_way_keeps_what_it_had_read() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000_000_000, 8, 1000, "movement");
    let backend = backend_for(&url, "secret", 0);

    // Stops after the fourth event has been walked.
    let walked = std::cell::Cell::new(0usize);
    let stop = || {
        walked.set(walked.get() + 1);
        walked.get() > 4
    };
    let pass = backend
        .scan_once(ScanKind::Startup, &stop, REQUEST_TIMEOUT)
        .await
        .unwrap();

    assert!(matches!(pass, ScanPass::Interrupted));
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        4,
        "what the pass had read before it stopped was thrown away"
    );
    assert!(
        backend.scanned_events().is_none(),
        "a half-walked archive was taken for a rebuilt index"
    );
}

/// A filmstrip frame the store lost leaves a hole, and counting up from
/// zero until the first miss reads that hole as the end of the filmstrip —
/// which on a lost `thumb_0` means an event indexed with no frames at all,
/// its remaining thumbnails invisible to the UI and unreachable by the
/// delete that walks `0..filmstrip_frames`. They would be leaked bytes the
/// budget never learns to reclaim.
#[tokio::test]
async fn a_scanned_filmstrip_with_a_gap_counts_the_frames_the_store_still_has() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        files.insert("cam/1000_1000.ts".to_string(), vec![0u8; 10]);
        // thumb_0 never landed; 1 and 3 did.
        files.insert("cam/1000_1000_thumb_1.jpg".to_string(), vec![0u8; 7]);
        files.insert("cam/1000_1000_thumb_3.jpg".to_string(), vec![0u8; 9]);
    }

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    let entry = backend.find_event("cam", url_key(1000, 1000)).unwrap();

    // Named to the highest frame that exists...
    assert_eq!(entry.filmstrip_frames, 4);
    // ...but charged only for the bytes the store really holds.
    assert_eq!(entry.thumbnail_bytes, 16);

    // And the delete reaches every one of them.
    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(!stub.has("cam/1000_1000_thumb_1.jpg"));
    assert!(!stub.has("cam/1000_1000_thumb_3.jpg"));
    assert!(stub.files.lock().unwrap().is_empty());
}

/// `reqwest` starts the TLS stack and reads the proxy environment when a
/// client is built, so this can fail on a box whose CA bundle or
/// `HTTPS_PROXY` is broken — a permanent, operator-fixable fault that used
/// to take the whole process down from inside a constructor.
#[test]
fn a_backend_whose_http_client_will_not_build_is_an_error_not_a_panic() {
    let config = StathostConfig {
        url: "http://127.0.0.1:1".to_string(),
        bucket: "cams".to_string(),
        token: "secret".to_string(),
        max_stored_bytes: 0,
        enabled: true,
    };
    assert!(
        StathostBackend::new(&config, &["cam".to_string()], StopFlag::never()).is_ok(),
        "the ordinary case stopped building"
    );

    force_client_build_failure(true);
    let refused = StathostBackend::new(&config, &["cam".to_string()], StopFlag::never());
    force_client_build_failure(false);

    let error = refused.err().expect("a client that will not build built");
    assert!(
        error.to_string().contains("HTTP client"),
        "an error an operator cannot act on: {error}"
    );
}

/// Overlapping the reads must not reach the index: entries stay sorted, and
/// each stored event is indexed exactly once and charged once.
#[tokio::test]
async fn a_concurrent_scan_indexes_every_event_once_and_in_order() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        for i in 0..40u64 {
            let stem = format!("{}_1000", 1_000 + i * SEC);
            files.insert(format!("cam/{stem}.ts"), vec![0u8; 10]);
            // Alternating types, so a result delivered against the wrong
            // event would show up as a type on the wrong entry.
            let event_type = if i % 2 == 0 { "object" } else { "continuous" };
            files.insert(
                format!("cam/{stem}.json"),
                format!(r#"{{"event_type":"{event_type}"}}"#).into_bytes(),
            );
        }
    }
    stub.get_delay_ms.store(5, Ordering::Relaxed);

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
    assert_eq!(entries.len(), 40, "an event was dropped or duplicated");
    for (i, entry) in entries.iter().enumerate() {
        assert_eq!(entry.start_pts_ns, 1_000 + i as u64 * SEC);
        assert_eq!(
            entry.event_type,
            if i % 2 == 0 {
                EventType::Object
            } else {
                EventType::Continuous
            }
        );
    }
    // The budget is the sum of the index, so a double insertion shows
    // here — and it is the sum of everything each event costs the host,
    // sidecars included, so a rebuild that priced videos alone would too.
    assert_eq!(backend.used(), stub.stored_bytes());
}

/// The distinction the whole sidecar path rests on survives the fan-out: a
/// read that failed is "unknown", never the confirmed absence that means
/// "movement". Half of these sidecars answer `500` while the other half
/// answer normally, in the same pass.
#[tokio::test]
async fn a_concurrent_scan_never_reads_a_failure_as_an_absence() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, 20, 1000, "object");
    seed_events(&stub, 1_000, 20, 2000, "object");
    // Only the 2000ms-duration events' sidecars are unreadable.
    stub.fail_gets("_2000.json");

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
    assert_eq!(entries.len(), 40);
    for entry in &entries {
        let key = event_key(entry);
        if entry.duration_ms == 1000 {
            assert_eq!(entry.event_type, EventType::Object);
            assert!(!backend.has_unknown_type("cam", key), "held a read event");
        } else {
            // Not indexed as a movement event: the type is on hold, so
            // pruning measures it against the longest retention.
            assert!(
                backend.has_unknown_type("cam", key),
                "a failed read was taken for a confirmed absence"
            );
        }
    }
}

#[tokio::test]
async fn write_retries_then_drops_on_persistent_failure() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_writes.store(true, Ordering::Relaxed);

    let outcome = backend.write_event("cam", &movement_event(6_000, 30)).await;
    assert_eq!(outcome, WriteOutcome::Failed);
    // The event was not indexed and nothing landed on the host.
    assert!(backend.find_event("cam", url_key(6_000, 1000)).is_none());
    assert!(stub.files.lock().unwrap().is_empty());
}

/// The sidecar is the only record of an event's type here, so a write that
/// could not store one must not report success: the in-RAM index would keep
/// serving the object event until a restart, after which the scan would
/// call the leftover `.ts` a movement and expire it 12 days early.
#[tokio::test]
async fn a_failed_sidecar_fails_the_write_before_the_video() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".json", false);

    let outcome = backend.write_event("cam", &object_event(11_000, 30)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert!(backend.find_event("cam", url_key(11_000, 1000)).is_none());
    assert_eq!(backend.used(), 0);
    // The video was never attempted, so there is no bare .ts for a later
    // scan to call a movement event.
    assert!(!stub.has("cam/11000_1000.ts"));
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert!(scanned.find_event("cam", url_key(11_000, 1000)).is_none());
}

/// A plain movement event is the one event a sidecar-less `.ts` already
/// scans back as unchanged, so its sidecar is not worth the footage.
#[tokio::test]
async fn a_failed_sidecar_does_not_cost_a_movement_event() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".json", false);

    let outcome = backend
        .write_event("cam", &movement_event(15_000, 30))
        .await;

    assert_eq!(outcome, WriteOutcome::Written);
    assert!(stub.has("cam/15000_1000.ts"));
    assert!(!stub.has("cam/15000_1000.json"));
}

/// The exemption above is only sound while a sidecar-less scan rebuilds a
/// movement event *exactly*. This fails the day a field is added to
/// `sidecar_json` or a scan default changes — which is the whole reason a
/// blanket "always require the sidecar" rule was tempting.
#[tokio::test]
async fn a_movement_event_scans_back_identically_without_its_sidecar() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    // Written without its sidecar rather than stripped of one afterwards,
    // so the two entries are priced against the same store.
    stub.fail_puts(".json", false);
    backend
        .write_event("cam", &movement_event(16_000, 30))
        .await;
    let written = backend.find_event("cam", url_key(16_000, 1000)).unwrap();

    stub.clear_faults();
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();

    assert_eq!(
        scanned.find_event("cam", url_key(16_000, 1000)).unwrap(),
        written
    );
}

/// A PUT that reports failure may still have committed — an upload timeout
/// or a proxy 5xx says nothing about the origin. The video is therefore
/// never rolled back: deleting the sidecar of a phantom `.ts` would leave
/// exactly the bare video this write order exists to prevent.
#[tokio::test]
async fn a_video_that_lands_despite_a_failed_put_keeps_its_type() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".ts", true); // committed server-side, 500 to the client

    let outcome = backend.write_event("cam", &object_event(12_000, 30)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert!(backend.find_event("cam", url_key(12_000, 1000)).is_none());
    // Both objects are still there, so the next scan adopts the phantom
    // video as the object event it is — not as a movement event on a
    // two-day retention.
    stub.clear_faults();
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(12_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.object_classes, vec!["car".to_string()]);
}

/// The mirror case: the video genuinely did not land. The orphan sidecar
/// left behind indexes nothing — the scan walks `.ts` objects only — and so
/// nothing else would ever delete it either: it is never indexed, never
/// counted against the budget, and never a sibling of an event.
///
/// It used to wait for the next *startup* to be collected, which on a flaky
/// uplink is one orphan per failed write for however many weeks the box
/// stays up. The write that created it collects it instead, at the one
/// moment there is no ambiguity about whose upload this is.
#[tokio::test]
async fn an_orphan_sidecar_is_collected_by_the_write_that_orphaned_it() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".ts", false);

    let outcome = backend.write_event("cam", &object_event(17_000, 30)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert!(
        !stub.has("cam/17000_1000.json"),
        "the sidecar of a video that never landed was left for a reboot"
    );
    // The probe is what authorises the delete, and it asks about the video.
    assert_eq!(stub.get_count("cam/17000_1000.ts"), 1);
    stub.clear_faults();
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert!(scanned.find_event("cam", url_key(17_000, 1000)).is_none());
}

/// And it asks rather than assumes. A `PUT` that reported failure may have
/// committed anyway, and deleting *that* video's sidecar would leave the
/// bare `.ts` the sidecar-first order exists to prevent — read back as a
/// plain movement on the wrong retention.
#[tokio::test]
async fn a_sidecar_is_kept_when_the_video_landed_despite_reporting_failure() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".ts", true); // committed server-side, 500 to the client

    assert_eq!(
        backend.write_event("cam", &object_event(18_500, 30)).await,
        WriteOutcome::Failed
    );

    assert!(
        stub.has("cam/18500_1000.json"),
        "collected the sidecar of a video that is on the host"
    );
    assert!(stub.has("cam/18500_1000.ts"));
}

/// A probe that cannot find out is not an absence. Nothing is deleted on a
/// maybe, and the startup sweep — which asks the same question later, when
/// nothing of this process's is in flight — remains the backstop.
#[tokio::test]
async fn nothing_is_collected_when_the_probe_itself_fails() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".ts", false);
    stub.fail_gets(".ts");

    assert_eq!(
        backend.write_event("cam", &object_event(18_700, 30)).await,
        WriteOutcome::Failed
    );

    assert!(stub.has("cam/18700_1000.json"));
}

/// Thumbnails orphan the same way: an upload that got as far as the
/// filmstrip before the video failed leaves them behind too.
#[tokio::test]
async fn orphaned_thumbnails_are_collected_and_live_ones_are_not() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(19_000, 30))
        .await;
    // Orphans of an event whose video is not on the host.
    stub.files
        .lock()
        .unwrap()
        .insert("cam/20000_1000_thumb_0.jpg".to_string(), vec![0x01]);
    stub.files
        .lock()
        .unwrap()
        .insert("cam/20000_1000.json".to_string(), b"{}".to_vec());

    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();

    assert!(!stub.has("cam/20000_1000_thumb_0.jpg"));
    assert!(!stub.has("cam/20000_1000.json"));
    // The live event keeps every one of its objects.
    assert_eq!(
        scanned
            .find_event("cam", url_key(19_000, 1000))
            .unwrap()
            .filmstrip_frames,
        2
    );
    assert!(stub.has("cam/19000_1000.json"));
    assert!(stub.has("cam/19000_1000_thumb_1.jpg"));
}

/// One failed upload orphans a sidecar and every filmstrip frame at once,
/// and all of them turn on the same question. The sweep asks the host once
/// per stem, not once per object it is about to delete.
#[tokio::test]
async fn the_sweep_probes_an_orphaned_stem_once() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        files.insert("cam/24000_1000.json".to_string(), b"{}".to_vec());
        for i in 0..6 {
            files.insert(format!("cam/24000_1000_thumb_{i}.jpg"), vec![0x01]);
        }
    }

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert_eq!(
        stub.get_count("cam/24000_1000.ts"),
        1,
        "the same video was probed once per orphaned object"
    );
    assert!(
        stub.files.lock().unwrap().is_empty(),
        "an orphaned object survived the sweep"
    );
}

/// The listing is a snapshot, and it is not necessarily *this* process's
/// snapshot: another camon on the same camera id, or a `PUT` that outlived
/// the process which issued it, can commit a video after the bucket was
/// listed. The sweep re-checks every candidate against the host immediately
/// before deleting it, so a sidecar whose video landed in that window keeps
/// the event's only record of its type.
#[tokio::test]
async fn the_sweep_keeps_a_sidecar_whose_video_landed_after_the_listing() {
    let (url, stub) = spawn_stub("secret").await;
    stub.files.lock().unwrap().insert(
        "cam/22000_1000.json".to_string(),
        br#"{"event_type":"object"}"#.to_vec(),
    );
    // The video commits the moment the scan has its listing — the shape of
    // an upload in flight under a camera id this process also owns.
    stub.commit_after_list
        .lock()
        .unwrap()
        .push("cam/22000_1000.ts".to_string());

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert!(stub.has("cam/22000_1000.json"), "live sidecar collected");
    assert!(stub.has("cam/22000_1000.ts"));
    // Not indexed — it was not in the listing — but the next start reads it
    // back as the object event its surviving sidecar says it is.
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert_eq!(
        scanned
            .find_event("cam", url_key(22_000, 1000))
            .unwrap()
            .event_type,
        EventType::Object
    );
}

/// A failure to find out whether the video is there is not an absence.
#[tokio::test]
async fn the_sweep_keeps_metadata_it_could_not_check() {
    let (url, stub) = spawn_stub("secret").await;
    stub.files
        .lock()
        .unwrap()
        .insert("cam/23000_1000.json".to_string(), b"{}".to_vec());
    // The re-check itself fails: a 500 on the video probe, not a 404.
    stub.fail_gets(".ts");

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert!(stub.has("cam/23000_1000.json"));
}

/// The sweep deletes, so it may only touch what this process is the
/// authority for: the cameras it owns, under names this backend writes. A
/// second camon sharing the bucket has uploads in flight that look exactly
/// like orphans — sidecar first, video second.
#[tokio::test]
async fn the_scan_only_collects_orphans_of_cameras_it_owns() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        // Another camon's in-flight write: sidecar up, video still going.
        files.insert("other/21000_1000.json".to_string(), b"{}".to_vec());
        // Ours, but not something this backend writes.
        files.insert("cam/notes.txt".to_string(), b"hi".to_vec());
        files.insert("cam/settings.json".to_string(), b"{}".to_vec());
        files.insert("cam/logo.jpg".to_string(), vec![0x01]);
    }

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert!(stub.has("other/21000_1000.json"));
    assert!(stub.has("cam/notes.txt"));
    assert!(stub.has("cam/settings.json"));
    assert!(stub.has("cam/logo.jpg"));
}

/// Thumbnails are decoration — the UI hides frames that fail to load — so
/// losing them must not cost the footage.
#[tokio::test]
async fn a_failed_thumbnail_is_not_fatal() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".jpg", false);

    let outcome = backend.write_event("cam", &object_event(13_000, 30)).await;

    assert_eq!(outcome, WriteOutcome::Written);
    let entry = backend.find_event("cam", url_key(13_000, 1000)).unwrap();
    assert_eq!(entry.event_type, EventType::Object);
    assert_eq!(entry.filmstrip_frames, 0);

    // ...and the index a restart rebuilds agrees with the one in RAM.
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(13_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.filmstrip_frames, 0);
    assert!(backend.read_thumbnail("cam", &entry).await.is_err());
}

/// 1s after the epoch: older than any retention a test configures.
const OLD_PTS: u64 = 1_000_000_000;

/// One flaky GET during startup must not decide a retention class. An
/// unreadable sidecar leaves the type *unknown*, and an unknown type is
/// not a movement type — the old scan collapsed both onto Movement, which
/// deletes an object event twelve days early from the read side alone.
#[tokio::test]
async fn an_unreadable_sidecar_is_not_pruned_as_a_movement_event() {
    let (url, stub) = spawn_stub("secret").await;
    backend_for(&url, "secret", 0)
        .write_event("cam", &object_event(OLD_PTS, 30))
        .await;

    // A restart that cannot read the sidecar: 500, not 404.
    stub.fail_gets(".json");
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();

    // Indexed and visible (losing the footage from the UI would be its own
    // bug), but not deleted by the sweep its placeholder type invites.
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    scanned
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    assert!(stub.has("cam/1000000000_1000.ts"));

    // The hold is a longer retention, not an immortal one: once every
    // configured age has passed, an event nobody can type still goes.
    scanned.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
    assert!(!stub.has("cam/1000000000_1000.ts"));
}

/// A scan that succeeded is not run again, so the sweep is the only place a
/// held event can ever be typed without a restart.
#[tokio::test]
async fn a_prune_tick_resolves_a_held_event() {
    let (url, stub) = spawn_stub("secret").await;
    backend_for(&url, "secret", 0)
        .write_event("cam", &object_event(OLD_PTS, 30))
        .await;
    stub.fail_gets(".json");
    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    // The store recovers; the next sweep reads the sidecar it could not.
    stub.clear_faults();
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    let entry = backend.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
    assert_eq!(entry.event_type, EventType::Object);
    assert_eq!(entry.object_classes, vec!["car".to_string()]);
    assert!(!backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    // Typed again, it prunes on its own retention: kept as an object...
    backend.prune(1, u64::MAX, 1, &AtomicBool::new(false)).await;
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    // ...and gone once the object retention itself expires.
    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
}

/// Valid JSON that names no type is not a movement event either — and
/// unlike a failed read it will never resolve itself, so it must be held
/// rather than quietly given a two-day retention.
#[tokio::test]
async fn a_sidecar_naming_no_type_is_held_not_assumed() {
    let (url, stub) = spawn_stub("secret").await;
    stub.files
        .lock()
        .unwrap()
        .insert("cam/1000000000_1000.ts".to_string(), vec![0u8; 10]);
    stub.files
        .lock()
        .unwrap()
        .insert("cam/1000000000_1000.json".to_string(), b"{}".to_vec());

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    // A re-read says the same thing, so the hold survives the sweep.
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));
}

/// Bytes that are not JSON are a failed read, not an absent sidecar: the
/// distinction is the difference between holding the event and pruning it
/// as a movement.
#[tokio::test]
async fn an_unparsable_sidecar_is_held_not_treated_as_absent() {
    let (url, stub) = spawn_stub("secret").await;
    stub.files
        .lock()
        .unwrap()
        .insert("cam/1000000000_1000.ts".to_string(), vec![0u8; 10]);
    stub.files
        .lock()
        .unwrap()
        .insert("cam/1000000000_1000.json".to_string(), b"not json".to_vec());

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
}

/// Eviction order is the other decision that reads the event type. A held
/// event's placeholder says movement; evicting on that would spend the
/// footage the hold exists to protect.
#[tokio::test]
async fn budget_eviction_tiers_an_unknown_type_with_the_objects() {
    let (url, stub) = spawn_stub("secret").await;
    let writer = backend_for(&url, "secret", 0);
    writer.write_event("cam", &object_event(1_000, 40)).await;
    writer.write_event("cam", &movement_event(2_000, 40)).await;

    // The object event's sidecar is unreadable on the next start...
    stub.fail_gets("1000_1000.json");
    let backend = backend_for(&url, "secret", cost_of(&object_event(1_000, 40)));
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (1_000, 1000)));

    // ...so the budget must still evict the genuine movement event first,
    // even though the held one is older and labelled movement too.
    backend.guard_free_space("cam", 0).await;
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_some());
}

/// A thumbnail gap stops the uploads: the scan counts frames contiguously
/// from 0, so continuing past a failure would index frames it will never
/// see again and leave the extra objects stranded on the host.
#[tokio::test]
async fn a_thumbnail_gap_stops_the_filmstrip() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    let mut event = movement_event(18_000, 30);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    stub.fail_puts("_thumb_1.jpg", false);

    backend.write_event("cam", &event).await;

    assert_eq!(
        backend
            .find_event("cam", url_key(18_000, 1000))
            .unwrap()
            .filmstrip_frames,
        1
    );
    assert!(stub.has("cam/18000_1000_thumb_0.jpg"));
    assert!(!stub.has("cam/18000_1000_thumb_2.jpg"));
}

/// The movement exemption rests on the sidecar of a plain movement event
/// saying nothing the scan does not already assume. Pinning the literal
/// bytes catches a field added to `sidecar_json` — which the entry-equality
/// test above cannot, since a new field would be default in both halves.
#[test]
fn a_plain_movement_sidecar_carries_nothing_but_its_type() {
    assert_eq!(
        sidecar_json(Some(EventType::Movement), None, None, &[], false),
        r#"{"detections":[],"event_type":"movement"}"#
    );
}

// ---- the shared storage contract --------------------------------------
//
// One assertion body, two backends: see `storage::contract::contract_tests`
// for why these are written there and called here.

#[tokio::test]
async fn contract_a_written_event_reads_back_whole() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    crate::storage::contract::contract_tests::a_written_event_reads_back_whole(&backend).await;
}

#[tokio::test]
async fn contract_an_event_costs_nothing_once_it_is_deleted() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    crate::storage::contract::contract_tests::an_event_costs_nothing_once_it_is_deleted(
        &backend,
        || stub.stored_bytes(),
    )
    .await;
}

#[tokio::test]
async fn contract_a_prune_that_starts_stopped_deletes_nothing() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    crate::storage::contract::contract_tests::a_prune_that_starts_stopped_deletes_nothing(&backend)
        .await;
}

#[tokio::test]
async fn contract_a_rewritten_event_replaces_its_entry() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    crate::storage::contract::contract_tests::a_rewritten_event_replaces_its_entry(&backend).await;
}

#[tokio::test]
async fn contract_an_upgrade_reclassifies_the_one_indexed_event() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    crate::storage::contract::contract_tests::an_upgrade_reclassifies_the_one_indexed_event(
        &backend,
    )
    .await;
}

#[tokio::test]
async fn contract_an_upgrade_of_a_deleted_event_indexes_nothing() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    crate::storage::contract::contract_tests::an_upgrade_of_a_deleted_event_indexes_nothing(
        &backend,
    )
    .await;
}

// ---- cancellation: a write inside the shutdown drain -------------------

/// One event write is up to eight sequential uploads — a sidecar and a
/// video, each with a retry, then four filmstrip frames — and each may sit
/// on its whole [`UPLOAD_TIMEOUT`]. Nothing used to interrupt that, so a
/// single event could hold a camera's writer for forty minutes: past the
/// drain's phase-3 budget, which is sized for *one* upload timeout and
/// nothing more (`crate::shutdown`).
///
/// The rule that makes the arithmetic true again is that no further request
/// is issued once the flag is up. Here the flag rises while the first
/// upload is in flight, and that upload is the last one there is.
///
/// What is asserted is the request *count*, not the clock. Post-stop time is
/// (requests issued) x (the per-request timeout), so "one request" is a
/// stricter statement about the drain's budget than any millisecond figure
/// could be — and it is the same statement on a loaded box, where a
/// wall-clock bound only measures the box.
#[tokio::test]
async fn a_write_issues_no_further_upload_once_shutdown_is_asked_for() {
    let (url, stub) = spawn_stub("secret").await;
    let flag = Arc::new(AtomicBool::new(false));
    let backend = backend_stopped_by(&url, "secret", 0, StopFlag::shared(Arc::clone(&flag)));
    // The first upload is held open until the flag is up, so the flag
    // really does rise *inside* it however busy the machine is.
    stub.hold(&stub.hold_puts);

    let event = movement_event(1_000, 40);
    let (outcome, ()) = tokio::join!(backend.write_event("cam", &event), async {
        wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
        flag.store(true, Ordering::SeqCst);
        stub.release(&stub.hold_puts);
    });

    assert_eq!(outcome, WriteOutcome::Failed);
    assert_eq!(
        stub.puts.lock().unwrap().len(),
        1,
        "requests were issued after shutdown was asked for: {:?}",
        stub.puts.lock().unwrap()
    );
}

/// And a write that starts stopped issues nothing at all — the drain is
/// waiting on this task, and an upload begun now is one it has to sit out.
#[tokio::test]
async fn a_write_that_starts_stopped_uploads_nothing() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_stopped_by(
        &url,
        "secret",
        0,
        StopFlag::shared(Arc::new(AtomicBool::new(true))),
    );

    let outcome = backend.write_event("cam", &movement_event(1_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert!(stub.puts.lock().unwrap().is_empty());
    assert!(backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .is_empty());
}

// ---- accounting: the whole cost, and room secured in advance -----------

/// The budget used to count `.ts` bytes and nothing else, so a store with a
/// sidecar and four filmstrip frames per event was permanently over a cap
/// it believed it was under — and every extra byte was one no eviction
/// would ever reclaim, because eviction is measured against the same
/// figure.
#[tokio::test]
async fn an_events_whole_cost_is_charged_and_not_just_its_video() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    let event = movement_event(1_000, 40);

    backend.write_event("cam", &event).await;

    let entry = backend.find_event("cam", url_key(1_000, 1000)).unwrap();
    assert_eq!(entry.file_size, 40, "the video's own size still means that");
    assert!(entry.sidecar_bytes > 0 && entry.thumbnail_bytes > 0);
    assert_eq!(backend.used(), cost_of(&event));
    assert_eq!(
        backend.used(),
        stub.stored_bytes(),
        "the client-side budget and the host disagree about what is stored"
    );
}

/// A rebuild has to price events the same way, off the listing — otherwise
/// a restart resets the budget to the sum of the videos and the store goes
/// over again until the next write.
#[tokio::test]
async fn a_rebuild_prices_events_the_way_the_write_path_did() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    for pts in [1_000u64, 2_000] {
        backend.write_event("cam", &movement_event(pts, 40)).await;
    }

    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();

    assert_eq!(scanned.used(), backend.used());
    assert_eq!(scanned.used(), stub.stored_bytes());
}

/// Two cameras write at once, which is the normal case on any installation
/// with more than one. The guard used to read the byte total on each
/// writer's own task, before either event was in it, so both saw room and
/// both wrote: the store landed a whole event over its cap for every write
/// in flight. A reservation puts both of them into the figure the eviction
/// is measured against.
#[tokio::test]
async fn two_writes_in_flight_cannot_both_walk_through_the_budget() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    // Room for three events, two of them already stored.
    let backend = over_budget_backend(
        &url,
        &[movement_event(1_000, 40), movement_event(2_000, 40)],
        cost * 3,
    )
    .await;
    assert_eq!(backend.used(), cost * 2);
    // Enough of a window that the second write is polled while the first is
    // still uploading. No gate is needed: both writes claim their room
    // before either awaits anything, so the overlap this is about is
    // settled by the time the first byte is sent.
    stub.put_delay_ms.store(20, Ordering::SeqCst);

    let (third, fourth) = (movement_event(3_000, 40), movement_event(4_000, 40));
    tokio::join!(
        backend.write_event("cam", &third),
        backend.write_event("cam", &fourth),
    );

    assert!(
        backend.used() <= cost * 3,
        "the store is {} bytes over a budget of {}",
        backend.used() - cost * 3,
        cost * 3
    );
    assert_eq!(backend.used(), stub.stored_bytes());
}

// ---- retry classification ---------------------------------------------

/// A store that refuses the request itself — a bad token, a path it will
/// not take — answers the second attempt exactly as it answered the first.
/// Sending it is an [`UPLOAD_TIMEOUT`]'s worth of a camera's writer, and on
/// a video it is the event's megabytes up the link again, to learn nothing.
#[tokio::test]
async fn a_refused_upload_is_not_sent_a_second_time() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.refuse_puts(".json", false, StatusCode::FORBIDDEN);

    let outcome = backend.write_event("cam", &object_event(1_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert_eq!(stub.put_count("cam/1000_1000.json"), 1);
}

/// A store having a moment gets its second attempt — that is what the
/// allowance is for, and the wait between the two is [`OBJECT_RETRY`]'s
/// (pinned in `storage::contract`).
#[tokio::test]
async fn a_store_having_a_moment_gets_its_second_attempt() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.refuse_puts(".json", false, StatusCode::SERVICE_UNAVAILABLE);

    let outcome = backend.write_event("cam", &object_event(1_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert_eq!(stub.put_count("cam/1000_1000.json"), 2);
}

// ---- an upgrade overtaken by the sweep --------------------------------

/// Retention runs on its own task, so it can overtake an upgrade between
/// the check that the event is indexed and the sidecar `PUT` that lands.
/// A `PUT` always succeeds, so what used to happen is that the sidecar of a
/// deleted event was written back onto the store — an orphan nothing but a
/// reboot collects — and the upgrade logged success for footage that was
/// gone. Local disk cannot reach this: its upgrade commits by renaming the
/// video, which simply fails once the video is not there.
#[tokio::test]
async fn an_upgrade_overtaken_by_a_sweep_leaves_no_sidecar_behind() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    backend
        .write_event("cam", &movement_event(OLD_PTS, 40))
        .await;
    // The upgrade's sidecar upload is held open while the sweep runs inside
    // it — the interleaving, rather than a store that is simply missing the
    // event when the upgrade starts.
    stub.take_puts();
    stub.hold(&stub.hold_puts);

    let upgrade = upgrade_for(OLD_PTS);
    tokio::join!(backend.upgrade_event("cam", &upgrade), async {
        wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
        backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
        stub.release(&stub.hold_puts);
    });

    assert!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty(),
        "the upgrade re-indexed an event the sweep had deleted"
    );
    assert!(
        !stub.has(&format!("cam/{OLD_PTS}_1000.json")),
        "the upgrade left the sidecar of a deleted event on the store"
    );
    // And it worked this out without asking the store. That is the whole
    // value of the index check over the probe that backs it up: in the
    // ordering a sweep usually leaves behind, the answer is already in RAM,
    // and a request here is one more thing for the drain to wait on.
    assert_eq!(
        stub.get_count(&format!("cam/{OLD_PTS}_1000.ts")),
        0,
        "the upgrade probed the store for something the index already knew"
    );
}

/// The scenario the whole cancellation guarantee turns on. A capped store
/// sits *at* its cap in the steady state — that is what reserve-then-evict
/// makes it do — and the drain keeps handing the warm writer events after
/// the flag goes up. So the first post-flag write finds itself over budget,
/// and without a gate it would evict real stored footage, with real
/// `DELETE`s, to make room for an event that the very next check is about
/// to abandon unsent: footage destroyed for a recording that never
/// happened, and up to five request timeouts of the drain's budget spent
/// destroying it.
#[tokio::test]
async fn a_stopped_write_over_a_full_store_deletes_nothing() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let seeder = backend_for(&url, "secret", 0);
    for pts in [1_000u64, 2_000] {
        seeder.write_event("cam", &movement_event(pts, 40)).await;
    }
    let flag = Arc::new(AtomicBool::new(false));
    // Exactly at the cap, which is where a capped store lives.
    let backend = backend_stopped_by(
        &url,
        "secret",
        cost * 2,
        StopFlag::shared(Arc::clone(&flag)),
    );
    backend.scan().await.unwrap();
    assert_eq!(backend.used(), cost * 2);
    stub.take_deletes();
    let stored = stub.stored_bytes();

    flag.store(true, Ordering::SeqCst);
    // What the drain does: the writer keeps draining its queue, guard and
    // all, after the flag is up.
    backend.guard_free_space("cam", 0).await;
    let outcome = backend.write_event("cam", &movement_event(3_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert_eq!(
        stub.take_deletes(),
        Vec::<String>::new(),
        "a stopped write evicted stored footage to make room for itself"
    );
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        2
    );
    assert_eq!(stub.stored_bytes(), stored);
}

/// The sweep's own deletes are gated request by request, not merely event by
/// event: one event is up to six of them and each can sit on a request
/// timeout, so a flag checked only between events leaves six minutes of
/// post-stop work inside one.
#[tokio::test]
async fn a_sweep_stops_between_the_deletes_of_a_single_event() {
    let (url, stub) = spawn_stub("secret").await;
    let flag = Arc::new(AtomicBool::new(false));
    let backend = backend_stopped_by(&url, "secret", 0, StopFlag::shared(Arc::clone(&flag)));
    backend.scan().await.unwrap();
    backend
        .write_event("cam", &movement_event(OLD_PTS, 40))
        .await;
    stub.take_deletes();
    // The first delete is held open until the flag is up.
    stub.hold(&stub.hold_deletes);

    let cancel = AtomicBool::new(false);
    tokio::join!(backend.prune(1, 1, 1, &cancel), async {
        wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
        flag.store(true, Ordering::SeqCst);
        stub.release(&stub.hold_deletes);
    });

    assert_eq!(
        stub.take_deletes().len(),
        1,
        "the sweep kept deleting an event's objects after shutdown was asked for"
    );
    // Nothing is flagged as having resisted deletion: the store was fine.
    assert!(
        !backend
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .delete_failed
    );
}

/// The eviction pass is the one the drain waits on with a camera's
/// recording queued behind it, and the flag can go up while it is already
/// running — after the guard has decided to evict and between two of the
/// events it is deleting. Neither of the checks that refuse a pass *before*
/// it starts can reach that; the skeleton's own `cancel` is what does.
#[tokio::test]
async fn an_eviction_already_under_way_stops_when_shutdown_arrives() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let seeder = backend_for(&url, "secret", 0);
    for pts in [1_000u64, 2_000, 3_000, 4_000] {
        seeder.write_event("cam", &movement_event(pts, 40)).await;
    }
    let flag = Arc::new(AtomicBool::new(false));
    // Room for one: the pass has three events to get through.
    let backend = backend_stopped_by(&url, "secret", cost, StopFlag::shared(Arc::clone(&flag)));
    backend.scan().await.unwrap();
    stub.take_deletes();
    // The first delete is held open until the flag is up.
    stub.hold(&stub.hold_deletes);

    tokio::join!(backend.guard_free_space("cam", 0), async {
        wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
        flag.store(true, Ordering::SeqCst);
        stub.release(&stub.hold_deletes);
    });

    assert_eq!(
        stub.take_deletes().len(),
        1,
        "the eviction kept deleting stored footage after shutdown was asked for"
    );
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        4,
        "an event was unindexed by a pass that never finished deleting it"
    );
}

/// The sweep's `cancel` is the flag the trait promises is honoured between
/// the requests one event's deletion is made of — and it has to be *that*
/// flag, not the backend's own. Production hands both the same
/// `AtomicBool`, so a backend polling only its constructor flag would look
/// correct here for ever and be wrong for any caller with a stop of its
/// own. This raises `cancel` alone, with the shutdown flag left down, and
/// the deletion stops all the same.
#[tokio::test]
async fn a_sweep_stops_between_deletes_on_the_cancel_it_was_given() {
    let (url, stub) = spawn_stub("secret").await;
    // No shutdown flag at all: the only stop in this test is `cancel`.
    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    backend
        .write_event("cam", &movement_event(OLD_PTS, 40))
        .await;
    stub.take_deletes();
    stub.hold(&stub.hold_deletes);

    let cancel = AtomicBool::new(false);
    tokio::join!(backend.prune(1, 1, 1, &cancel), async {
        wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
        cancel.store(true, Ordering::SeqCst);
        stub.release(&stub.hold_deletes);
    });

    assert_eq!(
        stub.take_deletes().len(),
        1,
        "the sweep kept deleting an event's objects after its own cancel was raised"
    );
    assert!(
        !backend
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .delete_failed,
        "a cancelled deletion was recorded as one the store refused"
    );
}

/// A deletion cut short mid-thumbnail must leave a *prefix*, because a
/// prefix is the only thing the next scan can see.
///
/// The scan counts an event's frames contiguously from 0 and stops at the
/// first gap. So deleting frame 0 first and stopping there strands the rest
/// where nothing can reach them: the scan records zero frames, the orphan
/// sweep passes over them because the video is still on the store, and the
/// event's own later deletion only removes as many frames as the entry
/// records — which is none of them. They would sit there until some restart
/// that happened to come *after* the video had gone, which on a box that
/// stays up for months is never.
///
/// This walks the whole scenario rather than just the ordering: interrupt a
/// sweep after one thumbnail delete, restart onto the store, and then let
/// retention take the event away. Nothing may be left.
#[tokio::test]
async fn a_deletion_cut_short_leaves_thumbnails_a_rebuild_can_still_see() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    let mut event = movement_event(OLD_PTS, 40);
    // Three frames, so "a prefix" is a claim with content: one is deleted
    // and two have to survive together, in order, from index 0.
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;
    let stem = format!("{OLD_PTS}_1000");
    stub.take_deletes();
    stub.hold(&stub.hold_deletes);

    let cancel = AtomicBool::new(false);
    tokio::join!(backend.prune(1, 1, 1, &cancel), async {
        wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
        cancel.store(true, Ordering::SeqCst);
        stub.release(&stub.hold_deletes);
    });

    // Exactly one frame went, and it was the *last* one.
    assert_eq!(
        stub.take_deletes(),
        vec![format!("cam/{stem}_thumb_2.jpg")],
        "the sweep deleted the wrong end of the filmstrip, or did not stop"
    );
    assert!(stub.has(&format!("cam/{stem}_thumb_0.jpg")));
    assert!(stub.has(&format!("cam/{stem}_thumb_1.jpg")));
    assert!(stub.has(&format!("cam/{stem}.ts")));

    // A restart sees the survivors, because they are a prefix.
    let restarted = backend_for(&url, "secret", 0);
    restarted.scan().await.unwrap();
    let entry = restarted.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
    assert_eq!(
        entry.filmstrip_frames, 2,
        "the rebuild cannot see the thumbnails the interrupted sweep left"
    );
    assert_eq!(
        restarted.used(),
        stub.stored_bytes(),
        "the rebuild is not charged for everything the store is holding"
    );

    // And retention finishes the job it was interrupted in the middle of,
    // taking every object with it.
    restarted.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(
        restarted
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty(),
        "the second sweep did not delete the event"
    );
    assert!(
        stub.files.lock().unwrap().is_empty(),
        "objects survived the event they belong to: {:?}",
        stub.files.lock().unwrap().keys().collect::<Vec<_>>()
    );
}

/// The same stranding as the test above, reached without any interruption:
/// one refused `DELETE` in the middle of an otherwise healthy pass.
///
/// A frame the store refuses is still there afterwards, so carrying on down
/// past it deletes *around* it and leaves a gap — and a gap is what the scan
/// cannot see past. Stopping the descent leaves a prefix instead.
///
/// Cancellation is deliberately not what this test uses, because it cannot
/// tell the two implementations apart: whether the loop breaks on the
/// refusal or carries on to the next frame, the very next thing either does
/// is read the stop flag and abandon, so both leave every frame in place. A
/// video that also refuses keeps the event alive with no timing in the test
/// at all, and separates them completely.
#[tokio::test]
async fn a_refused_filmstrip_frame_keeps_the_ones_below_it_contiguous() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    let mut event = movement_event(OLD_PTS, 40);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;
    let stem = format!("{OLD_PTS}_1000");
    stub.take_deletes();
    {
        let mut refused = stub.fail_delete_paths.lock().unwrap();
        // The top frame will not go...
        refused.insert(format!("cam/{stem}_thumb_2.jpg"));
        // ...and neither will the video, which is what keeps the event
        // indexed for a later pass to find — the state the leak needs.
        refused.insert(format!("cam/{stem}.ts"));
    }

    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

    // A refused frame does not stop the event's deletion: the video was
    // still attempted, and its outcome is the event's.
    let attempted = stub.take_deletes();
    assert!(
        attempted.contains(&format!("cam/{stem}.ts")),
        "a refused thumbnail stopped the event's own deletion: {attempted:?}"
    );
    assert!(
        backend
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .delete_failed,
        "the video refused and the event was not flagged for a retry"
    );
    for i in 0..3 {
        assert!(
            stub.has(&format!("cam/{stem}_thumb_{i}.jpg")),
            "frame {i} was deleted from under the frame that refused, leaving a gap"
        );
    }

    // So a restart sees all three, and the retry that follows takes them.
    let restarted = backend_for(&url, "secret", 0);
    restarted.scan().await.unwrap();
    assert_eq!(
        restarted
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .filmstrip_frames,
        3,
        "the rebuild cannot see the frames the refused delete left behind"
    );
    assert_eq!(restarted.used(), stub.stored_bytes());

    stub.fail_delete_paths.lock().unwrap().clear();
    restarted.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(
        stub.files.lock().unwrap().is_empty(),
        "objects survived the event they belong to: {:?}",
        stub.files.lock().unwrap().keys().collect::<Vec<_>>()
    );
}

/// And a refused frame is not the *event's* outcome. The video's is — that
/// rule predates all of this — so an expired recording is not held back
/// because a JPEG resisted, and nothing is flagged as a refusal on a
/// thumbnail's account: [`WarmEventEntry::delete_failed`] is what demotes an
/// event in eviction and what a sweep counts, and neither should turn on
/// decoration. What is left over becomes an orphan, which is what the
/// startup sweep is for.
#[tokio::test]
async fn a_refused_filmstrip_frame_is_not_the_events_outcome() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    let mut event = movement_event(OLD_PTS, 40);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;
    let stem = format!("{OLD_PTS}_1000");
    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert(format!("cam/{stem}_thumb_2.jpg"));

    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

    assert!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty(),
        "an expired recording was held back because a thumbnail resisted"
    );
    assert!(!stub.has(&format!("cam/{stem}.ts")));
    assert!(!stub.has(&format!("cam/{stem}.json")));

    // The frames left behind are orphans now — no video, no entry — and the
    // startup sweep is the thing that collects those.
    stub.fail_delete_paths.lock().unwrap().clear();
    let restarted = backend_for(&url, "secret", 0);
    restarted.scan().await.unwrap();
    assert!(
        stub.files.lock().unwrap().is_empty(),
        "the startup sweep did not collect the frames a refused delete stranded: {:?}",
        stub.files.lock().unwrap().keys().collect::<Vec<_>>()
    );
}

/// The policy when room cannot be made: record anyway, and say so. An event
/// bigger than the whole budget is the sharpest form of it — refusing would
/// mean this camera never stores anything again, on a cap that is a number
/// an operator typed rather than a disk that is actually full.
#[tokio::test]
async fn an_event_too_big_for_the_whole_budget_is_still_recorded() {
    let (url, _stub) = spawn_stub("secret").await;
    let budget = cost_of(&movement_event(0, 40)) / 2;
    let backend = scanned_backend_for(&url, "secret", budget).await;

    let outcome = backend.write_event("cam", &movement_event(1_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Written);
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        1
    );
    assert!(
        backend.used() > budget,
        "the store is under a budget it cannot fit one event into"
    );
    // And the overshoot is visible where the write path steers by it.
    assert_eq!(backend.free_space().unwrap(), 0);
}

/// The same when eviction is the thing that cannot free anything: a store
/// refusing `DELETE`s is a store that will still take footage, and refusing
/// to give it any would turn one outage into two.
#[tokio::test]
async fn a_store_that_refuses_deletes_still_gets_the_footage() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let backend = over_budget_backend(
        &url,
        &[movement_event(1_000, 40), movement_event(2_000, 40)],
        cost * 2,
    )
    .await;
    for pts in [1_000u64, 2_000] {
        stub.fail_delete_paths
            .lock()
            .unwrap()
            .insert(format!("cam/{pts}_1000.ts"));
    }

    let outcome = backend.write_event("cam", &movement_event(3_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Written);
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        3
    );
    assert!(backend.used() > cost * 2);
}

/// An upgraded sidecar is bigger than the one it replaces — it carries the
/// detections the movement event had none of — and it is stored before
/// anything accounts for it. The growth is claimed for the duration, so a
/// write racing the upgrade evicts against a total that includes it.
#[tokio::test]
async fn an_upgrade_claims_the_growth_of_the_sidecar_it_is_writing() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 100_000).await;
    backend.write_event("cam", &movement_event(1_000, 40)).await;
    let before = backend.free_space().unwrap();
    stub.take_puts();
    // Held open so the budget can be read from inside the upload.
    stub.hold(&stub.hold_puts);

    let upgrade = upgrade_for(1_000);
    let (_, during) = tokio::join!(backend.upgrade_event("cam", &upgrade), async {
        wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
        let during = backend.free_space().unwrap();
        stub.release(&stub.hold_puts);
        during
    });

    assert!(
        during < before,
        "the sidecar's growth was uploaded against a budget that had not been told"
    );
    // And the figure does not move again: the claim is handed to the index,
    // never counted on top of it.
    assert_eq!(backend.free_space().unwrap(), during);
}

/// The handover from reservation to index has to be complete before the
/// write's trailing cleanup, which is more network deletes. Holding the
/// claim through those would have every other camera's write count this
/// event twice for as long as they take, and evict a victim it did not need
/// to.
#[tokio::test]
async fn a_rewrites_reservation_is_released_before_its_thumbnails_are_trimmed() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 100_000).await;
    let mut event = movement_event(2_000, 30);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;

    let mut shorter = movement_event(2_000, 30);
    shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
    // Held open so the accounting can be read from inside the trim.
    stub.hold(&stub.hold_deletes);

    let (_, during) = tokio::join!(backend.write_event("cam", &shorter), async {
        wait_until(|| !stub.deletes.lock().unwrap().is_empty()).await;
        let during = backend.free_space().unwrap();
        stub.release(&stub.hold_deletes);
        during
    });

    assert_eq!(
        during,
        backend.free_space().unwrap(),
        "the write's reservation was still counted while its thumbnails were being trimmed"
    );
}

/// The budget bounds the *wait*, not just the gap between results: a store
/// that has stopped answering makes the read at the head of the fan-out two
/// request timeouts long, and a deadline checked only after a result
/// arrives never gets to say anything about it.
#[tokio::test]
async fn a_prune_does_not_wait_out_a_held_read_past_its_budget() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, OLD_PTS, 8, 1000, "object");
    stub.fail_gets(".json");
    let backend = scanned_backend_for(&url, "secret", 0).await;
    // The reads are held open and never answered — the host that accepted
    // the connection and then said nothing, which is the failure that costs
    // a whole request timeout. A pass whose budget bounds only the gap
    // *between* results never reaches the check at all and sits here for as
    // long as the reads take; a pass that bounds the wait returns while they
    // are still held. What is asserted is which of those happened, not how
    // many milliseconds it took.
    stub.hold(&stub.hold_gets);

    let finished = tokio::time::timeout(
        Duration::from_secs(5),
        backend.prune(1, 1, 1, &AtomicBool::new(false)),
    )
    .await;
    stub.release(&stub.hold_gets);

    assert!(
        finished.is_ok(),
        "the tick was still waiting on held sidecar reads, past a budget of {RESOLVE_BUDGET:?}"
    );
}

/// And the next tick starts where the last one stopped. The hold list has
/// whatever order its `HashSet` iterates in, so a pass that always starts at
/// the front re-reads the same unresolvable prefix every hour and never
/// reaches the tail behind it at all — the events furthest from being typed
/// would be exactly the ones nothing ever asks about again.
///
/// Asserted by membership rather than by counting: how *many* holds a tick
/// gets through depends on the machine, but which one it starts from does
/// not. The oldest hold is the head of the sorted list, so a pass that
/// always starts at the front reads it every single tick — and a pass that
/// starts where the last one stopped cannot read it again until the window
/// has been all the way round.
#[tokio::test]
async fn consecutive_prune_ticks_reach_different_held_events() {
    let (url, stub) = spawn_stub("secret").await;
    const HELD: u64 = 80;
    seed_events(&stub, OLD_PTS, HELD, 1000, "object");
    stub.fail_gets(".json");
    let backend = scanned_backend_for(&url, "secret", 0).await;
    stub.take_gets();
    // Slow enough that no single tick can get all the way round the hold
    // list and wrap back to its head.
    stub.get_delay_ms.store(25, Ordering::SeqCst);

    // Two ticks with no expiry, so only the re-reads happen.
    let oldest_hold = format!("cam/{OLD_PTS}_1000.json");
    let read_oldest = |stub: &Stub| stub.take_gets().contains(&oldest_hold);
    backend
        .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    let first = read_oldest(&stub);
    backend
        .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    let second = read_oldest(&stub);

    assert!(first, "the first tick did not start at the oldest hold");
    assert!(
        !second,
        "the second tick started at the oldest hold again; every tick re-reads the same \
         prefix and the holds behind it are never reached"
    );
}

/// Two cameras, and the reason the cursor cannot be shared between them.
///
/// A shared cursor is advanced by every camera's pass, so each camera's
/// window moves by the *sum* of what all of them read. Whenever that sum is
/// a multiple of a camera's hold count, that camera lands on the same window
/// every tick and the rest of its list is never read again — starvation
/// reached from the opposite direction to the one the rotation was added
/// for, and reached under exactly the conditions this pass is normally in.
///
/// Constructed without a clock in it. The sidecar reads are held open and
/// never answered, so every pass reads nothing back and advances its cursor
/// by the floor of one; the fan-out issues exactly [`SCAN_CONCURRENCY`]
/// requests and no more, because no slot is ever freed. With 32 holds per
/// camera the window is a strict half of the list, so which holds it covers
/// is an observable fact rather than a matter of timing:
///
/// * per camera, this camera's second tick starts at its own hold 1;
/// * shared, both cameras have moved it, so the second tick starts at 2 or 3
///   depending on which camera the index happened to walk first — and hold 1
///   is not read in either case.
#[tokio::test]
async fn a_second_cameras_holds_do_not_move_this_ones_window() {
    let (url, stub) = spawn_stub("secret").await;
    const HELD: u64 = 32;
    seed_events_for(&stub, "cam", OLD_PTS, HELD, 1000, "object");
    seed_events_for(&stub, "other", OLD_PTS, HELD, 1000, "object");
    stub.fail_gets(".json");
    let backend = scanned_backend_with_cameras(&url, &["cam", "other"]).await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        HELD as usize
    );
    assert_eq!(
        backend
            .query("other", EventPage::unbounded(0, u64::MAX))
            .len(),
        HELD as usize
    );
    stub.take_gets();
    // Held and never answered: every pass reads nothing back, so every
    // cursor advances by its floor of one and nothing depends on latency.
    stub.hold(&stub.hold_gets);

    let hold_at = |i: u64| format!("cam/{}_1000.json", OLD_PTS + i * SEC);
    let sweep = |stub: &Stub| -> HashSet<String> { stub.take_gets().into_iter().collect() };

    // Two ticks with nothing expired, so only the re-reads happen.
    backend
        .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    let first = sweep(&stub);
    backend
        .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    let second = sweep(&stub);
    stub.release(&stub.hold_gets);

    assert!(
        first.contains(&hold_at(0)),
        "the first tick did not start at this camera's oldest hold"
    );
    // Between them these four pin the second window to exactly `1..=16`,
    // which is the only placement a per-camera cursor that moved by one can
    // produce: hold 1 present rules out a window the other camera pushed
    // past it, hold 0 absent rules out one that never moved at all, hold 16
    // present rules out one narrower than the fan-out, and hold 17 absent
    // rules out one wider.
    assert!(
        second.contains(&hold_at(1)),
        "the second tick skipped this camera's hold 1: the other camera's reads moved \
         this camera's window, which is how a window comes to land on the same place \
         every tick and never read the rest of the list"
    );
    assert!(
        !second.contains(&hold_at(0)),
        "the second tick started where the first did: this camera's window is not moving"
    );
    assert!(
        second.contains(&hold_at(SCAN_CONCURRENCY as u64)),
        "the second tick's window is narrower than the fan-out that issues it, so what \
         it covers per tick is not what the budget and the fan-out say it is"
    );
    assert!(
        !second.contains(&hold_at(SCAN_CONCURRENCY as u64 + 1)),
        "the second tick's window is wider than the fan-out that issued it"
    );
}

/// An upgrade grows the sidecar and does not evict for it, so it can take
/// the store over the cap on its own — and it can be the *last* thing a
/// camera does, with the detection worker draining its queue after the last
/// ordinary write, so there is no later `make_room` to notice. It therefore
/// reports for itself, on the same streak the write path uses.
#[tokio::test]
async fn an_upgrade_that_crosses_the_budget_says_so() {
    let (url, stub) = spawn_stub("secret").await;
    let event = movement_event(1_000, 40);
    // Exactly enough for the event as written, and not a byte for the
    // detections the upgrade is about to add to its sidecar.
    let backend = scanned_backend_for(&url, "secret", cost_of(&event)).await;
    backend.write_event("cam", &event).await;
    assert_eq!(
        backend.free_space().unwrap(),
        0,
        "the store is not exactly full"
    );
    assert_eq!(
        backend.budget_overshoots.lock_recover().count(),
        0,
        "the write itself was already over"
    );

    backend.upgrade_event("cam", &upgrade_for(1_000)).await;

    assert!(
        backend.used() > cost_of(&event),
        "the upgrade's sidecar did not grow the store"
    );
    assert_eq!(
        backend.budget_overshoots.lock_recover().count(),
        1,
        "an upgrade took the store over its cap and recorded nothing"
    );

    // And on the *same* streak the write path uses, so a store that
    // alternates between the two sources cannot stay quiet by splitting its
    // occurrences across two schedules. The store refuses to give the one
    // stored event up, so the write cannot evict its way under either.
    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert("cam/1000_1000.ts".to_string());
    backend.write_event("cam", &movement_event(2_000, 40)).await;
    assert_eq!(
        backend.budget_overshoots.lock_recover().count(),
        2,
        "the write path counts its overshoots somewhere else"
    );
}

/// The hard ordering of the upgrade/sweep race, which the index check alone
/// cannot see: the sweep has deleted the objects and has *not* reached its
/// `index.remove` yet, so the reclassification lands on an entry that is
/// still there and the sidecar is written back onto a store with no video
/// under it. Constructed by deleting the objects directly while the
/// upgrade's `PUT` is in flight — which is exactly the state the sweep is
/// in between its own two steps.
#[tokio::test]
async fn an_upgrade_whose_video_vanished_mid_put_takes_itself_back() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    backend
        .write_event("cam", &movement_event(OLD_PTS, 40))
        .await;
    stub.take_puts();
    stub.hold(&stub.hold_puts);

    let upgrade = upgrade_for(OLD_PTS);
    tokio::join!(backend.upgrade_event("cam", &upgrade), async {
        wait_until(|| !stub.puts.lock().unwrap().is_empty()).await;
        // The sweep's object deletes, with its index removal still to come.
        {
            let mut files = stub.files.lock().unwrap();
            files.remove(&format!("cam/{OLD_PTS}_1000.ts"));
            files.remove(&format!("cam/{OLD_PTS}_1000.json"));
        }
        stub.release(&stub.hold_puts);
    });

    assert!(
        !stub.has(&format!("cam/{OLD_PTS}_1000.json")),
        "the upgrade left a sidecar on a store that has no video for it"
    );
    assert!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .is_empty(),
        "the index still describes footage that is gone"
    );
}

// ---- a prune tick that has held types to re-read -----------------------

/// The re-read of held sidecars runs ahead of every deletion the sweep
/// exists to make, and it used to be serial and unbounded: an archive's
/// worth of round trips, hourly, in front of the pass that reclaims the
/// space. It is fanned out and cut off at [`RESOLVE_BUDGET`] now, and the
/// sweep behind it runs either way.
#[tokio::test]
async fn a_prune_bounds_the_time_it_spends_re_reading_held_types() {
    let (url, stub) = spawn_stub("secret").await;
    const HELD: u64 = 100;
    seed_events(&stub, OLD_PTS, HELD, 1000, "object");
    // Every sidecar is unreadable, so the scan holds every event's type and
    // the prune tick has the whole archive to re-read.
    stub.fail_gets(".json");
    let backend = scanned_backend_for(&url, "secret", 0).await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        HELD as usize
    );
    // Only now: the scan's own reads are not what this is measuring.
    stub.take_gets();
    stub.get_delay_ms.store(25, Ordering::SeqCst);

    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

    let sidecar_reads = stub
        .take_gets()
        .iter()
        .filter(|p| p.ends_with(".json"))
        .count();
    assert!(
        sidecar_reads < HELD as usize,
        "the tick re-read {sidecar_reads} sidecars of {HELD} held events instead of \
         stopping at its budget"
    );
    assert!(
        stub.peak_gets.load(Ordering::SeqCst) > 1,
        "the re-reads were issued one at a time"
    );
    // And the sweep behind them still deleted its share (a quarter of the
    // archive, per `cap_sweep_deletions`).
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        HELD as usize * 3 / 4,
        "the sweep did not get to its deletions"
    );
}

#[tokio::test]
async fn read_thumbnail_errors_when_no_filmstrip() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    let event = continuous_event(7_000, 30); // no filmstrip frames
    backend.write_event("cam", &event).await;
    let entry = backend.find_event("cam", url_key(7_000, 1000)).unwrap();
    assert!(backend.read_thumbnail("cam", &entry).await.is_err());
}

// ---- streamed Range playback ------------------------------------------

#[tokio::test]
async fn read_video_serves_partial_and_suffix_ranges() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    // A 40-byte movement event (body is 40 × 0xab).
    backend.write_event("cam", &movement_event(8_000, 40)).await;
    let entry = backend.find_event("cam", url_key(8_000, 1000)).unwrap();

    // bytes=10-19 → a 206 with a 10-byte body and the right Content-Range.
    let vs = backend
        .read_video(
            "cam",
            &entry,
            Some(RangeRequest::FromTo {
                start: 10,
                end: Some(19),
            }),
        )
        .await
        .unwrap();
    assert_eq!(vs.range, ServedRange::Partial { start: 10, end: 19 });
    assert_eq!(vs.total_size, 40);
    assert_eq!(drain(vs).await, vec![0xab; 10]);

    // bytes=-5 → the last five bytes.
    let vs = backend
        .read_video("cam", &entry, Some(RangeRequest::Suffix(5)))
        .await
        .unwrap();
    assert_eq!(vs.range, ServedRange::Partial { start: 35, end: 39 });
    assert_eq!(drain(vs).await, vec![0xab; 5]);
}

#[tokio::test]
async fn read_video_reports_unsatisfiable_range() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend.write_event("cam", &movement_event(9_000, 40)).await;
    let entry = backend.find_event("cam", url_key(9_000, 1000)).unwrap();

    // start past EOF → the stub answers 416; we surface Unsatisfiable + size.
    let vs = backend
        .read_video(
            "cam",
            &entry,
            Some(RangeRequest::FromTo {
                start: 100,
                end: None,
            }),
        )
        .await
        .unwrap();
    assert_eq!(vs.range, ServedRange::Unsatisfiable);
    assert_eq!(vs.total_size, 40);
    assert!(drain(vs).await.is_empty());
}

/// A `206` says the body is a slice and its `Content-Range` says which one.
/// Without a usable header there is nothing to serve it as: passing the
/// slice off as the whole event hands the player bytes that are not at the
/// offsets it seeks to — silent corruption of exactly the footage someone
/// is scrubbing through — and relaying a range the arithmetic cannot make
/// sense of ends in a subtraction that underflows. Both are refused.
#[tokio::test]
async fn a_206_without_a_usable_content_range_is_refused_rather_than_served_whole() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(11_000, 40))
        .await;
    let entry = backend.find_event("cam", url_key(11_000, 1000)).unwrap();
    let ranged = RangeRequest::FromTo {
        start: 10,
        end: Some(19),
    };

    for header in [
        "",                                // no Content-Range at all
        "pages 10-19/40",                  // not a byte range
        "bytes 10-19",                     // no total
        "bytes ten-19/40",                 // unparsable bound
        "bytes 19-10/40",                  // reversed: the underflow shape
        "bytes 10-40/40",                  // end past the last byte
        "bytes 40-45/40",                  // wholly past the object
        "bytes 0-18446744073709551615/40", // an end nothing can be sliced to
    ] {
        *stub.bad_content_range.lock().unwrap() = Some(header.to_string());
        let refused = backend.read_video("cam", &entry, Some(ranged)).await;
        let error = refused
            .err()
            .unwrap_or_else(|| panic!("{header:?} was served as a video"));
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::InvalidData,
            "{header:?} failed for the wrong reason: {error}"
        );
    }

    // The honest header still plays, so the refusal above is about the
    // header rather than about ranged reads.
    *stub.bad_content_range.lock().unwrap() = None;
    let vs = backend
        .read_video("cam", &entry, Some(ranged))
        .await
        .unwrap();
    assert_eq!(vs.range, ServedRange::Partial { start: 10, end: 19 });
    assert_eq!(drain(vs).await, vec![0xab; 10]);
}

/// The bounds check itself, on the shapes a `Content-Range` can carry.
#[test]
fn a_partial_range_has_to_be_a_range_of_the_object() {
    assert_eq!(
        ServedRange::partial(10, 19, 40),
        Some(ServedRange::Partial { start: 10, end: 19 })
    );
    // Whole object, and a single byte, are both ranges of it.
    assert!(ServedRange::partial(0, 39, 40).is_some());
    assert!(ServedRange::partial(39, 39, 40).is_some());
    // Reversed — the subtraction that used to underflow.
    assert_eq!(ServedRange::partial(19, 10, 40), None);
    assert_eq!(ServedRange::partial(1, 0, 40), None);
    // Past the end, and of an empty object.
    assert_eq!(ServedRange::partial(10, 40, 40), None);
    assert_eq!(ServedRange::partial(40, 45, 40), None);
    assert_eq!(ServedRange::partial(0, 0, 0), None);
    assert_eq!(ServedRange::partial(0, u64::MAX, 40), None);
}

#[tokio::test]
async fn read_video_degrades_to_full_when_server_ignores_range() {
    let (url, stub) = spawn_stub("secret").await;
    // A 200 with the full body is a legal answer to a range request.
    stub.ignore_range.store(true, Ordering::Relaxed);
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(10_000, 40))
        .await;
    let entry = backend.find_event("cam", url_key(10_000, 1000)).unwrap();

    // A range was requested, but the full body comes back as a 200.
    let vs = backend
        .read_video(
            "cam",
            &entry,
            Some(RangeRequest::FromTo {
                start: 10,
                end: Some(19),
            }),
        )
        .await
        .unwrap();
    assert_eq!(vs.range, ServedRange::Full);
    assert_eq!(vs.total_size, 40);
    assert_eq!(drain(vs).await.len(), 40);
}

const SEC: u64 = 1_000_000_000;

/// A backend whose index holds `spans` and which never talks to a host.
fn indexed(spans: &[(u64, u32)]) -> StathostBackend {
    let backend = backend_for("http://127.0.0.1:1", "secret", 0);
    for &(start_pts_ns, duration_ms) in spans {
        backend.events.insert(
            "cam",
            WarmEventEntry {
                start_pts_ns,
                duration_ms,
                event_type: EventType::Continuous,
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
            },
        );
    }
    backend
}

#[test]
fn query_returns_long_events_that_started_before_the_window() {
    // A 100s chunk starting at 0, then two 1s events that end long before
    // the window: sorted by start, "ends before from" is false-then-true,
    // so a binary search on it skips right past the chunk that does overlap.
    let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
    let hits = backend.query("cam", EventPage::unbounded(50 * SEC, 60 * SEC));
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].start_pts_ns, 0);
}

#[test]
fn query_returns_every_overlapping_event_in_start_order() {
    let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000), (20 * SEC, 1_000)]);
    let starts: Vec<u64> = backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .iter()
        .map(|e| e.start_pts_ns)
        .collect();
    assert_eq!(starts, vec![0, 10 * SEC, 20 * SEC]);
    assert!(backend
        .query("unknown", EventPage::unbounded(0, u64::MAX))
        .is_empty());
}

#[test]
fn zero_duration_events_are_found_at_their_start() {
    let backend = indexed(&[(10 * SEC, 0)]);
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(10 * SEC, 10 * SEC))
            .len(),
        1
    );
    assert!(backend
        .query("cam", EventPage::unbounded(10 * SEC + 1, 20 * SEC))
        .is_empty());
}

#[test]
fn query_bounds_include_events_that_only_touch_them() {
    let backend = indexed(&[(10 * SEC, 5_000)]);
    // Ends exactly at from_ns.
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(15 * SEC, 20 * SEC))
            .len(),
        1
    );
    assert!(backend
        .query("cam", EventPage::unbounded(15 * SEC + 1, 20 * SEC))
        .is_empty());
    // Starts exactly at to_ns.
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, 10 * SEC))
            .len(),
        1
    );
    assert!(backend
        .query("cam", EventPage::unbounded(0, 10 * SEC - 1))
        .is_empty());
}

#[test]
fn query_with_an_inverted_range_is_empty() {
    // These bounds used to be computed independently and sliced, which
    // panicked here with start > end.
    let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000)]);
    assert!(backend
        .query("cam", EventPage::unbounded(u64::MAX, 0))
        .is_empty());
    assert!(backend
        .query("cam", EventPage::unbounded(20 * SEC, 5 * SEC))
        .is_empty());
}
