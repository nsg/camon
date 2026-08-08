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

#[derive(Clone)]
struct Stub {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    token: String,
    fail_writes: Arc<AtomicBool>,
    put_fault: Arc<Mutex<Option<PutFault>>>,
    fail_get_suffix: Arc<Mutex<Option<String>>>,
    fail_delete_paths: Arc<Mutex<HashSet<String>>>,
    ignore_range: Arc<AtomicBool>,
    bad_content_range: Arc<Mutex<Option<String>>>,
    commit_after_list: Arc<Mutex<Vec<String>>>,
    list_failures: Arc<AtomicUsize>,
    list_delay_ms: Arc<AtomicU64>,
    lists: Arc<AtomicUsize>,
    gets: Arc<Mutex<Vec<String>>>,
    in_flight: Arc<AtomicUsize>,
    peak_gets: Arc<AtomicUsize>,
    get_delay_ms: Arc<AtomicU64>,
    puts: Arc<Mutex<Vec<String>>>,
    put_delay_ms: Arc<AtomicU64>,
    deletes: Arc<Mutex<Vec<String>>>,
    delete_delay_ms: Arc<AtomicU64>,
    hold_puts: Arc<AtomicBool>,
    hold_deletes: Arc<AtomicBool>,
    hold_gets: Arc<AtomicBool>,
}

#[derive(Clone)]
struct PutFault {
    suffix: String,
    stored: bool,
    status: StatusCode,
}

impl Stub {
    fn fail_puts(&self, suffix: &str, stored: bool) {
        self.refuse_puts(suffix, stored, StatusCode::INTERNAL_SERVER_ERROR);
    }

    fn refuse_puts(&self, suffix: &str, stored: bool, status: StatusCode) {
        *self.put_fault.lock().unwrap() = Some(PutFault {
            suffix: suffix.to_string(),
            stored,
            status,
        });
    }

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

    fn fail_next_lists(&self, n: usize) {
        self.list_failures.store(n, Ordering::SeqCst);
    }

    fn serve_lists_again(&self) {
        self.list_failures.store(0, Ordering::SeqCst);
    }

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

    fn stored_bytes(&self) -> u64 {
        self.files
            .lock()
            .unwrap()
            .values()
            .map(|v| v.len() as u64)
            .sum()
    }

    fn take_gets(&self) -> Vec<String> {
        std::mem::take(&mut self.gets.lock().unwrap())
    }

    fn take_puts(&self) -> Vec<String> {
        std::mem::take(&mut self.puts.lock().unwrap())
    }

    fn take_deletes(&self) -> Vec<String> {
        std::mem::take(&mut self.deletes.lock().unwrap())
    }

    fn hold(&self, gate: &Arc<AtomicBool>) {
        gate.store(true, Ordering::SeqCst);
    }

    fn release(&self, gate: &Arc<AtomicBool>) {
        gate.store(false, Ordering::SeqCst);
    }

    fn get_count(&self, path: &str) -> usize {
        self.gets
            .lock()
            .unwrap()
            .iter()
            .filter(|p| *p == path)
            .count()
    }
}

async fn drain(vs: VideoStream) -> Vec<u8> {
    use futures_util::StreamExt;
    let mut buf = Vec::new();
    let mut stream = vs.stream;
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.unwrap());
    }
    buf
}

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
        _ => {
            stub.gets.lock().unwrap().push(path.clone());
            let now = stub.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            stub.peak_gets.fetch_max(now, Ordering::SeqCst);
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
        Some(_) if stub.ignore_range.load(Ordering::Relaxed) => full_200(bytes),
        Some(r) => match parse_stub_range(r, total) {
            Some((start, end)) => {
                let slice = bytes[start as usize..=end as usize].to_vec();
                let mut resp = (StatusCode::PARTIAL_CONTENT, slice).into_response();
                let content_range = match stub.bad_content_range.lock().unwrap().clone() {
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

async fn wait_on_gate(gate: &Arc<AtomicBool>) {
    for _ in 0..10_000 {
        if !gate.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    for _ in 0..5_000 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the stub never reached the state this test needs");
}

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

async fn scanned_backend_for(url: &str, token: &str, max_stored_bytes: u64) -> StathostBackend {
    let backend = backend_for(url, token, max_stored_bytes);
    backend.scan().await.unwrap();
    backend
}

fn segment(start_pts: u64, byte: u8, len: usize) -> GopSegment {
    GopSegment {
        start_pts,
        duration_ns: 1_000_000_000,
        data: Arc::new(vec![byte; len]),
        frame_count: 1,
    }
}

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

fn longer_movement_event(first_pts: u64, size: usize) -> FinishedEvent {
    let mut e = movement_event(first_pts, size);
    e.segments.push(segment(first_pts + SEC, 0xcd, size));
    e.total_bytes = size * 2;
    e
}

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

#[tokio::test]
async fn write_then_scan_round_trip_detailed_list() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);

    let event = movement_event(1_000, 40);
    assert_eq!(
        backend.write_event("cam", &event).await,
        WriteOutcome::Written
    );

    let entry = backend.find_event("cam", url_key(1_000, 1000)).unwrap();
    assert_eq!(entry.event_type, EventType::Movement);
    assert_eq!(entry.file_size, 40);
    assert_eq!(entry.filmstrip_frames, 2);

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

    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(1_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Movement);
    assert_eq!(e.file_size, 40);
    assert_eq!(e.filmstrip_frames, 2);
    assert_eq!(scanned.free_space().unwrap(), u64::MAX); // unlimited budget
}

#[tokio::test]
async fn a_rewritten_stem_replaces_its_entry_rather_than_adding_one() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);

    backend.write_event("cam", &movement_event(1_000, 40)).await;
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
    assert_eq!(backend.used(), stub.stored_bytes());
    assert_eq!(stub.files.lock().unwrap().len(), 4);
}

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
        let vs = backend.read_video("cam", &entry, None).await.unwrap();
        assert_eq!(vs.total_size, file_size);
        assert_eq!(drain(vs).await.len(), file_size as usize);
    }

    assert!(backend.find_event("cam", url_key(30_000, 3000)).is_none());
}

#[tokio::test]
async fn find_event_resolves_by_stem_across_an_upgrade() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(31_000, 40))
        .await;
    backend.upgrade_event("cam", &upgrade_for(31_000)).await;

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
    assert!(backend
        .find_event("cam", EventRef::new(31_000, 2000, EventType::Object))
        .is_none());
}

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

    stub.fail_gets("_2000.json");
    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 2000)));
    assert!(!backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    stub.fail_delete_paths
        .lock()
        .unwrap()
        .insert(format!("cam/{OLD_PTS}_2000.ts"));
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(sibling(&backend, 1000).is_none());
    assert!(sibling(&backend, 2000).is_some());

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

    let ts_key = "cam/5000_1000.ts";
    let before = stub.files.lock().unwrap().get(ts_key).cloned().unwrap();

    let mut upgrade = upgrade_for(5_000);
    upgrade.continues = true;
    backend.upgrade_event("cam", &upgrade).await;

    let e = backend.find_event("cam", url_key(5_000, 1000)).unwrap();
    assert!(e.continues);
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.object_classes, vec!["person".to_string()]);
    assert_eq!(
        stub.files.lock().unwrap().get(ts_key).cloned().unwrap(),
        before
    );
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
    let old_pts = 1_000_000_000; // 1s after epoch
    backend
        .write_event("cam", &movement_event(old_pts, 30))
        .await;
    assert_eq!(stub.files.lock().unwrap().len(), 4); // ts + json + 2 thumbs

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    assert!(backend.find_event("cam", url_key(old_pts, 1000)).is_none());
    assert!(stub.files.lock().unwrap().is_empty());
    assert_eq!(backend.used(), 0);
}

#[tokio::test]
async fn prune_caps_how_much_one_sweep_deletes() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    for i in 0..40u64 {
        backend
            .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
            .await;
    }

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

#[tokio::test]
async fn an_undeletable_event_does_not_block_the_sweep_forever() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    for i in 0..12u64 {
        backend
            .write_event("cam", &movement_event(1_000_000_000 + i * 1_000_000, 10))
            .await;
    }
    {
        let mut refused = stub.fail_delete_paths.lock().unwrap();
        for i in 0..4u64 {
            refused.insert(format!("cam/{}_1000.ts", 1_000_000_000 + i * 1_000_000));
        }
    }

    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        12
    );

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

    backend
        .write_event("cam", &movement_event(1_000_000_000, 30))
        .await;
    let stored = stub.files.lock().unwrap().len();

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

#[tokio::test]
async fn a_scan_that_never_listed_the_store_refuses_to_enforce_the_budget() {
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
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
    backend
        .prune(u64::MAX, 1, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(backend
        .query("cam", EventPage::unbounded(0, u64::MAX))
        .is_empty());
}

#[tokio::test]
async fn the_retention_tick_heals_an_index_the_startup_scan_never_built() {
    let (url, stub) = spawn_stub("secret").await;
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

    let listed = stub.lists();
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert_eq!(stub.lists(), listed, "kept re-scanning a scanned index");
}

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

#[tokio::test]
async fn a_healing_rescan_leaves_orphaned_metadata_for_the_next_startup() {
    let (url, stub) = spawn_stub("secret").await;
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
    let restarted = backend_for(&url, "secret", 0);
    restarted.scan().await.unwrap();
    assert!(!stub.has("cam/5000_1000.json"), "orphan sidecar kept");
}

#[tokio::test]
async fn a_listing_that_never_answers_gives_up_on_the_clock_not_the_attempt_count() {
    let (url, stub) = spawn_stub("secret").await;
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

#[tokio::test]
async fn a_shutdown_stops_the_scan_from_retrying() {
    let (url, stub) = spawn_stub("secret").await;
    stub.fail_next_lists(usize::MAX);
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
    sleep_unless(Duration::from_secs(30), &|| flag.load(Ordering::SeqCst)).await;
    let elapsed = started.elapsed();
    raiser.await.unwrap();

    assert!(
        elapsed < crate::storage::contract::SHUTDOWN_POLL * 4,
        "the wait sat through {elapsed:?} of a shutdown"
    );
    assert!(elapsed >= Duration::from_millis(20), "did not wait at all");
}

#[tokio::test]
async fn a_shutdown_part_way_through_a_scan_leaves_it_unscanned() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000_000_000, 4, 1000, "movement");
    stub.fail_next_lists(usize::MAX);
    let backend = Arc::new(backend_for(&url, "secret", 0));
    assert!(backend.scan().await.is_err());

    stub.serve_lists_again();
    stub.get_delay_ms.store(100, Ordering::Relaxed);
    let cancel = Arc::new(AtomicBool::new(false));
    let healing = tokio::spawn({
        let (backend, cancel) = (Arc::clone(&backend), Arc::clone(&cancel));
        async move { backend.prune(1, u64::MAX, u64::MAX, &cancel).await }
    });

    wait_until(|| !stub.gets.lock().unwrap().is_empty()).await;
    cancel.store(true, Ordering::SeqCst);
    healing.await.unwrap();

    assert!(stub.has("cam/1000000000_1000.ts"));
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

#[tokio::test]
async fn a_heal_yields_to_an_upgrade_that_landed_while_it_was_reading() {
    const LIVE_PTS: u64 = 5_000_000_000;
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000_000_000, 2, 1000, "movement");
    stub.fail_next_lists(usize::MAX);
    let backend = Arc::new(backend_for(&url, "secret", 0));
    assert!(backend.scan().await.is_err());

    backend
        .write_event("cam", &movement_event(LIVE_PTS, 40))
        .await;

    stub.serve_lists_again();
    stub.get_delay_ms.store(100, Ordering::Relaxed);
    let healing = tokio::spawn({
        let backend = Arc::clone(&backend);
        async move {
            backend
                .prune(u64::MAX, u64::MAX, u64::MAX, &AtomicBool::new(false))
                .await
        }
    });

    let sidecar = format!("cam/{LIVE_PTS}_1000.json");
    wait_until(|| stub.get_count(&sidecar) >= 1).await;
    backend.upgrade_event("cam", &upgrade_for(LIVE_PTS)).await;
    healing.await.unwrap();

    let entry = backend.find_event("cam", url_key(LIVE_PTS, 1000)).unwrap();
    assert_eq!(
        entry.event_type,
        EventType::Object,
        "the heal wrote a stale movement over an upgraded event"
    );
    assert_eq!(entry.object_classes, vec!["person".to_string()]);
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        3
    );
}

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
    assert_eq!(entry.file_size, 40);
    assert_eq!(entry.filmstrip_frames, 2);
}

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

    let backend = over_budget_backend(&url, &[continuous, obj, movement], budget).await;
    assert!(backend.used() > budget, "the store is not over its budget");

    backend.guard_free_space("cam", 0).await;

    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none()); // continuous evicted
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none()); // movement evicted
    assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some()); // object survives
    assert!(backend.used() <= budget);
}

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

    backend.guard_free_space("cam", 0).await;
    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_some());
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(3_000, 1000)).is_none());
    assert!(stub.has("cam/1000_1000.ts"));
}

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
    stub.files.lock().unwrap().remove("cam/1000_1000.ts");

    backend.guard_free_space("cam", 0).await;

    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(3_000, 1000)).is_some());
    assert_eq!(backend.used(), cost);
}

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

    backend
        .prune(u64::MAX, u64::MAX, 1, &AtomicBool::new(false))
        .await;

    let entry = backend.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
    assert!(entry.delete_failed);
    assert!(stub.has(&format!("cam/{OLD_PTS}_1000.ts")));
    assert!(stub.has(&format!("cam/{OLD_PTS}_1000.json")), "type lost");
    assert!(!stub.has(&format!("cam/{OLD_PTS}_1000_thumb_0.jpg")));

    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert_eq!(
        scanned
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .event_type,
        EventType::Continuous
    );
    stub.fail_delete_paths.lock().unwrap().clear();
    scanned
        .prune(u64::MAX, u64::MAX, 1, &AtomicBool::new(false))
        .await;
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
    assert!(stub.files.lock().unwrap().is_empty());
}

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
    stub.files
        .lock()
        .unwrap()
        .insert("cam/9000_1000.ts".to_string(), vec![0u8; 10]);

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    let e = backend.find_event("cam", url_key(9_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Movement);
    assert_eq!(e.filmstrip_frames, 0);
    assert!(e.object_classes.is_empty());
}

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

#[tokio::test]
async fn the_scan_reads_sidecars_concurrently() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, 64, 1000, "object");
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
    assert!(
        elapsed < Duration::from_millis(2_000),
        "sidecar reads were serial: {elapsed:?}"
    );
    let peak = stub.peak_gets.load(Ordering::SeqCst);
    assert!(peak > 1, "no reads overlapped");
    assert!(peak <= SCAN_CONCURRENCY, "fan-out ran unbounded: {peak}");
}

#[tokio::test]
async fn the_startup_scan_builds_the_index_without_shifting_it_into_shape() {
    const EVENTS: u64 = 1_000;
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, EVENTS, 1000, "object");

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    let entries = backend.query("cam", EventPage::unbounded(0, u64::MAX));
    assert_eq!(entries.len() as u64, EVENTS);
    assert!(entries
        .windows(2)
        .all(|w| w[0].start_pts_ns <= w[1].start_pts_ns));
    assert_eq!(
        backend.used(),
        EVENTS * (10 + r#"{"event_type":"object"}"#.len() as u64)
    );

    let shifted = backend.events.shifted_entries();
    assert!(
        shifted <= 4 * EVENTS,
        "the scan shifted {shifted} entries to index {EVENTS} events"
    );
}

#[tokio::test]
async fn a_startup_pass_that_stops_part_way_keeps_what_it_had_read() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000_000_000, 8, 1000, "movement");
    let backend = backend_for(&url, "secret", 0);

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

#[tokio::test]
async fn a_scanned_filmstrip_with_a_gap_counts_the_frames_the_store_still_has() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        files.insert("cam/1000_1000.ts".to_string(), vec![0u8; 10]);
        files.insert("cam/1000_1000_thumb_1.jpg".to_string(), vec![0u8; 7]);
        files.insert("cam/1000_1000_thumb_3.jpg".to_string(), vec![0u8; 9]);
    }

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();
    let entry = backend.find_event("cam", url_key(1000, 1000)).unwrap();

    assert_eq!(entry.filmstrip_frames, 4);
    assert_eq!(entry.thumbnail_bytes, 16);

    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(!stub.has("cam/1000_1000_thumb_1.jpg"));
    assert!(!stub.has("cam/1000_1000_thumb_3.jpg"));
    assert!(stub.files.lock().unwrap().is_empty());
}

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

#[tokio::test]
async fn a_concurrent_scan_indexes_every_event_once_and_in_order() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        for i in 0..40u64 {
            let stem = format!("{}_1000", 1_000 + i * SEC);
            files.insert(format!("cam/{stem}.ts"), vec![0u8; 10]);
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
    assert_eq!(backend.used(), stub.stored_bytes());
}

#[tokio::test]
async fn a_concurrent_scan_never_reads_a_failure_as_an_absence() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, 1_000, 20, 1000, "object");
    seed_events(&stub, 1_000, 20, 2000, "object");
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
    assert!(backend.find_event("cam", url_key(6_000, 1000)).is_none());
    assert!(stub.files.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_failed_sidecar_fails_the_write_before_the_video() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".json", false);

    let outcome = backend.write_event("cam", &object_event(11_000, 30)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert!(backend.find_event("cam", url_key(11_000, 1000)).is_none());
    assert_eq!(backend.used(), 0);
    assert!(!stub.has("cam/11000_1000.ts"));
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert!(scanned.find_event("cam", url_key(11_000, 1000)).is_none());
}

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

#[tokio::test]
async fn a_movement_event_scans_back_identically_without_its_sidecar() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
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

#[tokio::test]
async fn a_video_that_lands_despite_a_failed_put_keeps_its_type() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.fail_puts(".ts", true); // committed server-side, 500 to the client

    let outcome = backend.write_event("cam", &object_event(12_000, 30)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert!(backend.find_event("cam", url_key(12_000, 1000)).is_none());
    stub.clear_faults();
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(12_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.object_classes, vec!["car".to_string()]);
}

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
    assert_eq!(stub.get_count("cam/17000_1000.ts"), 1);
    stub.clear_faults();
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    assert!(scanned.find_event("cam", url_key(17_000, 1000)).is_none());
}

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

#[tokio::test]
async fn orphaned_thumbnails_are_collected_and_live_ones_are_not() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(19_000, 30))
        .await;
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

#[tokio::test]
async fn the_sweep_keeps_a_sidecar_whose_video_landed_after_the_listing() {
    let (url, stub) = spawn_stub("secret").await;
    stub.files.lock().unwrap().insert(
        "cam/22000_1000.json".to_string(),
        br#"{"event_type":"object"}"#.to_vec(),
    );
    stub.commit_after_list
        .lock()
        .unwrap()
        .push("cam/22000_1000.ts".to_string());

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert!(stub.has("cam/22000_1000.json"), "live sidecar collected");
    assert!(stub.has("cam/22000_1000.ts"));
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

#[tokio::test]
async fn the_sweep_keeps_metadata_it_could_not_check() {
    let (url, stub) = spawn_stub("secret").await;
    stub.files
        .lock()
        .unwrap()
        .insert("cam/23000_1000.json".to_string(), b"{}".to_vec());
    stub.fail_gets(".ts");

    let backend = backend_for(&url, "secret", 0);
    backend.scan().await.unwrap();

    assert!(stub.has("cam/23000_1000.json"));
}

#[tokio::test]
async fn the_scan_only_collects_orphans_of_cameras_it_owns() {
    let (url, stub) = spawn_stub("secret").await;
    {
        let mut files = stub.files.lock().unwrap();
        files.insert("other/21000_1000.json".to_string(), b"{}".to_vec());
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

    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();
    let e = scanned.find_event("cam", url_key(13_000, 1000)).unwrap();
    assert_eq!(e.event_type, EventType::Object);
    assert_eq!(e.filmstrip_frames, 0);
    assert!(backend.read_thumbnail("cam", &entry).await.is_err());
}

const OLD_PTS: u64 = 1_000_000_000;

#[tokio::test]
async fn an_unreadable_sidecar_is_not_pruned_as_a_movement_event() {
    let (url, stub) = spawn_stub("secret").await;
    backend_for(&url, "secret", 0)
        .write_event("cam", &object_event(OLD_PTS, 30))
        .await;

    stub.fail_gets(".json");
    let scanned = backend_for(&url, "secret", 0);
    scanned.scan().await.unwrap();

    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    scanned
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    assert!(stub.has("cam/1000000000_1000.ts"));

    scanned.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(scanned.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
    assert!(!stub.has("cam/1000000000_1000.ts"));
}

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

    stub.clear_faults();
    backend
        .prune(1, u64::MAX, u64::MAX, &AtomicBool::new(false))
        .await;

    let entry = backend.find_event("cam", url_key(OLD_PTS, 1000)).unwrap();
    assert_eq!(entry.event_type, EventType::Object);
    assert_eq!(entry.object_classes, vec!["car".to_string()]);
    assert!(!backend.has_unknown_type("cam", (OLD_PTS, 1000)));

    backend.prune(1, u64::MAX, 1, &AtomicBool::new(false)).await;
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_none());
}

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
    assert!(backend.find_event("cam", url_key(OLD_PTS, 1000)).is_some());
    assert!(backend.has_unknown_type("cam", (OLD_PTS, 1000)));
}

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

#[tokio::test]
async fn budget_eviction_tiers_an_unknown_type_with_the_objects() {
    let (url, stub) = spawn_stub("secret").await;
    let writer = backend_for(&url, "secret", 0);
    writer.write_event("cam", &object_event(1_000, 40)).await;
    writer.write_event("cam", &movement_event(2_000, 40)).await;

    stub.fail_gets("1000_1000.json");
    let backend = backend_for(&url, "secret", cost_of(&object_event(1_000, 40)));
    backend.scan().await.unwrap();
    assert!(backend.has_unknown_type("cam", (1_000, 1000)));

    backend.guard_free_space("cam", 0).await;
    assert!(backend.find_event("cam", url_key(2_000, 1000)).is_none());
    assert!(backend.find_event("cam", url_key(1_000, 1000)).is_some());
}

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

#[test]
fn a_plain_movement_sidecar_carries_nothing_but_its_type() {
    assert_eq!(
        sidecar_json(Some(EventType::Movement), None, None, &[], false),
        r#"{"detections":[],"event_type":"movement"}"#
    );
}

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

#[tokio::test]
async fn a_write_issues_no_further_upload_once_shutdown_is_asked_for() {
    let (url, stub) = spawn_stub("secret").await;
    let flag = Arc::new(AtomicBool::new(false));
    let backend = backend_stopped_by(&url, "secret", 0, StopFlag::shared(Arc::clone(&flag)));
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

#[tokio::test]
async fn two_writes_in_flight_cannot_both_walk_through_the_budget() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let backend = over_budget_backend(
        &url,
        &[movement_event(1_000, 40), movement_event(2_000, 40)],
        cost * 3,
    )
    .await;
    assert_eq!(backend.used(), cost * 2);
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

#[tokio::test]
async fn a_refused_upload_is_not_sent_a_second_time() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.refuse_puts(".json", false, StatusCode::FORBIDDEN);

    let outcome = backend.write_event("cam", &object_event(1_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert_eq!(stub.put_count("cam/1000_1000.json"), 1);
}

#[tokio::test]
async fn a_store_having_a_moment_gets_its_second_attempt() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    stub.refuse_puts(".json", false, StatusCode::SERVICE_UNAVAILABLE);

    let outcome = backend.write_event("cam", &object_event(1_000, 40)).await;

    assert_eq!(outcome, WriteOutcome::Failed);
    assert_eq!(stub.put_count("cam/1000_1000.json"), 2);
}

#[tokio::test]
async fn an_upgrade_overtaken_by_a_sweep_leaves_no_sidecar_behind() {
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
    assert_eq!(
        stub.get_count(&format!("cam/{OLD_PTS}_1000.ts")),
        0,
        "the upgrade probed the store for something the index already knew"
    );
}

#[tokio::test]
async fn a_stopped_write_over_a_full_store_deletes_nothing() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let seeder = backend_for(&url, "secret", 0);
    for pts in [1_000u64, 2_000] {
        seeder.write_event("cam", &movement_event(pts, 40)).await;
    }
    let flag = Arc::new(AtomicBool::new(false));
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
    assert!(
        !backend
            .find_event("cam", url_key(OLD_PTS, 1000))
            .unwrap()
            .delete_failed
    );
}

#[tokio::test]
async fn an_eviction_already_under_way_stops_when_shutdown_arrives() {
    let (url, stub) = spawn_stub("secret").await;
    let cost = cost_of(&movement_event(0, 40));
    let seeder = backend_for(&url, "secret", 0);
    for pts in [1_000u64, 2_000, 3_000, 4_000] {
        seeder.write_event("cam", &movement_event(pts, 40)).await;
    }
    let flag = Arc::new(AtomicBool::new(false));
    let backend = backend_stopped_by(&url, "secret", cost, StopFlag::shared(Arc::clone(&flag)));
    backend.scan().await.unwrap();
    stub.take_deletes();
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

#[tokio::test]
async fn a_sweep_stops_between_deletes_on_the_cancel_it_was_given() {
    let (url, stub) = spawn_stub("secret").await;
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

#[tokio::test]
async fn a_deletion_cut_short_leaves_thumbnails_a_rebuild_can_still_see() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 0).await;
    let mut event = movement_event(OLD_PTS, 40);
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

    assert_eq!(
        stub.take_deletes(),
        vec![format!("cam/{stem}_thumb_2.jpg")],
        "the sweep deleted the wrong end of the filmstrip, or did not stop"
    );
    assert!(stub.has(&format!("cam/{stem}_thumb_0.jpg")));
    assert!(stub.has(&format!("cam/{stem}_thumb_1.jpg")));
    assert!(stub.has(&format!("cam/{stem}.ts")));

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
        refused.insert(format!("cam/{stem}_thumb_2.jpg"));
        refused.insert(format!("cam/{stem}.ts"));
    }

    backend.prune(1, 1, 1, &AtomicBool::new(false)).await;

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

    stub.fail_delete_paths.lock().unwrap().clear();
    let restarted = backend_for(&url, "secret", 0);
    restarted.scan().await.unwrap();
    assert!(
        stub.files.lock().unwrap().is_empty(),
        "the startup sweep did not collect the frames a refused delete stranded: {:?}",
        stub.files.lock().unwrap().keys().collect::<Vec<_>>()
    );
}

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
    assert_eq!(backend.free_space().unwrap(), 0);
}

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

#[tokio::test]
async fn an_upgrade_claims_the_growth_of_the_sidecar_it_is_writing() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 100_000).await;
    backend.write_event("cam", &movement_event(1_000, 40)).await;
    let before = backend.free_space().unwrap();
    stub.take_puts();
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
    assert_eq!(backend.free_space().unwrap(), during);
}

#[tokio::test]
async fn a_rewrites_reservation_is_released_before_its_thumbnails_are_trimmed() {
    let (url, stub) = spawn_stub("secret").await;
    let backend = scanned_backend_for(&url, "secret", 100_000).await;
    let mut event = movement_event(2_000, 30);
    event.filmstrip_frames = Some(Arc::new(vec![vec![0x01], vec![0x02], vec![0x03]]));
    backend.write_event("cam", &event).await;

    let mut shorter = movement_event(2_000, 30);
    shorter.filmstrip_frames = Some(Arc::new(vec![vec![0x09]]));
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

#[tokio::test]
async fn a_prune_does_not_wait_out_a_held_read_past_its_budget() {
    let (url, stub) = spawn_stub("secret").await;
    seed_events(&stub, OLD_PTS, 8, 1000, "object");
    stub.fail_gets(".json");
    let backend = scanned_backend_for(&url, "secret", 0).await;
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

#[tokio::test]
async fn consecutive_prune_ticks_reach_different_held_events() {
    let (url, stub) = spawn_stub("secret").await;
    const HELD: u64 = 80;
    seed_events(&stub, OLD_PTS, HELD, 1000, "object");
    stub.fail_gets(".json");
    let backend = scanned_backend_for(&url, "secret", 0).await;
    stub.take_gets();
    stub.get_delay_ms.store(25, Ordering::SeqCst);

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
    stub.hold(&stub.hold_gets);

    let hold_at = |i: u64| format!("cam/{}_1000.json", OLD_PTS + i * SEC);
    let sweep = |stub: &Stub| -> HashSet<String> { stub.take_gets().into_iter().collect() };

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

#[tokio::test]
async fn an_upgrade_that_crosses_the_budget_says_so() {
    let (url, stub) = spawn_stub("secret").await;
    let event = movement_event(1_000, 40);
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

#[tokio::test]
async fn a_prune_bounds_the_time_it_spends_re_reading_held_types() {
    let (url, stub) = spawn_stub("secret").await;
    const HELD: u64 = 100;
    seed_events(&stub, OLD_PTS, HELD, 1000, "object");
    stub.fail_gets(".json");
    let backend = scanned_backend_for(&url, "secret", 0).await;
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(0, u64::MAX))
            .len(),
        HELD as usize
    );
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

#[tokio::test]
async fn read_video_serves_partial_and_suffix_ranges() {
    let (url, _stub) = spawn_stub("secret").await;
    let backend = backend_for(&url, "secret", 0);
    backend.write_event("cam", &movement_event(8_000, 40)).await;
    let entry = backend.find_event("cam", url_key(8_000, 1000)).unwrap();

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

    *stub.bad_content_range.lock().unwrap() = None;
    let vs = backend
        .read_video("cam", &entry, Some(ranged))
        .await
        .unwrap();
    assert_eq!(vs.range, ServedRange::Partial { start: 10, end: 19 });
    assert_eq!(drain(vs).await, vec![0xab; 10]);
}

#[test]
fn a_partial_range_has_to_be_a_range_of_the_object() {
    assert_eq!(
        ServedRange::partial(10, 19, 40),
        Some(ServedRange::Partial { start: 10, end: 19 })
    );
    assert!(ServedRange::partial(0, 39, 40).is_some());
    assert!(ServedRange::partial(39, 39, 40).is_some());
    assert_eq!(ServedRange::partial(19, 10, 40), None);
    assert_eq!(ServedRange::partial(1, 0, 40), None);
    assert_eq!(ServedRange::partial(10, 40, 40), None);
    assert_eq!(ServedRange::partial(40, 45, 40), None);
    assert_eq!(ServedRange::partial(0, 0, 0), None);
    assert_eq!(ServedRange::partial(0, u64::MAX, 40), None);
}

#[tokio::test]
async fn read_video_degrades_to_full_when_server_ignores_range() {
    let (url, stub) = spawn_stub("secret").await;
    stub.ignore_range.store(true, Ordering::Relaxed);
    let backend = backend_for(&url, "secret", 0);
    backend
        .write_event("cam", &movement_event(10_000, 40))
        .await;
    let entry = backend.find_event("cam", url_key(10_000, 1000)).unwrap();

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
    assert_eq!(
        backend
            .query("cam", EventPage::unbounded(15 * SEC, 20 * SEC))
            .len(),
        1
    );
    assert!(backend
        .query("cam", EventPage::unbounded(15 * SEC + 1, 20 * SEC))
        .is_empty());
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
    let backend = indexed(&[(0, 100_000), (10 * SEC, 1_000)]);
    assert!(backend
        .query("cam", EventPage::unbounded(u64::MAX, 0))
        .is_empty());
    assert!(backend
        .query("cam", EventPage::unbounded(20 * SEC, 5 * SEC))
        .is_empty());
}
