//! Home Assistant MQTT bridge: outbound-only publisher of discovery documents, sensor states
//! and snapshots. Nothing is ever subscribed to, so a misbehaving broker cannot drive camon.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use rumqttc::{AsyncClient, Event, Incoming, LastWill, MqttOptions, QoS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::buffer::HotBuffer;
use crate::config::MqttConfig;
use crate::durable::{create_dir_all_synced, parent_dir, sync_dir, tmp_path, write_synced};
use crate::locks::LockExt;

/// Depth of the analyzer/detector -> bridge event channel. Producers
/// `try_send` and drop on full rather than ever stalling the analysis threads.
pub const MQTT_EVENT_CAPACITY: usize = 64;

/// Headroom above one reconnect burst: room for the transitions and snapshots
/// raised while that burst is still being drained.
const REQUEST_QUEUE_HEADROOM: usize = 256;

/// How many publishes one reconnect burst is. A test ties this to what
/// [`reconnect_burst`] actually produces.
fn burst_len(cameras: usize, classes: usize, orphans: usize) -> usize {
    orphans + cameras * (3 + 3 * classes) + 1
}
/// Depth of the rumqttc request queue, sized from the config so that one whole reconnect burst
/// always fits: [`publish_burst`] is all-or-nothing, so a burst longer than the queue could
/// never be published at all.
fn request_queue_capacity(cameras: usize, classes: usize, orphans: usize) -> usize {
    burst_len(cameras, classes, orphans) + REQUEST_QUEUE_HEADROOM
}

/// Outgoing packet-size ceiling. rumqttc defaults to 10 KiB, which every
/// snapshot would exceed; a 1280x720 JPEG at quality 90 is a few hundred KiB.
const MAX_OUTGOING_PACKET_BYTES: usize = 4 * 1024 * 1024;

/// How many image bytes may be handed to the request queue between two ticks.
const MAX_IMAGE_BYTES_PER_TICK: usize = 16 * 1024 * 1024;

/// The image allowance of [`MAX_IMAGE_BYTES_PER_TICK`], shared by the bridge
/// loop and every detached snapshot task, since both queue images.
#[derive(Clone, Default)]
struct ImageBudget {
    spent: Arc<std::sync::atomic::AtomicUsize>,
    /// Whether this window's refusal has been reported: once a second, not
    /// once per camera per second.
    reported: Arc<AtomicBool>,
}

impl ImageBudget {
    /// Charge one image, reporting whether it may be queued. A refusal just
    /// drops the image; the camera tile keeps the frame it has.
    fn take(&self, bytes: usize) -> bool {
        let taken = self
            .spent
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |spent| {
                spent
                    .checked_add(bytes)
                    .filter(|&total| total <= MAX_IMAGE_BYTES_PER_TICK)
            })
            .is_ok();
        if !taken && !self.reported.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                bytes,
                limit = MAX_IMAGE_BYTES_PER_TICK,
                "mqtt image budget spent for this second, dropping images"
            );
        }
        taken
    }

    fn refill(&self) {
        self.spent.store(0, Ordering::Relaxed);
        self.reported.store(false, Ordering::Relaxed);
    }
}

/// The MQTT client id this instance connects under. See [`derive_client_id`].
fn client_id(ctx: &BridgeContext) -> String {
    derive_client_id(
        &hostname().unwrap_or_default(),
        ctx.entities_path.as_deref(),
    )
}

/// Derive the client id camon connects under, as `camon-<[`CLIENT_ID_HASH_BYTES`] hex>`.
fn derive_client_id(hostname: &str, entities_path: Option<&Path>) -> String {
    use std::os::unix::ffi::OsStrExt;
    let mut hasher = Sha256::new();
    hasher.update(hostname.as_bytes());
    // A separator no path or hostname can contain, so that a hostname ending
    // where a path begins cannot read as another pair.
    hasher.update([0u8]);
    if let Some(path) = entities_path {
        hasher.update(path.as_os_str().as_bytes());
    }
    let digest = hasher.finalize();
    let mut id = String::from("camon-");
    for byte in &digest[..CLIENT_ID_HASH_BYTES] {
        use std::fmt::Write as _;
        let _ = write!(id, "{byte:02x}");
    }
    id
}

/// How much of the digest the client id carries. Six bytes keeps the whole id
/// at 18 characters — inside the 23 that MQTT 3.1.1 requires every broker to
/// accept.
const CLIENT_ID_HASH_BYTES: usize = 6;

/// The kernel's hostname, or `None` when there is none to read.
fn hostname() -> Option<String> {
    let mut buf = [0u8; 256];
    // SAFETY: `gethostname` writes at most `buf.len()` bytes into a buffer of
    // exactly that size, which is ours alone for the length of the call.
    if unsafe { libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) } != 0 {
        return None;
    }
    // POSIX allows a truncated name to come back unterminated, so the end is
    // the first NUL *or* the end of the buffer.
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..end]).into_owned();
    (!name.is_empty()).then_some(name)
}

/// Delay before re-polling after a connection error. `poll()` reconnects on its
/// own but does not pace itself, so without this a down broker becomes a busy
/// loop.
const RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// Depth of the eventloop task -> bridge notification channel. Entries are
/// connection edges, already paced by [`RECONNECT_DELAY`].
const LINK_EVENT_CAPACITY: usize = 16;

/// How long shutdown waits for the eventloop task to end once told to stop. A
/// broker that leaves the socket open parks the task in `poll()`, where no
/// signal reaches; this bound keeps shutdown from waiting on it.
const EVENTLOOP_STOP_JOIN: Duration = Duration::from_millis(200);

/// Snapshot output size. Fixed regardless of camera aspect ratio (letterboxed
/// by the decode filter) so Home Assistant's camera tile never jumps around.
const SNAPSHOT_WIDTH: u32 = 1280;
const SNAPSHOT_HEIGHT: u32 = 720;

/// JPEG quality for snapshots, matching the analytics pipeline's.
const SNAPSHOT_JPEG_QUALITY: u8 = 90;

/// How long shutdown waits for the retained `offline` marker and the DISCONNECT
/// to reach the socket.
const SHUTDOWN_FLUSH: Duration = Duration::from_secs(2);

/// The longest snapshot cadence camon will honour, clamped with a warning:
/// an operator's `u64` must not be turned into an `Instant` that overflows.
const MAX_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// How soon a snapshot that produced nothing is attempted again, instead of
/// waiting out the whole cadence. Only ever shortens the wait.
const SNAPSHOT_RETRY_DELAY: Duration = Duration::from_secs(2);

/// How long one snapshot decode may run before its ffmpeg is killed — a bound
/// on wedged ffmpegs, not a policing of healthy decodes.
const SNAPSHOT_DECODE_TIMEOUT: Duration = Duration::from_secs(15);

/// How long shutdown waits for aborted snapshot tasks to actually end. Abort
/// lands at an await in microseconds; this exists so a task blocking its
/// worker thread cannot hold shutdown open.
const SNAPSHOT_ABORT_JOIN: Duration = Duration::from_millis(500);

/// Something the bridge should reflect to Home Assistant. Produced by the
/// analyzer threads (motion lifecycle) and the detection worker (verdicts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqttEvent {
    /// A motion run opened. Chunked continuations of one physical motion period
    /// do NOT produce a second start — see `analytics::pipeline`.
    MotionStart { camera_id: String },
    /// The motion run closed (post-padding elapsed, or a shutdown flush).
    MotionEnd { camera_id: String },
    /// A detection verdict, already deduplicated and confidence-filtered: one
    /// entry per named class, each with the frame that evidences it.
    Detections {
        camera_id: String,
        sightings: Vec<Sighting>,
    },
}

/// One class a verdict named, together with the picture behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sighting {
    /// The object class, as configured in `analytics.object_detection.classes`.
    pub class: String,
    /// The JPEG the vision model classified. `None` when the job carried no
    /// frame; the snapshot entity then keeps showing the previous sighting.
    pub frame_jpeg: Option<Vec<u8>>,
}

/// Hand an event to the bridge without ever blocking the producer — stalling
/// here would cost footage. A full queue drops the event with a warning; MQTT
/// state is refreshed on the next transition and on reconnect.
pub fn send_event(tx: &tokio::sync::mpsc::Sender<MqttEvent>, event: MqttEvent) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    match tx.try_send(event) {
        Ok(()) => true,
        Err(TrySendError::Full(event)) => {
            tracing::warn!(event = ?event, "mqtt event queue full, dropping event");
            false
        }
        Err(TrySendError::Closed(event)) => {
            tracing::debug!(event = ?event, "mqtt bridge gone, dropping event");
            false
        }
    }
}

/// Lowercase slug with every non-alphanumeric character folded to `_`, for
/// MQTT topics and Home Assistant unique ids. `Config::validate` uses the same
/// function to reject ids that would collide once normalized.
pub(crate) fn slugify(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Capitalize the first character for entity names ("person" -> "Person").
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Every topic the bridge writes to, derived once from the config.
struct Topics {
    prefix: String,
    discovery_prefix: String,
}

impl Topics {
    fn new(config: &MqttConfig) -> Self {
        Self::from_prefixes(&config.topic_prefix, &config.discovery_prefix)
    }

    /// Trailing slashes are trimmed here rather than at the call sites, so a
    /// remembered prefix compares equal to the same prefix written with one.
    fn from_prefixes(prefix: &str, discovery_prefix: &str) -> Self {
        Self {
            prefix: prefix.trim_end_matches('/').to_string(),
            discovery_prefix: discovery_prefix.trim_end_matches('/').to_string(),
        }
    }

    /// The one availability topic for every entity of this prefix — the *prefix's*, not the
    /// instance's.
    fn availability(&self) -> String {
        format!("{}/availability", self.prefix)
    }

    fn motion(&self, camera_id: &str) -> String {
        format!("{}/{}/motion", self.prefix, camera_id)
    }

    fn occupancy(&self, camera_id: &str, class: &str) -> String {
        format!("{}/{}/occupancy/{}", self.prefix, camera_id, class)
    }

    /// The retained crop behind the last sighting of `class`. A subtopic of the
    /// occupancy state topic so the two read as one pair in an MQTT explorer.
    fn occupancy_snapshot(&self, camera_id: &str, class: &str) -> String {
        format!("{}/snapshot", self.occupancy(camera_id, class))
    }

    fn snapshot(&self, camera_id: &str) -> String {
        format!("{}/{}/snapshot", self.prefix, camera_id)
    }

    fn discovery(&self, component: &str, object_id: &str) -> String {
        format!(
            "{}/{}/{}/config",
            self.discovery_prefix, component, object_id
        )
    }
}

/// The `device` block shared by every entity of one camera, so Home Assistant
/// groups them under a single device card.
fn device_block(camera_id: &str) -> serde_json::Value {
    serde_json::json!({
        "identifiers": [format!("camon_{}", slugify(camera_id))],
        "name": format!("Camon {camera_id}"),
        "manufacturer": "camon",
        "sw_version": env!("CAMON_VERSION"),
    })
}

/// The retained discovery payloads for one camera: the snapshot camera, the
/// motion sensor, and per configured class an occupancy sensor plus the camera
/// showing that class's last sighting.
fn discovery_payloads(
    topics: &Topics,
    camera_id: &str,
    classes: &[String],
) -> Vec<(String, serde_json::Value)> {
    let slug = slugify(camera_id);
    let device = device_block(camera_id);
    let availability = topics.availability();

    let mut out = vec![
        (
            topics.discovery("camera", &format!("camon_{slug}")),
            serde_json::json!({
                "name": "Snapshot",
                "unique_id": format!("camon_{slug}_snapshot"),
                "topic": topics.snapshot(camera_id),
                "availability_topic": availability,
                "device": device,
            }),
        ),
        (
            topics.discovery("binary_sensor", &format!("camon_{slug}_motion")),
            serde_json::json!({
                "name": "Motion",
                "unique_id": format!("camon_{slug}_motion"),
                "state_topic": topics.motion(camera_id),
                "device_class": "motion",
                "availability_topic": availability,
                "device": device,
            }),
        ),
    ];

    for class in classes {
        let class_slug = slugify(class);
        out.push((
            topics.discovery(
                "binary_sensor",
                &format!("camon_{slug}_occupancy_{class_slug}"),
            ),
            serde_json::json!({
                "name": format!("{} occupancy", capitalize(class)),
                "unique_id": format!("camon_{slug}_occupancy_{class_slug}"),
                "state_topic": topics.occupancy(camera_id, class),
                "device_class": "occupancy",
                "availability_topic": availability,
                "device": device,
            }),
        ));
        // The evidence tile for that sensor: retained, so it keeps showing the
        // last sighting long after the occupancy hold-off expires.
        out.push((
            topics.discovery("camera", &format!("camon_{slug}_occupancy_{class_slug}")),
            serde_json::json!({
                "name": format!("{} snapshot", capitalize(class)),
                "unique_id": format!("camon_{slug}_occupancy_{class_slug}_snapshot"),
                "topic": topics.occupancy_snapshot(camera_id, class),
                "availability_topic": availability,
                "device": device,
            }),
        ));
    }

    out
}

/// Every retained topic one camera's entities occupy. Read back out of the
/// discovery payloads rather than rebuilt alongside them, so an entity kind
/// cannot be announced without also being clearable.
fn retained_topics(topics: &Topics, camera_id: &str, classes: &[String]) -> Vec<String> {
    discovery_payloads(topics, camera_id, classes)
        .into_iter()
        .flat_map(|(discovery, payload)| {
            let state = payload["state_topic"]
                .as_str()
                .or_else(|| payload["topic"].as_str())
                .map(str::to_string);
            std::iter::once(discovery).chain(state)
        })
        .collect()
}

/// Format of [`EntityRecord`]. A record that does not say exactly this is not
/// camon's to act on — no deleting on the strength of a document this build
/// does not fully understand.
const ENTITY_RECORD_VERSION: u32 = 1;

/// The broker an entity set belongs to, as the record names it.
fn broker_id(config: &MqttConfig) -> String {
    format!("{}:{}", config.host, config.port)
}

/// The entity set camon announced to a broker, remembered across restarts so the next start can
/// tell Home Assistant to forget what the config dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EntityRecord {
    version: u32,
    /// `host:port` of the broker this set was announced to.
    broker: String,
    topic_prefix: String,
    discovery_prefix: String,
    cameras: Vec<String>,
    classes: Vec<String>,
    /// Clears queued but not known to have left the process. They ride the
    /// next run's burst too, until a clean disconnect proves the socket took
    /// them: re-clearing is a no-op, forgetting is a permanent ghost entity.
    pending_clears: Vec<String>,
}

impl EntityRecord {
    /// The set this run announces, given what the last one announced.
    fn current(previous: Option<&Self>, topics: &Topics, ctx: &BridgeContext) -> Self {
        let classes = match previous {
            Some(previous) if ctx.classes.is_empty() => previous.classes.clone(),
            _ => ctx.classes.clone(),
        };
        Self {
            version: ENTITY_RECORD_VERSION,
            broker: broker_id(&ctx.config),
            topic_prefix: topics.prefix.clone(),
            discovery_prefix: topics.discovery_prefix.clone(),
            cameras: ctx.camera_ids.clone(),
            classes,
            pending_clears: Vec::new(),
        }
    }

    fn topics(&self) -> Topics {
        Topics::from_prefixes(&self.topic_prefix, &self.discovery_prefix)
    }

    fn retained_topics(&self) -> HashSet<String> {
        let topics = self.topics();
        self.cameras
            .iter()
            .flat_map(|camera_id| retained_topics(&topics, camera_id, &self.classes))
            .collect()
    }
}

/// Retained topics the previous record holds that the current set does not explain, plus
/// whatever it left owed.
fn orphaned_topics(previous: &EntityRecord, current: &EntityRecord) -> Vec<String> {
    let live = current.retained_topics();
    let mut orphans: Vec<String> = previous
        .retained_topics()
        .into_iter()
        .chain(previous.pending_clears.iter().cloned())
        .filter(|topic| !live.contains(topic))
        .collect();
    // The availability topic is shared by every entity, so it is only ever
    // orphaned by the prefix itself moving.
    if previous.topic_prefix != current.topic_prefix {
        orphans.push(previous.topics().availability());
    }
    orphans.sort();
    orphans.dedup();
    orphans
}

fn load_record(path: &Path) -> Option<EntityRecord> {
    let data = std::fs::read(path).ok()?;
    let record: EntityRecord = match serde_json::from_slice(&data) {
        Ok(record) => record,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e,
                "mqtt entity record unreadable, clearing no entities this start");
            return None;
        }
    };
    if record.version != ENTITY_RECORD_VERSION {
        tracing::warn!(path = %path.display(), version = record.version,
            "mqtt entity record is a format this build does not know, clearing no entities");
        return None;
    }
    Some(record)
}

/// Persist the record: stage, fsync, rename, fsync the directory. A torn
/// record would read as no record and cost a cleanup; the rename makes even
/// that unreachable.
fn save_record(path: &Path, record: &EntityRecord) -> std::io::Result<()> {
    let dir = parent_dir(path).unwrap_or(Path::new("."));
    create_dir_all_synced(dir)?;

    let json = serde_json::to_vec_pretty(record).map_err(std::io::Error::other)?;
    let tmp = tmp_path(path);
    if let Err(e) = write_synced(&tmp, &json).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    sync_dir(dir)
}

/// The bridge's memory of what Home Assistant has been told about.
struct EntityMemory {
    /// `None` disables the memory entirely: nothing is read, cleared or
    /// written.
    path: Option<PathBuf>,
    /// The set this run announces. The burst is built from it rather than from
    /// the config directly, so what is recorded is exactly what was published.
    announced: EntityRecord,
    /// Retained topics of entities the current set does not explain. Published
    /// empty at the head of every reconnect burst for the life of the process.
    orphans: Vec<String>,
    /// The record believed to be on disk, so a write is skipped when it would
    /// change nothing and retried when it failed.
    on_disk: Option<EntityRecord>,
    /// Whether the clears sit in the request queue of the session up *now*.
    clears_queued: bool,
}

impl EntityMemory {
    fn load(topics: &Topics, ctx: &BridgeContext) -> Self {
        let Some(path) = ctx.entities_path.clone() else {
            return Self {
                path: None,
                announced: EntityRecord::current(None, topics, ctx),
                orphans: Vec::new(),
                on_disk: None,
                clears_queued: false,
            };
        };
        let previous = load_record(&path);
        // A record is authority over one broker's entities only; acting on
        // another broker's record here would delete entities this broker was
        // never told about.
        let broker = broker_id(&ctx.config);
        let authority = previous.as_ref().filter(|previous| {
            if previous.broker == broker {
                return true;
            }
            tracing::info!(
                recorded = %previous.broker,
                broker = %broker,
                "mqtt entity record was written for another broker, clearing no entities"
            );
            false
        });
        let announced = EntityRecord::current(authority, topics, ctx);
        let orphans = authority
            .map(|previous| orphaned_topics(previous, &announced))
            .unwrap_or_default();
        if !orphans.is_empty() {
            tracing::info!(
                topics = orphans.len(),
                "clearing retained topics of entities this config no longer describes"
            );
        }
        Self {
            path: Some(path),
            announced,
            orphans,
            on_disk: previous,
            clears_queued: false,
        }
    }

    /// Record the announced set, now that a burst carrying the clears has been
    /// taken by the request queue. The clears stay recorded as owed: queued is
    /// not written, and a process that dies in between must clear them again.
    fn note_burst_accepted(&mut self) {
        self.clears_queued = true;
        let owed = self.orphans.clone();
        self.write(owed);
    }

    /// A session ended, or a new one began: whatever the old one's queue held is unwritten and
    /// unreachable, so the clears are owed again.
    fn note_session_lost(&mut self) {
        self.clears_queued = false;
    }

    /// Drop the owed clears, now that a `Disconnect` has gone out: requests are written in
    /// queue order, so a disconnect on the wire puts every publish queued before it on the wire
    /// too.
    fn note_clears_flushed(&mut self) {
        if self.clears_queued {
            self.write(Vec::new());
        }
    }

    fn write(&mut self, pending_clears: Vec<String>) {
        let Some(path) = self.path.clone() else {
            return;
        };
        let mut record = self.announced.clone();
        record.pending_clears = pending_clears;
        if self.on_disk.as_ref() == Some(&record) {
            return;
        }
        match save_record(&path, &record) {
            // Only on success: an unwritten record must stay owed so the next
            // attempt tries again.
            Ok(()) => self.on_disk = Some(record),
            Err(e) => tracing::warn!(path = %path.display(), error = %e,
                "failed to record the mqtt entity set; a later removal cannot be cleaned up"),
        }
    }
}

/// What the eventloop task tells the bridge about the broker connection. Every
/// other rumqttc event is the eventloop's own business and never gets here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkEvent {
    /// A `ConnAck` arrived: the session is up. The bridge's only cue to
    /// republish, so these must reach it whole — neither dropped nor coalesced.
    Connected,
    /// The connection failed or dropped. The eventloop paces its own retry.
    Disconnected,
    /// The shutdown `Disconnect` reached the socket, which is also the proof
    /// that everything queued before it did — see [`shutdown_bridge`].
    DisconnectSent,
}

/// The bridge's half of the task that owns the rumqttc event loop; `poll()` is not
/// cancellation-safe, so it lives in a plain loop on its own task and the bridge learns about
/// the connection through [`LinkEvent`]s.
struct Eventloop {
    events: tokio::sync::mpsc::Receiver<LinkEvent>,
    /// Cuts the retry delay short at shutdown. Nothing else can hurry the task
    /// along — a `poll()` in progress is not interruptible.
    stop: Arc<tokio::sync::Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl Eventloop {
    /// Move `eventloop` onto its own task and start polling it.
    fn spawn(eventloop: rumqttc::EventLoop) -> Self {
        let (tx, events) = tokio::sync::mpsc::channel(LINK_EVENT_CAPACITY);
        let stop = Arc::new(tokio::sync::Notify::new());
        let task = tokio::spawn(run_eventloop(eventloop, tx, Arc::clone(&stop)));
        Self { events, stop, task }
    }

    /// End the task: ask, wait briefly, abort. Only called once the shutdown
    /// flush is over, so an aborted `poll()` costs nothing that matters.
    async fn stop(self) {
        let Self {
            events,
            stop,
            mut task,
        } = self;
        // Dropped first: a task parked on `send` has to be woken by the send
        // failing rather than by a notification it is not listening for.
        drop(events);
        stop.notify_one();
        if tokio::time::timeout(EVENTLOOP_STOP_JOIN, &mut task)
            .await
            .is_err()
        {
            task.abort();
        }
    }
}

/// Poll the event loop forever, reporting connection edges to the bridge. The
/// only await raced against anything is the retry sleep, which is safe to
/// drop.
async fn run_eventloop(
    mut eventloop: rumqttc::EventLoop,
    tx: tokio::sync::mpsc::Sender<LinkEvent>,
    stop: Arc<tokio::sync::Notify>,
) {
    loop {
        let event = match eventloop.poll().await {
            Ok(Event::Incoming(Incoming::ConnAck(_))) => LinkEvent::Connected,
            Ok(Event::Outgoing(rumqttc::Outgoing::Disconnect)) => LinkEvent::DisconnectSent,
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!(error = %e, "mqtt connection error, retrying");
                LinkEvent::Disconnected
            }
        };
        // The bridge is gone: nothing left to poll for.
        if tx.send(event).await.is_err() {
            return;
        }
        if event == LinkEvent::Disconnected {
            tokio::select! {
                _ = tokio::time::sleep(RECONNECT_DELAY) => {}
                _ = stop.notified() => return,
            }
        }
    }
}

/// What the bridge believes about the broker connection.
#[derive(Default)]
struct Link {
    connected: bool,
    republish_pending: bool,
    /// What the queue may still be asked to carry in images this tick.
    images: ImageBudget,
}

impl Link {
    /// Record what became of a publish. A rejected one is never retried on its own: the tick
    /// re-asserts every entity from the live `SensorState`. Without this a single dropped OFF
    /// would leave its retained ON standing for good.
    fn note(&mut self, published: Published) {
        if published == Published::QueueFull {
            self.republish_pending = true;
        }
    }
}

/// Pure sensor bookkeeping. No I/O and no clock of its own — `now` is always
/// injected — so the hold-off and cadence rules are exercised directly in
/// tests.
struct SensorState {
    snapshot_interval: Duration,
    occupancy_hold: Duration,
    motion_active: HashSet<String>,
    /// When each camera's next snapshot falls due. Absent means due now.
    next_snapshot: HashMap<String, Instant>,
    /// Cameras whose last decode produced no frame, so that a failure is
    /// reported when it starts rather than once per retry.
    snapshot_failing: HashSet<String>,
    /// How many motion runs each camera has opened. A decode outcome is only
    /// folded back into the cadence when it carries the run that is open now;
    /// see [`SnapshotTask`].
    snapshot_run: HashMap<String, u64>,
    /// `(camera_id, class)` -> last sighting. Presence means the sensor is ON.
    occupancy: HashMap<(String, String), Instant>,
}

impl SensorState {
    /// `snapshot_interval` is clamped to [`MAX_SNAPSHOT_INTERVAL`] because
    /// everything downstream is `now + interval`: `u64::MAX` seconds would
    /// panic on the first tick with motion open.
    fn new(snapshot_interval: Duration, occupancy_hold: Duration) -> Self {
        let snapshot_interval = if snapshot_interval > MAX_SNAPSHOT_INTERVAL {
            tracing::warn!(
                configured = ?snapshot_interval,
                clamped = ?MAX_SNAPSHOT_INTERVAL,
                "mqtt snapshot interval is longer than any motion run, clamping it"
            );
            MAX_SNAPSHOT_INTERVAL
        } else {
            snapshot_interval
        };
        Self {
            snapshot_interval,
            occupancy_hold,
            motion_active: HashSet::new(),
            next_snapshot: HashMap::new(),
            snapshot_failing: HashSet::new(),
            snapshot_run: HashMap::new(),
            occupancy: HashMap::new(),
        }
    }

    /// Open a motion run. `false` when it was already open (a duplicate start, which must not
    /// restart the snapshot cadence). A fresh run is a new generation: outcomes of decodes
    /// started for the previous run must not land on this one.
    fn motion_start(&mut self, camera_id: &str) -> bool {
        let opened = self.motion_active.insert(camera_id.to_string());
        if opened {
            let run = self.snapshot_run.entry(camera_id.to_string()).or_insert(0);
            *run = run.wrapping_add(1);
        }
        opened
    }

    /// The run a decode started now would belong to.
    fn snapshot_run(&self, camera_id: &str) -> u64 {
        self.snapshot_run.get(camera_id).copied().unwrap_or(0)
    }

    /// Close a motion run. `false` when nothing was open.
    fn motion_end(&mut self, camera_id: &str) -> bool {
        self.next_snapshot.remove(camera_id);
        self.motion_active.remove(camera_id)
    }

    fn has_motion(&self, camera_id: &str) -> bool {
        self.motion_active.contains(camera_id)
    }

    /// Record a sighting of `class`, turning its sensor ON. `true` when this is
    /// a fresh OFF -> ON transition.
    fn record_sighting(&mut self, camera_id: &str, class: &str, now: Instant) -> bool {
        self.occupancy
            .insert((camera_id.to_string(), class.to_string()), now)
            .is_none()
    }

    fn is_occupied(&self, camera_id: &str, class: &str) -> bool {
        self.occupancy
            .contains_key(&(camera_id.to_string(), class.to_string()))
    }

    /// Note that `camera_id` has just been snapshotted, restarting its cadence.
    fn mark_snapshot(&mut self, camera_id: &str, now: Instant) {
        self.next_snapshot
            .insert(camera_id.to_string(), now + self.snapshot_interval);
    }

    /// Note that a decode produced no frame.
    fn note_snapshot_failed(&mut self, camera_id: &str, now: Instant) -> bool {
        if self.motion_active.contains(camera_id) {
            let retry = now + SNAPSHOT_RETRY_DELAY.min(self.snapshot_interval);
            self.next_snapshot
                .entry(camera_id.to_string())
                .and_modify(|next| *next = (*next).min(retry))
                .or_insert(retry);
        }
        self.snapshot_failing.insert(camera_id.to_string())
    }

    /// Note that a decode produced a frame. Reports `true` when that ends a run
    /// of failures, so the recovery is as audible as the failure was.
    fn note_snapshot_decoded(&mut self, camera_id: &str) -> bool {
        self.snapshot_failing.remove(camera_id)
    }

    /// Turn OFF every occupancy sensor whose hold-off has elapsed, returning
    /// the pairs that just expired.
    fn expire_occupancy(&mut self, now: Instant) -> Vec<(String, String)> {
        let hold = self.occupancy_hold;
        let expired: Vec<(String, String)> = self
            .occupancy
            .iter()
            .filter(|(_, &seen)| now.saturating_duration_since(seen) >= hold)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &expired {
            self.occupancy.remove(key);
        }
        expired
    }

    /// Cameras whose next motion snapshot is due, putting each on the cadence again. The
    /// attempt is what is stamped here; a decode that produces nothing takes the stamp back
    /// with [`note_snapshot_failed`](Self::note_snapshot_failed).
    fn due_snapshots(&mut self, now: Instant) -> Vec<String> {
        let due: Vec<String> = self
            .motion_active
            .iter()
            .filter(|camera_id| match self.next_snapshot.get(*camera_id) {
                Some(&next) => now >= next,
                None => true,
            })
            .cloned()
            .collect();
        for camera_id in &due {
            self.next_snapshot
                .insert(camera_id.clone(), now + self.snapshot_interval);
        }
        due
    }
}

/// Everything the bridge task needs, assembled in `main` while the config and
/// camera handles are still in scope.
pub struct BridgeContext {
    pub config: MqttConfig,
    /// The same hot buffers the HTTP API serves from; snapshots are decoded
    /// from their newest segment.
    pub buffers: Arc<HashMap<String, Arc<RwLock<HotBuffer>>>>,
    pub camera_ids: Vec<String>,
    /// Object-detection classes to expose occupancy sensors for. Empty when
    /// object detection is off.
    pub classes: Vec<String>,
    /// Where the announced entity set is remembered between runs. `None` keeps
    /// no memory: nothing is read, cleared or written.
    pub entities_path: Option<PathBuf>,
    pub shutdown: Arc<AtomicBool>,
}

/// Whether the bridge's loop has nothing left to serve.
fn bridge_is_done(producers_gone: bool, shutdown: &AtomicBool) -> bool {
    producers_gone && shutdown.load(Ordering::Relaxed)
}

/// The bridge itself. Spawned under supervision by [`crate::app`] and joined
/// (with a timeout) during shutdown, so the retained `offline` marker gets
/// published.
pub async fn run_bridge(ctx: BridgeContext, rx: tokio::sync::mpsc::Receiver<MqttEvent>) {
    run_bridge_with(ctx, rx, Eventloop::spawn).await
}

/// [`run_bridge`], with the poller's construction handed in — a seam for the
/// test that kills the poller to prove the bridge notices.
async fn run_bridge_with<F>(
    ctx: BridgeContext,
    mut rx: tokio::sync::mpsc::Receiver<MqttEvent>,
    poller: F,
) where
    F: FnOnce(rumqttc::EventLoop) -> Eventloop,
{
    let topics = Topics::new(&ctx.config);
    // Loaded before the client: its queue has to be sized for the clears too.
    let mut memory = EntityMemory::load(&topics, &ctx);
    let client_id = client_id(&ctx);
    let mut options = MqttOptions::new(&client_id, &ctx.config.host, ctx.config.port);
    options.set_keep_alive(Duration::from_secs(30));
    options.set_max_packet_size(10 * 1024, MAX_OUTGOING_PACKET_BYTES);
    // Set before connect: the broker publishes this on our behalf if camon dies
    // without a clean disconnect, so HA marks the entities unavailable.
    options.set_last_will(LastWill::new(
        topics.availability(),
        "offline",
        QoS::AtLeastOnce,
        true,
    ));
    if let (Some(username), Some(password)) = (&ctx.config.username, &ctx.config.password) {
        options.set_credentials(username, password);
    }

    let (client, raw_eventloop) = AsyncClient::new(
        options,
        request_queue_capacity(
            memory.announced.cameras.len(),
            memory.announced.classes.len(),
            memory.orphans.len(),
        ),
    );
    let mut eventloop = poller(raw_eventloop);
    let mut state = SensorState::new(
        Duration::from_secs(ctx.config.snapshot_interval_secs),
        Duration::from_secs(ctx.config.occupancy_hold_secs),
    );
    let mut snapshot_tasks: HashMap<String, SnapshotTask> = HashMap::new();
    let mut link = Link::default();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    tracing::info!(
        host = %ctx.config.host,
        port = ctx.config.port,
        client_id = %client_id,
        cameras = ctx.camera_ids.len(),
        classes = ctx.classes.len(),
        "mqtt bridge started"
    );

    // Once every producer's sender has dropped, `recv()` returns `None`
    // immediately and forever; the branches are disabled so they cannot spin.
    let mut producers_gone = false;
    let mut eventloop_gone = false;

    loop {
        tokio::select! {
            event = eventloop.events.recv(), if !eventloop_gone => match event {
                Some(LinkEvent::Connected) => {
                    tracing::info!(host = %ctx.config.host, "mqtt connected, publishing discovery");
                    link.connected = true;
                    // A new session starts owing the clears again whatever the
                    // last one queued; the burst below is what re-queues them.
                    memory.note_session_lost();
                    link.republish_pending = republish(&client, &topics, &state, &mut memory);
                }
                Some(LinkEvent::Disconnected) => {
                    link.connected = false;
                    memory.note_session_lost();
                }
                // Nothing asks for a disconnect before shutdown does, and by
                // then this loop is over.
                Some(LinkEvent::DisconnectSent) => {}
                // The eventloop task only ends on purpose during shutdown, so reaching here
                // with the stop flag down means it died.
                None => {
                    eventloop_gone = true;
                    link.connected = false;
                    // Nothing queued will ever be written — clears included.
                    memory.note_session_lost();
                    if !ctx.shutdown.load(Ordering::Relaxed) {
                        tracing::error!(
                            "the mqtt event loop stopped polling while camon was running; \
                             nothing can be published or received until camon restarts"
                        );
                        break;
                    }
                }
            },
            event = rx.recv(), if !producers_gone => match event {
                Some(event) => handle_event(
                    event,
                    &client,
                    &topics,
                    &mut state,
                    &ctx,
                    &mut snapshot_tasks,
                    &mut link,
                    ctx.shutdown.load(Ordering::Relaxed),
                ),
                // Every producer is gone: keep serving ticks until the flag
                // confirms shutdown rather than exiting here with the
                // availability marker still reading `online`.
                None => {
                    producers_gone = true;
                    if bridge_is_done(producers_gone, &ctx.shutdown) {
                        break;
                    }
                }
            },
            _ = tick.tick() => {
                if bridge_is_done(producers_gone, &ctx.shutdown) {
                    break;
                }
                on_tick(
                    &client,
                    &topics,
                    &mut state,
                    &ctx,
                    &mut snapshot_tasks,
                    &mut link,
                    &mut memory,
                    ctx.shutdown.load(Ordering::Relaxed),
                );
            }
        }
    }

    shutdown_bridge(
        client,
        &topics,
        eventloop,
        snapshot_tasks,
        &mut memory,
        link.republish_pending,
    )
    .await;
}

/// `stopping` is the shutdown flag: state transitions still go out, but no new snapshot is
/// started — each forks an ffmpeg, and a drain is the one time the process is trying to get
/// every child it already has to exit.
#[allow(clippy::too_many_arguments)]
fn handle_event(
    event: MqttEvent,
    client: &AsyncClient,
    topics: &Topics,
    state: &mut SensorState,
    ctx: &BridgeContext,
    snapshot_tasks: &mut HashMap<String, SnapshotTask>,
    link: &mut Link,
    stopping: bool,
) {
    match event {
        MqttEvent::MotionStart { camera_id } => {
            if !state.motion_start(&camera_id) {
                return;
            }
            tracing::debug!(camera = %camera_id, "mqtt motion ON");
            link.note(publish_state(client, &topics.motion(&camera_id), "ON"));
            // The frame that opened the run is the interesting one: take it
            // now rather than waiting for the tick.
            if !stopping {
                state.mark_snapshot(&camera_id, Instant::now());
                let run = state.snapshot_run(&camera_id);
                spawn_snapshot(client, topics, ctx, &camera_id, run, snapshot_tasks, link);
            }
        }
        MqttEvent::MotionEnd { camera_id } => {
            if !state.motion_end(&camera_id) {
                return;
            }
            tracing::debug!(camera = %camera_id, "mqtt motion OFF");
            link.note(publish_state(client, &topics.motion(&camera_id), "OFF"));
            // One last frame so the tile shows the end of the event.
            if !stopping {
                let run = state.snapshot_run(&camera_id);
                spawn_snapshot(client, topics, ctx, &camera_id, run, snapshot_tasks, link);
            }
        }
        MqttEvent::Detections {
            camera_id,
            sightings,
        } => {
            let now = Instant::now();
            for sighting in sightings {
                let class = sighting.class;
                // An unconfigured class has no entity to publish to.
                if !ctx.classes.contains(&class) {
                    continue;
                }
                if state.record_sighting(&camera_id, &class, now) {
                    tracing::debug!(camera = %camera_id, class = %class, "mqtt occupancy ON");
                }
                link.note(publish_state(
                    client,
                    &topics.occupancy(&camera_id, &class),
                    "ON",
                ));
                // QoS 0 because the next verdict supersedes this one; retained
                // so the tile keeps the sighting; skipped while disconnected
                // for the same reason snapshots are.
                if let (true, Some(jpeg)) = (link.connected, sighting.frame_jpeg) {
                    if !link.images.take(jpeg.len()) {
                        tracing::debug!(camera = %camera_id, class = %class,
                            bytes = jpeg.len(), "image budget spent, dropping the sighting crop");
                        continue;
                    }
                    let topic = topics.occupancy_snapshot(&camera_id, &class);
                    // A rejected crop is gone, but the rejection still says
                    // the queue is full, so the sensor states get re-asserted.
                    link.note(publish_retained(client, &topic, QoS::AtMostOnce, jpeg));
                }
            }
        }
    }
}

/// `stopping` means here what it means in [`handle_event`]: publishing is
/// still welcome, forking new snapshot decodes is not.
#[allow(clippy::too_many_arguments)]
fn on_tick(
    client: &AsyncClient,
    topics: &Topics,
    state: &mut SensorState,
    ctx: &BridgeContext,
    snapshot_tasks: &mut HashMap<String, SnapshotTask>,
    link: &mut Link,
    memory: &mut EntityMemory,
    stopping: bool,
) {
    let now = Instant::now();
    link.images.refill();

    // Rebuilt from the live state on every attempt, never replayed from the
    // failed one: a retry must not assert a value that has since changed.
    if link.connected && link.republish_pending {
        link.republish_pending = republish(client, topics, state, memory);
    }

    // Before the cadence is consulted, so a decode that has just ended is
    // accounted for in the very tick that could act on it.
    retire_snapshots(snapshot_tasks, state, now);

    if !stopping {
        for camera_id in state.due_snapshots(now) {
            let run = state.snapshot_run(&camera_id);
            spawn_snapshot(client, topics, ctx, &camera_id, run, snapshot_tasks, link);
        }
    }

    for (camera_id, class) in state.expire_occupancy(now) {
        tracing::debug!(camera = %camera_id, class = %class, "mqtt occupancy OFF (hold elapsed)");
        link.note(publish_state(
            client,
            &topics.occupancy(&camera_id, &class),
            "OFF",
        ));
    }
}

/// Retire every decode that has ended, folding its outcome into the cadence.
/// The first failure of a run is said out loud: a camera whose decodes all
/// fail is a tile that quietly stops moving.
fn retire_snapshots(
    snapshot_tasks: &mut HashMap<String, SnapshotTask>,
    state: &mut SensorState,
    now: Instant,
) {
    let ended: Vec<(String, bool, u64)> = snapshot_tasks
        .iter()
        .filter(|(_, task)| task.handle.is_finished())
        .map(|(camera_id, task)| {
            (
                camera_id.clone(),
                task.decoded.load(Ordering::Relaxed),
                task.run,
            )
        })
        .collect();
    for (camera_id, decoded, run) in ended {
        snapshot_tasks.remove(&camera_id);
        // The outcome belongs to a run that has since closed, not to the one
        // open now.
        if run != state.snapshot_run(&camera_id) {
            tracing::debug!(camera = %camera_id, "a decode outlived its motion run, dropping its outcome");
            continue;
        }
        if decoded {
            if state.note_snapshot_decoded(&camera_id) {
                tracing::info!(camera = %camera_id, "snapshots are decoding again");
            }
        } else if state.note_snapshot_failed(&camera_id, now) {
            tracing::warn!(camera = %camera_id,
                "no snapshot could be decoded for this camera; its Home Assistant tile is not \
                 being updated");
        }
    }
}

/// What became of one `try_publish`. rumqttc reports a full queue and an
/// unpublishable topic as the same error, but they need opposite handling: the
/// first is transient and retried, the second never becomes publishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Published {
    Yes,
    QueueFull,
    /// A wildcard in a camera id or class name — a backstop against a
    /// livelock; `Config::validate` rejects both when the bridge is enabled.
    ImpossibleTopic,
}

/// Publish one retained message, so Home Assistant sees the current value the
/// moment it subscribes.
fn publish_retained(client: &AsyncClient, topic: &str, qos: QoS, payload: Vec<u8>) -> Published {
    if !rumqttc::valid_topic(topic) {
        tracing::error!(topic = %topic, "topic holds an MQTT wildcard and can never be published");
        return Published::ImpossibleTopic;
    }
    match client.try_publish(topic, qos, true, payload) {
        Ok(()) => Published::Yes,
        Err(e) => {
            tracing::warn!(topic = %topic, error = %e, "mqtt publish rejected");
            Published::QueueFull
        }
    }
}

fn publish_state(client: &AsyncClient, topic: &str, payload: &str) -> Published {
    publish_retained(client, topic, QoS::AtLeastOnce, payload.as_bytes().to_vec())
}

/// Queue everything the connection owes the broker and, if all of it was taken,
/// record the entity set it announced. Reports whether the burst is still owed.
fn republish(
    client: &AsyncClient,
    topics: &Topics,
    state: &SensorState,
    memory: &mut EntityMemory,
) -> bool {
    let burst = reconnect_burst(topics, state, &memory.announced, &memory.orphans);
    let accepted = publish_burst(client, burst);
    if accepted {
        memory.note_burst_accepted();
    }
    !accepted
}

/// Everything a (re)connect owes the broker, in publish order: orphan clears, discovery
/// payloads, every announced entity's current state, then the availability marker.
fn reconnect_burst(
    topics: &Topics,
    state: &SensorState,
    announced: &EntityRecord,
    orphans: &[String],
) -> Vec<(String, Vec<u8>)> {
    let mut burst = Vec::new();
    // First: Home Assistant must forget these before the device goes
    // available again, which would otherwise resurrect them. An empty
    // retained payload deletes both discovery document and held state.
    for topic in orphans {
        burst.push((topic.clone(), Vec::new()));
    }
    for camera_id in &announced.cameras {
        for (topic, payload) in discovery_payloads(topics, camera_id, &announced.classes) {
            match serde_json::to_vec(&payload) {
                Ok(bytes) => burst.push((topic, bytes)),
                Err(e) => {
                    tracing::error!(topic = %topic, error = %e, "discovery payload not serializable")
                }
            }
        }
    }
    for (topic, payload) in state_payloads(topics, state, &announced.cameras, &announced.classes) {
        burst.push((topic, payload.as_bytes().to_vec()));
    }
    // Last, so the retained values are current before HA hears `online`.
    burst.push((topics.availability(), b"online".to_vec()));
    burst
}

/// Hand the burst to the request queue, reporting whether all of it fit.
fn publish_burst(client: &AsyncClient, burst: Vec<(String, Vec<u8>)>) -> bool {
    let total = burst.len();
    let mut published = 0;
    for (topic, payload) in burst {
        match publish_retained(client, &topic, QoS::AtLeastOnce, payload) {
            // An impossible topic counts as done: it will never be
            // publishable, and stopping on it would hold up the burst forever.
            Published::Yes | Published::ImpossibleTopic => published += 1,
            Published::QueueFull => break,
        }
    }
    if published < total {
        tracing::warn!(
            published,
            total,
            "mqtt request queue full, retrying the reconnect burst"
        );
    }
    published == total
}

fn on_off(on: bool) -> &'static str {
    if on {
        "ON"
    } else {
        "OFF"
    }
}

/// The retained payload every configured sensor should currently be holding.
/// Enumerated from the config, not from the sensors that are ON: a stale
/// retained ON is only corrected by saying OFF out loud.
fn state_payloads(
    topics: &Topics,
    state: &SensorState,
    camera_ids: &[String],
    classes: &[String],
) -> Vec<(String, &'static str)> {
    let mut out = Vec::with_capacity(camera_ids.len() * (1 + classes.len()));
    for camera_id in camera_ids {
        out.push((
            topics.motion(camera_id),
            on_off(state.has_motion(camera_id)),
        ));
        for class in classes {
            out.push((
                topics.occupancy(camera_id, class),
                on_off(state.is_occupied(camera_id, class)),
            ));
        }
    }
    out
}

/// One camera's detached snapshot decode. The task is detached so a slow
/// ffmpeg cannot delay the bridge loop; a flag the task sets and the tick
/// reads carries the outcome back.
struct SnapshotTask {
    handle: tokio::task::JoinHandle<()>,
    /// Set by the task when the decode produced a frame. A task that panicked
    /// or was aborted leaves it down, which reads as a failure and is one.
    decoded: Arc<AtomicBool>,
    /// The motion run this decode was started for.
    run: u64,
}

/// Decode and publish one snapshot, unless this camera already has a decode in
/// flight. Detached so a slow ffmpeg never delays the event loop; the handle is
/// kept as the in-flight marker and as the carrier of the decode's outcome.
fn spawn_snapshot(
    client: &AsyncClient,
    topics: &Topics,
    ctx: &BridgeContext,
    camera_id: &str,
    run: u64,
    snapshot_tasks: &mut HashMap<String, SnapshotTask>,
    link: &Link,
) {
    // Nothing drains the request queue while the broker is unreachable, so a
    // snapshot queued now is a few hundred KiB parked in memory for the length
    // of the outage, to be delivered long after anything cared about it.
    if !link.connected {
        tracing::debug!(camera = %camera_id, "mqtt disconnected, skipping snapshot");
        return;
    }
    if let Some(task) = snapshot_tasks.get(camera_id) {
        if !task.handle.is_finished() {
            tracing::debug!(camera = %camera_id, "snapshot still decoding, skipping this one");
            return;
        }
    }

    // Clone the segment's Arc under the lock and let the guard go immediately:
    // the ingest thread must never wait on a decode.
    let data = match ctx.buffers.get(camera_id) {
        Some(buffer) => {
            let buf = buffer.read_recover();
            let last = buf.last_sequence();
            last.checked_sub(1)
                .and_then(|seq| buf.get_segment_by_sequence(seq))
                .map(|segment| Arc::clone(&segment.data))
        }
        None => None,
    };
    let Some(data) = data else {
        tracing::debug!(camera = %camera_id, "hot buffer empty, no snapshot");
        return;
    };

    let client = client.clone();
    let topic = topics.snapshot(camera_id);
    let camera = camera_id.to_string();
    let decoded = Arc::new(AtomicBool::new(false));
    let produced = Arc::clone(&decoded);
    let budget = link.images.clone();
    let handle = tokio::spawn(async move {
        match snapshot_jpeg(&data).await {
            Some(jpeg) => {
                // A frame exists, which is what the cadence asked for; whether
                // the queue then takes it is the queue's story, told by the
                // budget and by the reconnect burst.
                produced.store(true, Ordering::Relaxed);
                if !budget.take(jpeg.len()) {
                    tracing::debug!(camera = %camera, bytes = jpeg.len(),
                        "image budget spent, dropping the snapshot");
                    return;
                }
                // QoS 0 avoids retrying a soon-replaced frame; retained supplies new subscribers.
                publish_retained(&client, &topic, QoS::AtMostOnce, jpeg);
            }
            None => tracing::debug!(camera = %camera, "snapshot decode produced no frame"),
        }
    });
    snapshot_tasks.insert(
        camera_id.to_string(),
        SnapshotTask {
            handle,
            decoded,
            run,
        },
    );
}

/// Decode the first frame of an MPEG-TS segment and JPEG-encode it.
async fn snapshot_jpeg(segment: &[u8]) -> Option<Vec<u8>> {
    let expected = (SNAPSHOT_WIDTH * SNAPSHOT_HEIGHT * 3) as usize;
    let mut frame = piped_decode(
        snapshot_command(),
        segment,
        expected,
        SNAPSHOT_DECODE_TIMEOUT,
    )
    .await?;
    if frame.len() < expected {
        return None;
    }
    frame.truncate(expected);

    // Fixed-size CPU work that always terminates, so it needs no bound of its
    // own; off the runtime because a 720p encode is milliseconds the event loop
    // should not be spending.
    tokio::task::spawn_blocking(move || encode_jpeg(&frame, SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT))
        .await
        .map_err(|e| tracing::warn!(error = %e, "snapshot encode task failed"))
        .ok()?
}

fn snapshot_command() -> tokio::process::Command {
    let filter = format!(
        "scale={SNAPSHOT_WIDTH}:{SNAPSHOT_HEIGHT}:force_original_aspect_ratio=decrease,\
         pad={SNAPSHOT_WIDTH}:{SNAPSHOT_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    );
    let mut command = tokio::process::Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "quiet",
        "-f",
        "mpegts",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-vf",
        &filter,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]);
    command
}

/// Feed `input` to `command` on stdin and return everything it writes to stdout, killing the
/// child if it has not finished within `timeout`.
async fn piped_decode(
    mut command: tokio::process::Command,
    input: &[u8],
    capacity: usize,
    timeout: Duration,
) -> Option<Vec<u8>> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| tracing::warn!(error = %e, "failed to spawn snapshot ffmpeg"))
        .ok()?;

    let stdin = child.stdin.take().expect("stdin piped");
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut out = Vec::with_capacity(capacity);

    let piped = async {
        // ffmpeg exits after one frame and closes stdin, so the tail of the write failing with
        // EPIPE is the normal case, not an error.
        let write = async move {
            let mut stdin = stdin;
            let _ = stdin.write_all(input).await;
            drop(stdin);
        };
        let (_, read) = tokio::join!(write, stdout.read_to_end(&mut out));
        read
    };
    let piped = tokio::time::timeout(timeout, piped).await;

    match piped {
        Ok(Ok(_)) => Some(out),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "snapshot frame read failed");
            None
        }
        Err(_) => {
            tracing::warn!(timeout = ?timeout, "snapshot decode timed out, killing ffmpeg");
            None
        }
    }
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, SNAPSHOT_JPEG_QUALITY)
        .encode(rgb, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| tracing::warn!(error = %e, "snapshot jpeg encode failed"))
        .ok()?;
    Some(out)
}

/// Publish the retained `offline` marker and, if the queue took that marker, disconnect
/// cleanly, then wait briefly for both packets to actually reach the socket — `try_publish`
/// only queues them, the eventloop task is what writes them.
async fn shutdown_bridge(
    client: AsyncClient,
    topics: &Topics,
    mut eventloop: Eventloop,
    snapshot_tasks: HashMap<String, SnapshotTask>,
    memory: &mut EntityMemory,
    burst_owed: bool,
) {
    abort_snapshots(snapshot_tasks).await;

    // Whatever is already queued was raised before the disconnect was even requested and says
    // nothing about whether it reached the socket.
    while let Ok(event) = eventloop.events.try_recv() {
        if event != LinkEvent::DisconnectSent {
            memory.note_session_lost();
        }
    }

    // A clean DISCONNECT suppresses the LWT, so request it only after `offline` is queued.
    // If the marker is rejected, an unclean close lets the broker publish the LWT instead.
    if publish_state(&client, &topics.availability(), "offline") == Published::Yes {
        if let Err(e) = client.try_disconnect() {
            tracing::debug!(error = %e, "mqtt disconnect request failed");
        }
    } else {
        tracing::warn!("mqtt offline marker was not queued, leaving the LWT to publish it");
    }

    let flush = tokio::time::timeout(SHUTDOWN_FLUSH, async {
        loop {
            match eventloop.events.recv().await {
                Some(LinkEvent::DisconnectSent) => return true,
                // The socket went down with the disconnect still queued, or
                // the task is gone: either way nothing was written.
                Some(LinkEvent::Disconnected) | None => return false,
                // A connection came up *while this was waiting*, which means
                // the one the clears were queued on is gone and took them with
                // it. The disconnect that follows is this session's alone.
                Some(LinkEvent::Connected) => memory.note_session_lost(),
            }
        }
    });
    let disconnect_written = flush.await == Ok(true);
    if disconnect_written && !burst_owed {
        memory.note_clears_flushed();
    }
    eventloop.stop().await;
    tracing::info!("mqtt bridge stopped");
}

/// Cancel every in-flight snapshot decode.
async fn abort_snapshots(snapshot_tasks: HashMap<String, SnapshotTask>) {
    let handles: Vec<tokio::task::JoinHandle<()>> = snapshot_tasks
        .into_values()
        .map(|task| task.handle)
        .collect();
    for handle in &handles {
        handle.abort();
    }
    let joined = tokio::time::timeout(SNAPSHOT_ABORT_JOIN, async {
        for handle in handles {
            let _ = handle.await;
        }
    });
    if joined.await.is_err() {
        tracing::warn!("snapshot decode did not stop in time, leaving it detached");
    }
}

#[cfg(test)]
mod tests;
