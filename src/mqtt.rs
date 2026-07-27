//! Home Assistant MQTT bridge.
//!
//! One async task per process owns the broker connection and publishes camon's
//! state under a configurable topic prefix, plus the retained
//! `homeassistant/.../config` payloads that make Home Assistant materialize the
//! entities on its own (MQTT discovery). Nothing is ever subscribed to: the
//! bridge is strictly outbound, so a compromised or misbehaving broker cannot
//! drive camon.
//!
//! # Entities
//!
//! Per camera, four kinds of entity:
//!
//! - a **camera** fed by JPEG snapshots;
//! - a **motion** binary sensor, ON for the lifetime of a motion run;
//! - one **occupancy** binary sensor per configured object-detection class, ON
//!   from the moment a verdict names that class until `occupancy_hold_secs`
//!   pass with no new sighting;
//! - one **occupancy snapshot** camera per configured class, holding the crop
//!   the vision model classified — see below.
//!
//! # Per-class sighting snapshots
//!
//! A verdict carries, per named class, the JPEG the model actually looked at:
//! the motion crop of the frame with that class's strongest detection. It is
//! published retained, so the entity is not tied to the occupancy sensor's
//! hold-off — it keeps showing the last sighting of that class indefinitely,
//! long after the sensor went OFF, and is there the moment Home Assistant
//! subscribes. "When did a cat last walk past, and what did it look like" is
//! then one dashboard tile rather than an event search. The bytes are not held
//! in memory, so unlike the sensor states these are not re-asserted on
//! reconnect; a broker that loses its retained set fills the tile in again on
//! the next sighting.
//!
//! # Motion-gated snapshots
//!
//! Snapshots are published *only while motion is open*, never on a free-running
//! timer. This is a deliberate design decision, not a limitation. Each snapshot
//! costs an ffmpeg decode of a whole GOP segment, and the interesting frames are
//! by definition the ones with motion in them; a 24/7 cadence would burn CPU on
//! every camera continuously to produce identical still frames. The camera
//! entity therefore shows "the last thing that moved", which is what a
//! notification or dashboard tile actually wants. A final snapshot is taken as
//! the run closes so the tile does not freeze mid-event.
//!
//! # Never block the loop
//!
//! The event loop must be polled continuously — rumqttc does all of its I/O,
//! including draining the client's request queue, inside `poll()`. Two
//! consequences shape this module:
//!
//! - every publish uses `try_publish`, never the awaiting `publish`. While the
//!   broker is unreachable the request queue backs up, and an awaiting publish
//!   inside the `select!` body would stop the very `poll()` that drains it —
//!   a self-inflicted deadlock. Rejected publishes are recovered by the
//!   reconnect republish below.
//! - snapshot decoding runs in a detached task holding a killable ffmpeg child,
//!   with at most one in flight per camera, and is skipped entirely while
//!   disconnected — the queue is not being drained, so a snapshot pushed into
//!   it is memory parked for the length of the outage.
//!
//! # Reconnect republish
//!
//! On every `ConnAck` the bridge republishes the full discovery set, then an
//! explicit state for every configured entity, then the availability marker.
//! A broker restart loses retained messages, Home Assistant re-reads discovery
//! when *it* restarts, and anything dropped during an outage is made good here.
//! Republishing is idempotent, so doing it unconditionally is simpler and safer
//! than tracking what the broker still holds.
//!
//! The states are enumerated from the config rather than from what happens to
//! be ON, because the broker can be holding a retained ON that no live state
//! corresponds to: camon dying mid-motion-run leaves one behind, the LWT only
//! flips availability, and a fresh process would otherwise never contradict it.
//! "Every configured entity" means every entity *this* config describes — a
//! camera that has since been renamed or removed still has whatever the broker
//! retained for it, and nothing here cleans that up.
//!
//! The burst is queued at its least favourable moment: nothing has drained the
//! request queue for the length of the outage, so it can be rejected in full.
//! It is therefore retried from the tick, rebuilt from the live state each
//! time, until the whole of it is accepted — losing it would leave both a stale
//! retained ON and the LWT's `offline` standing on a healthy connection. For
//! that retry to be able to converge the queue is sized from the config to hold
//! a whole burst; see [`request_queue_capacity`].
//!
//! Order within the burst is discovery, then states, then `online`. MQTT only
//! guarantees ordering per topic, so this is not a promise about what Home
//! Assistant observes in what instant; it is that camon never leaves the
//! *final* retained set inconsistent, and that the availability flip is the
//! last thing queued rather than the first.

use std::collections::{HashMap, HashSet};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use rumqttc::{AsyncClient, Event, Incoming, LastWill, MqttOptions, QoS};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::buffer::HotBuffer;
use crate::config::MqttConfig;
use crate::locks::LockExt;

/// Depth of the analyzer/detector -> bridge event channel. Events are tiny and
/// rare (a motion transition or a verdict), so this is far more headroom than a
/// healthy bridge ever needs; producers `try_send` and drop on full rather than
/// ever stalling the analysis threads.
pub const MQTT_EVENT_CAPACITY: usize = 64;

/// Headroom above one reconnect burst: room for the transitions and snapshots
/// raised while that burst is still being drained.
const REQUEST_QUEUE_HEADROOM: usize = 256;

/// How many publishes one reconnect burst is — per camera two discovery
/// payloads plus two per class, one state plus one per class, then the
/// availability marker. A test ties this to what [`reconnect_burst`] actually
/// produces.
fn burst_len(cameras: usize, classes: usize) -> usize {
    cameras * (3 + 3 * classes) + 1
}
/// Depth of the rumqttc request queue, sized from the config so that one whole
/// reconnect burst always fits in it.
///
/// [`publish_burst`] is all-or-nothing and runs while `poll()` is draining
/// nothing, so a burst longer than the queue could never be published at all:
/// the tick would requeue the same prefix every second forever, and `online` —
/// at the tail — would never go out, leaving Home Assistant with a permanently
/// unavailable device. Fifteen cameras with the default five classes is already
/// 271 publishes, so this is not a hypothetical size.
///
/// Sizing up is only safe because snapshots are never queued while
/// disconnected: the slots this adds can hold nothing bigger than a few bytes
/// of retained state each, not the hundreds of KiB a JPEG would park here for
/// the length of an outage.
fn request_queue_capacity(cameras: usize, classes: usize) -> usize {
    burst_len(cameras, classes) + REQUEST_QUEUE_HEADROOM
}

/// Outgoing packet-size ceiling. rumqttc defaults to 10 KiB, which every
/// snapshot would exceed; a 1280x720 JPEG at quality 90 is a few hundred KiB.
const MAX_OUTGOING_PACKET_BYTES: usize = 4 * 1024 * 1024;

/// Delay before re-polling after a connection error. `poll()` reconnects on its
/// own but does not pace itself, so without this a down broker becomes a busy
/// loop.
const RECONNECT_DELAY: Duration = Duration::from_secs(5);

/// Snapshot output size. Fixed regardless of camera aspect ratio (letterboxed
/// by the decode filter) so Home Assistant's camera tile never jumps around.
const SNAPSHOT_WIDTH: u32 = 1280;
const SNAPSHOT_HEIGHT: u32 = 720;

/// JPEG quality for snapshots, matching the analytics pipeline's.
const SNAPSHOT_JPEG_QUALITY: u8 = 90;

/// How long shutdown waits for the retained `offline` marker and the DISCONNECT
/// to reach the socket.
const SHUTDOWN_FLUSH: Duration = Duration::from_secs(2);

/// How long one snapshot decode may run before its ffmpeg is killed. Generous
/// for a single-GOP decode: the point is that a wedged ffmpeg cannot hold a
/// camera's in-flight slot forever or outlive the process, not to police the
/// timing of healthy decodes.
const SNAPSHOT_DECODE_TIMEOUT: Duration = Duration::from_secs(15);

/// How long shutdown waits for aborted snapshot tasks to actually end. A
/// snapshot task is always parked on an await — the ffmpeg pipe, or the encode
/// it hands to a blocking thread — and abort takes effect there in microseconds,
/// so nothing is expected to reach this bound. It exists so that a task somehow
/// blocking its worker thread cannot hold shutdown open indefinitely.
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
    /// The JPEG the vision model classified — the motion crop of the frame
    /// holding this class's highest-confidence detection, falling back to the
    /// run's full frame. `None` when the job carried no frame at all, in which
    /// case nothing is published to the snapshot topic and the entity keeps
    /// showing the previous sighting.
    pub frame_jpeg: Option<Vec<u8>>,
}

/// Hand an event to the bridge without ever blocking the producer. Both
/// analyzer and detection worker call this from paths where stalling would cost
/// footage, so a full queue (or a bridge that is gone) drops the event with a
/// warning: MQTT state is refreshed on the next transition and on reconnect.
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

/// Lowercase slug with every non-alphanumeric character folded to `_`. Camera
/// ids are free-form, but they end up inside MQTT topics and Home Assistant
/// unique ids, so they are normalized once here. `Config::validate` uses the
/// same function to reject ids that would collide once normalized.
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
        Self {
            prefix: config.topic_prefix.trim_end_matches('/').to_string(),
            discovery_prefix: config.discovery_prefix.trim_end_matches('/').to_string(),
        }
    }

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
        // The evidence tile for that sensor. Its topic is retained, so this
        // entity survives the occupancy hold-off expiring and keeps showing the
        // last sighting of the class until the next one replaces it.
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

/// What the bridge believes about the broker connection.
///
/// `republish_pending` is the whole recovery story for a dropped burst. The
/// request queue is only drained while connected, so a reconnect after a real
/// outage finds it exactly as full as the outage left it and every publish in
/// the burst is rejected — including `online`, which would otherwise leave the
/// LWT's `offline` standing on a healthy connection. The tick retries until the
/// eventloop has drained enough for the burst to fit.
#[derive(Default)]
struct Link {
    connected: bool,
    republish_pending: bool,
}

impl Link {
    /// Record what became of a publish. A rejected one is never retried on its
    /// own: the tick re-asserts every entity from the live `SensorState`, which
    /// repairs this message and anything else the same full queue swallowed.
    /// Without this a single dropped OFF — an occupancy hold expiring into a
    /// queue with no room — would leave its retained ON standing for good,
    /// which is the bug the reconnect burst exists to prevent.
    fn note(&mut self, published: Published) {
        if published == Published::QueueFull {
            self.republish_pending = true;
        }
    }
}

/// Pure sensor bookkeeping: which cameras have motion open, when each last had
/// a snapshot taken, and which occupancy sensors are ON with their last
/// sighting. No I/O and no clock of its own — `now` is always injected — so the
/// hold-off and cadence rules are exercised directly in tests.
struct SensorState {
    snapshot_interval: Duration,
    occupancy_hold: Duration,
    motion_active: HashSet<String>,
    /// When each camera last had a snapshot published.
    last_snapshot: HashMap<String, Instant>,
    /// `(camera_id, class)` -> last sighting. Presence means the sensor is ON.
    occupancy: HashMap<(String, String), Instant>,
}

impl SensorState {
    fn new(snapshot_interval: Duration, occupancy_hold: Duration) -> Self {
        Self {
            snapshot_interval,
            occupancy_hold,
            motion_active: HashSet::new(),
            last_snapshot: HashMap::new(),
            occupancy: HashMap::new(),
        }
    }

    /// Open a motion run. `false` when it was already open (a duplicate start,
    /// which must not restart the snapshot cadence).
    fn motion_start(&mut self, camera_id: &str) -> bool {
        self.motion_active.insert(camera_id.to_string())
    }

    /// Close a motion run. `false` when nothing was open.
    fn motion_end(&mut self, camera_id: &str) -> bool {
        self.last_snapshot.remove(camera_id);
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
        self.last_snapshot.insert(camera_id.to_string(), now);
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

    /// Cameras whose next motion snapshot is due, marking each as taken. A
    /// camera that just opened its run has no `last_snapshot` and is therefore
    /// due immediately.
    fn due_snapshots(&mut self, now: Instant) -> Vec<String> {
        let interval = self.snapshot_interval;
        let due: Vec<String> = self
            .motion_active
            .iter()
            .filter(|camera_id| match self.last_snapshot.get(*camera_id) {
                Some(&last) => now.saturating_duration_since(last) >= interval,
                None => true,
            })
            .cloned()
            .collect();
        for camera_id in &due {
            self.last_snapshot.insert(camera_id.clone(), now);
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
    /// object detection is off — no occupancy entities are then created.
    pub classes: Vec<String>,
    pub shutdown: Arc<AtomicBool>,
}

/// Spawn the bridge. The returned handle is joined (with a timeout) during
/// shutdown so the retained `offline` marker gets published.
pub fn spawn_bridge(
    ctx: BridgeContext,
    rx: tokio::sync::mpsc::Receiver<MqttEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_bridge(ctx, rx))
}

async fn run_bridge(ctx: BridgeContext, mut rx: tokio::sync::mpsc::Receiver<MqttEvent>) {
    let topics = Topics::new(&ctx.config);
    let mut options = MqttOptions::new("camon", &ctx.config.host, ctx.config.port);
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

    let (client, mut eventloop) = AsyncClient::new(
        options,
        request_queue_capacity(ctx.camera_ids.len(), ctx.classes.len()),
    );
    let mut state = SensorState::new(
        Duration::from_secs(ctx.config.snapshot_interval_secs),
        Duration::from_secs(ctx.config.occupancy_hold_secs),
    );
    let mut snapshot_tasks: HashMap<String, tokio::task::JoinHandle<()>> = HashMap::new();
    let mut link = Link::default();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    tracing::info!(
        host = %ctx.config.host,
        port = ctx.config.port,
        cameras = ctx.camera_ids.len(),
        classes = ctx.classes.len(),
        "mqtt bridge started"
    );

    // Once every producer's sender has dropped, `recv()` returns `None`
    // immediately and forever; the branch is disabled so it cannot spin.
    let mut producers_gone = false;

    loop {
        tokio::select! {
            event = eventloop.poll() => match event {
                Ok(Event::Incoming(Incoming::ConnAck(_))) => {
                    tracing::info!(host = %ctx.config.host, "mqtt connected, publishing discovery");
                    link.connected = true;
                    link.republish_pending =
                        !publish_burst(&client, reconnect_burst(&topics, &state, &ctx));
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "mqtt connection error, retrying");
                    link.connected = false;
                    tokio::time::sleep(RECONNECT_DELAY).await;
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
                ),
                // Every producer is gone: the analyzers and detection worker
                // have exited, so shutdown is already under way. Keep serving
                // ticks until the flag confirms it rather than exiting here
                // with the availability marker still reading `online`.
                None => {
                    producers_gone = true;
                    if ctx.shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                }
            },
            _ = tick.tick() => {
                if ctx.shutdown.load(Ordering::Relaxed) {
                    break;
                }
                on_tick(&client, &topics, &mut state, &ctx, &mut snapshot_tasks, &mut link);
            }
        }
    }

    shutdown_bridge(client, &topics, &mut eventloop, snapshot_tasks).await;
}

fn handle_event(
    event: MqttEvent,
    client: &AsyncClient,
    topics: &Topics,
    state: &mut SensorState,
    ctx: &BridgeContext,
    snapshot_tasks: &mut HashMap<String, tokio::task::JoinHandle<()>>,
    link: &mut Link,
) {
    match event {
        MqttEvent::MotionStart { camera_id } => {
            if !state.motion_start(&camera_id) {
                return;
            }
            tracing::debug!(camera = %camera_id, "mqtt motion ON");
            link.note(publish_state(client, &topics.motion(&camera_id), "ON"));
            // The frame that opened the run is the interesting one: take it now
            // rather than waiting up to a full interval for the tick.
            state.mark_snapshot(&camera_id, Instant::now());
            spawn_snapshot(client, topics, ctx, &camera_id, snapshot_tasks, link);
        }
        MqttEvent::MotionEnd { camera_id } => {
            if !state.motion_end(&camera_id) {
                return;
            }
            tracing::debug!(camera = %camera_id, "mqtt motion OFF");
            link.note(publish_state(client, &topics.motion(&camera_id), "OFF"));
            // One last frame so the camera tile shows the end of the event
            // instead of freezing wherever the cadence happened to land.
            spawn_snapshot(client, topics, ctx, &camera_id, snapshot_tasks, link);
        }
        MqttEvent::Detections {
            camera_id,
            sightings,
        } => {
            let now = Instant::now();
            for sighting in sightings {
                let class = sighting.class;
                // A verdict can only name a configured class, but the sensor
                // set is built from the config, so anything else has no entity
                // to publish to.
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
                // No in-flight guard like `spawn_snapshot`'s: the bytes came
                // with the event, so this is a queue push and nothing else.
                // QoS 0 because the next verdict supersedes this one anyway;
                // retained so the tile keeps the sighting for good. Skipped
                // while disconnected for the same reason snapshots are: it
                // would sit in the queue until long after it mattered.
                if let (true, Some(jpeg)) = (link.connected, sighting.frame_jpeg) {
                    let topic = topics.occupancy_snapshot(&camera_id, &class);
                    // The crop itself is gone if this is rejected — the bytes
                    // are not kept — but a rejection still says the queue is
                    // full, so the sensor states get re-asserted.
                    link.note(publish_retained(client, &topic, QoS::AtMostOnce, jpeg));
                }
            }
        }
    }
}

fn on_tick(
    client: &AsyncClient,
    topics: &Topics,
    state: &mut SensorState,
    ctx: &BridgeContext,
    snapshot_tasks: &mut HashMap<String, tokio::task::JoinHandle<()>>,
    link: &mut Link,
) {
    let now = Instant::now();

    // Rebuilt from the live state on every attempt, never replayed from the
    // failed one: a retry a few seconds later must not assert a value that has
    // since changed.
    if link.connected && link.republish_pending {
        link.republish_pending = !publish_burst(client, reconnect_burst(topics, state, ctx));
    }

    for camera_id in state.due_snapshots(now) {
        spawn_snapshot(client, topics, ctx, &camera_id, snapshot_tasks, link);
    }

    for (camera_id, class) in state.expire_occupancy(now) {
        tracing::debug!(camera = %camera_id, class = %class, "mqtt occupancy OFF (hold elapsed)");
        link.note(publish_state(
            client,
            &topics.occupancy(&camera_id, &class),
            "OFF",
        ));
    }

    snapshot_tasks.retain(|_, handle| !handle.is_finished());
}

/// What became of one `try_publish`.
///
/// rumqttc reports a full queue and an unpublishable topic as the same error,
/// but they need opposite handling: the first is transient and the whole
/// recovery design rests on retrying it, while the second never becomes
/// publishable and retrying one is a 1 Hz loop that never gets past it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Published {
    Yes,
    QueueFull,
    /// A wildcard in a camera id or class name. `Config::validate` rejects both
    /// when the bridge is enabled, so this is a backstop against a livelock,
    /// not a path an operator is expected to reach.
    ImpossibleTopic,
}

/// Publish one retained message. Retained throughout: Home Assistant sees the
/// current value the moment it subscribes, rather than an unknown entity until
/// the next transition.
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

/// Everything a (re)connect owes the broker, in publish order: every camera's
/// discovery payloads, then every configured entity's current state, then the
/// availability marker.
///
/// One list rather than three calls so the order is a value that can be
/// asserted, instead of an implication of how a `select!` arm happens to be
/// written. Every payload is retained and idempotent, so republishing the whole
/// thing on a retry costs nothing but the bytes.
fn reconnect_burst(
    topics: &Topics,
    state: &SensorState,
    ctx: &BridgeContext,
) -> Vec<(String, Vec<u8>)> {
    let mut burst = Vec::new();
    for camera_id in &ctx.camera_ids {
        for (topic, payload) in discovery_payloads(topics, camera_id, &ctx.classes) {
            match serde_json::to_vec(&payload) {
                Ok(bytes) => burst.push((topic, bytes)),
                Err(e) => {
                    tracing::error!(topic = %topic, error = %e, "discovery payload not serializable")
                }
            }
        }
    }
    for (topic, payload) in state_payloads(topics, state, &ctx.camera_ids, &ctx.classes) {
        burst.push((topic, payload.as_bytes().to_vec()));
    }
    // Last: with every state already queued ahead of it, the connection Home
    // Assistant is told about is one whose retained values are current.
    burst.push((topics.availability(), b"online".to_vec()));
    burst
}

/// Hand the burst to the request queue, reporting whether all of it fit.
///
/// Nothing drains the queue between these publishes — `poll()` is not running
/// while this returns — so a rejection means the queue is full and every
/// publish after it fails too. That is the normal outcome of reconnecting after
/// an outage, so it is one warn line for the burst rather than one per topic,
/// and the caller retries rather than treating it as lost.
fn publish_burst(client: &AsyncClient, burst: Vec<(String, Vec<u8>)>) -> bool {
    let total = burst.len();
    let mut published = 0;
    for (topic, payload) in burst {
        match publish_retained(client, &topic, QoS::AtLeastOnce, payload) {
            // An impossible topic counts as done: it will never be publishable,
            // and stopping on it would hold up the rest of the burst — the
            // availability marker included — for as long as the process runs.
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

/// The retained payload every configured sensor should currently be holding,
/// on the same topics the discovery payloads point Home Assistant at.
///
/// Enumerated from the config, not from the sensors that are ON: an OFF that
/// was dropped by a full request queue, or an ON left retained by a process
/// that died mid-motion-run, is only corrected by saying OFF out loud.
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

/// Decode and publish one snapshot, unless this camera already has a decode in
/// flight. Detached so a slow ffmpeg never delays the event loop; the handle is
/// kept purely as the in-flight marker.
fn spawn_snapshot(
    client: &AsyncClient,
    topics: &Topics,
    ctx: &BridgeContext,
    camera_id: &str,
    snapshot_tasks: &mut HashMap<String, tokio::task::JoinHandle<()>>,
    link: &Link,
) {
    // Nothing drains the request queue while the broker is unreachable, so a
    // snapshot queued now is a few hundred KiB parked in memory for the length
    // of the outage, to be delivered long after anything cared about it.
    if !link.connected {
        tracing::debug!(camera = %camera_id, "mqtt disconnected, skipping snapshot");
        return;
    }
    if let Some(handle) = snapshot_tasks.get(camera_id) {
        if !handle.is_finished() {
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
    let handle = tokio::spawn(async move {
        match snapshot_jpeg(&data).await {
            Some(jpeg) => {
                // QoS 0: the next snapshot is at most one interval away, so a
                // lost frame costs nothing worth a retransmit, and this task
                // has no `Link` to report the rejection to. Retained so the
                // camera tile has an image right after HA subscribes.
                publish_retained(&client, &topic, QoS::AtMostOnce, jpeg);
            }
            None => tracing::debug!(camera = %camera, "snapshot decode produced no frame"),
        }
    });
    snapshot_tasks.insert(camera_id.to_string(), handle);
}

/// Decode the first frame of an MPEG-TS segment and JPEG-encode it.
///
/// Hot-buffer segments always start on a keyframe, so a single-frame decode of
/// one segment needs no priming. The scale/pad filter letterboxes into a fixed
/// [`SNAPSHOT_WIDTH`]x[`SNAPSHOT_HEIGHT`] frame so the output size is the same
/// for every camera whatever its native aspect ratio.
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

/// Feed `input` to `command` on stdin and return everything it writes to
/// stdout, killing the child if it has not finished within `timeout`.
///
/// stdin and stdout are driven concurrently because a whole GOP does not fit in
/// a pipe buffer, and writing it up front would deadlock against the child's
/// own output. `kill_on_drop` is what makes cancellation real: dropping this
/// future — the timeout firing, or shutdown aborting the task that holds it —
/// kills the child rather than orphaning it.
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

    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut out = Vec::with_capacity(capacity);

    let piped = async {
        // ffmpeg exits after one frame and closes stdin, so the tail of the
        // write failing with EPIPE is the normal case, not an error.
        let write = async {
            let _ = stdin.write_all(input).await;
            let _ = stdin.shutdown().await;
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

/// Publish the retained `offline` marker and disconnect cleanly, then keep
/// polling briefly so both packets actually reach the socket — `try_publish`
/// only queues them, `poll()` is what writes them.
async fn shutdown_bridge(
    client: AsyncClient,
    topics: &Topics,
    eventloop: &mut rumqttc::EventLoop,
    snapshot_tasks: HashMap<String, tokio::task::JoinHandle<()>>,
) {
    abort_snapshots(snapshot_tasks).await;

    // Nothing left to retry with: if this is rejected the DISCONNECT below is
    // too, so no clean disconnect goes out and the broker publishes the LWT.
    let _ = publish_state(&client, &topics.availability(), "offline");
    if let Err(e) = client.try_disconnect() {
        tracing::debug!(error = %e, "mqtt disconnect request failed");
    }

    let flush = tokio::time::timeout(SHUTDOWN_FLUSH, async {
        loop {
            match eventloop.poll().await {
                Ok(Event::Outgoing(rumqttc::Outgoing::Disconnect)) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });
    let _ = flush.await;
    tracing::info!("mqtt bridge stopped");
}

/// Cancel every in-flight snapshot decode. Aborting drops the task's future,
/// which drops its ffmpeg child and so kills it; joining afterwards is what
/// makes that ordering observable rather than a race with process exit. Bounded
/// because a task inside a blocking closure cannot be cancelled at all, and
/// shutdown must not wait on one.
async fn abort_snapshots(snapshot_tasks: HashMap<String, tokio::task::JoinHandle<()>>) {
    let handles: Vec<tokio::task::JoinHandle<()>> = snapshot_tasks.into_values().collect();
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
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_secs(5);
    const HOLD: Duration = Duration::from_secs(60);

    fn state() -> SensorState {
        SensorState::new(INTERVAL, HOLD)
    }

    #[test]
    fn slugify_folds_non_alphanumerics() {
        assert_eq!(slugify("Front Door"), "front_door");
        assert_eq!(slugify("front-door"), "front_door");
        assert_eq!(slugify("CAM.1"), "cam_1");
        assert_eq!(slugify("yard"), "yard");
    }

    #[test]
    fn capitalize_uppercases_first_char_only() {
        assert_eq!(capitalize("person"), "Person");
        assert_eq!(capitalize("delivery van"), "Delivery van");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn occupancy_turns_on_and_holds_then_expires() {
        let mut s = state();
        let t0 = Instant::now();

        // A sighting turns the sensor ON.
        assert!(s.record_sighting("yard", "person", t0));
        assert!(s.is_occupied("yard", "person"));

        // Within the hold-off nothing expires.
        assert!(s
            .expire_occupancy(t0 + HOLD - Duration::from_secs(1))
            .is_empty());
        assert!(s.is_occupied("yard", "person"));

        // At the hold-off it turns OFF.
        let expired = s.expire_occupancy(t0 + HOLD);
        assert_eq!(expired, vec![("yard".to_string(), "person".to_string())]);
        assert!(!s.is_occupied("yard", "person"));
    }

    #[test]
    fn new_sighting_extends_the_hold() {
        let mut s = state();
        let t0 = Instant::now();
        s.record_sighting("yard", "person", t0);

        // A second sighting inside the window is not a fresh transition but
        // does restart the countdown.
        let later = t0 + HOLD - Duration::from_secs(1);
        assert!(!s.record_sighting("yard", "person", later));
        assert!(s.expire_occupancy(t0 + HOLD).is_empty());
        assert!(s.is_occupied("yard", "person"));

        // The hold now runs from the newer sighting.
        assert_eq!(s.expire_occupancy(later + HOLD).len(), 1);
        assert!(!s.is_occupied("yard", "person"));
    }

    #[test]
    fn occupancy_re_arms_after_expiring() {
        let mut s = state();
        let t0 = Instant::now();
        s.record_sighting("yard", "person", t0);
        s.expire_occupancy(t0 + HOLD);
        // OFF, so the next sighting is a fresh OFF -> ON transition again.
        let t1 = t0 + HOLD + Duration::from_secs(30);
        assert!(s.record_sighting("yard", "person", t1));
        assert!(s.is_occupied("yard", "person"));
        assert!(s
            .expire_occupancy(t1 + HOLD - Duration::from_secs(1))
            .is_empty());
    }

    #[test]
    fn occupancy_is_tracked_per_camera_and_class() {
        let mut s = state();
        let t0 = Instant::now();
        s.record_sighting("yard", "person", t0);
        s.record_sighting("yard", "car", t0 + Duration::from_secs(30));
        s.record_sighting("gate", "person", t0 + Duration::from_secs(30));

        // Only the oldest pair expires at t0 + HOLD.
        let expired = s.expire_occupancy(t0 + HOLD);
        assert_eq!(expired, vec![("yard".to_string(), "person".to_string())]);
        assert!(s.is_occupied("yard", "car"));
        assert!(s.is_occupied("gate", "person"));
        assert!(!s.is_occupied("yard", "person"));
    }

    #[test]
    fn snapshots_are_due_immediately_then_on_the_interval() {
        let mut s = state();
        let t0 = Instant::now();
        // No motion: nothing to snapshot.
        assert!(s.due_snapshots(t0).is_empty());

        assert!(s.motion_start("yard"));
        // A freshly opened run is due at once.
        assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);
        // ...and not again until the interval elapses.
        assert!(s
            .due_snapshots(t0 + INTERVAL - Duration::from_millis(1))
            .is_empty());
        assert_eq!(s.due_snapshots(t0 + INTERVAL), vec!["yard".to_string()]);
    }

    #[test]
    fn motion_end_stops_the_snapshot_cadence() {
        let mut s = state();
        let t0 = Instant::now();
        s.motion_start("yard");
        s.due_snapshots(t0);
        assert!(s.motion_end("yard"));
        assert!(s.due_snapshots(t0 + INTERVAL * 10).is_empty());
        // A second end is a no-op — the caller must not publish OFF twice.
        assert!(!s.motion_end("yard"));
    }

    #[test]
    fn duplicate_motion_start_is_ignored() {
        let mut s = state();
        assert!(s.motion_start("yard"));
        assert!(!s.motion_start("yard"));
        assert_eq!(s.motion_active.len(), 1);
    }

    #[test]
    fn discovery_payloads_match_expected_json() {
        let config = MqttConfig {
            topic_prefix: "camon".to_string(),
            discovery_prefix: "homeassistant".to_string(),
            ..MqttConfig::default()
        };
        let topics = Topics::new(&config);
        let payloads = discovery_payloads(&topics, "Front Door", &["person".to_string()]);

        let device = serde_json::json!({
            "identifiers": ["camon_front_door"],
            "name": "Camon Front Door",
            "manufacturer": "camon",
            "sw_version": env!("CAMON_VERSION"),
        });

        assert_eq!(payloads.len(), 4);

        assert_eq!(
            payloads[0].0,
            "homeassistant/camera/camon_front_door/config"
        );
        assert_eq!(
            payloads[0].1,
            serde_json::json!({
                "name": "Snapshot",
                "unique_id": "camon_front_door_snapshot",
                "topic": "camon/Front Door/snapshot",
                "availability_topic": "camon/availability",
                "device": device,
            })
        );

        assert_eq!(
            payloads[1].0,
            "homeassistant/binary_sensor/camon_front_door_motion/config"
        );
        assert_eq!(
            payloads[1].1,
            serde_json::json!({
                "name": "Motion",
                "unique_id": "camon_front_door_motion",
                "state_topic": "camon/Front Door/motion",
                "device_class": "motion",
                "availability_topic": "camon/availability",
                "device": device,
            })
        );

        assert_eq!(
            payloads[2].0,
            "homeassistant/binary_sensor/camon_front_door_occupancy_person/config"
        );
        assert_eq!(
            payloads[2].1,
            serde_json::json!({
                "name": "Person occupancy",
                "unique_id": "camon_front_door_occupancy_person",
                "state_topic": "camon/Front Door/occupancy/person",
                "device_class": "occupancy",
                "availability_topic": "camon/availability",
                "device": device,
            })
        );

        assert_eq!(
            payloads[3].0,
            "homeassistant/camera/camon_front_door_occupancy_person/config"
        );
        assert_eq!(
            payloads[3].1,
            serde_json::json!({
                "name": "Person snapshot",
                "unique_id": "camon_front_door_occupancy_person_snapshot",
                "topic": "camon/Front Door/occupancy/person/snapshot",
                "availability_topic": "camon/availability",
                "device": device,
            })
        );
    }

    #[test]
    fn no_classes_means_no_occupancy_entities() {
        let topics = Topics::new(&MqttConfig::default());
        let payloads = discovery_payloads(&topics, "yard", &[]);
        assert_eq!(payloads.len(), 2);
    }

    #[test]
    fn every_class_adds_a_sensor_and_a_snapshot_camera() {
        let topics = Topics::new(&MqttConfig::default());
        let classes = ["person".to_string(), "cat".to_string()];
        let payloads = discovery_payloads(&topics, "yard", &classes);
        assert_eq!(payloads.len(), 2 + 2 * classes.len());
        // Unique ids must stay distinct across components.
        let ids: HashSet<&str> = payloads
            .iter()
            .map(|(_, payload)| payload["unique_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids.len(), payloads.len());
    }

    #[test]
    fn topics_tolerate_a_trailing_slash_in_the_prefix() {
        let config = MqttConfig {
            topic_prefix: "camon/".to_string(),
            discovery_prefix: "homeassistant/".to_string(),
            ..MqttConfig::default()
        };
        let topics = Topics::new(&config);
        assert_eq!(topics.availability(), "camon/availability");
        assert_eq!(topics.motion("yard"), "camon/yard/motion");
        assert_eq!(topics.occupancy("yard", "car"), "camon/yard/occupancy/car");
        assert_eq!(
            topics.occupancy_snapshot("yard", "car"),
            "camon/yard/occupancy/car/snapshot"
        );
        assert_eq!(topics.snapshot("yard"), "camon/yard/snapshot");
        assert_eq!(
            topics.discovery("camera", "camon_yard"),
            "homeassistant/camera/camon_yard/config"
        );
    }

    #[test]
    fn reconnect_publishes_an_explicit_state_for_every_entity() {
        let topics = Topics::new(&MqttConfig::default());
        let cameras = ["yard".to_string(), "gate".to_string()];
        let classes = ["person".to_string(), "cat".to_string()];
        let mut s = state();
        s.motion_start("yard");
        s.record_sighting("gate", "cat", Instant::now());

        let payloads = state_payloads(&topics, &s, &cameras, &classes);
        assert_eq!(payloads.len(), cameras.len() * (1 + classes.len()));

        let by_topic: HashMap<&str, &str> = payloads
            .iter()
            .map(|(topic, payload)| (topic.as_str(), *payload))
            .collect();
        assert_eq!(by_topic["camon/yard/motion"], "ON");
        assert_eq!(by_topic["camon/gate/occupancy/cat"], "ON");
        // The point of the enumeration: everything that is not ON says so,
        // instead of leaving whatever the broker still has retained.
        assert_eq!(by_topic["camon/gate/motion"], "OFF");
        assert_eq!(by_topic["camon/yard/occupancy/cat"], "OFF");
        assert_eq!(by_topic["camon/yard/occupancy/person"], "OFF");
        assert_eq!(by_topic["camon/gate/occupancy/person"], "OFF");
    }

    #[test]
    fn republished_topics_are_exactly_the_discovered_state_topics() {
        let topics = Topics::new(&MqttConfig::default());
        let cameras = ["Front Door".to_string(), "yard".to_string()];
        let classes = ["person".to_string()];

        let discovered: HashSet<String> = cameras
            .iter()
            .flat_map(|camera_id| discovery_payloads(&topics, camera_id, &classes))
            .filter_map(|(_, payload)| payload["state_topic"].as_str().map(str::to_string))
            .collect();
        let republished: HashSet<String> = state_payloads(&topics, &state(), &cameras, &classes)
            .into_iter()
            .map(|(topic, _)| topic)
            .collect();

        assert_eq!(discovered, republished);
    }

    fn bridge_context(cameras: &[&str], classes: &[&str]) -> BridgeContext {
        BridgeContext {
            config: MqttConfig::default(),
            buffers: Arc::new(HashMap::new()),
            camera_ids: cameras.iter().map(|c| c.to_string()).collect(),
            classes: classes.iter().map(|c| c.to_string()).collect(),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    fn capacity_for(ctx: &BridgeContext) -> usize {
        request_queue_capacity(ctx.camera_ids.len(), ctx.classes.len())
    }

    /// A client whose event loop is never polled, so its request queue only
    /// ever fills. That is precisely the state a reconnect after an outage
    /// finds it in. The event loop comes back too: dropping it closes the
    /// channel, which fails publishes for a reason a live bridge never has.
    fn unpolled_client(capacity: usize) -> (AsyncClient, rumqttc::EventLoop) {
        let options = MqttOptions::new("camon-test", "127.0.0.1", 1883);
        AsyncClient::new(options, capacity)
    }

    #[test]
    fn the_burst_is_discovery_then_states_then_availability() {
        let topics = Topics::new(&MqttConfig::default());
        let ctx = bridge_context(&["yard", "gate"], &["person"]);
        let burst = reconnect_burst(&topics, &state(), &ctx);

        let discovery = burst
            .iter()
            .rposition(|(topic, _)| topic.starts_with("homeassistant/"))
            .unwrap();
        let first_state = burst
            .iter()
            .position(|(topic, _)| topic.ends_with("/motion"))
            .unwrap();
        assert!(discovery < first_state);
        // The availability flip is the last thing queued, never the first.
        let (topic, payload) = burst.last().unwrap();
        assert_eq!(topic, "camon/availability");
        assert_eq!(payload, b"online");

        // Per camera: two discovery payloads plus two per class, one state
        // plus one per class. Then the marker.
        let (cameras, classes) = (ctx.camera_ids.len(), ctx.classes.len());
        assert_eq!(burst.len(), cameras * (3 + 3 * classes) + 1);
    }

    #[test]
    fn a_retried_burst_carries_the_state_of_the_retry_not_the_failure() {
        let topics = Topics::new(&MqttConfig::default());
        let ctx = bridge_context(&["yard"], &[]);
        let mut s = state();
        s.motion_start("yard");

        let motion = |burst: Vec<(String, Vec<u8>)>| {
            burst
                .into_iter()
                .find(|(topic, _)| topic == "camon/yard/motion")
                .map(|(_, payload)| payload)
                .unwrap()
        };
        assert_eq!(motion(reconnect_burst(&topics, &s, &ctx)), b"ON");

        // The run ends while the burst is still owed to the broker. Rebuilding
        // is what keeps the retry from re-asserting the value that has gone.
        s.motion_end("yard");
        assert_eq!(motion(reconnect_burst(&topics, &s, &ctx)), b"OFF");
    }

    #[test]
    fn the_queue_always_has_room_for_a_whole_burst() {
        let topics = Topics::new(&MqttConfig::default());
        // Well past any supported install, and past the fifteen-camera default
        // -class shape that does not fit a fixed 256.
        for (cameras, classes) in [(1, 0), (15, 5), (64, 16)] {
            let camera_ids: Vec<String> = (0..cameras).map(|i| format!("cam{i}")).collect();
            let class_names: Vec<String> = (0..classes).map(|i| format!("class{i}")).collect();
            let ctx = BridgeContext {
                config: MqttConfig::default(),
                buffers: Arc::new(HashMap::new()),
                camera_ids,
                classes: class_names,
                shutdown: Arc::new(AtomicBool::new(false)),
            };

            // The formula the queue is sized from must be the burst that is
            // actually built, or the sizing means nothing.
            let burst = reconnect_burst(&topics, &state(), &ctx);
            assert_eq!(burst.len(), burst_len(cameras, classes));

            // All-or-nothing publishing plus a burst that cannot fit is a
            // permanent retry loop that never reaches `online`.
            let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
            assert!(publish_burst(&client, burst));
        }
    }

    #[test]
    fn a_rejected_burst_goes_out_once_the_queue_drains() {
        let topics = Topics::new(&MqttConfig::default());
        let ctx = bridge_context(&["yard", "gate"], &["person"]);
        let burst = reconnect_burst(&topics, &state(), &ctx);

        // One message already queued leaves the burst a slot short of fitting.
        let (client, mut eventloop) = unpolled_client(burst.len());
        assert_eq!(publish_state(&client, "camon/filler", "x"), Published::Yes);
        assert!(!publish_burst(&client, burst.clone()));

        // What the event loop does when the connection comes back: take
        // everything the channel is holding. Recovery has to happen on this
        // queue, so the retry runs against the same client rather than a fresh
        // one that was never full.
        eventloop.clean();
        assert!(publish_burst(&client, burst));
    }

    #[tokio::test]
    async fn a_dropped_state_publish_asks_for_a_full_republish() {
        let topics = Topics::new(&MqttConfig::default());
        let ctx = bridge_context(&["yard"], &["person"]);
        // Zero hold, so the sighting recorded here expires on the next tick.
        let mut s = SensorState::new(INTERVAL, Duration::ZERO);
        s.record_sighting("yard", "person", Instant::now());

        let (client, _eventloop) = unpolled_client(1);
        assert_eq!(publish_state(&client, "camon/filler", "x"), Published::Yes);

        let mut link = Link {
            connected: true,
            republish_pending: false,
        };
        let mut tasks = HashMap::new();
        on_tick(&client, &topics, &mut s, &ctx, &mut tasks, &mut link);

        // The OFF had nowhere to go. Nothing else would ever re-assert it, so
        // the tick has to come back with the whole state or that sensor stays
        // ON in Home Assistant for good.
        assert!(link.republish_pending);
    }

    #[test]
    fn a_topic_that_can_never_be_published_stalls_nothing() {
        let topics = Topics::new(&MqttConfig::default());
        // `Config::validate` rejects this shape while the bridge is enabled;
        // the bridge defends itself anyway, because retrying it forever would
        // mean `online` never goes out.
        let ctx = bridge_context(&["ya+rd"], &[]);
        let burst = reconnect_burst(&topics, &state(), &ctx);
        let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
        assert!(publish_burst(&client, burst));

        assert_eq!(
            publish_state(&client, "camon/ya+rd/motion", "OFF"),
            Published::ImpossibleTopic
        );
        // ...and it must not ask for a retry that could not change anything.
        let mut link = Link::default();
        link.note(Published::ImpossibleTopic);
        assert!(!link.republish_pending);
        link.note(Published::QueueFull);
        assert!(link.republish_pending);
    }

    #[test]
    fn a_full_queue_fails_the_whole_burst_including_availability() {
        let topics = Topics::new(&MqttConfig::default());
        let ctx = bridge_context(&["yard"], &["person"]);
        let (client, _eventloop) = unpolled_client(2);
        // Fill it, exactly as an outage does.
        assert!(!publish_burst(
            &client,
            reconnect_burst(&topics, &state(), &ctx)
        ));
        // Nothing drains it, so a retry gets nowhere either — and reports so
        // rather than leaving `online` unsent and the entities unavailable.
        assert!(!publish_burst(
            &client,
            reconnect_burst(&topics, &state(), &ctx)
        ));
    }

    #[tokio::test]
    async fn the_tick_retries_the_burst_until_the_queue_takes_it() {
        let topics = Topics::new(&MqttConfig::default());
        let ctx = bridge_context(&["yard"], &["person"]);
        let mut s = state();
        let mut tasks = HashMap::new();
        let mut link = Link {
            connected: true,
            republish_pending: true,
        };

        let (small, _small_loop) = unpolled_client(2);
        on_tick(&small, &topics, &mut s, &ctx, &mut tasks, &mut link);
        // Still owed to the broker: the flag is what brings the next tick back
        // here, instead of leaving `online` unsent on a healthy connection.
        assert!(link.republish_pending);

        let (roomy, _roomy_loop) = unpolled_client(capacity_for(&ctx));
        on_tick(&roomy, &topics, &mut s, &ctx, &mut tasks, &mut link);
        assert!(!link.republish_pending);

        // A tick while disconnected must not spend the retry on a queue that
        // nothing is draining. Against a queue with room to spare, so the flag
        // can only still be pending because the guard held.
        link.republish_pending = true;
        link.connected = false;
        let (idle, _idle_loop) = unpolled_client(capacity_for(&ctx));
        on_tick(&idle, &topics, &mut s, &ctx, &mut tasks, &mut link);
        assert!(link.republish_pending);
    }

    #[tokio::test]
    async fn snapshots_are_not_queued_while_disconnected() {
        let topics = Topics::new(&MqttConfig::default());
        let buffer = HotBuffer::new("yard".to_string(), 60);
        {
            let mut buf = buffer.write_recover();
            for i in 0..2 {
                buf.push(crate::buffer::GopSegment {
                    start_pts: i * 1_000_000_000,
                    duration_ns: 1_000_000_000,
                    data: Arc::new(vec![0u8; 16]),
                    frame_count: 1,
                });
            }
        }
        let mut ctx = bridge_context(&["yard"], &[]);
        ctx.buffers = Arc::new(HashMap::from([("yard".to_string(), buffer)]));

        let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
        let mut tasks = HashMap::new();
        let mut link = Link::default();
        spawn_snapshot(&client, &topics, &ctx, "yard", &mut tasks, &link);
        // Disconnected: no decode, so no JPEG parked in a queue nothing is
        // draining until long after it mattered.
        assert!(tasks.is_empty());

        link.connected = true;
        spawn_snapshot(&client, &topics, &ctx, "yard", &mut tasks, &link);
        assert_eq!(tasks.len(), 1);
        for handle in tasks.values() {
            handle.abort();
        }
    }

    /// A pid is dead once `/proc` has lost it or reports it as a zombie —
    /// `kill_on_drop` reaps asynchronously, so the zombie window is normal.
    fn process_dead(pid: u32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat
                .rsplit(')')
                .next()
                .is_some_and(|rest| rest.trim_start().starts_with('Z')),
            Err(_) => true,
        }
    }

    #[tokio::test]
    async fn a_wedged_decode_is_bounded_and_kills_its_child() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("pid");
        let mut command = tokio::process::Command::new("sh");
        command.arg("-c").arg(format!(
            "echo $$ > {}; exec sleep 60",
            pidfile.to_str().unwrap()
        ));

        let started = std::time::Instant::now();
        let out = piped_decode(command, b"unread", 16, Duration::from_millis(200)).await;
        assert!(out.is_none());
        assert!(started.elapsed() < Duration::from_secs(5));

        // Written by the child before it execs `sleep`; on a loaded machine it
        // may not be there the moment the decode gives up.
        let mut pid = None;
        for _ in 0..100 {
            if let Ok(written) = std::fs::read_to_string(&pidfile) {
                if let Ok(parsed) = written.trim().parse::<u32>() {
                    pid = Some(parsed);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let pid = pid.expect("child never recorded its pid");
        for _ in 0..100 {
            if process_dead(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("child {pid} outlived the decode that owned it");
    }

    #[tokio::test]
    async fn a_decode_that_produces_nothing_is_not_an_error() {
        let mut command = tokio::process::Command::new("sh");
        command.arg("-c").arg("exit 0");
        let out = piped_decode(command, b"unread", 16, Duration::from_secs(5)).await;
        assert_eq!(out, Some(Vec::new()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_does_not_wait_on_a_decode_that_ignores_cancellation() {
        // A task with no await point at all, which `abort` cannot reach. Real
        // snapshot tasks are not this shape — they await the ffmpeg pipe and
        // then the encode, and abort lands there in microseconds — so this is
        // the bound's backstop case rather than its everyday one.
        let mut tasks = HashMap::new();
        tasks.insert(
            "yard".to_string(),
            tokio::spawn(async { std::thread::sleep(Duration::from_secs(1)) }),
        );

        let started = std::time::Instant::now();
        abort_snapshots(tasks).await;
        assert!(started.elapsed() < Duration::from_millis(900));
    }

    #[tokio::test]
    async fn shutdown_joins_cancellable_decodes_at_once() {
        let mut tasks = HashMap::new();
        tasks.insert(
            "yard".to_string(),
            tokio::spawn(async { tokio::time::sleep(Duration::from_secs(60)).await }),
        );

        let started = std::time::Instant::now();
        abort_snapshots(tasks).await;
        assert!(started.elapsed() < SNAPSHOT_ABORT_JOIN);
    }

    #[tokio::test]
    async fn send_event_drops_instead_of_blocking() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        assert!(send_event(
            &tx,
            MqttEvent::MotionStart {
                camera_id: "yard".to_string()
            }
        ));
        // Queue full: the producer moves on rather than stalling.
        assert!(!send_event(
            &tx,
            MqttEvent::MotionEnd {
                camera_id: "yard".to_string()
            }
        ));
        assert_eq!(
            rx.recv().await.unwrap(),
            MqttEvent::MotionStart {
                camera_id: "yard".to_string()
            }
        );

        drop(rx);
        assert!(!send_event(
            &tx,
            MqttEvent::MotionEnd {
                camera_id: "yard".to_string()
            }
        ));
    }
}
