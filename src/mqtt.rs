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
//! Per camera, three kinds of entity:
//!
//! - a **camera** fed by JPEG snapshots;
//! - a **motion** binary sensor, ON for the lifetime of a motion run;
//! - one **occupancy** binary sensor per configured object-detection class, ON
//!   from the moment a verdict names that class until `occupancy_hold_secs`
//!   pass with no new sighting.
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
//!   a self-inflicted deadlock. Dropped publishes are recovered by the
//!   reconnect republish below.
//! - snapshot decoding runs in a detached task (a `spawn_blocking` ffmpeg pipe
//!   inside it), with at most one in flight per camera.
//!
//! # Reconnect republish
//!
//! On every `ConnAck` the bridge republishes the full discovery set, the
//! availability marker and all current sensor states. A broker restart loses
//! retained messages, Home Assistant re-reads discovery when *it* restarts, and
//! anything dropped during an outage is made good here. Republishing is
//! idempotent, so doing it unconditionally is simpler and safer than tracking
//! what the broker still holds.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use rumqttc::{AsyncClient, Event, Incoming, LastWill, MqttOptions, QoS};

use crate::buffer::HotBuffer;
use crate::config::MqttConfig;
use crate::locks::LockExt;

/// Depth of the analyzer/detector -> bridge event channel. Events are tiny and
/// rare (a motion transition or a verdict), so this is far more headroom than a
/// healthy bridge ever needs; producers `try_send` and drop on full rather than
/// ever stalling the analysis threads.
pub const MQTT_EVENT_CAPACITY: usize = 64;

/// Depth of the rumqttc request queue. Sized for the reconnect burst (discovery
/// for every camera and class, plus every sensor's current state) so it is not
/// the thing that drops messages.
const REQUEST_QUEUE_CAPACITY: usize = 256;

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

/// Something the bridge should reflect to Home Assistant. Produced by the
/// analyzer threads (motion lifecycle) and the detection worker (verdicts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqttEvent {
    /// A motion run opened. Chunked continuations of one physical motion period
    /// do NOT produce a second start — see `analytics::pipeline`.
    MotionStart { camera_id: String },
    /// The motion run closed (post-padding elapsed, or a shutdown flush).
    MotionEnd { camera_id: String },
    /// A detection verdict, already deduplicated and confidence-filtered.
    Detections {
        camera_id: String,
        classes: Vec<String>,
    },
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
/// unique ids, so they are normalized once here.
fn slugify(id: &str) -> String {
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
/// motion sensor, and one occupancy sensor per configured class.
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
    }

    out
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

    fn motion_cameras(&self) -> impl Iterator<Item = &String> {
        self.motion_active.iter()
    }

    /// Record a sighting of `class`, turning its sensor ON. `true` when this is
    /// a fresh OFF -> ON transition.
    fn record_sighting(&mut self, camera_id: &str, class: &str, now: Instant) -> bool {
        self.occupancy
            .insert((camera_id.to_string(), class.to_string()), now)
            .is_none()
    }

    #[cfg(test)]
    fn is_occupied(&self, camera_id: &str, class: &str) -> bool {
        self.occupancy
            .contains_key(&(camera_id.to_string(), class.to_string()))
    }

    /// Note that `camera_id` has just been snapshotted, restarting its cadence.
    fn mark_snapshot(&mut self, camera_id: &str, now: Instant) {
        self.last_snapshot.insert(camera_id.to_string(), now);
    }

    fn occupied_pairs(&self) -> impl Iterator<Item = &(String, String)> {
        self.occupancy.keys()
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

    let (client, mut eventloop) = AsyncClient::new(options, REQUEST_QUEUE_CAPACITY);
    let mut state = SensorState::new(
        Duration::from_secs(ctx.config.snapshot_interval_secs),
        Duration::from_secs(ctx.config.occupancy_hold_secs),
    );
    let mut snapshot_tasks: HashMap<String, tokio::task::JoinHandle<()>> = HashMap::new();
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
                    publish_discovery(&client, &topics, &ctx);
                    republish_state(&client, &topics, &state);
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "mqtt connection error, retrying");
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
                on_tick(&client, &topics, &mut state, &ctx, &mut snapshot_tasks);
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
) {
    match event {
        MqttEvent::MotionStart { camera_id } => {
            if !state.motion_start(&camera_id) {
                return;
            }
            tracing::debug!(camera = %camera_id, "mqtt motion ON");
            publish_state(client, &topics.motion(&camera_id), "ON");
            // The frame that opened the run is the interesting one: take it now
            // rather than waiting up to a full interval for the tick.
            state.mark_snapshot(&camera_id, Instant::now());
            spawn_snapshot(client, topics, ctx, &camera_id, snapshot_tasks);
        }
        MqttEvent::MotionEnd { camera_id } => {
            if !state.motion_end(&camera_id) {
                return;
            }
            tracing::debug!(camera = %camera_id, "mqtt motion OFF");
            publish_state(client, &topics.motion(&camera_id), "OFF");
            // One last frame so the camera tile shows the end of the event
            // instead of freezing wherever the cadence happened to land.
            spawn_snapshot(client, topics, ctx, &camera_id, snapshot_tasks);
        }
        MqttEvent::Detections { camera_id, classes } => {
            let now = Instant::now();
            for class in classes {
                // A verdict can only name a configured class, but the sensor
                // set is built from the config, so anything else has no entity
                // to publish to.
                if !ctx.classes.contains(&class) {
                    continue;
                }
                if state.record_sighting(&camera_id, &class, now) {
                    tracing::debug!(camera = %camera_id, class = %class, "mqtt occupancy ON");
                }
                publish_state(client, &topics.occupancy(&camera_id, &class), "ON");
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
) {
    let now = Instant::now();

    for camera_id in state.due_snapshots(now) {
        spawn_snapshot(client, topics, ctx, &camera_id, snapshot_tasks);
    }

    for (camera_id, class) in state.expire_occupancy(now) {
        tracing::debug!(camera = %camera_id, class = %class, "mqtt occupancy OFF (hold elapsed)");
        publish_state(client, &topics.occupancy(&camera_id, &class), "OFF");
    }

    snapshot_tasks.retain(|_, handle| !handle.is_finished());
}

/// Publish a retained sensor state. Retained so Home Assistant sees the current
/// value the moment it subscribes, rather than an unknown sensor until the next
/// transition.
fn publish_state(client: &AsyncClient, topic: &str, payload: &str) {
    if let Err(e) = client.try_publish(topic, QoS::AtLeastOnce, true, payload) {
        tracing::warn!(topic = %topic, error = %e, "mqtt publish failed");
    }
}

fn publish_discovery(client: &AsyncClient, topics: &Topics, ctx: &BridgeContext) {
    for camera_id in &ctx.camera_ids {
        for (topic, payload) in discovery_payloads(topics, camera_id, &ctx.classes) {
            match serde_json::to_vec(&payload) {
                Ok(bytes) => {
                    if let Err(e) = client.try_publish(&topic, QoS::AtLeastOnce, true, bytes) {
                        tracing::warn!(topic = %topic, error = %e, "mqtt discovery publish failed");
                    }
                }
                Err(e) => {
                    tracing::error!(topic = %topic, error = %e, "discovery payload not serializable")
                }
            }
        }
    }
    publish_state(client, &topics.availability(), "online");
}

/// Re-assert every sensor's current value after a (re)connect.
fn republish_state(client: &AsyncClient, topics: &Topics, state: &SensorState) {
    for camera_id in state.motion_cameras() {
        publish_state(client, &topics.motion(camera_id), "ON");
    }
    for (camera_id, class) in state.occupied_pairs() {
        publish_state(client, &topics.occupancy(camera_id, class), "ON");
    }
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
) {
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
        let jpeg = tokio::task::spawn_blocking(move || decode_snapshot_jpeg(&data)).await;
        match jpeg {
            Ok(Some(jpeg)) => {
                // QoS 0: the next snapshot is at most one interval away, so a
                // lost frame costs nothing worth a retransmit. Retained so the
                // camera tile has an image right after HA subscribes.
                if let Err(e) = client.try_publish(&topic, QoS::AtMostOnce, true, jpeg) {
                    tracing::warn!(camera = %camera, error = %e, "snapshot publish failed");
                }
            }
            Ok(None) => tracing::debug!(camera = %camera, "snapshot decode produced no frame"),
            Err(e) => tracing::warn!(camera = %camera, error = %e, "snapshot decode task failed"),
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
///
/// Blocking: stdin is fed from a helper thread while this one drains stdout,
/// because a whole GOP does not fit in a pipe buffer and writing it up front
/// would deadlock against ffmpeg's own output.
fn decode_snapshot_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    let filter = format!(
        "scale={SNAPSHOT_WIDTH}:{SNAPSHOT_HEIGHT}:force_original_aspect_ratio=decrease,\
         pad={SNAPSHOT_WIDTH}:{SNAPSHOT_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    );
    let mut child = Command::new("ffmpeg")
        .args([
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
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| tracing::warn!(error = %e, "failed to spawn snapshot ffmpeg"))
        .ok()?;

    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut stdout = child.stdout.take().expect("stdout piped");
    let segment = data.to_vec();
    // ffmpeg exits after one frame and closes stdin, so the tail of the write
    // failing with EPIPE is the normal case, not an error.
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&segment);
        let _ = stdin.flush();
    });

    let expected = (SNAPSHOT_WIDTH * SNAPSHOT_HEIGHT * 3) as usize;
    let mut frame = Vec::with_capacity(expected);
    let read = stdout.read_to_end(&mut frame);
    let _ = child.wait();
    let _ = writer.join();
    if let Err(e) = read {
        tracing::warn!(error = %e, "snapshot frame read failed");
        return None;
    }
    if frame.len() < expected {
        return None;
    }
    frame.truncate(expected);

    encode_jpeg(&frame, SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT)
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
    for (_, handle) in snapshot_tasks {
        handle.abort();
    }

    publish_state(&client, &topics.availability(), "offline");
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
        assert_eq!(s.motion_cameras().count(), 1);
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

        assert_eq!(payloads.len(), 3);

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
    }

    #[test]
    fn no_classes_means_no_occupancy_entities() {
        let topics = Topics::new(&MqttConfig::default());
        let payloads = discovery_payloads(&topics, "yard", &[]);
        assert_eq!(payloads.len(), 2);
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
        assert_eq!(topics.snapshot("yard"), "camon/yard/snapshot");
        assert_eq!(
            topics.discovery("camera", "camon_yard"),
            "homeassistant/camera/camon_yard/config"
        );
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
