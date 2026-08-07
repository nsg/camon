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
//! # Never block the loop, and never cancel it either
//!
//! The event loop must be polled continuously — rumqttc does all of its I/O,
//! including draining the client's request queue, inside `poll()`. Three
//! consequences shape this module:
//!
//! - `poll()` never appears as a `select!` arm, because it is not
//!   cancellation-safe. It runs in a plain loop on a task of its own, which
//!   does nothing else; see [`Eventloop`]. A QoS-1 publish is booked as
//!   in-flight *before* its bytes are written, so a `poll()` dropped
//!   mid-write leaks that slot — no PUBACK can ever retire it — and a hundred
//!   leaked slots wedge the request path shut for good: a bridge that still
//!   answers keepalives, still accepts publishes and still reports `online`
//!   while nothing it queues ever reaches the broker. Cancelling a connect
//!   attempt is the same bug in miniature, since the handshake then restarts
//!   from the beginning every time and a broker slower than the cancelling
//!   arm's cadence is never reached at all.
//! - every publish uses `try_publish`, never the awaiting `publish`. While the
//!   broker is unreachable the request queue backs up and nothing drains it,
//!   so an awaiting publish would park the bridge loop for the length of the
//!   outage — no ticks, no events, and no reconnect burst to recover with.
//!   Rejected publishes are recovered by the reconnect republish below.
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
//! "Every configured entity" means every entity *this* config describes; what a
//! previous one described is dealt with below.
//!
//! The burst is queued at its least favourable moment: nothing has drained the
//! request queue for the length of the outage, so it can be rejected in full.
//! It is therefore retried from the tick, rebuilt from the live state each
//! time, until the whole of it is accepted — losing it would leave both a stale
//! retained ON and the LWT's `offline` standing on a healthy connection. For
//! that retry to be able to converge the queue is sized from the config to hold
//! a whole burst; see [`request_queue_capacity`].
//!
//! Order within the burst is orphan clears, then discovery, then states, then
//! `online`. MQTT only guarantees ordering per topic, so this is not a promise
//! about what Home Assistant observes in what instant; it is that camon never
//! leaves the *final* retained set inconsistent, and that the availability flip
//! is the last thing queued rather than the first.
//!
//! # Forgetting an entity the config dropped
//!
//! Renaming or removing a camera, or dropping an object class, leaves the
//! broker holding that entity's retained discovery document and its retained
//! state. The burst above contradicts neither — it enumerates the current
//! config — and since every entity shares one availability topic, publishing
//! `online` makes those orphans available again: an entity showing whatever
//! motion happened to be open when the camera was removed, permanently, in
//! every restart.
//!
//! So the bridge remembers the entity set it announced, in
//! `{data_dir}/mqtt_entities.json`, and clears what the current set no longer
//! explains at the head of the next reconnect burst: an empty retained payload
//! on the discovery topic, which is how Home Assistant is told to forget a
//! discovered entity, and one on the topic that document pointed at, which the
//! broker otherwise keeps holding for good — a per-class sighting crop is
//! hundreds of KiB of it.
//!
//! Clearing something that was only *temporarily* absent costs the operator an
//! entity's history and its place on a dashboard, so what counts as dropped is
//! deliberately narrow. Cameras come from the config outright, so one that is
//! not in it was renamed or removed. Classes do not: an empty class list means
//! object detection is off *this run*, which is equally what a misconfigured or
//! unreachable vision server looks like, so the remembered classes are carried
//! forward. Carried forward *and announced* — the burst is built from the same
//! record, so those entities keep their discovery documents and are restated
//! OFF, rather than being left for `online` to resurrect with whatever the
//! broker still held. No record at all — a first start, or one that cannot be
//! read — clears nothing.
//!
//! Deleting is only camon's to do where camon published: the record names the
//! broker it announced to and its format version, and a record that does not
//! match this build and this broker is not acted on. Two camon instances
//! sharing a broker, a discovery prefix *and* a camera id do co-own that
//! camera's entities, and a removal in one is a removal for both; give them
//! distinct camera ids or distinct discovery prefixes.
//!
//! Delivery is where the record is careful. `try_publish` only queues, so
//! acceptance is recorded *with the clears still marked owed*: a process that
//! dies before the eventloop writes them clears the same topics again on the
//! next start. They are dropped from the record only when the shutdown
//! `Disconnect` goes out, which — requests being written in queue order — puts
//! everything queued before it on the wire. An unclean exit therefore costs one
//! redundant round of clears, which is a no-op on a topic that no longer holds
//! anything; the opposite trade would cost an entity that never came back.
//!
//! # One connection per instance
//!
//! The broker identifies a session by its client id, and enforces that identity
//! by disconnecting the older session whenever a second one arrives claiming
//! the same id. Two camon instances pointed at one broker under a shared id
//! therefore kick each other off for ever — each reconnect ends the other's
//! connection, and neither ever finishes a reconnect burst. So the id is
//! derived per instance; see [`derive_client_id`]. It is camon's identity *to
//! the broker* and nothing else: Home Assistant discovers entities by the
//! object ids and `unique_id`s in the discovery payloads, which are built from
//! camera ids alone, so what this id is cannot move, rename or duplicate a
//! single entity.
//!
//! Coexisting is not the same as being independent. Two instances that share a
//! `topic_prefix` share the one availability topic under it, last writer and
//! last will included — see [`Topics::availability`]. The id makes them able to
//! stay connected at the same time; only distinct prefixes make what they say
//! about themselves distinguishable.

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

/// Depth of the analyzer/detector -> bridge event channel. Events are tiny and
/// rare (a motion transition or a verdict), so this is far more headroom than a
/// healthy bridge ever needs; producers `try_send` and drop on full rather than
/// ever stalling the analysis threads.
pub const MQTT_EVENT_CAPACITY: usize = 64;

/// Headroom above one reconnect burst: room for the transitions and snapshots
/// raised while that burst is still being drained.
const REQUEST_QUEUE_HEADROOM: usize = 256;

/// How many publishes one reconnect burst is — one clear per orphaned topic,
/// then per camera two discovery payloads plus two per class, one state plus
/// one per class, then the availability marker. A test ties this to what
/// [`reconnect_burst`] actually produces.
fn burst_len(cameras: usize, classes: usize, orphans: usize) -> usize {
    orphans + cameras * (3 + 3 * classes) + 1
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
fn request_queue_capacity(cameras: usize, classes: usize, orphans: usize) -> usize {
    burst_len(cameras, classes, orphans) + REQUEST_QUEUE_HEADROOM
}

/// Outgoing packet-size ceiling. rumqttc defaults to 10 KiB, which every
/// snapshot would exceed; a 1280x720 JPEG at quality 90 is a few hundred KiB.
const MAX_OUTGOING_PACKET_BYTES: usize = 4 * 1024 * 1024;

/// How many image bytes may be handed to the request queue between two ticks.
///
/// This is a bound on the *rate* at which camon can commit memory to the queue,
/// and it is worth being exact about what that does and does not buy, because
/// [`request_queue_capacity`] bounds the queue in slots only: fifteen cameras
/// with five classes size it at 527 of them, and a slot holding a JPEG is worth
/// up to [`MAX_OUTGOING_PACKET_BYTES`]. Only images come near that — every
/// other publish this bridge makes is a discovery document or a two-byte state.
///
/// What camon cannot do is bound residency directly. rumqttc offers no signal
/// for a request having left the queue, so nothing here can know how full it
/// is; a real residency budget would need drain feedback that does not exist.
/// That is the residual, and it leaves three regimes:
///
/// - **Healthy link.** rumqttc takes each request and writes it straight out,
///   so the queue is near enough empty and this ceiling is purely a rate cap.
///   It is set well above what a healthy second costs: fifteen cameras all
///   opening a motion run at once is fifteen snapshots of a few hundred KiB.
/// - **Outage.** The connection fails, whereupon rumqttc moves the queue into
///   its pending set — and *keeps* that set across every failed reconnect,
///   dropping it only at the first successful `ConnAck`, since camon's
///   sessions are clean ones. What that set holds is whatever the queue held
///   at the moment of failure. A socket that errors outright is caught within
///   a tick, so roughly one window, 16 MiB; a flush that stalls takes up to
///   rumqttc's five-second timeout to be declared dead, and the bridge counts
///   as connected for all of it, so up to five refilled windows — some 80 MiB
///   — can be admitted first; and a failure that ends the regime below
///   inherits everything that regime accumulated. Snapshots already admitted
///   before the disconnect also finish their publish from their detached
///   tasks (bounded by the one-in-flight-per-camera guard); only new decodes
///   stop, see [`spawn_snapshot`].
/// - **A broker that answers but never acknowledges.** The bad one. rumqttc
///   stops taking requests once 100 QoS-1 publishes are unacked, so a broker
///   that accepts packets and answers keepalives while withholding PUBACKs
///   holds a connection camon believes in while nothing drains — and the
///   channel fills to the item cap. (The images themselves are QoS 0 and hold
///   no inflight slots; it is the two-byte QoS-1 state publishes that fill
///   the window, and the disabled request branch that strands the images
///   behind them.) Residency is then bounded by that cap alone: 527 slots of
///   realistic ~300 KiB snapshots is some 160 MiB, and the 4 MiB packet
///   ceiling makes 2 GiB the theoretical worst. What this ceiling does buy is
///   time — filling it takes at least the total divided by 16 MiB in ticks,
///   ten seconds for the realistic figure and two minutes for the theoretical
///   one. By this regime's own premise the keepalive path never fires, so
///   camon does not leave it on its own: as the residual above says, it ends
///   when an operator restarts something or the broker's behaviour changes,
///   and only then does the outage regime inherit the accumulation.
const MAX_IMAGE_BYTES_PER_TICK: usize = 16 * 1024 * 1024;

/// The image allowance of [`MAX_IMAGE_BYTES_PER_TICK`], shared by the bridge
/// loop and every detached snapshot task, since both queue images and the
/// bound is on their sum.
#[derive(Clone, Default)]
struct ImageBudget {
    spent: Arc<std::sync::atomic::AtomicUsize>,
    /// Whether this window's refusal has already been reported. A stalled
    /// broker refuses every camera every second; the operator needs to hear
    /// that once a second, not once per camera per second.
    reported: Arc<AtomicBool>,
}

impl ImageBudget {
    /// Charge one image, reporting whether it may be queued. A refusal is a
    /// dropped image and nothing more: the camera tile keeps the frame it has
    /// until the next one is due, exactly as it does for a decode that produced
    /// nothing.
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

    /// A new window: on a healthy link whatever the last one queued has had a
    /// tick to reach the socket. On an unhealthy one the windows accumulate in
    /// the queue instead — see [`MAX_IMAGE_BYTES_PER_TICK`] for what bounds
    /// that.
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

/// Derive the client id camon connects under, as
/// `camon-<[`CLIENT_ID_HASH_BYTES`] hex>`.
///
/// Uniqueness is the load-bearing property: a broker disconnects the older
/// session the moment a second one claims the same id, so two camons sharing an
/// id spend their lives evicting each other. Stability across restarts is worth
/// less than it looks — camon's sessions are clean ones, so the broker keeps no
/// state under the id, retained messages belong to their topics and the last
/// will belongs to the connection, none of it held for an id that comes back —
/// but it is still worth having: it is what makes a broker's logs, its ACLs and
/// its connected-clients list say the same thing about this camon tomorrow as
/// they do today, rather than accumulating a new name per restart.
///
/// So it is derived from what identifies the instance rather than generated:
///
/// - the machine's hostname, which is what separates two camons on two hosts
///   publishing to one broker — their data dirs are very likely the same path;
/// - the path of the entity record, which lives in the data dir and so is what
///   separates two camons on one host: they cannot share a data dir, each
///   owning its own hot buffer state, warm index and entity record in it.
///
/// Deliberately nothing else. Not the camera list or the classes, which change
/// whenever the operator adds a camera and would take the session identity with
/// them; not the topic or discovery prefixes, which two instances may share —
/// permitted, though not advisable (see [`Topics::availability`]); and not the
/// broker address, which
/// is by definition the same for the two instances that would collide.
///
/// The path is taken as configured rather than canonicalized: resolving it is
/// I/O against a directory that need not exist yet, and a config that keeps
/// saying the same thing keeps producing the same id, which is all stability
/// asks for. It follows that the path separates instances only *as spelled* —
/// two camons started from different working directories with the same relative
/// `data_dir` are distinct instances that derive one id, and the operator who
/// arranges that wants absolute paths (or distinct ones).
///
/// Both halves of the identity are handed in rather than read here, so each can
/// be varied on its own.
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

/// How much of the digest the client id carries. Six bytes is 48 bits against a
/// handful of instances per broker, and keeps the whole id at 18 characters —
/// inside the 23 that MQTT 3.1.1 requires every broker to accept, which a
/// hostname pasted in raw would not be.
const CLIENT_ID_HASH_BYTES: usize = 6;

/// The kernel's hostname, or `None` when there is none to read. An unnamed host
/// costs [`derive_client_id`] one of its two dimensions, never the id itself.
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

/// Depth of the eventloop task -> bridge notification channel. Its entries are
/// connection edges, which [`RECONNECT_DELAY`] already paces to at most a
/// couple per five seconds, so this is several outages' worth of headroom.
const LINK_EVENT_CAPACITY: usize = 16;

/// How long shutdown waits for the eventloop task to end once it has been told
/// to stop. Usually far less: the broker closes the connection it was just told
/// to disconnect from, `poll()` returns the error and the task falls out. A
/// broker that leaves the socket open instead parks it in `poll()`, where no
/// signal reaches, and this bound is what keeps shutdown from waiting on it.
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

/// The longest snapshot cadence camon will honour. The cadence only runs while
/// a motion run is open, and a run that stays open a whole day paces at most
/// one snapshot in it either way; the ceiling exists so that an operator's
/// `u64` cannot be turned into an `Instant` that does not exist. A configured
/// cadence above a day is honoured as "one per day", clamped with a warning.
const MAX_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// How soon a snapshot that produced nothing is attempted again, instead of
/// waiting out the whole cadence. Short enough that a camera recovering within
/// a motion run still gets a tile out of it, long enough that a camera failing
/// for good forks one ffmpeg every couple of seconds rather than every tick.
/// Only ever shortens the wait: a cadence faster than this keeps its own pace.
const SNAPSHOT_RETRY_DELAY: Duration = Duration::from_secs(2);

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

    /// The one topic every entity of this prefix reports its availability on —
    /// and it is the *prefix's*, not the instance's. Two camons configured with
    /// the same `topic_prefix` therefore share it: last writer wins, the last
    /// will included, so a dying instance's retained `offline` sits over a
    /// perfectly healthy sibling until that sibling next reconnects and
    /// republishes `online`. Distinct instances want distinct prefixes for
    /// availability to mean anything about any one of them. The per-instance
    /// client id (see [`derive_client_id`]) makes coexistence possible at all —
    /// before it they evicted each other's sessions — but it does not divide
    /// this namespace, and nothing but the prefix can.
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

/// Every retained topic one camera's entities occupy: each discovery document
/// and the topic that document points Home Assistant at. Read back out of the
/// payloads rather than rebuilt alongside them, so an entity kind cannot be
/// announced without also being clearable.
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
/// camon's to act on: deleting a Home Assistant entity on the strength of a
/// document this build does not fully understand is how a wrong deletion
/// happens.
const ENTITY_RECORD_VERSION: u32 = 1;

/// The broker an entity set belongs to, as the record names it.
fn broker_id(config: &MqttConfig) -> String {
    format!("{}:{}", config.host, config.port)
}

/// The entity set camon announced to a broker, remembered across restarts so
/// the next start can tell Home Assistant to forget what the config dropped.
///
/// Every field is deletion authority, so every field is required and unknown
/// ones are refused: a record that is not exactly this shape reads as no record
/// rather than as an entity set with empty prefixes. `broker` scopes the
/// authority to where the entities were actually published — a data dir carried
/// to another broker, or pointed at one, must not delete entities inferred from
/// what a different broker was told. The prefixes decide every topic, so moving
/// one orphans that whole side of the set.
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
    /// Clears that were queued but are not known to have left the process. They
    /// ride the next run's burst as well, until a clean disconnect proves the
    /// socket took them: re-clearing a cleared topic is a no-op, while
    /// forgetting one that never went out is a permanent ghost entity.
    pending_clears: Vec<String>,
}

impl EntityRecord {
    /// The set this run announces, given what the last one announced.
    ///
    /// Classes are carried forward when this run has none: an empty list means
    /// object detection produced no classes *this run* — it is off, or its
    /// vision client could not be built — which is no evidence that the
    /// operator dropped those entities. What is carried forward is announced,
    /// not merely remembered: the burst is built from this record, so those
    /// occupancy entities keep their discovery documents and are restated OFF,
    /// which is what they are while nothing is looking for them. Cameras are
    /// taken as they are — that list is the config's own.
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

/// Retained topics the previous record holds that the current set does not
/// explain, plus whatever it left owed. Anything the current set does announce
/// is filtered back out — a camera removed and then added again is live, and
/// clearing it would delete an entity camon is publishing to. Sorted and
/// deduplicated so the burst is the same every time it is rebuilt.
fn orphaned_topics(previous: &EntityRecord, current: &EntityRecord) -> Vec<String> {
    let live = current.retained_topics();
    let mut orphans: Vec<String> = previous
        .retained_topics()
        .into_iter()
        .chain(previous.pending_clears.iter().cloned())
        .filter(|topic| !live.contains(topic))
        .collect();
    // The availability topic belongs to no single entity — every one of them
    // shares it — so it is only ever orphaned by the prefix itself moving.
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

/// Persist the record the way every other small camon file is written: stage,
/// fsync, rename, fsync the directory. A torn record reads as no record, which
/// costs a cleanup that would have happened; the rename makes even that
/// unreachable.
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
///
/// The writes are small, synchronous and at most two per process (one when the
/// burst is first accepted, one when the disconnect proves it went out). They
/// are not handed to a blocking task because the failure has to come back here:
/// a record that did not reach disk must stay owed, or a transient write error
/// silently costs the operator's *next* removal its cleanup.
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
    /// Whether the clears are sitting in the request queue of the session that
    /// is up *now*.
    ///
    /// Scoped to the session on purpose. A queued request survives exactly as
    /// long as the connection it was queued on: when that connection fails,
    /// rumqttc moves whatever the queue still held into its pending set, and
    /// the next connect — camon's sessions are clean ones — throws that set
    /// away unwritten. So a burst accepted on a connection that has since
    /// dropped proves nothing about the broker, and only a fresh burst on the
    /// live session can put the clears back in front of it.
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
        // A record is authority over one broker's entities only. One from
        // another broker describes entities over there: acting on it here would
        // delete entities this broker was never told about — or that another
        // camon still serves — and it is no reason to announce that broker's
        // classes here either.
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
    /// taken by the request queue. The clears stay in the record as owed: the
    /// queue having taken them is not the socket having written them, and a
    /// process that dies in between must clear them again rather than believe
    /// the job done.
    fn note_burst_accepted(&mut self) {
        self.clears_queued = true;
        let owed = self.orphans.clone();
        self.write(owed);
    }

    /// A session ended, or a new one began: whatever the old one's queue was
    /// holding is unwritten and unreachable, so the clears are owed again.
    ///
    /// Called for both edges because both destroy the same evidence. It costs
    /// nothing when the clears did in fact go out before the connection
    /// dropped — the next start clears topics that hold nothing — while the
    /// opposite mistake costs an entity that never goes away.
    fn note_session_lost(&mut self) {
        self.clears_queued = false;
    }

    /// Drop the owed clears, now that a `Disconnect` has gone out. Requests are
    /// written in the order they were queued, so a disconnect on the wire means
    /// every publish queued before it — the clears among them — is on the wire
    /// too. Only ever reached from a clean shutdown; a killed camon simply
    /// clears the same topics again next start, which is a no-op on a topic
    /// that no longer holds anything.
    ///
    /// The ordering argument only holds within one session, which is what
    /// [`clears_queued`](Self::clears_queued) tracks: a disconnect written on a
    /// *later* connection than the burst says nothing at all about the burst,
    /// since the reconnect in between threw its queue away. That is a real
    /// sequence, not a hypothetical one — a broker that drops out just before
    /// shutdown has the bridge queue its `offline` marker and its `Disconnect`
    /// while down, and the poller reconnects and writes exactly those two.
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
            // Only now: an unwritten record must stay owed so the next attempt
            // — the next accepted burst, or the shutdown — tries again.
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
    /// A `ConnAck` arrived: the session is up. One per successful connect, and
    /// the bridge's only cue to republish, so these must reach it whole —
    /// neither dropped nor coalesced with a later one.
    Connected,
    /// The connection failed or dropped. The eventloop paces its own retry.
    Disconnected,
    /// The shutdown `Disconnect` reached the socket, which is also the proof
    /// that everything queued before it did — see [`shutdown_bridge`].
    DisconnectSent,
}

/// The bridge's half of the task that owns the rumqttc event loop.
///
/// The event loop lives on a task of its own for one reason: `poll()` is not
/// cancellation-safe (see the module header), so it must never be raced against
/// anything. Here it is polled in a plain loop that does nothing else, and the
/// bridge learns about the connection through [`LinkEvent`]s instead.
///
/// The channel is small and the task sends into it with `send().await`, which
/// looks like the very thing this module forbids — an await that could stop
/// `poll()`. It cannot deadlock: the bridge loop never awaits anything but its
/// own `select!`, so the channel is always drained within one turn of it, and
/// the connection edges that flow here are rate-limited by [`RECONNECT_DELAY`]
/// to a couple per five seconds against [`LINK_EVENT_CAPACITY`] slots. Waiting
/// is chosen over dropping deliberately: a lost `Connected` is a lost republish
/// burst, which leaves Home Assistant looking at `offline` on a live
/// connection.
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

    /// End the task: ask, wait briefly, abort. Only ever called once the
    /// shutdown flush is over, so an aborted `poll()` costs nothing that
    /// matters — the in-flight bookkeeping it could leak dies with the process,
    /// and the session it belongs to has already been disconnected.
    async fn stop(self) {
        let Self {
            events,
            stop,
            mut task,
        } = self;
        // Dropped first: nothing drains the channel any more, so a task parked
        // on `send` has to be woken by the send failing rather than by a
        // notification it is not listening for.
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

/// Poll the event loop forever, reporting connection edges to the bridge.
///
/// The only awaits here are `poll()` itself and the paced retry, and the retry
/// is the only one raced against anything: it is a sleep, so dropping it drops
/// nothing rumqttc is keeping track of.
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
        // The bridge is gone: there is no one left to publish, so there is
        // nothing left to poll for either.
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
    /// What the queue may still be asked to carry in images this tick. Here
    /// because it is a property of the same request queue the rest of this
    /// struct is about, and because every site that queues an image already
    /// holds a `Link` to decide whether the connection is up.
    images: ImageBudget,
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
    /// When each camera's next snapshot falls due. Absent means "now": a run
    /// that has just opened is due immediately.
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
    /// `snapshot_interval` is an operator's number and is clamped to
    /// [`MAX_SNAPSHOT_INTERVAL`] on the way in, because everything downstream
    /// of it is `now + interval`: an interval of `u64::MAX` seconds is a
    /// panic on the first tick with motion open, and it is far likelier to be
    /// a fat-fingered "effectively never" than an attack.
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

    /// Open a motion run. `false` when it was already open (a duplicate start,
    /// which must not restart the snapshot cadence).
    ///
    /// A fresh run is a new generation: decodes started for the run that just
    /// closed are still out there, and their outcomes belong to that run and
    /// not to this one.
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

    /// Note that a decode produced no frame, so the camera's tile still holds
    /// whatever it held before the attempt.
    ///
    /// The cadence paces *pictures*, not attempts: a camera whose decodes all
    /// fail would otherwise sit out a full interval between failures and
    /// publish nothing at all, so the next attempt is brought forward to
    /// [`SNAPSHOT_RETRY_DELAY`] — never later than the cadence would have had
    /// it, and never sooner than the cadence when that is the shorter of the
    /// two. Only while the run is still open: a failure reported after
    /// `MotionEnd` closed it must not put the camera back on a schedule that
    /// [`motion_end`](Self::motion_end) just took it off.
    ///
    /// Reports `true` the first time a camera fails in a row, which is what the
    /// caller says out loud — a permanently failing camera is silence on a
    /// dashboard tile, and silence is what nobody notices.
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

    /// Cameras whose next motion snapshot is due, putting each on the cadence
    /// again. A camera that just opened its run has no due time recorded and is
    /// therefore due immediately.
    ///
    /// The attempt is what is stamped here, not its outcome, which is only
    /// known once the detached decode ends; a decode that produces nothing
    /// takes the stamp back with [`note_snapshot_failed`](Self::note_snapshot_failed).
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
    /// object detection is off — no occupancy entities are then created.
    pub classes: Vec<String>,
    /// Where the announced entity set is remembered between runs, so entities
    /// a later config no longer describes can be cleared from the broker.
    /// `None` keeps no memory: nothing is read, cleared or written.
    pub entities_path: Option<PathBuf>,
    pub shutdown: Arc<AtomicBool>,
}

/// Whether the bridge's loop has nothing left to serve. Both halves are
/// required, and the second is the one that is easy to leave out.
///
/// The stop flag says a stop has *begun*. It does not say the analyzers have
/// finished: phase 2 of the drain (see [`crate::shutdown`]) lets each of them
/// keep working for up to `TAIL_DRAIN_BOUND` past the flag, and the last thing
/// one does before it exits is flush its open motion run and send the
/// `MotionEnd` that clears Home Assistant's motion sensor. A bridge that
/// stopped receiving on the flag alone stopped one tick in and dropped that
/// transition on the floor, leaving Home Assistant holding movement until camon
/// came back. The producers dropping their senders is what actually says there
/// is nothing more coming, and by the time the drain joins this task they
/// always have.
///
/// If they somehow have not — an analyzer abandoned at its own bound still
/// holds an `mqtt_tx` clone — this loop keeps running and the drain's
/// `MQTT_SHUTDOWN_TIMEOUT` aborts it, which costs the retained `offline` marker
/// and lets the LWT publish it instead. That is the same fallback an
/// unreachable broker already gets.
fn bridge_is_done(producers_gone: bool, shutdown: &AtomicBool) -> bool {
    producers_gone && shutdown.load(Ordering::Relaxed)
}

/// The bridge itself. Spawned under supervision by [`crate::app`] and joined
/// (with a timeout) during shutdown, so the retained `offline` marker gets
/// published.
pub async fn run_bridge(ctx: BridgeContext, rx: tokio::sync::mpsc::Receiver<MqttEvent>) {
    run_bridge_with(ctx, rx, Eventloop::spawn).await
}

/// [`run_bridge`], with the poller's construction handed in.
///
/// The seam exists for one test — the one that kills the poller to prove the
/// bridge notices — because the eventloop task is created in here and a test
/// otherwise has no way to reach it.
async fn run_bridge_with<F>(
    ctx: BridgeContext,
    mut rx: tokio::sync::mpsc::Receiver<MqttEvent>,
    poller: F,
) where
    F: FnOnce(rumqttc::EventLoop) -> Eventloop,
{
    let topics = Topics::new(&ctx.config);
    // Before the client: its queue has to be sized for the clears too.
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
    // The client is a handle onto a request channel and stays usable from here;
    // only the polling half moves away.
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
    // immediately and forever; the branch is disabled so it cannot spin.
    let mut producers_gone = false;
    // Likewise for the eventloop task, which only ends when the bridge does.
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
                // Nothing is polling any more, so nothing can be published
                // either. The eventloop task only ends on purpose as part of
                // this bridge's own shutdown, so reaching here with the stop
                // flag still down means it died — panicked, or was aborted by
                // something that had no business doing so.
                //
                // That is a task death and it is fatal, exactly as the policy
                // table says for `mqtt-bridge`: a broker that has gone away is
                // the *poller's* problem and it reconnects through those on its
                // own, but the poller's absence cannot be recovered from in
                // here, because the event loop moved out of this task
                // deliberately (`poll()` is not cancellation-safe) and cannot
                // be moved back. Carrying on would leave a bridge ticking for
                // ever, publishing into a queue nobody drains, with Home
                // Assistant holding whatever it last heard and camon looking
                // perfectly healthy — the very failure supervision exists to
                // end. So the bridge returns; its `FatalGuard` names it, the
                // drain runs, and the restart brings up a poller that works.
                None => {
                    eventloop_gone = true;
                    link.connected = false;
                    // Nothing is polling, so nothing queued will ever be
                    // written — the clears included.
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
                // Every producer is gone: the analyzers and detection worker
                // have exited, so shutdown is already under way. Keep serving
                // ticks until the flag confirms it rather than exiting here
                // with the availability marker still reading `online`.
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

/// `stopping` is the shutdown flag, and it means here what it means in
/// [`on_tick`]: the state transitions still go out — they are publishes onto a
/// queue that is still being drained, and the `MotionEnd` below is the one that
/// clears Home Assistant's motion sensor — but no new snapshot is started,
/// because each one forks an ffmpeg to decode a GOP and a drain is the one time
/// this process is trying to get every child it already has to exit.
///
/// Both spawn sites are guarded, not just the tick's. The `MotionEnd` that
/// arrives during phase 2 is precisely an analyzer flushing its open run on the
/// way out, so leaving this one unguarded would fork a decode per camera at the
/// worst possible moment — a snapshot of the very last GOP, which is also the
/// least interesting frame anybody will ever not look at.
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
            // The frame that opened the run is the interesting one: take it now
            // rather than waiting up to a full interval for the tick.
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
            // One last frame so the camera tile shows the end of the event
            // instead of freezing wherever the cadence happened to land.
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
                    if !link.images.take(jpeg.len()) {
                        tracing::debug!(camera = %camera_id, class = %class,
                            bytes = jpeg.len(), "image budget spent, dropping the sighting crop");
                        continue;
                    }
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

/// `stopping` is the shutdown flag: the loop goes on serving ticks through the
/// drain (see [`bridge_is_done`]), but a stopping camon does not start new
/// snapshots. Each one forks an ffmpeg to decode a GOP, and a drain is the one
/// time this process is trying to get every child it already has to exit —
/// publishing is still welcome, forking is not. Everything else here is a
/// publish onto a queue that is still being drained.
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
    // A new window for the images the rest of this tick may queue.
    link.images.refill();

    // Rebuilt from the live state on every attempt, never replayed from the
    // failed one: a retry a few seconds later must not assert a value that has
    // since changed.
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
///
/// A decode that produced nothing published nothing, so the camera's tile still
/// shows whatever it showed before — and the cadence, which exists to pace
/// pictures, must not count the attempt as one. It is brought forward instead,
/// and the first failure of a run is said out loud: a camera whose decodes all
/// fail is a tile that quietly stops moving, which is the kind of fault an
/// operator only ever finds by going to look.
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
        // The run this decode was started for has closed and the camera has
        // opened another since. Whatever this decode did or did not manage is
        // that run's business, and that run is over.
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

/// Everything a (re)connect owes the broker, in publish order: the retained
/// topics of entities the announced set dropped, cleared; then every camera's
/// discovery payloads; then every announced entity's current state; then the
/// availability marker.
///
/// Built from the announced set rather than from the config, so what a run
/// records having announced is exactly what it published — including the
/// classes carried forward through a run that detects nothing, which are
/// discovered and restated OFF rather than being left for `online` to bring
/// back with whatever the broker still held.
///
/// One list rather than four calls so the order is a value that can be
/// asserted, instead of an implication of how a `select!` arm happens to be
/// written. Every payload is retained and idempotent, so republishing the whole
/// thing on a retry costs nothing but the bytes.
fn reconnect_burst(
    topics: &Topics,
    state: &SensorState,
    announced: &EntityRecord,
    orphans: &[String],
) -> Vec<(String, Vec<u8>)> {
    let mut burst = Vec::new();
    // First: Home Assistant is told to forget these before it is told the
    // device is available again, which is what would otherwise resurrect them.
    // An empty retained payload both deletes the discovery document and clears
    // whatever state the broker was holding under it.
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
    // Last: with every state already queued ahead of it, the connection Home
    // Assistant is told about is one whose retained values are current.
    burst.push((topics.availability(), b"online".to_vec()));
    burst
}

/// Hand the burst to the request queue, reporting whether all of it fit.
///
/// The eventloop is draining the queue from its own task while this runs, so a
/// rejection does not mean every publish after it is rejected too: a slot
/// freeing up mid-burst would let a later message through and leave the tail
/// published without its head — `online` standing on top of states that were
/// dropped. It therefore stops at the first rejection and reports the burst as
/// a whole, and the caller retries all of it from the live state rather than
/// treating any of it as lost. One warn line for the burst rather than one per
/// topic, because a rejection here is the normal outcome of reconnecting after
/// an outage.
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

/// One camera's detached snapshot decode, and where that decode leaves word of
/// whether it produced anything.
///
/// The outcome has to come back somehow: the task is detached precisely so a
/// slow ffmpeg cannot delay the bridge loop, so the loop cannot await it, and
/// the cadence it belongs to lives in the loop's `SensorState`. A flag the task
/// sets and the tick reads is the whole channel — the tick already looks at
/// every handle to retire the finished ones.
struct SnapshotTask {
    handle: tokio::task::JoinHandle<()>,
    /// Set by the task when the decode produced a frame. A task that panicked
    /// or was aborted leaves it down, which reads as a failure and is one.
    decoded: Arc<AtomicBool>,
    /// The motion run this decode was started for.
    ///
    /// A decode outlives the run that asked for it: fifteen seconds are allowed
    /// for one, and a run can close and the camera open a new one inside that.
    /// Without the tag the old outcome would land on the new run — shortening a
    /// cadence that never failed, or clearing a failure the new run is still
    /// having — so an outcome whose run has moved on is dropped instead.
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
                // QoS 0: the next snapshot is at most one interval away, so a
                // lost frame costs nothing worth a retransmit, and this task
                // has no `Link` to report the rejection to. Retained so the
                // camera tile has an image right after HA subscribes.
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

    let stdin = child.stdin.take().expect("stdin piped");
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut out = Vec::with_capacity(capacity);

    let piped = async {
        // ffmpeg exits after one frame and closes stdin, so the tail of the
        // write failing with EPIPE is the normal case, not an error.
        //
        // The handle is moved in here so that finishing the write *closes* the
        // pipe. `shutdown` on a child's stdin flushes and returns; it does not
        // close the descriptor, and only the close is an EOF. A demuxer reading
        // a stream that never ends is a decode that never starts: ffmpeg probes
        // for stream information before it will emit a frame, and against a
        // segment far smaller than its probe size it simply waits for input
        // that is not coming — which is the whole decode timeout, and no
        // snapshot, for every camera, every time.
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

/// Publish the retained `offline` marker and, if the queue took that marker,
/// disconnect cleanly, then wait briefly for both packets to actually reach the
/// socket — `try_publish` only queues them, the eventloop task is what writes
/// them. That task keeps polling
/// throughout; all this does is watch its notifications for the
/// [`LinkEvent::DisconnectSent`] that says the flush is over, and stop it after.
///
/// The `Disconnect` this waits for is also the only evidence camon gets that
/// the clears at the head of the reconnect burst were written rather than
/// merely queued: requests go out in the order they were queued, so a
/// disconnect on the wire puts everything queued before it on the wire too.
/// That is where the owed clears are dropped from the record, and two separate
/// things have to be true of them first. `burst_owed` says the burst was still
/// waiting for room, in which case the clears were never queued at all;
/// `EntityMemory::clears_queued` says which *session* queued them, because the
/// ordering argument only holds inside one — hence the connection edges below
/// being applied to the record rather than merely skipped over.
async fn shutdown_bridge(
    client: AsyncClient,
    topics: &Topics,
    mut eventloop: Eventloop,
    snapshot_tasks: HashMap<String, SnapshotTask>,
    memory: &mut EntityMemory,
    burst_owed: bool,
) {
    abort_snapshots(snapshot_tasks).await;

    // Whatever is already queued was raised before the disconnect was even
    // requested and says nothing about whether it reached the socket. Not read
    // as this flush's outcome, so that an edge left over from the loop — a
    // connection error a moment ago, say — cannot be taken for it having
    // failed. It does still say that the session which took the clears is over,
    // and that much is kept.
    while let Ok(event) = eventloop.events.try_recv() {
        if event != LinkEvent::DisconnectSent {
            memory.note_session_lost();
        }
    }

    // Nothing left to retry with, and the two are not independent: a clean
    // DISCONNECT tells the broker to *drop* the LWT, so sending one without the
    // `offline` marker queued ahead of it leaves the retained availability
    // reading `online` for as long as camon stays down. The queue can take the
    // disconnect and not the publish — it is a request like any other, and the
    // eventloop is draining slots the whole time — so the disconnect is only
    // asked for once the marker is actually in the queue behind it. Rejected,
    // the connection is left to die unclean instead, which is exactly what the
    // LWT is for: the broker publishes `offline` on camon's behalf.
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

/// Cancel every in-flight snapshot decode. Aborting drops the task's future,
/// which drops its ffmpeg child and so kills it; joining afterwards is what
/// makes that ordering observable rather than a race with process exit. Bounded
/// because a task inside a blocking closure cannot be cancelled at all, and
/// shutdown must not wait on one.
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
