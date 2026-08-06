use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// The ranges the motion sliders are held to. Read from the module that owns
/// them, so the bounds this file corrects against and the bounds the detector
/// is finally given can never drift apart.
use crate::analytics::motion_settings;

const DEFAULT_CONFIG_PATH: &str = "config.toml";

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("no cameras configured")]
    NoCameras,
    #[error(
        "invalid [http] bind {value:?}: expected an IP address, e.g. \"0.0.0.0\" (all \
         interfaces) or \"127.0.0.1\" (this machine only)"
    )]
    HttpBind { value: String },
    #[error(
        "[http] token is set but empty; remove the key to run without authentication, or \
         give it a real value, e.g. `openssl rand -hex 32`"
    )]
    HttpTokenEmpty,
    #[error("a [[cameras]] entry has a blank id; every camera needs an id with something in it")]
    BlankCameraId,
    #[error(
        "camera id {id:?} has leading or trailing whitespace; it would be invisible \
         everywhere the id is shown — write it as {trimmed:?}"
    )]
    PaddedCameraId { id: String, trimmed: String },
    #[error(
        "camera id {id:?} contains {character:?}; an id is used verbatim as a directory name \
         under the storage data_dir, so it must be a single path component"
    )]
    InvalidCameraIdCharacter { id: String, character: char },
    #[error(
        "camera id {id:?} is not a usable directory name; an id is used verbatim as a \
         directory name under the storage data_dir"
    )]
    ReservedCameraId { id: String },
    #[error(
        "camera id {id:?} is used by more than one [[cameras]] entry; duplicates share one \
         storage directory and only one of them is reachable over the API"
    )]
    DuplicateCameraId { id: String },
    #[error(
        "[storage] max_event_duration_secs is 0, which means \"never chunk\" — but with \
         [analytics] disabled camon records continuously and rolls a chunk only when the cap \
         is reached, so nothing would be written until shutdown; set a real cap, e.g. 120"
    )]
    ZeroMaxEventDurationInContinuousMode,
    #[error(
        "[storage] max_event_duration_secs ({cap}) must be below [buffer] hot_duration_secs \
         ({hot}): continuous chunks are cut straight from the hot buffer, so a chunk that \
         long loses its oldest segments to eviction before the recorder rolls it. Raise \
         hot_duration_secs, or lower max_event_duration_secs — it is 120 unless you set it"
    )]
    ContinuousChunkExceedsHotBuffer { cap: u64, hot: u64 },
    #[error(
        "[storage] {key} is 0. Retention is in days, so 0 says every stored event is already \
         past it and the next hourly sweep would delete the whole archive. Set at least 1 \
         day, or turn recording off with [storage] enabled = false"
    )]
    ZeroRetentionDays { key: &'static str },
    #[error(
        "[storage] {key} is {days}, which is beyond the {max} days camon can hold: retention \
         that long is really a disk-space limit, which is what min_free_bytes is for"
    )]
    RetentionDaysTooLarge {
        key: &'static str,
        days: u64,
        max: u64,
    },
    #[error(
        "[analytics] sample_fps is 0, which reaches ffmpeg as an `fps=0` filter: not one frame \
         is ever pulled out of a motion run, so events get no thumbnails and the vision model \
         is never asked anything. Motion detection and recording carry on — they decode \
         keyframes through a different filter — so nothing at all reports this. Set at least \
         1 — it is 5 unless you set it"
    )]
    ZeroSampleFps,
    #[error(
        "[analytics.motion] {key} is {value}, which is not a real number. A value merely out \
         of range is clamped into it, but this one cannot be: no comparison against it is \
         ever true, so the detector would run on every frame and find motion in none of them. \
         Write a number"
    )]
    NonFiniteMotionDefault { key: &'static str, value: f64 },
    #[error(
        "[analytics.object_detection] confidence_threshold is {value}, which is not a number. \
         The filter is `confidence < threshold`, and that is false for every detection when \
         the threshold is not one — so instead of a stricter detector you get no filtering at \
         all, and every sighting of an allowed class is kept however unsure the model was. \
         Write a confidence between 0.0 and 1.0; a real number outside that is clamped into it"
    )]
    ConfidenceThresholdNotANumber { value: f32 },
    #[error(
        "[buffer] hot_duration_secs is 0, so every segment is evicted by the same push that \
         added it: the analyzer is handed nothing to look at, no event can be cut from a \
         buffer holding no footage, and nothing says why. Set the seconds of video to keep in \
         memory — it is 600 unless you set it"
    )]
    ZeroHotDuration,
    #[error(
        "[{section}] {key} is {secs} seconds, beyond the {max} camon can hold: every one of \
         these durations is multiplied out to nanoseconds, which wraps past that into a very \
         short duration — an absurd setting would quietly become no recording rather than \
         obviously too much of one"
    )]
    DurationTooLarge {
        section: &'static str,
        key: &'static str,
        secs: u64,
        max: u64,
    },
    #[error(
        "[analytics.object_detection] classes is empty, so nothing could ever be detected. \
         Remove the key to detect the defaults ({defaults}), or say so outright with \
         enabled = false"
    )]
    EmptyObjectClasses { defaults: String },
    #[error(
        "[analytics.object_detection] classes contains a blank entry; it names no object, so \
         the model can never return it, and it would still get Home Assistant entities of \
         its own"
    )]
    BlankObjectClass,
    #[error(
        "camera id {id:?} contains {character:?}, which is an MQTT topic wildcard; \
         rename the camera or disable [mqtt]"
    )]
    MqttWildcardCameraId { id: String, character: char },
    #[error(
        "object detection class {class:?} contains {character:?}, which is an MQTT topic \
         wildcard; its occupancy topic could never be published — rename the class or \
         disable [mqtt]"
    )]
    MqttWildcardClass { class: String, character: char },
    #[error(
        "camera ids {first:?} and {second:?} both normalize to MQTT slug {slug:?}; \
         rename one so Home Assistant entities don't collide"
    )]
    MqttSlugCollision {
        first: String,
        second: String,
        slug: String,
    },
}

/// A single `--set <dotted.path>=<value>` startup override, applied to the
/// parsed config tree before it is deserialized into [`Config`]. Overrides win
/// over the file's values and can create missing intermediate tables. The
/// value's TOML type comes from the setting it names rather than from how the
/// text looks — see [`Override::reading_against`].
#[derive(Debug, Clone)]
pub struct Override {
    path: Vec<String>,
    raw: String,
}

impl Override {
    /// Parse a `dotted.path=value` argument. Splits on the first `=`; the value
    /// stays raw text until it is typed against the config schema. An argument
    /// without `=`, with an empty key, or with an empty path segment is
    /// rejected.
    pub fn parse(arg: &str) -> Result<Self, String> {
        let (path, raw) = arg
            .split_once('=')
            .ok_or_else(|| format!("invalid --set {arg:?}: expected <dotted.path>=<value>"))?;
        if path.is_empty() {
            return Err(format!("invalid --set {arg:?}: empty key before '='"));
        }
        let segments: Vec<String> = path.split('.').map(str::to_string).collect();
        if segments.iter().any(String::is_empty) {
            return Err(format!("invalid --set {arg:?}: empty path segment"));
        }
        Ok(Self {
            path: segments,
            raw: raw.to_string(),
        })
    }

    /// The value read by shape alone: bool, then integer, then float, else a
    /// string. Where the schema has no answer this is what gets inserted, so
    /// the load reports the real problem instead of a type camon invented.
    fn guess(&self) -> toml::Value {
        parse_scalar(&self.raw)
    }

    /// The raw text as-is, the other reading every value has.
    fn text(&self) -> toml::Value {
        toml::Value::String(self.raw.clone())
    }

    /// The TOML scalar this override should insert, or `None` when `base`
    /// gives no answer.
    ///
    /// The type has to come from the target field. `--set http.port=8080` must
    /// become an integer while `--set mqtt.password=8080` must stay a string,
    /// and nothing about the text itself tells them apart — reading by shape
    /// alone mistyped every all-digit secret. A hand-kept list of string keys
    /// would instead mistype the next key someone adds, silently and only for
    /// whoever overrides it.
    ///
    /// A reading is only ever taken from a [`Config`] that deserialized, so a
    /// failure — here or anywhere else in `base` — is never mistaken for an
    /// answer about this key. Callers pass a `base` that isolates the question
    /// as far as it can be isolated; see [`Config::load_from_with_overrides`].
    fn reading_against(&self, base: &toml::Value) -> Option<toml::Value> {
        let guess = self.guess();
        // A string reading is the last resort anyway: nothing else applies, so
        // there is no question to answer.
        if matches!(guess, toml::Value::String(_)) {
            return Some(guess);
        }
        if self.accepted_by(base, &guess) {
            return Some(guess);
        }
        let text = self.text();
        self.accepted_by(base, &text).then_some(text)
    }

    /// Whether [`Config`] deserializes from `base` once `value` is set at this
    /// path.
    fn accepted_by(&self, base: &toml::Value, value: &toml::Value) -> bool {
        let mut probe = base.clone();
        insert_at(&mut probe, &self.path, value.clone());
        probe.try_into::<Config>().is_ok()
    }
}

/// An empty tree: the base that isolates an override from everything else.
fn empty_tree() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

/// Insert `value` at a dotted path, creating (or replacing non-table)
/// intermediate tables as needed.
fn insert_at(root: &mut toml::Value, path: &[String], value: toml::Value) {
    let (last, parents) = path
        .split_last()
        .expect("Override::parse guarantees a non-empty path");
    let mut current = root;
    for segment in parents {
        if !current.is_table() {
            *current = toml::Value::Table(toml::map::Map::new());
        }
        current = current
            .as_table_mut()
            .expect("just ensured a table")
            .entry(segment.clone())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    }
    if !current.is_table() {
        *current = toml::Value::Table(toml::map::Map::new());
    }
    current
        .as_table_mut()
        .expect("just ensured a table")
        .insert(last.clone(), value);
}

/// Read a raw `--set` value by shape alone: bool, then integer, then float,
/// otherwise a string. Only a starting point — [`Override::resolve`] decides.
fn parse_scalar(raw: &str) -> toml::Value {
    if let Ok(b) = raw.parse::<bool>() {
        toml::Value::Boolean(b)
    } else if let Ok(i) = raw.parse::<i64>() {
        toml::Value::Integer(i)
    } else if let Ok(f) = raw.parse::<f64>() {
        toml::Value::Float(f)
    } else {
        toml::Value::String(raw.to_string())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    pub id: String,
    pub url: String,
}

impl CameraConfig {
    /// The camera URL with any userinfo password masked, safe for logs.
    pub fn redacted_url(&self) -> String {
        match url_password_range(&self.url) {
            Some(range) => {
                let mut url = self.url.clone();
                url.replace_range(range, "****");
                url
            }
            None => self.url.clone(),
        }
    }
}

/// The password embedded in a URL's userinfo (`scheme://user:pass@host/...`),
/// if any. Used to keep credentials out of logs.
pub fn url_password(url: &str) -> Option<&str> {
    url_password_range(url).map(|range| &url[range])
}

fn url_password_range(url: &str) -> Option<std::ops::Range<usize>> {
    let userinfo_start = url.find("://")? + 3;
    let rest = &url[userinfo_start..];
    let authority = &rest[..rest.find('/').unwrap_or(rest.len())];
    let userinfo = &authority[..authority.rfind('@')?];
    let password_offset = userinfo.find(':')? + 1;
    (password_offset < userinfo.len())
        .then(|| userinfo_start + password_offset..userinfo_start + userinfo.len())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferConfig {
    #[serde(default = "default_hot_duration")]
    pub hot_duration_secs: u64,
}

fn default_hot_duration() -> u64 {
    600
}

fn default_http_port() -> u16 {
    8080
}

/// All interfaces. The Home Assistant add-on is reached through ingress over
/// the container network, and standalone installs are reached over the LAN, so
/// loopback would break both out of the box.
fn default_http_bind() -> String {
    "0.0.0.0".to_string()
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            hot_duration_secs: default_hot_duration(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpConfig {
    #[serde(default = "default_http_port")]
    pub port: u16,
    /// Address the listener binds to. Validated as an [`IpAddr`] at load time.
    #[serde(default = "default_http_bind")]
    pub bind: String,
    /// Shared secret required on every `/api` request, reads included. `None`
    /// (the default) does *not* mean "open": on a non-loopback bind camon
    /// generates a token of its own and requires it for anything that changes
    /// state. See [`crate::api::ApiAuth`] for the full table.
    #[serde(default)]
    pub token: Option<String>,
    /// Declares that something in front of camon is the authentication boundary
    /// (Home Assistant ingress, an authenticating reverse proxy). Camon then
    /// asks for nothing itself: no generated token, no startup warning. The
    /// add-on forces this on, because ingress reaches camon over the container
    /// network and could never present a token camon invented.
    #[serde(default)]
    pub allow_open: bool,
}

impl HttpConfig {
    /// The parsed [`bind`](Self::bind) address. `Config::validate` rejects an
    /// unparseable value at load time, so this cannot fail on a loaded config.
    pub fn bind_addr(&self) -> IpAddr {
        self.bind
            .parse()
            .expect("bind address validated at config load")
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            port: default_http_port(),
            bind: default_http_bind(),
            token: None,
            allow_open: false,
        }
    }
}

fn default_sample_fps() -> u32 {
    5
}

fn default_confidence_threshold() -> f32 {
    0.5
}

fn default_classes() -> Vec<String> {
    vec![
        "person".to_string(),
        "car".to_string(),
        "truck".to_string(),
        "dog".to_string(),
        "cat".to_string(),
    ]
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_model() -> String {
    "gemma4:e4b".to_string()
}

/// Per-request timeout for Ollama calls. Warm-inference latency on modest
/// GPUs runs up to ~50s per frame (measured 2026-07-23), so 90s gives real
/// headroom; a timeout only costs the object upgrade of an event, never the
/// footage.
fn default_ollama_timeout_secs() -> u64 {
    90
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaServerConfig {
    #[serde(default = "default_ollama_url")]
    pub url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
    /// Request timeout in seconds, applied to primary and fallback alike.
    #[serde(default = "default_ollama_timeout_secs")]
    pub timeout_secs: u64,
    pub fallback: Option<OllamaServerConfig>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            url: default_ollama_url(),
            model: default_ollama_model(),
            timeout_secs: default_ollama_timeout_secs(),
            fallback: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectDetectionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_classes")]
    pub classes: Vec<String>,
    #[serde(default)]
    pub ollama: OllamaConfig,
}

impl Default for ObjectDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            confidence_threshold: default_confidence_threshold(),
            classes: default_classes(),
            ollama: OllamaConfig::default(),
        }
    }
}

// Deterministic motion-detection defaults. These seed a camera's
// motion_settings.json the first time it is seen; thereafter the per-camera
// file (edited live from the web UI) wins. Ranges must match the clamps in
// `analytics::motion_settings`.
fn default_motion_var_threshold() -> f64 {
    16.0 // sensitivity; range 4..=96, higher = less sensitive
}

fn default_motion_min_contour_area() -> f64 {
    200.0 // min object size in foreground pixels; range 50..=2000
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionConfig {
    /// MOG2 var_threshold. Higher = less sensitive.
    #[serde(default = "default_motion_var_threshold")]
    pub var_threshold: f64,
    /// Minimum connected-component area (foreground pixels) to count as motion.
    #[serde(default = "default_motion_min_contour_area")]
    pub min_contour_area: f64,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            var_threshold: default_motion_var_threshold(),
            min_contour_area: default_motion_min_contour_area(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyticsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_sample_fps")]
    pub sample_fps: u32,
    /// Global default motion-detection settings (deterministic, user-tunable
    /// per camera at runtime).
    #[serde(default)]
    pub motion: MotionConfig,
    #[serde(default)]
    pub object_detection: ObjectDetectionConfig,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_fps: default_sample_fps(),
            motion: MotionConfig::default(),
            object_detection: ObjectDetectionConfig::default(),
        }
    }
}

fn default_warm_enabled() -> bool {
    true
}

fn default_warm_data_dir() -> String {
    "/var/camon/storage".to_string()
}

fn default_warm_pre_padding_secs() -> u64 {
    5
}

fn default_warm_post_padding_secs() -> u64 {
    10
}

fn default_max_event_duration_secs() -> u64 {
    120
}

fn default_movement_retention_days() -> u64 {
    2
}

fn default_object_retention_days() -> u64 {
    14
}

fn default_continuous_retention_days() -> u64 {
    1
}

/// Upper bound on any `*_retention_days`. Retention is held in nanoseconds
/// (`days * 86400 * 1e9`), which overflows `u64` around 213000 days and wraps
/// into a *short* retention — the one failure mode that deletes footage instead
/// of keeping it. 10 years is far past any real archive and leaves the product
/// nowhere near the wrap.
const MAX_RETENTION_DAYS: u64 = 3650;

/// Upper bound on any `*_secs` duration in the config, and the same ten years
/// [`MAX_RETENTION_DAYS`] allows. See [`Config::validate_durations`] for the
/// arithmetic it protects.
const MAX_DURATION_SECS: u64 = MAX_RETENTION_DAYS * 86_400;

/// 2 GiB. Roughly an hour of footage at a typical 4 Mbps camera bitrate —
/// enough slack for the hourly retention prune to catch up before the disk
/// actually fills — while also keeping the filesystem out of the near-full
/// regime where allocation slows down and other services start failing.
fn default_min_free_bytes() -> u64 {
    2 * 1024 * 1024 * 1024
}

fn default_stathost_enabled() -> bool {
    true
}

/// Remote "stathost" warm-storage backend (github.com/nsg/stathost). Presence
/// of a `[storage.stathost]` section — with `enabled` left at its default of
/// `true` — switches the warm backend from local disk to this static file host.
/// Analytics, motion settings, and the hot buffer always stay local.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StathostConfig {
    /// Base URL of the host, e.g. `https://files.example.com`.
    pub url: String,
    /// Bucket that events are written into.
    pub bucket: String,
    /// Per-bucket bearer token, sent as `Authorization: Bearer <token>`.
    pub token: String,
    /// Client-side storage budget in bytes. The client can't see the server's
    /// disk, so retention-by-space becomes a budget: when tracked usage exceeds
    /// it, the oldest events are pruned (continuous → movements → objects).
    /// 0 (the default) means unlimited — rely on time-based retention only.
    #[serde(default)]
    pub max_stored_bytes: u64,
    /// Set to `false` to keep the section but fall back to local-disk storage.
    #[serde(default = "default_stathost_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmConfig {
    #[serde(default = "default_warm_enabled")]
    pub enabled: bool,
    #[serde(default = "default_warm_data_dir")]
    pub data_dir: String,
    /// When present (and `enabled`), warm events go to this remote host instead
    /// of the local `data_dir`. See [`StathostConfig`].
    #[serde(default)]
    pub stathost: Option<StathostConfig>,
    #[serde(default = "default_warm_pre_padding_secs")]
    pub pre_padding_secs: u64,
    #[serde(default = "default_warm_post_padding_secs")]
    pub post_padding_secs: u64,
    /// Cap on the wall-clock length of a single event. A run exceeding this is
    /// split into chained, independently playable chunks. 0 disables chunking,
    /// which continuous recording cannot do — there it is the only thing that
    /// rolls a chunk. A recording has to fit in the hot buffer: fatal in
    /// continuous mode, a warning in event mode, where
    /// [`pre_padding_secs`](Self::pre_padding_secs) counts too.
    #[serde(default = "default_max_event_duration_secs")]
    pub max_event_duration_secs: u64,
    #[serde(default = "default_movement_retention_days")]
    pub movement_retention_days: u64,
    #[serde(default = "default_object_retention_days")]
    pub object_retention_days: u64,
    /// Retention for continuous-recording chunks (analytics disabled). Kept
    /// short by default: continuous at ~4 Mbps is roughly 43 GB/day/camera.
    #[serde(default = "default_continuous_retention_days")]
    pub continuous_retention_days: u64,
    /// Low-space guard: before each event write, if the storage filesystem
    /// has less than this many bytes free, the oldest events are
    /// emergency-pruned (continuous → movements → objects) until space
    /// recovers. 0 disables the guard.
    #[serde(default = "default_min_free_bytes")]
    pub min_free_bytes: u64,
}

impl Default for WarmConfig {
    fn default() -> Self {
        Self {
            enabled: default_warm_enabled(),
            data_dir: default_warm_data_dir(),
            stathost: None,
            pre_padding_secs: default_warm_pre_padding_secs(),
            post_padding_secs: default_warm_post_padding_secs(),
            max_event_duration_secs: default_max_event_duration_secs(),
            movement_retention_days: default_movement_retention_days(),
            object_retention_days: default_object_retention_days(),
            continuous_retention_days: default_continuous_retention_days(),
            min_free_bytes: default_min_free_bytes(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateConfig {
    #[serde(default = "default_update_enabled")]
    pub enabled: bool,
}

/// Opt-in: an update is only checked against the sha256sums.txt published
/// beside it in the same GitHub release, which protects against a corrupt
/// download but not against a tampered release, and the installed service runs
/// as root.
fn default_update_enabled() -> bool {
    false
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            enabled: default_update_enabled(),
        }
    }
}

fn default_mqtt_host() -> String {
    "localhost".to_string()
}

fn default_mqtt_port() -> u16 {
    1883
}

fn default_mqtt_topic_prefix() -> String {
    "camon".to_string()
}

/// Home Assistant's own default discovery prefix. Changing it here must match
/// the `discovery_prefix` of HA's MQTT integration or nothing is discovered.
fn default_mqtt_discovery_prefix() -> String {
    "homeassistant".to_string()
}

/// Snapshot cadence while motion is active. Snapshots are motion-gated by
/// design (see `crate::mqtt`), so this only ever costs decode work during a
/// run — 5s is a compromise between a responsive HA camera tile and the
/// per-frame ffmpeg decode.
fn default_mqtt_snapshot_interval_secs() -> u64 {
    5
}

/// How long an occupancy sensor stays ON after the last sighting of its class.
/// The vision model only sees frames during motion runs, so without a hold-off
/// a parked person would flap OFF between runs; a minute reads as "still here"
/// for automations without pinning the sensor ON indefinitely.
fn default_mqtt_occupancy_hold_secs() -> u64 {
    60
}

/// MQTT bridge to Home Assistant. Off by default: camon is fully usable
/// standalone, and an unreachable broker should never be a startup concern for
/// users who don't run HA.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MqttConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_mqtt_host")]
    pub host: String,
    #[serde(default = "default_mqtt_port")]
    pub port: u16,
    /// Broker credentials. Both must be set for authentication to be attempted.
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    /// Root of camon's own state topics (`<prefix>/<camera>/motion`, ...).
    #[serde(default = "default_mqtt_topic_prefix")]
    pub topic_prefix: String,
    /// Root of the Home Assistant discovery topics.
    #[serde(default = "default_mqtt_discovery_prefix")]
    pub discovery_prefix: String,
    #[serde(default = "default_mqtt_snapshot_interval_secs")]
    pub snapshot_interval_secs: u64,
    #[serde(default = "default_mqtt_occupancy_hold_secs")]
    pub occupancy_hold_secs: u64,
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: default_mqtt_host(),
            port: default_mqtt_port(),
            username: None,
            password: None,
            topic_prefix: default_mqtt_topic_prefix(),
            discovery_prefix: default_mqtt_discovery_prefix(),
            snapshot_interval_secs: default_mqtt_snapshot_interval_secs(),
            occupancy_hold_secs: default_mqtt_occupancy_hold_secs(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub update: UpdateConfig,
    #[serde(default)]
    pub buffer: BufferConfig,
    #[serde(default)]
    pub http: HttpConfig,
    #[serde(default)]
    pub analytics: AnalyticsConfig,
    #[serde(default)]
    pub storage: WarmConfig,
    #[serde(default)]
    pub mqtt: MqttConfig,
    #[serde(default)]
    pub cameras: Vec<CameraConfig>,
    /// The file this config was read from, kept so camon can put things beside
    /// it — today only the API token it generates for an otherwise-open
    /// deployment, which belongs where the operator already looks for camon's
    /// settings. `None` for a config that did not come from a file (tests, and
    /// `toml::from_str`), which costs that config a *persisted* token, never
    /// the protection itself.
    ///
    /// Not a setting: `deny_unknown_fields` plus `serde(skip)` means a
    /// `source_path` key in the TOML is refused like any other typo.
    #[serde(skip)]
    source_path: Option<PathBuf>,
}

/// Where a generated API token is kept, relative to the config file that did
/// not name one. See [`crate::api::ApiAuth`].
const API_TOKEN_FILE: &str = "api-token";

impl Config {
    /// The file a generated API token is read from and written to: beside the
    /// config file, so `/etc/camon/config.toml` puts it at
    /// `/etc/camon/api-token`. `None` when the config did not come from a file
    /// and there is therefore nowhere obvious to keep it.
    pub fn token_file_path(&self) -> Option<PathBuf> {
        let source = self.source_path.as_ref()?;
        // A bare `config.toml` has an empty parent, which joins to a plain
        // relative name — resolved against the working directory the service
        // unit sets, exactly as the config path itself was.
        Some(
            source
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .join(API_TOKEN_FILE),
        )
    }

    /// Load from the default `config.toml` in the current working directory.
    pub fn load(overrides: &[Override]) -> Result<Self, ConfigError> {
        Self::load_from_with_overrides(DEFAULT_CONFIG_PATH, overrides)
    }

    /// Load from an explicit TOML path, applying each `--set` override into the
    /// parsed value tree before deserializing. Overrides win over file values.
    ///
    /// An override's TOML type is read from the setting it names, in two
    /// passes that keep the answer independent of everything else — of the
    /// other overrides, of the order they were given in, and of any unrelated
    /// defect in the file:
    ///
    /// 1. Against an empty tree, so the only thing the schema can complain
    ///    about is the key under test. This answers for every setting whose
    ///    enclosing tables are all `#[serde(default)]` — which is all of them
    ///    but `[storage.stathost]`, whose `url`/`bucket`/`token` are required.
    /// 2. For the leftovers, against the whole config once every override is
    ///    in place and retired keys are gone, so the required siblings exist.
    ///    That tree is built from pass-1 readings alone, so it does not depend
    ///    on the order the overrides were given in either.
    ///
    /// A reading is only ever taken from a `Config` that deserialized, so a
    /// failure elsewhere can never be mistaken for an answer: it leaves the
    /// plain reading in place, and the load reports the real problem.
    pub fn load_from_with_overrides<P: AsRef<Path>>(
        path: P,
        overrides: &[Override],
    ) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let mut value: toml::Value = toml::from_str(&content)?;

        let bare = empty_tree();
        let mut undecided = Vec::new();
        for (index, ov) in overrides.iter().enumerate() {
            match ov.reading_against(&bare) {
                Some(reading) => insert_at(&mut value, &ov.path, reading),
                None => {
                    insert_at(&mut value, &ov.path, ov.guess());
                    undecided.push(index);
                }
            }
        }

        for (key, advice) in strip_retired_keys(&mut value) {
            tracing::warn!(key = %key, "ignoring retired config key: {advice}");
        }

        if !undecided.is_empty() {
            // A key can only be judged against a tree the rest of which
            // parses, so the leftovers get two views of the finished config:
            // one holding them at their plain reading, one holding all of them
            // as text — with several numeric-looking secrets in the same
            // table, the first view never parses.
            let plain = value.clone();
            let mut as_text = value.clone();
            for &index in &undecided {
                let ov = &overrides[index];
                insert_at(&mut as_text, &ov.path, ov.text());
            }
            for index in undecided {
                let ov = &overrides[index];
                if let Some(reading) = ov
                    .reading_against(&plain)
                    .or_else(|| ov.reading_against(&as_text))
                {
                    insert_at(&mut value, &ov.path, reading);
                }
            }
        }

        let mut config: Config = value.try_into()?;
        config.source_path = Some(path.to_path_buf());
        config.normalize();
        // Corrections before checks, so a value `repair` has already put right
        // is never also refused, and the operator gets one line about it.
        config.repair();
        config.validate()?;
        Ok(config)
    }

    /// Canonicalize values that several consumers have to agree on. Object
    /// classes are matched case-insensitively by the Ollama client but compared
    /// verbatim by the MQTT bridge, so `classes = ["Person"]` would otherwise
    /// yield an occupancy sensor that can never turn on. Deduplicating is part
    /// of the same fix: once folded, `["Person", "person"]` would produce two
    /// discovery payloads sharing one unique id. Surrounding whitespace goes
    /// for the same reason — `" person"` reaches the topic verbatim but comes
    /// back from the model trimmed.
    fn normalize(&mut self) {
        let classes = &mut self.analytics.object_detection.classes;
        for class in classes.iter_mut() {
            *class = class.trim().to_lowercase();
        }
        let mut seen = HashSet::new();
        classes.retain(|class| seen.insert(class.clone()));
    }

    /// Post-parse checks that TOML deserialization can't express. Failing here
    /// is deliberate: a bad camera id or an unrollable event cap costs silently
    /// lost footage, which is worse than refusing to start.
    fn validate(&self) -> Result<(), ConfigError> {
        if self.cameras.is_empty() {
            return Err(ConfigError::NoCameras);
        }

        if self.http.bind.parse::<IpAddr>().is_err() {
            return Err(ConfigError::HttpBind {
                value: self.http.bind.clone(),
            });
        }

        // An empty token is worse than no token: it silences the open-API
        // warning while `?token=` or a bare `Bearer ` satisfies the check.
        if self
            .http
            .token
            .as_ref()
            .is_some_and(|t| t.trim().is_empty())
        {
            return Err(ConfigError::HttpTokenEmpty);
        }

        // Before the storage rules that compare these durations against each
        // other, so a hot buffer of 0 is reported as itself rather than as the
        // chunk that will not fit in it.
        self.validate_durations()?;

        // The cap is only read by the warm writer and the continuous recorder,
        // neither of which is spawned with storage off.
        if self.storage.enabled {
            // Retention has no "off" value: 0 does not mean "keep forever", it
            // expires everything ever recorded.
            for (key, days) in [
                (
                    "movement_retention_days",
                    self.storage.movement_retention_days,
                ),
                ("object_retention_days", self.storage.object_retention_days),
                (
                    "continuous_retention_days",
                    self.storage.continuous_retention_days,
                ),
            ] {
                if days == 0 {
                    return Err(ConfigError::ZeroRetentionDays { key });
                }
                if days > MAX_RETENTION_DAYS {
                    return Err(ConfigError::RetentionDaysTooLarge {
                        key,
                        days,
                        max: MAX_RETENTION_DAYS,
                    });
                }
            }

            let cap = self.storage.max_event_duration_secs;
            // Continuous recording (storage on, analytics off) has no motion
            // run to close a chunk, so the cap is the only thing that rolls
            // one. In event mode 0 is a real setting: don't chunk, and let
            // motion end close the event.
            //
            // Continuous mode is settled here, before any event-mode rule is
            // considered, so a config that is wrong in both readings is always
            // reported as the mode it will actually run in.
            if !self.analytics.enabled {
                if cap == 0 {
                    return Err(ConfigError::ZeroMaxEventDurationInContinuousMode);
                }
                // Continuous chunks are cut straight from the buffer with no
                // padding at all, so the cap is the whole span. Fatal because
                // `plan_continuous_roll` can then never fire: nothing at all
                // is written, rather than something imperfect.
                if cap >= self.buffer.hot_duration_secs {
                    return Err(ConfigError::ContinuousChunkExceedsHotBuffer {
                        cap,
                        hot: self.buffer.hot_duration_secs,
                    });
                }
            } else {
                // Event mode. Two different losses can be read off the config
                // here, they are caused by different things, and a config can
                // hold both — so each is reported on its own terms rather than
                // one standing in for the other.
                let post = self.storage.post_padding_secs;
                let hot = self.buffer.hot_duration_secs;
                let evicted_head = crate::buffer::warm::EVICTED_HEAD_WARNING;

                // What closes a run is a quiet stretch outlasting
                // post_padding_secs, measured on the clock — the tracker never
                // consults the buffer. So the run does close, but if the
                // padding is as wide as the buffer, everything the run was made
                // of has been evicted by then. Below this threshold the same
                // loss is possible and depends on the scene
                // (motion + post_padding must fit), so only this much can be
                // said from the file alone.
                if post >= hot {
                    if cap == 0 {
                        // Nothing else can close a run, so this is every event
                        // the camera will ever write.
                        tracing::warn!(
                            post_padding_secs = post,
                            hot_duration_secs = hot,
                            "[storage] post_padding_secs ({post}) is not shorter than [buffer] \
                             hot_duration_secs ({hot}), and max_event_duration_secs is 0: the \
                             quiet window is the only thing that closes a run, and by the time \
                             it has elapsed the run's own footage has been evicted, so every \
                             event will hold nothing but the padding that outlived it \
                             (\"{evicted_head}\" at runtime). Lower post_padding_secs — it is \
                             10 unless you set it"
                        );
                    } else if cap < hot {
                        // A chunk closes at whichever comes first, and both are
                        // measured from the chunk's own start: the cap at `cap`,
                        // the quiet window at `e + post` where `e` is the last
                        // motion inside the chunk (`RunTracker::observe` tests
                        // the window with `>` and the cap with `>=`, so an exact
                        // tie goes to the cap). With `cap < hot`
                        // the cap is always the earlier of the two, so every
                        // chunk holding motion is assembled less than a buffer's
                        // worth of time after it opened, with its footage
                        // resident. Nothing is lost *here* — said that way and
                        // not as "no footage is lost", because the span rule
                        // below may be reporting a loss of its own in the very
                        // next line, by a mechanism this one says nothing about.
                        //
                        // The cost is the tail. The quiet window still has to
                        // elapse before the run ends, and the cap keeps slicing
                        // chunks the whole way; how many of them hold nothing
                        // but padding depends on where motion stopped relative
                        // to the next cap boundary — anywhere from none to the
                        // whole window, hence "up to".
                        tracing::warn!(
                            post_padding_secs = post,
                            hot_duration_secs = hot,
                            cap,
                            "[storage] post_padding_secs ({post}) is not shorter than [buffer] \
                             hot_duration_secs ({hot}): the {cap}s cap closes every chunk that \
                             holds motion well inside the buffer, so nothing is lost to the \
                             quiet window — but each motion run then ends in up to ~{post}s of \
                             padding-only chunks while that window elapses. Lower \
                             post_padding_secs — it is 10 unless you set it"
                        );
                    } else {
                        // `cap >= hot` is legal in event mode (only continuous
                        // recording refuses it), and here it costs footage. A
                        // chunk that holds motion closes at
                        // `min(cap, e + post)`, and with both `cap` and `post`
                        // at least `hot` that is at least `hot` however the race
                        // goes — so by the time the chunk is assembled its
                        // oldest segments have been evicted. The run's first
                        // chunk opens *on* motion, so what it loses is motion;
                        // a later chunk opens whenever the previous one rolled,
                        // so how much of its motion survives depends on where in
                        // the chunk that motion falls.
                        //
                        // The span rule below always fires too (`cap + pre` is
                        // at least `cap`, so at least `hot`), and the two do not
                        // contradict: this is the quiet window beating the cap,
                        // that is the cap's own span not fitting the buffer.
                        // Both are true, both name their own mechanism, and they
                        // are fixed by different numbers.
                        tracing::warn!(
                            post_padding_secs = post,
                            hot_duration_secs = hot,
                            cap,
                            "[storage] post_padding_secs ({post}) is not shorter than [buffer] \
                             hot_duration_secs ({hot}), and neither is the {cap}s cap: whichever \
                             of them closes a chunk that holds motion, that chunk is at least \
                             {hot}s old by then, so its oldest footage has already been evicted \
                             (\"{evicted_head}\" at runtime; a follow-on holding only padding \
                             can close young and lose nothing). The chunk a run opens with loses the \
                             motion it opened on; how much later chunks lose depends on where \
                             their motion falls. Bring max_event_duration_secs under \
                             hot_duration_secs, and lower post_padding_secs — it is 10 unless \
                             you set it"
                        );
                    }
                }

                // A different mechanism with a different cost: here the chunk
                // closes on time, but the span it asks the buffer for — the cap
                // plus the pre-padding reach-back — is wider than the buffer
                // holds, so a long event loses its opening seconds. It can hold
                // at the same time as the relation above, and then both are
                // true and both are said.
                //
                // A warning, not an error: unlike continuous mode this still
                // records. Refusing to boot over it would be the worse trade,
                // since the cap is 120 by default (so it may never have been
                // written), trimming hot_duration_secs is exactly what a
                // RAM-pressured box would do, and config load precedes the
                // updater, so an auto-updating install would stay down until
                // someone read the log.
                if cap != 0 {
                    if let Some(total) = event_span_overrun(cap, self.storage.pre_padding_secs, hot)
                    {
                        tracing::warn!(
                            cap,
                            pre_padding_secs = self.storage.pre_padding_secs,
                            total,
                            hot_duration_secs = hot,
                            "[storage] max_event_duration_secs ({cap}) plus pre_padding_secs \
                             ({}) is {total}s, which does not fit in [buffer] \
                             hot_duration_secs ({hot}): events running longer than about {}s \
                             will lose their opening seconds to eviction. Raise \
                             hot_duration_secs, or lower max_event_duration_secs — it is 120 \
                             unless you set it",
                            self.storage.pre_padding_secs,
                            hot.saturating_sub(self.storage.pre_padding_secs),
                        );
                    }
                }
            }
        }

        self.validate_analytics()?;

        // An empty allowlist reads as either "detect the defaults" or "detect
        // nothing" depending on who is asked, and both readings are already
        // spelled out unambiguously elsewhere: omit the key, or set
        // enabled = false. Rejecting it keeps the detector and the MQTT bridge
        // from ever disagreeing about what is being looked for.
        if self.analytics.enabled && self.analytics.object_detection.enabled {
            let classes = &self.analytics.object_detection.classes;
            if classes.is_empty() {
                return Err(ConfigError::EmptyObjectClasses {
                    defaults: default_classes().join(", "),
                });
            }
            // Blank after normalize()'s trim: an empty list one entry down.
            if classes.iter().any(String::is_empty) {
                return Err(ConfigError::BlankObjectClass);
            }
        }

        let mut seen: HashSet<&str> = HashSet::new();
        for camera in &self.cameras {
            validate_camera_id(&camera.id)?;
            if !seen.insert(camera.id.as_str()) {
                return Err(ConfigError::DuplicateCameraId {
                    id: camera.id.clone(),
                });
            }
        }

        // Wildcards and slug collisions are purely MQTT concerns: both shapes
        // are legal ids and legal directory names, so without the bridge there
        // is nothing to break.
        if !self.mqtt.enabled {
            return Ok(());
        }
        let mut slugs: HashMap<String, &str> = HashMap::new();
        for camera in &self.cameras {
            if let Some(character) = camera.id.chars().find(|c| matches!(c, '+' | '#')) {
                return Err(ConfigError::MqttWildcardCameraId {
                    id: camera.id.clone(),
                    character,
                });
            }
            let slug = crate::mqtt::slugify(&camera.id);
            if let Some(first) = slugs.insert(slug.clone(), &camera.id) {
                return Err(ConfigError::MqttSlugCollision {
                    first: first.to_string(),
                    second: camera.id.clone(),
                    slug,
                });
            }
        }

        // Classes reach the topic verbatim too, so the same wildcard rules
        // apply: an occupancy topic holding one is rejected by the client on
        // every attempt, and the bridge would retry it for ever.
        for class in &self.analytics.object_detection.classes {
            if let Some(character) = class.chars().find(|c| matches!(c, '+' | '#')) {
                return Err(ConfigError::MqttWildcardClass {
                    class: class.clone(),
                    character,
                });
            }
        }

        Ok(())
    }

    /// Analytics numbers that a value can switch *off* rather than tune, held
    /// to the line this file draws between refusing to start and correcting a
    /// value: **fatal only where the loss would be silent.** Where the run
    /// already complains about the consequence, or the value has a reading
    /// somebody could have meant, [`repair`](Self::repair) fixes it and warns.
    ///
    /// Silence is what makes the difference worth a refusal, because config
    /// load runs before the self-updater: an install that auto-updates into a
    /// stricter camon and then refuses to start cannot update out of it again,
    /// which is why retired keys are dropped rather than rejected. That price
    /// is only worth paying against a failure nobody would otherwise see.
    ///
    /// - `sample_fps = 0` becomes an `fps=0` filter on the crop decoder, and
    ///   nothing else: motion analysis is keyframe-driven through a different
    ///   decoder, so recording carries on. What stops is every frame the
    ///   vision model and the event thumbnails are made from — ffmpeg spawns,
    ///   emits nothing, and exits, with its output silenced (`-loglevel
    ///   quiet`) and its status unread. Nothing anywhere says so, and no
    ///   operator means "zero frames per second".
    /// - a motion default that is not a number used to reach the detector
    ///   through a clamp that cannot bound it, where `area >= NaN` is false for
    ///   every blob there will ever be: motion detection off, events stopping
    ///   altogether, and only the recording watchdog's eventual "recorded
    ///   nothing" line — which names no cause — to show for it.
    ///   [`MotionSettings::sanitize`](crate::analytics::motion_settings::MotionSettings::sanitize)
    ///   now substitutes the default before the detector is built, so this
    ///   refusal is the first of two nets rather than the only one: it is what
    ///   makes the mistake *visible*, since the second net is silent by nature
    ///   and would leave a camera running settings nobody chose.
    /// - a confidence floor that is not a number turns the filter off rather
    ///   than up: `confidence < NaN` is false, so every detection of an
    ///   allowed class is kept whatever the model thought of it.
    ///
    /// Only what the run will actually read: with analytics off no analyzer is
    /// spawned and no motion settings store is built, and with object detection
    /// off no client is created — same reasoning as the storage checks above,
    /// which a disabled `[storage]` skips.
    fn validate_analytics(&self) -> Result<(), ConfigError> {
        if !self.analytics.enabled {
            return Ok(());
        }

        if self.analytics.sample_fps == 0 {
            return Err(ConfigError::ZeroSampleFps);
        }

        // These two seed every camera's motion_settings.json, and a value that
        // is not a number is the one thing `repair`'s clamp cannot correct:
        // there is no nearest valid slider to a NaN. It is refused rather than
        // substituted because nobody writes `nan` meaning anything, and a
        // detector quietly running on a default the operator did not choose is
        // worth less than being told to fix the line.
        //
        // The first of two nets, not the only one: `MotionSettings::sanitize`
        // substitutes the module default for a non-finite slider before the
        // detector is built, so this refusal is what makes the loss *visible*
        // rather than what makes it survivable. Both are deliberate — the check
        // here covers the config file, sanitize covers the per-camera files and
        // the API, and neither can see the other's input.
        for (key, value) in [
            ("var_threshold", self.analytics.motion.var_threshold),
            ("min_contour_area", self.analytics.motion.min_contour_area),
        ] {
            if !value.is_finite() {
                return Err(ConfigError::NonFiniteMotionDefault { key, value });
            }
        }

        if !self.analytics.object_detection.enabled {
            return Ok(());
        }
        // Only the reading that cannot be corrected: a finite threshold out of
        // range is clamped by `repair`, the way an out-of-range motion slider
        // is, but there is no nearest sensible value for a number that is not
        // one, and getting it wrong here means keeping detections rather than
        // dropping them.
        let confidence = self.analytics.object_detection.confidence_threshold;
        if !confidence.is_finite() {
            return Err(ConfigError::ConfidenceThresholdNotANumber { value: confidence });
        }

        Ok(())
    }

    /// Values camon corrects instead of refusing, warning as it does.
    ///
    /// The other half of [`validate_analytics`](Self::validate_analytics)'s
    /// line. Each of these is either already complained about where it bites —
    /// so the config is not the operator's only clue — or has a reading
    /// somebody plausibly meant, and both are worth less than a box that will
    /// not boot and cannot self-update.
    ///
    /// Correcting is only defensible while it is *said*, so every branch here
    /// warns with the field, what was written and what is being used. That is
    /// also why the motion sliders are corrected here rather than left to
    /// `MotionSettings::sanitize`, which bounds them per camera without a word:
    /// an operator who wrote `var_threshold = 1000` was running a detector
    /// pinned at its least sensitive setting and was never told, while the one
    /// who wrote a confidence of 4.5 was. Sanitize stays as the second net,
    /// under the API and the per-camera files this pass never sees.
    fn repair(&mut self) {
        if !self.analytics.enabled {
            return;
        }

        // The same correction the settings store would make silently, made
        // once, out loud, over the value the operator actually wrote. Finite
        // only: a slider that is not a number is fatal in `validate_analytics`,
        // because there is no nearest value to correct it to.
        for (key, value, min, max) in [
            (
                "var_threshold",
                &mut self.analytics.motion.var_threshold,
                motion_settings::VAR_THRESHOLD_MIN,
                motion_settings::VAR_THRESHOLD_MAX,
            ),
            (
                "min_contour_area",
                &mut self.analytics.motion.min_contour_area,
                motion_settings::MIN_CONTOUR_AREA_MIN,
                motion_settings::MIN_CONTOUR_AREA_MAX,
            ),
        ] {
            if value.is_finite() && !(min..=max).contains(value) {
                let clamped = value.clamp(min, max);
                tracing::warn!(
                    key,
                    configured = *value,
                    using = clamped,
                    "[analytics.motion] {key} ({value}) is outside {min}-{max}; using {clamped}"
                );
                *value = clamped;
            }
        }

        if !self.analytics.object_detection.enabled {
            return;
        }

        // Clamped rather than refused, matching the motion sliders above: one
        // policy for "a real number, out of its range" across the whole file. A
        // threshold above 1.0 drops every detection and a negative one is the
        // same as 0.0 — the model's own confidences are validated into [0, 1]
        // before they are compared, so neither can do anything a bound cannot.
        // Finite only, which is the other half of that one policy: `inf` does
        // have a clamp, but taking it would mean this file rejected `nan` and
        // accepted `inf` for the same field while rejecting both for the motion
        // sliders next door.
        let confidence = &mut self.analytics.object_detection.confidence_threshold;
        if confidence.is_finite() && !(0.0..=1.0).contains(confidence) {
            let clamped = confidence.clamp(0.0, 1.0);
            tracing::warn!(
                configured = *confidence,
                using = clamped,
                "[analytics.object_detection] confidence_threshold ({confidence}) is outside \
                 0.0-1.0; using {clamped}"
            );
            *confidence = clamped;
        }

        // Substituted rather than refused: "0 means no timeout" is a widespread
        // convention and someone may well have written it meaning that. Here it
        // is the opposite — reqwest deadlines every request immediately — but
        // the detection worker already warns on each failed job, so the
        // consequence is visible either way.
        let timeout = &mut self.analytics.object_detection.ollama.timeout_secs;
        if *timeout == 0 {
            *timeout = default_ollama_timeout_secs();
            tracing::warn!(
                using = *timeout,
                "[analytics.object_detection.ollama] timeout_secs is 0, which is not \"no \
                 timeout\" here but a deadline every request misses; using {timeout}s"
            );
        }
    }

    /// Seconds that are multiplied out to nanoseconds somewhere downstream, and
    /// the one duration that must not be zero.
    ///
    /// The same discipline as [`MAX_RETENTION_DAYS`], for the same arithmetic:
    /// `secs * 1_000_000_000` wraps `u64` around 584 years, and a wrapped
    /// duration is not a large one but a tiny or a zero one — which is how an
    /// absurd setting turns into no recording instead of obviously too much of
    /// it. The bound is the ten years `MAX_RETENTION_DAYS` already allows,
    /// expressed in seconds: far past any real padding, buffer or chunk, and 58
    /// times short of the wrap.
    ///
    /// Each bound is gated where its arithmetic actually runs, and nowhere
    /// wider — a value nothing will read must not stop the load, or turning a
    /// feature off would be a way to strand an install that a stricter camon
    /// updated into. The three multiplications, and their gates:
    ///
    /// - `hot_duration_secs`: `HotBuffer::new` multiplies it for every camera
    ///   before any mode is consulted, and the live view reads that buffer with
    ///   analytics and storage both off. **Ungated**, and the only one that
    ///   also may not be 0.
    /// - `pre_padding_secs`: multiplied bare (`* 1_000_000_000`) where the
    ///   analyzer is spawned, inside the analytics gate; continuous recording
    ///   never pads. **Gated on `[analytics] enabled`.**
    /// - `max_event_duration_secs`: converted with `as_nanos() as u64` in the
    ///   continuous recorder, which truncates rather than wraps above the same
    ///   horizon. **Gated on `[storage] enabled`** — a superset of the
    ///   continuous mode that does the conversion, kept together with the other
    ///   rules about this field so they cannot disagree about when the cap
    ///   matters.
    ///
    /// `post_padding_secs` is deliberately **not** bounded here. It was, and
    /// the justification did not survive being checked: it only ever becomes a
    /// `Duration::from_secs` and is compared with `saturating_duration_since`,
    /// neither of which can overflow. What a huge one really costs — a run
    /// whose footage is evicted before anything closes it — is a relation to
    /// the buffer rather than a magnitude, and `validate` warns about that
    /// relation from a far lower and more meaningful threshold.
    fn validate_durations(&self) -> Result<(), ConfigError> {
        // A zero-length hot buffer evicts every segment in the same call that
        // pushed it, so the analyzer is fed nothing, no event can be assembled
        // from a buffer that holds none, and continuous recording has nothing
        // to cut. Fatal for the same reason `sample_fps = 0` is: nothing says
        // so, and nobody means "buffer nothing".
        if self.buffer.hot_duration_secs == 0 {
            return Err(ConfigError::ZeroHotDuration);
        }

        let bounded = [
            (
                true,
                "buffer",
                "hot_duration_secs",
                self.buffer.hot_duration_secs,
            ),
            (
                self.analytics.enabled,
                "storage",
                "pre_padding_secs",
                self.storage.pre_padding_secs,
            ),
            (
                self.storage.enabled,
                "storage",
                "max_event_duration_secs",
                self.storage.max_event_duration_secs,
            ),
        ];
        for (read_by_this_run, section, key, secs) in bounded {
            if read_by_this_run && secs > MAX_DURATION_SECS {
                return Err(ConfigError::DurationTooLarge {
                    section,
                    key,
                    secs,
                    max: MAX_DURATION_SECS,
                });
            }
        }

        Ok(())
    }
}

/// The span of an event that will not fit in the hot buffer, or `None` when it
/// fits. Only `pre` widens an event — it reaches back before the first motion
/// segment at assembly time — while post-padding cannot, because
/// `RunTracker::observe` tests the post-padding close before the cap, so a
/// chunk still ends at the cap. The comparison is strict because both the
/// recorder's tick and segment rounding overshoot slightly; with integer
/// seconds that leaves at least a second of slack.
fn event_span_overrun(cap: u64, pre: u64, hot: u64) -> Option<u64> {
    let total = cap.saturating_add(pre);
    (total >= hot).then_some(total)
}

/// Keys camon itself shipped and later removed, with the advice to print when
/// one turns up. They are dropped with a warning rather than rejected by the
/// strict unknown-field check: a config that booted before must keep booting.
/// An install that auto-updated into a stricter camon would otherwise be stuck
/// — config load happens before the updater runs, so it could never start long
/// enough to update out of the failure.
const RETIRED_KEYS: &[(&[&str], &str)] = &[
    (
        &["analytics", "object_detection", "backend"],
        "object detection has been Ollama-only since 0.2.1; the server is configured under \
         [analytics.object_detection.ollama]",
    ),
    (
        &["analytics", "object_detection", "model_path"],
        "the bundled ONNX model was removed in 0.2.1; set `model` under \
         [analytics.object_detection.ollama] instead",
    ),
];

/// Remove every [`RETIRED_KEYS`] entry present in `root`, returning the dotted
/// key and its advice for each one actually found.
fn strip_retired_keys(root: &mut toml::Value) -> Vec<(String, &'static str)> {
    let mut found = Vec::new();
    for (path, advice) in RETIRED_KEYS {
        let (last, parents) = path.split_last().expect("retired key paths are non-empty");
        let mut current = Some(&mut *root);
        for segment in parents {
            current = current
                .and_then(toml::Value::as_table_mut)
                .and_then(|table| table.get_mut(*segment));
        }
        let removed = current
            .and_then(toml::Value::as_table_mut)
            .and_then(|table| table.remove(*last));
        if removed.is_some() {
            found.push((path.join("."), *advice));
        }
    }
    found
}

/// A camera id is used verbatim as a directory name under the storage
/// `data_dir`, so it must be a single path component. This is footgun
/// protection rather than a security boundary — ids reaching `data_dir.join()`
/// come only from operator config, and API handlers resolve a request to a
/// known camera before any path is built — so it denies the few shapes that
/// misbehave and accepts everything else, punctuation and non-ASCII included.
fn validate_camera_id(id: &str) -> Result<(), ConfigError> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return Err(ConfigError::BlankCameraId);
    }
    if trimmed != id {
        return Err(ConfigError::PaddedCameraId {
            id: id.to_string(),
            trimmed: trimmed.to_string(),
        });
    }
    if id == "." || id == ".." {
        return Err(ConfigError::ReservedCameraId { id: id.to_string() });
    }
    match id.chars().find(|c| matches!(c, '/' | '\\' | '\0')) {
        Some(character) => Err(ConfigError::InvalidCameraIdCharacter {
            id: id.to_string(),
            character,
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const TOML_SAMPLE: &str = r#"
[http]
port = 9090

[analytics]
enabled = true

[analytics.object_detection]
enabled = true

[analytics.object_detection.ollama]
url = "http://ollama.local:11434"
model = "gemma4:e4b"

[storage]
enabled = true
data_dir = "/data/storage"

[[cameras]]
id = "front-door"
url = "rtsp://user:pass@10.0.0.5:554/stream1"

[[cameras]]
id = "yard"
url = "rtsp://user:pass@10.0.0.6:554/stream1"
"#;

    fn write_temp(name: &str, content: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut file = std::fs::File::create(dir.path().join(name)).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        dir
    }

    fn assert_matches_sample(config: &Config) {
        assert_eq!(config.http.port, 9090);
        assert!(config.analytics.enabled);
        assert!(config.analytics.object_detection.enabled);
        assert_eq!(
            config.analytics.object_detection.ollama.url,
            "http://ollama.local:11434"
        );
        assert!(config.storage.enabled);
        assert_eq!(config.storage.data_dir, "/data/storage");
        assert_eq!(config.cameras.len(), 2);
        assert_eq!(config.cameras[0].id, "front-door");
        assert_eq!(
            config.cameras[1].url,
            "rtsp://user:pass@10.0.0.6:554/stream1"
        );
    }

    #[test]
    fn loads_a_toml_config() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let config = Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap();
        assert_matches_sample(&config);
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err =
            Config::load_from_with_overrides("/nonexistent/camon/config.toml", &[]).unwrap_err();
        assert!(matches!(err, ConfigError::Io(_)), "got {err:?}");
    }

    #[test]
    fn empty_cameras_is_rejected() {
        let dir = write_temp("config.toml", "[http]\nport = 8080\n");
        let err =
            Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap_err();
        assert!(matches!(err, ConfigError::NoCameras), "got {err:?}");
    }

    /// Type and insert one override the way the first pass of
    /// `load_from_with_overrides` does: judged against nothing but itself.
    fn apply_override(spec: &str, tree: &mut toml::Value) {
        let ov = Override::parse(spec).unwrap();
        let reading = ov.reading_against(&empty_tree()).unwrap_or_else(|| {
            // Pass 2 answers these; the callers below never use one.
            ov.guess()
        });
        insert_at(tree, &ov.path, reading);
    }

    fn load_with_overrides(toml: &str, specs: &[&str]) -> Result<Config, ConfigError> {
        let dir = write_temp("config.toml", toml);
        let overrides: Vec<Override> = specs
            .iter()
            .map(|spec| Override::parse(spec).unwrap())
            .collect();
        Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides)
    }

    #[test]
    fn override_sets_a_bool() {
        let mut v: toml::Value = toml::from_str("[update]\nenabled = true\n").unwrap();
        apply_override("update.enabled=false", &mut v);
        assert_eq!(v["update"]["enabled"].as_bool(), Some(false));
    }

    #[test]
    fn override_sets_an_integer() {
        let mut v: toml::Value = toml::from_str("[http]\nport = 8080\n").unwrap();
        apply_override("http.port=22666", &mut v);
        assert_eq!(v["http"]["port"].as_integer(), Some(22666));
    }

    #[test]
    fn override_sets_a_string() {
        let mut v: toml::Value = toml::from_str("[storage]\ndata_dir = \"/var/camon\"\n").unwrap();
        apply_override("storage.data_dir=/data/storage", &mut v);
        assert_eq!(v["storage"]["data_dir"].as_str(), Some("/data/storage"));
    }

    /// The add-on forwards Supervisor-generated MQTT credentials through
    /// `--set`, and those are regularly all digits.
    #[test]
    fn override_of_a_string_key_stays_a_string_when_it_looks_numeric() {
        let mut v: toml::Value = toml::from_str("").unwrap();
        apply_override("mqtt.password=12345", &mut v);
        apply_override("mqtt.username=0042", &mut v);
        apply_override("http.token=1e5", &mut v);
        assert_eq!(v["mqtt"]["password"].as_str(), Some("12345"));
        assert_eq!(v["mqtt"]["username"].as_str(), Some("0042"));
        assert_eq!(v["http"]["token"].as_str(), Some("1e5"));
    }

    /// The first pass judges an override against a tree holding nothing else,
    /// which only works while every table in `Config` has a serde default. Add
    /// a required field to one and its keys silently fall through to the
    /// second pass instead — which is weaker, so this is worth knowing about.
    /// `[storage.stathost]` is the one table that is already like that.
    #[test]
    fn every_setting_in_the_example_can_be_judged_on_its_own() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml.example");
        let example: toml::Value = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut settings = Vec::new();
        collect_scalars(&example, &mut Vec::new(), &mut settings);
        assert!(settings.len() > 20, "only found {}", settings.len());

        for (path, value) in settings {
            // The known exception, documented above; the example has no
            // stathost section today, so this is here for the day it does.
            if path.starts_with("storage.stathost.") {
                continue;
            }
            let segments: Vec<String> = path.split('.').map(str::to_string).collect();
            let mut probe = empty_tree();
            insert_at(&mut probe, &segments, value);
            assert!(
                probe.try_into::<Config>().is_ok(),
                "{path} can no longer be judged on its own: a table on the way to it must \
                 have gained a field without a serde default"
            );
        }
    }

    /// Every scalar in `value` as a dotted path and its value, skipping
    /// arrays: `[[cameras]]` and `classes` are not reachable with `--set`.
    fn collect_scalars(
        value: &toml::Value,
        prefix: &mut Vec<String>,
        out: &mut Vec<(String, toml::Value)>,
    ) {
        let Some(table) = value.as_table() else {
            return;
        };
        for (key, child) in table {
            prefix.push(key.clone());
            match child {
                toml::Value::Table(_) => collect_scalars(child, prefix, out),
                toml::Value::Array(_) => {}
                _ => out.push((prefix.join("."), child.clone())),
            }
            prefix.pop();
        }
    }

    /// `[storage.stathost]`'s `url`/`bucket`/`token` are required, so none of
    /// its keys can be judged alone — the second pass answers for them once
    /// the rest of the table is in place, whether it comes from the file...
    #[test]
    fn override_of_a_string_key_in_a_table_with_required_fields_stays_a_string() {
        let toml = format!(
            "[storage.stathost]\nurl = \"https://host\"\nbucket = \"camon\"\n\
             token = \"placeholder\"\n{}",
            one_camera("yard")
        );
        let config = load_with_overrides(&toml, &["storage.stathost.token=12345"]).unwrap();
        assert_eq!(config.storage.stathost.unwrap().token, "12345");
    }

    /// ...or from the overrides themselves. Keeping a bearer token out of the
    /// config file is the reason to reach for `--set` here, so the table has
    /// to be constructible this way.
    #[test]
    fn a_table_with_required_fields_can_be_built_from_overrides_alone() {
        let config = load_with_overrides(
            &one_camera("yard"),
            &[
                "storage.stathost.url=https://host",
                "storage.stathost.bucket=camon",
                "storage.stathost.token=12345",
            ],
        )
        .unwrap();
        let stathost = config.storage.stathost.unwrap();
        assert_eq!(stathost.token, "12345");
        assert_eq!(stathost.bucket, "camon");
    }

    /// Typing may not depend on the order the flags were written in — the
    /// numeric-looking token is as likely to be written first as last — nor on
    /// how many of the values in one table look numeric.
    #[test]
    fn override_typing_is_independent_of_argument_order() {
        let forwards = [
            "storage.stathost.url=12345",
            "storage.stathost.bucket=camon",
            "storage.stathost.token=67890",
        ];
        let mut backwards = forwards;
        backwards.reverse();

        for order in [forwards, backwards] {
            let config = load_with_overrides(&one_camera("yard"), &order).unwrap();
            let stathost = config.storage.stathost.unwrap();
            assert_eq!(stathost.url, "12345", "order {order:?}");
            assert_eq!(stathost.token, "67890", "order {order:?}");
        }
    }

    /// Retired keys are stripped before anything is typed against the whole
    /// config, so a leftover from an older camon cannot decide a type. Both
    /// halves matter: the config still boots, and the token is still a string.
    #[test]
    fn a_retired_key_does_not_affect_typing() {
        let toml = format!(
            "[analytics.object_detection]\nmodel_path = \"/opt/yolo26.onnx\"\n\n\
             [storage.stathost]\nurl = \"https://host\"\nbucket = \"camon\"\n\
             token = \"placeholder\"\n{}",
            one_camera("yard")
        );
        let config = load_with_overrides(&toml, &["storage.stathost.token=12345"]).unwrap();
        assert_eq!(config.storage.stathost.unwrap().token, "12345");
    }

    #[test]
    fn override_of_a_float_key_accepts_a_whole_number() {
        let config =
            load_with_overrides(TOML_SAMPLE, &["analytics.motion.var_threshold=20"]).unwrap();
        assert_eq!(config.analytics.motion.var_threshold, 20.0);
    }

    /// A path the schema rejects whatever its type keeps its plain reading, so
    /// the load fails with the real complaint rather than an invented type.
    #[test]
    fn override_of_an_unknown_key_keeps_its_plain_reading() {
        let mut v: toml::Value = toml::from_str("").unwrap();
        apply_override("http.prot=9090", &mut v);
        assert_eq!(v["http"]["prot"].as_integer(), Some(9090));
    }

    #[test]
    fn override_creates_missing_intermediate_tables() {
        let mut v: toml::Value = toml::from_str("").unwrap();
        apply_override("analytics.object_detection.enabled=true", &mut v);
        assert_eq!(
            v["analytics"]["object_detection"]["enabled"].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn malformed_override_is_rejected() {
        assert!(Override::parse("no-equals-sign").is_err());
        assert!(Override::parse("=value").is_err());
        assert!(Override::parse("a..b=1").is_err());
    }

    #[test]
    fn overrides_win_over_file_values() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let overrides = [
            Override::parse("http.port=22666").unwrap(),
            Override::parse("update.enabled=true").unwrap(),
        ];
        let config =
            Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides).unwrap();
        // File said 9090 / default-false; the overrides win.
        assert_eq!(config.http.port, 22666);
        assert!(config.update.enabled);
    }

    /// Both shipped deployments are reached from another machine, so the bind
    /// stays on every interface and neither the token nor the opt-out is set by
    /// default. What that combination *means* is [`crate::api::ApiAuth`]'s to
    /// say — writes end up behind a token camon makes for itself.
    #[test]
    fn http_defaults_to_all_interfaces_with_nothing_configured() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let config = Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap();
        assert_eq!(config.http.bind, "0.0.0.0");
        assert_eq!(config.http.bind_addr(), IpAddr::from([0, 0, 0, 0]));
        assert!(config.http.token.is_none());
        assert!(!config.http.allow_open);
    }

    /// A generated token goes beside the config file, wherever that turned out
    /// to be — `/etc/camon/config.toml` puts it in `/etc/camon`, the same
    /// directory the operator already goes to when camon needs something. A
    /// config that came from no file at all has nowhere to put it and says so
    /// rather than guessing at the working directory.
    #[test]
    fn a_generated_token_is_kept_beside_the_config_file() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let path = dir.path().join("config.toml");
        let config = Config::load_from_with_overrides(&path, &[]).unwrap();
        assert_eq!(config.token_file_path(), Some(dir.path().join("api-token")));

        // A bare relative config path keeps the token relative too, resolved
        // against the same working directory the config path was.
        let mut relative = config.clone();
        relative.source_path = Some(PathBuf::from(DEFAULT_CONFIG_PATH));
        assert_eq!(relative.token_file_path(), Some(PathBuf::from("api-token")));

        let from_text: Config = toml::from_str(TOML_SAMPLE).unwrap();
        assert_eq!(from_text.token_file_path(), None);
    }

    /// The updater installs an unsigned binary into a service that runs as
    /// root, so it stays off until the operator asks for it.
    #[test]
    fn self_update_is_off_unless_asked_for() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let config = Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap();
        assert!(!config.update.enabled);
        assert!(!UpdateConfig::default().enabled);
    }

    #[test]
    fn http_auth_keys_are_settable_from_the_command_line() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let overrides = [
            Override::parse("http.token=s3cr3t").unwrap(),
            Override::parse("http.bind=127.0.0.1").unwrap(),
            Override::parse("http.allow_open=true").unwrap(),
        ];
        let config =
            Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides).unwrap();
        assert_eq!(config.http.token.as_deref(), Some("s3cr3t"));
        assert_eq!(config.http.bind_addr(), IpAddr::from([127, 0, 0, 1]));
        assert!(config.http.allow_open);
    }

    #[test]
    fn empty_http_token_is_rejected() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        for empty in ["", "   "] {
            let overrides = [Override::parse(&format!("http.token={empty}")).unwrap()];
            let err = Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides)
                .unwrap_err();
            assert!(
                matches!(err, ConfigError::HttpTokenEmpty),
                "got {err:?} for {empty:?}"
            );
        }
    }

    #[test]
    fn unparseable_bind_is_rejected() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let overrides = [Override::parse("http.bind=localhost").unwrap()];
        let err = Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides)
            .unwrap_err();
        assert!(matches!(err, ConfigError::HttpBind { .. }), "got {err:?}");
        assert!(err.to_string().contains("localhost"), "got {err}");
    }

    #[test]
    fn mqtt_defaults_to_disabled() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let config = Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap();
        assert!(!config.mqtt.enabled);
        assert_eq!(config.mqtt.host, "localhost");
        assert_eq!(config.mqtt.port, 1883);
        assert_eq!(config.mqtt.topic_prefix, "camon");
        assert_eq!(config.mqtt.discovery_prefix, "homeassistant");
        assert_eq!(config.mqtt.snapshot_interval_secs, 5);
        assert_eq!(config.mqtt.occupancy_hold_secs, 60);
        assert!(config.mqtt.username.is_none());
    }

    #[test]
    fn mqtt_section_is_parsed_and_overridable() {
        let toml = format!(
            "{TOML_SAMPLE}\n[mqtt]\nenabled = true\nhost = \"10.0.0.2\"\nusername = \"ha\"\n\
             password = \"secret\"\n"
        );
        let dir = write_temp("config.toml", &toml);
        let overrides = [Override::parse("mqtt.occupancy_hold_secs=120").unwrap()];
        let config =
            Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides).unwrap();
        assert!(config.mqtt.enabled);
        assert_eq!(config.mqtt.host, "10.0.0.2");
        assert_eq!(config.mqtt.username.as_deref(), Some("ha"));
        assert_eq!(config.mqtt.password.as_deref(), Some("secret"));
        // Unset keys keep their defaults; `--set` reaches the new section
        // through the generic override path.
        assert_eq!(config.mqtt.port, 1883);
        assert_eq!(config.mqtt.occupancy_hold_secs, 120);
    }

    const WILDCARD_CAMERAS: &str = r#"
[[cameras]]
id = "cam+1"
url = "rtsp://10.0.0.5:554/stream1"
"#;

    const COLLIDING_CAMERAS: &str = r#"
[[cameras]]
id = "Front Door"
url = "rtsp://10.0.0.5:554/stream1"

[[cameras]]
id = "front-door"
url = "rtsp://10.0.0.6:554/stream1"
"#;

    fn load_with_mqtt(cameras: &str, enabled: bool) -> Result<Config, ConfigError> {
        let toml = format!("[mqtt]\nenabled = {enabled}\n{cameras}");
        let dir = write_temp("config.toml", &toml);
        Config::load_from_with_overrides(dir.path().join("config.toml"), &[])
    }

    #[test]
    fn mqtt_rejects_wildcard_camera_id() {
        let err = load_with_mqtt(WILDCARD_CAMERAS, true).unwrap_err();
        assert!(
            matches!(err, ConfigError::MqttWildcardCameraId { .. }),
            "got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains("cam+1"), "got {message}");
        assert!(message.contains('+'), "got {message}");
    }

    const WILDCARD_CLASS: &str = r##"
[analytics.object_detection]
classes = ["person", "#"]

[[cameras]]
id = "yard"
url = "rtsp://10.0.0.5:554/stream1"
"##;

    #[test]
    fn mqtt_rejects_wildcard_object_class() {
        let err = load_with_mqtt(WILDCARD_CLASS, true).unwrap_err();
        assert!(
            matches!(err, ConfigError::MqttWildcardClass { .. }),
            "got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains('#'), "got {message}");
    }

    #[test]
    fn disabled_mqtt_leaves_object_classes_unconstrained() {
        load_with_mqtt(WILDCARD_CLASS, false).unwrap();
    }

    #[test]
    fn mqtt_rejects_colliding_camera_slugs() {
        let err = load_with_mqtt(COLLIDING_CAMERAS, true).unwrap_err();
        assert!(
            matches!(err, ConfigError::MqttSlugCollision { .. }),
            "got {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains("Front Door"), "got {message}");
        assert!(message.contains("front-door"), "got {message}");
        assert!(message.contains("front_door"), "got {message}");
    }

    #[test]
    fn disabled_mqtt_leaves_camera_ids_unconstrained() {
        // Wildcards and colliding slugs are legal ids and legal directory
        // names; only the bridge cares.
        load_with_mqtt(WILDCARD_CAMERAS, false).unwrap();
        load_with_mqtt(COLLIDING_CAMERAS, false).unwrap();
    }

    #[test]
    fn mqtt_accepts_distinct_camera_ids() {
        let config = load_with_mqtt(
            "\n[[cameras]]\nid = \"front-door\"\nurl = \"rtsp://10.0.0.5:554/stream1\"\n\n\
             [[cameras]]\nid = \"yard\"\nurl = \"rtsp://10.0.0.6:554/stream1\"\n",
            true,
        )
        .unwrap();
        assert_eq!(config.cameras.len(), 2);
    }

    fn load_cameras(cameras: &str) -> Result<Config, ConfigError> {
        let dir = write_temp("config.toml", cameras);
        Config::load_from_with_overrides(dir.path().join("config.toml"), &[])
    }

    fn one_camera(id: &str) -> String {
        format!("[[cameras]]\nid = {id:?}\nurl = \"rtsp://10.0.0.5:554/stream1\"\n")
    }

    #[test]
    fn blank_camera_ids_are_rejected() {
        for id in ["", "   ", "\t"] {
            let err = load_cameras(&one_camera(id)).unwrap_err();
            assert!(matches!(err, ConfigError::BlankCameraId), "got {err:?}");
        }
    }

    #[test]
    fn space_padded_camera_ids_are_rejected() {
        let err = load_cameras(&one_camera(" yard ")).unwrap_err();
        assert!(
            matches!(err, ConfigError::PaddedCameraId { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("\"yard\""), "got {err}");
    }

    #[test]
    fn path_like_camera_ids_are_rejected() {
        for id in ["../other", "/var/camon", "a/b", "a\\b"] {
            let err = load_cameras(&one_camera(id)).unwrap_err();
            assert!(
                matches!(err, ConfigError::InvalidCameraIdCharacter { .. }),
                "got {err:?} for {id:?}"
            );
            assert!(
                err.to_string().contains(&format!("{id:?}")),
                "got {err} for {id:?}"
            );
        }
        for id in [".", ".."] {
            let err = load_cameras(&one_camera(id)).unwrap_err();
            assert!(
                matches!(err, ConfigError::ReservedCameraId { .. }),
                "got {err:?} for {id:?}"
            );
        }
    }

    #[test]
    fn duplicate_camera_ids_are_rejected() {
        let toml = format!("{}{}", one_camera("yard"), one_camera("yard"));
        let err = load_cameras(&toml).unwrap_err();
        assert!(
            matches!(err, ConfigError::DuplicateCameraId { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("yard"), "got {err}");
    }

    /// Non-ASCII and punctuation are explicitly in scope: the repo owner uses
    /// Swedish camera names, and every one of these is a fine directory name.
    #[test]
    fn ordinary_camera_ids_are_accepted() {
        for id in [
            "front-door",
            "Front Door",
            "Trädgård",
            "Entré",
            "Garage (side)",
            "door:west",
            "user@door",
            "cam+garage",
            "cam_1",
            "cam.1",
            "..a",
        ] {
            load_cameras(&one_camera(id)).unwrap_or_else(|e| panic!("{id:?} rejected: {e}"));
        }
    }

    /// Continuous mode (storage on, analytics off) rolls a chunk only when the
    /// cap is reached, so 0 would write nothing until shutdown.
    #[test]
    fn zero_max_event_duration_is_rejected_only_in_continuous_mode() {
        let cameras = one_camera("yard");
        let continuous = format!(
            "[analytics]\nenabled = false\n[storage]\nmax_event_duration_secs = 0\n{cameras}"
        );
        let err = load_cameras(&continuous).unwrap_err();
        assert!(
            matches!(err, ConfigError::ZeroMaxEventDurationInContinuousMode),
            "got {err:?}"
        );

        // Event mode: 0 means "don't chunk", and motion end still closes runs.
        let event_mode = format!(
            "[analytics]\nenabled = true\n[storage]\nmax_event_duration_secs = 0\n{cameras}"
        );
        let config = load_cameras(&event_mode).unwrap();
        assert_eq!(config.storage.max_event_duration_secs, 0);
    }

    /// Event mode still records when the span overruns — it only loses the
    /// head of long events — so this warns and boots rather than refusing.
    #[test]
    fn event_longer_than_the_hot_buffer_still_loads() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        for cap in [600, 900] {
            let overrides =
                [Override::parse(&format!("storage.max_event_duration_secs={cap}")).unwrap()];
            let config =
                Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides)
                    .unwrap_or_else(|e| panic!("cap {cap} rejected: {e}"));
            assert_eq!(config.storage.max_event_duration_secs, cap);
        }
    }

    /// Only pre_padding widens an event. post_padding cannot: `observe` tests
    /// the post-padding close before the cap, so the chunk still ends at the
    /// cap.
    #[test]
    fn the_event_span_counts_pre_padding_only() {
        // 594 + 5 = 599 < 600 fits; 595 + 5 = 600 does not.
        assert_eq!(event_span_overrun(594, 5, 600), None);
        assert_eq!(event_span_overrun(595, 5, 600), Some(600));
        // post_padding is not an input at all, so it cannot move the boundary.
        assert_eq!(event_span_overrun(590, 0, 600), None);
        assert_eq!(event_span_overrun(u64::MAX, 5, 600), Some(u64::MAX));
    }

    /// Continuous chunks are cut straight from the buffer with no padding, so
    /// the cap alone is the span.
    #[test]
    fn the_continuous_bound_is_the_cap_alone() {
        let cameras = one_camera("yard");
        let load = |cap: u64| {
            let toml = format!(
                "[buffer]\nhot_duration_secs = 600\n[analytics]\nenabled = false\n\
                 [storage]\nmax_event_duration_secs = {cap}\npre_padding_secs = 5\n\
                 post_padding_secs = 10\n{cameras}"
            );
            load_cameras(&toml)
        };
        // 590 works in the current release and must keep working; 599 is the
        // last accepted value.
        load(590).unwrap();
        load(599).unwrap();
        let err = load(600).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::ContinuousChunkExceedsHotBuffer { cap: 600, hot: 600 }
            ),
            "got {err:?}"
        );
        let message = err.to_string();
        assert!(!message.contains("padding"), "got {message}");
    }

    /// 0 is not "keep forever": every stored event is instantly past a
    /// zero-day retention, so the next hourly sweep would delete the archive.
    #[test]
    fn zero_retention_days_is_rejected_for_every_class() {
        let cameras = one_camera("yard");
        for key in [
            "movement_retention_days",
            "object_retention_days",
            "continuous_retention_days",
        ] {
            let err = load_cameras(&format!("[storage]\n{key} = 0\n{cameras}")).unwrap_err();
            assert!(
                matches!(err, ConfigError::ZeroRetentionDays { key: k } if k == key),
                "got {err:?}"
            );
            // Nothing sweeps with storage off, so nothing to refuse to boot for.
            let off = format!("[storage]\nenabled = false\n{key} = 0\n{cameras}");
            load_cameras(&off).unwrap_or_else(|e| panic!("{key} = 0 with storage off: {e}"));
        }
    }

    /// Retention is held in nanoseconds, where a big enough value wraps into a
    /// very short one — the only way this setting can delete footage.
    #[test]
    fn retention_days_past_the_bound_is_rejected() {
        let cameras = one_camera("yard");
        let toml = format!("[storage]\nobject_retention_days = 3651\n{cameras}");
        let err = load_cameras(&toml).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::RetentionDaysTooLarge {
                    key: "object_retention_days",
                    days: 3651,
                    ..
                }
            ),
            "got {err:?}"
        );
        for days in [1, MAX_RETENTION_DAYS] {
            let toml = format!("[storage]\nobject_retention_days = {days}\n{cameras}");
            load_cameras(&toml).unwrap_or_else(|e| panic!("{days} days rejected: {e}"));
        }
    }

    /// With storage off no writer and no continuous recorder are spawned, so
    /// the cap is never read and must not be able to block startup.
    #[test]
    fn disabled_storage_skips_the_duration_checks() {
        let cameras = one_camera("yard");
        for storage in [
            "enabled = false\nmax_event_duration_secs = 0",
            "enabled = false\nmax_event_duration_secs = 9000",
        ] {
            let toml = format!("[buffer]\nhot_duration_secs = 60\n[storage]\n{storage}\n{cameras}");
            load_cameras(&toml).unwrap_or_else(|e| panic!("{storage:?} rejected: {e}"));
        }
    }

    #[test]
    fn unknown_config_key_is_rejected() {
        let toml = format!(
            "[storage]\nmovment_retention_days = 3\n{}",
            one_camera("yard")
        );
        let err = load_cameras(&toml).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "got {err:?}");
        assert!(
            err.to_string().contains("movment_retention_days"),
            "got {err}"
        );
    }

    #[test]
    fn unknown_override_key_is_rejected() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let overrides = [Override::parse("http.prot=9090").unwrap()];
        let err = Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides)
            .unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "got {err:?}");
    }

    /// Every key documented in `config.toml.example` must still parse — with
    /// `deny_unknown_fields` a stale example is a startup failure, not a
    /// no-op.
    #[test]
    fn example_config_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml.example");
        let config = Config::load_from_with_overrides(&path, &[]).unwrap();
        assert_eq!(config.cameras.len(), 1);
        assert_eq!(config.buffer.hot_duration_secs, 600);
    }

    /// The `--set` arguments `run.sh` really passes, read from the script
    /// itself so this cannot drift from it, with the Supervisor's shell
    /// variables filled in. Comment lines are skipped: the header explains
    /// `--set` in prose.
    fn addon_run_sh_overrides() -> Vec<String> {
        const RUN_SH: &str = include_str!("../camon-addon/run.sh");
        // Values the Supervisor generates. The password is all digits on
        // purpose — those are generated too, and one read as an integer stops
        // the add-on from starting.
        const SUPERVISOR: [(&str, &str); 4] = [
            ("${MQTT_HOST}", "core-mosquitto"),
            ("${MQTT_PORT}", "1883"),
            ("${MQTT_USERNAME}", "addons"),
            ("${MQTT_PASSWORD}", "48151623"),
        ];

        let mut out = Vec::new();
        for line in RUN_SH.lines().filter(|l| !l.trim_start().starts_with('#')) {
            let mut tokens = line.split_whitespace();
            while let Some(token) = tokens.next() {
                // Ends with rather than equals: the MQTT block builds an array,
                // so the first flag on those lines reads `MQTT_ARGS+=(--set`.
                if !token.ends_with("--set") {
                    continue;
                }
                let mut arg = tokens
                    .next()
                    .expect("--set in run.sh without an argument")
                    .trim_matches(|c| c == '"' || c == ')')
                    .to_string();
                for (variable, value) in SUPERVISOR {
                    arg = arg.replace(variable, value);
                }
                assert!(arg.contains('='), "unparsed --set argument {arg:?}");
                out.push(arg);
            }
        }
        out
    }

    /// The add-on forces these on every start, so a key renamed in camon is a
    /// container that will not boot. The file underneath sets every one of
    /// them to something else, so an override that stopped being applied is a
    /// failure here rather than a value that happens to match a default.
    #[test]
    fn addon_overrides_still_apply() {
        let specs = addon_run_sh_overrides();
        assert_eq!(specs.len(), 10, "run.sh changed: {specs:?}");
        let file = format!(
            "[update]\nenabled = true\n\n[http]\nport = 9090\nbind = \"127.0.0.1\"\n\
             allow_open = false\n\n[storage]\ndata_dir = \"/var/lib/camon\"\n\n\
             [mqtt]\nenabled = false\nhost = \"mqtt.example\"\nport = 1884\n\
             username = \"fileuser\"\npassword = \"filepass\"\n{}",
            one_camera("yard")
        );
        let refs: Vec<&str> = specs.iter().map(String::as_str).collect();
        let config = load_with_overrides(&file, &refs).unwrap();

        assert!(!config.update.enabled);
        assert_eq!(config.http.port, 22666);
        assert_eq!(config.http.bind, "0.0.0.0");
        assert!(config.http.allow_open);
        assert_eq!(config.storage.data_dir, "/data/storage");
        assert!(config.mqtt.enabled);
        assert_eq!(config.mqtt.host, "core-mosquitto");
        assert_eq!(config.mqtt.port, 1883);
        assert_eq!(config.mqtt.username.as_deref(), Some("addons"));
        assert_eq!(config.mqtt.password.as_deref(), Some("48151623"));
    }

    /// Retired keys have to stay bootable: rejecting them would strand an
    /// install that auto-updated into strict parsing, since config load runs
    /// before the updater.
    #[test]
    fn retired_keys_are_dropped_with_advice_instead_of_rejected() {
        let toml = format!(
            "[analytics.object_detection]\nenabled = true\nbackend = \"onnx\"\n\
             model_path = \"/opt/yolo26.onnx\"\n{}",
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert!(config.analytics.object_detection.enabled);

        let mut value: toml::Value = toml::from_str(&toml).unwrap();
        let found = strip_retired_keys(&mut value);
        let keys: Vec<&str> = found.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "analytics.object_detection.backend",
                "analytics.object_detection.model_path"
            ]
        );
        for (_, advice) in &found {
            assert!(advice.contains("ollama"), "got {advice}");
        }
        // Removed from the tree, so the strict unknown-field check never sees
        // them, and a second pass finds nothing.
        assert!(strip_retired_keys(&mut value).is_empty());
    }

    /// A retired key must stay distinguishable from a typo — the latter is
    /// still a hard error.
    #[test]
    fn a_typo_next_to_a_retired_key_is_still_rejected() {
        let toml = format!(
            "[analytics.object_detection]\nmodel_path = \"/opt/yolo26.onnx\"\n\
             confidance_threshold = 0.5\n{}",
            one_camera("yard")
        );
        let err = load_cameras(&toml).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "got {err:?}");
        assert!(
            err.to_string().contains("confidance_threshold"),
            "got {err}"
        );
    }

    #[test]
    fn object_classes_are_lowercased_at_load() {
        let toml = format!(
            "[analytics.object_detection]\nenabled = true\nclasses = [\"Person\", \"CAR\"]\n{}",
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(
            config.analytics.object_detection.classes,
            vec!["person".to_string(), "car".to_string()]
        );
    }

    /// Two spellings of one class would otherwise become two discovery
    /// payloads sharing a unique id.
    #[test]
    fn object_classes_are_deduplicated_after_folding() {
        let toml = format!(
            "[analytics.object_detection]\nclasses = [\"Person\", \"person\", \"Cat\"]\n{}",
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(
            config.analytics.object_detection.classes,
            vec!["person".to_string(), "cat".to_string()]
        );
    }

    /// The detector and the MQTT bridge would otherwise read an empty list
    /// differently: built-in defaults for one, no entities at all for the
    /// other.
    #[test]
    fn empty_object_classes_is_rejected_while_detection_is_on() {
        let toml = format!(
            "[analytics]\nenabled = true\n\n\
             [analytics.object_detection]\nenabled = true\nclasses = []\n{}",
            one_camera("yard")
        );
        let err = load_cameras(&toml).unwrap_err();
        assert!(
            matches!(err, ConfigError::EmptyObjectClasses { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("person, car, truck, dog, cat"));
    }

    /// With nothing asking for classes there is nothing to disagree about —
    /// and both halves of the gate turn the detector off, so both are allowed
    /// to leave the list empty.
    #[test]
    fn empty_object_classes_is_accepted_while_nothing_detects() {
        for analytics in [
            "[analytics]\nenabled = true\n\n[analytics.object_detection]\nenabled = false",
            "[analytics]\nenabled = false\n\n[analytics.object_detection]\nenabled = true",
        ] {
            let toml = format!("{analytics}\nclasses = []\n{}", one_camera("yard"));
            let config =
                load_cameras(&toml).unwrap_or_else(|e| panic!("rejected with {analytics:?}: {e}"));
            assert!(config.analytics.object_detection.classes.is_empty());
        }
    }

    /// A blank entry is an empty list one step down: the model can never
    /// return it, and it would still get entities of its own.
    #[test]
    fn blank_object_classes_are_rejected_while_detection_is_on() {
        for classes in [r#"[""]"#, r#"["  "]"#, r#"["person", "\t"]"#] {
            let toml = format!(
                "[analytics]\nenabled = true\n\n\
                 [analytics.object_detection]\nenabled = true\nclasses = {classes}\n{}",
                one_camera("yard")
            );
            let err = load_cameras(&toml).unwrap_err();
            assert!(
                matches!(err, ConfigError::BlankObjectClass),
                "{classes} got {err:?}"
            );
        }
    }

    /// Padding is folded away rather than rejected, like the case folding it
    /// travels with: the topic and the prompt have to spell a class the same.
    #[test]
    fn padded_object_classes_are_trimmed() {
        let toml = format!(
            "[analytics.object_detection]\nclasses = [\" Person \", \"person\"]\n{}",
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(
            config.analytics.object_detection.classes,
            vec!["person".to_string()]
        );
    }

    /// Omitting the key is the documented way to ask for the built-in list,
    /// and it must stay distinct from writing an empty one.
    #[test]
    fn omitted_object_classes_are_the_defaults() {
        let toml = format!(
            "[analytics]\nenabled = true\n\n\
             [analytics.object_detection]\nenabled = true\n{}",
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(config.analytics.object_detection.classes, default_classes());
    }

    /// `fps=0` is a filter ffmpeg accepts and emits nothing through, so the
    /// analyzer runs for ever on a stream it never sees a frame of.
    #[test]
    fn zero_sample_fps_is_rejected() {
        let cameras = one_camera("yard");
        let toml = format!("[analytics]\nenabled = true\nsample_fps = 0\n{cameras}");
        let err = load_cameras(&toml).unwrap_err();
        assert!(matches!(err, ConfigError::ZeroSampleFps), "got {err:?}");
        assert!(err.to_string().contains("sample_fps"), "got {err}");

        // 1 is the smallest rate that decodes anything, and nothing analyzes
        // at all with analytics off.
        load_cameras(&format!(
            "[analytics]\nenabled = true\nsample_fps = 1\n{cameras}"
        ))
        .unwrap();
        load_cameras(&format!(
            "[analytics]\nenabled = false\nsample_fps = 0\n{cameras}"
        ))
        .unwrap();
    }

    /// The original defect, through the parser that admits it: TOML has a `nan`
    /// literal, `f64::clamp` returns NaN for NaN, and the detector's
    /// `area >= min_contour_area` is then false for every blob it will ever
    /// see. Motion detection off, nothing in the log, a healthy-looking camera.
    #[test]
    fn motion_defaults_that_are_not_numbers_are_rejected() {
        let cameras = one_camera("yard");
        for (key, literal) in [
            ("var_threshold", "nan"),
            ("var_threshold", "inf"),
            ("min_contour_area", "nan"),
            ("min_contour_area", "-inf"),
        ] {
            let toml = format!(
                "[analytics]\nenabled = true\n[analytics.motion]\n{key} = {literal}\n{cameras}"
            );
            let err = load_cameras(&toml).unwrap_err();
            assert!(
                matches!(err, ConfigError::NonFiniteMotionDefault { key: k, .. } if k == key),
                "{key} = {literal} got {err:?}"
            );
            assert!(err.to_string().contains(key), "got {err}");
        }

        // A real number out of range is corrected instead — the difference
        // between the two policies, on one field.
        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.motion]\nvar_threshold = 1000\n{cameras}"
        );
        assert_eq!(
            load_cameras(&toml).unwrap().analytics.motion.var_threshold,
            motion_settings::VAR_THRESHOLD_MAX
        );
    }

    /// A threshold that is not a number turns the filter off — `x < NaN` is
    /// false, so every detection of an allowed class is kept — and no clamp can
    /// guess what was meant, so this one is fatal.
    #[test]
    fn a_confidence_threshold_that_is_not_a_number_is_rejected() {
        let cameras = one_camera("yard");
        let detection = "[analytics]\nenabled = true\n[analytics.object_detection]\nenabled = true";
        for literal in ["nan", "inf", "-inf"] {
            let toml = format!("{detection}\nconfidence_threshold = {literal}\n{cameras}");
            let err = load_cameras(&toml).unwrap_err();
            assert!(
                matches!(err, ConfigError::ConfidenceThresholdNotANumber { .. }),
                "{literal} got {err:?}"
            );
            assert!(
                err.to_string().contains("confidence_threshold"),
                "got {err}"
            );
        }
        // Both ends are meaningful settings: keep everything, or keep only a
        // certain detection.
        for literal in ["0.0", "0.5", "1.0"] {
            let toml = format!("{detection}\nconfidence_threshold = {literal}\n{cameras}");
            load_cameras(&toml).unwrap_or_else(|e| panic!("{literal} rejected: {e}"));
        }
        // Nothing reads it while detection is off.
        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.object_detection]\nenabled = false\n\
             confidence_threshold = nan\n{cameras}"
        );
        load_cameras(&toml).unwrap();
    }

    /// A real number out of range is corrected, not refused — the same policy
    /// the motion sliders have, and the model's own confidences are validated
    /// into [0, 1] anyway, so a bound can express everything either end means.
    #[test]
    fn a_confidence_threshold_outside_zero_to_one_is_clamped_into_it() {
        let cameras = one_camera("yard");
        let detection = "[analytics]\nenabled = true\n[analytics.object_detection]\nenabled = true";
        for (literal, expected) in [("1.5", 1.0), ("-0.1", 0.0), ("100", 1.0)] {
            let toml = format!("{detection}\nconfidence_threshold = {literal}\n{cameras}");
            let config = load_cameras(&toml).unwrap_or_else(|e| panic!("{literal} rejected: {e}"));
            assert_eq!(
                config.analytics.object_detection.confidence_threshold, expected,
                "{literal}"
            );
        }
    }

    /// 0 here is not "wait forever" but a deadline every request misses. It is
    /// corrected rather than refused: the convention it borrows is common
    /// enough to have been meant, and the detection worker warns on every job
    /// it costs, so the config file is not the only place this shows up.
    #[test]
    fn a_zero_ollama_timeout_becomes_the_default() {
        let cameras = one_camera("yard");
        let detection = "[analytics]\nenabled = true\n[analytics.object_detection]\nenabled = true";
        let toml = format!(
            "{detection}\n[analytics.object_detection.ollama]\ntimeout_secs = 0\n{cameras}"
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(
            config.analytics.object_detection.ollama.timeout_secs,
            default_ollama_timeout_secs()
        );

        // A budget that was asked for is left alone.
        let toml = format!(
            "{detection}\n[analytics.object_detection.ollama]\ntimeout_secs = 1\n{cameras}"
        );
        assert_eq!(
            load_cameras(&toml)
                .unwrap()
                .analytics
                .object_detection
                .ollama
                .timeout_secs,
            1
        );
    }

    /// Somewhere for a subscriber to write, so a test can read what an operator
    /// would have seen. The same shape `supervise` uses.
    #[derive(Clone, Default)]
    struct CapturedLog(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Everything a load of `toml` says at warn level — what an operator on a
    /// production box (`camon=warn`) would actually see of it.
    fn warnings_from(toml: &str) -> String {
        let logs = CapturedLog::default();
        {
            let _reader = tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .with_writer(logs.clone())
                    .with_max_level(tracing::Level::WARN)
                    .with_ansi(false)
                    .finish(),
            );
            load_cameras(toml).expect("config under test must load");
        }
        let written = logs.0.lock().unwrap().clone();
        String::from_utf8(written).unwrap()
    }

    /// Correcting a value instead of refusing it is only defensible while the
    /// operator is told: this line is the entire difference between a repair
    /// and a silent override, and production logs warnings and above. It has to
    /// name the field — the config file is large — and say what camon used.
    #[test]
    fn every_repaired_value_is_reported_at_warn() {
        let written = warnings_from(&format!(
            "[analytics]\nenabled = true\n[analytics.motion]\nvar_threshold = 1000\n\
             min_contour_area = 1\n[analytics.object_detection]\nenabled = true\n\
             confidence_threshold = 4.5\n[analytics.object_detection.ollama]\n\
             timeout_secs = 0\n{}",
            one_camera("yard")
        ));
        for expected in [
            "var_threshold",
            "1000",
            "using 96",
            "min_contour_area",
            "using 50",
            "confidence_threshold",
            "4.5",
            "using 1",
            "timeout_secs",
            "using 90s",
        ] {
            assert!(written.contains(expected), "no {expected:?} in: {written}");
        }
        assert_eq!(written.matches("WARN").count(), 4, "got: {written}");
    }

    /// The correction the operator is told about is the one the detector gets:
    /// the config value is clamped here, and `sanitize` — which would otherwise
    /// have made the same correction per camera without a word — has nothing
    /// left to do to it.
    #[test]
    fn out_of_range_motion_sliders_are_clamped_on_the_config_path() {
        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.motion]\nvar_threshold = 1000\n\
             min_contour_area = 1\n{}",
            one_camera("yard")
        );
        let motion = load_cameras(&toml).unwrap().analytics.motion;
        assert_eq!(motion.var_threshold, motion_settings::VAR_THRESHOLD_MAX);
        assert_eq!(
            motion.min_contour_area,
            motion_settings::MIN_CONTOUR_AREA_MIN
        );

        let seeded = crate::analytics::motion_settings::MotionSettings::from_defaults(
            motion.var_threshold,
            motion.min_contour_area,
        );
        assert_eq!(seeded.var_threshold, motion.var_threshold);
        assert_eq!(seeded.min_contour_area, motion.min_contour_area);

        // In range is left alone, and says nothing.
        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.motion]\nvar_threshold = 24\n{}",
            one_camera("yard")
        );
        assert!(warnings_from(&toml).is_empty());
        assert_eq!(
            load_cameras(&toml).unwrap().analytics.motion.var_threshold,
            24.0
        );
    }

    /// Nothing that only the detector reads may stop a camon that is not
    /// detecting: with object detection off both of the above are left exactly
    /// as written.
    #[test]
    fn repairs_are_confined_to_what_object_detection_reads() {
        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.object_detection]\nenabled = false\n\
             confidence_threshold = 9.0\n[analytics.object_detection.ollama]\n\
             timeout_secs = 0\n{}",
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(config.analytics.object_detection.confidence_threshold, 9.0);
        assert_eq!(config.analytics.object_detection.ollama.timeout_secs, 0);
    }

    /// A post-padding that is not shorter than the buffer means a run's own
    /// footage can be gone by the time the quiet window closes it — but what
    /// that costs turns on what else closes chunks, and the warning has to say
    /// the true version of it. Three shapes, three different truths:
    ///
    /// - no cap: the quiet window is the only close there is, so it is every
    ///   event, and the runtime warning follows every one of them.
    /// - a cap under the buffer: the cap always wins the race, so every chunk
    ///   holding motion is assembled while its footage is resident — nothing is
    ///   lost *to this mechanism*, and the cost is a padding-only tail.
    /// - a cap at or above the buffer (legal in event mode): a motion-bearing
    ///   chunk closes at `min(cap, e + post)`, which is at least a buffer's
    ///   worth of time however the race goes, so footage is lost again.
    ///
    /// None of the three refuses the boot: the events are written, and the
    /// assembly warns about a lost head as it happens.
    #[test]
    fn a_post_padding_that_outlasts_the_buffer_warns_in_the_terms_that_are_true() {
        let cameras = one_camera("yard");
        let event_mode = |post: u64, cap: u64| {
            format!(
                "[buffer]\nhot_duration_secs = 600\n[analytics]\nenabled = true\n[storage]\n\
                 post_padding_secs = {post}\nmax_event_duration_secs = {cap}\n{cameras}"
            )
        };
        let evicted_head = crate::buffer::warm::EVICTED_HEAD_WARNING;

        // No cap: the quiet window is the only thing that closes a run, so the
        // loss really is every event.
        let written = warnings_from(&event_mode(600, 0));
        assert!(written.contains("every event"), "{written}");
        assert!(
            written.contains(evicted_head),
            "does not point at the runtime warning: {written}"
        );

        // A cap under the buffer wins the race every time, so nothing is
        // evicted and no runtime warning fires — the cost is a padding-only
        // tail, and the size of it depends on where motion stopped. 599 is the
        // last cap that is safe here, and the boundary is pinned rather than a
        // comfortable value, because the whole point of the split is where it
        // falls.
        for cap in [120, 599] {
            let written = warnings_from(&event_mode(600, cap));
            assert!(written.contains("padding-only"), "cap {cap}: {written}");
            assert!(
                written.contains("up to ~600s"),
                "cap {cap} states the tail as a certainty: {written}"
            );
            assert!(
                !written.contains("every event"),
                "cap {cap} overclaims: {written}"
            );
            assert!(
                !written.contains(evicted_head),
                "cap {cap} promises a runtime warning this shape never emits: {written}"
            );
            // The one loss claim it makes is tied to its own mechanism, because
            // the span rule can be reporting a real loss in the next line.
            assert!(
                written.contains("nothing is lost to the quiet window"),
                "cap {cap} makes an unqualified loss claim: {written}"
            );
            assert!(
                !written.contains("no footage is lost"),
                "cap {cap}: {written}"
            );
        }

        // A cap at or above the buffer is legal here, and it loses footage: a
        // motion-bearing chunk closes at min(cap, e + post), which is at least
        // the buffer's own length whichever way the race goes. 600 is the first
        // cap that costs anything — one second more than the last one that
        // does not.
        for cap in [600, 1000] {
            let written = warnings_from(&event_mode(600, cap));
            assert!(
                written.contains(&format!("neither is the {cap}s cap")),
                "cap {cap}: {written}"
            );
            assert!(
                written.contains(evicted_head),
                "cap {cap} does not point at the runtime warning: {written}"
            );
            assert!(
                !written.contains("no footage"),
                "cap {cap} contradicts itself: {written}"
            );
            assert!(
                !written.contains("every event"),
                "cap {cap} overclaims: {written}"
            );
            assert!(
                !written.contains("padding-only"),
                "cap {cap} claims the safe shape's cost: {written}"
            );
        }

        // The ordinary shape of the same pair says nothing at all.
        assert!(warnings_from(&event_mode(10, 0)).is_empty());
        assert!(warnings_from(&event_mode(599, 0)).is_empty());

        // Continuous mode has no runs to close, so the relation cannot apply.
        let continuous = format!(
            "[buffer]\nhot_duration_secs = 600\n[analytics]\nenabled = false\n[storage]\n\
             post_padding_secs = 600\nmax_event_duration_secs = 120\n{cameras}"
        );
        assert!(warnings_from(&continuous).is_empty());
    }

    /// Two relations, two mechanisms, two costs — a config can hold both, and
    /// then both are said. They used to share an if/else, where the padding
    /// relation silently swallowed the span warning even though the span is a
    /// separate thing to fix.
    #[test]
    fn each_event_mode_warning_fires_for_its_own_mechanism() {
        let cameras = one_camera("yard");
        let event_mode = |post: u64, cap: u64, pre: u64| {
            format!(
                "[buffer]\nhot_duration_secs = 600\n[analytics]\nenabled = true\n[storage]\n\
                 post_padding_secs = {post}\npre_padding_secs = {pre}\n\
                 max_event_duration_secs = {cap}\n{cameras}"
            )
        };

        // The span alone: the cap plus its reach-back does not fit, while the
        // padding is ordinary.
        let written = warnings_from(&event_mode(10, 595, 5));
        assert!(written.contains("max_event_duration_secs"), "{written}");
        assert!(!written.contains("padding-only"), "{written}");
        assert_eq!(written.matches("WARN").count(), 1, "{written}");

        // The padding alone: the span fits.
        let written = warnings_from(&event_mode(600, 120, 5));
        assert!(written.contains("padding-only"), "{written}");
        assert_eq!(written.matches("WARN").count(), 1, "{written}");

        // Both, with the cap under the buffer: neither hides the other, because
        // they are fixed by different numbers. This is also why the padding
        // line may not say "no footage is lost" — the span line two lines down
        // is saying footage is lost, and both are true of their own mechanism.
        let written = warnings_from(&event_mode(600, 595, 5));
        assert!(
            written.contains("padding-only"),
            "no padding warning: {written}"
        );
        assert!(
            written.contains("lose their opening seconds"),
            "the span warning was swallowed: {written}"
        );
        assert!(
            written.contains("nothing is lost to the quiet window"),
            "the loss claim is not tied to its mechanism: {written}"
        );
        assert!(
            !written.contains("no footage is lost"),
            "contradicts the span warning beside it: {written}"
        );
        assert_eq!(written.matches("WARN").count(), 2, "{written}");

        // Both, with the cap at or above the buffer: the span rule always fires
        // here (cap + pre >= cap >= hot), and now both lines report a loss —
        // different losses, each naming its own cause, neither denying the
        // other.
        let written = warnings_from(&event_mode(600, 1000, 5));
        assert!(written.contains("neither is the 1000s cap"), "{written}");
        assert!(
            written.contains("lose their opening seconds"),
            "the span warning was swallowed: {written}"
        );
        assert!(!written.contains("nothing is lost"), "{written}");
        assert_eq!(written.matches("WARN").count(), 2, "{written}");
    }

    /// Continuous mode is settled before any event-mode rule, so a config that
    /// is wrong in both readings is reported as the mode it will run in. The
    /// variant is the assertion: with the checks the other way round this would
    /// report a padding relation that a mode without runs cannot have.
    #[test]
    fn continuous_mode_reports_its_own_cap_rules_first() {
        let cameras = one_camera("yard");
        let continuous = |cap: u64| {
            format!(
                "[buffer]\nhot_duration_secs = 600\n[analytics]\nenabled = false\n[storage]\n\
                 post_padding_secs = 900\nmax_event_duration_secs = {cap}\n{cameras}"
            )
        };

        let err = load_cameras(&continuous(0)).unwrap_err();
        assert!(
            matches!(err, ConfigError::ZeroMaxEventDurationInContinuousMode),
            "got {err:?}"
        );
        let err = load_cameras(&continuous(600)).unwrap_err();
        assert!(
            matches!(err, ConfigError::ContinuousChunkExceedsHotBuffer { .. }),
            "got {err:?}"
        );
    }

    /// A hot buffer of 0 evicts each segment in the push that added it, so the
    /// analyzer is fed nothing and no event can be cut. Continuous mode has
    /// always caught this sideways, through the chunk that cannot fit; event
    /// mode said nothing at all.
    #[test]
    fn a_zero_hot_buffer_is_rejected() {
        let cameras = one_camera("yard");
        for analytics in ["true", "false"] {
            let toml = format!(
                "[buffer]\nhot_duration_secs = 0\n[analytics]\nenabled = {analytics}\n{cameras}"
            );
            let err = load_cameras(&toml).unwrap_err();
            assert!(
                matches!(err, ConfigError::ZeroHotDuration),
                "analytics {analytics} got {err:?}"
            );
            assert!(err.to_string().contains("hot_duration_secs"), "got {err}");
        }
        let toml =
            format!("[buffer]\nhot_duration_secs = 1\n[analytics]\nenabled = true\n{cameras}");
        load_cameras(&toml).unwrap();
    }

    /// Seconds that reach a nanosecond multiplication are bounded, each one
    /// gated where that multiplication actually runs — a value this run will
    /// never read must not stop it loading, or switching a feature off would
    /// become a way to strand a box that a stricter camon updated into.
    #[test]
    fn durations_past_the_bound_are_rejected_where_they_are_read() {
        let cameras = one_camera("yard");
        let too_large = MAX_DURATION_SECS + 1;
        // Per field: the whole config that makes its arithmetic run, and the
        // one that leaves the field unread. `{}` is the value under test.
        let cases: [(&str, &str, Option<&str>); 3] = [
            // HotBuffer::new multiplies this for every camera before any mode
            // is consulted, so no configuration leaves it unread.
            (
                "hot_duration_secs",
                "[buffer]\nhot_duration_secs = {}\n[analytics]\nenabled = false\n\
                 [storage]\nenabled = false\n",
                None,
            ),
            // app.rs multiplies this bare where the analyzer is spawned;
            // continuous recording never pads.
            (
                "pre_padding_secs",
                "[analytics]\nenabled = true\n[storage]\npre_padding_secs = {}\n",
                Some("[analytics]\nenabled = false\n[storage]\npre_padding_secs = {}\n"),
            ),
            // The continuous recorder converts it with `as_nanos() as u64`;
            // gated on the storage that owns the field.
            (
                "max_event_duration_secs",
                "[analytics]\nenabled = true\n[storage]\nenabled = true\n\
                 max_event_duration_secs = {}\n",
                Some(
                    "[analytics]\nenabled = true\n[storage]\nenabled = false\n\
                     max_event_duration_secs = {}\n",
                ),
            ),
        ];

        for (key, reads_it, ignores_it) in cases {
            let build = |template: &str, value: u64| {
                format!("{}{cameras}", template.replace("{}", &value.to_string()))
            };

            let err = load_cameras(&build(reads_it, too_large)).unwrap_err();
            assert!(
                matches!(err, ConfigError::DurationTooLarge { key: k, .. } if k == key),
                "{key} got {err:?}"
            );
            assert!(err.to_string().contains(key), "got {err}");

            // The bound itself is accepted, and it is generous: a decade of
            // padding is nothing an operator would ever ask for.
            load_cameras(&build(reads_it, MAX_DURATION_SECS))
                .unwrap_or_else(|e| panic!("{key} at the bound rejected: {e}"));

            // And a run that will never read it loads with it absurd, so the
            // value is judged on the restart that starts using it.
            if let Some(ignores_it) = ignores_it {
                load_cameras(&build(ignores_it, too_large))
                    .unwrap_or_else(|e| panic!("{key} rejected while nothing reads it: {e}"));
            }
        }

        assert_eq!(MAX_DURATION_SECS, MAX_RETENTION_DAYS * 86_400);
        // Far enough below the nanosecond wrap that no arithmetic downstream
        // can reach it.
        assert!(MAX_DURATION_SECS.checked_mul(1_000_000_000).is_some());
    }

    /// post_padding_secs is deliberately unbounded: it only ever becomes a
    /// `Duration::from_secs` compared with `saturating_duration_since`, and
    /// neither can overflow, so a bound would be a rule with no mechanism
    /// under it. The relation that does cost something is warned about
    /// separately, from a much lower threshold.
    #[test]
    fn a_post_padding_of_any_size_still_loads() {
        let toml = format!(
            "[analytics]\nenabled = true\n[storage]\npost_padding_secs = {}\n{}",
            i64::MAX,
            one_camera("yard")
        );
        let config = load_cameras(&toml).unwrap();
        assert_eq!(config.storage.post_padding_secs, i64::MAX as u64);
    }

    /// The wrap this bound is for: the pre-padding is multiplied out with no
    /// guard at all when an analyzer is spawned, which panics in a debug build
    /// and wraps in a release one.
    #[test]
    fn the_bounded_durations_survive_the_arithmetic_downstream() {
        let cameras = one_camera("yard");
        let toml = format!(
            "[buffer]\nhot_duration_secs = {MAX_DURATION_SECS}\n[analytics]\nenabled = true\n\
             [storage]\npre_padding_secs = {MAX_DURATION_SECS}\n\
             post_padding_secs = {MAX_DURATION_SECS}\n\
             max_event_duration_secs = {MAX_DURATION_SECS}\n{cameras}"
        );
        let config = load_cameras(&toml).unwrap();
        // The three multiplications that would wrap, spelled as their callers
        // spell them (app.rs, buffer::hot, buffer::warm).
        assert!(config
            .storage
            .pre_padding_secs
            .checked_mul(1_000_000_000)
            .is_some());
        assert!(config
            .buffer
            .hot_duration_secs
            .checked_mul(1_000_000_000)
            .is_some());
        let cap = std::time::Duration::from_secs(config.storage.max_event_duration_secs);
        assert!(u64::try_from(cap.as_nanos()).is_ok());
    }

    /// A zero that says something. Each of these is documented as a setting in
    /// its own right, so no pass added to `validate` may start reading it as
    /// the nonsense the ones above are.
    #[test]
    fn documented_sentinel_values_still_load() {
        let cameras = one_camera("yard");
        for (setting, section) in [
            // "Never chunk; let motion end close the event" — event mode only,
            // which the continuous check next door still refuses.
            ("max_event_duration_secs = 0", "[storage]"),
            // "No low-space guard."
            ("min_free_bytes = 0", "[storage]"),
        ] {
            let toml = format!("[analytics]\nenabled = true\n{section}\n{setting}\n{cameras}");
            load_cameras(&toml).unwrap_or_else(|e| panic!("{setting} rejected: {e}"));
        }
        // "Unlimited remote budget; rely on time-based retention alone."
        let toml = format!(
            "[storage.stathost]\nurl = \"https://host\"\nbucket = \"camon\"\n\
             token = \"t\"\nmax_stored_bytes = 0\n{cameras}"
        );
        load_cameras(&toml).unwrap();
    }

    #[test]
    fn url_password_extracts_userinfo_password() {
        assert_eq!(
            url_password("rtsp://ubnt:s3cret@10.0.0.5:554/s0"),
            Some("s3cret")
        );
        assert_eq!(url_password("rtsp://10.0.0.5:554/s0"), None);
        assert_eq!(url_password("rtsp://ubnt@10.0.0.5:554/s0"), None);
        assert_eq!(url_password("rtsp://ubnt:@10.0.0.5:554/s0"), None);
        assert_eq!(url_password("not a url"), None);
    }

    #[test]
    fn redacted_url_masks_only_the_password() {
        let camera = |url: &str| CameraConfig {
            id: "cam".to_string(),
            url: url.to_string(),
        };
        assert_eq!(
            camera("rtsp://ubnt:s3cret@10.0.0.5:554/s0").redacted_url(),
            "rtsp://ubnt:****@10.0.0.5:554/s0"
        );
        // Username identical to the password stays visible.
        assert_eq!(
            camera("rtsp://ubnt:ubnt@10.0.0.5:554/s0").redacted_url(),
            "rtsp://ubnt:****@10.0.0.5:554/s0"
        );
        assert_eq!(
            camera("rtsp://10.0.0.5:554/s0").redacted_url(),
            "rtsp://10.0.0.5:554/s0"
        );
    }
}
