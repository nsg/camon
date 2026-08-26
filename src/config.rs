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
        "camera {id:?} has an empty sub_url; remove the key when no substream is available, or \
         set it to the camera's low-resolution RTSP URL"
    )]
    EmptyCameraSubUrl { id: String },
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

/// A single `--set <dotted.path>=<value>` startup override, applied to the parsed config tree
/// before it is deserialized into [`Config`].
#[derive(Debug, Clone)]
pub struct Override {
    path: Vec<String>,
    raw: String,
}

impl Override {
    /// Parse a `dotted.path=value` argument. Splits on the first `=`; the value stays raw text
    /// until it is typed against the config schema. An argument without `=`, with an empty key,
    /// or with an empty path segment is rejected.
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

    /// The TOML scalar this override should insert, or `None` when `base` gives no answer.
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
    pub sub_url: Option<String>,
}

impl CameraConfig {
    /// The camera URL with any userinfo password masked, safe for logs.
    pub fn redacted_url(&self) -> String {
        redact_url(&self.url)
    }

    /// The optional substream URL with any userinfo password masked, safe for logs.
    pub fn redacted_sub_url(&self) -> Option<String> {
        self.sub_url.as_deref().map(redact_url)
    }
}

fn redact_url(url: &str) -> String {
    match url_password_range(url) {
        Some(range) => {
            let mut redacted = url.to_string();
            redacted.replace_range(range, "****");
            redacted
        }
        None => url.to_string(),
    }
}

/// The password embedded in a URL's userinfo (`scheme://user:pass@host/...`),
/// if any. Keeps credentials out of logs.
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
    /// Shared secret required on every `/api` request, reads included.
    #[serde(default)]
    pub token: Option<String>,
    /// Declares that something in front of camon is the authentication boundary (Home Assistant
    /// ingress, an authenticating reverse proxy).
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

/// Per-request Ollama timeout; expiry costs an object upgrade, never footage.
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

// Deterministic motion-detection defaults. These seed a camera's motion_settings.json the first
// time it is seen; thereafter the per-camera file (edited live from the web UI) wins. Ranges
// must match the clamps in `analytics::motion_settings`.
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

/// Upper bound on any `*_retention_days`.
const MAX_RETENTION_DAYS: u64 = 3650;

/// Upper bound on any `*_secs` duration in the config, and the same ten years
/// [`MAX_RETENTION_DAYS`] allows. See [`Config::validate_durations`] for the
/// arithmetic it protects.
const MAX_DURATION_SECS: u64 = MAX_RETENTION_DAYS * 86_400;

/// Reserve 2 GiB so hourly retention can catch up before the filesystem fills.
fn default_min_free_bytes() -> u64 {
    2 * 1024 * 1024 * 1024
}

fn default_stathost_enabled() -> bool {
    true
}

/// Remote "stathost" warm-storage backend (github.com/nsg/stathost).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StathostConfig {
    /// Base URL of the host, e.g. `https://files.example.com`.
    pub url: String,
    /// Bucket that events are written into.
    pub bucket: String,
    /// Per-bucket bearer token, sent as `Authorization: Bearer <token>`.
    pub token: String,
    /// Client-side storage budget in bytes.
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
    /// Cap on the wall-clock length of a single event.
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
    /// Low-space guard: before each event write, if the storage filesystem has less than this
    /// many bytes free, the oldest events are emergency-pruned (continuous → movements →
    /// objects) until space recovers. 0 disables the guard.
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

/// Opt-in: an update is only checked against the sha256sums.txt published beside it in the same
/// GitHub release, which protects against a corrupt download but not against a tampered
/// release, and the installed service runs as root.
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

/// Snapshot cadence while motion is active. Snapshots are motion-gated by design (see
/// `crate::mqtt`), so this only ever costs decode work during a run — 5s is a compromise
/// between a responsive HA camera tile and the per-frame ffmpeg decode.
fn default_mqtt_snapshot_interval_secs() -> u64 {
    5
}

/// How long an occupancy sensor stays ON after the last sighting of its class.
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
    /// The file this config was read from, kept so camon can put things beside it — today
    /// only the API token it generates for an otherwise-open deployment, which belongs where
    /// the operator already looks for camon's settings.
    #[serde(skip)]
    source_path: Option<PathBuf>,
}

/// Where a generated API token is kept, relative to the config file that did
/// not name one. See [`crate::api::ApiAuth`].
const API_TOKEN_FILE: &str = "api-token";

impl Config {
    /// The file a generated API token is read from and written to: beside the config file, so
    /// `/etc/camon/config.toml` puts it at `/etc/camon/api-token`.
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

    /// Load from an explicit TOML path, applying each `--set` override into the parsed value
    /// tree before deserializing. Overrides win over file values.
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
            // Unknown keys need both plain and string-valued parses because secrets may look
            // numeric while siblings require their native types.
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

    /// Canonicalize values that several consumers have to agree on.
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
            // Continuous recording (storage on, analytics off) has no motion run to close a
            // chunk, so the cap is the only thing that rolls one. In event mode 0 is a real
            // setting: don't chunk, and let motion end close the event.
            if !self.analytics.enabled {
                if cap == 0 {
                    return Err(ConfigError::ZeroMaxEventDurationInContinuousMode);
                }
                // Continuous chunks are cut straight from the buffer with no padding at all, so
                // the cap is the whole span. Fatal because `plan_continuous_roll` can then
                // never fire: nothing at all is written, rather than something imperfect.
                if cap >= self.buffer.hot_duration_secs {
                    return Err(ConfigError::ContinuousChunkExceedsHotBuffer {
                        cap,
                        hot: self.buffer.hot_duration_secs,
                    });
                }
            } else {
                // Quiet-window and event-span losses are independent, so report both.
                let post = self.storage.post_padding_secs;
                let hot = self.buffer.hot_duration_secs;
                let evicted_head = crate::buffer::warm::EVICTED_HEAD_WARNING;

                // Run closure is clock-based and independent of buffer residency.
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
                        // A cap below the hot window preserves each chunk, but padding after the
                        // last cap boundary belongs to no chunk.
                        tracing::warn!(
                            post_padding_secs = post,
                            hot_duration_secs = hot,
                            cap,
                            "[storage] post_padding_secs ({post}) is not shorter than [buffer] \
                             hot_duration_secs ({hot}): the {cap}s cap closes every chunk while \
                             its footage is still in the buffer, so nothing is lost to the quiet \
                             window — but the quiet past a run's last chunk is recorded nowhere, \
                             so this buys at most {cap}s of trailing context while still taking \
                             up to ~{post}s to end the run. Lower post_padding_secs — it is 10 \
                             unless you set it"
                        );
                    } else {
                        // `cap >= hot` is legal in event mode (only continuous recording
                        // refuses it), and here it costs footage.
                        tracing::warn!(
                            post_padding_secs = post,
                            hot_duration_secs = hot,
                            cap,
                            "[storage] post_padding_secs ({post}) is not shorter than [buffer] \
                             hot_duration_secs ({hot}), and neither is the {cap}s cap: whichever \
                             of them closes a chunk, that chunk is at least {hot}s old by then, \
                             so its oldest footage has already been evicted (\"{evicted_head}\" \
                             at runtime). No chunk escapes it, because none opens on padding — \
                             every one of them holds motion, and the motion it opened on is \
                             exactly what it loses. Bring max_event_duration_secs under \
                             hot_duration_secs, and lower post_padding_secs — it is 10 unless \
                             you set it"
                        );
                    }
                }

                // A cap plus pre-padding wider than the hot buffer loses the event head,
                // independently of the quiet-window relation above.
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

        // An empty allowlist reads as either "detect the defaults" or "detect nothing"
        // depending on who is asked, and both readings are already spelled out unambiguously
        // elsewhere: omit the key, or set enabled = false.
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
            if camera
                .sub_url
                .as_ref()
                .is_some_and(|url| url.trim().is_empty())
            {
                return Err(ConfigError::EmptyCameraSubUrl {
                    id: camera.id.clone(),
                });
            }
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

    /// Reject values that silently disable enabled analytics; repair values with an
    /// unambiguous correction.
    fn validate_analytics(&self) -> Result<(), ConfigError> {
        if !self.analytics.enabled {
            return Ok(());
        }

        if self.analytics.sample_fps == 0 {
            return Err(ConfigError::ZeroSampleFps);
        }

        // NaN has no nearest valid slider value for `repair` to choose.
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
        // Finite thresholds can be clamped; NaN would silently admit every detection.
        let confidence = self.analytics.object_detection.confidence_threshold;
        if !confidence.is_finite() {
            return Err(ConfigError::ConfidenceThresholdNotANumber { value: confidence });
        }

        Ok(())
    }

    /// Values camon corrects instead of refusing, warning as it does.
    fn repair(&mut self) {
        if !self.analytics.enabled {
            return;
        }

        // The same correction the settings store would make silently, made once, out loud, over
        // the value the operator actually wrote.
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

        // Clamped rather than refused, matching the motion sliders above: one policy for "a
        // real number, out of its range" across the whole file.
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

        // Substituted rather than refused: "0 means no timeout" is a widespread convention and
        // someone may well have written it meaning that.
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

    /// Seconds that are multiplied out to nanoseconds somewhere downstream, and the one
    /// duration that must not be zero.
    fn validate_durations(&self) -> Result<(), ConfigError> {
        // A zero-length hot buffer evicts every segment in the same call that pushed it, so the
        // analyzer is fed nothing, no event can be assembled from a buffer that holds none, and
        // continuous recording has nothing to cut.
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

/// The span of an event that will not fit in the hot buffer, or `None` when it fits.
fn event_span_overrun(cap: u64, pre: u64, hot: u64) -> Option<u64> {
    let total = cap.saturating_add(pre);
    (total >= hot).then_some(total)
}

/// Keys camon itself shipped and later removed, with the advice to print when one turns up.
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

/// A camera id is used verbatim as a directory name under the storage `data_dir`, so it must be
/// a single path component.
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

    fn apply_override(spec: &str, tree: &mut toml::Value) {
        let ov = Override::parse(spec).unwrap();
        let reading = ov.reading_against(&empty_tree()).unwrap_or_else(|| {
            // Without a schema reading, infer the scalar from its text.
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

    #[test]
    fn every_setting_in_the_example_can_be_judged_on_its_own() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml.example");
        let example: toml::Value = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut settings = Vec::new();
        collect_scalars(&example, &mut Vec::new(), &mut settings);
        assert!(settings.len() > 20, "only found {}", settings.len());

        for (path, value) in settings {
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
        assert_eq!(config.http.port, 22666);
        assert!(config.update.enabled);
    }

    #[test]
    fn http_defaults_to_all_interfaces_with_nothing_configured() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let config = Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap();
        assert_eq!(config.http.bind, "0.0.0.0");
        assert_eq!(config.http.bind_addr(), IpAddr::from([0, 0, 0, 0]));
        assert!(config.http.token.is_none());
        assert!(!config.http.allow_open);
    }

    #[test]
    fn a_generated_token_is_kept_beside_the_config_file() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let path = dir.path().join("config.toml");
        let config = Config::load_from_with_overrides(&path, &[]).unwrap();
        assert_eq!(config.token_file_path(), Some(dir.path().join("api-token")));

        let mut relative = config.clone();
        relative.source_path = Some(PathBuf::from(DEFAULT_CONFIG_PATH));
        assert_eq!(relative.token_file_path(), Some(PathBuf::from("api-token")));

        let from_text: Config = toml::from_str(TOML_SAMPLE).unwrap();
        assert_eq!(from_text.token_file_path(), None);
    }

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

    #[test]
    fn substream_url_parses() {
        let with_substream = load_cameras(
            "[[cameras]]\nid = \"yard\"\nurl = \"rtsp://main/stream\"\n\
             sub_url = \"rtsp://sub/stream\"\n",
        )
        .unwrap();
        assert_eq!(
            with_substream.cameras[0].sub_url.as_deref(),
            Some("rtsp://sub/stream")
        );
    }

    #[test]
    fn substream_url_is_optional() {
        let without_substream = load_cameras(&one_camera("yard")).unwrap();
        assert_eq!(without_substream.cameras[0].sub_url, None);
    }

    #[test]
    fn empty_or_blank_substream_urls_are_rejected() {
        for sub_url in ["", "   ", "\t"] {
            let toml = format!(
                "[[cameras]]\nid = \"yard\"\nurl = \"rtsp://main/stream\"\nsub_url = {sub_url:?}\n"
            );
            let err = load_cameras(&toml).unwrap_err();
            assert!(
                matches!(err, ConfigError::EmptyCameraSubUrl { .. }),
                "got {err:?} for {sub_url:?}"
            );
        }
    }

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

        let event_mode = format!(
            "[analytics]\nenabled = true\n[storage]\nmax_event_duration_secs = 0\n{cameras}"
        );
        let config = load_cameras(&event_mode).unwrap();
        assert_eq!(config.storage.max_event_duration_secs, 0);
    }

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

    #[test]
    fn the_event_span_counts_pre_padding_only() {
        assert_eq!(event_span_overrun(594, 5, 600), None);
        assert_eq!(event_span_overrun(595, 5, 600), Some(600));
        assert_eq!(event_span_overrun(590, 0, 600), None);
        assert_eq!(event_span_overrun(u64::MAX, 5, 600), Some(u64::MAX));
    }

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
            let off = format!("[storage]\nenabled = false\n{key} = 0\n{cameras}");
            load_cameras(&off).unwrap_or_else(|e| panic!("{key} = 0 with storage off: {e}"));
        }
    }

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

    #[test]
    fn example_config_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml.example");
        let config = Config::load_from_with_overrides(&path, &[]).unwrap();
        assert_eq!(config.cameras.len(), 1);
        assert_eq!(config.buffer.hot_duration_secs, 600);
    }

    fn addon_run_sh_overrides() -> Vec<String> {
        const RUN_SH: &str = include_str!("../camon-addon/run.sh");
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
        assert!(strip_retired_keys(&mut value).is_empty());
    }

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

    #[test]
    fn zero_sample_fps_is_rejected() {
        let cameras = one_camera("yard");
        let toml = format!("[analytics]\nenabled = true\nsample_fps = 0\n{cameras}");
        let err = load_cameras(&toml).unwrap_err();
        assert!(matches!(err, ConfigError::ZeroSampleFps), "got {err:?}");
        assert!(err.to_string().contains("sample_fps"), "got {err}");

        load_cameras(&format!(
            "[analytics]\nenabled = true\nsample_fps = 1\n{cameras}"
        ))
        .unwrap();
        load_cameras(&format!(
            "[analytics]\nenabled = false\nsample_fps = 0\n{cameras}"
        ))
        .unwrap();
    }

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

        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.motion]\nvar_threshold = 1000\n{cameras}"
        );
        assert_eq!(
            load_cameras(&toml).unwrap().analytics.motion.var_threshold,
            motion_settings::VAR_THRESHOLD_MAX
        );
    }

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
        for literal in ["0.0", "0.5", "1.0"] {
            let toml = format!("{detection}\nconfidence_threshold = {literal}\n{cameras}");
            load_cameras(&toml).unwrap_or_else(|e| panic!("{literal} rejected: {e}"));
        }
        let toml = format!(
            "[analytics]\nenabled = true\n[analytics.object_detection]\nenabled = false\n\
             confidence_threshold = nan\n{cameras}"
        );
        load_cameras(&toml).unwrap();
    }

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

        let written = warnings_from(&event_mode(600, 0));
        assert!(written.contains("every event"), "{written}");
        assert!(
            written.contains(evicted_head),
            "does not point at the runtime warning: {written}"
        );

        for cap in [120, 599] {
            let written = warnings_from(&event_mode(600, cap));
            assert!(written.contains("recorded nowhere"), "cap {cap}: {written}");
            assert!(
                written.contains(&format!("at most {cap}s of trailing context")),
                "cap {cap} promises the padding the setting names: {written}"
            );
            assert!(
                written.contains("up to ~600s"),
                "cap {cap} states the wait as a certainty: {written}"
            );
            assert!(
                !written.contains("every event"),
                "cap {cap} overclaims: {written}"
            );
            assert!(
                !written.contains(evicted_head),
                "cap {cap} promises a runtime warning this shape never emits: {written}"
            );
            assert!(
                written.contains("nothing is lost to the quiet window"),
                "cap {cap} makes an unqualified loss claim: {written}"
            );
            assert!(
                !written.contains("no footage is lost"),
                "cap {cap}: {written}"
            );
        }

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
                !written.contains("recorded nowhere"),
                "cap {cap} claims the safe shape's cost: {written}"
            );
            for absolution in ["only padding", "lose nothing", "loses nothing"] {
                assert!(
                    !written.contains(absolution),
                    "cap {cap} offers a chunk that loses nothing: {written}"
                );
            }
        }

        assert!(warnings_from(&event_mode(10, 0)).is_empty());
        assert!(warnings_from(&event_mode(599, 0)).is_empty());

        let continuous = format!(
            "[buffer]\nhot_duration_secs = 600\n[analytics]\nenabled = false\n[storage]\n\
             post_padding_secs = 600\nmax_event_duration_secs = 120\n{cameras}"
        );
        assert!(warnings_from(&continuous).is_empty());
    }

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

        let written = warnings_from(&event_mode(10, 595, 5));
        assert!(written.contains("max_event_duration_secs"), "{written}");
        assert!(!written.contains("recorded nowhere"), "{written}");
        assert_eq!(written.matches("WARN").count(), 1, "{written}");

        let written = warnings_from(&event_mode(600, 120, 5));
        assert!(written.contains("recorded nowhere"), "{written}");
        assert_eq!(written.matches("WARN").count(), 1, "{written}");

        let written = warnings_from(&event_mode(600, 595, 5));
        assert!(
            written.contains("recorded nowhere"),
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

        let written = warnings_from(&event_mode(600, 1000, 5));
        assert!(written.contains("neither is the 1000s cap"), "{written}");
        assert!(
            written.contains("lose their opening seconds"),
            "the span warning was swallowed: {written}"
        );
        assert!(!written.contains("nothing is lost"), "{written}");
        assert!(!written.contains("only padding"), "{written}");
        assert_eq!(written.matches("WARN").count(), 2, "{written}");
    }

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

    #[test]
    fn durations_past_the_bound_are_rejected_where_they_are_read() {
        let cameras = one_camera("yard");
        let too_large = MAX_DURATION_SECS + 1;
        let cases: [(&str, &str, Option<&str>); 3] = [
            (
                "hot_duration_secs",
                "[buffer]\nhot_duration_secs = {}\n[analytics]\nenabled = false\n\
                 [storage]\nenabled = false\n",
                None,
            ),
            (
                "pre_padding_secs",
                "[analytics]\nenabled = true\n[storage]\npre_padding_secs = {}\n",
                Some("[analytics]\nenabled = false\n[storage]\npre_padding_secs = {}\n"),
            ),
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

            load_cameras(&build(reads_it, MAX_DURATION_SECS))
                .unwrap_or_else(|e| panic!("{key} at the bound rejected: {e}"));

            if let Some(ignores_it) = ignores_it {
                load_cameras(&build(ignores_it, too_large))
                    .unwrap_or_else(|e| panic!("{key} rejected while nothing reads it: {e}"));
            }
        }

        assert_eq!(MAX_DURATION_SECS, MAX_RETENTION_DAYS * 86_400);
        assert!(MAX_DURATION_SECS.checked_mul(1_000_000_000).is_some());
    }

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

    #[test]
    fn documented_sentinel_values_still_load() {
        let cameras = one_camera("yard");
        for (setting, section) in [
            ("max_event_duration_secs = 0", "[storage]"),
            ("min_free_bytes = 0", "[storage]"),
        ] {
            let toml = format!("[analytics]\nenabled = true\n{section}\n{setting}\n{cameras}");
            load_cameras(&toml).unwrap_or_else(|e| panic!("{setting} rejected: {e}"));
        }
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
            sub_url: None,
        };
        assert_eq!(
            camera("rtsp://ubnt:s3cret@10.0.0.5:554/s0").redacted_url(),
            "rtsp://ubnt:****@10.0.0.5:554/s0"
        );
        assert_eq!(
            camera("rtsp://ubnt:ubnt@10.0.0.5:554/s0").redacted_url(),
            "rtsp://ubnt:****@10.0.0.5:554/s0"
        );
        assert_eq!(
            camera("rtsp://10.0.0.5:554/s0").redacted_url(),
            "rtsp://10.0.0.5:554/s0"
        );
    }

    #[test]
    fn redacted_substream_url_masks_its_password() {
        let camera = CameraConfig {
            id: "cam".to_string(),
            url: "rtsp://main:secret@10.0.0.5/main".to_string(),
            sub_url: Some("rtsp://sub:another-secret@10.0.0.5/sub".to_string()),
        };
        assert_eq!(
            camera.redacted_sub_url().as_deref(),
            Some("rtsp://sub:****@10.0.0.5/sub")
        );
    }
}
