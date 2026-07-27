use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::Path;
use thiserror::Error;

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
/// over the file's values and can create missing intermediate tables.
#[derive(Debug, Clone)]
pub struct Override {
    path: Vec<String>,
    value: toml::Value,
}

impl Override {
    /// Parse a `dotted.path=value` argument. Splits on the first `=`; the value
    /// is coerced to the first TOML scalar that accepts it — bool, then
    /// integer, then float, otherwise a plain string. An argument without `=`,
    /// with an empty key, or with an empty path segment is rejected.
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
            value: parse_scalar(raw),
        })
    }

    /// Apply this override into `root`, walking the dotted path and creating
    /// (or replacing non-table) intermediate tables as needed.
    fn apply(&self, root: &mut toml::Value) {
        let (last, parents) = self
            .path
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
            .insert(last.clone(), self.value.clone());
    }
}

/// Coerce a raw `--set` value to a TOML scalar: bool, then integer, then float,
/// otherwise a string.
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
    /// Shared secret required on every `/api` request. `None` (the default)
    /// leaves the API open to anyone who can reach the port.
    #[serde(default)]
    pub token: Option<String>,
    /// Suppresses the open-API startup warning for deployments where an outer
    /// layer authenticates (e.g. Home Assistant ingress).
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
}

impl Config {
    /// Load from the default `config.toml` in the current working directory.
    pub fn load(overrides: &[Override]) -> Result<Self, ConfigError> {
        Self::load_from_with_overrides(DEFAULT_CONFIG_PATH, overrides)
    }

    /// Load from an explicit TOML path, applying each `--set` override into the
    /// parsed value tree before deserializing. Overrides win over file values.
    pub fn load_from_with_overrides<P: AsRef<Path>>(
        path: P,
        overrides: &[Override],
    ) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let mut value: toml::Value = toml::from_str(&content)?;
        for ov in overrides {
            ov.apply(&mut value);
        }
        for (key, advice) in strip_retired_keys(&mut value) {
            tracing::warn!(key = %key, "ignoring retired config key: {advice}");
        }
        let mut config: Config = value.try_into()?;
        config.normalize();
        config.validate()?;
        Ok(config)
    }

    /// Canonicalize values that several consumers have to agree on. Object
    /// classes are matched case-insensitively by the Ollama client but compared
    /// verbatim by the MQTT bridge, so `classes = ["Person"]` would otherwise
    /// yield an occupancy sensor that can never turn on. Deduplicating is part
    /// of the same fix: once folded, `["Person", "person"]` would produce two
    /// discovery payloads sharing one unique id.
    fn normalize(&mut self) {
        let classes = &mut self.analytics.object_detection.classes;
        for class in classes.iter_mut() {
            *class = class.to_lowercase();
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

        // The cap is only read by the warm writer and the continuous recorder,
        // neither of which is spawned with storage off.
        if self.storage.enabled {
            let cap = self.storage.max_event_duration_secs;
            // Continuous recording (storage on, analytics off) has no motion
            // run to close a chunk, so the cap is the only thing that rolls
            // one. In event mode 0 is a real setting: don't chunk, and let
            // motion end close the event.
            if cap == 0 {
                if !self.analytics.enabled {
                    return Err(ConfigError::ZeroMaxEventDurationInContinuousMode);
                }
            } else if !self.analytics.enabled {
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
            } else if let Some(total) = event_span_overrun(
                cap,
                self.storage.pre_padding_secs,
                self.buffer.hot_duration_secs,
            ) {
                // A warning, not an error: unlike continuous mode this still
                // records, it just loses the head of events that run long —
                // the degradation camon already has today. Refusing to boot
                // over it would be the worse trade, since the cap is 120 by
                // default (so it may never have been written), trimming
                // hot_duration_secs is exactly what a RAM-pressured box would
                // do, and config load precedes the updater, so an auto-updating
                // install would stay down until someone read the log.
                tracing::warn!(
                    cap,
                    pre_padding_secs = self.storage.pre_padding_secs,
                    total,
                    hot_duration_secs = self.buffer.hot_duration_secs,
                    "[storage] max_event_duration_secs ({cap}) plus pre_padding_secs ({}) is \
                     {total}s, which does not fit in [buffer] hot_duration_secs ({}): events \
                     running longer than about {}s will lose their opening seconds to \
                     eviction. Raise hot_duration_secs, or lower max_event_duration_secs — it \
                     is 120 unless you set it",
                    self.storage.pre_padding_secs,
                    self.buffer.hot_duration_secs,
                    self.buffer
                        .hot_duration_secs
                        .saturating_sub(self.storage.pre_padding_secs),
                );
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

    #[test]
    fn override_sets_a_bool() {
        let mut v: toml::Value = toml::from_str("[update]\nenabled = true\n").unwrap();
        Override::parse("update.enabled=false")
            .unwrap()
            .apply(&mut v);
        assert_eq!(v["update"]["enabled"].as_bool(), Some(false));
    }

    #[test]
    fn override_sets_an_integer() {
        let mut v: toml::Value = toml::from_str("[http]\nport = 8080\n").unwrap();
        Override::parse("http.port=22666").unwrap().apply(&mut v);
        assert_eq!(v["http"]["port"].as_integer(), Some(22666));
    }

    #[test]
    fn override_sets_a_string() {
        let mut v: toml::Value = toml::from_str("[storage]\ndata_dir = \"/var/camon\"\n").unwrap();
        Override::parse("storage.data_dir=/data/storage")
            .unwrap()
            .apply(&mut v);
        assert_eq!(v["storage"]["data_dir"].as_str(), Some("/data/storage"));
    }

    #[test]
    fn override_creates_missing_intermediate_tables() {
        let mut v: toml::Value = toml::from_str("").unwrap();
        Override::parse("analytics.object_detection.enabled=true")
            .unwrap()
            .apply(&mut v);
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

    #[test]
    fn http_defaults_to_open_on_all_interfaces() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let config = Config::load_from_with_overrides(dir.path().join("config.toml"), &[]).unwrap();
        assert_eq!(config.http.bind, "0.0.0.0");
        assert_eq!(config.http.bind_addr(), IpAddr::from([0, 0, 0, 0]));
        assert!(config.http.token.is_none());
        assert!(!config.http.allow_open);
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

    /// Mirrors the `--set` list in `camon-addon/run.sh:90-96` (plus the
    /// conditional MQTT block above it), which the add-on forces on every
    /// start. A key renamed here is a container that will not boot.
    #[test]
    fn addon_overrides_still_apply() {
        let dir = write_temp("config.toml", TOML_SAMPLE);
        let overrides: Vec<Override> = [
            "update.enabled=false",
            "http.port=22666",
            "http.bind=0.0.0.0",
            "http.allow_open=true",
            "storage.data_dir=/data/storage",
            "mqtt.enabled=true",
            "mqtt.host=core-mosquitto",
            "mqtt.port=1883",
            "mqtt.username=addons",
            "mqtt.password=s3cr3t",
        ]
        .iter()
        .map(|a| Override::parse(a).unwrap())
        .collect();
        let config =
            Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides).unwrap();
        assert!(!config.update.enabled);
        assert_eq!(config.http.port, 22666);
        assert!(config.http.allow_open);
        assert_eq!(config.storage.data_dir, "/data/storage");
        assert!(config.mqtt.enabled);
        assert_eq!(config.mqtt.host, "core-mosquitto");
        assert_eq!(config.mqtt.username.as_deref(), Some("addons"));
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
