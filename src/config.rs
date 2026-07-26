use serde::Deserialize;
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
pub struct CameraConfig {
    pub id: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
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

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            hot_duration_secs: default_hot_duration(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpConfig {
    #[serde(default = "default_http_port")]
    pub port: u16,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            port: default_http_port(),
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
pub struct OllamaServerConfig {
    #[serde(default = "default_ollama_url")]
    pub url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
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
    /// split into chained, independently playable chunks. 0 disables the cap.
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
pub struct UpdateConfig {
    #[serde(default = "default_update_enabled")]
    pub enabled: bool,
}

fn default_update_enabled() -> bool {
    true
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
        let config: Config = value.try_into()?;

        if config.cameras.is_empty() {
            return Err(ConfigError::NoCameras);
        }

        Ok(config)
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
            Override::parse("update.enabled=false").unwrap(),
        ];
        let config =
            Config::load_from_with_overrides(dir.path().join("config.toml"), &overrides).unwrap();
        // File said 9090 / default-true; the overrides win.
        assert_eq!(config.http.port, 22666);
        assert!(!config.update.enabled);
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
}
