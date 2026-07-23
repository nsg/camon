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

#[derive(Debug, Clone, Deserialize)]
pub struct WarmConfig {
    #[serde(default = "default_warm_enabled")]
    pub enabled: bool,
    #[serde(default = "default_warm_data_dir")]
    pub data_dir: String,
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
    pub cameras: Vec<CameraConfig>,
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_from(DEFAULT_CONFIG_PATH)
    }

    pub fn load_from<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;

        if config.cameras.is_empty() {
            return Err(ConfigError::NoCameras);
        }

        Ok(config)
    }
}
