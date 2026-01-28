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

fn default_motion_threshold() -> f32 {
    0.05
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyticsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_sample_fps")]
    pub sample_fps: u32,
    #[serde(default = "default_motion_threshold")]
    pub motion_threshold: f32,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_fps: default_sample_fps(),
            motion_threshold: default_motion_threshold(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub buffer: BufferConfig,
    #[serde(default)]
    pub http: HttpConfig,
    #[serde(default)]
    pub analytics: AnalyticsConfig,
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
