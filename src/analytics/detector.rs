//! Shared object-detector types and backend dispatch.

use super::{OllamaClient, TpueClient};

/// One validated object detection in normalized image coordinates.
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    /// Normalized bounding box (x, y, w, h) in 0.0-1.0 image coordinates.
    pub bbox: Option<(f32, f32, f32, f32)>,
}

/// Result from a single frame detection call.
#[derive(Debug)]
pub struct FrameDetectResult {
    pub detections: Vec<Detection>,
    pub raw_response: String,
    pub model: String,
}

/// Configured object-detection backend.
pub enum Detector {
    Ollama(OllamaClient),
    Tpue(TpueClient),
}

impl Detector {
    /// Stable backend name persisted with detections and verdicts.
    pub fn backend(&self) -> &'static str {
        match self {
            Self::Ollama(_) => "ollama",
            Self::Tpue(_) => "tpue",
        }
    }

    /// Configured model name, or the backend's default placeholder.
    pub fn model(&self) -> &str {
        match self {
            Self::Ollama(client) => client.model(),
            Self::Tpue(client) => client.model(),
        }
    }

    /// Configured object-class allowlist.
    pub fn allowed_classes(&self) -> &[String] {
        match self {
            Self::Ollama(client) => client.allowed_classes(),
            Self::Tpue(client) => client.allowed_classes(),
        }
    }

    /// Run the backend's startup sanity checks. Failures are logged because a
    /// temporarily unavailable detector must not prevent recording.
    pub async fn check_ready(&self) {
        match self {
            Self::Ollama(client) => client.check_models().await,
            Self::Tpue(client) => client.check_ready().await,
        }
    }

    /// Detect objects in one JPEG-encoded frame.
    pub async fn detect_jpeg(
        &self,
        jpeg: &[u8],
    ) -> Result<FrameDetectResult, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Ollama(client) => client.detect_jpeg(jpeg).await,
            Self::Tpue(client) => client.detect_jpeg(jpeg).await,
        }
    }
}

/// Validate and normalize a bounding box. The origin must lie inside the
/// frame and the size must be positive; a box slightly overhanging the right
/// or bottom edge (model slop) is clamped back in. Anything else is garbage.
pub(super) fn validate_bbox(x: f32, y: f32, w: f32, h: f32) -> Option<(f32, f32, f32, f32)> {
    if ![x, y, w, h].iter().all(|v| v.is_finite()) {
        return None;
    }
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) || w <= 0.0 || h <= 0.0 {
        return None;
    }
    if w > 1.0 || h > 1.0 {
        return None;
    }
    let w = w.min(1.0 - x);
    let h = h.min(1.0 - y);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    Some((x, y, w, h))
}
