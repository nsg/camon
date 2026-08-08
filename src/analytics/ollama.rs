//! Ollama vision-model client for object detection.

use base64::Engine;
use serde::{Deserialize, Serialize};

/// TCP connect timeout. A down or unreachable server fails in seconds instead
/// of eating the whole request timeout.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Hard cap on generated tokens. Fifteen schema-shaped detections fit in well
/// under half of this; the cap only exists to bound a runaway generation.
const NUM_PREDICT: u32 = 768;

/// Cap on the detections array in the response schema. Keeps a degenerate
/// "everything is a person" response bounded in both tokens and latency.
const MAX_DETECTIONS: usize = 15;

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    /// Normalized bounding box (x, y, w, h) in 0.0-1.0 image coordinates.
    pub bbox: Option<(f32, f32, f32, f32)>,
}

/// Result from a single frame detection call.
pub struct FrameDetectResult {
    pub detections: Vec<Detection>,
    pub raw_response: String,
    pub model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    /// JSON schema for structured output (Ollama constrains decoding to it).
    format: serde_json::Value,
    options: RequestOptions,
}

#[derive(Serialize)]
struct RequestOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<String>,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Option<ResponseMessage>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

/// The shape the response schema enforces; serde does the strict parse.
#[derive(Deserialize)]
struct DetectionsPayload {
    detections: Vec<RawDetection>,
}

#[derive(Deserialize)]
struct RawDetection {
    class: String,
    confidence: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

struct OllamaServer {
    base_url: String,
    model: String,
}

pub struct OllamaClient {
    client: reqwest::Client,
    primary: OllamaServer,
    fallback: Option<OllamaServer>,
    confidence_threshold: f32,
    /// The configured allowlist, lowercased.
    allowed_classes: Vec<String>,
}

impl OllamaClient {
    pub fn new(
        base_url: &str,
        model: &str,
        timeout_secs: u64,
        confidence_threshold: f32,
        allowed_classes: Vec<String>,
        fallback: Option<(&str, &str)>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;

        let primary = OllamaServer {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        };
        let fallback = fallback.map(|(url, model)| OllamaServer {
            base_url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        });

        let allowed_classes = allowed_classes
            .into_iter()
            .map(|c| c.to_lowercase())
            .collect();

        Ok(Self {
            client,
            primary,
            fallback,
            confidence_threshold,
            allowed_classes,
        })
    }

    pub fn model(&self) -> &str {
        &self.primary.model
    }

    /// The classes this client will ask the model about, for callers that have
    /// to publish the same set (the MQTT bridge's occupancy entities).
    pub fn allowed_classes(&self) -> &[String] {
        &self.allowed_classes
    }

    /// Startup sanity check: ask each configured server for its pulled models (`/api/tags`) and
    /// warn loudly if the configured model is missing.
    pub async fn check_models(&self) {
        for server in std::iter::once(&self.primary).chain(self.fallback.as_ref()) {
            let url = format!("{}/api/tags", server.base_url);
            let names: Vec<String> = match self.client.get(&url).send().await {
                Ok(resp) => match resp.json::<TagsResponse>().await {
                    Ok(tags) => tags.models.into_iter().map(|m| m.name).collect(),
                    Err(e) => {
                        tracing::warn!(url = %server.base_url, error = %e,
                            "could not parse ollama /api/tags response, skipping model check");
                        continue;
                    }
                },
                Err(e) => {
                    tracing::warn!(url = %server.base_url, error = %e,
                        "ollama server unreachable, skipping model check");
                    continue;
                }
            };
            // A model configured without a tag implies ":latest" on the server.
            let with_latest = format!("{}:latest", server.model);
            if names
                .iter()
                .any(|n| *n == server.model || *n == with_latest)
            {
                tracing::info!(url = %server.base_url, model = %server.model,
                    "ollama model available");
            } else {
                tracing::warn!(
                    url = %server.base_url,
                    model = %server.model,
                    available = ?names,
                    "configured model is NOT pulled on the ollama server — object \
                     detection will fail until you run: ollama pull {}",
                    server.model
                );
            }
        }
    }

    /// Detect objects in one JPEG-encoded frame. Tries the primary server,
    /// then the fallback (if configured). The caller (the serial detection
    /// worker) guarantees no other request is in flight.
    pub async fn detect_jpeg(
        &self,
        jpeg: &[u8],
    ) -> Result<FrameDetectResult, Box<dyn std::error::Error + Send + Sync>> {
        let image_b64 = base64::engine::general_purpose::STANDARD.encode(jpeg);

        match self.call_server(&self.primary, &image_b64).await {
            Ok(result) => Ok(result),
            Err(primary_err) => {
                if let Some(ref fallback) = self.fallback {
                    tracing::warn!(
                        error = %primary_err,
                        "primary ollama failed, trying fallback"
                    );
                    self.call_server(fallback, &image_b64).await.map_err(|e| {
                        format!("both servers failed — primary: {primary_err}, fallback: {e}")
                            .into()
                    })
                } else {
                    Err(primary_err)
                }
            }
        }
    }

    async fn call_server(
        &self,
        server: &OllamaServer,
        image_b64: &str,
    ) -> Result<FrameDetectResult, Box<dyn std::error::Error + Send + Sync>> {
        let request = ChatRequest {
            model: server.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: self.build_prompt(),
                images: vec![image_b64.to_string()],
            }],
            stream: false,
            format: build_format_schema(&self.allowed_classes),
            options: RequestOptions {
                temperature: 0.0,
                num_predict: NUM_PREDICT,
            },
        };

        let url = format!("{}/api/chat", server.base_url);
        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("ollama API error {}: {}", status, body).into());
        }

        let chat_response: ChatResponse = response.json().await?;
        let content = chat_response.message.map(|m| m.content).unwrap_or_default();
        let detections =
            parse_detections(&content, self.confidence_threshold, &self.allowed_classes);

        Ok(FrameDetectResult {
            detections,
            raw_response: content,
            model: server.model.clone(),
        })
    }

    fn build_prompt(&self) -> String {
        let classes = self.allowed_classes.join(", ");
        format!(
            "Security camera frame. List objects: {classes}.\n\
             Return JSON matching the schema: a \"detections\" array. Each detection has \
             \"class\" (one of the listed objects), \"confidence\" (0.0-1.0), and bounding box \
             \"x\",\"y\",\"w\",\"h\" as fractions of image size (0.0-1.0), where x,y is the \
             top-left corner and w,h are the width and height.\n\
             If nothing noteworthy, return an empty detections array. No other text."
        )
    }
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
}

/// The JSON schema sent in the request's `format` field. Flat numeric-only
/// detection objects, class constrained to the allowlist enum, and a hard cap
/// on the array length.
fn build_format_schema(allowed_classes: &[String]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "maxItems": MAX_DETECTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {"type": "string", "enum": allowed_classes},
                        "confidence": {"type": "number"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "w": {"type": "number"},
                        "h": {"type": "number"},
                    },
                    "required": ["class", "confidence", "x", "y", "w", "h"],
                },
            }
        },
        "required": ["detections"],
    })
}

/// Parse a structured response and apply semantic validation.
fn parse_detections(content: &str, threshold: f32, allowed_classes: &[String]) -> Vec<Detection> {
    let payload: DetectionsPayload = match serde_json::from_str(content) {
        Ok(p) => p,
        Err(e) => match salvage_truncated(content) {
            // Ollama may ignore maxItems and hit num_predict mid-array; complete leading
            // detections remain valid.
            Some(p) => {
                tracing::warn!(
                    count = p.detections.len(),
                    "ollama response truncated at token cap, salvaged complete detections"
                );
                p
            }
            None => {
                tracing::warn!(error = %e, raw = %content,
                    "ollama response is not valid schema JSON, dropping");
                return Vec::new();
            }
        },
    };

    let mut detections = Vec::new();
    // maxItems is advisory to the model; enforce the cap for real here.
    for raw in payload.detections.into_iter().take(MAX_DETECTIONS) {
        let class_name = raw.class.to_lowercase();
        if !allowed_classes.contains(&class_name) {
            tracing::debug!(class = %raw.class, "dropping detection with class outside allowlist");
            continue;
        }
        if !raw.confidence.is_finite() || !(0.0..=1.0).contains(&raw.confidence) {
            tracing::debug!(class = %class_name, confidence = raw.confidence,
                "dropping detection with nonsensical confidence");
            continue;
        }
        if raw.confidence < threshold {
            continue;
        }
        let bbox = validate_bbox(raw.x, raw.y, raw.w, raw.h);
        if bbox.is_none() {
            tracing::debug!(class = %class_name, x = raw.x, y = raw.y, w = raw.w, h = raw.h,
                "dropping detection with nonsensical bounding box");
            continue;
        }
        detections.push(Detection {
            class_name,
            confidence: raw.confidence,
            bbox,
        });
    }
    detections
}

/// Repair a generation cut off mid-array by the `num_predict` cap: walk the closing braces from
/// the end, cut back to the last complete detection object, close the array, and re-parse.
/// Returns `None` when nothing salvageable remains.
fn salvage_truncated(content: &str) -> Option<DetectionsPayload> {
    for (i, _) in content.char_indices().rev().filter(|&(_, c)| c == '}') {
        let candidate = format!("{}]}}", &content[..=i]);
        if let Ok(payload) = serde_json::from_str::<DetectionsPayload>(&candidate) {
            return Some(payload);
        }
    }
    None
}

/// Validate and normalize a bounding box. The origin must lie inside the
/// frame and the size must be positive; a box slightly overhanging the right
/// or bottom edge (model slop) is clamped back in. Anything else is garbage.
fn validate_bbox(x: f32, y: f32, w: f32, h: f32) -> Option<(f32, f32, f32, f32)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn classes() -> Vec<String> {
        vec!["person".to_string(), "car".to_string(), "dog".to_string()]
    }

    fn make_client() -> OllamaClient {
        OllamaClient::new("http://localhost:11434", "test", 90, 0.5, classes(), None).unwrap()
    }

    #[test]
    fn format_schema_exact_shape() {
        let schema = build_format_schema(&classes());
        let expected = serde_json::json!({
            "type": "object",
            "properties": {
                "detections": {
                    "type": "array",
                    "maxItems": 15,
                    "items": {
                        "type": "object",
                        "properties": {
                            "class": {"type": "string", "enum": ["person", "car", "dog"]},
                            "confidence": {"type": "number"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                        "required": ["class", "confidence", "x", "y", "w", "h"],
                    },
                }
            },
            "required": ["detections"],
        });
        assert_eq!(schema, expected);
    }

    #[test]
    fn chat_request_serializes_format_and_options() {
        let request = ChatRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "prompt".to_string(),
                images: vec!["QUJD".to_string()],
            }],
            stream: false,
            format: build_format_schema(&classes()),
            options: RequestOptions {
                temperature: 0.0,
                num_predict: NUM_PREDICT,
            },
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["stream"], serde_json::json!(false));
        assert_eq!(value["options"]["temperature"], serde_json::json!(0.0));
        assert_eq!(value["options"]["num_predict"], serde_json::json!(768));
        assert_eq!(value["format"]["type"], serde_json::json!("object"));
        assert_eq!(
            value["format"]["properties"]["detections"]["maxItems"],
            serde_json::json!(15)
        );
        assert_eq!(value["messages"][0]["images"][0], serde_json::json!("QUJD"));
    }

    #[test]
    fn parse_valid_detections_kept() {
        let content = r#"{"detections": [
            {"class": "person", "confidence": 0.95, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            {"class": "car", "confidence": 0.8, "x": 0.5, "y": 0.6, "w": 0.2, "h": 0.3}
        ]}"#;
        let results = parse_detections(content, 0.5, &classes());
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].class_name, "person");
        assert!((results[0].confidence - 0.95).abs() < 0.01);
        let bbox = results[0].bbox.unwrap();
        assert!((bbox.0 - 0.1).abs() < 0.01);
        assert!((bbox.3 - 0.4).abs() < 0.01);
        assert_eq!(results[1].class_name, "car");
    }

    #[test]
    fn parse_empty_detections() {
        assert!(parse_detections(r#"{"detections": []}"#, 0.5, &classes()).is_empty());
    }

    #[test]
    fn parse_garbage_json_dropped() {
        assert!(parse_detections("NONE", 0.5, &classes()).is_empty());
        assert!(parse_detections("", 0.5, &classes()).is_empty());
        assert!(parse_detections(r#"{"foo": 1}"#, 0.5, &classes()).is_empty());
        assert!(parse_detections(
            r#"{"detections": [{"class": "person", "confidence": 0.9}]}"#,
            0.5,
            &classes()
        )
        .is_empty());
    }

    #[test]
    fn parse_unknown_class_dropped() {
        let content = r#"{"detections": [
            {"class": "unicorn", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            {"class": "person", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
        ]}"#;
        let results = parse_detections(content, 0.5, &classes());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "person");
    }

    #[test]
    fn parse_class_case_insensitive() {
        let content = r#"{"detections": [{"class": "PERSON", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]}"#;
        let results = parse_detections(content, 0.5, &classes());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "person");
    }

    #[test]
    fn parse_below_threshold_filtered() {
        let content = r#"{"detections": [
            {"class": "person", "confidence": 0.3, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            {"class": "car", "confidence": 0.8, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
        ]}"#;
        let results = parse_detections(content, 0.5, &classes());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "car");
    }

    #[test]
    fn parse_nonsensical_confidence_dropped() {
        let content = r#"{"detections": [
            {"class": "person", "confidence": 1.5, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            {"class": "person", "confidence": -0.2, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
        ]}"#;
        assert!(parse_detections(content, 0.5, &classes()).is_empty());
    }

    #[test]
    fn parse_garbage_bbox_dropped() {
        let content = r#"{"detections": [
            {"class": "person", "confidence": 0.9, "x": 1.2, "y": 0.1, "w": 0.2, "h": 0.2},
            {"class": "person", "confidence": 0.9, "x": 0.1, "y": -0.5, "w": 0.2, "h": 0.2},
            {"class": "person", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": -0.2, "h": 0.2},
            {"class": "person", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.0},
            {"class": "person", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": 5.0, "h": 0.2}
        ]}"#;
        assert!(parse_detections(content, 0.5, &classes()).is_empty());
    }

    #[test]
    fn parse_overhanging_bbox_clamped() {
        let content = r#"{"detections": [{"class": "car", "confidence": 0.9, "x": 0.9, "y": 0.9, "w": 0.3, "h": 0.3}]}"#;
        let results = parse_detections(content, 0.5, &classes());
        assert_eq!(results.len(), 1);
        let bbox = results[0].bbox.unwrap();
        assert!((bbox.2 - 0.1).abs() < 0.001);
        assert!((bbox.3 - 0.1).abs() < 0.001);
    }

    #[test]
    fn parse_salvages_truncated_response() {
        let content = r#"{"detections": [
            {"class": "person", "confidence": 0.95, "x": 0.01, "y": 0.53, "w": 0.03, "h": 0.47},
            {"class": "person", "confidence": 0.9, "x": 0.02, "y": 0.55, "w": 0.02, "h": 0.45},
            {"class": "person", "confidence": 0.9, "x": 0.18, "y": 0.68, "w"#;
        let results = parse_detections(content, 0.5, &classes());
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|d| d.class_name == "person"));
    }

    #[test]
    fn parse_unsalvageable_truncation_is_empty() {
        assert!(parse_detections(r#"{"detections": [{"class": "per"#, 0.5, &classes()).is_empty());
    }

    #[test]
    fn parse_enforces_detection_cap_client_side() {
        let one =
            r#"{"class": "person", "confidence": 0.9, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}"#;
        let content = format!(r#"{{"detections": [{}]}}"#, vec![one; 40].join(","));
        let results = parse_detections(&content, 0.5, &classes());
        assert_eq!(results.len(), 15);
    }

    #[test]
    fn prompt_uses_allowed_classes() {
        let client = make_client();
        let prompt = client.build_prompt();
        assert!(prompt.contains("person, car, dog"));
        assert!(!prompt.contains("truck"));
        assert!(prompt.contains("detections"));
    }

    #[test]
    fn empty_class_list_is_not_substituted() {
        let client =
            OllamaClient::new("http://localhost:11434", "test", 90, 0.5, vec![], None).unwrap();
        assert!(client.allowed_classes.is_empty());
    }

    #[test]
    fn classes_lowercased_at_construction() {
        let client = OllamaClient::new(
            "http://localhost:11434",
            "test",
            90,
            0.5,
            vec!["Person".to_string(), "CAR".to_string()],
            None,
        )
        .unwrap();
        assert_eq!(client.allowed_classes, vec!["person", "car"]);
    }
}
