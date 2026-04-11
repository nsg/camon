use base64::Engine;
use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Result from a single frame detection call.
struct FrameDetectResult {
    detections: Vec<Detection>,
    raw_response: String,
    model: String,
}

/// Full result from detecting across multiple frames.
pub struct DetectResult {
    pub detections: Vec<Detection>,
    pub frame_jpegs: Vec<Vec<u8>>,
    pub raw_responses: Vec<String>,
    pub model: String,
}

const DETECT_PROMPT: &str = "This image is a single frame from a security camera during a motion event. \
List every noteworthy object you see (person, car, truck, dog, cat, bird, bicycle, motorcycle, bus, boat). \
For each distinct object, respond with exactly one line: CLASS CONFIDENCE\n\
CONFIDENCE = a number 0.0 to 1.0 indicating how certain you are.\n\
If you see nothing noteworthy, respond with exactly: NONE\n\
Do not add any other text, headers, or explanations.";

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
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

struct OllamaServer {
    base_url: String,
    model: String,
}

pub struct OllamaDetector {
    client: reqwest::Client,
    rt: Handle,
    primary: OllamaServer,
    fallback: Option<OllamaServer>,
    confidence_threshold: f32,
    allowed_classes: Vec<String>,
}

impl OllamaDetector {
    pub fn new(
        base_url: &str,
        model: &str,
        confidence_threshold: f32,
        allowed_classes: Vec<String>,
        fallback: Option<(&str, &str)>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        let rt = Handle::current();

        let primary = OllamaServer {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        };

        let fallback = fallback.map(|(url, model)| OllamaServer {
            base_url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        });

        Ok(Self {
            client,
            rt,
            primary,
            fallback,
            confidence_threshold,
            allowed_classes,
        })
    }

    pub fn model(&self) -> &str {
        &self.primary.model
    }

    pub fn detect_frames(
        &self,
        frames: &[opencv::core::Mat],
        cx: f32,
        cy: f32,
    ) -> Result<DetectResult, Box<dyn std::error::Error + Send + Sync>> {
        if frames.is_empty() {
            return Ok(DetectResult {
                detections: Vec::new(),
                frame_jpegs: Vec::new(),
                raw_responses: Vec::new(),
                model: self.primary.model.clone(),
            });
        }

        let mut frame_jpegs = Vec::with_capacity(frames.len());
        let mut all_detections = Vec::new();
        let mut raw_responses = Vec::new();
        let mut model = self.primary.model.clone();

        for frame in frames.iter().take(4) {
            let jpeg = self.encode_frame_jpeg(frame)?;
            let image_b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);

            let result = self.call_single_frame(&image_b64, cx, cy);
            match result {
                Ok(r) => {
                    all_detections.extend(r.detections);
                    raw_responses.push(r.raw_response);
                    model = r.model;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "frame detection failed");
                    raw_responses.push(format!("ERROR: {e}"));
                }
            }

            frame_jpegs.push(jpeg);
        }

        Ok(DetectResult {
            detections: all_detections,
            frame_jpegs,
            raw_responses,
            model,
        })
    }

    fn call_single_frame(
        &self,
        image_b64: &str,
        cx: f32,
        cy: f32,
    ) -> Result<FrameDetectResult, Box<dyn std::error::Error + Send + Sync>> {
        match self.call_server(&self.primary, image_b64, cx, cy) {
            Ok((detections, raw_response, model)) => Ok(FrameDetectResult {
                detections,
                raw_response,
                model,
            }),
            Err(primary_err) => {
                if let Some(ref fallback) = self.fallback {
                    tracing::warn!(
                        error = %primary_err,
                        "primary ollama failed, trying fallback"
                    );
                    match self.call_server(fallback, image_b64, cx, cy) {
                        Ok((detections, raw_response, model)) => Ok(FrameDetectResult {
                            detections,
                            raw_response,
                            model,
                        }),
                        Err(fallback_err) => Err(format!(
                            "both servers failed — primary: {primary_err}, fallback: {fallback_err}"
                        )
                        .into()),
                    }
                } else {
                    Err(primary_err)
                }
            }
        }
    }

    fn call_server(
        &self,
        server: &OllamaServer,
        image_b64: &str,
        cx: f32,
        cy: f32,
    ) -> Result<(Vec<Detection>, String, String), Box<dyn std::error::Error + Send + Sync>> {
        let request = ChatRequest {
            model: server.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: DETECT_PROMPT.to_string(),
                images: vec![image_b64.to_string()],
            }],
            stream: false,
        };

        let url = format!("{}/api/chat", server.base_url);
        let response = self
            .rt
            .block_on(async { self.client.post(&url).json(&request).send().await })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = self.rt.block_on(response.text()).unwrap_or_default();
            return Err(format!("ollama API error {}: {}", status, body).into());
        }

        let chat_response: ChatResponse = self.rt.block_on(response.json())?;
        let content = chat_response.message.map(|m| m.content).unwrap_or_default();
        let detections = self.parse_response(&content, cx, cy);

        Ok((detections, content, server.model.clone()))
    }

    fn encode_frame_jpeg(
        &self,
        frame: &opencv::core::Mat,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let mut buf = Vector::<u8>::new();
        let params = Vector::<i32>::new();
        imgcodecs::imencode(".jpg", frame, &mut buf, &params)?;
        Ok(buf.to_vec())
    }

    fn parse_response(&self, content: &str, cx: f32, cy: f32) -> Vec<Detection> {
        let mut detections = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.eq_ignore_ascii_case("NONE") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let class_name = parts[0].to_lowercase();
            let confidence: f32 = match parts[1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            if confidence < self.confidence_threshold {
                continue;
            }

            if !self.allowed_classes.is_empty() && !self.allowed_classes.contains(&class_name) {
                continue;
            }

            detections.push(Detection {
                class_name,
                confidence: confidence.clamp(0.0, 1.0),
                cx,
                cy,
            });
        }

        detections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> OllamaDetector {
        let rt = tokio::runtime::Runtime::new().unwrap();
        OllamaDetector {
            client: reqwest::Client::new(),
            rt: rt.handle().clone(),
            primary: OllamaServer {
                base_url: "http://localhost:11434".to_string(),
                model: "test".to_string(),
            },
            fallback: None,
            confidence_threshold: 0.5,
            allowed_classes: vec!["person".to_string(), "car".to_string(), "dog".to_string()],
        }
    }

    #[test]
    fn parse_valid_detections() {
        let det = make_detector();
        let response = "person 0.95\ncar 0.80\n";
        let results = det.parse_response(response, 0.5, 0.6);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].class_name, "person");
        assert!((results[0].confidence - 0.95).abs() < 0.01);
        assert_eq!(results[0].cx, 0.5);
        assert_eq!(results[0].cy, 0.6);
        assert_eq!(results[1].class_name, "car");
    }

    #[test]
    fn parse_none_response() {
        let det = make_detector();
        assert!(det.parse_response("NONE", 0.5, 0.5).is_empty());
        assert!(det.parse_response("none", 0.5, 0.5).is_empty());
        assert!(det.parse_response("NONE\n", 0.5, 0.5).is_empty());
    }

    #[test]
    fn parse_filters_low_confidence() {
        let det = make_detector();
        let response = "person 0.3\ncar 0.8\n";
        let results = det.parse_response(response, 0.5, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "car");
    }

    #[test]
    fn parse_filters_disallowed_classes() {
        let det = make_detector();
        let response = "truck 0.9\nperson 0.8\n";
        let results = det.parse_response(response, 0.5, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "person");
    }

    #[test]
    fn parse_ignores_garbage() {
        let det = make_detector();
        let response = "Here are the objects I see:\nperson 0.9\nsome random text\n";
        let results = det.parse_response(response, 0.5, 0.5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn parse_uses_provided_coordinates() {
        let det = make_detector();
        let response = "person 0.9\n";
        let results = det.parse_response(response, 0.3, 0.7);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cx, 0.3);
        assert_eq!(results[0].cy, 0.7);
    }

    #[test]
    fn parse_empty_response() {
        let det = make_detector();
        assert!(det.parse_response("", 0.5, 0.5).is_empty());
        assert!(det.parse_response("\n\n", 0.5, 0.5).is_empty());
    }
}
