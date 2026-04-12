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
    /// Normalized bounding box (x, y, w, h) in 0.0-1.0 image coordinates.
    pub bbox: Option<(f32, f32, f32, f32)>,
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

const DETECT_PROMPT_TEMPLATE: &str = "Security camera frame. List objects: {classes}.\n\
One line per object: CLASS CONF X Y W H\n\
CONF is 0.0-1.0. X Y W H are bounding box fractions of image size (0.0-1.0), top-left corner and size.\n\
Examples:\n\
person 0.92 0.40 0.10 0.15 0.60\n\
car 0.87 0.05 0.50 0.30 0.20\n\
If nothing noteworthy: NONE\n\
No other text.";

const DEFAULT_CLASSES: &str = "person, car, truck, dog, cat, bird, bicycle, motorcycle, bus, boat";

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

    fn build_prompt(&self) -> String {
        let classes = if self.allowed_classes.is_empty() {
            DEFAULT_CLASSES.to_string()
        } else {
            self.allowed_classes.join(", ")
        };
        DETECT_PROMPT_TEMPLATE.replace("{classes}", &classes)
    }

    pub fn detect_frames(
        &self,
        frames: &[opencv::core::Mat],
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

            let result = self.call_single_frame(&image_b64);
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
    ) -> Result<FrameDetectResult, Box<dyn std::error::Error + Send + Sync>> {
        match self.call_server(&self.primary, image_b64) {
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
                    match self.call_server(fallback, image_b64) {
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
    ) -> Result<(Vec<Detection>, String, String), Box<dyn std::error::Error + Send + Sync>> {
        let request = ChatRequest {
            model: server.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: self.build_prompt(),
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
        let detections = self.parse_response(&content);

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

    fn parse_response(&self, content: &str) -> Vec<Detection> {
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

            // Find the class name (first token) and then scan forward for the
            // first parseable float as the confidence value. This handles cases
            // where the model inserts extra words like "CONFIDENCE" between the
            // class and the number.
            let class_name = parts[0].to_lowercase();
            let mut conf_idx = None;
            for (i, part) in parts.iter().enumerate().skip(1) {
                if let Ok(_v) = part.parse::<f32>() {
                    conf_idx = Some(i);
                    break;
                }
            }
            let conf_idx = match conf_idx {
                Some(i) => i,
                None => continue,
            };

            let confidence: f32 = parts[conf_idx].parse().unwrap();

            if confidence < self.confidence_threshold {
                continue;
            }

            if !self.allowed_classes.is_empty() && !self.allowed_classes.contains(&class_name) {
                continue;
            }

            // Parse bounding box from the 4 floats after the confidence value.
            let remaining = &parts[conf_idx + 1..];
            let bbox = if remaining.len() >= 4 {
                match (
                    remaining[0].parse::<f32>(),
                    remaining[1].parse::<f32>(),
                    remaining[2].parse::<f32>(),
                    remaining[3].parse::<f32>(),
                ) {
                    (Ok(x), Ok(y), Ok(w), Ok(h)) => {
                        let x = x.clamp(0.0, 1.0);
                        let y = y.clamp(0.0, 1.0);
                        let w = w.clamp(0.0, 1.0 - x);
                        let h = h.clamp(0.0, 1.0 - y);
                        if w > 0.0 && h > 0.0 {
                            Some((x, y, w, h))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            detections.push(Detection {
                class_name,
                confidence: confidence.clamp(0.0, 1.0),
                bbox,
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
        let results = det.parse_response(response);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].class_name, "person");
        assert!((results[0].confidence - 0.95).abs() < 0.01);
        assert!(results[0].bbox.is_none());
        assert_eq!(results[1].class_name, "car");
    }

    #[test]
    fn parse_detections_with_bbox() {
        let det = make_detector();
        let response = "person 0.95 0.1 0.2 0.3 0.4\ncar 0.80 0.5 0.6 0.2 0.3\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 2);
        let bbox = results[0].bbox.unwrap();
        assert!((bbox.0 - 0.1).abs() < 0.01);
        assert!((bbox.1 - 0.2).abs() < 0.01);
        assert!((bbox.2 - 0.3).abs() < 0.01);
        assert!((bbox.3 - 0.4).abs() < 0.01);
        assert!(results[1].bbox.is_some());
    }

    #[test]
    fn parse_mixed_bbox_and_no_bbox() {
        let det = make_detector();
        let response = "person 0.95 0.1 0.2 0.3 0.4\ncar 0.80\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 2);
        assert!(results[0].bbox.is_some());
        assert!(results[1].bbox.is_none());
    }

    #[test]
    fn parse_none_response() {
        let det = make_detector();
        assert!(det.parse_response("NONE").is_empty());
        assert!(det.parse_response("none").is_empty());
        assert!(det.parse_response("NONE\n").is_empty());
    }

    #[test]
    fn parse_filters_low_confidence() {
        let det = make_detector();
        let response = "person 0.3\ncar 0.8\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "car");
    }

    #[test]
    fn parse_filters_disallowed_classes() {
        let det = make_detector();
        let response = "truck 0.9\nperson 0.8\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "person");
    }

    #[test]
    fn parse_ignores_garbage() {
        let det = make_detector();
        let response = "Here are the objects I see:\nperson 0.9\nsome random text\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn parse_empty_response() {
        let det = make_detector();
        assert!(det.parse_response("").is_empty());
        assert!(det.parse_response("\n\n").is_empty());
    }

    #[test]
    fn parse_confidence_word_before_number() {
        let det = make_detector();
        let response = "car CONFIDENCE 0.95 0.065 0.22 0.12 0.08\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].class_name, "car");
        assert!((results[0].confidence - 0.95).abs() < 0.01);
        let bbox = results[0].bbox.unwrap();
        assert!((bbox.0 - 0.065).abs() < 0.01);
    }

    #[test]
    fn parse_all_caps_class() {
        let det = make_detector();
        let response = "CAR 0.85 0.73 0.12 0.65 0.25\nPERSON 0.90 0.1 0.2 0.3 0.4\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].class_name, "car");
        assert_eq!(results[1].class_name, "person");
    }

    #[test]
    fn parse_extra_tokens_before_confidence() {
        let det = make_detector();
        let response = "dog confidence_score 0.75 0.1 0.2 0.3 0.4\n";
        let results = det.parse_response(response);
        assert_eq!(results.len(), 1);
        assert!((results[0].confidence - 0.75).abs() < 0.01);
        assert!(results[0].bbox.is_some());
    }

    #[test]
    fn prompt_uses_allowed_classes() {
        let det = make_detector();
        let prompt = det.build_prompt();
        assert!(prompt.contains("person, car, dog"));
        assert!(!prompt.contains("truck"));
    }

    #[test]
    fn prompt_uses_defaults_when_no_classes() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let det = OllamaDetector {
            client: reqwest::Client::new(),
            rt: rt.handle().clone(),
            primary: OllamaServer {
                base_url: "http://localhost:11434".to_string(),
                model: "test".to_string(),
            },
            fallback: None,
            confidence_threshold: 0.5,
            allowed_classes: vec![],
        };
        let prompt = det.build_prompt();
        assert!(prompt.contains("person, car, truck, dog, cat, bird"));
    }
}
