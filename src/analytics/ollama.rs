use base64::Engine;
use opencv::core::{Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;

use super::object::Detection;

const DETECT_PROMPT: &str = "This image is a 2x2 grid of 4 frames from a security camera, \
showing a motion event over time (top-left is earliest, bottom-right is latest). \
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

pub struct OllamaDetector {
    client: reqwest::Client,
    rt: Handle,
    base_url: String,
    model: String,
    confidence_threshold: f32,
    allowed_classes: Vec<String>,
}

impl OllamaDetector {
    pub fn new(
        base_url: &str,
        model: &str,
        confidence_threshold: f32,
        allowed_classes: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;

        let rt = Handle::current();
        let base_url = base_url.trim_end_matches('/').to_string();

        Ok(Self {
            client,
            rt,
            base_url,
            model: model.to_string(),
            confidence_threshold,
            allowed_classes,
        })
    }

    pub fn detect_grid(
        &self,
        frames: &[opencv::core::Mat],
        cx: f32,
        cy: f32,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error + Send + Sync>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let grid = self.stitch_grid(frames)?;
        let image_b64 = self.encode_frame(&grid)?;

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: DETECT_PROMPT.to_string(),
                images: vec![image_b64],
            }],
            stream: false,
        };

        let url = format!("{}/api/chat", self.base_url);
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

        Ok(self.parse_response(&content, cx, cy))
    }

    fn stitch_grid(
        &self,
        frames: &[opencv::core::Mat],
    ) -> Result<opencv::core::Mat, Box<dyn std::error::Error + Send + Sync>> {
        let cell_w = 320;
        let cell_h = 240;
        let grid_w = cell_w * 2;
        let grid_h = cell_h * 2;

        let mut grid = opencv::core::Mat::zeros(grid_h, grid_w, opencv::core::CV_8UC3)?.to_mat()?;

        let positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)];

        for (i, frame) in frames.iter().take(4).enumerate() {
            if frame.empty() {
                continue;
            }
            let mut resized = opencv::core::Mat::default();
            imgproc::resize(
                frame,
                &mut resized,
                Size::new(cell_w, cell_h),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            let (px, py) = positions[i];
            let roi = opencv::core::Rect::new(px, py, cell_w, cell_h);
            let mut dst = opencv::core::Mat::roi_mut(&mut grid, roi)?;
            resized.copy_to(&mut dst)?;
        }

        Ok(grid)
    }

    fn encode_frame(
        &self,
        frame: &opencv::core::Mat,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut buf = Vector::<u8>::new();
        let params = Vector::<i32>::new();
        imgcodecs::imencode(".jpg", frame, &mut buf, &params)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(buf.to_vec()))
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
            base_url: "http://localhost:11434".to_string(),
            model: "test".to_string(),
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
