//! tpue Edge TPU object-detection client.

use serde::Deserialize;

use super::detector::{validate_bbox, Detection, FrameDetectResult};

/// TCP connect timeout. A down or unreachable server fails in seconds instead
/// of eating the whole request timeout.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Client for a tpue object-detection service.
pub struct TpueClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
    configured_model: Option<String>,
    threshold: Option<f32>,
    max_detections: usize,
    /// The configured allowlist, lowercased.
    allowed_classes: Vec<String>,
}

#[derive(Deserialize)]
struct DetectResponse {
    detections: Vec<RawDetection>,
    model: String,
    #[serde(rename = "image")]
    _image: Option<ImageInfo>,
    timing_ms: Option<Timing>,
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

#[derive(Deserialize)]
struct ImageInfo {
    #[serde(rename = "width")]
    _width: u32,
    #[serde(rename = "height")]
    _height: u32,
}

#[derive(Deserialize)]
struct Timing {
    total: Option<f64>,
    inference: Option<f64>,
}

#[derive(Deserialize)]
struct ApiError {
    error: String,
}

#[derive(Deserialize)]
struct HealthResponse {
    status: String,
    model: String,
    device: String,
}

#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    name: String,
    #[serde(default)]
    default: bool,
}

impl TpueClient {
    /// Build a client for a tpue service and its optional request overrides.
    pub fn new(
        base_url: &str,
        model: Option<&str>,
        timeout_secs: u64,
        threshold: Option<f32>,
        max_detections: usize,
        allowed_classes: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;
        let configured_model = model.map(str::to_string);
        let model = configured_model
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let allowed_classes = allowed_classes
            .into_iter()
            .map(|class| class.to_lowercase())
            .collect();

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            configured_model,
            threshold,
            max_detections,
            allowed_classes,
        })
    }

    /// Configured model name, or `"default"` when tpue chooses it.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The class allowlist sent to tpue and enforced again on its response.
    pub fn allowed_classes(&self) -> &[String] {
        &self.allowed_classes
    }

    /// Detect objects in one JPEG-encoded frame. The serial detection worker
    /// guarantees no other request is in flight.
    pub async fn detect_jpeg(
        &self,
        jpeg: &[u8],
    ) -> Result<FrameDetectResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut query = vec![
            ("classes", self.allowed_classes.join(",")),
            ("max_detections", self.max_detections.to_string()),
        ];
        if let Some(threshold) = self.threshold {
            query.push(("threshold", threshold.to_string()));
        }
        if let Some(model) = &self.configured_model {
            query.push(("model", model.clone()));
        }

        let mut url = reqwest::Url::parse(&format!("{}/v1/detect", self.base_url))?;
        {
            let mut pairs = url.query_pairs_mut();
            for (key, value) in &query {
                pairs.append_pair(key, value);
            }
        }
        let response = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "image/jpeg")
            .body(jpeg.to_vec())
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            let message = serde_json::from_str::<ApiError>(&body)
                .map(|error| error.error)
                .unwrap_or(body);
            return Err(format!("tpue API error {status}: {message}").into());
        }

        let parsed: DetectResponse = serde_json::from_str(&body)?;
        if let Some(timing) = &parsed.timing_ms {
            tracing::debug!(
                total_ms = timing.total,
                inference_ms = timing.inference,
                "tpue inference timing"
            );
        }

        let mut detections = Vec::new();
        for raw in parsed.detections {
            let class_name = raw.class.to_lowercase();
            if !self.allowed_classes.contains(&class_name) {
                tracing::debug!(class = %raw.class, "dropping detection with class outside allowlist");
                continue;
            }
            if !raw.confidence.is_finite() || !(0.0..=1.0).contains(&raw.confidence) {
                tracing::debug!(class = %class_name, confidence = raw.confidence,
                    "dropping detection with nonsensical confidence");
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
            if detections.len() == self.max_detections {
                break;
            }
        }

        Ok(FrameDetectResult {
            detections,
            raw_response: body,
            model: parsed.model,
        })
    }

    /// Check service health and, when configured, that the selected model is
    /// present. Failures are warnings so a temporarily unavailable TPU never
    /// prevents camon from recording.
    pub async fn check_ready(&self) {
        let health_url = format!("{}/healthz", self.base_url);
        match self.client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<HealthResponse>().await {
                    Ok(health) => tracing::info!(
                        url = %self.base_url,
                        status = %health.status,
                        model = %health.model,
                        device = %health.device,
                        "tpue ready"
                    ),
                    Err(error) => tracing::warn!(
                        url = %self.base_url,
                        error = %error,
                        "could not parse tpue /healthz response; server is not ready or unreachable and detection will fail until it is"
                    ),
                }
            }
            Ok(response) => tracing::warn!(
                url = %self.base_url,
                status = %response.status(),
                "tpue server is not ready or unreachable; detection will fail until it is"
            ),
            Err(error) => tracing::warn!(
                url = %self.base_url,
                error = %error,
                "tpue server is not ready or unreachable; detection will fail until it is"
            ),
        }

        let Some(configured_model) = &self.configured_model else {
            return;
        };
        let models_url = format!("{}/v1/models", self.base_url);
        let models = match self.client.get(&models_url).send().await {
            Ok(response) => match response.json::<ModelsResponse>().await {
                Ok(models) => models.models,
                Err(error) => {
                    tracing::warn!(url = %self.base_url, error = %error,
                        "could not parse tpue /v1/models response, skipping model check");
                    return;
                }
            },
            Err(error) => {
                tracing::warn!(url = %self.base_url, error = %error,
                    "tpue server unreachable, skipping model check");
                return;
            }
        };
        if models.iter().any(|model| model.name == *configured_model) {
            tracing::info!(url = %self.base_url, model = %configured_model,
                "tpue model available");
        } else {
            let available: Vec<&str> = models.iter().map(|model| model.name.as_str()).collect();
            let server_default = models
                .iter()
                .find(|model| model.default)
                .map(|model| model.name.as_str());
            tracing::warn!(
                url = %self.base_url,
                model = %configured_model,
                available = ?available,
                server_default,
                "configured model is NOT available on the tpue server — object detection will fail until it is configured there"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use axum::body::Bytes;
    use axum::extract::{Query, State};
    use axum::http::{header, HeaderMap, StatusCode};
    use axum::routing::post;
    use axum::{Json, Router};

    use super::*;

    async fn serve(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    fn client(url: &str) -> TpueClient {
        TpueClient::new(
            url,
            None,
            5,
            None,
            15,
            vec!["person".to_string(), "car".to_string()],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn parses_and_validates_a_successful_response() {
        let app = Router::new().route(
            "/v1/detect",
            post(|| async {
                Json(serde_json::json!({
                    "detections": [
                        {"class":"PERSON", "confidence":0.83, "x":0.41, "y":0.22, "w":0.09, "h":0.31},
                        {"class":"cat", "confidence":0.9, "x":0.1, "y":0.1, "w":0.2, "h":0.2},
                        {"class":"car", "confidence":0.9, "x":1.2, "y":0.1, "w":0.2, "h":0.2}
                    ],
                    "model":"yolov9s-512",
                    "image":{"width":640,"height":360},
                    "timing_ms":{"inference":34.9,"total":38.1}
                }))
            }),
        );
        let result = client(&serve(app).await)
            .detect_jpeg(b"jpeg")
            .await
            .unwrap();

        assert_eq!(result.model, "yolov9s-512");
        assert_eq!(result.detections.len(), 1);
        assert_eq!(result.detections[0].class_name, "person");
        assert!(result.raw_response.contains("timing_ms"));
    }

    #[tokio::test]
    async fn sends_raw_jpeg_content_type_and_class_allowlist() {
        async fn handler(
            headers: HeaderMap,
            Query(query): Query<HashMap<String, String>>,
            body: Bytes,
        ) -> Json<serde_json::Value> {
            assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
            assert_eq!(&body[..], b"raw jpeg bytes");
            assert_eq!(query.get("classes").map(String::as_str), Some("person,car"));
            assert_eq!(query.get("max_detections").map(String::as_str), Some("15"));
            Json(serde_json::json!({"detections":[], "model":"test"}))
        }
        let app = Router::new().route("/v1/detect", post(handler));

        client(&serve(app).await)
            .detect_jpeg(b"raw jpeg bytes")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn api_error_includes_the_servers_message() {
        let app = Router::new().route(
            "/v1/detect",
            post(|| async {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error":"TPU queue is full"})),
                )
            }),
        );
        let error = client(&serve(app).await)
            .detect_jpeg(b"jpeg")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("TPU queue is full"), "{error}");
    }

    #[tokio::test]
    async fn optional_query_parameters_are_sent_only_when_configured() {
        type Seen = Arc<Mutex<Vec<HashMap<String, String>>>>;
        async fn handler(
            State(seen): State<Seen>,
            Query(query): Query<HashMap<String, String>>,
        ) -> Json<serde_json::Value> {
            seen.lock().unwrap().push(query);
            Json(serde_json::json!({"detections":[], "model":"test"}))
        }

        let seen = Seen::default();
        let app = Router::new()
            .route("/v1/detect", post(handler))
            .with_state(Arc::clone(&seen));
        let url = serve(app).await;
        client(&url).detect_jpeg(b"jpeg").await.unwrap();
        TpueClient::new(
            &url,
            Some("configured"),
            5,
            Some(0.2),
            7,
            vec!["person".to_string()],
        )
        .unwrap()
        .detect_jpeg(b"jpeg")
        .await
        .unwrap();

        let requests = seen.lock().unwrap();
        assert!(!requests[0].contains_key("threshold"));
        assert!(!requests[0].contains_key("model"));
        assert_eq!(
            requests[1].get("threshold").map(String::as_str),
            Some("0.2")
        );
        assert_eq!(
            requests[1].get("model").map(String::as_str),
            Some("configured")
        );
    }
}
