use ndarray::{Array4, ArrayViewD};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

const YOLO_INPUT_SIZE: u32 = 640;
const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
}

pub struct ObjectDetector {
    session: Session,
    confidence_threshold: f32,
    allowed_classes: Vec<String>,
}

impl ObjectDetector {
    pub fn new(
        model_path: &str,
        confidence_threshold: f32,
        allowed_classes: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        let session = if model_path.starts_with("http://") || model_path.starts_with("https://") {
            builder.commit_from_url(model_path)?
        } else {
            builder.commit_from_file(model_path)?
        };

        Ok(Self {
            session,
            confidence_threshold,
            allowed_classes,
        })
    }

    pub fn detect(
        &mut self,
        frame: &opencv::core::Mat,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error + Send + Sync>> {
        use opencv::prelude::*;

        let rows = frame.rows();
        let cols = frame.cols();
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }

        let input_tensor = self.preprocess(frame)?;

        let tensor_ref = TensorRef::from_array_view(input_tensor.view())?.into_dyn();
        let outputs = self.session.run(ort::inputs![tensor_ref])?;

        // YOLO26 format: separate "logits" and "pred_boxes" outputs
        let Some(logits_val) = outputs.get("logits") else {
            return Err("Unsupported model format: expected YOLO26 with 'logits' output".into());
        };

        let logits = logits_val.try_extract_array::<f32>()?;
        let logits_owned = logits.to_owned();
        drop(outputs);

        let detections = Self::postprocess_yolo26(
            &logits_owned.view(),
            self.confidence_threshold,
            &self.allowed_classes,
        )?;

        Ok(detections)
    }

    fn preprocess(
        &self,
        frame: &opencv::core::Mat,
    ) -> Result<Array4<f32>, Box<dyn std::error::Error + Send + Sync>> {
        use opencv::core::{Mat, Size, BORDER_CONSTANT};
        use opencv::imgproc;
        use opencv::prelude::*;

        let rows = frame.rows() as f32;
        let cols = frame.cols() as f32;
        let input_size = YOLO_INPUT_SIZE as f32;

        let scale = (input_size / cols).min(input_size / rows);
        let new_w = (cols * scale).round() as i32;
        let new_h = (rows * scale).round() as i32;

        let mut resized = Mat::default();
        imgproc::resize(
            frame,
            &mut resized,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let pad_x = (input_size as i32 - new_w) / 2;
        let pad_y = (input_size as i32 - new_h) / 2;

        let mut padded = Mat::default();
        opencv::core::copy_make_border(
            &resized,
            &mut padded,
            pad_y,
            input_size as i32 - new_h - pad_y,
            pad_x,
            input_size as i32 - new_w - pad_x,
            BORDER_CONSTANT,
            opencv::core::Scalar::new(114.0, 114.0, 114.0, 0.0),
        )?;

        let mut rgb = Mat::default();
        imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

        let data = rgb.data_bytes()?;
        let total_size = (YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3) as usize;
        if data.len() < total_size {
            return Err("Frame data too small".into());
        }

        let mut tensor =
            Array4::<f32>::zeros((1, 3, YOLO_INPUT_SIZE as usize, YOLO_INPUT_SIZE as usize));
        for y in 0..YOLO_INPUT_SIZE as usize {
            for x in 0..YOLO_INPUT_SIZE as usize {
                let idx = (y * YOLO_INPUT_SIZE as usize + x) * 3;
                tensor[[0, 0, y, x]] = data[idx] as f32 / 255.0;
                tensor[[0, 1, y, x]] = data[idx + 1] as f32 / 255.0;
                tensor[[0, 2, y, x]] = data[idx + 2] as f32 / 255.0;
            }
        }

        Ok(tensor)
    }

    fn postprocess_yolo26(
        logits: &ArrayViewD<f32>,
        confidence_threshold: f32,
        allowed_classes: &[String],
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error + Send + Sync>> {
        let logits_shape = logits.shape();

        // Expected shape: logits [1, 300, 80]
        if logits_shape.len() < 2 {
            return Ok(Vec::new());
        }

        let num_detections = if logits_shape.len() == 3 {
            logits_shape[1]
        } else {
            logits_shape[0]
        };
        let num_classes = if logits_shape.len() == 3 {
            logits_shape[2]
        } else {
            logits_shape[1]
        };

        let logits_flat = logits.as_slice().ok_or("Cannot get logits slice")?;
        let mut detections = Vec::new();

        for i in 0..num_detections {
            // Find max class score (apply sigmoid)
            let mut max_score = 0.0f32;
            let mut max_class = 0usize;

            for j in 0..num_classes {
                let logit = logits_flat[i * num_classes + j];
                let score = 1.0 / (1.0 + (-logit).exp()); // sigmoid
                if score > max_score {
                    max_score = score;
                    max_class = j;
                }
            }

            if max_score < confidence_threshold {
                continue;
            }

            let class_name = if max_class < COCO_CLASSES.len() {
                COCO_CLASSES[max_class].to_string()
            } else {
                format!("class_{}", max_class)
            };

            if !allowed_classes.is_empty() && !allowed_classes.contains(&class_name) {
                continue;
            }

            detections.push(Detection {
                class_name,
                confidence: max_score,
            });
        }

        Ok(detections)
    }
}
