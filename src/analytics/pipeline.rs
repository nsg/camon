use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use opencv::core::Mat;
use opencv::prelude::*;

use crate::buffer::HotBuffer;
use crate::config::AnalyticsConfig;
use crate::storage::{MotionEntry, MotionStore};

use super::decoder::FrameDecoder;
use super::motion::MotionDetector;

const POLL_INTERVAL: Duration = Duration::from_millis(200);

pub struct MotionAnalyzer {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    store: MotionStore,
    config: AnalyticsConfig,
    detector: MotionDetector,
    decoder: FrameDecoder,
    last_processed: u64,
}

impl MotionAnalyzer {
    fn new(
        camera_id: String,
        buffer: Arc<RwLock<HotBuffer>>,
        store: MotionStore,
        config: AnalyticsConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let detector = MotionDetector::new()?;
        let decoder = FrameDecoder::new(config.sample_fps)?;

        let last_processed = store.last_sequence(&camera_id).map(|s| s + 1).unwrap_or(0);

        Ok(Self {
            camera_id,
            buffer,
            store,
            config,
            detector,
            decoder,
            last_processed,
        })
    }

    fn run(mut self, shutdown: Arc<AtomicBool>) {
        tracing::info!(camera = %self.camera_id, "motion analyzer started");

        while !shutdown.load(Ordering::Relaxed) {
            if !self.decoder.is_alive() {
                tracing::warn!(camera = %self.camera_id, "decoder process died, restarting");
                match FrameDecoder::new(self.config.sample_fps) {
                    Ok(d) => self.decoder = d,
                    Err(e) => {
                        tracing::error!(camera = %self.camera_id, error = %e, "failed to restart decoder");
                        thread::sleep(Duration::from_secs(5));
                        continue;
                    }
                }
            }

            if let Err(e) = self.process_new_segments() {
                tracing::error!(
                    camera = %self.camera_id,
                    error = %e,
                    "motion analysis error"
                );
            }

            thread::sleep(POLL_INTERVAL);
        }

        tracing::info!(camera = %self.camera_id, "motion analyzer stopped");
    }

    fn process_new_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let segments_to_process = {
            let buffer = self.buffer.read().map_err(|_| "buffer lock poisoned")?;
            let first_seq = buffer.first_sequence();
            let last_seq = buffer.last_sequence();

            if first_seq > 0 {
                self.store.cleanup(&self.camera_id, first_seq);
            }

            if self.last_processed < first_seq {
                self.last_processed = first_seq;
            }

            let mut segments = Vec::new();
            for seq in self.last_processed..last_seq {
                if let Some(segment) = buffer.get_segment_by_sequence(seq) {
                    segments.push((
                        seq,
                        segment.data.clone(),
                        segment.start_pts,
                        segment.duration_ns,
                    ));
                }
            }
            segments
        };

        for (seq, data, start_pts, duration_ns) in segments_to_process {
            let score = self.analyze_segment(&data, duration_ns)?;

            if score >= self.config.motion_threshold {
                self.store.insert(
                    &self.camera_id,
                    MotionEntry {
                        segment_sequence: seq,
                        start_time_ns: start_pts,
                        end_time_ns: start_pts + duration_ns,
                        motion_score: score,
                    },
                );

                tracing::debug!(
                    camera = %self.camera_id,
                    sequence = seq,
                    score = format!("{:.3}", score),
                    "motion detected"
                );
            }

            self.last_processed = seq + 1;
        }

        Ok(())
    }

    fn analyze_segment(
        &mut self,
        data: &[u8],
        duration_ns: u64,
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let raw_frames = self.decoder.decode_segment(data, duration_ns);

        if raw_frames.is_empty() {
            return Ok(0.0);
        }

        let height = self.decoder.height() as i32;
        let mut total_score = 0.0f32;
        let mut frame_count = 0u32;

        for frame_data in &raw_frames {
            let mat = Mat::from_slice(frame_data)?;
            let mat = mat.reshape(1, height)?;

            match self.detector.process_frame(&mat) {
                Ok(score) => {
                    total_score += score.score;
                    frame_count += 1;
                }
                Err(e) => {
                    tracing::trace!(error = %e, "frame processing error");
                }
            }
        }

        if frame_count == 0 {
            return Ok(0.0);
        }

        Ok(total_score / frame_count as f32)
    }
}

pub fn spawn_analyzer(
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    store: MotionStore,
    config: AnalyticsConfig,
    shutdown: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        match MotionAnalyzer::new(camera_id.clone(), buffer, store, config) {
            Ok(analyzer) => analyzer.run(shutdown),
            Err(e) => {
                tracing::error!(camera = %camera_id, error = %e, "failed to create motion analyzer");
            }
        }
    })
}
