use std::sync::{Arc, RwLock};

use gstreamer::prelude::*;
use gstreamer::{ClockTime, Element, Pipeline};
use gstreamer_app::AppSink;
use thiserror::Error;

use crate::buffer::{GopSegment, HotBuffer};
use crate::config::CameraConfig;

#[derive(Debug, Error)]
pub enum RtspError {
    #[error("gstreamer error: {0}")]
    Gstreamer(#[from] gstreamer::glib::Error),
    #[error("gstreamer bool error: {0}")]
    GstreamerBool(#[from] gstreamer::glib::BoolError),
    #[error("missing element")]
    MissingElement,
    #[error("pipeline error: {0}")]
    Pipeline(String),
}

pub struct RtspPipeline {
    pipeline: Pipeline,
    camera_id: String,
}

impl RtspPipeline {
    pub fn new(config: &CameraConfig, buffer: Arc<RwLock<HotBuffer>>) -> Result<Self, RtspError> {
        let pipeline = Pipeline::with_name(&format!("camera-{}", config.id));

        let rtspsrc = gstreamer::ElementFactory::make("rtspsrc")
            .name("src")
            .property("location", &config.url)
            .property("latency", 100u32)
            .build()?;

        let depay = gstreamer::ElementFactory::make("rtph264depay")
            .name("depay")
            .build()?;

        let parse = gstreamer::ElementFactory::make("h264parse")
            .name("parse")
            .build()?;

        let appsink = gstreamer::ElementFactory::make("appsink")
            .name("sink")
            .build()?;

        pipeline.add_many([&rtspsrc, &depay, &parse, &appsink])?;
        Element::link_many([&depay, &parse, &appsink])?;

        let depay_weak = depay.downgrade();
        rtspsrc.connect_pad_added(move |_, src_pad| {
            let Some(depay) = depay_weak.upgrade() else {
                return;
            };

            let sink_pad = depay.static_pad("sink").expect("depay has sink pad");
            if sink_pad.is_linked() {
                return;
            }

            if let Err(e) = src_pad.link(&sink_pad) {
                tracing::error!("failed to link rtspsrc pad: {}", e);
            }
        });

        let appsink = appsink
            .dynamic_cast::<AppSink>()
            .map_err(|_| RtspError::MissingElement)?;

        appsink.set_property("emit-signals", true);
        appsink.set_property("sync", false);

        let camera_id = config.id.clone();
        let accumulator = GopAccumulator::new(camera_id.clone(), buffer);

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gstreamer::FlowError::Eos)?;
                    accumulator.lock().unwrap().handle_sample(&sample);
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            camera_id: config.id.clone(),
        })
    }

    pub fn start(&self) -> Result<(), RtspError> {
        self.pipeline
            .set_state(gstreamer::State::Playing)
            .map_err(|e| RtspError::Pipeline(e.to_string()))?;
        tracing::info!(camera = %self.camera_id, "RTSP pipeline started");
        Ok(())
    }

    pub fn stop(&self) -> Result<(), RtspError> {
        self.pipeline
            .set_state(gstreamer::State::Null)
            .map_err(|e| RtspError::Pipeline(e.to_string()))?;
        tracing::info!(camera = %self.camera_id, "RTSP pipeline stopped");
        Ok(())
    }

    pub fn bus(&self) -> Option<gstreamer::Bus> {
        self.pipeline.bus()
    }
}

struct GopAccumulator {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    current_gop: Option<GopSegment>,
}

impl GopAccumulator {
    fn new(camera_id: String, buffer: Arc<RwLock<HotBuffer>>) -> std::sync::Mutex<Self> {
        std::sync::Mutex::new(Self {
            camera_id,
            buffer,
            current_gop: None,
        })
    }

    fn handle_sample(&mut self, sample: &gstreamer::Sample) {
        let Some(buffer) = sample.buffer() else {
            return;
        };

        let flags = buffer.flags();
        let is_keyframe = !flags.contains(gstreamer::BufferFlags::DELTA_UNIT);

        let pts = buffer.pts().map(ClockTime::nseconds).unwrap_or(0);

        if is_keyframe {
            if let Some(mut gop) = self.current_gop.take() {
                gop.finalize(pts);
                if gop.frame_count > 0 {
                    if let Ok(mut hot) = self.buffer.write() {
                        hot.push(gop);
                    }
                }
            }
            self.current_gop = Some(GopSegment::new(pts));
            tracing::trace!(camera = %self.camera_id, pts, "keyframe detected, starting new GOP");
        }

        if let Some(ref mut gop) = self.current_gop {
            let map = buffer.map_readable().ok();
            if let Some(map) = map {
                gop.append_frame(map.as_slice(), pts);
            }
        }
    }
}
