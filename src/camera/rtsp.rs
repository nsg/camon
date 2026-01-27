use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use thiserror::Error;

use crate::buffer::{GopSegment, HotBuffer};
use crate::config::CameraConfig;

#[derive(Debug, Error)]
pub enum RtspError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffmpeg not found")]
    FfmpegNotFound,
    #[error("ffmpeg failed: {0}")]
    FfmpegFailed(String),
}

pub struct FfmpegPipeline {
    camera_id: String,
    url: String,
    buffer: Arc<RwLock<HotBuffer>>,
}

impl FfmpegPipeline {
    pub fn new(config: &CameraConfig, buffer: Arc<RwLock<HotBuffer>>) -> Result<Self, RtspError> {
        Ok(Self {
            camera_id: config.id.clone(),
            url: config.url.clone(),
            buffer,
        })
    }

    /// Run the ffmpeg pipeline, blocking until error or shutdown
    pub fn run(&self, shutdown: &std::sync::atomic::AtomicBool) -> Result<(), RtspError> {
        let mut child = self.spawn_ffmpeg()?;
        let stdout = child.stdout.take().ok_or(RtspError::FfmpegFailed(
            "failed to capture stdout".to_string(),
        ))?;

        tracing::info!(camera = %self.camera_id, "ffmpeg pipeline started");

        let result = self.process_stream(stdout, shutdown);

        // Clean up child process
        let _ = child.kill();
        let _ = child.wait();

        result
    }

    fn spawn_ffmpeg(&self) -> Result<Child, RtspError> {
        // Output MPEG-TS format which includes keyframe flags in adaptation field
        // -fflags +genpts ensures proper timestamps
        // -rtsp_transport tcp for reliable delivery
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "warning",
                "-rtsp_transport",
                "tcp",
                "-i",
                &self.url,
                "-c:v",
                "copy", // No re-encoding
                "-an",  // No audio
                "-f",
                "mpegts", // MPEG-TS container with keyframe flags
                "-mpegts_copyts",
                "1", // Preserve timestamps
                "-", // Output to stdout
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    RtspError::FfmpegNotFound
                } else {
                    RtspError::Io(e)
                }
            })
    }

    fn process_stream<R: Read>(
        &self,
        mut reader: R,
        shutdown: &std::sync::atomic::AtomicBool,
    ) -> Result<(), RtspError> {
        let mut parser = MpegTsParser::new();
        let mut accumulator = GopAccumulator::new(self.camera_id.clone(), Arc::clone(&self.buffer));
        let mut buf = [0u8; 188 * 64]; // Read multiple TS packets at once

        let start = Instant::now();

        while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                tracing::warn!(camera = %self.camera_id, "ffmpeg stream ended");
                return Ok(());
            }

            // Parse MPEG-TS packets
            for frame in parser.parse(&buf[..n]) {
                let pts_ns = frame
                    .pts
                    .map(|p| p * 1_000_000_000 / 90_000)
                    .unwrap_or_else(|| start.elapsed().as_nanos() as u64);
                accumulator.handle_frame(&frame.data, pts_ns, frame.is_keyframe);
            }
        }

        Ok(())
    }
}

/// Accumulates frames into GOP segments
struct GopAccumulator {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    current_gop: Option<GopSegment>,
}

impl GopAccumulator {
    fn new(camera_id: String, buffer: Arc<RwLock<HotBuffer>>) -> Self {
        Self {
            camera_id,
            buffer,
            current_gop: None,
        }
    }

    fn handle_frame(&mut self, data: &[u8], pts_ns: u64, is_keyframe: bool) {
        if is_keyframe {
            // Finalize and push current GOP
            if let Some(mut gop) = self.current_gop.take() {
                gop.finalize(pts_ns);
                if gop.frame_count > 0 {
                    if let Ok(mut hot) = self.buffer.write() {
                        hot.push(gop);
                    }
                }
            }
            self.current_gop = Some(GopSegment::new(pts_ns));
            tracing::debug!(camera = %self.camera_id, "keyframe detected, starting new GOP");
        }

        // Initialize first GOP if needed
        if self.current_gop.is_none() {
            self.current_gop = Some(GopSegment::new(pts_ns));
            tracing::debug!(camera = %self.camera_id, "initializing first GOP");
        }

        if let Some(ref mut gop) = self.current_gop {
            gop.append_frame(data, pts_ns);
        }
    }
}

/// MPEG-TS parser that extracts H.264 frames and keyframe flags
struct MpegTsParser {
    video_pid: Option<u16>,
    buffer: Vec<u8>,
    current_pts: Option<u64>,
    current_is_keyframe: bool,
}

struct ParsedFrame {
    data: Vec<u8>,
    pts: Option<u64>,
    is_keyframe: bool,
}

impl MpegTsParser {
    fn new() -> Self {
        Self {
            video_pid: None,
            buffer: Vec::new(),
            current_pts: None,
            current_is_keyframe: false,
        }
    }

    fn parse(&mut self, data: &[u8]) -> Vec<ParsedFrame> {
        let mut frames = Vec::new();
        let mut offset = 0;

        while offset + 188 <= data.len() {
            if data[offset] != 0x47 {
                // Sync byte not found, try to resync
                offset += 1;
                continue;
            }

            let packet = &data[offset..offset + 188];
            if let Some(frame) = self.parse_packet(packet) {
                frames.push(frame);
            }
            offset += 188;
        }

        frames
    }

    fn parse_packet(&mut self, packet: &[u8]) -> Option<ParsedFrame> {
        let pid = ((packet[1] as u16 & 0x1F) << 8) | packet[2] as u16;
        let payload_start = (packet[1] & 0x40) != 0;
        let has_adaptation = (packet[3] & 0x20) != 0;
        let has_payload = (packet[3] & 0x10) != 0;

        // Handle PAT (Program Association Table)
        if pid == 0 {
            self.parse_pat(packet);
            return None;
        }

        // Handle PMT (Program Map Table) - we detect video PID here
        if pid == 0x1000 {
            // Common PMT PID, but we should get it from PAT
            self.parse_pmt(packet);
            return None;
        }

        // Only process video PID
        let video_pid = self.video_pid.unwrap_or(0x100); // Default video PID
        if pid != video_pid {
            // Try common video PIDs if not yet detected
            if self.video_pid.is_none() && (pid == 0x100 || pid == 0x101 || pid == 0x1011) {
                self.video_pid = Some(pid);
            } else {
                return None;
            }
        }

        let mut payload_offset = 4;

        // Parse adaptation field
        if has_adaptation {
            let adaptation_len = packet[4] as usize;
            if adaptation_len > 0 && adaptation_len < 184 {
                let adaptation = &packet[5..5 + adaptation_len.min(183)];

                // Check random_access_indicator (bit 6 of adaptation flags)
                if !adaptation.is_empty() && (adaptation[0] & 0x40) != 0 {
                    self.current_is_keyframe = true;
                }

                // Parse PCR/PTS if present
                if adaptation.len() >= 6 && (adaptation[0] & 0x10) != 0 {
                    // PCR present - could extract timing here
                }
            }
            payload_offset = 5 + adaptation_len.min(183);
        }

        if !has_payload || payload_offset >= 188 {
            return None;
        }

        let payload = &packet[payload_offset..188];

        // If payload starts new PES packet
        if payload_start && payload.len() >= 9 {
            // Emit previous frame if we have data
            let result = if !self.buffer.is_empty() {
                Some(ParsedFrame {
                    data: std::mem::take(&mut self.buffer),
                    pts: self.current_pts.take(),
                    is_keyframe: std::mem::replace(&mut self.current_is_keyframe, false),
                })
            } else {
                None
            };

            // Parse PES header
            if payload[0] == 0x00 && payload[1] == 0x00 && payload[2] == 0x01 {
                let stream_id = payload[3];

                // Video stream IDs: 0xE0-0xEF
                if (0xE0..=0xEF).contains(&stream_id) {
                    let pes_header_len = payload[8] as usize;
                    let pts_dts_flags = (payload[7] >> 6) & 0x03;

                    // Extract PTS if present
                    if pts_dts_flags >= 2 && payload.len() >= 14 {
                        let pts = self.parse_pts(&payload[9..14]);
                        self.current_pts = Some(pts);
                    }

                    // Skip PES header to get to H.264 data
                    let h264_start = 9 + pes_header_len;
                    if h264_start < payload.len() {
                        self.buffer.extend_from_slice(&payload[h264_start..]);
                    }
                }
            }

            return result;
        }

        // Continuation of PES packet
        self.buffer.extend_from_slice(payload);

        None
    }

    fn parse_pat(&mut self, packet: &[u8]) {
        // Simplified PAT parsing - just look for PMT PID
        let payload_offset = if (packet[3] & 0x20) != 0 {
            5 + packet[4] as usize
        } else {
            4
        };

        if payload_offset + 12 > 188 {
            return;
        }

        // PAT starts with pointer field when payload_unit_start is set
        let start = if (packet[1] & 0x40) != 0 {
            payload_offset + 1 + packet[payload_offset] as usize
        } else {
            payload_offset
        };

        if start + 12 > 188 {
            return;
        }

        // Skip table header and find first program
        if start + 8 + 4 <= 188 {
            let pmt_pid = ((packet[start + 10] as u16 & 0x1F) << 8) | packet[start + 11] as u16;
            if pmt_pid != 0 && pmt_pid != 0x1FFF {
                tracing::trace!(pmt_pid, "found PMT PID in PAT");
            }
        }
    }

    fn parse_pmt(&mut self, packet: &[u8]) {
        // Simplified PMT parsing to find video PID
        let payload_offset = if (packet[3] & 0x20) != 0 {
            5 + packet[4] as usize
        } else {
            4
        };

        if payload_offset >= 188 {
            return;
        }

        let start = if (packet[1] & 0x40) != 0 && payload_offset < 188 {
            payload_offset + 1 + packet[payload_offset] as usize
        } else {
            payload_offset
        };

        if start + 12 > 188 {
            return;
        }

        // Look for H.264 stream type (0x1B) in program loop
        let program_info_len = ((packet.get(start + 10).copied().unwrap_or(0) as usize & 0x0F)
            << 8)
            | packet.get(start + 11).copied().unwrap_or(0) as usize;

        let mut pos = start + 12 + program_info_len;
        while pos + 5 <= 188 {
            let stream_type = packet[pos];
            let elem_pid = ((packet[pos + 1] as u16 & 0x1F) << 8) | packet[pos + 2] as u16;
            let es_info_len = ((packet[pos + 3] as usize & 0x0F) << 8) | packet[pos + 4] as usize;

            // H.264 stream type
            if stream_type == 0x1B && self.video_pid.is_none() {
                self.video_pid = Some(elem_pid);
                tracing::debug!(video_pid = elem_pid, "detected H.264 video PID");
                break;
            }

            pos += 5 + es_info_len;
        }
    }

    fn parse_pts(&self, data: &[u8]) -> u64 {
        ((data[0] as u64 >> 1) & 0x07) << 30
            | (data[1] as u64) << 22
            | ((data[2] as u64 >> 1) & 0x7F) << 15
            | (data[3] as u64) << 7
            | ((data[4] as u64 >> 1) & 0x7F)
    }
}
