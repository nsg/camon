use std::io::Read;
use std::os::unix::io::{AsRawFd, RawFd};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

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

    pub fn run(&self, shutdown: &std::sync::atomic::AtomicBool) -> Result<(), RtspError> {
        let mut child = self.spawn_ffmpeg()?;
        let stdout = child.stdout.take().ok_or(RtspError::FfmpegFailed(
            "failed to capture stdout".to_string(),
        ))?;

        tracing::info!(camera = %self.camera_id, "ffmpeg pipeline started");

        let result = self.process_stream(stdout, shutdown);

        let _ = child.kill();
        let _ = child.wait();

        result
    }

    fn spawn_ffmpeg(&self) -> Result<Child, RtspError> {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "warning",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "10000000",
                "-i",
                &self.url,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-f",
                "mpegts",
                "-",
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

    fn process_stream<R: Read + AsRawFd>(
        &self,
        mut reader: R,
        shutdown: &std::sync::atomic::AtomicBool,
    ) -> Result<(), RtspError> {
        let mut segmenter = MpegTsSegmenter::new(self.camera_id.clone(), Arc::clone(&self.buffer));
        let mut buf = [0u8; 188 * 64];
        let fd = reader.as_raw_fd();

        while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            // Poll with timeout so we can check the shutdown flag
            if !poll_readable(fd, 500) {
                continue;
            }
            let n = reader.read(&mut buf)?;
            if n == 0 {
                tracing::warn!(camera = %self.camera_id, "ffmpeg stream ended");
                return Ok(());
            }
            segmenter.process(&buf[..n]);
        }

        Ok(())
    }
}

/// Segments raw MPEG-TS stream based on keyframe detection
/// Stores raw MPEG-TS packets directly - no re-muxing needed
struct MpegTsSegmenter {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    current_segment: Option<GopSegment>,
    video_pid: Option<u16>,
    audio_pid: Option<u16>,
    pat_packet: Option<[u8; 188]>,
    pmt_packet: Option<[u8; 188]>,
    pmt_pid: Option<u16>,
    partial_packet: Vec<u8>,
    current_media_pts: Option<u64>,
    prev_media_pts: Option<u64>,
}

impl MpegTsSegmenter {
    fn new(camera_id: String, buffer: Arc<RwLock<HotBuffer>>) -> Self {
        Self {
            camera_id,
            buffer,
            current_segment: None,
            video_pid: None,
            audio_pid: None,
            pat_packet: None,
            pmt_packet: None,
            pmt_pid: None,
            partial_packet: Vec::with_capacity(188),
            current_media_pts: None,
            prev_media_pts: None,
        }
    }

    fn process(&mut self, data: &[u8]) {
        // Handle partial packet from previous read
        let mut offset = 0;
        if !self.partial_packet.is_empty() {
            let needed = 188 - self.partial_packet.len();
            if data.len() >= needed {
                self.partial_packet.extend_from_slice(&data[..needed]);
                let packet: [u8; 188] = self.partial_packet[..188].try_into().unwrap();
                self.partial_packet.clear();
                if packet[0] == 0x47 {
                    self.process_packet(&packet);
                }
                offset = needed;
            } else {
                self.partial_packet.extend_from_slice(data);
                return;
            }
        }

        // Find sync byte and process aligned packets
        while offset < data.len() {
            // Look for sync byte
            if data[offset] != 0x47 {
                offset += 1;
                continue;
            }

            // Check if we have a complete packet
            if offset + 188 <= data.len() {
                let packet: &[u8; 188] = data[offset..offset + 188].try_into().unwrap();
                self.process_packet(packet);
                offset += 188;
            } else {
                // Save partial packet for next read
                self.partial_packet.extend_from_slice(&data[offset..]);
                break;
            }
        }
    }

    fn process_packet(&mut self, packet: &[u8]) {
        let pid = ((packet[1] as u16 & 0x1F) << 8) | packet[2] as u16;
        let has_adaptation = (packet[3] & 0x20) != 0;

        // Capture PAT
        if pid == 0 {
            let mut pat = [0u8; 188];
            pat.copy_from_slice(packet);
            self.pat_packet = Some(pat);
            self.parse_pat(packet);
        }

        // Capture PMT
        if Some(pid) == self.pmt_pid {
            let mut pmt = [0u8; 188];
            pmt.copy_from_slice(packet);
            self.pmt_packet = Some(pmt);
            self.parse_pmt(packet);
        }

        // Detect keyframe from random_access_indicator
        let is_keyframe = if has_adaptation && Some(pid) == self.video_pid {
            let adaptation_len = packet[4] as usize;
            if adaptation_len > 0 && adaptation_len < 184 {
                (packet[5] & 0x40) != 0
            } else {
                false
            }
        } else {
            false
        };

        // Extract media PTS from video packets with PES header
        if Some(pid) == self.video_pid {
            let pusi = (packet[1] & 0x40) != 0;
            if pusi {
                if let Some(pts) = Self::extract_pes_pts(packet) {
                    self.current_media_pts = Some(pts);
                }
            }
        }

        // Start new segment on keyframe
        if is_keyframe {
            let pts_ns = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            self.finalize_segment(pts_ns);
            self.start_segment(pts_ns);
        }

        // Append packet to current segment
        if let Some(ref mut segment) = self.current_segment {
            segment.data.extend_from_slice(packet);
            if Some(pid) == self.video_pid {
                segment.frame_count += 1;
            }
        }
    }

    fn start_segment(&mut self, pts_ns: u64) {
        let mut segment = GopSegment::new(pts_ns);

        // Prepend PAT and PMT for segment independence
        // Reset continuity counters to 0 for clean segment start
        if let Some(mut pat) = self.pat_packet {
            pat[3] &= 0xF0; // Reset continuity counter to 0
            segment.data.extend_from_slice(&pat);
        }
        if let Some(mut pmt) = self.pmt_packet {
            pmt[3] &= 0xF0; // Reset continuity counter to 0
            segment.data.extend_from_slice(&pmt);
        }

        self.current_segment = Some(segment);
    }

    fn finalize_segment(&mut self, end_pts_ns: u64) {
        if let Some(mut segment) = self.current_segment.take() {
            segment.finalize_with_media_pts(
                end_pts_ns,
                self.current_media_pts,
                self.prev_media_pts,
            );
            self.prev_media_pts = self.current_media_pts;
            if segment.frame_count > 0 {
                if let Ok(mut hot) = self.buffer.write() {
                    hot.push(segment);
                }
            }
        }
    }

    fn extract_pes_pts(packet: &[u8]) -> Option<u64> {
        let has_adaptation = (packet[3] & 0x20) != 0;
        let has_payload = (packet[3] & 0x10) != 0;
        if !has_payload {
            return None;
        }

        let payload_start = if has_adaptation {
            5 + packet[4] as usize
        } else {
            4
        };

        // PES header: 0x00 0x00 0x01 stream_id
        if payload_start + 14 > 188 {
            return None;
        }
        let p = &packet[payload_start..];
        if p[0] != 0x00 || p[1] != 0x00 || p[2] != 0x01 {
            return None;
        }

        // Check PTS_DTS_flags (bits 7-6 of byte 7)
        let pts_dts_flags = (p[7] >> 6) & 0x03;
        if pts_dts_flags < 2 {
            return None; // No PTS present
        }

        // Parse 33-bit PTS from 5 bytes (bytes 9-13)
        if payload_start + 14 > 188 {
            return None;
        }
        let pts = ((p[9] as u64 & 0x0E) << 29)
            | ((p[10] as u64) << 22)
            | ((p[11] as u64 & 0xFE) << 14)
            | ((p[12] as u64) << 7)
            | ((p[13] as u64) >> 1);

        Some(pts)
    }

    fn parse_pat(&mut self, packet: &[u8]) {
        let payload_offset = if (packet[3] & 0x20) != 0 {
            5 + packet[4] as usize
        } else {
            4
        };

        if payload_offset + 12 > 188 {
            return;
        }

        let start = if (packet[1] & 0x40) != 0 {
            payload_offset + 1 + packet[payload_offset] as usize
        } else {
            payload_offset
        };

        if start + 12 > 188 {
            return;
        }

        if start + 8 + 4 <= 188 {
            let pmt_pid = ((packet[start + 10] as u16 & 0x1F) << 8) | packet[start + 11] as u16;
            if pmt_pid != 0 && pmt_pid != 0x1FFF && self.pmt_pid.is_none() {
                self.pmt_pid = Some(pmt_pid);
                tracing::debug!(camera = %self.camera_id, pmt_pid, "detected PMT PID");
            }
        }
    }

    fn parse_pmt(&mut self, packet: &[u8]) {
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

        let program_info_len = ((packet.get(start + 10).copied().unwrap_or(0) as usize & 0x0F)
            << 8)
            | packet.get(start + 11).copied().unwrap_or(0) as usize;

        let mut pos = start + 12 + program_info_len;
        while pos + 5 <= 188 {
            let stream_type = packet[pos];
            let elem_pid = ((packet[pos + 1] as u16 & 0x1F) << 8) | packet[pos + 2] as u16;
            let es_info_len = ((packet[pos + 3] as usize & 0x0F) << 8) | packet[pos + 4] as usize;

            // H.264 stream type = 0x1B
            if stream_type == 0x1B && self.video_pid.is_none() {
                self.video_pid = Some(elem_pid);
                tracing::debug!(camera = %self.camera_id, video_pid = elem_pid, "detected H.264 video PID");
            }

            // AAC audio stream types: 0x0F (MPEG-2 AAC), 0x11 (MPEG-4 AAC), 0x81 (AC-3)
            if (stream_type == 0x0F || stream_type == 0x11 || stream_type == 0x81)
                && self.audio_pid.is_none()
            {
                self.audio_pid = Some(elem_pid);
                tracing::debug!(camera = %self.camera_id, audio_pid = elem_pid, stream_type, "detected audio PID");
            }

            pos += 5 + es_info_len;
        }
    }
}

/// Poll a file descriptor for readability with a timeout in milliseconds.
/// Returns true if the fd is readable, false on timeout.
fn poll_readable(fd: RawFd, timeout_ms: i32) -> bool {
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
    ret > 0
}
