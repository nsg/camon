use std::io::{BufRead, BufReader, Read};
use std::os::unix::io::{AsRawFd, RawFd};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use thiserror::Error;

use crate::buffer::{GopSegment, HotBuffer};
use crate::config::CameraConfig;
use crate::locks::LockExt;
use crate::retry::Streak;

/// Reconnect if no bytes are read from ffmpeg for this long.
const DATA_TIMEOUT_SECS: u64 = 30;
/// Reconnect if bytes are flowing but no segment (keyframe) is produced for this long.
const NO_SEGMENT_TIMEOUT_SECS: u64 = 60;
/// Consecutive runs that recorded nothing before the diagnosis is raised from a
/// warning about this connection to an error about the camera as a whole.
const ESCALATE_AFTER: u32 = 4;
/// Video must be watched for at least this long before anything is said about
/// its keyframes. A PMT parsed just before the watchdog fires leaves too short
/// a window for "no keyframe" to mean more than "camon did not look for long",
/// and no stream is accused on that.
const MIN_KEYFRAME_WINDOW_SECS: u64 = 30;
/// A run that kept going this long was a working stream that hiccupped, not a
/// camera that never gets going, so its next stop is worth a line again.
const SETTLED_RUN_SECS: u64 = 600;

#[derive(Debug, Error)]
pub enum RtspError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffmpeg not found")]
    FfmpegNotFound,
    #[error("ffmpeg failed: {0}")]
    FfmpegFailed(String),
    /// A watchdog fired or ffmpeg ended the stream. Carries what the stream
    /// actually contained, so a run that recorded nothing can say why instead
    /// of reconnecting forever on an unexplained "no segments".
    #[error("{}", .0.summary())]
    NoRecording(StreamFailure),
}

/// What the segmenter counted during one connection. Bytes at the socket prove
/// nothing on their own: a stream can flow at full rate and still be unusable,
/// so every step between "bytes" and "segment" is counted separately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StreamCounts {
    bytes: u64,
    /// Bytes thrown away while hunting for the next sync byte. The scan
    /// resynchronizes on any 0x47, so garbage still yields "TS packets"; the
    /// ratio of these to `bytes` is what tells transport stream from noise.
    skipped_bytes: u64,
    ts_packets: u64,
    /// PMT PID as named by the PAT, so a PAT that never parsed is not blamed
    /// on the PMT.
    pmt_pid: Option<u16>,
    /// Packets on the PMT PID that parsed — repeats of one table, not distinct
    /// sections: ffmpeg re-emits the PMT about ten times a second.
    pmt_packets: u64,
    video_pid: Option<u16>,
    video_packets: u64,
    keyframes: u64,
    segments: u64,
}

/// The counts plus the timings that say what they are worth. Zero keyframes
/// means nothing without the window it was counted over: video identified one
/// second before the watchdog fired proves only that camon did not look long.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    counts: StreamCounts,
    /// Length of the whole run, which is not the length of the keyframe window:
    /// it starts when the segmenter is built, before ffmpeg has connected.
    run_secs: u64,
    /// Seconds between identifying the video PID and the end of the run; 0
    /// when no video PID was ever identified.
    video_secs: u64,
    /// How long ago the first and the most recent keyframe arrived.
    first_keyframe_secs: Option<u64>,
    last_keyframe_secs: Option<u64>,
}

/// How a run ended. Structural faults hold however a run ended, but anything
/// said about keyframes depends on the run reaching the no-segment watchdog: a
/// run cut short by EOF or a stall may simply not have got that far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunEnd {
    DataTimeout,
    SegmentTimeout,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamFault {
    NoData,
    NotTransportStream,
    NoProgramMap,
    NoVideoStream,
    NoVideoPackets,
    NoKeyframeFlag,
    OneKeyframeOnly,
    EndedEarly,
    WindowTooShort,
}

/// A finished run and the evidence it produced.
///
/// Every message built here reports what camon observed and the window it was
/// observed over. These counters cannot see the camera, only ffmpeg's stdout,
/// so none of them may be phrased as a verdict on the camera: an operator who
/// replaces a working camera on camon's say-so was misled by this code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamFailure {
    stats: StreamStats,
    end: RunEnd,
}

impl StreamFailure {
    /// The fault this run demonstrates, or `None` when the stream did record
    /// and merely stopped — an ordinary reconnect, nothing to look into.
    fn fault(&self) -> Option<StreamFault> {
        let c = &self.stats.counts;
        if c.segments > 0 {
            return None;
        }
        if c.bytes == 0 {
            return Some(StreamFault::NoData);
        }
        // Not "no packets": the scan resynchronizes on any sync byte, so noise
        // yields packets too. Mostly-skipped output is not a transport stream.
        if c.ts_packets == 0 || c.skipped_bytes > c.bytes / 2 {
            return Some(StreamFault::NotTransportStream);
        }
        if c.video_pid.is_none() {
            return Some(if c.pmt_packets == 0 {
                StreamFault::NoProgramMap
            } else {
                StreamFault::NoVideoStream
            });
        }
        if c.video_packets == 0 {
            return Some(StreamFault::NoVideoPackets);
        }
        // Past here the verdict is about keyframes, which takes both a run that
        // reached the watchdog and a long enough look at the video PID.
        if self.end != RunEnd::SegmentTimeout {
            return Some(StreamFault::EndedEarly);
        }
        if self.stats.video_secs < MIN_KEYFRAME_WINDOW_SECS {
            return Some(StreamFault::WindowTooShort);
        }
        if c.keyframes == 0 {
            return Some(StreamFault::NoKeyframeFlag);
        }
        // Only one keyframe can reach here: the second one finalizes a segment
        // that always holds at least the frame of the first, so it records.
        Some(StreamFault::OneKeyframeOnly)
    }

    fn summary(&self) -> String {
        let c = &self.stats.counts;
        let secs = self.stats.video_secs;
        let run = self.stats.run_secs;
        let pid = c.video_pid.unwrap_or(0);
        match self.fault() {
            None => match self.end {
                RunEnd::DataTimeout => format!(
                    "no data from ffmpeg for {DATA_TIMEOUT_SECS}s after {} segments in {run}s, \
                     reconnecting",
                    c.segments
                ),
                RunEnd::SegmentTimeout => format!(
                    "no keyframe for {NO_SEGMENT_TIMEOUT_SECS}s after {} segments in {run}s, \
                     reconnecting",
                    c.segments
                ),
                RunEnd::Eof => format!(
                    "ffmpeg ended the stream after {} segments in {run}s",
                    c.segments
                ),
            },
            Some(StreamFault::NoData) => {
                let observed = match self.end {
                    RunEnd::Eof => "ffmpeg exited without writing a byte to camon".to_string(),
                    _ => format!("no bytes from ffmpeg for {DATA_TIMEOUT_SECS}s"),
                };
                format!(
                    "{observed} — camon saw nothing to segment, so whatever went wrong is \
                     upstream of it: ffmpeg itself, the URL, the credentials or the network \
                     path. ffmpeg's own errors say which, and camon logs them at debug level, \
                     so re-run with RUST_LOG=debug to see them"
                )
            }
            Some(StreamFault::NotTransportStream) if c.ts_packets == 0 => format!(
                "{} bytes of ffmpeg output with no complete 188-byte TS packet found in them — \
                 either the output is not the MPEG-TS camon asked for, or the run ended inside \
                 the first packet. ffmpeg's own output is logged at debug level (RUST_LOG=debug)",
                c.bytes
            ),
            Some(StreamFault::NotTransportStream) => format!(
                "{} of {} bytes of ffmpeg output belonged to no TS packet, leaving {} packets \
                 recovered by resynchronizing — this does not look like the MPEG-TS camon asked \
                 for. ffmpeg's own output is logged at debug level (RUST_LOG=debug)",
                c.skipped_bytes, c.bytes, c.ts_packets
            ),
            Some(StreamFault::NoProgramMap) => match c.pmt_pid {
                None => format!(
                    "{} TS packets parsed and no PAT naming a PMT PID, so camon never learned \
                     which PID carries video",
                    c.ts_packets
                ),
                Some(pmt_pid) => format!(
                    "{} TS packets parsed and no readable PMT on PID {pmt_pid}, the PID the PAT \
                     named, so camon never learned which PID carries video",
                    c.ts_packets
                ),
            },
            Some(StreamFault::NoVideoStream) => format!(
                "camon parsed the PMT {} times and it never listed an H.264 elementary stream \
                 (stream type 0x1B) — camon records H.264 only, so an H.265/HEVC or MJPEG \
                 stream is not recorded; check which codec this stream serves",
                c.pmt_packets
            ),
            Some(StreamFault::NoVideoPackets) => format!(
                "camon took video PID {pid} from the first PMT it parsed and then saw no packet \
                 on it in {} TS packets — the stream may be carrying video on a PID that a \
                 later PMT announced",
                c.ts_packets
            ),
            Some(StreamFault::WindowTooShort) => format!(
                "no segment completed, and video PID {pid} was identified only {secs}s before \
                 the {NO_SEGMENT_TIMEOUT_SECS}s watchdog fired — too short a look to say \
                 anything about this stream's keyframes"
            ),
            Some(StreamFault::NoKeyframeFlag) => format!(
                "{} packets on video PID {pid} in {secs}s, none of them flagging \
                 random_access_indicator in the MPEG-TS adaptation field — camon starts a \
                 segment only on that flag, so nothing was recorded. Either this stream never \
                 sets it (some encoders and remuxers do not) or its I-frame interval is longer \
                 than the {secs}s observed; RUST_LOG=debug shows what ffmpeg reported",
                c.video_packets
            ),
            Some(StreamFault::OneKeyframeOnly) => format!(
                "{} keyframe on video PID {pid} {}s ago and no further one in the {secs}s of \
                 video observed, so no segment completed — camon closes a segment only when the \
                 next keyframe arrives. A long I-frame (GOP) interval on the camera looks like \
                 this, and so does a connection that spent most of the run's {run}s getting \
                 started",
                c.keyframes,
                self.stats.last_keyframe_secs.unwrap_or(0)
            ),
            Some(StreamFault::EndedEarly) => match self.end {
                RunEnd::Eof => format!(
                    "ffmpeg ended the stream after {} bytes and {} video packets in {run}s, \
                     before a segment completed",
                    c.bytes, c.video_packets
                ),
                _ => format!(
                    "no data from ffmpeg for {DATA_TIMEOUT_SECS}s after {} bytes and {} video \
                     packets, before a segment completed — ffmpeg may still have been running",
                    c.bytes, c.video_packets
                ),
            },
        }
    }
}

/// What a finished run should produce in the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Report {
    /// The run recorded and then stopped, on a milestone of its own streak.
    Stopped(u32),
    /// The same between milestones: a camera that records one segment and then
    /// wedges would otherwise repeat an identical warning every reconnect.
    StoppedQuiet,
    /// A run that recorded nothing, between milestones.
    Quiet,
    /// A milestone below the escalation threshold: about this connection.
    Connection(u32),
    /// A milestone at or above it: about the camera as a whole.
    Camera(u32),
}

/// Decides how loudly a finished run is reported.
///
/// Each streak is cleared only by evidence that the condition it counts has
/// gone away: a run that recorded clears the fault streak, a run that lasted
/// clears the stall streak. Outcomes that show neither — an unrelated error, a
/// panic — never reach this type at all, so they cannot defer a diagnosis by
/// resetting a count they say nothing about.
#[derive(Default)]
pub struct NoRecordingTracker {
    fault: Streak,
    stall: Streak,
}

impl NoRecordingTracker {
    fn classify(&mut self, failure: &StreamFailure) -> Report {
        if failure.fault().is_none() {
            self.fault.reset();
            if failure.stats.run_secs >= SETTLED_RUN_SECS {
                self.stall.reset();
            }
            return match self.stall.record() {
                Some(count) => Report::Stopped(count),
                None => Report::StoppedQuiet,
            };
        }
        match self.fault.record() {
            Some(streak) if streak >= ESCALATE_AFTER => Report::Camera(streak),
            Some(streak) => Report::Connection(streak),
            None => Report::Quiet,
        }
    }

    pub fn report(&mut self, camera_id: &str, failure: &StreamFailure) {
        let summary = failure.summary();
        match self.classify(failure) {
            Report::Stopped(_) => tracing::warn!(camera = %camera_id, "{summary}"),
            Report::StoppedQuiet => tracing::debug!(camera = %camera_id, "{summary}"),
            Report::Quiet => {
                tracing::debug!(camera = %camera_id, "recorded nothing this connection: {summary}")
            }
            Report::Connection(_) => {
                tracing::warn!(camera = %camera_id, "recorded nothing this connection: {summary}")
            }
            Report::Camera(streak) => tracing::error!(
                camera = %camera_id,
                "recorded nothing in {streak} connection attempts in a row: {summary}"
            ),
        }
    }
}

pub struct FfmpegPipeline {
    camera_id: String,
    url: String,
    buffer: Arc<RwLock<HotBuffer>>,
}

impl FfmpegPipeline {
    pub fn new(config: &CameraConfig, buffer: Arc<RwLock<HotBuffer>>) -> Self {
        Self {
            camera_id: config.id.clone(),
            url: config.url.clone(),
            buffer,
        }
    }

    pub fn run(&self, shutdown: &std::sync::atomic::AtomicBool) -> Result<(), RtspError> {
        let mut child = self.spawn_ffmpeg()?;
        let stdout = child.stdout.take().ok_or(RtspError::FfmpegFailed(
            "failed to capture stdout".to_string(),
        ))?;

        // Drain stderr so the pipe never fills and blocks ffmpeg. Runs on a
        // plain thread (this fn is inside spawn_blocking, no async runtime) and
        // exits on EOF when the child is killed/exits below. ffmpeg echoes the
        // full RTSP URL in its messages, so the password is scrubbed before
        // anything reaches the log.
        let stderr_thread = child.stderr.take().map(|stderr| {
            let camera_id = self.camera_id.clone();
            let password = crate::config::url_password(&self.url).map(str::to_owned);
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(line) => {
                            let line = match &password {
                                Some(pw) => line.replace(pw, "****"),
                                None => line,
                            };
                            tracing::debug!(camera = %camera_id, "ffmpeg: {}", line);
                        }
                        Err(_) => break,
                    }
                }
            })
        });

        tracing::info!(camera = %self.camera_id, "ffmpeg pipeline started");

        let result = self.process_stream(stdout, shutdown);

        let _ = child.kill();
        let _ = child.wait();

        if let Some(handle) = stderr_thread {
            let _ = handle.join();
        }

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
        reader: R,
        shutdown: &std::sync::atomic::AtomicBool,
    ) -> Result<(), RtspError> {
        let mut segmenter = MpegTsSegmenter::new(self.camera_id.clone(), Arc::clone(&self.buffer));
        let result = segmenter.read_stream(reader, shutdown);
        // After the diagnosis is built, so the flushed segment cannot turn a run
        // that recorded nothing into one that appears to have recorded.
        segmenter.flush_end_of_stream();
        result
    }
}

/// Segments raw MPEG-TS stream based on keyframe detection
/// Stores raw MPEG-TS packets directly - no re-muxing needed
struct MpegTsSegmenter {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    current_segment: Option<GopSegment>,
    /// Incremental byte buffer for the in-progress segment; wrapped in an Arc
    /// once at finalize time so readers share it without copying.
    current_data: Vec<u8>,
    video_pid: Option<u16>,
    audio_pid: Option<u16>,
    pat_packet: Option<[u8; 188]>,
    pmt_packet: Option<[u8; 188]>,
    pmt_pid: Option<u16>,
    partial_packet: Vec<u8>,
    current_media_pts: Option<u64>,
    prev_media_pts: Option<u64>,
    /// PES starts on the video PID inside the segment being filled. Two of them
    /// mean its first frame is complete, which is what an end-of-stream flush
    /// needs to know.
    video_pes_starts: u32,
    last_segment_at: Instant,
    counts: StreamCounts,
    started: Instant,
    /// When the video PID was identified and when keyframes arrived: a count
    /// of zero keyframes is only worth the window it was counted over.
    video_pid_at: Option<Instant>,
    first_keyframe_at: Option<Instant>,
    last_keyframe_at: Option<Instant>,
}

impl MpegTsSegmenter {
    fn new(camera_id: String, buffer: Arc<RwLock<HotBuffer>>) -> Self {
        Self {
            camera_id,
            buffer,
            current_segment: None,
            current_data: Vec::new(),
            video_pid: None,
            audio_pid: None,
            pat_packet: None,
            pmt_packet: None,
            pmt_pid: None,
            partial_packet: Vec::with_capacity(188),
            current_media_pts: None,
            prev_media_pts: None,
            video_pes_starts: 0,
            last_segment_at: Instant::now(),
            counts: StreamCounts::default(),
            started: Instant::now(),
            video_pid_at: None,
            first_keyframe_at: None,
            last_keyframe_at: None,
        }
    }

    fn read_stream<R: Read + AsRawFd>(
        &mut self,
        mut reader: R,
        shutdown: &std::sync::atomic::AtomicBool,
    ) -> Result<(), RtspError> {
        let mut buf = [0u8; 188 * 64];
        let fd = reader.as_raw_fd();
        let mut last_data = Instant::now();

        while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            // Data watchdog: no bytes at all for too long means the stream wedged.
            if last_data.elapsed() >= Duration::from_secs(DATA_TIMEOUT_SECS) {
                return Err(self.failure(RunEnd::DataTimeout));
            }
            // No-segment tripwire: bytes flowing but no keyframe-bounded segment produced.
            if self.last_segment_at.elapsed() >= Duration::from_secs(NO_SEGMENT_TIMEOUT_SECS) {
                return Err(self.failure(RunEnd::SegmentTimeout));
            }

            // Poll with timeout so we can check the shutdown flag
            if !poll_readable(fd, 500) {
                continue;
            }
            let n = reader.read(&mut buf)?;
            if n == 0 {
                // ffmpeg gave up on its own. Reported through the same path as
                // the watchdogs so a stream that ends before it ever records
                // gets the diagnosis, not a bare "stream ended" per reconnect.
                return Err(self.failure(RunEnd::Eof));
            }
            last_data = Instant::now();
            self.process(&buf[..n]);
        }

        Ok(())
    }

    /// Push the GOP that was still being filled when the stream ended, instead
    /// of losing it with the connection.
    ///
    /// The cut lands mid-packet, so the segment's last frame is truncated. It is
    /// only offered once a second frame has started, which proves the keyframe
    /// the segment opens on arrived whole: with PAT and PMT already prepended
    /// that segment decodes on its own like any other, a decoder discarding the
    /// unterminated tail. Below that the segment holds a fragment of a keyframe
    /// and nothing can be decoded from it, so it is dropped rather than left in
    /// the buffer for the analyzer and the player to choke on.
    fn flush_end_of_stream(&mut self) {
        if self.video_pes_starts < 2 {
            return;
        }
        let now_ns = wall_clock_ns();
        self.finalize_segment(now_ns);
    }

    fn failure(&self, end: RunEnd) -> RtspError {
        RtspError::NoRecording(StreamFailure {
            stats: StreamStats {
                counts: self.counts,
                run_secs: self.started.elapsed().as_secs(),
                video_secs: self.video_pid_at.map_or(0, |at| at.elapsed().as_secs()),
                first_keyframe_secs: self.first_keyframe_at.map(|at| at.elapsed().as_secs()),
                last_keyframe_secs: self.last_keyframe_at.map(|at| at.elapsed().as_secs()),
            },
            end,
        })
    }

    fn process(&mut self, data: &[u8]) {
        self.counts.bytes += data.len() as u64;

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
                } else {
                    self.counts.skipped_bytes += 188;
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
                self.counts.skipped_bytes += 1;
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
        let pid = crate::mpegts::packet_pid(packet);
        self.counts.ts_packets += 1;

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

        // Detect keyframe from random_access_indicator. Shared with the
        // analyzer's keyframe count, which must see exactly what this cuts on.
        let is_keyframe =
            Some(pid) == self.video_pid && crate::mpegts::has_random_access_indicator(packet);

        let pusi = (packet[1] & 0x40) != 0;

        // Extract media PTS from video packets with PES header
        if Some(pid) == self.video_pid {
            self.counts.video_packets += 1;
            if is_keyframe {
                self.counts.keyframes += 1;
                let now = Instant::now();
                self.first_keyframe_at.get_or_insert(now);
                self.last_keyframe_at = Some(now);
            }
            if pusi {
                if let Some(pts) = crate::mpegts::extract_pes_pts(packet) {
                    self.current_media_pts = Some(pts);
                }
            }
        }

        // Start new segment on keyframe
        if is_keyframe {
            let pts_ns = wall_clock_ns();
            self.finalize_segment(pts_ns);
            self.start_segment(pts_ns);
        }

        // Append packet to current segment
        if let Some(ref mut segment) = self.current_segment {
            self.current_data.extend_from_slice(packet);
            if Some(pid) == self.video_pid {
                segment.frame_count += 1;
                if pusi {
                    self.video_pes_starts += 1;
                }
            }
        }
    }

    fn start_segment(&mut self, pts_ns: u64) {
        let segment = GopSegment::new(pts_ns);
        self.video_pes_starts = 0;

        // Prepend PAT and PMT for segment independence
        // Reset continuity counters to 0 for clean segment start
        if let Some(mut pat) = self.pat_packet {
            pat[3] &= 0xF0; // Reset continuity counter to 0
            self.current_data.extend_from_slice(&pat);
        }
        if let Some(mut pmt) = self.pmt_packet {
            pmt[3] &= 0xF0; // Reset continuity counter to 0
            self.current_data.extend_from_slice(&pmt);
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
            // Wrap the accumulated bytes once; readers share via Arc clone.
            // Drop the Vec's growth slack first — the segment lives in the
            // hot buffer for minutes, so excess capacity is held that long.
            self.current_data.shrink_to_fit();
            segment.data = Arc::new(std::mem::take(&mut self.current_data));
            if segment.frame_count > 0 {
                self.buffer.write_recover().push(segment);
                self.last_segment_at = Instant::now();
                self.counts.segments += 1;
            }
        }
    }

    fn parse_pat(&mut self, packet: &[u8]) {
        let start = match table_section_start(packet) {
            Some(s) => s,
            None => return,
        };

        if start + 12 > 188 {
            return;
        }

        let pmt_pid = ((packet[start + 10] as u16 & 0x1F) << 8) | packet[start + 11] as u16;
        if pmt_pid != 0 && pmt_pid != 0x1FFF && self.pmt_pid.is_none() {
            self.pmt_pid = Some(pmt_pid);
            self.counts.pmt_pid = Some(pmt_pid);
            tracing::debug!(camera = %self.camera_id, pmt_pid, "detected PMT PID");
        }
    }

    fn parse_pmt(&mut self, packet: &[u8]) {
        let start = match table_section_start(packet) {
            Some(s) => s,
            None => return,
        };

        if start + 12 > 188 {
            return;
        }
        self.counts.pmt_packets += 1;

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
                self.counts.video_pid = Some(elem_pid);
                self.video_pid_at = Some(Instant::now());
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

fn wall_clock_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Compute the start of a PSI table section inside an MPEG-TS packet.
/// Handles adaptation field and PUSI pointer field. Returns None if out of bounds.
fn table_section_start(packet: &[u8]) -> Option<usize> {
    let payload_offset = if (packet[3] & 0x20) != 0 {
        5 + packet[4] as usize
    } else {
        4
    };

    if payload_offset >= 188 {
        return None;
    }

    let start = if (packet[1] & 0x40) != 0 {
        payload_offset + 1 + packet[payload_offset] as usize
    } else {
        payload_offset
    };

    if start + 12 > 188 {
        return None;
    }

    Some(start)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpegts::testutil::{keyframe_packet, pes_packet};
    use crate::mpegts::TS_PACKET_SIZE;
    use crate::retry::MAX_REPORT_GAP;
    use std::sync::Mutex;
    use tracing::Level;

    const VIDEO_PID: u16 = 0x100;
    const AUDIO_PID: u16 = 0x101;
    const OTHER_PID: u16 = 0x200;
    const PMT_PID: u16 = 0x1000;
    const H264: u8 = 0x1B;
    const HEVC: u8 = 0x24;
    const VIDEO_STREAM_ID: u8 = 0xE0;
    const AUDIO_STREAM_ID: u8 = 0xC0;

    fn pat(pmt_pid: u16) -> [u8; TS_PACKET_SIZE] {
        let mut p = [0xFFu8; TS_PACKET_SIZE];
        p[0] = crate::mpegts::SYNC_BYTE;
        p[1] = 0x40; // PUSI, PID 0
        p[2] = 0x00;
        p[3] = 0x10; // payload only
        p[4] = 0x00; // pointer field, section starts at 5
        p[5] = 0x00; // table_id: PAT
        p[6] = 0xB0;
        p[7] = 0x0D; // section length
        p[8] = 0x00;
        p[9] = 0x01; // transport_stream_id
        p[10] = 0xC1;
        p[11] = 0x00;
        p[12] = 0x00;
        p[13] = 0x00;
        p[14] = 0x01; // program_number
        p[15] = 0xE0 | ((pmt_pid >> 8) as u8 & 0x1F);
        p[16] = pmt_pid as u8;
        p
    }

    fn pmt(stream_type: u8, elem_pid: u16) -> [u8; TS_PACKET_SIZE] {
        let mut p = [0xFFu8; TS_PACKET_SIZE];
        p[0] = crate::mpegts::SYNC_BYTE;
        p[1] = 0x40 | ((PMT_PID >> 8) as u8 & 0x1F);
        p[2] = PMT_PID as u8;
        p[3] = 0x10;
        p[4] = 0x00; // pointer field, section starts at 5
        p[5] = 0x02; // table_id: PMT
        p[6] = 0xB0;
        p[7] = 0x12;
        p[8] = 0x00;
        p[9] = 0x01;
        p[10] = 0xC1;
        p[11] = 0x00;
        p[12] = 0x00;
        p[13] = 0xE0 | ((elem_pid >> 8) as u8 & 0x1F);
        p[14] = elem_pid as u8; // PCR PID
        p[15] = 0xF0;
        p[16] = 0x00; // program_info_length = 0
        p[17] = stream_type;
        p[18] = 0xE0 | ((elem_pid >> 8) as u8 & 0x1F);
        p[19] = elem_pid as u8;
        p[20] = 0xF0;
        p[21] = 0x00; // ES_info_length = 0
        p
    }

    fn segmenter() -> MpegTsSegmenter {
        MpegTsSegmenter::new("cam".to_string(), HotBuffer::new("cam".to_string(), 60))
    }

    /// Counters from the real segmenter. The clock is supplied by the tests
    /// below instead of waited out, so the observation window a verdict needs
    /// can be chosen.
    fn segment_stats(packets: &[[u8; TS_PACKET_SIZE]]) -> StreamStats {
        let mut segmenter = segmenter();
        for packet in packets {
            segmenter.process(packet);
        }
        counted(segmenter.counts)
    }

    fn segment_bytes(data: &[u8]) -> StreamStats {
        let mut segmenter = segmenter();
        segmenter.process(data);
        counted(segmenter.counts)
    }

    fn counted(counts: StreamCounts) -> StreamStats {
        StreamStats {
            counts,
            ..StreamStats::default()
        }
    }

    fn watched_for(stats: StreamStats, video_secs: u64) -> StreamStats {
        StreamStats {
            video_secs,
            ..stats
        }
    }

    fn failed(stats: StreamStats, end: RunEnd) -> StreamFailure {
        StreamFailure { stats, end }
    }

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<(Level, String)>>>);

    struct Message(String);

    impl tracing::field::Visit for Message {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Captured {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut message = Message(String::new());
            event.record(&mut message);
            self.0
                .lock()
                .expect("log capture poisoned")
                .push((*event.metadata().level(), message.0));
        }
    }

    /// Everything logged inside `body`, so the levels asserted are the ones
    /// `report` really emits rather than the ones its inputs imply.
    fn capture(body: impl FnOnce()) -> Vec<(Level, String)> {
        use tracing_subscriber::layer::SubscriberExt;
        let captured = Captured::default();
        let subscriber = tracing_subscriber::registry().with(captured.clone());
        tracing::subscriber::with_default(subscriber, body);
        let logs = captured.0.lock().expect("log capture poisoned").clone();
        logs
    }

    fn levels(logs: &[(Level, String)]) -> Vec<Level> {
        logs.iter().map(|(level, _)| *level).collect()
    }

    /// A stream carrying video that never flags a random access point.
    fn no_keyframe_packets() -> Vec<[u8; TS_PACKET_SIZE]> {
        let mut packets = vec![pat(PMT_PID), pmt(H264, VIDEO_PID)];
        packets.extend((1..20).map(|i| pes_packet(VIDEO_PID, i * 3_000)));
        packets
    }

    fn no_keyframe_failure() -> StreamFailure {
        failed(
            watched_for(segment_stats(&no_keyframe_packets()), 58),
            RunEnd::SegmentTimeout,
        )
    }

    #[test]
    fn missing_keyframe_flag_is_reported_as_an_observation() {
        let stats = segment_stats(&no_keyframe_packets());
        assert_eq!(stats.counts.video_pid, Some(VIDEO_PID));
        assert_eq!(stats.counts.pmt_pid, Some(PMT_PID));
        assert_eq!(stats.counts.pmt_packets, 1);
        assert_eq!(stats.counts.video_packets, 19);
        assert_eq!(stats.counts.keyframes, 0);
        assert_eq!(stats.counts.segments, 0);

        let failure = no_keyframe_failure();
        assert_eq!(failure.fault(), Some(StreamFault::NoKeyframeFlag));
        let summary = failure.summary();
        assert!(summary.contains("random_access_indicator"), "{summary}");
        assert!(summary.contains("in 58s"), "{summary}");
        // Both explanations are offered; neither is asserted as the cause.
        assert!(
            summary.contains("Either this stream never sets it"),
            "{summary}"
        );
        assert!(summary.contains("I-frame interval is longer"), "{summary}");
    }

    #[test]
    fn a_short_look_at_video_yields_no_keyframe_verdict() {
        // The PMT arrives just before the watchdog fires: the same counters as
        // above, but they say nothing about the stream yet.
        let stats = watched_for(segment_stats(&no_keyframe_packets()), 1);
        let failure = failed(stats, RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::WindowTooShort));
        let summary = failure.summary();
        assert!(summary.contains("only 1s before"), "{summary}");
        assert!(summary.contains("too short a look"), "{summary}");
        assert!(!summary.contains("random_access_indicator"), "{summary}");

        // One second under the threshold is still no verdict; one over is.
        let counts = segment_stats(&no_keyframe_packets()).counts;
        for (secs, expected) in [
            (MIN_KEYFRAME_WINDOW_SECS - 1, StreamFault::WindowTooShort),
            (MIN_KEYFRAME_WINDOW_SECS, StreamFault::NoKeyframeFlag),
        ] {
            let stats = watched_for(counted(counts), secs);
            assert_eq!(
                failed(stats, RunEnd::SegmentTimeout).fault(),
                Some(expected),
                "{secs}s"
            );
        }
    }

    #[test]
    fn one_keyframe_in_the_window_reports_only_what_arrived() {
        // A camera whose second keyframe is not due before the watchdog fires.
        // Only the first one was seen, so only that may be said.
        let stats = StreamStats {
            counts: StreamCounts {
                bytes: 900_000,
                ts_packets: 4_000,
                pmt_pid: Some(PMT_PID),
                pmt_packets: 600,
                video_pid: Some(VIDEO_PID),
                video_packets: 3_800,
                keyframes: 1,
                ..StreamCounts::default()
            },
            run_secs: 62,
            video_secs: 60,
            first_keyframe_secs: Some(21),
            last_keyframe_secs: Some(21),
        };
        let failure = failed(stats, RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::OneKeyframeOnly));
        let summary = failure.summary();
        assert!(
            summary.contains("1 keyframe on video PID 256 21s ago"),
            "{summary}"
        );
        assert!(
            summary.contains("in the 60s of video observed"),
            "{summary}"
        );
        // No claim about an interval that was never observed.
        assert!(!summary.contains("interval is longer"), "{summary}");
    }

    #[test]
    fn audio_random_access_points_are_not_keyframes() {
        // ffmpeg flags every audio packet as a random access point. Counting
        // those would turn a stream with no video keyframe into a stream that
        // looks like it has plenty.
        let mut packets = vec![pat(PMT_PID), pmt(H264, VIDEO_PID)];
        for i in 1..20 {
            packets.push(pes_packet(VIDEO_PID, i * 3_000));
            packets.push(keyframe_packet(AUDIO_PID, i * 3_000, AUDIO_STREAM_ID));
        }
        let stats = watched_for(segment_stats(&packets), 58);

        assert_eq!(stats.counts.keyframes, 0);
        assert_eq!(stats.counts.segments, 0);
        assert_eq!(
            failed(stats, RunEnd::SegmentTimeout).fault(),
            Some(StreamFault::NoKeyframeFlag)
        );
    }

    #[test]
    fn a_late_pid_change_is_reported_as_camon_latching_the_first_pmt() {
        // The first PMT wins for the life of the connection, so video on the
        // PID a later PMT announces is never picked up.
        let mut packets = vec![pat(PMT_PID), pmt(H264, VIDEO_PID), pmt(H264, OTHER_PID)];
        packets.extend((1..10).map(|i| pes_packet(OTHER_PID, i * 3_000)));
        let stats = watched_for(segment_stats(&packets), 58);

        assert_eq!(stats.counts.video_pid, Some(VIDEO_PID));
        assert_eq!(stats.counts.video_packets, 0);
        let failure = failed(stats, RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoVideoPackets));
        let summary = failure.summary();
        assert!(summary.contains("first PMT it parsed"), "{summary}");
        assert!(summary.contains("a later PMT announced"), "{summary}");
    }

    #[test]
    fn non_h264_program_map_is_named_as_such() {
        let packets = vec![pat(PMT_PID), pmt(HEVC, VIDEO_PID), pes_packet(VIDEO_PID, 0)];
        let stats = watched_for(segment_stats(&packets), 58);

        assert_eq!(stats.counts.video_pid, None);
        assert_eq!(stats.counts.pmt_packets, 1);
        let failure = failed(stats, RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoVideoStream));
        let summary = failure.summary();
        assert!(summary.contains("H.265"), "{summary}");
        // The count is of parsed PMT packets, which are repeats of one table.
        assert!(summary.contains("parsed the PMT 1 times"), "{summary}");
    }

    #[test]
    fn structural_faults_hold_however_the_run_ended() {
        // What the stream is missing does not depend on which watchdog fired,
        // so only the keyframe verdicts may be withheld from a short run.
        let no_pat = counted(StreamCounts {
            bytes: 4096,
            ts_packets: 20,
            ..StreamCounts::default()
        });
        let no_video_stream = counted(StreamCounts {
            bytes: 4096,
            ts_packets: 20,
            pmt_pid: Some(PMT_PID),
            pmt_packets: 3,
            ..StreamCounts::default()
        });
        let no_video_packets = counted(StreamCounts {
            bytes: 4096,
            ts_packets: 20,
            pmt_pid: Some(PMT_PID),
            pmt_packets: 3,
            video_pid: Some(VIDEO_PID),
            ..StreamCounts::default()
        });
        for end in [RunEnd::SegmentTimeout, RunEnd::DataTimeout, RunEnd::Eof] {
            assert_eq!(
                failed(no_pat, end).fault(),
                Some(StreamFault::NoProgramMap),
                "{end:?}"
            );
            assert_eq!(
                failed(no_video_stream, end).fault(),
                Some(StreamFault::NoVideoStream),
                "{end:?}"
            );
            assert_eq!(
                failed(no_video_packets, end).fault(),
                Some(StreamFault::NoVideoPackets),
                "{end:?}"
            );
        }
        // A PAT that never named a PMT PID is not blamed on the PMT.
        assert!(failed(no_pat, RunEnd::SegmentTimeout)
            .summary()
            .contains("no PAT naming a PMT PID"));
        assert!(!failed(no_video_packets, RunEnd::SegmentTimeout)
            .summary()
            .contains("no readable PMT"));
    }

    /// The GOP being filled when the connection drops is worth up to a second
    /// of footage, and nothing else will ever finalize it.
    #[test]
    fn the_open_gop_is_flushed_when_the_stream_ends() {
        let mut segmenter = segmenter();
        for packet in [pat(PMT_PID), pmt(H264, VIDEO_PID)] {
            segmenter.process(&packet);
        }
        segmenter.process(&keyframe_packet(VIDEO_PID, 0, VIDEO_STREAM_ID));
        segmenter.process(&pes_packet(VIDEO_PID, 3_000));
        segmenter.process(&keyframe_packet(VIDEO_PID, 90_000, VIDEO_STREAM_ID));
        for pts in [93_000, 96_000] {
            segmenter.process(&pes_packet(VIDEO_PID, pts));
        }
        // Only the closed GOP is in the buffer; nothing else will ever close
        // the one still being filled.
        assert_eq!(segmenter.buffer.read_recover().segment_count(), 1);

        segmenter.flush_end_of_stream();

        let buffer = segmenter.buffer.read_recover();
        assert_eq!(buffer.segment_count(), 2);
        let segment = buffer.segments().back().unwrap();
        // PAT and PMT are prepended, so the flushed segment opens on its
        // keyframe and stands alone like every other one.
        assert_eq!(crate::mpegts::keyframe_count(&segment.data), 1);
        // Media PTS, not the wall clock: the flush closes the segment on the
        // last frame that arrived, not on the moment the connection dropped.
        assert_eq!(segment.duration_ns, 6_000 * 1_000_000_000 / 90_000);
    }

    /// A segment cut before its second frame started holds a fragment of a
    /// keyframe and decodes to nothing. Publishing it would put a broken
    /// segment in the buffer for the sake of no picture at all.
    #[test]
    fn a_keyframe_fragment_is_dropped_rather_than_flushed() {
        let mut segmenter = segmenter();
        for packet in [pat(PMT_PID), pmt(H264, VIDEO_PID)] {
            segmenter.process(&packet);
        }
        segmenter.process(&keyframe_packet(VIDEO_PID, 0, VIDEO_STREAM_ID));
        segmenter.process(&pes_packet(VIDEO_PID, 3_000));
        segmenter.process(&keyframe_packet(VIDEO_PID, 90_000, VIDEO_STREAM_ID));

        segmenter.flush_end_of_stream();

        assert_eq!(segmenter.buffer.read_recover().segment_count(), 1);
    }

    #[test]
    fn keyframes_produce_segments_and_no_fault() {
        let mut packets = vec![pat(PMT_PID), pmt(H264, VIDEO_PID)];
        packets.push(keyframe_packet(VIDEO_PID, 0, VIDEO_STREAM_ID));
        packets.extend((1..5).map(|i| pes_packet(VIDEO_PID, i * 3_000)));
        packets.push(keyframe_packet(VIDEO_PID, 90_000, VIDEO_STREAM_ID));
        let stats = watched_for(segment_stats(&packets), 58);

        assert_eq!(stats.counts.keyframes, 2);
        assert_eq!(stats.counts.segments, 1);
        // A stream that did record is a plain reconnect, nothing to diagnose.
        for end in [RunEnd::SegmentTimeout, RunEnd::DataTimeout, RunEnd::Eof] {
            assert_eq!(failed(stats, end).fault(), None, "{end:?}");
        }
    }

    #[test]
    fn resynchronized_noise_is_not_reported_as_transport_stream() {
        // The scan resynchronizes on any sync byte, so noise with one 0x47 in
        // it still yields a "TS packet". The verdict has to weigh that.
        let mut data = vec![0u8; 1000];
        data[500] = crate::mpegts::SYNC_BYTE;
        let stats = segment_bytes(&data);

        assert_eq!(stats.counts.ts_packets, 1);
        assert_eq!(stats.counts.skipped_bytes, 812);
        let failure = failed(stats, RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NotTransportStream));
        let summary = failure.summary();
        assert!(summary.contains("812 of 1000 bytes"), "{summary}");
        assert!(
            summary.contains("recovered by resynchronizing"),
            "{summary}"
        );
    }

    #[test]
    fn faults_separate_no_bytes_from_bytes_without_video() {
        for end in [RunEnd::DataTimeout, RunEnd::Eof, RunEnd::SegmentTimeout] {
            let failure = failed(counted(StreamCounts::default()), end);
            assert_eq!(failure.fault(), Some(StreamFault::NoData), "{end:?}");
            // No cause is assigned to a camera camon never heard from.
            let summary = failure.summary();
            assert!(summary.contains("upstream of it"), "{summary}");
            assert!(summary.contains("RUST_LOG=debug"), "{summary}");
        }

        let not_ts = counted(StreamCounts {
            bytes: 4096,
            ..StreamCounts::default()
        });
        let failure = failed(not_ts, RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NotTransportStream));
        // A torn short read looks the same, and the message says so.
        assert!(failure.summary().contains("ended inside the first packet"));

        let flowing = StreamCounts {
            bytes: 900_000,
            ts_packets: 4_000,
            pmt_pid: Some(PMT_PID),
            pmt_packets: 600,
            video_pid: Some(VIDEO_PID),
            video_packets: 3_800,
            ..StreamCounts::default()
        };
        // A run cut short says nothing about keyframes, however long the video
        // PID had been known by then.
        for end in [RunEnd::DataTimeout, RunEnd::Eof] {
            let failure = failed(watched_for(counted(flowing), 300), end);
            assert_eq!(failure.fault(), Some(StreamFault::EndedEarly), "{end:?}");
            assert!(!failure.summary().contains("random_access_indicator"));
        }
        // A stall is not phrased as ffmpeg having stopped: it may still run.
        assert!(
            failed(watched_for(counted(flowing), 300), RunEnd::DataTimeout)
                .summary()
                .contains("may still have been running")
        );
    }

    #[test]
    fn report_escalates_then_goes_quiet() {
        let mut tracker = NoRecordingTracker::default();
        let failure = no_keyframe_failure();
        let logs = capture(|| {
            for _ in 0..12 {
                tracker.report("yard", &failure);
            }
        });

        assert_eq!(
            levels(&logs),
            vec![
                Level::WARN,  // streak 1
                Level::WARN,  // 2
                Level::DEBUG, // 3
                Level::ERROR, // 4
                Level::DEBUG, // 5
                Level::DEBUG, // 6
                Level::DEBUG, // 7
                Level::ERROR, // 8
                Level::DEBUG, // 9
                Level::DEBUG, // 10
                Level::DEBUG, // 11
                Level::DEBUG, // 12
            ]
        );
        assert!(logs[0].1.starts_with("recorded nothing this connection"));
        assert!(logs[3]
            .1
            .starts_with("recorded nothing in 4 connection attempts in a row"));
        assert!(logs[7]
            .1
            .starts_with("recorded nothing in 8 connection attempts in a row"));
        // Every line carries the diagnosis, whatever its level.
        assert!(logs
            .iter()
            .all(|(_, line)| line.contains("random_access_indicator")));
        // Reporting advances the streak it reports on: a report that cleared it
        // would restart at one here and never escalate at all.
        assert_eq!(tracker.classify(&failure), Report::Quiet);
        assert_eq!(tracker.fault.count(), 13);
    }

    #[test]
    fn reports_never_fall_more_than_the_cap_apart() {
        let mut tracker = NoRecordingTracker::default();
        let failure = no_keyframe_failure();
        let mut milestones = Vec::new();
        for _ in 0..400 {
            match tracker.classify(&failure) {
                Report::Connection(streak) | Report::Camera(streak) => milestones.push(streak),
                _ => {}
            }
        }
        assert_eq!(&milestones[..7], &[1, 2, 4, 8, 16, 32, 64]);
        // Doubling would next report at 128, then 256: a camera dead for half a
        // day would go silent for hours at a time.
        assert!(
            milestones
                .windows(2)
                .all(|pair| pair[1] - pair[0] <= MAX_REPORT_GAP),
            "{milestones:?}"
        );
        assert!(milestones.last().unwrap() > &300, "{milestones:?}");
    }

    #[test]
    fn only_a_run_that_recorded_clears_the_streak() {
        let mut tracker = NoRecordingTracker::default();
        let broken = no_keyframe_failure();
        assert_eq!(tracker.classify(&broken), Report::Connection(1));
        assert_eq!(tracker.classify(&broken), Report::Connection(2));
        assert_eq!(tracker.classify(&broken), Report::Quiet);
        // An unrelated error or a panic never reaches the tracker, so the
        // fourth run that recorded nothing still escalates on schedule.
        assert_eq!(tracker.classify(&broken), Report::Camera(4));

        let recorded = failed(
            StreamStats {
                counts: StreamCounts {
                    segments: 5,
                    ..StreamCounts::default()
                },
                run_secs: 65,
                ..StreamStats::default()
            },
            RunEnd::SegmentTimeout,
        );
        let logs = capture(|| tracker.report("yard", &recorded));
        assert_eq!(logs[0].0, Level::WARN);
        assert!(
            logs[0].1.contains("after 5 segments in 65s"),
            "{:?}",
            logs[0]
        );
        assert_eq!(tracker.classify(&broken), Report::Connection(1));
    }

    #[test]
    fn a_camera_that_wedges_after_one_segment_stops_repeating() {
        let mut tracker = NoRecordingTracker::default();
        let wedged = failed(
            StreamStats {
                counts: StreamCounts {
                    segments: 1,
                    ..StreamCounts::default()
                },
                run_secs: 65,
                ..StreamStats::default()
            },
            RunEnd::SegmentTimeout,
        );
        let logs = capture(|| {
            for _ in 0..6 {
                tracker.report("yard", &wedged);
            }
        });
        assert_eq!(
            levels(&logs),
            vec![
                Level::WARN,
                Level::WARN,
                Level::DEBUG,
                Level::WARN,
                Level::DEBUG,
                Level::DEBUG,
            ]
        );

        // A run that kept going for ten minutes was a working stream, so its
        // stop is worth a line again.
        let settled = failed(
            StreamStats {
                run_secs: SETTLED_RUN_SECS,
                ..wedged.stats
            },
            RunEnd::DataTimeout,
        );
        let logs = capture(|| {
            tracker.report("yard", &settled);
            tracker.report("yard", &settled);
        });
        assert_eq!(levels(&logs), vec![Level::WARN, Level::WARN]);
    }
}
