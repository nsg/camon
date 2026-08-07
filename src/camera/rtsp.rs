use std::io::{BufRead, BufReader, Read};
use std::os::unix::io::{AsRawFd, RawFd};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::buffer::{wall_clock_ns, GopSegment, HotBuffer, MAX_SEGMENT_SPAN_NS};
use crate::config::CameraConfig;
use crate::locks::LockExt;
use crate::retry::Streak;

/// Reconnect if no bytes are read from ffmpeg for this long.
const DATA_TIMEOUT_SECS: u64 = 30;
/// Reconnect if bytes are flowing but no segment (keyframe) is produced for this
/// long. This is also what makes a segment's span bounded, so raising it past
/// [`MAX_SEGMENT_SPAN_NS`] would leave every real segment looking implausible —
/// both duration instruments refused, every duration zero. The assertion below
/// is what makes that mistake fail to compile instead of shipping.
const NO_SEGMENT_TIMEOUT_SECS: u64 = 60;
const _: () = assert!(
    MAX_SEGMENT_SPAN_NS > NO_SEGMENT_TIMEOUT_SECS * 1_000_000_000,
    "the span bound must sit above the no-segment watchdog, or real GOPs lose their durations"
);
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

/// The GOP being filled, paired with the monotonic instant the segmenter
/// opened it at.
///
/// Two clocks, deliberately, for two different jobs: [`GopSegment::start_pts`]
/// is the wall clock stamp the event's identity on disk is built from, while
/// the duration is measured from this [`Instant`], which no clock adjustment
/// can move. Held together so a segment's span can only ever be measured
/// against its own anchor.
struct OpenSegment {
    segment: GopSegment,
    opened_at: Instant,
}

/// Segments raw MPEG-TS stream based on keyframe detection
/// Stores raw MPEG-TS packets directly - no re-muxing needed
struct MpegTsSegmenter {
    camera_id: String,
    buffer: Arc<RwLock<HotBuffer>>,
    current_segment: Option<OpenSegment>,
    /// Incremental byte buffer for the in-progress segment; wrapped in an Arc
    /// once at finalize time so readers share it without copying.
    current_data: Vec<u8>,
    video_pid: Option<u16>,
    pat_packet: Option<[u8; 188]>,
    pmt_packet: Option<[u8; 188]>,
    pmt_pid: Option<u16>,
    partial_packet: Vec<u8>,
    current_media_pts: Option<u64>,
    prev_media_pts: Option<u64>,
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
            pat_packet: None,
            pmt_packet: None,
            pmt_pid: None,
            partial_packet: Vec::with_capacity(188),
            current_media_pts: None,
            prev_media_pts: None,
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
            match poll_readable(fd, 500) {
                Readiness::Readable => {}
                Readiness::Timeout => continue,
                // The descriptor hung up, errored, or is no longer a
                // descriptor at all. Reported as the end of the stream, like
                // the zero-length read below, because that is what it is —
                // and it routes to the same full diagnosis. (The old code
                // reached that diagnosis too, through the read the error bits
                // provoked; what classification buys is the honest path there,
                // not an escape from a spin the read already broke.)
                Readiness::Ended => return Err(self.failure(RunEnd::Eof)),
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
    ///
    /// "A second frame started" is exactly what the open segment's frame count
    /// reads, so it is read from there rather than tallied a second time
    /// alongside it.
    fn flush_end_of_stream(&mut self) {
        let frames_started = self
            .current_segment
            .as_ref()
            .map_or(0, |open| open.segment.frame_count);
        if frames_started < 2 {
            return;
        }
        self.finalize_segment(Instant::now());
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
            // One instant for both ends, so the closing segment's span and the
            // opening one's anchor meet exactly instead of leaving a gap
            // between them that the durations would never account for. The
            // wall clock is read after the close, not with it: the stamp names
            // when this GOP begins, and closing the last one takes long enough
            // (a shrink, a lock, a push) to be worth not backdating it by.
            let now = Instant::now();
            self.finalize_segment(now);
            self.start_segment(wall_clock_ns(), now);
        }

        // Append packet to current segment
        if let Some(ref mut open) = self.current_segment {
            self.current_data.extend_from_slice(packet);
            // Frames, not packets. One video frame is one PES packet spread
            // over dozens of TS packets, and only its first carries the
            // payload_unit_start_indicator — counting every packet on the
            // video PID inflates the count by however many packets a frame
            // happens to take.
            if Some(pid) == self.video_pid && pusi {
                open.segment.frame_count += 1;
            }
        }
    }

    fn start_segment(&mut self, pts_ns: u64, opened_at: Instant) {
        let segment = GopSegment::new(pts_ns);

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

        self.current_segment = Some(OpenSegment { segment, opened_at });
    }

    /// Close the GOP being filled, as of `at` — the same instant the next one
    /// is opened at, or [`Instant::now`] for the end-of-stream flush.
    fn finalize_segment(&mut self, at: Instant) {
        if let Some(OpenSegment {
            mut segment,
            opened_at,
        }) = self.current_segment.take()
        {
            segment.finalize_with_media_pts(
                at.saturating_duration_since(opened_at),
                self.current_media_pts,
                self.prev_media_pts,
            );
            self.prev_media_pts = self.current_media_pts;
            // Wrap the accumulated bytes once; readers share via Arc clone.
            // Drop the Vec's growth slack first — the segment lives in the
            // hot buffer for minutes, so excess capacity is held that long.
            //
            // What goes on filling is a buffer sized to the GOP just closed,
            // not an empty one: taking the Vec leaves capacity zero behind, so
            // every GOP regrew from nothing through some twenty reallocations,
            // each copying everything accumulated so far. The trade is one
            // allocation per GOP against those, at the cost of holding one
            // GOP's worth of empty bytes per camera between GOPs — memory the
            // buffer reaches anyway a moment later while filling. The estimate
            // is only as good as the last GOP: the first of a connection still
            // grows from zero, and a GOP bigger than its predecessor regrows
            // over the gap. Nothing is pinned at a peak, though — an outsized
            // estimate is given back at the very next finalize.
            let next_capacity = self.current_data.len();
            self.current_data.shrink_to_fit();
            segment.data = Arc::new(std::mem::replace(
                &mut self.current_data,
                Vec::with_capacity(next_capacity),
            ));
            // At least one frame started inside it; a segment holding only the
            // tail of a frame begun in the previous one decodes to nothing.
            // This gate rests on RAI implying PUSI (a keyframe flag on a
            // packet that also starts its frame) — true of ffmpeg's mpegts
            // muxer, whose output is all camon ever reads. A stream that
            // flagged RAI on a non-start packet would count zero frames and
            // lose every segment here, so if that assumption ever breaks,
            // this is the line to suspect.
            if segment.frame_count > 0 {
                self.buffer.write_recover().push(segment);
                self.last_segment_at = Instant::now();
                self.counts.segments += 1;
            }
        }
    }

    /// Take the PMT PID from the first entry the PAT lists.
    ///
    /// The first entry, not the first program: an entry with program_number 0
    /// names the network information table, and its PID would be taken for a
    /// program map here. ffmpeg's muxer emits no NIT and one program, so the
    /// first entry is the program — this reads that stream, not the standard.
    ///
    /// The entry read here is only there if the section says it is: a PAT
    /// listing no programs at all ends its five-byte header with the CRC_32,
    /// and reading the entry regardless would take two checksum bytes for a
    /// PID. Bounding by `section_length` refuses that section instead, and the
    /// run then reports the PAT as naming no PMT PID — which it does not.
    fn parse_pat(&mut self, packet: &[u8]) {
        let start = match table_section_start(packet) {
            Some(s) => s,
            None => return,
        };

        // The first program entry ends at `start + 12`; anything less and the
        // bytes there belong to the CRC or to another packet entirely.
        let Some(contents_end) = section_contents_end(packet, start) else {
            return;
        };
        if start + 12 > contents_end {
            return;
        }

        let pmt_pid = ((packet[start + 10] as u16 & 0x1F) << 8) | packet[start + 11] as u16;
        if pmt_pid != 0 && pmt_pid != 0x1FFF && self.pmt_pid.is_none() {
            self.pmt_pid = Some(pmt_pid);
            self.counts.pmt_pid = Some(pmt_pid);
            tracing::debug!(camera = %self.camera_id, pmt_pid, "detected PMT PID");
        }
    }

    /// Latch the video and audio elementary PIDs from the program map.
    ///
    /// The elementary-stream loop stops where the section's own
    /// `section_length` says the streams stop, four bytes short of the end so
    /// the CRC_32 is never walked into. Running to the end of the packet
    /// instead reads that checksum as a stream entry, and about one PMT layout
    /// in 256 has 0x1B — H.264 — sitting in the first CRC byte: a PID no
    /// packet ever carries, latched for the life of the connection, after
    /// which the run blames the camera for sending no video on a PID the
    /// camera never announced.
    ///
    /// A section that cannot hold its own fixed header, or that claims more
    /// bytes than the packet carries, is refused whole rather than half-read,
    /// and is not counted among the PMTs that parsed. The run then reports no
    /// readable PMT on this PID, which is exactly what was observed.
    ///
    /// What the bound does not promise is that every PID latched here was
    /// announced: a section whose `program_info_length` lies while staying
    /// inside its own bounds leaves the loop misaligned, reading a stream type
    /// out of descriptor bytes. That takes a malformed section rather than the
    /// one-in-256 accident above, and it can only misread bytes the section
    /// itself carries — never the CRC, never the stuffing past it.
    fn parse_pmt(&mut self, packet: &[u8]) {
        let start = match table_section_start(packet) {
            Some(s) => s,
            None => return,
        };

        let Some(es_end) = section_contents_end(packet, start) else {
            return;
        };
        // The fixed header runs to `start + 12`, `program_info_length` last.
        if start + 12 > es_end {
            return;
        }
        self.counts.pmt_packets += 1;

        let program_info_len =
            ((packet[start + 10] as usize & 0x0F) << 8) | packet[start + 11] as usize;

        // Descriptor and stream lengths are the section's own claims: a length
        // reaching past `es_end` walks `pos` off the end of the loop rather
        // than off the end of the packet.
        let mut pos = start + 12 + program_info_len;
        while pos + 5 <= es_end {
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

            pos += 5 + es_info_len;
        }
    }
}

/// Where the contents of the PSI section beginning at `start` end: one past
/// the last byte a table may be read from.
///
/// `section_length` counts the bytes following it, the four-byte CRC_32
/// included, so the section spans up to `start + 3 + section_length` and its
/// contents stop four bytes short of that. Returning that boundary is what
/// keeps a checksum from being parsed as table data. `None` for a section too
/// short to hold even its CRC, or one claiming more bytes than the 188-byte
/// packet carries — nothing in either can be trusted to be what it looks like.
///
/// A section is only ever parsed out of the one packet that starts it. PSI
/// sections may in principle span packets, and one that claims more bytes than
/// its own packet holds is refused whole rather than reassembled: refusing
/// beats guessing, since a half-read table names PIDs that were never in it,
/// and the run then reports no readable PMT on that PID — which is literally
/// what happened. The transport camon reads is produced by its own ffmpeg
/// remuxing one or two elementary streams, whose program map is a single small
/// packet, so this refusal is not expected to fire on a working camera.
///
/// Callers must still check that the fields they read fall before the
/// boundary: this says where the section ends, not that it holds anything.
fn section_contents_end(packet: &[u8], start: usize) -> Option<usize> {
    let section_length = ((packet[start + 1] as usize & 0x0F) << 8) | packet[start + 2] as usize;
    let end = start + 3 + section_length;
    (section_length >= 4 && end <= 188).then_some(end - 4)
}

/// Compute the start of a PSI table section inside an MPEG-TS packet, past the
/// adaptation field and the pointer field. `None` if the packet begins no
/// section, or if the pointer lands outside it.
///
/// Only a packet flagging `payload_unit_start_indicator` begins a section, and
/// only such a packet carries the pointer field that says where. A
/// continuation packet is the middle of a section already in progress: its
/// payload starts wherever the previous packet's section left off, so reading
/// a table_id and a `section_length` from its first bytes invents a section
/// out of another one's descriptors. Refusing it is what keeps the parsers
/// from latching a PID that no table ever named.
fn table_section_start(packet: &[u8]) -> Option<usize> {
    if (packet[1] & 0x40) == 0 {
        return None;
    }

    let payload_offset = if (packet[3] & 0x20) != 0 {
        5 + packet[4] as usize
    } else {
        4
    };

    if payload_offset >= 188 {
        return None;
    }

    let start = payload_offset + 1 + packet[payload_offset] as usize;

    if start + 12 > 188 {
        return None;
    }

    Some(start)
}

/// What one poll of ffmpeg's stdout said about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readiness {
    /// There are bytes to read, or a read will at least tell us something.
    Readable,
    /// Nothing happened within the timeout. Poll again.
    Timeout,
    /// The descriptor will never carry data again.
    Ended,
}

/// Poll a file descriptor for readability with a timeout in milliseconds.
///
/// The kernel's return value alone does not say the fd is readable: `poll`
/// reports `POLLERR`, `POLLHUP` and `POLLNVAL` whether or not they were asked
/// for, and each of them makes the call return a positive count. Reading them
/// as readability sends the caller into a read that fails or comes back
/// empty, and the diagnosis then depends on which of those the kernel picked;
/// classifying here routes every dead-descriptor shape to the one stream-end
/// diagnosis instead.
fn poll_readable(fd: RawFd, timeout_ms: i32) -> Readiness {
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
    readiness(if ret < 0 {
        Err(std::io::Error::last_os_error().kind())
    } else if ret > 0 {
        Ok(pollfd.revents)
    } else {
        // A timeout leaves nothing behind to classify.
        Ok(0)
    })
}

/// Read one completed poll: the `revents` it filled in, or the error it failed
/// with. Split from the syscall so the decision can be tested without arranging
/// a descriptor in each of these states.
///
/// `POLLIN` wins over a hangup reported beside it, because a pipe whose writer
/// closed still hands over whatever it buffered before the close; reading it
/// out is what turns a hangup into the ordinary zero-length read the caller
/// already diagnoses. (Should that read itself fail — possible for descriptor
/// kinds that raise `POLLERR` with data queued, which a pipe read end does not
/// — it surfaces as a plain I/O error, as before this classification existed.)
/// Only when nothing is readable do the error bits end the stream.
///
/// An interrupted poll is a wait that ended early and nothing more — the
/// descriptor is untouched, so the caller simply waits again. That is a spin
/// for as long as the signals keep arriving, which the data watchdog bounds at
/// [`DATA_TIMEOUT_SECS`]; any other failure is treated as a poll that cannot
/// be made to work on this fd and ends the run. (Strictly, `poll` can also
/// fail transiently — `ENOMEM` under pressure — where the updater's taxonomy
/// would retry; here the run's ending IS the retry, through the ordinary
/// reconnect backoff, which beats spinning on a wait that just failed.)
fn readiness(poll: Result<libc::c_short, std::io::ErrorKind>) -> Readiness {
    let revents = match poll {
        Ok(revents) => revents,
        Err(std::io::ErrorKind::Interrupted) => return Readiness::Timeout,
        Err(_) => return Readiness::Ended,
    };
    if revents & libc::POLLIN != 0 {
        Readiness::Readable
    } else if revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
        Readiness::Ended
    } else {
        Readiness::Timeout
    }
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

    /// A video packet continuing the frame already in progress: the same PID,
    /// no payload_unit_start_indicator. Every frame is one of the builders
    /// above followed by dozens of these.
    fn continuation(pid: u16) -> [u8; TS_PACKET_SIZE] {
        let mut p = pes_packet(pid, 0);
        p[1] &= !0x40;
        p
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
    fn a_program_map_names_the_streams_its_section_actually_lists() {
        let stats = segment_stats(&[pat(PMT_PID), pmt(H264, VIDEO_PID)]);

        assert_eq!(stats.counts.pmt_pid, Some(PMT_PID));
        assert_eq!(stats.counts.pmt_packets, 1);
        assert_eq!(stats.counts.video_pid, Some(VIDEO_PID));
    }

    #[test]
    fn a_crc_shaped_like_a_video_stream_entry_latches_no_video_pid() {
        // The four bytes past the last elementary stream are the section's
        // CRC_32, and roughly one PMT layout in 256 opens it with 0x1B. Read
        // as a stream entry it names a PID nothing is ever sent on.
        let mut packet = pmt(0x0F, AUDIO_PID);
        packet[22..26].copy_from_slice(&[H264, 0xE0 | (OTHER_PID >> 8) as u8, OTHER_PID as u8, 0]);
        let stats = segment_stats(&[pat(PMT_PID), packet]);

        assert_eq!(stats.counts.video_pid, None);
        // The section itself is well formed: it lists audio and no video.
        assert_eq!(stats.counts.pmt_packets, 1);
        let failure = failed(watched_for(stats, 58), RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoVideoStream));
    }

    #[test]
    fn the_four_bytes_a_section_ends_on_are_never_a_stream_entry() {
        // A section whose stream lengths leave a byte of slack before the
        // CRC_32 puts a five-byte read within reach of the section's end. Only
        // stopping four bytes short of it keeps the checksum out of the loop.
        let mut packet = pmt(0x0F, AUDIO_PID);
        packet[7] = 0x13; // section_length: one byte past the last stream
        packet[22..27].copy_from_slice(&[
            H264,
            0xE0 | (OTHER_PID >> 8) as u8,
            OTHER_PID as u8,
            0xF0,
            0x00,
        ]);
        let stats = segment_stats(&[pat(PMT_PID), packet]);

        assert_eq!(stats.counts.video_pid, None);
    }

    #[test]
    fn a_program_map_claiming_more_bytes_than_the_packet_holds_is_refused() {
        let mut packet = pmt(H264, VIDEO_PID);
        packet[6] = 0xBF;
        packet[7] = 0xFF; // section_length 4095, twenty times the packet
        let stats = segment_stats(&[pat(PMT_PID), packet]);

        assert_eq!(stats.counts.video_pid, None);
        // Refused, not parsed — so the run says the PMT was unreadable rather
        // than that the camera listed no video in it.
        assert_eq!(stats.counts.pmt_packets, 0);
        let failure = failed(watched_for(stats, 58), RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoProgramMap));
        let summary = failure.summary();
        assert!(summary.contains("no readable PMT on PID 4096"), "{summary}");
    }

    #[test]
    fn a_program_map_that_would_span_packets_is_refused_rather_than_reassembled() {
        // A section too long for its own packet continues in the next one.
        // camon does not reassemble; it says so rather than reading half a
        // table, and the fault names the PID it could not read.
        let mut packet = pmt(H264, VIDEO_PID);
        packet[7] = 0xC8; // section_length 200, past the 188-byte packet
        let stats = segment_stats(&[pat(PMT_PID), packet]);

        assert_eq!(stats.counts.video_pid, None);
        assert_eq!(stats.counts.pmt_packets, 0);
        let failure = failed(watched_for(stats, 58), RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoProgramMap));
        assert!(
            failure.summary().contains("no readable PMT on PID 4096"),
            "{}",
            failure.summary()
        );
    }

    #[test]
    fn a_continuation_packet_is_never_read_as_a_section_of_its_own() {
        // The rest of a spanning section carries no pointer field and no
        // table header: its payload is descriptor bytes that happen to sit
        // where a section's would. Laid out as a plausible PMT naming
        // OTHER_PID, it must name nothing at all.
        // The bytes are laid out to parse as such a section whether the
        // payload is taken to start at the payload offset or one pointer
        // field past it, so no accident of where the section is looked for
        // can pass for the flag being honoured.
        let mut continuation = [0xFFu8; TS_PACKET_SIZE];
        continuation[0] = crate::mpegts::SYNC_BYTE;
        continuation[1] = (PMT_PID >> 8) as u8 & 0x1F; // no payload_unit_start
        continuation[2] = PMT_PID as u8;
        continuation[3] = 0x10;
        continuation[4] = 0x00; // table_id, or a pointer field of zero
        continuation[5] = 0xB0;
        continuation[6] = 0x30;
        continuation[7] = 0x30; // section_length 48, read from either pair
        continuation[14] = 0xF0;
        continuation[15] = 0x10;
        continuation[16] = 0x0F; // program_info_length, either pair
        continuation[32] = H264; // where both readings put a stream entry
        continuation[33] = 0xE0 | (OTHER_PID >> 8) as u8;
        continuation[34] = OTHER_PID as u8;
        continuation[35] = 0xF0;
        continuation[36] = 0x00;

        let stats = segment_stats(&[pat(PMT_PID), continuation]);
        assert_eq!(stats.counts.video_pid, None);
        assert_eq!(stats.counts.pmt_packets, 0);
        let failure = failed(watched_for(stats, 58), RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoProgramMap));
        assert!(
            failure.summary().contains("no readable PMT on PID 4096"),
            "{}",
            failure.summary()
        );

        // The same bytes on the PAT's PID, where either reading takes them
        // for a program entry and names a PMT PID that was never announced.
        continuation[1] = 0x00;
        continuation[2] = 0x00;
        assert_eq!(segment_stats(&[continuation]).counts.pmt_pid, None);
    }

    #[test]
    fn the_shortest_well_formed_program_map_is_accepted_though_it_lists_nothing() {
        // Nine bytes of fixed header and four of CRC, listing nothing: the
        // exact boundary between a section camon reads and one it refuses. It
        // parsed, and it named no video — a different fault from an unreadable
        // one, and the 0x1B the CRC opens with is still not a stream.
        let mut packet = pmt(H264, VIDEO_PID);
        packet[7] = 0x0D;
        let stats = segment_stats(&[pat(PMT_PID), packet]);

        assert_eq!(stats.counts.pmt_packets, 1);
        assert_eq!(stats.counts.video_pid, None);
        let failure = failed(watched_for(stats, 58), RunEnd::SegmentTimeout);
        assert_eq!(failure.fault(), Some(StreamFault::NoVideoStream));
    }

    #[test]
    fn a_program_map_section_shorter_than_its_own_header_is_refused() {
        // Nine bytes of fixed header and four of CRC: below thirteen the
        // section cannot reach the elementary streams at all.
        for section_length in [0x00, 0x03, 0x04, 0x0C] {
            let mut packet = pmt(H264, VIDEO_PID);
            packet[7] = section_length;
            let stats = segment_stats(&[pat(PMT_PID), packet]);

            assert_eq!(
                stats.counts.video_pid, None,
                "section_length {section_length}"
            );
            assert_eq!(
                stats.counts.pmt_packets, 0,
                "section_length {section_length}"
            );
        }
    }

    #[test]
    fn a_program_association_table_listing_no_programs_names_no_pmt_pid() {
        // Five bytes of header and four of CRC, no program entry: the PID read
        // from one would be two bytes of the checksum.
        let mut packet = pat(PMT_PID);
        packet[7] = 0x09;
        let stats = segment_stats(&[packet, pmt(H264, VIDEO_PID)]);

        assert_eq!(stats.counts.pmt_pid, None);
        assert_eq!(stats.counts.video_pid, None);
    }

    #[test]
    fn malformed_table_packets_are_refused_without_panicking() {
        let mut adaptation_only = pmt(H264, VIDEO_PID);
        adaptation_only[3] = 0x20; // adaptation field, no payload at all
        adaptation_only[4] = 183;
        let mut pointer_past_end = pmt(H264, VIDEO_PID);
        pointer_past_end[4] = 0xFF; // pointer field past the last byte
        let mut every_bit_set = [0xFFu8; TS_PACKET_SIZE];
        every_bit_set[0] = crate::mpegts::SYNC_BYTE;
        every_bit_set[1] = 0x40 | ((PMT_PID >> 8) as u8 & 0x1F);
        every_bit_set[2] = PMT_PID as u8;

        for packet in [adaptation_only, pointer_past_end, every_bit_set] {
            let stats = segment_stats(&[pat(PMT_PID), packet]);
            assert_eq!(stats.counts.video_pid, None);
            assert_eq!(stats.counts.pmt_packets, 0);
        }

        // Every length a twelve-bit field can claim, against a payload whose
        // trailing bytes are all 0xFF: none of them may index past the packet.
        for length in 0..=0x0FFFu16 {
            let mut packet = pmt(H264, VIDEO_PID);
            packet[6] = 0xB0 | ((length >> 8) as u8 & 0x0F);
            packet[7] = length as u8;
            segment_stats(&[pat(PMT_PID), packet]);

            let mut packet = pmt(H264, VIDEO_PID);
            packet[15] = 0xF0 | ((length >> 8) as u8 & 0x0F);
            packet[16] = length as u8; // program_info_length
            segment_stats(&[pat(PMT_PID), packet]);

            let mut packet = pmt(H264, VIDEO_PID);
            packet[20] = 0xF0 | ((length >> 8) as u8 & 0x0F);
            packet[21] = length as u8; // ES_info_length
            segment_stats(&[pat(PMT_PID), packet]);
        }
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

    /// A GOP is measured from the monotonic instant the segmenter opened it,
    /// never from the wall clock stamps at its two ends. The two agree until
    /// the clock steps between them — which on a box with no battery-backed
    /// clock happens on every boot, the moment NTP lands.
    ///
    /// Aging the anchor stands in for a GOP that really did take two seconds;
    /// the wall clock is untouched, so a duration that followed it would be the
    /// microseconds this test actually takes.
    #[test]
    fn a_gop_is_measured_from_the_instant_it_was_opened_at() {
        let mut segmenter = segmenter();
        for packet in [pat(PMT_PID), pmt(H264, VIDEO_PID)] {
            segmenter.process(&packet);
        }
        segmenter.process(&keyframe_packet(VIDEO_PID, 0, VIDEO_STREAM_ID));
        // No predecessor to subtract a media PTS from, so this first segment is
        // the one the monotonic span has to carry. Checked, because the
        // monotonic clock starts at boot and subtracting from it is only
        // representable once the box has been up that long.
        let open = segmenter.current_segment.as_mut().unwrap();
        open.opened_at = open
            .opened_at
            .checked_sub(Duration::from_secs(2))
            .expect("host booted less than two seconds ago");
        segmenter.process(&pes_packet(VIDEO_PID, 3_000));
        segmenter.process(&keyframe_packet(VIDEO_PID, 90_000, VIDEO_STREAM_ID));

        let buffer = segmenter.buffer.read_recover();
        let first = buffer.segments().front().unwrap();
        assert!(
            first.duration_ns >= 2 * 1_000_000_000,
            "measured {} ns, not the two seconds it was open",
            first.duration_ns
        );
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

    /// A frame is one PES packet spread over many TS packets. Counting the
    /// packets instead reports a GOP of three frames as one of a dozen.
    #[test]
    fn a_frame_spread_over_many_packets_is_counted_once() {
        let mut segmenter = segmenter();
        for packet in [pat(PMT_PID), pmt(H264, VIDEO_PID)] {
            segmenter.process(&packet);
        }
        // Three frames of four packets each, the first of them the keyframe
        // the GOP opens on, with audio interleaved as a real stream has it.
        segmenter.process(&keyframe_packet(VIDEO_PID, 0, VIDEO_STREAM_ID));
        for frame in 1..3 {
            for _ in 0..3 {
                segmenter.process(&continuation(VIDEO_PID));
            }
            segmenter.process(&pes_packet(AUDIO_PID, frame * 3_000));
            segmenter.process(&pes_packet(VIDEO_PID, frame * 3_000));
        }
        for _ in 0..3 {
            segmenter.process(&continuation(VIDEO_PID));
        }
        segmenter.process(&keyframe_packet(VIDEO_PID, 90_000, VIDEO_STREAM_ID));

        let buffer = segmenter.buffer.read_recover();
        let segment = buffer.segments().front().unwrap();
        assert_eq!(segment.frame_count, 3);
        // The packets really are all in there: the count is a reading of the
        // stream, not of how much was stored.
        assert_eq!(segment.data.len() / TS_PACKET_SIZE, 16);
    }

    /// The buffer the next GOP is accumulated into keeps the capacity the last
    /// one needed, while the bytes handed to the hot buffer carry no slack at
    /// all — they are held there for minutes, the working buffer for a second.
    #[test]
    fn the_segment_buffer_keeps_its_capacity_for_the_next_gop() {
        let mut segmenter = segmenter();
        for packet in [pat(PMT_PID), pmt(H264, VIDEO_PID)] {
            segmenter.process(&packet);
        }
        segmenter.process(&keyframe_packet(VIDEO_PID, 0, VIDEO_STREAM_ID));
        for i in 1..40 {
            segmenter.process(&pes_packet(VIDEO_PID, i * 3_000));
        }
        segmenter.process(&keyframe_packet(VIDEO_PID, 90_000, VIDEO_STREAM_ID));

        let closed_len = {
            let buffer = segmenter.buffer.read_recover();
            let segment = buffer.segments().front().unwrap();
            assert_eq!(
                segment.data.capacity(),
                segment.data.len(),
                "growth slack shipped to the hot buffer"
            );
            segment.data.len()
        };
        assert!(closed_len > 7_000, "{closed_len} is too small to tell");
        assert!(
            segmenter.current_data.capacity() >= closed_len,
            "regrown from {} rather than reusing {closed_len}",
            segmenter.current_data.capacity()
        );
    }

    /// A descriptor that hangs up or errors is reported ready by `poll` — with
    /// an error bit, not `POLLIN`. Taking that for readability leaves the loop
    /// polling a dead descriptor as fast as the kernel answers.
    #[test]
    fn a_descriptor_in_error_ends_the_run_instead_of_being_polled_again() {
        use std::os::unix::io::FromRawFd;

        let mut fds = [0 as libc::c_int; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "pipe");
        // The write end of a pipe with no reader: a live descriptor that poll
        // flags POLLERR on and that can never be read from.
        let write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        drop(unsafe { std::fs::File::from_raw_fd(fds[0]) });

        let shutdown = std::sync::atomic::AtomicBool::new(false);
        let result = std::thread::scope(|scope| {
            // Bounds a run that spins instead of ending, so the mistake shows
            // up as a failed assertion rather than a hung test.
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(250));
                shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            segmenter().read_stream(write_end, &shutdown)
        });

        match result {
            Err(RtspError::NoRecording(failure)) => assert_eq!(failure.end, RunEnd::Eof),
            // Ok means the poll was read as a timeout and the loop spun until
            // the shutdown flag stopped it; an io error means it was read as
            // readable and the read was attempted on a write-only descriptor.
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn poll_readiness_separates_data_from_a_dead_descriptor() {
        // Bytes to read, whatever else is flagged beside them: a pipe whose
        // writer closed still hands over what it buffered first, and the
        // zero-length read that follows is the caller's own end-of-stream.
        for revents in [
            libc::POLLIN,
            libc::POLLIN | libc::POLLHUP,
            libc::POLLIN | libc::POLLERR,
        ] {
            assert_eq!(readiness(Ok(revents)), Readiness::Readable, "{revents:#x}");
        }
        // Nothing to read and the descriptor is finished.
        for revents in [libc::POLLERR, libc::POLLHUP, libc::POLLNVAL] {
            assert_eq!(readiness(Ok(revents)), Readiness::Ended, "{revents:#x}");
        }
        // Nothing happened, and a signal that ended the wait early is nothing
        // happening: the descriptor is untouched, so the caller waits again.
        assert_eq!(readiness(Ok(0)), Readiness::Timeout);
        assert_eq!(
            readiness(Err(std::io::ErrorKind::Interrupted)),
            Readiness::Timeout
        );
        // Any other failure is one that would repeat on every call.
        assert_eq!(
            readiness(Err(std::io::ErrorKind::InvalidInput)),
            Readiness::Ended
        );
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
