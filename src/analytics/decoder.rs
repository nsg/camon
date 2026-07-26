use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const ANALYSIS_WIDTH: u32 = 320;
const ANALYSIS_HEIGHT: u32 = 240;
const FRAME_SIZE: usize = (ANALYSIS_WIDTH * ANALYSIS_HEIGHT) as usize;
const FRAME_READ_TIMEOUT: Duration = Duration::from_millis(500);

/// How long a segment hand-off may stay blocked before the pipe is declared
/// wedged. The segment channel holds 16 whole GOP segments — roughly 16
/// seconds of video — and a healthy ffmpeg is never more than one segment
/// decode behind, so slots free up in milliseconds. A channel that stays full
/// for five seconds means the child stopped consuming stdin altogether
/// (descheduled in a swap storm, SIGSTOPped, hung) rather than merely running
/// behind, and nothing will drain it on its own.
const SEND_DEADLINE: Duration = Duration::from_secs(5);
/// Pause between `try_send` attempts while the segment channel is full. std's
/// `SyncSender` has no `send_timeout`, so a bounded hand-off has to poll.
const SEND_RETRY_INTERVAL: Duration = Duration::from_millis(50);

/// Outcome of handing one segment to a decoder's ffmpeg child.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SendOutcome {
    /// Queued for ffmpeg's stdin.
    Sent,
    /// The pipe is gone — the child exited and its writer thread stopped. Fails
    /// immediately; the caller's liveness check handles the respawn.
    Closed,
    /// The channel stayed full for [`SEND_DEADLINE`]: ffmpeg is alive but no
    /// longer reading stdin. Distinct from `Closed` because only this case
    /// needs the child killed to make progress possible again.
    Wedged,
}

/// Crop decoder output size when the frames feed the vision model: its native
/// input resolution, so a small object survives the crop.
pub const DETECTION_CROP_SIZE: (u32, u32) = (1920, 1080);
/// Crop decoder output size when the frames only become event thumbnails.
pub const THUMBNAIL_CROP_SIZE: (u32, u32) = (640, 360);

struct FfmpegPipe {
    segment_tx: Option<SyncSender<Vec<u8>>>,
    frame_rx: Receiver<Vec<u8>>,
    child: Option<Child>,
    _writer_handle: JoinHandle<()>,
    _reader_handle: JoinHandle<()>,
}

fn spawn_ffmpeg_pipe(
    args: &[&str],
    frame_size: usize,
    segment_channel_size: usize,
    frame_channel_size: usize,
) -> Result<FfmpegPipe, std::io::Error> {
    let mut child = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");

    let (segment_tx, segment_rx) = mpsc::sync_channel::<Vec<u8>>(segment_channel_size);
    let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(frame_channel_size);

    let writer_handle = thread::spawn(move || {
        let mut stdin = stdin;
        while let Ok(data) = segment_rx.recv() {
            if stdin.write_all(&data).is_err() {
                break;
            }
            if stdin.flush().is_err() {
                break;
            }
        }
    });

    let reader_handle = thread::spawn(move || {
        let mut stdout = stdout;
        let mut buf = vec![0u8; frame_size];
        while stdout.read_exact(&mut buf).is_ok() {
            if frame_tx.send(buf.clone()).is_err() {
                break;
            }
        }
    });

    Ok(FfmpegPipe {
        segment_tx: Some(segment_tx),
        frame_rx,
        child: Some(child),
        _writer_handle: writer_handle,
        _reader_handle: reader_handle,
    })
}

impl FfmpegPipe {
    /// Close stdin and kill the child. Idempotent, and leaves the pipe in the
    /// same state a dead child would: `FrameDecoder::is_alive` reports `false`
    /// afterwards, so the analyzer's respawn path takes over.
    fn kill(&mut self) {
        self.segment_tx.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for FfmpegPipe {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Hand `data` to the pipe's writer thread, giving up after `deadline` of a
/// permanently full channel. The deadline is a parameter so tests can trip the
/// wedge path without waiting out [`SEND_DEADLINE`].
fn send_with_deadline(tx: &SyncSender<Vec<u8>>, data: Vec<u8>, deadline: Duration) -> SendOutcome {
    let start = Instant::now();
    let mut data = data;
    loop {
        match tx.try_send(data) {
            Ok(()) => return SendOutcome::Sent,
            Err(TrySendError::Disconnected(_)) => return SendOutcome::Closed,
            // `try_send` hands the value back, so retrying costs no copy.
            Err(TrySendError::Full(returned)) => {
                if start.elapsed() >= deadline {
                    return SendOutcome::Wedged;
                }
                data = returned;
                thread::sleep(SEND_RETRY_INTERVAL);
            }
        }
    }
}

fn send_segment(pipe: &FfmpegPipe, data: &[u8]) -> SendOutcome {
    match pipe.segment_tx.as_ref() {
        Some(tx) => send_with_deadline(tx, data.to_vec(), SEND_DEADLINE),
        None => SendOutcome::Closed,
    }
}

/// What one call to [`FrameDecoder::decode_segment`] produced.
pub enum DecodeOutcome {
    /// The frames the segment yielded. Possibly empty: pipe buffering can push
    /// a frame past the read timeout into the next segment's window, so a
    /// single empty decode is normal. A *streak* of them is not — see the
    /// analyzer's zero-frame tripwire.
    Frames(Vec<Vec<u8>>),
    /// ffmpeg is alive but stopped consuming stdin. Motion analysis cannot
    /// resume until the child is killed and respawned.
    Wedged,
}

pub struct FrameDecoder {
    pipe: FfmpegPipe,
}

impl FrameDecoder {
    pub fn new() -> Result<Self, std::io::Error> {
        let scale_filter =
            format!("select=eq(pict_type\\,I),scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT}");
        let pipe = spawn_ffmpeg_pipe(
            &[
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-skip_frame",
                "nokey",
                "-f",
                "mpegts",
                "-i",
                "pipe:0",
                "-vf",
                &scale_filter,
                "-vsync",
                "vfr",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "pipe:1",
            ],
            FRAME_SIZE,
            16,
            64,
        )?;

        Ok(Self { pipe })
    }

    pub fn decode_segment(&self, data: &[u8]) -> DecodeOutcome {
        match send_segment(&self.pipe, data) {
            SendOutcome::Sent => {}
            // A closed pipe means the child already died; `is_alive` reports it
            // and the caller respawns without any special handling here.
            SendOutcome::Closed => return DecodeOutcome::Frames(Vec::new()),
            SendOutcome::Wedged => return DecodeOutcome::Wedged,
        }

        let mut frames = Vec::with_capacity(2);
        while let Ok(frame) = self.pipe.frame_rx.recv_timeout(FRAME_READ_TIMEOUT) {
            frames.push(frame);
        }
        DecodeOutcome::Frames(frames)
    }

    /// Kill the ffmpeg child so the caller's liveness check respawns it. Used
    /// when the pipe is wedged or the decoder has gone blind — neither of which
    /// the child recovers from on its own.
    pub fn kill(&mut self) {
        self.pipe.kill();
    }

    pub fn is_alive(&mut self) -> bool {
        self.pipe
            .child
            .as_mut()
            .map(|c| c.try_wait().ok().flatten().is_none())
            .unwrap_or(false)
    }
}

pub struct CropDecoder {
    pipe: FfmpegPipe,
    sample_fps: u32,
    width: u32,
    height: u32,
}

impl CropDecoder {
    pub fn new(sample_fps: u32, (width, height): (u32, u32)) -> Result<Self, std::io::Error> {
        let scale_filter = format!("fps={sample_fps},scale={width}:{height}");
        let pipe = spawn_ffmpeg_pipe(
            &[
                "-hide_banner",
                "-loglevel",
                "quiet",
                // A crop decoder is spawned per batch and fed ~4s of segments,
                // less than ffmpeg's default stream-analysis window — without
                // these it emits nothing before the pipe goes idle.
                "-probesize",
                "262144",
                "-analyzeduration",
                "0",
                "-fflags",
                "nobuffer",
                "-f",
                "mpegts",
                "-i",
                "pipe:0",
                "-vf",
                &scale_filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            (width * height * 3) as usize,
            16,
            16,
        )?;

        Ok(Self {
            pipe,
            sample_fps,
            width,
            height,
        })
    }

    pub fn decode_segment(&self, data: &[u8], duration_ns: u64) -> Vec<Vec<u8>> {
        match send_segment(&self.pipe, data) {
            SendOutcome::Sent => {}
            SendOutcome::Closed => return Vec::new(),
            // A crop decoder is spawned per batch and dropped after it, which
            // kills the child — so the segment is simply skipped rather than
            // respawned. The blast radius is this batch's event frames.
            SendOutcome::Wedged => {
                tracing::warn!("crop decoder stopped consuming input, skipping segment");
                return Vec::new();
            }
        }

        let expected_frames = expected_frame_count(duration_ns, self.sample_fps);
        let mut frames = Vec::with_capacity(expected_frames);
        for _ in 0..expected_frames {
            match self.pipe.frame_rx.recv_timeout(FRAME_READ_TIMEOUT) {
                Ok(frame) => frames.push(frame),
                Err(_) => break,
            }
        }
        frames
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

fn expected_frame_count(duration_ns: u64, sample_fps: u32) -> usize {
    let duration_secs = duration_ns as f64 / 1_000_000_000.0;
    (duration_secs * sample_fps as f64).ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test deadline: long enough for a couple of retry rounds, short enough
    /// that a wedged send costs CI a blink instead of [`SEND_DEADLINE`].
    const TEST_DEADLINE: Duration = Duration::from_millis(200);

    #[test]
    fn send_succeeds_while_the_channel_has_room() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        assert_eq!(
            send_with_deadline(&tx, vec![1, 2, 3], TEST_DEADLINE),
            SendOutcome::Sent
        );
        assert_eq!(rx.recv().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn send_to_a_full_channel_wedges_at_the_deadline() {
        // Nothing ever receives, so the single slot stays taken: the exact
        // shape of an ffmpeg that stopped reading its stdin.
        let (tx, _rx) = mpsc::sync_channel::<Vec<u8>>(1);
        assert_eq!(
            send_with_deadline(&tx, vec![0], TEST_DEADLINE),
            SendOutcome::Sent
        );

        let start = Instant::now();
        assert_eq!(
            send_with_deadline(&tx, vec![1], TEST_DEADLINE),
            SendOutcome::Wedged
        );
        let elapsed = start.elapsed();
        assert!(elapsed >= TEST_DEADLINE, "gave up early: {elapsed:?}");
        assert!(
            elapsed < TEST_DEADLINE * 5,
            "overshot the deadline: {elapsed:?}"
        );
    }

    #[test]
    fn send_to_a_closed_channel_reports_closed_immediately() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        drop(rx);
        let start = Instant::now();
        assert_eq!(
            send_with_deadline(&tx, vec![1], TEST_DEADLINE),
            SendOutcome::Closed
        );
        assert!(start.elapsed() < TEST_DEADLINE, "closed should not wait");
    }

    #[test]
    fn send_retries_until_a_slot_frees_up() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        assert_eq!(
            send_with_deadline(&tx, vec![0], TEST_DEADLINE),
            SendOutcome::Sent
        );
        // The receiver is handed back rather than dropped: a dropped `rx` would
        // disconnect the channel and race the retry loop into `Closed`.
        let drainer = thread::spawn(move || {
            thread::sleep(SEND_RETRY_INTERVAL * 2);
            (rx.recv().unwrap(), rx)
        });
        // Deliberately generous: the point is that a slow-but-alive consumer
        // is not mistaken for a wedge.
        assert_eq!(
            send_with_deadline(&tx, vec![1], Duration::from_secs(2)),
            SendOutcome::Sent
        );
        assert_eq!(drainer.join().unwrap().0, vec![0]);
    }

    /// Wedge detection against a real ffmpeg child, stopped mid-flight with
    /// SIGSTOP — the closest stand-in for the descheduled-under-memory-pressure
    /// case the bounded send exists for. Ignored by default: it needs an
    /// `ffmpeg` binary and burns [`SEND_DEADLINE`] once the pipe is full.
    #[test]
    #[ignore]
    fn frame_decoder_wedges_when_ffmpeg_stops_reading_stdin() {
        let decoder = FrameDecoder::new().expect("spawn ffmpeg");
        let pid = decoder.pipe.child.as_ref().expect("child").id() as libc::pid_t;
        assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0, "SIGSTOP");

        // Junk bytes: ffmpeg never gets to parse them. Enough per segment that
        // the kernel pipe buffer fills within a few sends, after which the 16
        // channel slots are all that is left.
        let segment = vec![0u8; 256 * 1024];
        let mut outcome = None;
        for _ in 0..64 {
            match decoder.decode_segment(&segment) {
                DecodeOutcome::Wedged => {
                    outcome = Some(DecodeOutcome::Wedged);
                    break;
                }
                DecodeOutcome::Frames(frames) => {
                    assert!(frames.is_empty(), "stopped ffmpeg cannot emit frames");
                }
            }
        }
        // Let Drop kill the stopped child (SIGKILL is not blocked by SIGSTOP).
        assert!(
            matches!(outcome, Some(DecodeOutcome::Wedged)),
            "a stopped ffmpeg should wedge the segment hand-off"
        );
    }
}
