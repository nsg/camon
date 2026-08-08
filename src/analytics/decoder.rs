use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const ANALYSIS_WIDTH: u32 = 320;
const ANALYSIS_HEIGHT: u32 = 240;
const FRAME_SIZE: usize = (ANALYSIS_WIDTH * ANALYSIS_HEIGHT) as usize;
/// Safety net, not the normal exit path: a healthy decode returns as soon as the segment's own
/// frames arrive (single-digit milliseconds).
const FRAME_READ_TIMEOUT: Duration = Duration::from_millis(500);

/// How long a segment hand-off may stay blocked before the pipe is declared wedged.
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
    /// Segments are handed over as they are held in the hot buffer — shared,
    /// not copied: the channel keeps up to `segment_channel_size` of them in
    /// flight, and each is a whole GOP.
    segment_tx: Option<SyncSender<Arc<Vec<u8>>>>,
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

    let (segment_tx, segment_rx) = mpsc::sync_channel::<Arc<Vec<u8>>>(segment_channel_size);
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
        loop {
            // A buffer per frame, handed over by move: a reused buffer would
            // memcpy 6 MB per frame at the detection crop size, while a fresh
            // allocation is lazily zeroed pages the read overwrites anyway.
            let mut buf = vec![0u8; frame_size];
            if stdout.read_exact(&mut buf).is_err() {
                break;
            }
            if frame_tx.send(buf).is_err() {
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
fn send_with_deadline<T>(tx: &SyncSender<T>, mut data: T, deadline: Duration) -> SendOutcome {
    let start = Instant::now();
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

/// Collect one segment's frames: at most `expected` of them, blocking only until they arrive or
/// `deadline` passes.
fn collect_frames(rx: &Receiver<Vec<u8>>, expected: usize, deadline: Instant) -> Vec<Vec<u8>> {
    let mut frames = Vec::with_capacity(expected);
    while frames.len() < expected {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(frame) => frames.push(frame),
            Err(_) => break,
        }
    }
    frames
}

/// Pull up to `count` frames out of the channel and throw them away, blocking for them like
/// [`collect_frames`] does, and reporting how many were there.
fn discard_frames(rx: &Receiver<Vec<u8>>, count: usize, deadline: Instant) -> usize {
    let mut discarded = 0;
    while discarded < count {
        if rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .is_err()
        {
            break;
        }
        discarded += 1;
    }
    discarded
}

/// What one decode was owed and what turned up, split by who owed it: the
/// arrears it inherited and the segment's own frames are separate claims and a
/// decode can settle one without settling the other.
struct FrameLedger {
    arrears: usize,
    /// Arrears that turned up and were discarded.
    paid: usize,
    /// Frames this segment's keyframes promise.
    expected: usize,
    /// Frames collected for this segment.
    collected: usize,
    /// Whether the collect phase had any of the deadline left to wait in. The
    /// arrears can drain the whole budget, and a collect phase that never got
    /// to wait proves nothing about the frames it did not see.
    waited: bool,
}

/// Frames still owed after a decode, to be discarded rather than scored when they finally
/// arrive.
fn frames_still_unclaimed(ledger: &FrameLedger) -> usize {
    let unpaid_arrears = ledger.arrears - ledger.paid;
    let uncollected = ledger.expected - ledger.collected;
    if ledger.paid + ledger.collected == 0 {
        return unpaid_arrears + uncollected;
    }
    if ledger.waited {
        0
    } else {
        uncollected
    }
}

fn send_segment(pipe: &FfmpegPipe, data: Arc<Vec<u8>>) -> SendOutcome {
    match pipe.segment_tx.as_ref() {
        Some(tx) => send_with_deadline(tx, data, SEND_DEADLINE),
        None => SendOutcome::Closed,
    }
}

/// What one call to [`FrameDecoder::decode_segment`] produced.
pub enum DecodeOutcome {
    /// The frames the segment yielded. Possibly empty — a freshly spawned ffmpeg swallows
    /// input while probing — and empty means *not analyzed*, never *no motion*. A streak of
    /// empties is not normal; see the analyzer's zero-frame tripwire.
    Frames(Vec<Vec<u8>>),
    /// ffmpeg is alive but stopped consuming stdin. Motion analysis cannot
    /// resume until the child is killed and respawned.
    Wedged,
}

pub struct FrameDecoder {
    pipe: FfmpegPipe,
    /// Frames earlier segments were promised and never got, discarded on
    /// arrival rather than scored — see [`discard_frames`]. Zero in the
    /// steady state.
    unclaimed_frames: usize,
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

        Ok(Self {
            pipe,
            unclaimed_frames: 0,
        })
    }

    pub fn decode_segment(&mut self, data: &Arc<Vec<u8>>) -> DecodeOutcome {
        match send_segment(&self.pipe, Arc::clone(data)) {
            SendOutcome::Sent => {}
            // A closed pipe means the child already died; `is_alive` reports it
            // and the caller respawns without any special handling here.
            SendOutcome::Closed => return DecodeOutcome::Frames(Vec::new()),
            SendOutcome::Wedged => return DecodeOutcome::Wedged,
        }

        // The decoder keeps keyframes only, so the segment's keyframe count is exactly how many
        // frames it owns.
        let expected = crate::mpegts::keyframe_count(data).max(1);
        // One budget for the whole decode: what the analyzer must never exceed
        // is the time it spends on a segment, not the time per phase.
        let deadline = Instant::now() + FRAME_READ_TIMEOUT;
        let discarded = discard_frames(&self.pipe.frame_rx, self.unclaimed_frames, deadline);
        if discarded > 0 {
            tracing::debug!(frames = discarded, "dropped frames no segment can claim");
        }
        let waited = !deadline.saturating_duration_since(Instant::now()).is_zero();
        let frames = collect_frames(&self.pipe.frame_rx, expected, deadline);
        if frames.len() < expected {
            // A steady stream of this line means the segment's keyframe count
            // no longer matches what ffmpeg emits.
            tracing::debug!(
                expected,
                collected = frames.len(),
                "segment frames did not arrive"
            );
        }
        self.unclaimed_frames = frames_still_unclaimed(&FrameLedger {
            arrears: self.unclaimed_frames,
            paid: discarded,
            expected,
            collected: frames.len(),
            waited,
        });
        DecodeOutcome::Frames(frames)
    }

    /// A decoder whose child is already gone, built without forking one first. For the
    /// shutdown-drain tests: the dead-decoder drain path must be reachable without ffmpeg
    /// installed, or only `#[ignore]`d tests would pin it.
    #[cfg(test)]
    pub(crate) fn dead() -> Self {
        let (_frame_tx, frame_rx) = std::sync::mpsc::channel();
        Self {
            pipe: FfmpegPipe {
                segment_tx: None,
                frame_rx,
                child: None,
                _writer_handle: std::thread::spawn(|| {}),
                _reader_handle: std::thread::spawn(|| {}),
            },
            unclaimed_frames: 0,
        }
    }

    /// Kill the ffmpeg child so the caller's liveness check respawns it. Used
    /// when the pipe is wedged or the decoder has gone blind — neither of which
    /// the child recovers from on its own.
    pub fn kill(&mut self) {
        self.pipe.kill();
        // Arrears belong to the dead child's stream; a replacement would spend
        // them discarding the new stream's first frames.
        self.unclaimed_frames = 0;
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
                // A crop decoder goes long stretches with nothing to decode and is then fed a
                // few seconds of segments at a time, less than ffmpeg's default stream-analysis
                // window — without these it emits nothing before the pipe goes idle again.
                "-probesize",
                "262144",
                "-analyzeduration",
                "0",
                "-fflags",
                "nobuffer",
                // A decoder kept for the life of a camera sees a discontinuous timeline: quiet
                // stretches arrive as PTS jumps and the 33-bit MPEG-TS timestamp wraps about
                // daily.
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
            // Four frames, not sixteen: at 6 MB each on the detection crop
            // size, sixteen is 100 MB of headroom no decode ever needed. A
            // full channel simply backpressures ffmpeg.
            4,
        )?;

        Ok(Self {
            pipe,
            sample_fps,
            width,
            height,
        })
    }

    /// Feed one segment and hand each frame it yields to `sink` as it arrives. Streamed rather
    /// than collected: `sample_fps` is config with no ceiling, so a collected decode's peak
    /// memory would scale with a number an operator can set to anything.
    pub fn decode_segment(
        &mut self,
        data: &Arc<Vec<u8>>,
        duration_ns: u64,
        mut sink: impl FnMut(Vec<u8>),
    ) {
        match send_segment(&self.pipe, Arc::clone(data)) {
            SendOutcome::Sent => {}
            SendOutcome::Closed => return,
            // A wedge never clears on its own, and this decoder outlives the batch that first
            // needed it — so the child is killed here rather than left for the caller to
            // notice.
            SendOutcome::Wedged => {
                tracing::warn!("crop decoder stopped consuming input, killing it");
                self.pipe.kill();
                return;
            }
        }

        for _ in 0..expected_frame_count(duration_ns, self.sample_fps) {
            match self.pipe.frame_rx.recv_timeout(FRAME_READ_TIMEOUT) {
                Ok(frame) => sink(frame),
                Err(_) => break,
            }
        }
    }

    /// Throw away every frame already waiting in the pipe, reporting how many.
    pub fn drain(&self) -> usize {
        let mut dropped = 0;
        while self.pipe.frame_rx.try_recv().is_ok() {
            dropped += 1;
        }
        dropped
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Whether the ffmpeg behind this decoder is still there.
    pub fn is_alive(&mut self) -> bool {
        self.pipe
            .child
            .as_mut()
            .map(|c| c.try_wait().ok().flatten().is_none())
            .unwrap_or(false)
    }

    /// The pid of the ffmpeg behind this decoder. For the tests that have to
    /// tell one child from its replacement, which is the only way to see from
    /// the outside whether a decoder was reused or re-forked.
    #[cfg(test)]
    pub(crate) fn child_id(&self) -> Option<u32> {
        self.pipe.child.as_ref().map(Child::id)
    }

    /// A decoder with no ffmpeg behind it and a frame channel the caller fills itself, handed
    /// back alongside it.
    #[cfg(test)]
    pub(crate) fn detached() -> (Self, SyncSender<Vec<u8>>) {
        let (frame_tx, frame_rx) = mpsc::sync_channel(4);
        let decoder = Self {
            pipe: FfmpegPipe {
                segment_tx: None,
                frame_rx,
                child: None,
                _writer_handle: thread::spawn(|| {}),
                _reader_handle: thread::spawn(|| {}),
            },
            sample_fps: 5,
            width: ANALYSIS_WIDTH,
            height: ANALYSIS_HEIGHT,
        };
        (decoder, frame_tx)
    }
}

fn expected_frame_count(duration_ns: u64, sample_fps: u32) -> usize {
    let duration_secs = duration_ns as f64 / 1_000_000_000.0;
    (duration_secs * sample_fps as f64).ceil().max(1.0) as usize
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

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
        let drainer = thread::spawn(move || {
            thread::sleep(SEND_RETRY_INTERVAL * 2);
            (rx.recv().unwrap(), rx)
        });
        assert_eq!(
            send_with_deadline(&tx, vec![1], Duration::from_secs(2)),
            SendOutcome::Sent
        );
        assert_eq!(drainer.join().unwrap().0, vec![0]);
    }

    fn frame(tag: u8) -> Vec<u8> {
        vec![tag; 4]
    }

    fn generous() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    #[test]
    fn collect_frames_returns_once_the_segment_is_done() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
        tx.send(frame(1)).unwrap();
        tx.send(frame(2)).unwrap();
        let start = Instant::now();
        let frames = collect_frames(&rx, 2, generous());
        assert_eq!(frames, vec![frame(1), frame(2)]);
        assert!(start.elapsed() < TEST_DEADLINE, "waited on the deadline");
    }

    #[test]
    fn collect_frames_waits_for_a_frame_still_in_flight() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            tx.send(frame(7)).unwrap();
        });
        assert_eq!(collect_frames(&rx, 1, generous()), vec![frame(7)]);
        producer.join().unwrap();
    }

    #[test]
    fn collect_frames_takes_no_more_than_the_segment_owns() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
        for tag in 1..=3 {
            tx.send(frame(tag)).unwrap();
        }
        let start = Instant::now();
        assert_eq!(collect_frames(&rx, 1, generous()), vec![frame(1)]);
        assert!(start.elapsed() < TEST_DEADLINE, "waited on the deadline");
        assert_eq!(collect_frames(&rx, 2, generous()), vec![frame(2), frame(3)]);
    }

    #[test]
    fn collect_frames_gives_up_at_the_deadline() {
        let (_tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let start = Instant::now();
        assert!(collect_frames(&rx, 1, Instant::now() + TEST_DEADLINE).is_empty());
        let elapsed = start.elapsed();
        assert!(elapsed >= TEST_DEADLINE, "gave up early: {elapsed:?}");
        assert!(elapsed < TEST_DEADLINE * 5, "overshot: {elapsed:?}");
    }

    #[test]
    fn discard_frames_takes_arrears_out_of_the_channel() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
        for tag in 1..=3 {
            tx.send(frame(tag)).unwrap();
        }
        let start = Instant::now();
        assert_eq!(discard_frames(&rx, 2, generous()), 2);
        assert!(start.elapsed() < TEST_DEADLINE, "waited on the deadline");
        assert_eq!(collect_frames(&rx, 1, generous()), vec![frame(3)]);
    }

    #[test]
    fn discard_frames_gives_up_at_the_deadline() {
        let (_tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let start = Instant::now();
        assert_eq!(discard_frames(&rx, 3, Instant::now() + TEST_DEADLINE), 0);
        assert!(start.elapsed() >= TEST_DEADLINE, "gave up early");
        let start = Instant::now();
        assert_eq!(discard_frames(&rx, 0, generous()), 0);
        assert!(start.elapsed() < TEST_DEADLINE, "waited for nothing");
    }

    fn ledger(arrears: usize, paid: usize, expected: usize, collected: usize) -> FrameLedger {
        FrameLedger {
            arrears,
            paid,
            expected,
            collected,
            waited: true,
        }
    }

    #[test]
    fn unclaimed_frames_survive_a_silent_decode() {
        assert_eq!(frames_still_unclaimed(&ledger(0, 0, 1, 0)), 1);
        assert_eq!(frames_still_unclaimed(&ledger(4, 0, 1, 0)), 5);
        assert_eq!(frames_still_unclaimed(&ledger(0, 0, 0, 0)), 0);
    }

    #[test]
    fn unclaimed_frames_are_written_off_once_frames_flow() {
        assert_eq!(frames_still_unclaimed(&ledger(2, 2, 1, 1)), 0);
        assert_eq!(frames_still_unclaimed(&ledger(0, 0, 2, 1)), 0);
        assert_eq!(frames_still_unclaimed(&ledger(3, 1, 1, 1)), 0);
    }

    #[test]
    fn unclaimed_frames_keep_a_segment_that_never_got_waited_for() {
        let starved = FrameLedger {
            arrears: 5,
            paid: 5,
            expected: 1,
            collected: 0,
            waited: false,
        };
        assert_eq!(frames_still_unclaimed(&starved), 1);
        assert_eq!(frames_still_unclaimed(&ledger(5, 5, 1, 0)), 0);
    }

    pub(crate) fn recorded_segments(count: usize) -> Vec<Arc<Vec<u8>>> {
        recorded_segments_starting_at(count, 0)
    }

    fn recorded_segments_starting_at(count: usize, offset_secs: u64) -> Vec<Arc<Vec<u8>>> {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=640x480:rate=25",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440",
                "-t",
                &count.to_string(),
                "-c:v",
                "libx264",
                "-g",
                "25",
                "-keyint_min",
                "25",
                "-sc_threshold",
                "0",
                "-c:a",
                "aac",
                "-output_ts_offset",
                &offset_secs.to_string(),
                "-f",
                "segment",
                "-segment_time",
                "1",
            ])
            .arg(dir.path().join("seg%03d.ts"))
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg failed to generate segments");
        (0..count)
            .map(|i| {
                Arc::new(std::fs::read(dir.path().join(format!("seg{i:03}.ts"))).expect("segment"))
            })
            .collect()
    }

    const WRAP_SECS: u64 = (1 << 33) / 90_000;

    fn picture(frame: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        frame.hash(&mut hasher);
        hasher.finish()
    }

    fn pictures_of(decoder: &mut CropDecoder, segments: &[Arc<Vec<u8>>]) -> Vec<u64> {
        let mut pictures = Vec::new();
        for segment in segments {
            decoder.decode_segment(segment, 1_000_000_000, |frame| {
                pictures.push(picture(&frame))
            });
        }
        pictures
    }

    fn frames_of(decoder: &mut CropDecoder, segments: &[Arc<Vec<u8>>]) -> usize {
        pictures_of(decoder, segments).len()
    }

    fn longest_repeat(pictures: &[u64]) -> usize {
        let mut longest = 0;
        let mut run = 0;
        let mut previous = None;
        for &p in pictures {
            run = if previous == Some(p) { run + 1 } else { 1 };
            previous = Some(p);
            longest = longest.max(run);
        }
        longest
    }

    const MAX_REPEATS: usize = 3;

    fn frames_until_quiet(decoder: &CropDecoder, cap: usize) -> usize {
        const QUIET_ROUNDS: u32 = 4;
        let mut total = 0;
        let mut quiet = 0;
        while quiet < QUIET_ROUNDS && total < cap {
            match decoder.drain() {
                0 => {
                    quiet += 1;
                    thread::sleep(Duration::from_millis(50));
                }
                n => {
                    quiet = 0;
                    total += n;
                }
            }
        }
        total
    }

    #[test]
    #[ignore]
    fn frame_decoder_ends_the_decode_when_the_frames_are_done() {
        const SEGMENTS: usize = 10;
        let segments = recorded_segments(SEGMENTS);

        let mut decoder = FrameDecoder::new().expect("spawn ffmpeg");
        let mut decodes = Vec::new();
        for (i, segment) in segments.iter().enumerate() {
            assert_eq!(
                crate::mpegts::keyframe_count(segment),
                1,
                "segment {i} is one GOP"
            );

            let start = Instant::now();
            let frames = match decoder.decode_segment(segment) {
                DecodeOutcome::Frames(frames) => frames,
                DecodeOutcome::Wedged => panic!("healthy decoder wedged"),
            };
            decodes.push((frames.len(), start.elapsed()));
        }

        let burst = decodes
            .iter()
            .position(|&(frames, _)| frames > 0)
            .expect("decoder produced no frames at all");
        assert!(burst < SEGMENTS - 2, "no steady state to measure");
        for (i, &(frames, elapsed)) in decodes.iter().enumerate().skip(burst + 1) {
            assert!(frames > 0, "segment {i} decoded nothing");
            assert!(
                elapsed < FRAME_READ_TIMEOUT / 2,
                "segment {i} ended on the safety timeout: {elapsed:?}"
            );
        }
    }

    #[test]
    #[ignore]
    fn frame_decoder_safety_net_ends_a_decode_that_owes_frames() {
        let segment = recorded_segments(1).remove(0);
        assert_eq!(crate::mpegts::keyframe_count(&segment), 1);

        let mut decoder = FrameDecoder::new().expect("spawn ffmpeg");
        let pid = decoder.pipe.child.as_ref().expect("child").id() as libc::pid_t;
        assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0, "SIGSTOP");

        let start = Instant::now();
        match decoder.decode_segment(&segment) {
            DecodeOutcome::Frames(frames) => assert!(frames.is_empty(), "stopped ffmpeg emitted"),
            DecodeOutcome::Wedged => panic!("one segment cannot fill the hand-off channel"),
        }
        let elapsed = start.elapsed();
        assert!(elapsed >= FRAME_READ_TIMEOUT, "gave up early: {elapsed:?}");
        assert!(elapsed < SEND_DEADLINE, "blocked past the read timeout");
    }

    #[test]
    #[ignore]
    fn frame_decoder_wedges_when_ffmpeg_stops_reading_stdin() {
        let mut decoder = FrameDecoder::new().expect("spawn ffmpeg");
        let pid = decoder.pipe.child.as_ref().expect("child").id() as libc::pid_t;
        assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0, "SIGSTOP");

        let segment = Arc::new(vec![0u8; 256 * 1024]);
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
        assert!(
            matches!(outcome, Some(DecodeOutcome::Wedged)),
            "a stopped ffmpeg should wedge the segment hand-off"
        );
    }

    #[test]
    #[ignore]
    fn crop_decoder_reuses_one_child_across_a_gap_between_batches() {
        const FPS: u32 = 5;
        const SPANNED: usize = 16;
        let segments = recorded_segments(16);
        let mut decoder = CropDecoder::new(FPS, (320, 180)).expect("spawn ffmpeg");

        let mut emitted = frames_of(&mut decoder, &segments[..3]);
        emitted += decoder.drain();

        let first_batch = frames_of(&mut decoder, &segments[3..9]);
        let second_batch = frames_of(&mut decoder, &segments[13..16]);

        assert!(first_batch > 0, "the primed decoder produced nothing");
        assert!(
            second_batch > 0,
            "the child did not survive the gap between batches"
        );
        assert!(decoder.is_alive(), "the child died between batches");

        let budget = SPANNED * FPS as usize * 2;
        emitted += first_batch + second_batch + frames_until_quiet(&decoder, budget + 1);
        assert!(
            emitted <= budget,
            "twelve seconds of footage across sixteen came back as {emitted} frames"
        );
    }

    #[test]
    #[ignore]
    fn a_kept_crop_decoder_answers_jumped_timestamps_with_its_own_frames() {
        const FPS: u32 = 5;
        const BATCH: usize = 3;
        let now = recorded_segments(20);
        let an_hour_on = recorded_segments_starting_at(BATCH, 3600);
        let before_the_wrap = recorded_segments_starting_at(BATCH, WRAP_SECS - 6);
        let after_the_wrap = recorded_segments_starting_at(BATCH + 1, WRAP_SECS + 2);
        let fed = now.len() + an_hour_on.len() + before_the_wrap.len() + after_the_wrap.len();

        let mut decoder = CropDecoder::new(FPS, (320, 180)).expect("spawn ffmpeg");
        pictures_of(&mut decoder, &now[..BATCH]);
        decoder.drain();

        let before_the_jump = pictures_of(&mut decoder, &now[BATCH..]);
        let mut emitted = before_the_jump.len();
        let settled = &before_the_jump[before_the_jump.len().saturating_sub(MAX_REPEATS * 2)..];
        assert!(
            !settled.is_empty() && longest_repeat(settled) <= MAX_REPEATS,
            "the child was still padding when the first jump was measured: \
             {} frames before it, ending in a run of {}",
            before_the_jump.len(),
            longest_repeat(settled)
        );

        for (what, batch) in [
            ("an hour on", &an_hour_on[..]),
            ("just before the wrap", &before_the_wrap[..]),
            ("just after the wrap", &after_the_wrap[..BATCH]),
        ] {
            let pictures = pictures_of(&mut decoder, batch);
            assert!(
                !pictures.is_empty(),
                "the batch {what} got no frames at all"
            );
            let repeated = longest_repeat(&pictures);
            assert!(
                repeated <= MAX_REPEATS,
                "{repeated} of the {} frames the batch {what} got were the same \
                 picture over again: the jump was filled in, not crossed",
                pictures.len()
            );
            assert!(decoder.is_alive(), "the child died {what}");
            emitted += pictures.len();
        }

        let budget = fed * FPS as usize * 2;
        emitted += frames_until_quiet(&decoder, budget + 1);

        let sentinel = pictures_of(&mut decoder, &after_the_wrap[BATCH..]);
        assert!(
            !sentinel.is_empty() && longest_repeat(&sentinel) <= MAX_REPEATS,
            "the child answered a fresh segment with {} of the same picture: \
             the silence the count stopped on was a pause in a flood",
            longest_repeat(&sentinel)
        );
        emitted += sentinel.len();
        emitted += frames_until_quiet(&decoder, budget + 1);

        assert!(
            emitted <= budget,
            "{fed}s of footage came back as {emitted} frames: the jumps were filled in"
        );
    }

    #[test]
    #[ignore]
    fn crop_decoder_kills_a_child_that_stopped_reading_stdin() {
        let mut decoder = CropDecoder::new(5, (320, 180)).expect("spawn ffmpeg");
        let pid = decoder.pipe.child.as_ref().expect("child").id() as libc::pid_t;
        assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0, "SIGSTOP");
        assert!(decoder.is_alive(), "a stopped child is still a child");

        let segment = Arc::new(vec![0u8; 256 * 1024]);
        for _ in 0..64 {
            decoder.decode_segment(&segment, 1, |_| panic!("a stopped ffmpeg emitted a frame"));
            if !decoder.is_alive() {
                return;
            }
        }
        panic!("a stopped ffmpeg was left wedged and alive");
    }

    #[test]
    #[ignore]
    fn crop_decoder_streams_a_run_through_a_four_slot_channel() {
        const PRIMING: usize = 3;
        const SAMPLED: usize = 4;
        let segments = recorded_segments(PRIMING + SAMPLED);
        let mut decoder = CropDecoder::new(5, (320, 180)).expect("spawn ffmpeg");

        for segment in &segments[..PRIMING] {
            decoder.decode_segment(segment, 1_000_000_000, |_| {});
        }
        decoder.drain();
        assert_eq!(decoder.drain(), 0, "a drained pipe drains to nothing");

        let mut sizes = Vec::new();
        let started = Instant::now();
        for segment in &segments[PRIMING..] {
            decoder.decode_segment(segment, 1_000_000_000, |frame| sizes.push(frame.len()));
        }
        let elapsed = started.elapsed();

        assert!(
            !sizes.is_empty(),
            "no frames survived the run: a wedged hand-off returns empty"
        );
        assert!(
            sizes.iter().all(|&len| len == 320 * 180 * 3),
            "the pipe must deliver whole frames: {sizes:?}"
        );
        assert!(
            elapsed < SEND_DEADLINE,
            "the sampled segments took {elapsed:?}, which is wedge territory"
        );
    }
}
