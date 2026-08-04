use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const ANALYSIS_WIDTH: u32 = 320;
const ANALYSIS_HEIGHT: u32 = 240;
const FRAME_SIZE: usize = (ANALYSIS_WIDTH * ANALYSIS_HEIGHT) as usize;
/// Safety net, not the normal exit path: how long a decode waits for frames
/// the segment says are coming before giving up on them. A healthy decode
/// returns as soon as the segment's own frames have arrived — single-digit
/// milliseconds — because the expected count comes from the segment itself.
/// Only a wedged ffmpeg, or one still probing its input after a spawn, waits
/// this out, and it must never block the analyzer longer than that.
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
            // A buffer per frame, handed over by move. Reading into one reused
            // buffer and cloning it out would memcpy the whole frame — 6 MB at
            // the detection crop size, on every frame of every crop decode —
            // where a fresh allocation of that size is served by lazily zeroed
            // pages the read overwrites anyway. A frame that fails to read is
            // dropped rather than sent, exactly as the reused buffer's partial
            // contents were before.
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

/// Collect one segment's frames: at most `expected` of them, blocking only
/// until they arrive or `deadline` passes.
///
/// `expected` is what the segment itself promises, so a healthy decode returns
/// the moment ffmpeg has emitted it — the deadline is only reached when a
/// promised frame never comes. Taking no more than the segment owns is what
/// keeps a decoder that releases a backlog in one burst from having all of it
/// averaged into whichever segment happened to be in flight.
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

/// Pull up to `count` frames out of the channel and throw them away, reporting
/// how many were there. Blocks for them like [`collect_frames`] does: they are
/// still inside ffmpeg, not in the channel, which is exactly why they have to
/// be waited for.
///
/// These are frames earlier segments were promised and never received — a fresh
/// ffmpeg holds several seconds of input back while it probes the stream. They
/// are real footage, but the segments that own them are already past. Scoring
/// them against whichever segment is in flight would put their motion in the
/// wrong second of video and dilute that segment's own score; leaving them
/// queued would do the same to every later segment. So they are taken out and
/// dropped, which puts the next frame out of ffmpeg back in step with the next
/// segment in.
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

/// Frames still owed after a decode, to be discarded rather than scored when
/// they finally arrive.
///
/// Silence is the only evidence that missing frames are still coming: a freshly
/// spawned ffmpeg emits nothing at all until it has probed several seconds of
/// input, then releases that backlog in order, and only a decoder that
/// remembers what it is owed can drop the backlog instead of scoring it against
/// the wrong segments. Once frames *are* flowing, whatever failed to arrive
/// alongside them is gone — ffmpeg dropped an undecodable keyframe — and
/// waiting for it again would cost the safety timeout on every later segment.
///
/// The one exception is this segment's own frames when the arrears ate the
/// budget before they could be waited for. Writing those off would hand them to
/// the next segment and leave a one-segment lag that nothing afterwards can
/// detect, since every later decode then finds a frame waiting and never times
/// out.
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
    /// The frames the segment yielded. Possibly empty: a freshly spawned
    /// ffmpeg swallows several seconds of input while it probes the stream, so
    /// the segments fed meanwhile decode to nothing and a frame that arrives
    /// after its own decode gave up is dropped rather than credited to a later
    /// segment. A single empty decode is therefore normal, and means *not
    /// analyzed* rather than *no motion*. A streak of them is not normal — see
    /// the analyzer's zero-frame tripwire.
    Frames(Vec<Vec<u8>>),
    /// ffmpeg is alive but stopped consuming stdin. Motion analysis cannot
    /// resume until the child is killed and respawned.
    Wedged,
}

pub struct FrameDecoder {
    pipe: FfmpegPipe,
    /// Frames earlier segments were promised and never got. Whoever owned them
    /// is past, so they are discarded on arrival rather than scored — see
    /// [`discard_frames`]. Zero in the steady state; non-zero while a freshly
    /// spawned ffmpeg holds its first seconds of input back, or while a blind
    /// one emits nothing at all.
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

        // The decoder keeps keyframes only, so the segment's own keyframe count
        // is exactly how many frames it owns — no timer has to stand in for the
        // end of the segment. A hot-buffer segment always opens on a keyframe,
        // so a count of zero means the parse found no video PES at all: wait for
        // one frame anyway, because expecting none would leave this segment's
        // own frame to be discarded as another segment's arrears.
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
            // Every decode short of its keyframe count spends the full budget,
            // and frames still flow, so neither the tripwire nor the throughput
            // numbers show it. A steady stream of this line means the segment's
            // keyframe count no longer matches what ffmpeg emits.
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

    /// A decoder whose child is already gone, built without forking one first.
    ///
    /// For the shutdown-drain tests. The path where the decoder dies before the
    /// drain begins is the path where a recording most easily loses its tail,
    /// and it has to be reachable from a test that does not depend on ffmpeg
    /// being installed — every test here that forks one is `#[ignore]`d, which
    /// would leave that path unpinned in the suite that actually gates commits.
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
            // Four frames, not the sixteen a gray analysis frame can afford:
            // these are 6 MB each at the detection crop size, so the channel
            // alone was 100 MB of headroom no decode ever needed — a read takes
            // at most one segment's frames and consumes them as they arrive.
            // A full channel simply backpressures ffmpeg, which the reader
            // thread is there to absorb.
            4,
        )?;

        Ok(Self {
            pipe,
            sample_fps,
            width,
            height,
        })
    }

    /// Feed one segment and hand each frame it yields to `sink` as it arrives.
    ///
    /// Streamed rather than returned as a `Vec` because a segment owns
    /// `sample_fps` frames per second of footage and `sample_fps` is config
    /// with no ceiling, while every caller keeps a fixed handful. Collecting
    /// the segment first would put the whole decode live at once — at the
    /// detection crop size, 6 MB a frame — and make the peak scale with a
    /// number an operator can set to anything.
    pub fn decode_segment(
        &self,
        data: &Arc<Vec<u8>>,
        duration_ns: u64,
        mut sink: impl FnMut(Vec<u8>),
    ) {
        match send_segment(&self.pipe, Arc::clone(data)) {
            SendOutcome::Sent => {}
            SendOutcome::Closed => return,
            // A crop decoder is spawned per batch and dropped after it, which
            // kills the child — so the segment is simply skipped rather than
            // respawned. The blast radius is this batch's event frames.
            SendOutcome::Wedged => {
                tracing::warn!("crop decoder stopped consuming input, skipping segment");
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
    ///
    /// This decoder has no arrears ledger of the kind [`FrameDecoder`] keeps,
    /// and cannot usefully have one: its expected count is `sample_fps` times a
    /// duration, an estimate of what ffmpeg's `fps` filter will emit rather
    /// than a parsed keyframe count, so a debt tracked against it would drift
    /// and start discarding real frames. What a caller *can* say is "everything
    /// emitted up to here belongs to footage I am not keeping", which is
    /// exactly the priming segments' case.
    ///
    /// Only what has already arrived is dropped. A frame still inside ffmpeg is
    /// not waited for, so this bounds the misalignment between segments and
    /// frames rather than removing it — no caller may depend on the frame after
    /// a drain belonging to the next segment fed.
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

    fn frame(tag: u8) -> Vec<u8> {
        vec![tag; 4]
    }

    /// A deadline no healthy path may ever reach.
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
        // Frames beyond the segment's keyframe count are somebody else's:
        // averaging them in would misplace the motion they hold and dilute this
        // segment's own score. They stay queued for the arrears accounting.
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
        // A wedged decoder owes a frame it will never send: the safety net has
        // to end the decode, and only this path may reach it.
        let (_tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let start = Instant::now();
        assert!(collect_frames(&rx, 1, Instant::now() + TEST_DEADLINE).is_empty());
        let elapsed = start.elapsed();
        assert!(elapsed >= TEST_DEADLINE, "gave up early: {elapsed:?}");
        assert!(elapsed < TEST_DEADLINE * 5, "overshot: {elapsed:?}");
    }

    #[test]
    fn discard_frames_takes_arrears_out_of_the_channel() {
        // The frames a probing ffmpeg finally releases: they belong to segments
        // already passed, so they are dropped — but they must leave the channel
        // or the next segment is scored against them.
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
        for tag in 1..=3 {
            tx.send(frame(tag)).unwrap();
        }
        let start = Instant::now();
        assert_eq!(discard_frames(&rx, 2, generous()), 2);
        assert!(start.elapsed() < TEST_DEADLINE, "waited on the deadline");
        // Only the arrears are dropped; the rest is the next segment's.
        assert_eq!(collect_frames(&rx, 1, generous()), vec![frame(3)]);
    }

    #[test]
    fn discard_frames_gives_up_at_the_deadline() {
        let (_tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let start = Instant::now();
        assert_eq!(discard_frames(&rx, 3, Instant::now() + TEST_DEADLINE), 0);
        assert!(start.elapsed() >= TEST_DEADLINE, "gave up early");
        // Nothing owed is nothing to wait for.
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
        // Nothing at all came out: ffmpeg is still buffering, and what it owes
        // has to be remembered so the backlog can be dropped when it arrives
        // instead of scored against whichever segment is in flight then.
        assert_eq!(frames_still_unclaimed(&ledger(0, 0, 1, 0)), 1);
        assert_eq!(frames_still_unclaimed(&ledger(4, 0, 1, 0)), 5);
        assert_eq!(frames_still_unclaimed(&ledger(0, 0, 0, 0)), 0);
    }

    #[test]
    fn unclaimed_frames_are_written_off_once_frames_flow() {
        // A settled decode owes nothing.
        assert_eq!(frames_still_unclaimed(&ledger(2, 2, 1, 1)), 0);
        // Frames arrived but not all of them: ffmpeg dropped an undecodable
        // keyframe. Waiting for it again would cost the safety timeout on
        // every later segment.
        assert_eq!(frames_still_unclaimed(&ledger(0, 0, 2, 1)), 0);
        assert_eq!(frames_still_unclaimed(&ledger(3, 1, 1, 1)), 0);
    }

    #[test]
    fn unclaimed_frames_keep_a_segment_that_never_got_waited_for() {
        // The arrears drained the whole budget, so the collect phase returned
        // empty-handed without ever waiting. Writing this segment's frame off
        // would hand it to the next segment and leave a one-segment lag that
        // nothing afterwards can detect.
        let starved = FrameLedger {
            arrears: 5,
            paid: 5,
            expected: 1,
            collected: 0,
            waited: false,
        };
        assert_eq!(frames_still_unclaimed(&starved), 1);
        // Having waited and seen nothing is evidence; the frame is gone.
        assert_eq!(frames_still_unclaimed(&ledger(5, 5, 1, 0)), 0);
    }

    /// One-GOP MPEG-TS segments with an audio track, straight out of ffmpeg's
    /// muxer — what the hot buffer holds, near enough. Needs an `ffmpeg`
    /// binary, so only the `#[ignore]`d tests use it.
    fn recorded_segments(count: usize) -> Vec<Arc<Vec<u8>>> {
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

    /// Fed the way the analyzer feeds them, real segments must pin the two
    /// properties the count-based drain exists for: the keyframe count matches
    /// what ffmpeg emits (the audio track's packets are *all* flagged as random
    /// access points and must not be counted), and a decode ends on its frames
    /// rather than on [`FRAME_READ_TIMEOUT`].
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

        // A freshly spawned ffmpeg swallows the first few seconds of input
        // while it probes the stream and releases those frames in one burst;
        // every decode after that is the steady state this fix is about.
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

    /// The safety net against a real ffmpeg that owes a frame and will never
    /// send it: a stopped child still accepts one segment into the pipe, so the
    /// send succeeds and only [`FRAME_READ_TIMEOUT`] can end the decode. The
    /// segment must yield nothing rather than borrow a later frame — the
    /// analyzer treats an empty decode as "not analyzed", never as "quiet".
    /// Ignored by default: it needs an `ffmpeg` binary and burns the timeout.
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

    /// Wedge detection against a real ffmpeg child, stopped mid-flight with
    /// SIGSTOP — the closest stand-in for the descheduled-under-memory-pressure
    /// case the bounded send exists for. Ignored by default: it needs an
    /// `ffmpeg` binary and burns [`SEND_DEADLINE`] once the pipe is full.
    #[test]
    #[ignore]
    fn frame_decoder_wedges_when_ffmpeg_stops_reading_stdin() {
        let mut decoder = FrameDecoder::new().expect("spawn ffmpeg");
        let pid = decoder.pipe.child.as_ref().expect("child").id() as libc::pid_t;
        assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0, "SIGSTOP");

        // Junk bytes: ffmpeg never gets to parse them. Enough per segment that
        // the kernel pipe buffer fills within a few sends, after which the 16
        // channel slots are all that is left.
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
        // Let Drop kill the stopped child (SIGKILL is not blocked by SIGSTOP).
        assert!(
            matches!(outcome, Some(DecodeOutcome::Wedged)),
            "a stopped ffmpeg should wedge the segment hand-off"
        );
    }

    /// The crop decoder driven exactly as the analyzer drives it — three
    /// priming segments discarded, a drain, then the sampled segments — against
    /// a real ffmpeg.
    ///
    /// The frame channel holds four of these rather than sixteen, because at
    /// the detection crop size sixteen is 100 MB of headroom no decode ever
    /// reads. A channel that small is only safe if a full one backpressures
    /// ffmpeg instead of deadlocking it: the writer thread would stop draining
    /// stdin, the segment hand-off would fill, and `SEND_DEADLINE` would report
    /// the pipe wedged. That cannot be reasoned about from the types, so it is
    /// tested. Ignored by default: it needs an `ffmpeg` binary.
    #[test]
    #[ignore]
    fn crop_decoder_streams_a_run_through_a_four_slot_channel() {
        const PRIMING: usize = 3;
        const SAMPLED: usize = 4;
        let segments = recorded_segments(PRIMING + SAMPLED);
        let decoder = CropDecoder::new(5, (320, 180)).expect("spawn ffmpeg");

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
        // A wedge costs SEND_DEADLINE per segment before it is reported.
        assert!(
            elapsed < SEND_DEADLINE,
            "the sampled segments took {elapsed:?}, which is wedge territory"
        );
    }
}
