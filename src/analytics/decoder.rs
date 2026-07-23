use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const ANALYSIS_WIDTH: u32 = 320;
const ANALYSIS_HEIGHT: u32 = 240;
const FRAME_SIZE: usize = (ANALYSIS_WIDTH * ANALYSIS_HEIGHT) as usize;
const FRAME_READ_TIMEOUT: Duration = Duration::from_millis(500);

const CROP_WIDTH: u32 = 1920;
const CROP_HEIGHT: u32 = 1080;
const CROP_FRAME_SIZE: usize = (CROP_WIDTH * CROP_HEIGHT * 3) as usize;

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

impl Drop for FfmpegPipe {
    fn drop(&mut self) {
        self.segment_tx.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn send_segment(pipe: &FfmpegPipe, data: &[u8]) -> bool {
    pipe.segment_tx
        .as_ref()
        .map(|tx| tx.send(data.to_vec()).is_ok())
        .unwrap_or(false)
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

    pub fn decode_segment(&self, data: &[u8]) -> Vec<Vec<u8>> {
        if !send_segment(&self.pipe, data) {
            return Vec::new();
        }

        let mut frames = Vec::with_capacity(2);
        while let Ok(frame) = self.pipe.frame_rx.recv_timeout(FRAME_READ_TIMEOUT) {
            frames.push(frame);
        }
        frames
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
}

impl CropDecoder {
    pub fn new(sample_fps: u32) -> Result<Self, std::io::Error> {
        let scale_filter = format!("fps={sample_fps},scale={CROP_WIDTH}:{CROP_HEIGHT}");
        let pipe = spawn_ffmpeg_pipe(
            &[
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-f",
                "mpegts",
                "-i",
                "pipe:0",
                "-vf",
                &scale_filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            CROP_FRAME_SIZE,
            16,
            16,
        )?;

        Ok(Self { pipe, sample_fps })
    }

    pub fn decode_segment(&self, data: &[u8], duration_ns: u64) -> Vec<Vec<u8>> {
        if !send_segment(&self.pipe, data) {
            return Vec::new();
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

    pub fn height(&self) -> u32 {
        CROP_HEIGHT
    }
}

fn expected_frame_count(duration_ns: u64, sample_fps: u32) -> usize {
    let duration_secs = duration_ns as f64 / 1_000_000_000.0;
    (duration_secs * sample_fps as f64).ceil().max(1.0) as usize
}
