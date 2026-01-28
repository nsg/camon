use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const ANALYSIS_WIDTH: u32 = 320;
const ANALYSIS_HEIGHT: u32 = 240;
const FRAME_SIZE: usize = (ANALYSIS_WIDTH * ANALYSIS_HEIGHT) as usize;
const FRAME_READ_TIMEOUT: Duration = Duration::from_millis(500);

pub struct FrameDecoder {
    segment_tx: Option<SyncSender<Vec<u8>>>,
    frame_rx: Receiver<Vec<u8>>,
    sample_fps: u32,
    child: Option<Child>,
    _writer_handle: JoinHandle<()>,
    _reader_handle: JoinHandle<()>,
}

impl FrameDecoder {
    pub fn new(sample_fps: u32) -> Result<Self, std::io::Error> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-f",
                "mpegts",
                "-i",
                "pipe:0",
                "-vf",
                &format!("fps={sample_fps},scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT}"),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");

        let (segment_tx, segment_rx) = mpsc::sync_channel::<Vec<u8>>(16);
        let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(64);

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
            let mut buf = vec![0u8; FRAME_SIZE];
            while stdout.read_exact(&mut buf).is_ok() {
                if frame_tx.send(buf.clone()).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            segment_tx: Some(segment_tx),
            frame_rx,
            sample_fps,
            child: Some(child),
            _writer_handle: writer_handle,
            _reader_handle: reader_handle,
        })
    }

    pub fn decode_segment(&self, data: &[u8], duration_ns: u64) -> Vec<Vec<u8>> {
        if let Some(tx) = &self.segment_tx {
            if tx.send(data.to_vec()).is_err() {
                return Vec::new();
            }
        }

        let duration_secs = duration_ns as f64 / 1_000_000_000.0;
        let expected_frames = (duration_secs * self.sample_fps as f64).ceil() as usize;
        let expected_frames = expected_frames.max(1);

        let mut frames = Vec::with_capacity(expected_frames);
        for _ in 0..expected_frames {
            match self.frame_rx.recv_timeout(FRAME_READ_TIMEOUT) {
                Ok(frame) => frames.push(frame),
                Err(_) => break,
            }
        }

        frames
    }

    pub fn is_alive(&mut self) -> bool {
        self.child
            .as_mut()
            .map(|c| c.try_wait().ok().flatten().is_none())
            .unwrap_or(false)
    }

    pub fn height(&self) -> u32 {
        ANALYSIS_HEIGHT
    }
}

impl Drop for FrameDecoder {
    fn drop(&mut self) {
        // Close the segment channel so the writer thread exits
        self.segment_tx.take();
        // Kill FFmpeg so the reader thread exits
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
