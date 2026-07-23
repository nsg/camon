//! Run the camon motion detector over a file of raw 8-bit grayscale frames.
//!
//! Frames are consumed back-to-back (`width * height` bytes each), the same
//! raw format the analyzer's ffmpeg decoder produces. One line per frame is
//! printed to stdout: `<index> <score> <bbox-count>`. Stage masks can
//! optionally be dumped (concatenated, same raw format) for offline
//! inspection and tuning; frames suppressed by warmup or scene-change
//! handling produce all-zero masks.
//!
//! Usage:
//!   motion_frames <frames.raw> <width> <height>
//!                 [--final-masks FILE] [--raw-masks FILE] [--morph-masks FILE]

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use camon::analytics::MotionDetector;

fn main() {
    let mut args = std::env::args().skip(1);
    let (frames_path, width, height) = match (args.next(), args.next(), args.next()) {
        (Some(p), Some(w), Some(h)) => {
            let w: usize = w.parse().expect("width must be an integer");
            let h: usize = h.parse().expect("height must be an integer");
            (PathBuf::from(p), w, h)
        }
        _ => {
            eprintln!(
                "usage: motion_frames <frames.raw> <width> <height> \
                 [--final-masks FILE] [--raw-masks FILE] [--morph-masks FILE]"
            );
            std::process::exit(2);
        }
    };

    let mut final_masks: Option<File> = None;
    let mut raw_masks: Option<File> = None;
    let mut morph_masks: Option<File> = None;
    while let Some(flag) = args.next() {
        let path = args.next().unwrap_or_else(|| {
            eprintln!("missing file argument for {flag}");
            std::process::exit(2);
        });
        let file = File::create(&path).expect("create mask output file");
        match flag.as_str() {
            "--final-masks" => final_masks = Some(file),
            "--raw-masks" => raw_masks = Some(file),
            "--morph-masks" => morph_masks = Some(file),
            _ => {
                eprintln!("unknown flag {flag}");
                std::process::exit(2);
            }
        }
    }

    let data = std::fs::read(&frames_path).expect("read frames file");
    let frame_size = width * height;
    assert!(frame_size > 0, "invalid dimensions");
    let nframes = data.len() / frame_size;
    eprintln!("{nframes} frames of {width}x{height}");

    // Detector state (tuner params) goes to a throwaway directory.
    let state_dir =
        std::env::temp_dir().join(format!("camon-motion-frames-{}", std::process::id()));
    let mut detector = MotionDetector::new("frames", &state_dir);

    let zeros = vec![0u8; frame_size];
    let write_mask =
        |file: &mut Option<File>, mask: Option<(&[u8], usize, usize)>, processed: bool| {
            if let Some(f) = file {
                let data = match mask {
                    Some((m, _, _)) if processed => m,
                    _ => &zeros,
                };
                f.write_all(data).expect("write mask");
            }
        };

    for (i, frame) in data.chunks_exact(frame_size).enumerate() {
        let score = detector.process_frame(frame, width, height);
        // Suppressed frames (warmup / scene change) leave the final and morph
        // masks stale, so substitute zeros for them.
        let processed = detector.is_warmed_up();
        println!("{i} {score:.6} {}", detector.motion_bboxes().len());
        write_mask(&mut final_masks, detector.fg_mask(), processed);
        write_mask(&mut raw_masks, detector.raw_mask(), true);
        write_mask(&mut morph_masks, detector.morph_mask(), processed);
    }

    let _ = std::fs::remove_dir_all(&state_dir);
}
