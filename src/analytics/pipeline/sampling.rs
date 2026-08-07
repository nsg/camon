//! Filmstrip sampling: which frames of a run are kept, how accumulators stay
//! bounded while they arrive, and the JPEG encoding of what survives.

use std::collections::HashMap;
use std::sync::Arc;

use super::framing::{NormalizedRect, RgbFrame};
use super::MotionSegment;

/// The JPEG thumbnails of one closed motion run, shared with the event that
/// carries them to warm storage.
pub(super) type Filmstrip = Arc<Vec<Vec<u8>>>;

/// Frames kept per event once the run closes.
///
/// Visible to the crate because storage counts the frames back off the store
/// and has to know where to stop looking: see
/// [`MAX_FILMSTRIP_FRAMES`](crate::storage::event_index::MAX_FILMSTRIP_FRAMES),
/// which is pinned to this number rather than agreeing with it by convention.
pub(crate) const FILMSTRIP_FRAMES: usize = 4;
/// Working size of an open run's accumulator. A run can last for hours, so
/// past this the strip is halved rather than grown.
pub(super) const FILMSTRIP_ACCUMULATOR_CAP: usize = 8;

/// Thumbnails extracted so far for the motion run that is currently open.
/// Frames arrive batch by batch and belong to the run as a whole, not to any
/// single segment, so they live here until the run closes.
#[derive(Default)]
pub(super) struct RunFilmstrip {
    pub(super) frames: Vec<Vec<u8>>,
}

/// Drop every second entry once `acc` outgrows `cap`. Halving keeps the whole
/// span covered at coarser spacing instead of truncating it to its beginning or
/// end, and the first entry always survives.
///
/// Used wherever the final length is not known while the entries arrive: an
/// open run lasts for as many batches as the motion does, and a segment holds
/// `sample_fps` frames per second of footage, which is config with no ceiling.
/// Where the count *is* known up front, thinning to it directly beats halving
/// down to it — see [`frames_per_segment`].
pub(super) fn halve_past<T>(acc: &mut Vec<T>, cap: usize) {
    if acc.len() <= cap {
        return;
    }
    let mut seen = 0;
    acc.retain(|_| {
        seen += 1;
        seen % 2 == 1
    });
}

impl RunFilmstrip {
    /// Add a batch's frames, halving the accumulator whenever it outgrows its
    /// cap.
    pub(super) fn push(&mut self, frames: Vec<Vec<u8>>) {
        self.frames.extend(frames);
        halve_past(&mut self.frames, FILMSTRIP_ACCUMULATOR_CAP);
    }

    /// Snapshot and reset, subsampled to at most [`FILMSTRIP_FRAMES`] frames
    /// spread from the first to the last. `None` when nothing was extracted.
    pub(super) fn take(&mut self) -> Option<Filmstrip> {
        let frames = std::mem::take(&mut self.frames);
        if frames.is_empty() {
            return None;
        }
        Some(Arc::new(subsample_filmstrip(frames)))
    }
}

fn subsample_filmstrip(frames: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let n = frames.len();
    if n <= FILMSTRIP_FRAMES {
        return frames;
    }
    let picks = [0, n / 3, 2 * n / 3, n - 1];
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, frame)| frame)
        .collect()
}

pub(super) fn sample_indices(len: usize) -> Vec<usize> {
    if len <= 4 {
        (0..len).collect()
    } else {
        vec![0, len / 3, 2 * len / 3, len - 1]
    }
}

/// Frames kept out of one segment's decode, given how many segments the run
/// contributes. [`sample_indices`] has already spread those segments over the
/// run and the final pick is positional — it never compares pixels — so one
/// frame per segment is all [`subsample_tagged`] strictly needs; the spare is
/// what keeps a segment that decoded short, or a pipe running a frame behind
/// its segments, from costing the strip a picture it cannot backfill.
pub(super) fn frames_per_segment(segments: usize) -> usize {
    FILMSTRIP_FRAMES.div_ceil(segments.max(1)) + 1
}

/// Live raw frames one run may hold while its filmstrip is chosen. Not a policy
/// of its own — the largest [`frames_per_segment`] ever grants across the
/// segments [`sample_indices`] yields — but pinned so the peak stays a number
/// this file states rather than one a reader has to derive.
pub(super) const RUN_FRAME_ACCUMULATOR_CAP: usize = 9;

/// Keep at most `keep` of one segment's frames, spread from its first to its
/// last. Both ends are kept because they are the two the run's selection can
/// least afford to lose: the first frame is the segment's keyframe, the one the
/// crop tag was measured on, and the last is the furthest whatever moved has
/// travelled by the time the next segment starts.
///
/// The two picks over a whole run — [`subsample_filmstrip`] and
/// [`subsample_tagged`] — instead space themselves at `n/3` and land on the
/// last frame only by way of `n - 1`. They are picking moments out of an event,
/// where the exact endpoints carry nothing in particular; this is picking
/// frames out of one segment, where they carry the two things above.
pub(super) fn thin_evenly<T>(frames: Vec<T>, keep: usize) -> Vec<T> {
    let n = frames.len();
    if n <= keep {
        return frames;
    }
    let span = keep.saturating_sub(1).max(1);
    let picks: Vec<usize> = (0..keep).map(|k| k * (n - 1) / span).collect();
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, frame)| frame)
        .collect()
}

/// Decode the sampled segments of `run` and reduce them to the frames the event
/// filmstrip and the vision model get, each tagged with its own segment's crop.
///
/// `decode` is a parameter so the selection can be driven without an ffmpeg;
/// [`MotionAnalyzer::extract_run_frames`] passes the crop decoder.
///
/// No segment is ever materialised whole. Frames are thinned as they arrive,
/// through an accumulator [`halve_past`] holds just above `keep` — a segment
/// owns `sample_fps` frames per second and `sample_fps` has no configured
/// ceiling, so the live cost has to be a function of what is kept rather than
/// of what is decoded.
pub(super) fn sample_run_frames(
    run: &[MotionSegment],
    crops: &HashMap<u64, NormalizedRect>,
    width: usize,
    height: usize,
    mut decode: impl FnMut(&Arc<Vec<u8>>, u64, &mut dyn FnMut(Vec<u8>)),
) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
    let indices = sample_indices(run.len());
    let keep = frames_per_segment(indices.len());
    let mut all_frames: Vec<(RgbFrame, Option<NormalizedRect>)> =
        Vec::with_capacity(RUN_FRAME_ACCUMULATOR_CAP);

    for &idx in &indices {
        let seg = &run[idx];
        let crop = crops.get(&seg.seq).copied();
        // One frame past `keep`, because `halve_past` needs one in hand beyond
        // its cap to halve against, and one more transiently while it is held.
        // Stated once: the two must not drift apart, and every frame either
        // covers is 6 MB at the detection crop size. The width is invisible in
        // the result — the final thin lands on the same frames whatever it is —
        // so it is purely how much memory the decode is allowed to use.
        let reservoir = keep + 1;
        let mut held: Vec<Vec<u8>> = Vec::with_capacity(reservoir + 1);
        decode(&seg.data, seg.duration_ns, &mut |frame_data: Vec<u8>| {
            // The pipe delivers exact fixed-size frames; anything else is a
            // torn read from a dying ffmpeg and gets skipped.
            if frame_data.len() == width * height * 3 {
                held.push(frame_data);
                halve_past(&mut held, reservoir);
            }
        });
        all_frames.extend(thin_evenly(held, keep).into_iter().map(|data| {
            (
                RgbFrame {
                    data,
                    width,
                    height,
                },
                crop,
            )
        }));
    }

    subsample_tagged(all_frames)
}

pub(super) fn subsample_tagged(
    frames: Vec<(RgbFrame, Option<NormalizedRect>)>,
) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
    if frames.len() <= 4 {
        return frames;
    }
    let n = frames.len();
    // Moved out rather than indexed and cloned, as in [`subsample_filmstrip`]:
    // a kept frame is a whole raw RGB image, several megabytes at the detection
    // crop size.
    let picks = [0, n / 3, 2 * n / 3, n - 1];
    frames
        .into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, tagged)| tagged)
        .collect()
}

/// JPEG quality for frames sent to the vision model and served to the UI.
/// High enough that compression artifacts don't cost the model detections.
const JPEG_QUALITY: u8 = 90;

fn encode_jpeg_raw(
    data: &[u8],
    width: usize,
    height: usize,
    color: image::ExtendedColorType,
) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    encoder
        .encode(data, width as u32, height as u32, color)
        .ok()?;
    Some(buf)
}

/// Encode a raw RGB frame as JPEG.
pub(super) fn rgb_jpeg(frame: &RgbFrame) -> Option<Vec<u8>> {
    if frame.data.len() != frame.width * frame.height * 3 {
        return None;
    }
    encode_jpeg_raw(
        &frame.data,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgb8,
    )
}

/// Encode an 8-bit grayscale buffer (detector masks, background model) as
/// JPEG for the debug endpoints.
pub(super) fn gray_jpeg((data, w, h): (&[u8], usize, usize)) -> Option<Vec<u8>> {
    if w == 0 || h == 0 || data.len() != w * h {
        return None;
    }
    encode_jpeg_raw(data, w, h, image::ExtendedColorType::L8)
}
