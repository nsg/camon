//! Rect and crop geometry: normalized regions, crop math and the detection
//! mask blackout applied to every frame the vision model sees.

use crate::analytics::motion::MotionBox;
use crate::analytics::motion_settings::{MASK_CELLS, MASK_COLS, MASK_ROWS};
use crate::config::{DetectionCrop, DetectionFramingConfig};

#[derive(Clone, Copy)]
pub(super) struct NormalizedRect {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) w: f32,
    pub(super) h: f32,
}

/// The whole frame in normalized coordinates. Used as the crop region for the
/// full-frame fallback (a frame with no motion crop, or a lighting-driven crop
/// that spans the entire frame) so the detection mask is applied consistently.
pub(super) const FULL_FRAME: NormalizedRect = NormalizedRect {
    x: 0.0,
    y: 0.0,
    w: 1.0,
    h: 1.0,
};

pub(super) fn normalize_rect(r: MotionBox, frame_w: i32, frame_h: i32) -> NormalizedRect {
    NormalizedRect {
        x: r.x as f32 / frame_w as f32,
        y: r.y as f32 / frame_h as f32,
        w: r.width as f32 / frame_w as f32,
        h: r.height as f32 / frame_h as f32,
    }
}

pub(super) fn union_rects_padded(
    rects: &[NormalizedRect],
    padding: f32,
    min_fraction: f32,
) -> Option<NormalizedRect> {
    if rects.is_empty() {
        return None;
    }
    let min_x = rects.iter().map(|r| r.x).fold(f32::MAX, f32::min);
    let min_y = rects.iter().map(|r| r.y).fold(f32::MAX, f32::min);
    let max_x = rects.iter().map(|r| r.x + r.w).fold(0.0f32, f32::max);
    let max_y = rects.iter().map(|r| r.y + r.h).fold(0.0f32, f32::max);

    let w = (max_x - min_x).max(min_fraction);
    let h = (max_y - min_y).max(min_fraction);
    let pad_x = w * padding;
    let pad_y = h * padding;

    let x = (min_x - pad_x).max(0.0);
    let y = (min_y - pad_y).max(0.0);
    Some(NormalizedRect {
        x,
        y,
        w: (w + 2.0 * pad_x).min(1.0 - x),
        h: (h + 2.0 * pad_y).min(1.0 - y),
    })
}

/// Grow the smaller normalized side around the centre until both sides match, then slide the
/// square back inside the unit frame.
pub(super) fn square_up(r: NormalizedRect) -> NormalizedRect {
    let side = r.w.max(r.h).min(1.0);
    let x = (r.x + (r.w - side) / 2.0).clamp(0.0, 1.0 - side);
    let y = (r.y + (r.h - side) / 2.0).clamp(0.0, 1.0 - side);
    NormalizedRect {
        x,
        y,
        w: side,
        h: side,
    }
}

/// The region of the frame the detector is shown for a segment with these motion rects, or
/// `None` when it had none.
pub(super) fn detection_region(
    rects: &[NormalizedRect],
    framing: &DetectionFramingConfig,
) -> Option<NormalizedRect> {
    if rects.is_empty() {
        return None;
    }
    if framing.crop == DetectionCrop::Full {
        return Some(FULL_FRAME);
    }
    let union = union_rects_padded(rects, framing.padding, framing.min_fraction)?;
    Some(if framing.preserve_aspect {
        square_up(union)
    } else {
        union
    })
}

pub(super) fn union_two_rects(a: NormalizedRect, b: NormalizedRect) -> NormalizedRect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let max_x = (a.x + a.w).max(b.x + b.w);
    let max_y = (a.y + a.h).max(b.y + b.h);
    NormalizedRect {
        x,
        y,
        w: max_x - x,
        h: max_y - y,
    }
}

/// A raw 8-bit RGB frame (3 bytes per pixel, row-major, no padding), as
/// produced by the crop decoder's ffmpeg pipe.
#[derive(Clone)]
pub(super) struct RgbFrame {
    pub(super) data: Vec<u8>,
    pub(super) width: usize,
    pub(super) height: usize,
}

/// Cut a normalized region out of a frame with pure row copying. The region
/// is clamped to the frame bounds; a region that leaves no visible area
/// yields `None`.
pub(super) fn crop_frame(frame: &RgbFrame, region: &NormalizedRect) -> Option<RgbFrame> {
    let cols = frame.width as i32;
    let rows = frame.height as i32;
    if cols == 0 || rows == 0 {
        return None;
    }

    let x = ((region.x * cols as f32) as i32).max(0);
    let y = ((region.y * rows as f32) as i32).max(0);
    let w = ((region.w * cols as f32) as i32).min(cols - x);
    let h = ((region.h * rows as f32) as i32).min(rows - y);
    if w <= 0 || h <= 0 {
        return None;
    }

    let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        let start = (row * frame.width + x) * 3;
        data.extend_from_slice(&frame.data[start..start + w * 3]);
    }
    Some(RgbFrame {
        data,
        width: w,
        height: h,
    })
}

/// Black out (set to RGB black) every pixel of `frame` that belongs to a painted detection-mask
/// cell.
pub(super) fn apply_detection_mask(frame: &mut RgbFrame, crop: NormalizedRect, mask: &[bool]) {
    if mask.len() != MASK_CELLS
        || mask.iter().all(|&m| !m)
        || crop.w <= 0.0
        || crop.h <= 0.0
        || frame.width == 0
        || frame.height == 0
    {
        return;
    }
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    for row in 0..MASK_ROWS {
        for col in 0..MASK_COLS {
            if !mask[row * MASK_COLS + col] {
                continue;
            }
            // Cell rectangle in full-frame normalized coordinates.
            let cx0 = col as f32 / MASK_COLS as f32;
            let cx1 = (col + 1) as f32 / MASK_COLS as f32;
            let cy0 = row as f32 / MASK_ROWS as f32;
            let cy1 = (row + 1) as f32 / MASK_ROWS as f32;
            // Intersect with the crop region.
            let ix0 = cx0.max(crop.x);
            let ix1 = cx1.min(crop.x + crop.w);
            let iy0 = cy0.max(crop.y);
            let iy1 = cy1.min(crop.y + crop.h);
            if ix1 <= ix0 || iy1 <= iy0 {
                continue;
            }
            // Translate into crop-local pixel coordinates, rounding outward.
            let px0 = ((((ix0 - crop.x) / crop.w) * fw).floor() as i64).clamp(0, frame.width as i64)
                as usize;
            let px1 = ((((ix1 - crop.x) / crop.w) * fw).ceil() as i64).clamp(0, frame.width as i64)
                as usize;
            let py0 = ((((iy0 - crop.y) / crop.h) * fh).floor() as i64)
                .clamp(0, frame.height as i64) as usize;
            let py1 = ((((iy1 - crop.y) / crop.h) * fh).ceil() as i64).clamp(0, frame.height as i64)
                as usize;
            for py in py0..py1 {
                let start = (py * frame.width + px0) * 3;
                let end = (py * frame.width + px1) * 3;
                for b in &mut frame.data[start..end] {
                    *b = 0;
                }
            }
        }
    }
}

const OUTLINE_RGB: [u8; 3] = [255, 0, 0];

/// Draw `rect` (full-frame normalized) as an unfilled red outline on `frame`, which shows the
/// `crop` region of the full frame. Whatever falls outside the frame is clipped away.
pub(super) fn draw_outline(frame: &mut RgbFrame, crop: NormalizedRect, rect: NormalizedRect) {
    if crop.w <= 0.0 || crop.h <= 0.0 || frame.width == 0 || frame.height == 0 {
        return;
    }
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    let x0 = (((rect.x - crop.x) / crop.w) * fw).floor() as i64;
    let x1 = (((rect.x + rect.w - crop.x) / crop.w) * fw).ceil() as i64;
    let y0 = (((rect.y - crop.y) / crop.h) * fh).floor() as i64;
    let y1 = (((rect.y + rect.h - crop.y) / crop.h) * fh).ceil() as i64;
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let stroke = ((0.004 * frame.width.min(frame.height) as f32).round() as i64).max(2);
    fill_rgb(frame, x0, y0, x1, y0 + stroke);
    fill_rgb(frame, x0, y1 - stroke, x1, y1);
    fill_rgb(frame, x0, y0, x0 + stroke, y1);
    fill_rgb(frame, x1 - stroke, y0, x1, y1);
}

fn fill_rgb(frame: &mut RgbFrame, x0: i64, y0: i64, x1: i64, y1: i64) {
    let (w, h) = (frame.width as i64, frame.height as i64);
    let (x0, x1) = (x0.clamp(0, w) as usize, x1.clamp(0, w) as usize);
    let (y0, y1) = (y0.clamp(0, h) as usize, y1.clamp(0, h) as usize);
    for py in y0..y1 {
        let row = py * frame.width;
        for px in x0..x1 {
            let i = (row + px) * 3;
            frame.data[i..i + 3].copy_from_slice(&OUTLINE_RGB);
        }
    }
}
