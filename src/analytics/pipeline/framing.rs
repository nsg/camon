//! Rect and crop geometry: normalized regions, crop math and the detection
//! mask applied to every frame the vision model sees.

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

/// Where a frame sits in the picture it was cut from, in whole pixels of that picture.
#[derive(Clone, Copy)]
pub(super) struct Placement {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) source_width: usize,
    pub(super) source_height: usize,
}

impl Placement {
    /// A frame that is the whole picture.
    pub(super) fn whole(frame: &RgbFrame) -> Self {
        Self {
            x: 0,
            y: 0,
            source_width: frame.width,
            source_height: frame.height,
        }
    }
}

/// Cut a normalized region out of a frame with pure row copying. The region
/// is clamped to the frame bounds; a region that leaves no visible area
/// yields `None`. The cut falls on whole pixels, so it is returned with where
/// it really sits, which is not quite the region asked for.
pub(super) fn crop_frame(
    frame: &RgbFrame,
    region: &NormalizedRect,
) -> Option<(RgbFrame, Placement)> {
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
    let placement = Placement {
        x,
        y,
        source_width: frame.width,
        source_height: frame.height,
    };
    let cropped = RgbFrame {
        data,
        width: w,
        height: h,
    };
    Some((cropped, placement))
}

/// Each mask cell is split into this many boxes to draw the mask's edge with. On a 16:9 frame
/// they come out square.
const EDGE_BOXES_X: usize = 12;
const EDGE_BOXES_Y: usize = 9;
const BOX_COLS: usize = MASK_COLS * EDGE_BOXES_X;
const BOX_ROWS: usize = MASK_ROWS * EDGE_BOXES_Y;
/// How many boxes the edge reaches out of the painted cells, and into them: a third of a
/// cell's height, a quarter of its width.
const EDGE_REACH: i32 = 3;
/// The lightest grey a box inside the painted cells can take.
const EDGE_MAX_GREY: f32 = 55.0;

/// A box close enough to the mask's edge to be drawn as part of it.
struct EdgeBox {
    col: usize,
    row: usize,
    painted: bool,
    /// Distance, in boxes, to the nearest box on the other side of the edge.
    distance: f32,
}

fn cell_painted(mask: &[bool], col: i32, row: i32) -> Option<bool> {
    let inside = (0..MASK_COLS as i32).contains(&col) && (0..MASK_ROWS as i32).contains(&row);
    inside.then(|| mask[row as usize * MASK_COLS + col as usize])
}

fn box_painted(mask: &[bool], col: i32, row: i32) -> Option<bool> {
    let inside = (0..BOX_COLS as i32).contains(&col) && (0..BOX_ROWS as i32).contains(&row);
    inside.then(|| mask[row as usize / EDGE_BOXES_Y * MASK_COLS + col as usize / EDGE_BOXES_X])
}

/// The boxes within `EDGE_REACH` of the boundary between painted and unpainted cells. The
/// frame's border is not such a boundary, so a mask running off the frame stays solid there.
fn edge_boxes(mask: &[bool]) -> Vec<EdgeBox> {
    let reach = EDGE_REACH;
    let mut boxes = Vec::new();
    for cell_row in 0..MASK_ROWS as i32 {
        for cell_col in 0..MASK_COLS as i32 {
            let painted = mask[cell_row as usize * MASK_COLS + cell_col as usize];
            // The edge is narrower than a cell, so only a cell beside the boundary holds any.
            let beside_boundary = (-1..=1).any(|dy| {
                (-1..=1)
                    .any(|dx| cell_painted(mask, cell_col + dx, cell_row + dy) == Some(!painted))
            });
            if !beside_boundary {
                continue;
            }
            for row in cell_row * EDGE_BOXES_Y as i32..(cell_row + 1) * EDGE_BOXES_Y as i32 {
                for col in cell_col * EDGE_BOXES_X as i32..(cell_col + 1) * EDGE_BOXES_X as i32 {
                    let nearest = (-reach..=reach)
                        .flat_map(|dy| (-reach..=reach).map(move |dx| (dx, dy)))
                        .filter(|&(dx, dy)| box_painted(mask, col + dx, row + dy) == Some(!painted))
                        .map(|(dx, dy)| dx * dx + dy * dy)
                        .min();
                    if let Some(distance_sq) = nearest {
                        boxes.push(EdgeBox {
                            col: col as usize,
                            row: row as usize,
                            painted,
                            distance: (distance_sq as f32).sqrt(),
                        });
                    }
                }
            }
        }
    }
    boxes
}

/// Two fixed pseudo-random values in `0.0..=1.0` for a box. They depend only on where the box
/// is in the scene, so the edge looks the same in every frame and every crop.
fn box_noise(col: usize, row: usize) -> (f32, f32) {
    let mut h = ((row * BOX_COLS + col) as u32).wrapping_add(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    ((h & 0xffff) as f32 / 65535.0, (h >> 16) as f32 / 65535.0)
}

/// The pixels along one axis of a frame that grid division `index` of `count` covers, where
/// the grid spans `source` pixels and the frame shows `len` of them starting at `origin`. With
/// `outward` every pixel the division touches is included; without it the edges round to the
/// nearest pixel, so neighbouring divisions never share one.
fn pixel_span(
    index: usize,
    count: usize,
    (source, origin, len): (usize, usize, usize),
    outward: bool,
) -> (usize, usize) {
    let (lo, hi) = if outward {
        (
            index * source / count,
            ((index + 1) * source).div_ceil(count),
        )
    } else {
        let nearest = |edge: usize| (2 * edge * source + count) / (2 * count);
        (nearest(index), nearest(index + 1))
    };
    (
        lo.saturating_sub(origin).min(len),
        hi.saturating_sub(origin).min(len),
    )
}

/// The pixels of `frame` covered by the division at (`col`, `row`) of a `cols` by `rows` grid
/// laid over the whole picture.
fn pixel_rect(
    frame: &RgbFrame,
    placement: Placement,
    (col, row): (usize, usize),
    (cols, rows): (usize, usize),
    outward: bool,
) -> Option<(usize, usize, usize, usize)> {
    let across = (placement.source_width, placement.x, frame.width);
    let down = (placement.source_height, placement.y, frame.height);
    let (px0, px1) = pixel_span(col, cols, across, outward);
    let (py0, py1) = pixel_span(row, rows, down, outward);
    (px1 > px0 && py1 > py0).then_some((px0, px1, py0, py1))
}

fn for_each_byte(
    frame: &mut RgbFrame,
    (px0, px1, py0, py1): (usize, usize, usize, usize),
    mut f: impl FnMut(&mut u8),
) {
    for py in py0..py1 {
        let start = (py * frame.width + px0) * 3;
        let end = (py * frame.width + px1) * 3;
        frame.data[start..end].iter_mut().for_each(&mut f);
    }
}

/// Hide every pixel of `frame` that belongs to a painted detection-mask cell, and soften the
/// mask's edge with small boxes.
///
/// A painted cell is filled black, shading to dark grey boxes over the last `EDGE_REACH` boxes
/// before its edge; none of the picture survives anywhere inside it. Outside, the picture is
/// dimmed box by box over the same distance, fading out.
///
/// `placement` says where `frame` sits in the picture the mask was painted on. The mask is
/// laid out in that picture's whole pixels, so a crop hides exactly the pixels the full
/// frame would.
pub(super) fn apply_detection_mask(frame: &mut RgbFrame, placement: Placement, mask: &[bool]) {
    if mask.len() != MASK_CELLS || mask.iter().all(|&m| !m) {
        return;
    }
    let edge = edge_boxes(mask);
    let boxes = (BOX_COLS, BOX_ROWS);
    let reach = EDGE_REACH as f32;

    // Dim the picture outside the mask first. Everything after this rounds outward and
    // overwrites, so a pixel straddling a painted cell's edge ends up hidden.
    for b in edge.iter().filter(|b| !b.painted) {
        let fade = 1.0 - (b.distance - 0.5) / reach;
        if fade <= 0.0 {
            continue;
        }
        let dim = (fade + (box_noise(b.col, b.row).0 - 0.5) * 0.7).clamp(0.0, 1.0);
        let keep = ((1.0 - dim) * 256.0) as u32;
        if let Some(rect) = pixel_rect(frame, placement, (b.col, b.row), boxes, false) {
            for_each_byte(frame, rect, |v| *v = ((*v as u32 * keep) >> 8) as u8);
        }
    }

    for row in 0..MASK_ROWS {
        for col in 0..MASK_COLS {
            if !mask[row * MASK_COLS + col] {
                continue;
            }
            let cells = (MASK_COLS, MASK_ROWS);
            if let Some(rect) = pixel_rect(frame, placement, (col, row), cells, true) {
                for_each_byte(frame, rect, |v| *v = 0);
            }
        }
    }

    for b in edge.iter().filter(|b| b.painted) {
        let lightness = ((reach + 0.5 - b.distance) / (2.0 * reach)).clamp(0.0, 1.0);
        let grey = (EDGE_MAX_GREY * lightness * (0.3 + 0.7 * box_noise(b.col, b.row).1)) as u8;
        if grey == 0 {
            continue;
        }
        if let Some(rect) = pixel_rect(frame, placement, (b.col, b.row), boxes, true) {
            for_each_byte(frame, rect, |v| *v = grey);
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
