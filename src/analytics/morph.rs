//! Binary morphology on 8-bit masks: elliptical structuring elements and
//! opening (erosion followed by dilation), matching OpenCV's
//! `getStructuringElement(MORPH_ELLIPSE, ..)` + `morphologyEx(MORPH_OPEN, ..)`
//! behavior on 0/255 masks, including its border convention (out-of-bounds
//! pixels never constrain erosion and never contribute to dilation).

/// A structuring element stored as one horizontal span per kernel row:
/// `(dy, dx_start, dx_end)` offsets relative to the anchor (kernel center),
/// with `dx_end` exclusive.
pub struct StructuringElement {
    rows: Vec<(i32, i32, i32)>,
}

impl StructuringElement {
    /// Elliptical kernel of `size` x `size` (odd, ≥ 1), generated with the
    /// same integer rasterization OpenCV uses, so kernel shapes match exactly.
    pub fn ellipse(size: i32) -> Self {
        let size = size.max(1) | 1;
        let r = size / 2;
        let inv_r2 = if r > 0 { 1.0 / f64::from(r * r) } else { 0.0 };

        let mut rows = Vec::with_capacity(size as usize);
        for i in 0..size {
            let dy = i - r;
            let dx = (f64::from(r) * (f64::from(r * r - dy * dy) * inv_r2).sqrt()).round_ties_even()
                as i32;
            let j1 = (r - dx).max(0);
            let j2 = (r + dx + 1).min(size);
            if j1 < j2 {
                rows.push((dy, j1 - r, j2 - r));
            }
        }
        Self { rows }
    }

    #[cfg(test)]
    fn contains(&self, dy: i32, dx: i32) -> bool {
        self.rows
            .iter()
            .any(|&(ry, x1, x2)| ry == dy && dx >= x1 && dx < x2)
    }
}

/// Morphological opening: erosion then dilation with the same element.
/// Removes foreground features smaller than the structuring element (isolated
/// noise pixels) while preserving the shape of larger regions.
///
/// `tmp` and `dst` are reusable buffers; both are resized to `w * h`.
pub fn open(
    src: &[u8],
    w: usize,
    h: usize,
    se: &StructuringElement,
    tmp: &mut Vec<u8>,
    dst: &mut Vec<u8>,
) {
    erode(src, w, h, se, tmp);
    dilate(tmp, w, h, se, dst);
}

/// Binary erosion: a pixel stays foreground only if every in-bounds pixel
/// under the structuring element is foreground.
pub fn erode(src: &[u8], w: usize, h: usize, se: &StructuringElement, dst: &mut Vec<u8>) {
    debug_assert!(src.len() >= w * h);
    dst.clear();
    dst.resize(w * h, 0);
    let (wi, hi) = (w as i32, h as i32);

    for y in 0..hi {
        'pixels: for x in 0..wi {
            let idx = (y * wi + x) as usize;
            if src[idx] == 0 {
                continue; // the anchor is always inside an elliptical element
            }
            for &(dy, dx1, dx2) in &se.rows {
                let yy = y + dy;
                if yy < 0 || yy >= hi {
                    continue; // out of bounds never constrains erosion
                }
                let x1 = (x + dx1).max(0);
                let x2 = (x + dx2).min(wi);
                let row = (yy * wi) as usize;
                for xx in x1..x2 {
                    if src[row + xx as usize] == 0 {
                        continue 'pixels;
                    }
                }
            }
            dst[idx] = 255;
        }
    }
}

/// Binary dilation: a pixel becomes foreground if any in-bounds pixel under
/// the (symmetric) structuring element is foreground.
pub fn dilate(src: &[u8], w: usize, h: usize, se: &StructuringElement, dst: &mut Vec<u8>) {
    debug_assert!(src.len() >= w * h);
    dst.clear();
    dst.resize(w * h, 0);
    let (wi, hi) = (w as i32, h as i32);

    for y in 0..hi {
        'pixels: for x in 0..wi {
            let idx = (y * wi + x) as usize;
            if src[idx] != 0 {
                dst[idx] = 255;
                continue;
            }
            for &(dy, dx1, dx2) in &se.rows {
                let yy = y + dy;
                if yy < 0 || yy >= hi {
                    continue; // out of bounds never contributes to dilation
                }
                let x1 = (x + dx1).max(0);
                let x2 = (x + dx2).min(wi);
                let row = (yy * wi) as usize;
                for xx in x1..x2 {
                    if src[row + xx as usize] != 0 {
                        dst[idx] = 255;
                        continue 'pixels;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_mask(src: &[u8], w: usize, h: usize, size: i32) -> Vec<u8> {
        let se = StructuringElement::ellipse(size);
        let mut tmp = Vec::new();
        let mut dst = Vec::new();
        open(src, w, h, &se, &mut tmp, &mut dst);
        dst
    }

    #[test]
    fn ellipse_5_matches_opencv_shape() {
        // cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)):
        //   0 0 1 0 0
        //   1 1 1 1 1
        //   1 1 1 1 1
        //   1 1 1 1 1
        //   0 0 1 0 0
        let se = StructuringElement::ellipse(5);
        let expected = [
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(
                    se.contains(i as i32 - 2, j as i32 - 2),
                    v == 1,
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn ellipse_3_is_cross() {
        let se = StructuringElement::ellipse(3);
        let expected = [[0, 1, 0], [1, 1, 1], [0, 1, 0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(se.contains(i as i32 - 1, j as i32 - 1), v == 1);
            }
        }
    }

    #[test]
    fn opening_removes_salt_noise() {
        let (w, h) = (32, 24);
        let mut mask = vec![0u8; w * h];
        for &(x, y) in &[(3, 3), (10, 17), (30, 2), (16, 12)] {
            mask[y * w + x] = 255;
        }
        let out = open_mask(&mask, w, h, 5);
        assert!(out.iter().all(|&v| v == 0), "isolated pixels must vanish");
    }

    #[test]
    fn opening_preserves_large_blob_bbox() {
        let (w, h) = (64, 48);
        let mut mask = vec![0u8; w * h];
        for y in 10..30 {
            for x in 20..40 {
                mask[y * w + x] = 255;
            }
        }
        let out = open_mask(&mask, w, h, 5);
        // Opening rounds the corners but keeps the extent of a large square.
        assert_eq!(out[10 * w + 30], 255, "top edge center survives");
        assert_eq!(out[20 * w + 20], 255, "left edge center survives");
        assert_eq!(out[29 * w + 39 - 2], 255, "bottom edge survives");
        assert_eq!(out[9 * w + 30], 0, "no growth above");
        assert_eq!(out[10 * w + 19], 0, "no growth left");
        let area: usize = out.iter().filter(|&&v| v != 0).count();
        assert!(
            (380..=400).contains(&area),
            "area {area} ≈ 400 minus corners"
        );
    }

    #[test]
    fn opening_keeps_full_frame_solid() {
        let (w, h) = (20, 16);
        let mask = vec![255u8; w * h];
        let out = open_mask(&mask, w, h, 5);
        assert!(
            out.iter().all(|&v| v == 255),
            "border must not erode a solid mask"
        );
    }
}
