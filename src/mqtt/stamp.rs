//! The capture time drawn into the top-left corner of the stills published to Home Assistant,
//! as local `HH:MM:SS`. Only digits and `:` are ever drawn, so a 5x7 bitmap font of eleven
//! glyphs covers it without a font-rendering crate.

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;

fn glyph(c: u8) -> [u8; GLYPH_H] {
    match c {
        b'0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        b'1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        b'2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        b'3' => [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
        b'4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        b'5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        b'6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        b'7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        b'8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        b'9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        b':' => [
            0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000,
        ],
        _ => [0; GLYPH_H],
    }
}

/// `captured_ns` (wall clock nanoseconds since the epoch) as `HH:MM:SS` in the process's local
/// time zone — `TZ`, else `/etc/localtime`, else UTC.
fn clock_text(captured_ns: u64) -> Option<String> {
    let secs = libc::time_t::try_from(captured_ns / 1_000_000_000).ok()?;
    // SAFETY: `tm` is plain data that `localtime_r` fills in; both pointers are valid for the
    // call and the reentrant variant keeps no reference to either.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&secs, &mut tm) }.is_null() {
        return None;
    }
    Some(format!(
        "{:02}:{:02}:{:02}",
        tm.tm_hour, tm.tm_min, tm.tm_sec
    ))
}

/// Draw the capture time onto a packed RGB24 frame. Scaled with the frame's height so it reads
/// the same on a 720p snapshot and on a small crop, and clipped rather than skipped when the
/// frame is narrower than the label.
pub(crate) fn stamp_rgb(rgb: &mut [u8], width: usize, height: usize, captured_ns: u64) {
    if rgb.len() < width * height * 3 {
        return;
    }
    if let Some(text) = clock_text(captured_ns) {
        draw_label(rgb, width, height, text.as_bytes());
    }
}

/// [`stamp_rgb`] for a JPEG: decode, stamp, re-encode. `None` when the bytes do not decode.
pub(crate) fn stamp_jpeg(jpeg: &[u8], captured_ns: u64, quality: u8) -> Option<Vec<u8>> {
    let mut frame = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)
        .ok()?
        .to_rgb8();
    let (width, height) = frame.dimensions();
    stamp_rgb(&mut frame, width as usize, height as usize, captured_ns);
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
        .encode(&frame, width, height, image::ExtendedColorType::Rgb8)
        .ok()?;
    Some(out)
}

fn draw_label(rgb: &mut [u8], width: usize, height: usize, text: &[u8]) {
    let scale = (height / 180).max(1);
    let pad = 2 * scale;
    let advance = (GLYPH_W + 1) * scale;
    let box_w = text.len() * advance - scale + 2 * pad;
    let box_h = GLYPH_H * scale + 2 * pad;
    let (left, top) = (pad, pad);

    // The live view's label is black at 70% opacity; the same here keeps the time readable over
    // a bright sky without hiding what is behind it.
    for y in top..(top + box_h).min(height) {
        for x in left..(left + box_w).min(width) {
            let px = (y * width + x) * 3;
            for channel in &mut rgb[px..px + 3] {
                *channel = (u16::from(*channel) * 3 / 10) as u8;
            }
        }
    }

    for (i, &c) in text.iter().enumerate() {
        let glyph_left = left + pad + i * advance;
        for (row, bits) in glyph(c).iter().enumerate() {
            for col in 0..GLYPH_W {
                if bits >> (GLYPH_W - 1 - col) & 1 == 0 {
                    continue;
                }
                let x0 = glyph_left + col * scale;
                let y0 = top + pad + row * scale;
                for y in y0..(y0 + scale).min(height) {
                    for x in x0..(x0 + scale).min(width) {
                        let px = (y * width + x) * 3;
                        rgb[px..px + 3].fill(255);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAY: u8 = 128;

    #[test]
    fn the_label_darkens_the_top_left_corner_and_leaves_the_rest_alone() {
        let (width, height) = (320, 240);
        let mut rgb = vec![GRAY; width * height * 3];
        draw_label(&mut rgb, width, height, b"12:34:56");

        let at = |x: usize, y: usize| rgb[(y * width + x) * 3];
        // Inside the box but between glyphs: darkened, not drawn on.
        assert_eq!(at(2, 2), (u16::from(GRAY) * 3 / 10) as u8);
        assert!(rgb[..width * 3 * 4].iter().any(|&v| v != GRAY));
        assert!((0..width).all(|x| at(x, height - 1) == GRAY));
        assert!((0..height).all(|y| at(width - 1, y) == GRAY));
        // Some glyph pixel was lit.
        assert!(rgb.chunks(3).any(|px| px == [255, 255, 255]));
    }

    #[test]
    fn a_frame_smaller_than_the_label_is_clipped_not_overrun() {
        let (width, height) = (12, 6);
        let mut rgb = vec![GRAY; width * height * 3];
        draw_label(&mut rgb, width, height, b"12:34:56");
        assert_eq!(rgb.len(), width * height * 3);
    }
}
