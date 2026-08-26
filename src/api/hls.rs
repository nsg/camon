use std::sync::Arc;

use bytes::Bytes;

use crate::buffer::HotBuffer;

const NANOS_PER_SEC: f64 = 1_000_000_000.0;

pub fn generate_playlist(
    buffer: &HotBuffer,
    tail_count: Option<usize>,
    segment_uri_suffix: &str,
) -> String {
    let segments = buffer.segments();
    let first_sequence = buffer.first_sequence();

    let skip = match tail_count {
        Some(n) if segments.len() > n => segments.len() - n,
        _ => 0,
    };
    let base_sequence = first_sequence + skip as u64;

    if segments.len() <= skip {
        return "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n"
            .to_string();
    }

    // RFC 8216 §4.3.3.1 compares TARGETDURATION with each EXTINF rounded to the nearest integer.
    let max_duration = segments
        .iter()
        .skip(skip)
        .map(|s| (s.duration_ns as f64 / NANOS_PER_SEC).round() as u64)
        .max()
        .unwrap_or(2)
        .max(1);

    let mut playlist = String::new();
    playlist.push_str("#EXTM3U\n");
    playlist.push_str("#EXT-X-VERSION:3\n");
    playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", max_duration));
    playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", base_sequence));

    // Previous segment's stamp and end: the gap test needs both, see
    // [`discontinuous`].
    let mut previous: Option<(u64, u64)> = None;
    for (i, segment) in segments.iter().skip(skip).enumerate() {
        let sequence = base_sequence + i as u64;
        let duration = segment.duration_ns as f64 / NANOS_PER_SEC;
        if let Some((prev_start, prev_end)) = previous {
            if discontinuous(prev_start, prev_end, segment.start_pts) {
                playlist.push_str("#EXT-X-DISCONTINUITY\n");
            }
        }
        previous = Some((
            segment.start_pts,
            segment.start_pts.saturating_add(segment.duration_ns),
        ));
        let secs = (segment.start_pts / 1_000_000_000) as i64;
        let millis = ((segment.start_pts % 1_000_000_000) / 1_000_000) as u32;
        let dt = format_datetime(secs, millis);
        playlist.push_str(&format!("#EXT-X-PROGRAM-DATE-TIME:{}\n", dt));
        playlist.push_str(&format!("#EXTINF:{:.3},\n", duration));
        playlist.push_str(&format!("segment/{}{}\n", sequence, segment_uri_suffix));
    }

    playlist
}

/// Whether a segment fails to continue the one before it — a gap between the previous end and
/// this stamp — so the player re-aligns its decoder.
fn discontinuous(prev_start: u64, prev_end: u64, start: u64) -> bool {
    const MAX_GAP_NS: u64 = 100_000_000;
    if prev_start == 0 && start == 0 {
        return false;
    }
    start.abs_diff(prev_end) > MAX_GAP_NS
}

/// Format unix timestamp as ISO 8601 for EXT-X-PROGRAM-DATE-TIME
fn format_datetime(secs: i64, millis: u32) -> String {
    const SECS_PER_DAY: i64 = 86400;
    const DAYS_FROM_UNIX_TO_0000: i64 = 719_468;

    let days = secs.div_euclid(SECS_PER_DAY) + DAYS_FROM_UNIX_TO_0000;
    let time_of_day = secs.rem_euclid(SECS_PER_DAY) as u32;

    // Civil date from day count (Euclidean affine algorithm)
    let era = days.div_euclid(146097);
    let doe = days.rem_euclid(146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, m, d, h, min, s, millis
    )
}

pub fn generate_segment(buffer: &HotBuffer, sequence: u64) -> Option<Arc<Vec<u8>>> {
    let segment = buffer.get_segment_by_sequence(sequence)?;
    // Cloning the Arc shares the bytes; the read lock drops without a copy.
    Some(Arc::clone(&segment.data))
}

/// A stored segment, borrowed as a byte slice so it can *be* a response body rather than be
/// copied into one.
struct SegmentBody(Arc<Vec<u8>>);

impl AsRef<[u8]> for SegmentBody {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Hand a stored segment to the HTTP layer as a body over the hot buffer's own allocation, no
/// copy at any point.
pub fn segment_body(data: Arc<Vec<u8>>) -> Bytes {
    Bytes::from_owner(SegmentBody(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::GopSegment;
    use crate::locks::LockExt;

    const SEC: u64 = 1_000_000_000;

    fn buffer_of(stamps: &[u64]) -> Arc<std::sync::RwLock<HotBuffer>> {
        let buffer = HotBuffer::new("cam".to_string(), 600);
        for &start_pts in stamps {
            let mut segment = GopSegment::new(start_pts);
            segment.duration_ns = 2 * SEC;
            segment.frame_count = 1;
            segment.data = Arc::new(vec![0x47; 188]);
            buffer.write_recover().push(segment);
        }
        buffer
    }

    fn buffer_with_durations(durations: &[u64]) -> Arc<std::sync::RwLock<HotBuffer>> {
        let buffer = HotBuffer::new("cam".to_string(), 600);
        let mut start_pts = 0_u64;
        for &duration_ns in durations {
            let mut segment = GopSegment::new(start_pts);
            segment.duration_ns = duration_ns;
            segment.frame_count = 1;
            segment.data = Arc::new(vec![0x47; 188]);
            buffer.write_recover().push(segment);
            start_pts = start_pts.saturating_add(duration_ns);
        }
        buffer
    }

    fn markers(playlist: &str) -> usize {
        playlist.matches("#EXT-X-DISCONTINUITY").count()
    }

    #[test]
    fn segments_stamped_by_an_unset_clock_are_not_all_marked_discontinuous() {
        let buffer = buffer_of(&[0, 0, 0, 0]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert_eq!(markers(&playlist), 0, "{playlist}");
    }

    #[test]
    fn the_segment_stamped_once_the_clock_lands_is_marked_discontinuous() {
        let buffer = buffer_of(&[0, 0, 1_700_000_000 * SEC, 1_700_000_002 * SEC]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert_eq!(markers(&playlist), 1, "{playlist}");
    }

    #[test]
    fn a_gap_between_stamped_segments_is_still_marked_discontinuous() {
        let buffer = buffer_of(&[1_700_000_000 * SEC, 1_700_000_060 * SEC]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert_eq!(markers(&playlist), 1, "{playlist}");
    }

    #[test]
    fn saturated_stamps_do_not_overflow_the_playlist() {
        let buffer = buffer_of(&[u64::MAX, u64::MAX]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert_eq!(markers(&playlist), 0, "{playlist}");
        assert!(playlist.contains("#EXTINF:2.000"), "{playlist}");
    }

    #[test]
    fn target_duration_rounds_to_the_nearest_second() {
        let buffer = buffer_with_durations(&[2_020_000_000, 2_000_000_000, 1_980_000_000]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert!(playlist.contains("#EXT-X-TARGETDURATION:2\n"), "{playlist}");

        let buffer = buffer_with_durations(&[2_600_000_000]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert!(playlist.contains("#EXT-X-TARGETDURATION:3\n"), "{playlist}");

        let buffer = buffer_with_durations(&[200_000_000]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert!(playlist.contains("#EXT-X-TARGETDURATION:1\n"), "{playlist}");
    }

    #[test]
    fn an_empty_segment_uri_suffix_preserves_the_existing_playlist_bytes() {
        let buffer = buffer_of(&[0]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "");
        assert_eq!(
            playlist,
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n\
             #EXT-X-MEDIA-SEQUENCE:0\n\
             #EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:00.000Z\n\
             #EXTINF:2.000,\nsegment/0\n"
        );
    }

    #[test]
    fn a_segment_uri_suffix_is_appended_to_every_segment() {
        let buffer = buffer_of(&[0, 0, 0]);
        let playlist = generate_playlist(&buffer.read_recover(), None, "?stream=sub");
        let segment_uris: Vec<&str> = playlist
            .lines()
            .filter(|line| line.starts_with("segment/"))
            .collect();
        assert_eq!(
            segment_uris,
            [
                "segment/0?stream=sub",
                "segment/1?stream=sub",
                "segment/2?stream=sub"
            ]
        );
    }

    #[test]
    fn generate_segment_shares_bytes_without_copying() {
        let buffer = HotBuffer::new("cam".to_string(), 60);
        let mut segment = GopSegment::new(0);
        segment.data = Arc::new(vec![1, 2, 3, 4]);
        segment.frame_count = 1;
        segment.duration_ns = 1_000_000;
        let stored = Arc::clone(&segment.data);
        buffer.write().unwrap().push(segment);

        let out = generate_segment(&buffer.read().unwrap(), 0).expect("segment present");
        assert!(Arc::ptr_eq(&out, &stored));
        assert_eq!(&*out, &[1, 2, 3, 4]);
    }

    #[test]
    fn the_served_body_is_the_buffers_own_allocation() {
        let buffer = HotBuffer::new("cam".to_string(), 60);
        let mut segment = GopSegment::new(0);
        segment.data = Arc::new(vec![0x47; 4096]);
        segment.frame_count = 1;
        segment.duration_ns = SEC;
        let stored = Arc::clone(&segment.data);
        buffer.write_recover().push(segment);

        let data = generate_segment(&buffer.read_recover(), 0).expect("segment present");
        let body = segment_body(data);

        assert_eq!(
            body.as_ptr(),
            stored.as_ptr(),
            "body was copied out of the buffer"
        );
        assert_eq!(body.len(), stored.len());
    }

    #[test]
    fn a_segment_evicted_while_it_is_being_served_still_arrives_whole() {
        let buffer = HotBuffer::new("cam".to_string(), 2);
        let mut first = GopSegment::new(0);
        first.data = Arc::new(vec![0xA5; 4096]);
        first.frame_count = 1;
        first.duration_ns = 2 * SEC;
        buffer.write_recover().push(first);

        let body = segment_body(generate_segment(&buffer.read_recover(), 0).expect("segment 0"));

        let mut second = GopSegment::new(2 * SEC);
        second.data = Arc::new(vec![0x5A; 4096]);
        second.frame_count = 1;
        second.duration_ns = 2 * SEC;
        buffer.write_recover().push(second);

        assert!(
            generate_segment(&buffer.read_recover(), 0).is_none(),
            "segment 0 should have aged out of the window"
        );
        assert_eq!(body.len(), 4096);
        assert!(body.iter().all(|&b| b == 0xA5));
    }
}
