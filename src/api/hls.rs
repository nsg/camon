use std::sync::Arc;

use crate::buffer::HotBuffer;

const NANOS_PER_SEC: f64 = 1_000_000_000.0;

pub fn generate_playlist(buffer: &HotBuffer, tail_count: Option<usize>) -> String {
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

    let max_duration = segments
        .iter()
        .skip(skip)
        .map(|s| (s.duration_ns as f64 / NANOS_PER_SEC).ceil() as u64)
        .max()
        .unwrap_or(2);

    let mut playlist = String::new();
    playlist.push_str("#EXTM3U\n");
    playlist.push_str("#EXT-X-VERSION:3\n");
    playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", max_duration));
    playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", base_sequence));

    let mut prev_end_pts: Option<u64> = None;
    for (i, segment) in segments.iter().skip(skip).enumerate() {
        let sequence = base_sequence + i as u64;
        let duration = segment.duration_ns as f64 / NANOS_PER_SEC;
        if let Some(prev_end) = prev_end_pts {
            let gap = segment.start_pts.abs_diff(prev_end);
            if gap > 100_000_000 {
                playlist.push_str("#EXT-X-DISCONTINUITY\n");
            }
        }
        prev_end_pts = Some(segment.start_pts + segment.duration_ns);
        let secs = (segment.start_pts / 1_000_000_000) as i64;
        let millis = ((segment.start_pts % 1_000_000_000) / 1_000_000) as u32;
        let dt = format_datetime(secs, millis);
        playlist.push_str(&format!("#EXT-X-PROGRAM-DATE-TIME:{}\n", dt));
        playlist.push_str(&format!("#EXTINF:{:.3},\n", duration));
        playlist.push_str(&format!("segment/{}\n", sequence));
    }

    playlist
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
    // Return raw MPEG-TS data directly - already properly formatted with PAT/PMT.
    // Cloning the Arc shares the bytes, so the read lock drops without a byte copy.
    Some(Arc::clone(&segment.data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::GopSegment;

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
}
