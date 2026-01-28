use crate::buffer::HotBuffer;

const NANOS_PER_SEC: f64 = 1_000_000_000.0;

pub fn generate_playlist(buffer: &HotBuffer) -> String {
    let segments = buffer.segments();
    let first_sequence = buffer.first_sequence();

    if segments.is_empty() {
        return "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n"
            .to_string();
    }

    let max_duration = segments
        .iter()
        .map(|s| (s.duration_ns as f64 / NANOS_PER_SEC).ceil() as u64)
        .max()
        .unwrap_or(2);

    let mut playlist = String::new();
    playlist.push_str("#EXTM3U\n");
    playlist.push_str("#EXT-X-VERSION:3\n");
    playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", max_duration));
    playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", first_sequence));

    for (i, segment) in segments.iter().enumerate() {
        let sequence = first_sequence + i as u64;
        let duration = segment.duration_ns as f64 / NANOS_PER_SEC;
        // Mark discontinuity for each segment since they have independent timestamps
        if i > 0 {
            playlist.push_str("#EXT-X-DISCONTINUITY\n");
        }
        playlist.push_str(&format!("#EXTINF:{:.3},\n", duration));
        playlist.push_str(&format!("segment/{}\n", sequence));
    }

    playlist
}

pub fn generate_segment(buffer: &HotBuffer, sequence: u64) -> Option<Vec<u8>> {
    let segment = buffer.get_segment_by_sequence(sequence)?;
    // Return raw MPEG-TS data directly - already properly formatted with PAT/PMT
    Some(segment.data.clone())
}
