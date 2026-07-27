//! Shared MPEG-TS packet parsing.
//!
//! Two consumers with very different lifetimes share this bit-twiddling:
//!
//! - the live segmenter ([`crate::camera`]) extracts a PES PTS per video
//!   packet while slicing the stream into GOP segments, and
//! - startup orphan recovery ([`crate::storage`]) trims a torn `.ts.tmp` file
//!   to its last intact packet and reads its first/last PTS to recompute the
//!   real duration of footage that survived a crash or power cut.
//!
//! Recovery is READ-ONLY reuse: nothing here may change behavior for the live
//! path (PTS extraction, segment durations, or the filename-stem convention).

/// Size of one MPEG-TS packet.
pub const TS_PACKET_SIZE: usize = 188;

/// Every MPEG-TS packet starts with this sync byte.
pub const SYNC_BYTE: u8 = 0x47;

/// PID reserved for null (stuffing) packets, so never an elementary stream.
const NULL_PID: u16 = 0x1FFF;

/// 33-bit PTS values wrap at this modulus (~26.5 hours at 90 kHz).
const PTS_MODULUS: u64 = 1 << 33;

/// Length of the longest prefix of `data` that consists of whole, plausible
/// TS packets: a multiple of 188 bytes where every packet starts with the
/// 0x47 sync byte.
///
/// Used to trim a torn tail write off an orphaned `.ts.tmp`: a partial final
/// packet, or garbage after a corrupted packet, must not poison the file. The
/// scan stops at the first bad sync byte, so anything after in-file corruption
/// is discarded along with the tail.
pub fn valid_prefix_len(data: &[u8]) -> usize {
    let mut end = 0;
    while end + TS_PACKET_SIZE <= data.len() && data[end] == SYNC_BYTE {
        end += TS_PACKET_SIZE;
    }
    end
}

/// Extract the 33-bit PTS (90 kHz units) from a TS packet that starts a PES
/// packet, or `None` if the payload is not a PES header carrying a PTS.
///
/// This is the exact parser the live segmenter uses on video packets; keep
/// behavior identical for 188-byte input.
pub fn extract_pes_pts(packet: &[u8]) -> Option<u64> {
    if packet.len() < TS_PACKET_SIZE {
        return None;
    }
    let has_adaptation = (packet[3] & 0x20) != 0;
    let has_payload = (packet[3] & 0x10) != 0;
    if !has_payload {
        return None;
    }

    let payload_start = if has_adaptation {
        5 + packet[4] as usize
    } else {
        4
    };

    // PES header: 0x00 0x00 0x01 stream_id
    if payload_start + 14 > TS_PACKET_SIZE {
        return None;
    }
    let p = &packet[payload_start..];
    if p[0] != 0x00 || p[1] != 0x00 || p[2] != 0x01 {
        return None;
    }

    // Check PTS_DTS_flags (bits 7-6 of byte 7)
    let pts_dts_flags = (p[7] >> 6) & 0x03;
    if pts_dts_flags < 2 {
        return None; // No PTS present
    }

    // Parse 33-bit PTS from 5 bytes (bytes 9-13)
    let pts = ((p[9] as u64 & 0x0E) << 29)
        | ((p[10] as u64) << 22)
        | ((p[11] as u64 & 0xFE) << 14)
        | ((p[12] as u64) << 7)
        | ((p[13] as u64) >> 1);

    Some(pts)
}

/// PTS of a single packet, but only when the payload_unit_start_indicator is
/// set (a PES header can only start there). Any elementary stream qualifies —
/// video and audio share the same 90 kHz clock, which is all recovery needs.
fn payload_unit_pts(packet: &[u8]) -> Option<u64> {
    if (packet[1] & 0x40) == 0 {
        return None; // no PUSI, cannot start a PES packet
    }
    extract_pes_pts(packet)
}

/// First PES PTS found in a buffer of whole TS packets.
pub fn first_pts(data: &[u8]) -> Option<u64> {
    data.chunks_exact(TS_PACKET_SIZE).find_map(payload_unit_pts)
}

/// Last PES PTS found in a buffer of whole TS packets.
pub fn last_pts(data: &[u8]) -> Option<u64> {
    data.chunks_exact(TS_PACKET_SIZE)
        .rev()
        .find_map(payload_unit_pts)
}

/// PID a TS packet belongs to. Short input yields no meaningful PID, so it maps
/// to the null PID, which no elementary stream uses.
pub fn packet_pid(packet: &[u8]) -> u16 {
    if packet.len() < TS_PACKET_SIZE {
        return NULL_PID;
    }
    (((packet[1] & 0x1F) as u16) << 8) | packet[2] as u16
}

/// Whether this packet starts a video PES packet (stream_id 0xE0-0xEF). Used
/// to find the video PID without parsing the PMT.
fn starts_video_pes(packet: &[u8]) -> bool {
    if packet.len() < TS_PACKET_SIZE {
        return false;
    }
    if (packet[1] & 0x40) == 0 || (packet[3] & 0x10) == 0 {
        return false; // no PUSI, or no payload at all
    }
    let payload_start = if (packet[3] & 0x20) != 0 {
        5 + packet[4] as usize
    } else {
        4
    };
    if payload_start + 4 > TS_PACKET_SIZE {
        return false;
    }
    let p = &packet[payload_start..];
    p[0] == 0x00 && p[1] == 0x00 && p[2] == 0x01 && (0xE0..=0xEF).contains(&p[3])
}

/// Whether the packet's adaptation field flags a random access point — a
/// keyframe, on a video PID.
///
/// The live segmenter cuts a new segment on exactly this predicate, and
/// [`keyframe_count`] counts the result, so the two must agree byte for byte:
/// they share this one definition rather than a comment promising they match.
pub fn has_random_access_indicator(packet: &[u8]) -> bool {
    if packet.len() < TS_PACKET_SIZE || (packet[3] & 0x20) == 0 {
        return false;
    }
    let adaptation_len = packet[4] as usize;
    adaptation_len > 0 && adaptation_len < 184 && (packet[5] & 0x40) != 0
}

/// Number of video keyframes in a buffer of whole TS packets, read from the
/// adaptation field's random_access_indicator — the same signal the live
/// segmenter cuts on, so a hot-buffer segment contains exactly one.
///
/// The count is restricted to the video PID because ffmpeg's muxer flags
/// *every* audio packet as a random access point; counting those would inflate
/// the keyframe count several-fold. `0` when the buffer holds no video PES at
/// all (an empty or garbled segment).
pub fn keyframe_count(data: &[u8]) -> usize {
    let packets = || data.chunks_exact(TS_PACKET_SIZE);
    let video_pid = match packets().find(|p| starts_video_pes(p)) {
        Some(packet) => packet_pid(packet),
        None => return 0,
    };
    packets()
        .filter(|p| packet_pid(p) == video_pid && has_random_access_indicator(p))
        .count()
}

/// Milliseconds between two 90 kHz PTS values, tolerating one wrap of the
/// 33-bit counter (which wraps about every 26.5 hours).
pub fn pts_delta_ms(first: u64, last: u64) -> u64 {
    let delta = last.wrapping_sub(first) & (PTS_MODULUS - 1);
    delta / 90
}

/// Builders for minimal-but-valid TS packets, shared by the unit tests here
/// and the orphan-recovery tests.
#[cfg(test)]
pub(crate) mod testutil {
    use super::{SYNC_BYTE, TS_PACKET_SIZE};

    /// A TS packet (PUSI set) whose payload is a PES header carrying `pts`.
    pub fn pes_packet(pid: u16, pts: u64) -> [u8; TS_PACKET_SIZE] {
        let mut p = [0xFFu8; TS_PACKET_SIZE];
        p[0] = SYNC_BYTE;
        p[1] = 0x40 | ((pid >> 8) as u8 & 0x1F); // PUSI + PID high bits
        p[2] = pid as u8;
        p[3] = 0x10; // payload only, continuity counter 0
                     // PES header at payload offset 4.
        p[4] = 0x00;
        p[5] = 0x00;
        p[6] = 0x01;
        p[7] = 0xE0; // video stream id
        p[8] = 0x00;
        p[9] = 0x00; // PES packet length (0 = unbounded)
        p[10] = 0x80; // marker bits
        p[11] = 0x80; // PTS_DTS_flags = '10' (PTS only)
        p[12] = 0x05; // PES header data length
                      // 33-bit PTS in 5 bytes with '0010' prefix and marker bits.
        p[13] = 0x21 | ((pts >> 29) as u8 & 0x0E);
        p[14] = (pts >> 22) as u8;
        p[15] = ((pts >> 14) as u8 & 0xFE) | 0x01;
        p[16] = (pts >> 7) as u8;
        p[17] = ((pts << 1) as u8) | 0x01;
        p
    }

    /// A PES packet carrying `stream_id` (0xE0 video, 0xC0 audio) whose
    /// adaptation field flags a random access point — how a muxer marks a
    /// keyframe.
    pub fn keyframe_packet(pid: u16, pts: u64, stream_id: u8) -> [u8; TS_PACKET_SIZE] {
        let mut p = pes_packet(pid, pts);
        // Insert a one-byte adaptation field ahead of the PES header, which
        // moves the payload from offset 4 to offset 6.
        p.copy_within(4..TS_PACKET_SIZE - 2, 6);
        p[3] = 0x30; // adaptation field + payload
        p[4] = 0x01; // adaptation field length
        p[5] = 0x40; // random_access_indicator
        p[9] = stream_id;
        p
    }

    /// A null packet (PID 0x1FFF) — valid TS, no PES payload.
    pub fn null_packet() -> [u8; TS_PACKET_SIZE] {
        let mut p = [0xFFu8; TS_PACKET_SIZE];
        p[0] = SYNC_BYTE;
        p[1] = 0x1F;
        p[2] = 0xFF;
        p[3] = 0x10;
        p
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{keyframe_packet, null_packet, pes_packet};
    use super::*;

    const VIDEO: u8 = 0xE0;
    const AUDIO: u8 = 0xC0;

    #[test]
    fn extract_pes_pts_round_trips_synthetic_packet() {
        for pts in [0u64, 1, 90_000, PTS_MODULUS - 1] {
            let packet = pes_packet(0x100, pts);
            assert_eq!(extract_pes_pts(&packet), Some(pts), "pts {pts}");
        }
    }

    #[test]
    fn extract_pes_pts_rejects_non_pes_and_short_input() {
        assert_eq!(extract_pes_pts(&null_packet()), None);
        // No payload flag at all.
        let mut p = pes_packet(0x100, 42);
        p[3] = 0x00;
        assert_eq!(extract_pes_pts(&p), None);
        // PTS_DTS_flags say no PTS.
        let mut p = pes_packet(0x100, 42);
        p[11] = 0x00;
        assert_eq!(extract_pes_pts(&p), None);
        // Truncated input must not panic.
        assert_eq!(extract_pes_pts(&[0x47, 0x40, 0x00]), None);
    }

    #[test]
    fn valid_prefix_empty_and_garbage() {
        assert_eq!(valid_prefix_len(&[]), 0);
        assert_eq!(valid_prefix_len(&[0x00; 400]), 0);
    }

    #[test]
    fn valid_prefix_trims_torn_tail() {
        let mut data = Vec::new();
        data.extend_from_slice(&null_packet());
        data.extend_from_slice(&pes_packet(0x100, 1000));
        // Torn tail: a partial packet that even starts with a sync byte.
        data.push(SYNC_BYTE);
        data.extend_from_slice(&[0xAB; 50]);
        assert_eq!(valid_prefix_len(&data), 2 * TS_PACKET_SIZE);
    }

    #[test]
    fn valid_prefix_stops_at_bad_sync_byte() {
        let mut data = Vec::new();
        data.extend_from_slice(&pes_packet(0x100, 1000));
        // A full-length "packet" of garbage: everything after it is discarded
        // too, even a valid-looking packet.
        data.extend_from_slice(&[0x00; TS_PACKET_SIZE]);
        data.extend_from_slice(&pes_packet(0x100, 2000));
        assert_eq!(valid_prefix_len(&data), TS_PACKET_SIZE);
    }

    #[test]
    fn first_and_last_pts_skip_non_pes_packets() {
        let mut data = Vec::new();
        data.extend_from_slice(&null_packet());
        data.extend_from_slice(&pes_packet(0x100, 5_000));
        data.extend_from_slice(&null_packet());
        data.extend_from_slice(&pes_packet(0x100, 95_000));
        data.extend_from_slice(&null_packet());
        assert_eq!(first_pts(&data), Some(5_000));
        assert_eq!(last_pts(&data), Some(95_000));
        assert_eq!(first_pts(&null_packet()[..]), None);
        assert_eq!(last_pts(&[]), None);
    }

    #[test]
    fn keyframe_count_counts_one_per_segment_start() {
        // A hot-buffer segment: keyframe, then plain video packets.
        let mut data = Vec::new();
        data.extend_from_slice(&keyframe_packet(0x100, 0, VIDEO));
        for i in 1..5 {
            data.extend_from_slice(&pes_packet(0x100, i * 3_000));
        }
        assert_eq!(keyframe_count(&data), 1);
        // Two concatenated segments count as two.
        data.extend_from_slice(&keyframe_packet(0x100, 90_000, VIDEO));
        data.extend_from_slice(&pes_packet(0x100, 93_000));
        assert_eq!(keyframe_count(&data), 2);
    }

    #[test]
    fn keyframe_count_ignores_audio_random_access_points() {
        // The muxer flags every audio packet as a random access point; only
        // video keyframes may be counted.
        let mut data = Vec::new();
        data.extend_from_slice(&keyframe_packet(0x100, 0, VIDEO));
        for i in 0..8 {
            data.extend_from_slice(&keyframe_packet(0x101, i * 3_000, AUDIO));
        }
        assert_eq!(keyframe_count(&data), 1);
    }

    #[test]
    fn random_access_indicator_is_the_segmenter_predicate() {
        // What the live segmenter cuts on is what the analyzer counts.
        assert!(has_random_access_indicator(&keyframe_packet(
            0x100, 0, VIDEO
        )));
        // No adaptation field at all.
        assert!(!has_random_access_indicator(&pes_packet(0x100, 0)));
        // Adaptation field present, flag clear.
        let mut p = keyframe_packet(0x100, 0, VIDEO);
        p[5] = 0x00;
        assert!(!has_random_access_indicator(&p));
        // Adaptation field length out of range.
        let mut p = keyframe_packet(0x100, 0, VIDEO);
        p[4] = 0;
        assert!(!has_random_access_indicator(&p));
        p[4] = 200;
        assert!(!has_random_access_indicator(&p));
        // Short input must not panic on any of the packet accessors.
        assert!(!has_random_access_indicator(&[0x47, 0x40]));
        assert!(!starts_video_pes(&[0x47]));
        assert_eq!(packet_pid(&[0x47, 0x40]), NULL_PID);
    }

    #[test]
    fn keyframe_count_of_non_video_data_is_zero() {
        assert_eq!(keyframe_count(&[]), 0);
        assert_eq!(keyframe_count(&null_packet()[..]), 0);
        assert_eq!(keyframe_count(&[0u8; 4 * TS_PACKET_SIZE]), 0);
        // Video packets without a flagged random access point.
        assert_eq!(keyframe_count(&pes_packet(0x100, 5_000)[..]), 0);
    }

    #[test]
    fn pts_delta_handles_wraparound() {
        assert_eq!(pts_delta_ms(0, 90_000), 1_000);
        assert_eq!(pts_delta_ms(90_000, 90_000), 0);
        // Wrap: last restarted from zero after the 33-bit rollover.
        let just_before_wrap = PTS_MODULUS - 45_000; // 0.5s before wrap
        assert_eq!(pts_delta_ms(just_before_wrap, 45_000), 1_000);
    }
}
