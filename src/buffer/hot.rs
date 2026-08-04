use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use super::GopSegment;
use crate::shutdown::Watermark;

const NANOS_PER_SEC: u64 = 1_000_000_000;

pub struct HotBuffer {
    segments: VecDeque<GopSegment>,
    max_duration_ns: u64,
    current_duration_ns: u64,
    camera_id: String,
    first_sequence: u64,
    /// The camera's terminal watermark, or `None` while it is still producing.
    /// See [`HotBuffer::seal`].
    terminal: Option<Watermark>,
}

impl HotBuffer {
    pub fn new(camera_id: String, max_duration_secs: u64) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            segments: VecDeque::new(),
            max_duration_ns: max_duration_secs * NANOS_PER_SEC,
            current_duration_ns: 0,
            camera_id,
            first_sequence: 0,
            terminal: None,
        }))
    }

    pub fn push(&mut self, segment: GopSegment) {
        tracing::trace!(
            camera = %self.camera_id,
            frames = segment.frame_count,
            duration_ms = segment.duration_ns / 1_000_000,
            data_size = segment.data.len(),
            "pushing GOP segment"
        );

        self.current_duration_ns += segment.duration_ns;
        self.segments.push_back(segment);

        self.evict_old();
    }

    /// Drop segments that aged out of the retention window. Events are
    /// persisted the moment their motion run ends, so eviction only frees RAM.
    fn evict_old(&mut self) {
        while self.current_duration_ns > self.max_duration_ns {
            if let Some(old) = self.segments.pop_front() {
                self.current_duration_ns = self.current_duration_ns.saturating_sub(old.duration_ns);
                self.first_sequence += 1;
                tracing::trace!(
                    camera = %self.camera_id,
                    evicted_duration_ms = old.duration_ns / 1_000_000,
                    first_sequence = self.first_sequence,
                    "evicted old segment"
                );
            } else {
                break;
            }
        }
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn current_duration_secs(&self) -> f64 {
        self.current_duration_ns as f64 / NANOS_PER_SEC as f64
    }

    pub fn segments(&self) -> &VecDeque<GopSegment> {
        &self.segments
    }

    /// Get segment by absolute sequence number (accounts for evicted segments)
    pub fn get_segment_by_sequence(&self, sequence: u64) -> Option<&GopSegment> {
        if sequence < self.first_sequence {
            return None; // Already evicted
        }
        let index = (sequence - self.first_sequence) as usize;
        self.segments.get(index)
    }

    /// Get the sequence number of the first segment in the buffer
    pub fn first_sequence(&self) -> u64 {
        self.first_sequence
    }

    /// Get the sequence number of the last segment in the buffer (exclusive)
    pub fn last_sequence(&self) -> u64 {
        self.first_sequence + self.segments.len() as u64
    }

    /// Publish this camera's terminal watermark: the camera thread has stopped,
    /// and the sequence one past its last segment is now final. Returns it.
    ///
    /// This is the promise phase 2 of the stop is built on (see
    /// [`crate::shutdown`]) — everything below the watermark is in this buffer
    /// and a consumer that reaches it has seen everything the camera produced.
    /// It is therefore published *after* the camera thread is joined, by the
    /// drain rather than by the camera, so that no push can slip in behind it.
    pub fn seal(&mut self) -> Watermark {
        self.publish_watermark(false)
    }

    /// Publish a watermark for a camera that did *not* stop — phase 1's join
    /// bound tripped and the thread is still running. The sequence is a floor:
    /// the camera may push past it, so consumers read the `provisional` flag
    /// and keep draining to their own bound rather than exiting on a target
    /// that can still move.
    ///
    /// Published at all, rather than left absent, because a consumer waiting
    /// for a watermark that never comes waits out its whole bound learning
    /// nothing, where this one at least tells it what has arrived so far.
    pub fn seal_provisionally(&mut self) -> Watermark {
        self.publish_watermark(true)
    }

    /// Idempotent, and deliberately keeps the first value: a second stop signal
    /// arriving mid-drain must not move a target the consumers may already have
    /// reached, and a late push must not extend a watermark that was published
    /// as final.
    fn publish_watermark(&mut self, provisional: bool) -> Watermark {
        let sequence = self.last_sequence();
        *self.terminal.get_or_insert(Watermark {
            sequence,
            provisional,
        })
    }

    /// The terminal watermark, or `None` while the camera may still produce and
    /// nothing has been published on its behalf.
    pub fn terminal_watermark(&self) -> Option<Watermark> {
        self.terminal
    }

    /// Get total duration of all segments in nanoseconds
    pub fn total_duration_ns(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ns).sum()
    }

    /// Convert a segment sequence number to timeline offset in nanoseconds
    /// Returns the cumulative duration of all segments before the given sequence
    pub fn sequence_to_offset_ns(&self, sequence: u64) -> Option<u64> {
        if sequence < self.first_sequence {
            return None;
        }
        let index = (sequence - self.first_sequence) as usize;
        if index > self.segments.len() {
            return None;
        }
        Some(
            self.segments
                .iter()
                .take(index)
                .map(|s| s.duration_ns)
                .sum(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locks::LockExt;

    const SEC: u64 = 1_000_000_000;

    fn segment(start_pts: u64) -> GopSegment {
        GopSegment {
            start_pts,
            duration_ns: SEC,
            data: Arc::new(vec![0; 4]),
            frame_count: 1,
        }
    }

    #[test]
    fn eviction_advances_first_sequence_and_frees_duration() {
        let buffer = HotBuffer::new("cam".to_string(), 3);
        let mut buf = buffer.write_recover();
        for seq in 0..5u64 {
            buf.push(segment(seq * SEC));
        }
        assert_eq!(buf.first_sequence(), 2);
        assert_eq!(buf.last_sequence(), 5);
        assert_eq!(buf.segment_count(), 3);
        assert!(buf.current_duration_secs() <= 3.0);
    }

    #[test]
    fn get_segment_by_sequence_accounts_for_eviction() {
        let buffer = HotBuffer::new("cam".to_string(), 3);
        let mut buf = buffer.write_recover();
        for seq in 0..5u64 {
            buf.push(segment(seq * SEC));
        }
        assert!(buf.get_segment_by_sequence(1).is_none());
        assert_eq!(buf.get_segment_by_sequence(2).unwrap().start_pts, 2 * SEC);
        assert_eq!(buf.get_segment_by_sequence(4).unwrap().start_pts, 4 * SEC);
        assert!(buf.get_segment_by_sequence(5).is_none());
    }

    /// A buffer nobody has sealed has no watermark, and sealing it names the
    /// sequence one past everything the camera pushed.
    #[test]
    fn sealing_publishes_the_sequence_one_past_the_last_segment() {
        let buffer = HotBuffer::new("cam".to_string(), 30);
        let mut buf = buffer.write_recover();
        assert_eq!(buf.terminal_watermark(), None);
        for seq in 0..4u64 {
            buf.push(segment(seq * SEC));
        }
        let expected = Watermark {
            sequence: 4,
            provisional: false,
        };
        assert_eq!(buf.seal(), expected);
        assert_eq!(buf.terminal_watermark(), Some(expected));
    }

    /// A camera the drain gave up joining is still running, and a watermark
    /// published on its behalf has to say so — a consumer that read it as final
    /// would exit on a number the camera is still moving past.
    #[test]
    fn a_watermark_published_for_a_running_camera_says_it_is_provisional() {
        let buffer = HotBuffer::new("cam".to_string(), 30);
        let mut buf = buffer.write_recover();
        buf.push(segment(0));
        assert_eq!(
            buf.seal_provisionally(),
            Watermark {
                sequence: 1,
                provisional: true
            }
        );
    }

    /// A second stop signal arriving mid-drain must not move a watermark the
    /// consumers may already have drained through, and a camera that was
    /// abandoned in phase 1 must not have its provisional watermark quietly
    /// extended by whatever it pushed afterwards.
    #[test]
    fn sealing_twice_keeps_the_first_watermark() {
        let buffer = HotBuffer::new("cam".to_string(), 30);
        let mut buf = buffer.write_recover();
        buf.push(segment(0));
        assert_eq!(buf.seal().sequence, 1);
        buf.push(segment(SEC));
        assert_eq!(buf.seal().sequence, 1, "a late push moved the watermark");
        assert_eq!(buf.terminal_watermark().unwrap().sequence, 1);
        // Nor can a later provisional seal downgrade a final one.
        assert!(!buf.seal_provisionally().provisional);
    }
}
