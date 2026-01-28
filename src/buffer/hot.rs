use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use super::GopSegment;

const NANOS_PER_SEC: u64 = 1_000_000_000;

pub struct HotBuffer {
    segments: VecDeque<GopSegment>,
    max_duration_ns: u64,
    current_duration_ns: u64,
    camera_id: String,
    first_sequence: u64,
}

impl HotBuffer {
    pub fn new(camera_id: String, max_duration_secs: u64) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            segments: VecDeque::new(),
            max_duration_ns: max_duration_secs * NANOS_PER_SEC,
            current_duration_ns: 0,
            camera_id,
            first_sequence: 0,
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
}
