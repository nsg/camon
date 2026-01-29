use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use tokio::sync::mpsc;

use super::GopSegment;

const NANOS_PER_SEC: u64 = 1_000_000_000;

pub struct EvictedSegment {
    pub segment: GopSegment,
    pub camera_id: String,
    pub sequence: u64,
}

pub struct HotBuffer {
    segments: VecDeque<GopSegment>,
    max_duration_ns: u64,
    current_duration_ns: u64,
    camera_id: String,
    first_sequence: u64,
    eviction_tx: Option<mpsc::UnboundedSender<EvictedSegment>>,
}

impl HotBuffer {
    pub fn new(camera_id: String, max_duration_secs: u64) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            segments: VecDeque::new(),
            max_duration_ns: max_duration_secs * NANOS_PER_SEC,
            current_duration_ns: 0,
            camera_id,
            first_sequence: 0,
            eviction_tx: None,
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
                let evicted_sequence = self.first_sequence;
                self.current_duration_ns = self.current_duration_ns.saturating_sub(old.duration_ns);
                self.first_sequence += 1;
                tracing::trace!(
                    camera = %self.camera_id,
                    evicted_duration_ms = old.duration_ns / 1_000_000,
                    first_sequence = self.first_sequence,
                    "evicted old segment"
                );
                if let Some(tx) = &self.eviction_tx {
                    let _ = tx.send(EvictedSegment {
                        segment: old,
                        camera_id: self.camera_id.clone(),
                        sequence: evicted_sequence,
                    });
                }
            } else {
                break;
            }
        }
    }

    pub fn set_eviction_sender(&mut self, tx: mpsc::UnboundedSender<EvictedSegment>) {
        self.eviction_tx = Some(tx);
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
