#[derive(Debug, Clone)]
pub struct GopSegment {
    pub start_pts: u64,
    pub duration_ns: u64,
    pub data: Vec<u8>,
    pub frame_count: u32,
}

impl GopSegment {
    pub fn new(start_pts: u64) -> Self {
        Self {
            start_pts,
            duration_ns: 0,
            data: Vec::new(),
            frame_count: 0,
        }
    }

    pub fn finalize(&mut self, end_pts: u64) {
        if end_pts > self.start_pts {
            self.duration_ns = end_pts - self.start_pts;
        }
    }
}
