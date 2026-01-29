mod hot;
mod segment;
pub mod warm;

pub use hot::{EvictedSegment, HotBuffer};
pub use segment::GopSegment;
