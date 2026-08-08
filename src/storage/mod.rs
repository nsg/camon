pub mod anchor;
pub mod backend;
pub mod contract;
mod debug_store;
mod detection_store;
pub mod event_index;
pub mod event_registry;
mod recovery;
pub mod stathost;
mod store;
pub mod warm_index;
pub mod watchdog;

pub use anchor::StorageAnchor;
pub use backend::{
    LocalDiskBackend, RangeRequest, ServedRange, ThumbnailError, VideoStream, WarmStorageBackend,
};
pub use contract::StopFlag;
pub use debug_store::DetectionDebugStore;
pub use detection_store::{DetectionEntry, DetectionStore};
pub use event_index::{EventCursor, EventPage, EventRef, EventType, WarmEventEntry};
pub use event_registry::{EventRegistry, PendingEvent, UpgradeTarget, Verdict, VerdictId};
pub use recovery::recover_orphans;
pub use stathost::StathostBackend;
pub use store::{MapKind, MotionEntry, MotionStore};
pub use warm_index::WarmEventIndex;
pub use watchdog::{RecordingMode, RecordingWatchdog};

/// An `Instant` `ago` in the past, so a test can reach across a demand window instead of
/// waiting one out.
#[cfg(test)]
pub(crate) fn back_date(ago: std::time::Duration) -> std::time::Instant {
    let now = std::time::Instant::now();
    now.checked_sub(ago).unwrap_or(now)
}
