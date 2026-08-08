pub mod analytics;
pub mod api;
pub mod app;
pub mod buffer;
pub mod camera;
pub mod config;
// Public only for the binary's self-updater, which stages its download with
// the same atomic write + fsync helpers as the storage layer.
pub mod durable;
pub mod locks;
pub mod mpegts;
pub mod mqtt;
pub mod retry;
pub mod shutdown;
pub mod storage;
pub mod supervise;
