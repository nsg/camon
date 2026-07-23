mod ccl;
mod decoder;
mod mog2;
mod morph;
mod motion;
pub mod motion_settings;
pub mod ollama;
pub mod pipeline;
pub mod run_tracker;

// Consumed through the library crate root (examples/, integration tests);
// the binary crate reaches MotionDetector via the pipeline instead.
#[allow(unused_imports)]
pub use motion::{MotionBox, MotionDetector};
pub use motion_settings::{MotionSettingsStore, SettingsUpdate};
pub use ollama::OllamaDetector;
pub use pipeline::{spawn_analyzer, AnalyzerContext};
