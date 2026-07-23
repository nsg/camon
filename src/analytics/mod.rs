mod ccl;
mod decoder;
pub mod detection_grid;
mod mog2;
mod morph;
mod motion;
pub mod ollama;
pub mod pipeline;
pub mod run_tracker;

pub use motion::TunerStats;
// Consumed through the library crate root (examples/, integration tests);
// the binary crate reaches MotionDetector via the pipeline instead.
#[allow(unused_imports)]
pub use motion::{MotionBox, MotionDetector};
pub use ollama::OllamaDetector;
pub use pipeline::{spawn_analyzer, AnalyzerContext};
