mod decoder;
pub mod detection_grid;
mod motion;
pub mod ollama;
pub mod pipeline;
pub mod run_tracker;

pub use motion::TunerStats;
pub use ollama::OllamaDetector;
pub use pipeline::{spawn_analyzer, AnalyzerContext};
