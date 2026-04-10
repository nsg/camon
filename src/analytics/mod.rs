mod decoder;
pub mod detection_grid;
mod motion;
pub mod ollama;
pub mod pipeline;

pub use motion::TunerStats;
pub use ollama::OllamaDetector;
pub use pipeline::spawn_analyzer;
