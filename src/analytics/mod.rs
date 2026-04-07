mod decoder;
pub mod detection_grid;
mod motion;
mod object;
mod pipeline;

pub use object::ObjectDetector;
pub use pipeline::spawn_analyzer;
