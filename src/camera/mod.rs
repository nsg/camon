// Public so a library consumer can name `RtspError::NoRecording`'s payload
// (`rtsp::StreamFailure`), which a re-export cannot make nameable on its own.
pub mod rtsp;

pub use rtsp::{FfmpegPipeline, NoRecordingTracker, RtspError};
