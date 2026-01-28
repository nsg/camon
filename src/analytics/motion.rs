use opencv::{
    core::Mat,
    prelude::*,
    video::{self, BackgroundSubtractorTrait},
    Result as CvResult,
};

pub struct MotionScore {
    pub score: f32,
}

pub struct MotionDetector {
    mog2: opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
    fg_mask: Mat,
    learning_rate: f64,
}

impl MotionDetector {
    pub fn new() -> CvResult<Self> {
        let mog2 = video::create_background_subtractor_mog2(500, 16.0, true)?;
        let fg_mask = Mat::default();

        Ok(Self {
            mog2,
            fg_mask,
            learning_rate: -1.0,
        })
    }

    pub fn process_frame(
        &mut self,
        frame: &impl opencv::core::ToInputArray,
    ) -> CvResult<MotionScore> {
        BackgroundSubtractorTrait::apply(
            &mut self.mog2,
            frame,
            &mut self.fg_mask,
            self.learning_rate,
        )?;

        let total_pixels = self.fg_mask.rows() * self.fg_mask.cols();
        if total_pixels == 0 {
            return Ok(MotionScore { score: 0.0 });
        }

        let fg_pixels = opencv::core::count_non_zero(&self.fg_mask)? as f32;
        let foreground_ratio = fg_pixels / total_pixels as f32;

        let score = (foreground_ratio * 10.0).min(1.0);

        Ok(MotionScore { score })
    }
}
