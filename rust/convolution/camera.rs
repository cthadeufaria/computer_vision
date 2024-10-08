use anyhow::Result;
use opencv::{
    prelude::*,
    highgui,
    videoio,
};

fn main() -> Result<()> {
    highgui::named_window("video", highgui::WINDOW__FULLSCREEN)?;
    
}