extern crate opencv;
extern crate nalgebra as na;
extern crate glob;

use opencv::core::{
    TermCriteria, 
    TermCriteria_MAX_ITER, 
    TermCriteria_EPS,
    Size,
    Mat,
    Vector,
    Point2f
};
use opencv::imgcodecs::imread;
use opencv::imgproc::{
    cvt_color, 
    COLOR_BGR2GRAY
};
use opencv::calib3d::{
    find_chessboard_corners, 
    CALIB_CB_ADAPTIVE_THRESH,
    CALIB_CB_NORMALIZE_IMAGE,
    CALIB_CB_FAST_CHECK
};
use glob::glob;



fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the termination criteria (similar to Python's criteria)
    let criteria = TermCriteria::new(TermCriteria_MAX_ITER as i32 + TermCriteria_EPS as i32, 30, 0.001,)?;

    // Initialize empty vector for names of images in a path
    let mut files: Vec<String> = Vec::new();

    // Prepare object points like (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
    let mut objp = na::DMatrix::<f32>::zeros(6 * 7, 3);

    // Fill the object points matrix with 2D grid coordinates for (0,0), (1,0), (2,0)...., (6,5)
    for j in 0..6 {
        for i in 0..7 {
            let index = j * 7 + i;
            objp[(index, 0)] = i as f32; // x-coordinate
            objp[(index, 1)] = j as f32; // y-coordinate
            // z-coordinate is already set to 0 in initialization
        }
    }

    // Get all the images in the path with glob pattern matching
    for entry in glob("./calibration_images/images/*.jpg").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => files.push(path.display().to_string()),
            Err(e) => println!("{:?}", e),
        }
    }

    for file in files {
        let image = imread(&file, 1)?;

        // let mut gray = Mat::default(); // CALIB_CB_ADAPTIVE_THRESH already converts to black and white
        // cvt_color(&image, &mut gray, COLOR_BGR2GRAY, 0)?;
        
        let mut corners = Vector::<Point2f>::new();
        if let Ok(found) = find_chessboard_corners(
            &image, Size { width: 7, height: 6 }, &mut corners, 
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK
        ) {
            if found {
                println!("Found corners in image: {}", file);
                println!("{:?}", corners);
            }
            else {
                println!("Couldn't find corners in image: {}", file);
            }
        }
    }

    Ok(())
}