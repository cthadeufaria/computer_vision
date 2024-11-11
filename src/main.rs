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
    corner_sub_pix,
    COLOR_BGR2GRAY
};
use opencv::calib3d::{
    find_chessboard_corners,
    draw_chessboard_corners,
    calibrate_camera,
    CALIB_CB_ADAPTIVE_THRESH,
    CALIB_CB_NORMALIZE_IMAGE,
    CALIB_CB_FAST_CHECK
};
use opencv::highgui::{
    imshow,
    wait_key,
    destroy_all_windows
};
use glob::glob;



fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the termination criteria (similar to Python's criteria)
    let criteria = TermCriteria::new(TermCriteria_MAX_ITER as i32 + TermCriteria_EPS as i32, 30, 0.001,)?;

    // Initialize empty vector for names of images in a path
    let mut files: Vec<String> = Vec::new();

    // Prepare object points like (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
    // let mut objp = na::DMatrix::<f32>::zeros(6 * 7, 3);
    let mut objp = na::DMatrix::<f32>::zeros(3 * 6, 3);

    // Fill the object points matrix with 2D grid coordinates for (0,0), (1,0), (2,0)...., (6,5)
    for j in 0..3 {
        for i in 0..6 {
            let index = j * 6 + i;
            objp[(index, 0)] = i as f32; // x-coordinate
            objp[(index, 1)] = j as f32; // y-coordinate
            // z-coordinate is already set to 0 in initialization
        }
    }

    // Get all the images in the path with glob pattern matching
    for entry in glob("./calibration_images/images/*.png").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => files.push(path.display().to_string()),
            Err(e) => println!("{:?}", e),
        }
    }

    // Create a vector to store object points
    let mut objpoints: Vec<na::DMatrix<f32>> = Vec::new();
    let mut imgpoints: Vec<Vector<Point2f>> = Vec::new();

    for file in files {
        let mut image = imread(&file, 1)?;

        let mut gray = Mat::default(); // CALIB_CB_ADAPTIVE_THRESH already converts to black and white
        cvt_color(&image, &mut gray, COLOR_BGR2GRAY, 0)?;
        
        let mut corners = Vector::<Point2f>::new();
        if let Ok(found) = find_chessboard_corners(&image, Size { width: 6, height: 3 }, &mut corners, 
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK
        ) {
            if found {
                println!("Found corners in image: {}", file);
                println!("{:?}", corners);
                                
                // Append the object points to the vector
                objpoints.push(objp.clone());

                corner_sub_pix(&gray, &mut corners, Size { width: 11, height: 11 }, 
                    Size { width: -1, height: -1 }, criteria)?;
                
                imgpoints.push(corners.clone());

                // Draw and display the corners
                draw_chessboard_corners(&mut image, Size { width: 7, height: 6 }, &corners, true,)?;
                imshow("Corners", &image)?;
                wait_key(1000)?;
                
            }
            else {
                println!("Couldn't find corners in image: {}", file);
            }
        }
    }

    destroy_all_windows()?;

    // calibrate_camera(&objpoints, &imgpoints, Size { width: 640, height: 480 }, 0, 0, 
    //     None, None, 0, criteria)?;

    Ok(())
}