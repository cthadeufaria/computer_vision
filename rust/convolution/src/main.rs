use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::highgui::{named_window, imshow, wait_key};



fn main() -> opencv::Result<()> {
    let image = imread("bird.jpg", IMREAD_COLOR)?;

    named_window("Image", 0)?;
    imshow("Image", &image)?;
    wait_key(0)?; // Wait for a key press

    // window.set_image("image-001", &image)?;
    // viz::imshow("image", &image, opencv::core::Size::new(800, 600))?;

    Ok(())

}

//====================
// use opencv::imgcodecs::{
//     imread,
//     IMREAD_COLOR
// };
// use opencv::highgui::imshow;
// use opencv::core::{
//     Mat, 
//     CV_32FC1
// };


// fn main() {
//     let filename = "../../../images/bird.png";
//     let image = imread(filename, IMREAD_COLOR);
    
//     let image = match image {
//         Ok(img) => img,
//         Err(e) => {
//             eprintln!("Error reading image: {:?}", e);
//             return;
//         }
//     };

//     println!("Image slice: {:?}", image);

//     let kernel = Mat::eye(3, 3, CV_32FC1);

//     // let result = convolve(&image, &kernel, None, 0, CV_32FC1).unwrap();
//     // imshow("Result", &image);
// }