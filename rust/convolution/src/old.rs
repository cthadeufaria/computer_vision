use opencv::imgcodecs::imread;
use convolve2d::{convolve2d, DynamicMatrix};


fn flip(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut flipped = vec![vec![0.0; a[0].len()]; a.len()];
    let rows = a.len();
    let cols = a[0].len();

    for i in 0..rows {
        for j in 0..cols {
            flipped[rows - 1 - i][cols - 1 - j] = a[i][j];
        }
    }

    flipped
}

fn add_padding(a: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let cols = a[0].len();
    let mut result = vec![vec![0; cols + 2]];

    for row in a {
        let mut padded_row = vec![0];
        padded_row.extend_from_slice(row);
        padded_row.push(0);
        result.push(padded_row);
    }

    result.push(vec![0; cols + 2]);

    result
}

fn normalize(a: &[Vec<i32>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();
    
    let sum: i32 = a.iter()
        .flat_map(|row| row.iter())
        .sum();

    let mut normalized = vec![vec![0.0; cols]; rows];

    for row in 0..rows {
        for col in 0..cols {
            normalized[row][col] = a[row][col] as f64 / sum as f64;
        }
    }

    normalized
}

fn convolve(a: &[Vec<i32>], kernel: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let x = a.len();
    let y = a[0].len();
    let u = kernel.len();
    let v = kernel[0].len();

    let mut convolved = vec![vec![0.0; y - v + 1]; x - u + 1];

    for i in 0..(x - u + 1) {
        for j in 0..(y - v + 1) {
            let mut sum = 0.0;
            for k in 0..u {
                for l in 0..v {
                    sum += a[i + k][j + l] as f64 * kernel[k][l];
                }
            }

            convolved[i][j] = sum;
        }
    }

    convolved
}

fn convolve2d(matrix: &DynamicMatrix, kernel: &DynamicMatrix) {
    let result = convolve2d(matrix, kernel);
    result
}

fn main() {
    let image_path = "../../../images/bird.png";
    let image = imread(image_path);
    // let convolved = convolve2d(&DynamicMatrix::from(&image), &DynamicMatrix::from(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]));
    // println!("{:?}", convolved);    
}