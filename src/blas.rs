extern crate blas;
extern crate rulinalg;

use rulinalg::vector::Vector;
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use blas::blas::dgemv;

pub fn matrix_multiply(a: &Matrix<f64>, x: &Vector<f64>) -> Vector<f64> {
    let m = a.cols() as i32;
    let n = a.rows() as i32;
    let mut y = Vector::zeros(n as usize);
    unsafe {
        dgemv(
            b'T', // transpose because rulinalg stores matrices row-major.
            m,
            n,
            1.,
            a.data().as_slice(),
            m,
            x.data().as_slice(),
            1,
            0.,
            y.mut_data(),
            1
        );
    }
    y
}

#[cfg(test)]
mod tests {
    use super::matrix_multiply;

    #[test]
    fn test_matrix_multiply() {
        let a = matrix![
            1., 1.;
            2., 3.;
            3., 5.;
            4., 7.
        ];
        let x = vector![4., 6.7];
        let y = matrix_multiply(&a, &x);
        assert_eq!(y[0], 10.7);
        assert_eq!(y[1], 28.1);
        assert_eq!(y[2], 45.5);
        assert_eq!(y[3], 62.9);
    }
}