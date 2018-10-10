extern crate blas;
#[cfg(target_os = "macos")]
extern crate accelerate_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;
extern crate rulinalg;

use rulinalg::vector::Vector;
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use blas::blas::{
    dgemv,
    dgbmv,
};

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

/**
 * Computes alpha*a.*b + beta * c
 */
pub fn elemul_affine(a: &Vector<f64>, b: &Vector<f64>, alpha: f64, c: &Vector<f64>, beta: f64) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    assert_eq!(a.size(), c.size());
    let n = a.size() as i32;
    let mut y = c.clone();
    unsafe {
        dgbmv(
            b'N',
            n,
            n,
            0, // no sub-diagonals
            0, // no super-diagonals
            alpha,
            a.data().as_slice(),
            1,
            b.data().as_slice(),
            1,
            beta,
            y.mut_data(),
            1
        );
    }
    y
}

/**
 * Computes alpha*a.*b
 */
pub fn elemul_scalar(a: &Vector<f64>, b: &Vector<f64>, alpha: f64) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    let n = a.size() as i32;
    let mut y = Vector::zeros(n as usize);
    unsafe {
        dgbmv(
            b'N',
            n,
            n,
            0, // no sub-diagonals
            0, // no super-diagonals
            alpha,
            a.data().as_slice(),
            1,
            b.data().as_slice(),
            1,
            0.,
            y.mut_data(),
            1
        );
    }
    y
}

/**
 * Computes a.*b
 */
pub fn elemul(a: &Vector<f64>, b: &Vector<f64>) -> Vector<f64> {
    elemul_scalar(a, b, 1.)
}

#[cfg(test)]
mod tests {
    use super::{matrix_multiply, elemul, elemul_affine};

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

    #[test]
    fn test_elemul() {
        let a = vector![1., 2., 3.];
        let b = vector![1., 3., 5.];
        let y = elemul(&a, &b);
        assert_eq!(y[0], 1.);
        assert_eq!(y[1], 6.);
        assert_eq!(y[2], 15.);
    }

    #[test]
    fn test_elemul_affine() {
        let a = vector![1., 2., 3.];
        let b = vector![1., 3., 5.];
        let c = vector![0.1, 0.2, 0.3];
        let y = elemul_affine(&a, &b, 2., &c, 3.);
        assert_eq!(y[0], 2.3);
        assert_eq!(y[1], 12.6);
        assert_eq!(y[2], 30.9);
    }
}