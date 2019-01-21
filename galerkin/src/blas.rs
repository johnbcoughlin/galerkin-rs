#[cfg(target_os = "macos")]
extern crate accelerate_src;
extern crate blas;
extern crate lapack;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;
extern crate rulinalg;

use crate::blas::blas::{daxpy, dgbmv, dgemv, dscal};
use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;

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
            1,
        );
    }
    y
}

/**
 * Computes alpha*a.*b + beta * c
 */
pub fn elemul_affine_(
    a: &Vector<f64>,
    b: &Vector<f64>,
    alpha: f64,
    mut y: Vector<f64>,
    beta: f64,
) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    assert_eq!(a.size(), y.size());
    let n = a.size() as i32;
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
            1,
        );
    }
    y
}

/**
 * Computes alpha*a.*b + beta * c
 */
pub fn elemul_affine(
    a: &Vector<f64>,
    b: &Vector<f64>,
    alpha: f64,
    c: &Vector<f64>,
    beta: f64,
) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    assert_eq!(a.size(), c.size());
    let y = c.clone();
    elemul_affine_(a, b, alpha, y, beta)
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
            1,
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

/**
 * Computes a + b
 */
pub fn vector_add(a: &Vector<f64>, b: &Vector<f64>) -> Vector<f64> {
    let y = b.clone();
    vector_add_(a, y)
}

pub fn vector_add_(a: &Vector<f64>, mut b: Vector<f64>) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    let n = a.size() as i32;
    unsafe { daxpy(n, 1., a.data().as_slice(), 1, b.mut_data(), 1) }
    b
}

/**
 * Computes a - b
 */
pub fn vector_sub(a: &Vector<f64>, b: &Vector<f64>) -> Vector<f64> {
    let y = a.clone();
    vector_sub_(y, b)
}

pub fn vector_sub_(mut a: Vector<f64>, b: &Vector<f64>) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    let n = a.size() as i32;
    unsafe { daxpy(n, -1., b.data().as_slice(), 1, a.mut_data(), 1) }
    a
}

/**
 * Computes alpha * a + b
 */
pub fn vector_affine(a: &Vector<f64>, alpha: f64, b: &Vector<f64>) -> Vector<f64> {
    let b = b.clone();
    vector_affine_(a, alpha, b)
}

/**
 * Computes alpha * a + b
 */
pub fn vector_affine_(a: &Vector<f64>, alpha: f64, mut b: Vector<f64>) -> Vector<f64> {
    assert_eq!(a.size(), b.size());
    let n = a.size() as i32;
    unsafe { daxpy(n, alpha, a.data().as_slice(), 1, b.mut_data(), 1) }
    b
}

/**
 * Computes alpha * a
 */
pub fn vector_scale_(mut a: Vector<f64>, alpha: f64) -> Vector<f64> {
    let n = a.size() as i32;
    unsafe { dscal(n, alpha, a.mut_data(), 1) }
    a
}

pub fn vector_scale(a: &Vector<f64>, alpha: f64) -> Vector<f64> {
    let x = a.clone();
    vector_scale_(x, alpha)
}

#[cfg(test)]
mod tests {
    use super::{elemul, elemul_affine, matrix_multiply};

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
