extern crate itertools;
extern crate rulinalg;

use std::fmt;

use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use rulinalg::vector::Vector;

use crate::blas;
use crate::functions::vandermonde::{grad_vandermonde_2d, vandermonde, vandermonde_2d};
use crate::galerkin_2d::grid::LocalMetric;
use crate::galerkin_2d::grid::XYTuple;
use crate::galerkin_2d::reference_element::ReferenceElement;
use crate::galerkin_2d::unknowns::Unknown;

#[derive(Debug)]
pub struct Operators {
    // The Vandermonde matrix
    pub v: Matrix<f64>,

    // The D_r derivative matrix. D_r*V = V_r
    pub d_r: Matrix<f64>,
    // The D_s derivative matrix. D_s*V = V_s
    pub d_s: Matrix<f64>,

    // The weak derivative operators
    pub d_r_w: Matrix<f64>,
    pub d_s_w: Matrix<f64>,

    // The matrix lifting the surface integral on the simplex edges to the
    // area integral over the simplex.
    pub lift: FaceLift,
}

#[derive(Debug)]
pub struct FaceLift {
    pub face1: Matrix<f64>,
    pub face2: Matrix<f64>,
    pub face3: Matrix<f64>,
}

impl fmt::Display for FaceLift {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Face 1:")?;
        writeln!(f, "{}", self.face1)?;
        writeln!(f, "Face 2:")?;
        writeln!(f, "{}", self.face2)?;
        writeln!(f, "Face 3:")?;
        writeln!(f, "{}", self.face3)
    }
}

pub trait FaceLiftable: Unknown {
    fn lift_faces(
        face_lift: &FaceLift,
        face1: &Self::Line,
        face2: &Self::Line,
        face3: &Self::Line,
    ) -> Self;
}

pub fn assemble_operators(reference_element: &ReferenceElement) -> Operators {
    let n = reference_element.n as i32;
    let rs = &reference_element.rs;
    let ss = &reference_element.ss;

    let (a, b) = ReferenceElement::rs_to_ab(&rs, &ss);
    let v = vandermonde_2d(n, &a, &b);
    let v_inv = v
        .clone()
        .inverse()
        .expect("Non-invertible Vandermonde matrix");
    let (v_r, v_s) = grad_vandermonde_2d(n, &a, &b);

    let d_r = &v_r * &v_inv;
    let d_s = &v_s * &v_inv;

    let v_vt = &v * v.transpose();
    let v_vt_inv = v_vt
        .inverse()
        .expect("Non-invertible weak Vandermonde matrix");
    let v_v_rt = &v * v_r.transpose();
    let v_v_st = &v * v_s.transpose();

    let d_r_w = v_v_rt * &v_vt_inv;
    let d_s_w = v_v_st * &v_vt_inv;

    let lift = assemble_lift(reference_element, &v);

    Operators {
        v,
        d_r,
        d_s,
        d_r_w,
        d_s_w,
        lift,
    }
}

fn assemble_lift(reference_element: &ReferenceElement, v2d: &Matrix<f64>) -> FaceLift {
    let inv_mass_matrix = v2d * v2d.transpose();
    let n = reference_element.n as i32;
    let n_p = (n + 1) * (n + 2) / 2;
    let n_fp: usize = (n + 1) as usize;

    let ss = &reference_element.ss;
    let rs = &reference_element.rs;

    let mut face1: Matrix<f64> = Matrix::zeros(n_p as usize, n_fp as usize);
    let face1_r: Vector<f64> = rs.select(&reference_element.face1.as_slice());
    let v = vandermonde(&face1_r, n);
    let mass_face1 = (&v * &v.transpose()).inverse().expect("non-invertible");
    &reference_element
        .face1
        .iter()
        .enumerate()
        .for_each(|(j, &i)| {
            face1
                .row_mut(i as usize)
                .iter_mut()
                .zip(mass_face1.row(j).into_iter())
                .for_each(|(dest, x)| *dest = *x)
        });
    let lift_face_1 = &inv_mass_matrix * face1;

    let mut face2: Matrix<f64> = Matrix::zeros(n_p as usize, n_fp as usize);
    // Can use either r or s here; the important thing is that they are distributed in the
    // same way along the diagonal edge.
    let face2_r: Vector<f64> = rs.select(&reference_element.face2.as_slice());
    let v = vandermonde(&face2_r, n);
    let _mass_face2 = (&v * &v.transpose()).inverse().expect("non-invertible");
    &reference_element
        .face2
        .iter()
        .enumerate()
        .for_each(|(j, &i)| {
            face2
                .row_mut(i as usize)
                .iter_mut()
                .zip(mass_face1.row(j).into_iter())
                .for_each(|(dest, x)| *dest = *x)
        });
    let lift_face_2 = &inv_mass_matrix * face2;

    let mut face3: Matrix<f64> = Matrix::zeros(n_p as usize, n_fp as usize);
    let face3_s: Vector<f64> = ss.select(&reference_element.face3.as_slice());
    let v = vandermonde(&face3_s, n_p);
    let _mass_face3 = (&v * &v.transpose()).inverse().expect("non-invertible");
    &reference_element
        .face3
        .iter()
        .enumerate()
        .for_each(|(j, &i)| {
            face3
                .row_mut(i as usize)
                .iter_mut()
                .zip(mass_face1.row(j).into_iter())
                .for_each(|(dest, x)| *dest = *x)
        });
    let lift_face_3 = &inv_mass_matrix * face3;

    FaceLift {
        face1: lift_face_1,
        face2: lift_face_2,
        face3: lift_face_3,
    }
}

pub fn cutoff_filter(operators: &Operators, n: i32, frac: f64) -> Matrix<f64> {
    let _n_p = (n + 1) * (n + 2) / 2;
    let mut array = [1.; 55];
    let sk: usize = 0;
    for i in 0..n + 1 {
        for j in 0..n {
            if i + j > n {
                array[sk] = frac;
            }
        }
    }
    let diag = Matrix::from_diag(&array);
    &operators.v
        * &(diag
            * &operators
                .v
                .clone()
                .inverse()
                .expect("non-invertible Vandermonde matrix"))
}

pub fn grad(
    u: &Vector<f64>,
    operators: &Operators,
    local_metric: &LocalMetric,
) -> XYTuple<Vector<f64>> {
    let u_r = blas::matrix_multiply(&operators.d_r, u);
    let u_s = blas::matrix_multiply(&operators.d_s, u);
    let u_x = blas::elemul_affine_(
        &local_metric.r_x,
        &u_r,
        1.,
        blas::elemul(&local_metric.s_x, &u_s),
        1.,
    );
    let u_y = blas::elemul_affine_(
        &local_metric.r_y,
        &u_r,
        1.,
        blas::elemul(&local_metric.s_y, &u_s),
        1.,
    );
    XYTuple { x: u_x, y: u_y }
}

pub fn curl_2d(
    u_x: &Vector<f64>,
    u_y: &Vector<f64>,
    operators: &Operators,
    local_metric: &LocalMetric,
) -> Vector<f64> {
    let u_xr = blas::matrix_multiply(&operators.d_r, u_x);
    let u_xs = blas::matrix_multiply(&operators.d_s, u_x);
    let u_yr = blas::matrix_multiply(&operators.d_r, u_y);
    let u_ys = blas::matrix_multiply(&operators.d_s, u_y);
    blas::vector_sub_(
        blas::elemul_affine_(
            &u_yr,
            &local_metric.r_x,
            1.,
            blas::elemul(&u_ys, &local_metric.s_x),
            1.,
        ),
        &blas::elemul_affine_(
            &u_xr,
            &local_metric.r_y,
            1.,
            blas::elemul(&u_xs, &local_metric.s_y),
            1.,
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::galerkin_2d::reference_element::*;

    use super::*;

    #[test]
    fn computes_vandermonde_matrix() {
        assert_eq!(operators().v[[5, 1]], -0.48198050606196585);
        assert_eq!(operators().v[[7, 8]], 0.3877560604205198);
    }

    #[test]
    fn computes_strong_derivative_matrix() {
        // Strong derivative operators
        assert_eq!(operators().d_r[[5, 2]], -2.0125329589324465);
        assert_eq!(operators().d_r[[9, 4]], -0.8328718938848465);
        assert_eq!(operators().d_s[[3, 2]], -2.1282604157187315);
        assert_eq!(operators().d_s[[9, 4]], 0.00000000000000030531133177191805);
    }

    #[test]
    fn computes_weak_derivative_matrix() {
        // Weak derivative operators
        assert_eq!(operators().d_r_w[[6, 2]], -0.7557453508986891);
        assert_eq!(operators().d_r_w[[9, 4]], 1.5047468938848467);
        assert_eq!(operators().d_s_w[[3, 2]], -0.5384062509479254);
        assert_eq!(
            operators().d_s_w[[9, 4]],
            -0.0000000000000007147060721024445
        );
    }

    fn operators() -> Operators {
        let reference_element = ReferenceElement::legendre(4);
        assemble_operators(&reference_element)
    }
}
