extern crate rulinalg;

use opencl::galerkin_1d::galerkin::GalerkinScheme;
use opencl::arrays::{self, CLVector};
use rulinalg::vector::Vector;
use opencl::galerkin_1d::unknowns::Unknown;
use ocl::{OclPrm, Kernel};
use ocl::builders::KernelBuilder;
use std::clone::Clone;
use galerkin_1d::grid::ReferenceElement;
use galerkin_1d::grid::SpatialFlux;
use galerkin_1d::operators::Operators;
use ocl::Buffer;
use std::marker::PhantomData;

pub trait SpatialFlux {
    type Unit: Sized + Copy;

    fn first(&self) -> Self::Unit;

    fn last(&self) -> Self::Unit;

    fn zero() -> Self::Unit;
}

pub struct Element<GS: GalerkinScheme> {
    pub index: usize,
    pub x_left: f64,
    pub x_right: f64,

    pub x_k: Vector<f64>,

    pub left_face: Face<GS>,
    pub right_face: Face<GS>,

    pub left_outward_normal: f64,
    pub right_outward_normal: f64,

    pub r_x: Vector<f64>,
    pub r_x_left: f64,
    pub r_x_right: f64,

    pub f_left_minus: f64,
    pub f_left_plus: f64,
    pub f_right_minus: f64,
    pub f_right_plus: f64,
}

impl<GS: GalerkinScheme> Element<GS> {
}

pub struct ElementStorage<U: Unknown> {
    pub u_k: Buffer<U>,

    pub u_left_minus: Buffer<U>,
    pub u_left_plus: Buffer<U>,
    pub u_right_minus: Buffer<U>,
    pub u_right_plus: Buffer<U>,

    pub du_left: Buffer<U>,
    pub du_right: Buffer<U>,

    pub u_k_rhs: Buffer<U>,
}

pub enum FaceType<GS: GalerkinScheme> {
    /**
     * An interior face with the index of the element on the other side.
     */
    Interior(usize),

    /**
     * Boundary condition which depends on the interior side of the boundary as well as
     * a time parameter.
     */
    Boundary(BoundaryCondition, GS::F)
}

pub struct BoundaryCondition {
    pub function_name: String,
}

impl BoundaryCondition {
}

pub struct Face<GS: GalerkinScheme> {
    pub face_type: FaceType<GS>,
}

pub struct Grid<GS: GalerkinScheme> {
    pub x_min: f64,
    pub x_max: f64,
    pub elements: Vec<Element<GS>>
}

pub fn generate_grid<GS, Fx>(
    x_min: f64,
    x_max: f64,
    n_k: i32,
    reference_element: &ReferenceElement,
    operators: &Operators,
    left_boundary: Face<GS>,
    right_boundary: Face<GS>,
    f: Fx,
) -> Grid<GS>
where
GS: GalerkinScheme,
Fx: Fn(&Vector<f64>) -> GS::F,
{
    let x_ks = compute_x_k(x_min, x_max, n_k, reference_element);
    let mut elements = vec![];

    for i in 0..n_k {
        let x_k = x_ks[i];

        let d_r_x_k = &operators.d_r * &x_k;
        let r_x = Vector::ones(d_r_x_k.size()).elediv(&d_r_x_k);
        let r_x_left = r_x[0];
        let r_x_right = r_x[reference_element.n_p as usize];

        let spatial_flux = f(&x_k);
        elements.push(Element {
            index: 0,
            x_left: x_k[0],
            x_right: x_k[x_k.len() - 1],
            x_k,
            r_x,
            r_x_left,
            r_x_right,
            left_face: left_boundary,
            right_face: Face {
                face_type: FaceType::Interior(1),
            },
            left_outward_normal: -1.,
            right_outward_normal: 1.,
            f_left_minus: spatial_flux.left(),
            f_left_plus: match left_boundary.face_type {
                FaceType::Interior(i) => f(&x_ks[i]).right(),
                FaceType::Boundary(_, f) => f,
            },
            f_right_minus: spatial_flux.right(),
            f_right_plus: match right_boundary.face_type {
                FaceType::Interior(i) => f(&x_ks[i]).left(),
                FaceType::Boundary(_, f) => f,
            },
        });
    }


    Grid {
        x_min,
        x_max,
        elements,
    }
}

fn compute_x_k(x_min: f64, x_max: f64, n_k: i32, reference_element: &ReferenceElement) -> Vec<Vector<f64>> {
    assert!(x_max > x_min);
    let diff = (x_max - x_min) / (n_k as f64);
    let transform = |left| {
        let s = (&reference_element.rs + 1.) / 2.;
        s * diff + left
    };
    0..n_k.map(|i| {
        transform(x_min + diff * i as f64)
    }).collect::<Vec<Vector<f64>>>()
}