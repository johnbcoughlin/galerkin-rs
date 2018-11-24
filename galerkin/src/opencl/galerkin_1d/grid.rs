extern crate rulinalg;

use opencl::galerkin_1d::galerkin::GalerkinScheme;
use opencl::arrays::{self, CLVector};
use rulinalg::vector::Vector;
use opencl::galerkin_1d::unknowns::Unknown;
use ocl::{OclPrm, Kernel};
use ocl::builders::KernelBuilder;
use std::clone::Clone;
use galerkin_1d::grid::ReferenceElement;
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
    pub r_x_at_faces: Vector<f64>,
}

impl<GS: GalerkinScheme> Element<GS> {
}

pub struct ElementStorage<U: Unknown> {
    pub u_k: Buffer<U>,

    pub u_left_minus: Buffer<U>,
    pub u_left_plus: Buffer<U>,
    pub u_right_minus: Buffer<U>,
    pub u_right_plus: Buffer<U>,
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
    Boundary(BoundaryCondition<GS>)
}

pub struct BoundaryCondition<GS: GalerkinScheme> {
    /**
     * An OpenCL kernel which takes as input the coordinates and time,
     * and returns a boundary value.
     */
    source: String,

    pub function_name: String,

    // TODO remove this or replace with a host-memory function
    marker: PhantomData<GS>,
}

impl<'prog, GS:GalerkinScheme> BoundaryCondition<GS> {
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
    assert!(x_max > x_min);
    let diff = (x_max - x_min) / (n_k as f64);
    let transform = |left| {
        let s = (&reference_element.rs + 1.) / 2.;
        s * diff + left
    };
    let mut elements = vec![];
    let x_k = transform(x_min);

    let d_r_x_k = &operators.d_r * &x_k;
    let r_x = Vector::ones(d_r_x_k.size()).elediv(&d_r_x_k);
    let r_x_at_faces = vector![r_x[0], r_x[reference_element.n_p as usize]];

    let spatial_flux = f(&x_k);
    elements.push(Element {
        index: 0,
        x_left: x_min,
        x_right: x_min + diff,
        x_k,
        r_x,
        r_x_at_faces,
        left_face: left_boundary,
        right_face: Face {
            face_type: FaceType::Interior(1),
        },
        left_outward_normal: -1.,
        right_outward_normal: 1.,
    });

    Grid {
        x_min,
        x_max,
        elements,
    }
}