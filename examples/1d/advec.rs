extern crate galerkin;
#[macro_use]
extern crate rulinalg;

use self::rulinalg::vector::Vector;
use galerkin::functions::range_kutta::{RKA, RKB, RKC};
use galerkin::galerkin_1d::flux::{FluxEnum, FluxScheme, FreeflowFlux, LaxFriedrichs};
use galerkin::galerkin_1d::galerkin::{compute_flux, Formulation, GalerkinScheme};
use galerkin::galerkin_1d::grid::{self, generate_grid, FaceType, ReferenceElement};
use galerkin::galerkin_1d::operators::{assemble_operators, Operators};
use galerkin::galerkin_1d::unknowns::{communicate, initialize_storage, Unknown};
use galerkin::plot::plot2d::Plotter2D;
use std::f64::consts;
use std::iter::repeat;

fn main() {
    advec_1d_example();
}

#[inline(never)]
pub fn advec_1d<Fx>(
    u_0: Fx,
    grid: &Grid,
    reference_element: &ReferenceElement,
    operators: &Operators,
    a: f64,
) -> Vec<UStorage>
where
    Fx: Fn(&Vector<f64>) -> U,
{
    let mut plotter = Plotter2D::create(0.0, 2.0, -1.0, 1.0);

    let final_time = 1.3;

    let cfl = 0.75;
    let x_scale = 0.01;
    let dt: f64 = 0.5 * cfl / (consts::PI * 2.) * x_scale;
    let n_t = (final_time / dt).ceil() as i32;
    let dt = final_time / n_t as f64;

    let mut t: f64 = 0.0;

    let mut storage: Vec<UStorage> =
        initialize_storage(u_0, reference_element.n_p, grid, operators);
    let mut residuals: Vec<Vector<f64>> = repeat(Vector::zeros(reference_element.n_p as usize + 1))
        .take(grid.elements.len())
        .collect();

    for epoch in 0..n_t {
        for int_rk in 0..5 {
            let t = t + RKC[int_rk] * dt;

            // communicate current values of u across faces
            communicate(t, grid, &storage);

            // update each element's local solution
            for elt in (*grid).elements.iter() {
                let mut storage = &mut storage[elt.index as usize];

                let residuals_u = {
                    let residuals_u = &(residuals[elt.index as usize]);

                    let rhs_u = advec_rhs_1d(&elt, &storage, &operators, a);
                    residuals_u * RKA[int_rk] + rhs_u * dt
                };

                let new_u = {
                    let u_ref: &U = &storage.u_k;
                    &u_ref.u + &residuals_u * RKB[int_rk]
                };

                residuals[elt.index as usize] = residuals_u;

                storage.u_k = U { u: new_u };
            }
        }
        if epoch % 20 == 0 {
            plotter.header();
            for elt in (*grid).elements.iter() {
                let storage = &storage[elt.index as usize];
                plotter.plot(&elt.x_k, &storage.u_k.u);
            }
            plotter.replot();
        }
        t = t + dt;
    }
    for elt in (*grid).elements.iter() {
        let storage = &storage[elt.index as usize];
        println!("{:?}", &storage.u_k);
    }

    storage
}

fn advec_rhs_1d(
    elt: &Element,
    elt_storage: &UStorage,
    operators: &Operators,
    a: f64,
) -> Vector<f64> {
    let (du_left, du_right) = compute_flux(elt, elt_storage);
    let du: Vector<f64> = vector![du_left, du_right];
    let dr_u = &operators.d_r * &elt_storage.u_k.u;
    let a_rx = &elt_storage.r_x * (-a);
    let rhs_u = &a_rx.elemul(&dr_u);
    let scaled_du = &elt_storage.r_x_at_faces.elemul(&du);
    let lifted_flux = &operators.lift * scaled_du;
    let result = rhs_u + lifted_flux;
    result
}

#[derive(Debug)]
pub struct U {
    u: Vector<f64>,
}

impl Unknown for U {
    type Unit = f64;

    fn first(&self) -> f64 {
        self.u[0]
    }

    fn last(&self) -> f64 {
        self.u[self.u.size() - 1]
    }

    fn zero() -> f64 {
        0.0
    }
}

type UStorage = grid::ElementStorage<U, LinearFlux>;

type Grid = grid::Grid<Advec>;

type Element = grid::Element<Advec>;

#[derive(Clone, Copy, Debug)]
pub struct LinearFlux(pub f64);

impl grid::SpatialFlux for LinearFlux {
    type Unit = f64;

    fn first(&self) -> Self::Unit {
        self.0
    }

    fn last(&self) -> Self::Unit {
        self.0
    }

    fn zero() -> Self::Unit {
        0.
    }
}

pub struct AdvecFluxScheme {}

impl FluxScheme<U, LinearFlux> for AdvecFluxScheme {
    type Left = LaxFriedrichs;
    type Right = FreeflowFlux;
    type Interior = LaxFriedrichs;
}

pub struct Advec {}

impl GalerkinScheme for Advec {
    type U = U;
    type F = LinearFlux;
    type FS = AdvecFluxScheme;

    const FORMULATION: Formulation = Formulation::Strong;
}

fn u_0(xs: &Vector<f64>) -> U {
    U {
        u: xs.iter().map(|x: &f64| x.sin()).collect(),
    }
}

pub fn advec_1d_example() {
    let n_p = 8;
    let reference_element = ReferenceElement::legendre(n_p);
    let a = consts::PI * 2.;
    let left_boundary_face = grid::Face {
        face_type: FaceType::Boundary(Box::new(move |t: f64, _| -(a * t).sin()), a),
        flux: FluxEnum::Left(LaxFriedrichs { alpha: 1. }),
    };
    let right_boundary_face = grid::Face {
        face_type: grid::free_flow_boundary(a),
        flux: FluxEnum::Right(FreeflowFlux {}),
    };
    let grid: Grid = generate_grid(
        0.0,
        2.0,
        10,
        &reference_element,
        left_boundary_face,
        right_boundary_face,
        LaxFriedrichs { alpha: 1. },
        move |_| LinearFlux(a),
    );

    let operators = assemble_operators::<U>(&reference_element);
    advec_1d(&u_0, &grid, &reference_element, &operators, a);
}
