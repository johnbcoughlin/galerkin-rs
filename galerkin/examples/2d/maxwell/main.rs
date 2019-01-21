#![feature(test)]
#[macro_use]
extern crate galerkin;
extern crate rulinalg;
extern crate test;

use std::f64::consts;
use std::iter::repeat_with;

use rulinalg::vector::Vector;

use galerkin::blas::*;
use galerkin::distmesh::distmesh_2d::unit_square;
use galerkin::functions::range_kutta::RKA;
use galerkin::functions::range_kutta::RKB;
use galerkin::galerkin_2d::flux::compute_flux;
use galerkin::galerkin_2d::galerkin::GalerkinScheme;
use galerkin::galerkin_2d::grid::{assemble_grid, Grid};
use galerkin::galerkin_2d::grid::Element;
use galerkin::galerkin_2d::grid::ElementStorage;
use galerkin::galerkin_2d::operators::{assemble_operators, Operators};
use galerkin::galerkin_2d::operators::curl_2d;
use galerkin::galerkin_2d::operators::FaceLiftable;
use galerkin::galerkin_2d::operators::grad;
use galerkin::galerkin_2d::reference_element::ReferenceElement;
use galerkin::galerkin_2d::unknowns::{communicate, initialize_storage, Unknown};
use galerkin::plot::plot3d::{GnuplotPlotter3D, Plotter3D};

use crate::flux::*;
use crate::unknowns::*;

mod flux;
mod unknowns;

fn main() {
    maxwell_2d_example(true, 10.0);
}

#[derive(Debug)]
pub struct Maxwell2D {
    flux_scheme: Vacuum,
}

impl GalerkinScheme for Maxwell2D {
    type U = EH;
    type FS = Vacuum;
}

type EHElement<'grid> = Element<'grid, Maxwell2D>;

pub fn maxwell_2d<'grid, Fx>(
    grid: &Grid<Maxwell2D>,
    reference_element: &ReferenceElement,
    operators: &Operators,
    u_0: Fx,
    plot: bool,
    final_time: f64,
) where
    Fx: Fn(&Vector<f64>, &Vector<f64>) -> EH,
{
    let mut plotter = if plot {
        Some(GnuplotPlotter3D::create(-1., 1., -1., 1., -1., 1.))
    } else {
        None
    };

    let dt: f64 = 0.003668181816046;
    let n_t = (final_time / dt).ceil() as i32;

    let mut t: f64 = 0.0;

    let mut storage: Vec<ElementStorage<Maxwell2D>> =
        initialize_storage(u_0, reference_element, grid);

    let mut residuals: Vec<EH> = repeat_with(|| EH::zero(reference_element))
        .take(grid.elements.len())
        .collect();

    for epoch in 0..n_t {
        for int_rk in 0..5 {
            communicate(t, reference_element, grid, &mut storage);

            for elt in (*grid).elements.iter() {
                let mut storage = &mut storage[elt.index as usize];

                let residuals_eh = {
                    let residuals_eh = &(residuals[elt.index as usize]);
                    let rhs = maxwell_rhs_2d(&elt, &storage, &operators);
                    residuals_eh * RKA[int_rk] + rhs * dt
                };

                let eh = {
                    let eh: &EH = &storage.u_k;
                    eh + &(&residuals_eh * RKB[int_rk])
                };

                residuals[elt.index as usize] = residuals_eh;
                storage.u_k = eh;
            }
        }
        println!("epoch: {}", epoch);
        t = t + dt;
        if epoch % 20 == 0 {
            match plotter {
                None => {}
                Some(ref mut plotter) => {
                    plotter.header();
                    for elt in (*grid).elements.iter() {
                        let storage = &storage[elt.index as usize];
                        plotter.plot(&elt.x_k, &elt.y_k, &storage.u_k.Ez);
                    }
                    plotter.replot();
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn maxwell_rhs_2d<'grid>(
    elt: &EHElement<'grid>,
    elt_storage: &ElementStorage<Maxwell2D>,
    operators: &Operators,
) -> EH {
    let (face1_flux, face2_flux, face3_flux) = compute_flux(elt, elt_storage);

    let flux = EH::lift_faces(
        &operators.lift,
        &(face1_flux * &elt.face1.f_scale),
        &(face2_flux * &elt.face2.f_scale),
        &(face3_flux * &elt.face3.f_scale),
    );

    if elt.index == 0 {
        //        println!("flux: {}", flux);
        //        println!("current_value: {}", elt_storage.u_k);
    }

    //    println!("{:?}", elt.local_metric.jacobian);

    let grad_ez = grad(&elt_storage.u_k.Ez, operators, &elt.local_metric);
    let curl_h = curl_2d(
        &elt_storage.u_k.Hx,
        &elt_storage.u_k.Hy,
        operators,
        &elt.local_metric,
    );

    let Hx = vector_affine_(&flux.Hx, 0.5, -grad_ez.y);
    let Hy = vector_affine_(&flux.Hy, 0.5, grad_ez.x);
    let Ez = vector_affine_(&flux.Ez, 0.5, curl_h);

    EH { Hx, Hy, Ez }
}

pub fn maxwell_2d_example(plot: bool, final_time: f64) {
    let n_p = 10;
    let reference_element = ReferenceElement::legendre(n_p);
    let operators = assemble_operators(&reference_element);
    let mesh = unit_square();
    let boundary_condition =
        |_t: f64, _x: &Vector<f64>, _y: &Vector<f64>| EH::face1_zero(&reference_element);
    let grid: Grid<Maxwell2D> = assemble_grid(
        &reference_element,
        &operators,
        &mesh,
        &boundary_condition,
        &|| Null(),
        |_, _| Null(),
        MaxwellFluxType::Interior,
        MaxwellFluxType::Exterior,
    );

    //    println!("{}", operators.lift);
    maxwell_2d(
        &grid,
        &reference_element,
        &operators,
        &exact_cavity_solution_eh0,
        plot,
        final_time,
    );
}

#[allow(non_snake_case)]
fn exact_cavity_solution_eh0(xs: &Vector<f64>, ys: &Vector<f64>) -> EH {
    let pi = consts::PI;
    let omega = pi * consts::SQRT_2;
    let t = 0.;
    let Hx: Vector<f64> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| -pi / omega * (pi * x).sin() * (pi * y).cos() * (omega * t).sin())
        .collect();
    let Hy: Vector<f64> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| pi / omega * (pi * x).cos() * (pi * y).sin() * (omega * t).sin())
        .collect();
    let Ez: Vector<f64> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (pi * x).sin() * (pi * y).sin() * (omega * t).cos())
        .collect();

    EH { Hx, Hy, Ez }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use test::Bencher;

    use super::*;

    #[bench]
    pub fn bench(b: &mut Bencher) {
        let n_p = 10;
        let reference_element = ReferenceElement::legendre(n_p);
        let operators = assemble_operators(&reference_element);
        let mesh = unit_square();
        let boundary_condition =
            |_t: f64, _x: &Vector<f64>, _y: &Vector<f64>| EH::face1_zero(&reference_element);
        let grid: Grid<Maxwell2D> = assemble_grid(
            &reference_element,
            &operators,
            &mesh,
            &boundary_condition,
            &|| Null(),
            |_, _| Null(),
            MaxwellFluxType::Interior,
            MaxwellFluxType::Exterior,
        );

        b.iter(|| {
            maxwell_2d(
                &grid,
                &reference_element,
                &operators,
                &exact_cavity_solution_eh0,
                false,
                0.01,
            )
        });
    }
}
