#![feature(test)]
#[macro_use]
extern crate galerkin;
#[macro_use]
extern crate rulinalg;

use flux::*;
use galerkin::galerkin_2d::flux::compute_flux;
use galerkin::galerkin_2d::galerkin::GalerkinScheme;
use galerkin::galerkin_2d::grid::{assemble_grid, Element, ElementStorage, Grid};
use galerkin::galerkin_2d::operators::{assemble_operators, cutoff_filter, FaceLiftable, Operators};
use galerkin::galerkin_2d::reference_element::ReferenceElement;
use galerkin::distmesh::distmesh_2d;
use galerkin::galerkin_2d::unknowns::Unknown;
use rulinalg::vector::Vector;
use unknowns::*;
use std::f64::consts;
use galerkin::galerkin_2d::grid::FaceNumber;
use galerkin::galerkin_2d::unknowns::initialize_storage;
use std::iter::repeat_with;
use galerkin::galerkin_2d::unknowns::communicate;
use galerkin::functions::range_kutta::*;
use galerkin::plot::plot3d::{Plotter3D, GnuplotPlotter3D};
use galerkin::plot::glium::run_inside_plot;
use std::sync::mpsc::Sender;

mod flux;
mod unknowns;

fn main() {
    euler_2d_example();
}

pub fn euler_2d_example() {
    let n_p = 9;
    let reference_element = ReferenceElement::legendre(n_p);
    let operators = assemble_operators(&reference_element);
    let mesh = distmesh_2d::isentropic_vortex();

    let grid: Grid<Euler2D> = assemble_grid(
        &reference_element,
        &operators,
        &mesh,
        &isentropic_vortex,
        &|| (),
        |_, _| (),
        EulerFluxType::LF,
        EulerFluxType::LF,
    );

    let expanded_mesh = grid.to_mesh(&reference_element);
    run_inside_plot(expanded_mesh, move |sender| {
        euler_2d(
            &grid,
            &reference_element,
            &operators,
            &isentropic_vortex,
            1.,
            false,
            sender,
        );
    });
}

fn euler_2d<'grid, Fx>(
    grid: &Grid<Euler2D>,
    reference_element: &ReferenceElement,
    operators: &Operators,
    exact_solution: Fx,
    final_time: f64,
    plot: bool,
    sender: Sender<Vec<f64>>,
) where
    Fx: Fn(f64, &Vector<f64>, &Vector<f64>) -> Q,
{
    let mut storage: Vec<ElementStorage<Euler2D>> = initialize_storage(
        |x, y| exact_solution(0., x, y),
        reference_element,
        grid,
    );
    let mut residuals: Vec<Q> = repeat_with(|| Q::zero(reference_element))
        .take(grid.elements.len())
        .collect();

    let filter = cutoff_filter(operators, reference_element.n, 0.95);

    for s in storage.iter_mut() {
        s.u_k = s.u_k.matrix_multiply(&filter);
    }

    let mut t: f64 = 0.0;

    let mut dt = timestep(
        grid,
        &storage,
        reference_element,
    );

    let mut epoch = 0;
    while t < final_time {
        for int_rk in 0..5 {
            communicate(t, reference_element, grid, &mut storage);

            for elt in (*grid).elements.iter() {
                let mut storage = &mut storage[elt.index as usize];

                let residuals_q = {
                    let residuals_q = &(residuals[elt.index as usize]);
                    let rhs = euler_rhs_2d(&elt, &storage, &operators);
                    residuals_q * RKA[int_rk] + rhs * dt
                }.matrix_multiply(&filter);

                let q = {
                    let q: &Q = &storage.u_k;
                    q + &(&residuals_q * RKB[int_rk])
                };

                residuals[elt.index as usize] = residuals_q;
                storage.u_k = q;
            }
        }
        println!("time: {}", t);
        t = t + dt;
        dt = timestep(
            grid,
            &storage,
            reference_element,
        );
        // plot
        {
            let rho_u = (*grid).elements.iter()
                .flat_map(|ref elt| storage[elt.index as usize].u_k.rho_u.iter())
                .map(|&e| e)
                .collect();
            sender.send(rho_u);
        }
        epoch += 1;
    }
}

fn euler_rhs_2d<'grid>(
    elt: &Element<'grid, Euler2D>,
    elt_storage: &ElementStorage<Euler2D>,
    operators: &Operators,
) -> Q {
    let volume_term = {
        let (f, g, uvp) = elt_storage.u_k.to_FG_UVP();
        let df_dr = f.matrix_multiply(&operators.d_r_w);
        let df_ds = f.matrix_multiply(&operators.d_s_w);
        let dg_dr = g.matrix_multiply(&operators.d_r_w);
        let dg_ds = g.matrix_multiply(&operators.d_s_w);
        (df_dr * &elt.local_metric.r_x + df_ds * &elt.local_metric.s_x)
            + (dg_dr * &elt.local_metric.r_y + dg_ds * &elt.local_metric.s_y)
    };

    // returns nflux / 2
    let (face1_flux, face2_flux, face3_flux) = compute_flux(elt, elt_storage);
    let surface_term = Q::lift_faces(
        &operators.lift,
        &(face1_flux * &elt.face1.f_scale),
        &(face2_flux * &elt.face2.f_scale),
        &(face3_flux * &elt.face3.f_scale),
    );

    volume_term - surface_term
}

fn timestep(
    grid: &Grid<Euler2D>,
    storages: &Vec<ElementStorage<Euler2D>>,
    reference_element: &ReferenceElement,
) -> f64 {
    let mut dt_max_recip: f64 = 0.;
    for elt in &grid.elements {
        let elt_storage = &storages[elt.index as usize];
        let q = &elt_storage.u_k;

        let face_max = |face_num: FaceNumber| {
            let q: Q = q.face(face_num, reference_element);
            let (_, _, uvp) = q.to_FG_UVP();
            let c: Vector<f64> = ((uvp.p.elediv(&q.rho)) * Q::GAMMA).into_iter()
                .map(|x| x.abs().sqrt())
                .collect();
            let uv_norm: Vector<f64> = (uvp.u.elemul(&uvp.u) + uvp.v.elemul(&uvp.v)).into_iter()
                .map(&f64::sqrt)
                .collect();
            let n = reference_element.n as f64;
            let term = &elt.face(face_num).f_scale * (n + 1.) * (n + 1.) * 0.5;
            (term.elemul(&(uv_norm + c))).argmax().1
        };
        dt_max_recip = dt_max_recip.max(
            face_max(FaceNumber::One).max(
                face_max(FaceNumber::Two).max(
                    face_max(FaceNumber::Three))));
    }
    1. / dt_max_recip
}

fn isentropic_vortex(t: f64, x: &Vector<f64>, y: &Vector<f64>) -> Q {
    let x_0: f64 = 5.;
    let y_0: f64 = 0.;
    let beta: f64 = 5.;
    let gamma: f64 = Q::GAMMA;

    let xt = (x - t - x_0);
    let yt = (y - y_0);
    let r: Vector<f64> = (xt.elemul(&xt) + yt.elemul(&yt)).apply(&f64::sqrt);
    let beta_exp = r.apply(&|r: f64| f64::exp(1. - r * r)) * beta;

    let u = -(y - y_0).elemul(&beta_exp) / (2. * consts::PI) + 1.;
    let v = (x - t - x_0).elemul(&beta_exp) / (2. * consts::PI);
    let rho = (-beta_exp.elemul(&beta_exp) * (gamma - 1.) /
        (16. * gamma * consts::PI * consts::PI) + 1.)
        .apply(&|x| x.powf(1. / (gamma - 1.)));
    let rho2 = rho.clone();
    let p = rho.apply(&|x| x.powf(gamma));

    Q {
        rho: rho2.clone(),
        rho_u: rho2.elemul(&u),
        rho_v: rho2.elemul(&v),
        E: p / (gamma - 1.) + rho2.elemul(&(u.elemul(&u) + v.elemul(&v))) * 0.5,
    }
}

struct Euler2D;

impl GalerkinScheme for Euler2D {
    type U = Q;
    type FS = EulerFlux;
}

#[cfg(test)]
mod tests {
    extern crate rulinalg;

    use super::*;
    use rulinalg::vector::Vector;

    #[test]
    fn test_isentropic_vortex_t0() {
        let q = isentropic_vortex(0., &vector![5.1], &vector![0.2]);
        println!("{:?}", q);
        assert_eq!(q.rho[0], 0.40642807906722056);
        assert_eq!(q.rho_u[0], 0.2391713520277582);
        assert_eq!(q.rho_v[0], 0.0836283635197309);
        assert_eq!(q.E[0], 0.7877659976239465);
    }

    #[test]
    fn test_isentropic_vortex_t1() {
        let q = isentropic_vortex(1., &vector![5.1], &vector![0.2]);
        println!("{:?}", q);
        assert_eq!(q.rho[0], 0.8542740991590158);
        assert_eq!(q.rho_u[0], 0.6963088550105302);
        assert_eq!(q.rho_v[0], -0.7108435986681854);
        assert_eq!(q.E[0], 2.5848091516382796);
    }
}
