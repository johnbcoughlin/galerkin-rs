extern crate galerkin;
extern crate rulinalg;

use galerkin::galerkin_2d::flux::*;
use rulinalg::vector::Vector;
use unknowns::*;

#[derive(Debug, Copy, Clone)]
pub enum EulerFluxType {
    LF,
}

impl FluxKey for EulerFluxType {}

#[derive(Debug)]
pub struct EulerFlux;

impl EulerFlux {
    fn lax_friedrichs(
        minus: Side<Q, ()>,
        plus: Side<Q, ()>,
        outward_normal_x: &Vector<f64>,
        outward_normal_y: &Vector<f64>,
    ) -> Q {
        let process = |side: &Side<Q, ()>| {
            let rho = &side.u.rho;
            let (f, g, uvp) = side.u.to_FG_UVP();
            let (u, v, p) = (uvp.u, uvp.v, uvp.p);
            let lambda = (u.elemul(&u) + v.elemul(&v)).apply(&f64::sqrt)
                + (p.elediv(&rho) * Q::GAMMA).apply(&|x: f64| x.abs().sqrt());
            (f, g, lambda)
        };
        let (f_minus, g_minus, lambda_minus) = process(&minus);
        let (f_plus, g_plus, lambda_plus) = process(&plus);
        let lambda: Vector<f64> = lambda_minus
            .into_iter()
            .zip(lambda_plus.into_iter())
            .map(|(m, p)| m.max(p))
            .collect();
        let f_flux = (&f_plus + &f_minus) * outward_normal_x;
        let g_flux = (&g_plus + &g_minus) * outward_normal_y;

        ((f_flux + g_flux) + (minus.u - plus.u) * &lambda) * 0.5
    }
}

impl FluxScheme<Q> for EulerFlux {
    type F = ();
    type K = EulerFluxType;

    fn flux_type(
        key: EulerFluxType,
        minus: Side<Q, ()>,
        plus: Side<Q, ()>,
        outward_normal_x: &Vector<f64>,
        outward_normal_y: &Vector<f64>,
    ) -> Q {
        match key {
            EulerFluxType::LF => {
                EulerFlux::lax_friedrichs(minus, plus, outward_normal_x, outward_normal_y)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate rulinalg;
    extern crate std;

    use super::super::unknowns::*;
    use super::*;
    use rulinalg::vector::Vector;
    use std::f64::consts;

    #[test]
    fn test_lax_friedrichs() {
        let q_minus = Q {
            rho: vector![1., 1., 1., 1., 1., 1., 1.],
            rho_u: vector![1., 1., 1., 1., 1., 1., 1.],
            rho_v: vector![
                -1.502078308897470e-10,
                -3.922177487488449e-10,
                -2.368813487705917e-09,
                -1.216915023201200e-08,
                -5.316720217008582e-08,
                -1.975148273437542e-07,
                -3.582105865656032e-07
            ],
            E: vector![3., 3., 3., 3., 3., 3., 3.],
        };
        let q_plus = &q_minus * 2.;
        let outward_normal_x = vector![
            -consts::SQRT_2,
            -consts::SQRT_2,
            -consts::SQRT_2,
            -consts::SQRT_2,
            -consts::SQRT_2,
            -consts::SQRT_2,
            -consts::SQRT_2
        ] / 2.;
        let outward_normal_y = -&outward_normal_x;
        let actual = EulerFlux::lax_friedrichs(
            Side {
                u: &q_minus,
                f: &(),
            },
            Side { u: &q_plus, f: &() },
            &outward_normal_x,
            &outward_normal_y,
        );
        let rho_expected = vector![
            -2.1522681502491023,
            -2.152268150505793,
            -2.1522681526022893,
            -2.152268162997116,
            -2.152268206482117,
            -2.152268359585901,
            -2.15226853002951
        ];
        assert_eq!(actual.rho, rho_expected);
        let rho_u_expected = vector![
            -3.2129283220289238,
            -3.212928322285614,
            -3.2129283243821103,
            -3.212928334776937,
            -3.212928378261938,
            -3.212928531365714,
            -3.212928701809304
        ];
        assert_eq!(actual.rho_u, rho_u_expected);
        let rho_v_expected = vector![
            1.0606601721031086,
            1.0606601726239786,
            1.060660176878143,
            1.0606601979710957,
            1.0606602862098995,
            1.0606605968847262,
            1.0606609427451668
        ];
        assert_eq!(actual.rho_v, rho_v_expected);
        let E_expected = vector![
            -7.5174646226864485,
            -7.5174646237132094,
            -7.517464632099196,
            -7.517464673678503,
            -7.5174648476185055,
            -7.517465460033627,
            -7.5174661418080255
        ];
        assert_eq!(actual.E, E_expected);
    }
}
