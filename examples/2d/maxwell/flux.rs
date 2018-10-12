extern crate galerkin;
extern crate rulinalg;

use galerkin::galerkin_2d::flux::{FluxKey, FluxScheme, Side};
use galerkin::galerkin_2d::grid::SpatialVariable;
use galerkin::galerkin_2d::grid::Vec2;
use galerkin::galerkin_2d::reference_element::ReferenceElement;
use rulinalg::vector::Vector;
use unknowns::{EH};
use galerkin::blas;

#[derive(Debug, Clone, Copy)]
pub struct Null();

impl SpatialVariable for Null {
    type Line = Null;

    fn edge_1(&self, _reference_element: &ReferenceElement) -> Null {
        Null()
    }

    fn edge_2(&self, _reference_element: &ReferenceElement) -> Null {
        Null()
    }

    fn edge_3(&self, _reference_element: &ReferenceElement) -> Null {
        Null()
    }

    fn face1_zero(_reference_element: &ReferenceElement) -> Null {
        Null()
    }

    fn face2_zero(_reference_element: &ReferenceElement) -> Null {
        Null()
    }

    fn face3_zero(_reference_element: &ReferenceElement) -> Null {
        Null()
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MaxwellFluxType {
    Interior,
    Exterior,
}

impl FluxKey for MaxwellFluxType {}

#[derive(Debug)]
pub struct Vacuum {}

impl Vacuum {
    fn interior_flux(
        minus: Side<EH, Null>,
        plus: Side<EH, Null>,
        outward_normal: &[Vec2],
    ) -> EH {
        let d_eh = minus.u - plus.u;
        let (d_hx, d_hy, d_ez) = (d_eh.Hx, d_eh.Hy, d_eh.Ez);
        Self::flux_calculation(d_hx, d_hy, d_ez, outward_normal)
    }

    fn exterior_flux(
        minus: Side<EH, Null>,
        _plus: Side<EH, Null>,
        outward_normal: &[Vec2],
    ) -> EH {
        let d_hx = Vector::zeros(minus.u.Hx.size());
        let d_hy = Vector::zeros(minus.u.Hy.size());
        let d_ez = &minus.u.Ez * 2.;
        Self::flux_calculation(d_hx, d_hy, d_ez, outward_normal)
    }

    fn flux_calculation(
        d_hx: Vector<f64>,
        d_hy: Vector<f64>,
        d_ez: Vector<f64>,
        outward_normal: &[Vec2],
    ) -> EH {
        let alpha = 1.;
        let (n_x, n_y): (Vector<f64>, Vector<f64>) = (
            outward_normal.iter().map(|ref n| n.x).collect(),
            outward_normal.iter().map(|ref n| n.y).collect(),
        );

        let n_dot_dh = blas::elemul_affine_(&d_hx, &n_x, 1., blas::elemul(&d_hy, &n_y), 1.);

        // -d_hy.*n_x + d_hx.*n_y - alpha * d_ez
        let flux_ez = blas::elemul_affine_(&d_hy, &n_x, -1., blas::elemul_affine(&d_hx, &n_y, 1., &d_ez, -alpha), 1.);

        let flux_hx = blas::elemul_affine_(&d_ez, &n_y, 1., blas::elemul_affine_(&n_dot_dh, &n_x, 1., d_hx, -1.), alpha);

        // -d_ez.*n_x + alpha(n_dot_dh.*n_y - d_hy)
        let flux_hy = blas::elemul_affine_(&d_ez, &n_x, -1., blas::elemul_affine_(&n_dot_dh, &n_y, 1., d_hy, -1.), alpha);

        EH {
            Ez: flux_ez,
            Hx: flux_hx,
            Hy: flux_hy,
        }
    }
}

impl FluxScheme<EH> for Vacuum {
    // In the vacuum, we can normalize all constants to 1, and there is no spatial variation
    // of the permittivity, so the spatial variable is ().
    type F = Null;
    type K = MaxwellFluxType;

    fn flux_type(
        key: Self::K,
        minus: Side<EH, Null>,
        plus: Side<EH, Null>,
        outward_normal: &[Vec2],
    ) -> EH {
        match key {
            MaxwellFluxType::Interior => Vacuum::interior_flux(minus, plus, &outward_normal),
            MaxwellFluxType::Exterior => Vacuum::exterior_flux(minus, plus, &outward_normal),
        }
    }
}
