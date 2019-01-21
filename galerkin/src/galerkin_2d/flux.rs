use std::fmt::Debug;
use std::ops::Deref;

use rulinalg::vector::Vector;

use crate::galerkin_2d::galerkin::GalerkinScheme;
use crate::galerkin_2d::grid::Element;
use crate::galerkin_2d::grid::ElementStorage;
use crate::galerkin_2d::grid::SpatialVariable;
use crate::galerkin_2d::unknowns::Unknown;

#[derive(Debug)]
pub struct Side<'iter, U, F>
where
    U: Unknown,
    F: SpatialVariable,
    <U as Unknown>::Line: 'iter,
    F::Line: 'iter,
{
    pub u: &'iter <U as Unknown>::Line,
    pub f: &'iter F::Line,
}

pub trait FluxKey: Copy + Debug {}

pub trait FluxScheme<U>: Debug
where
    U: Unknown,
{
    type F: SpatialVariable;
    type K: FluxKey;

    fn flux_type<'iter>(
        key: Self::K,
        minus: Side<'iter, U, Self::F>,
        plus: Side<'iter, U, Self::F>,
        outward_normal_x: &Vector<f64>,
        outward_normal_y: &Vector<f64>,
    ) -> U::Line;
}

pub trait NumericalFlux<U, F>
where
    U: Unknown,
    F: SpatialVariable,
{
    fn flux<'iter>(
        &self,
        minus: Side<'iter, U, F>,
        plus: Side<'iter, U, F>,
        outward_normal_x: &Vector<f64>,
        outward_normal_y: &Vector<f64>,
    ) -> U::Line;
}

pub fn compute_flux<'grid, GS>(
    elt: &Element<'grid, GS>,
    elt_storage: &ElementStorage<GS>,
) -> (
    <GS::U as Unknown>::Line,
    <GS::U as Unknown>::Line,
    <GS::U as Unknown>::Line,
)
where
    GS: GalerkinScheme,
{
    let face1_flux = {
        let u_minus = elt_storage.u_face1_minus.borrow();
        let minus = Side {
            u: u_minus.deref(),
            f: &elt_storage.f_face1_minus,
        };
        let u_plus = elt_storage.u_face1_plus.borrow();
        let plus = Side {
            u: u_plus.deref(),
            f: &elt_storage.f_face1_plus,
        };
        <GS::FS as FluxScheme<GS::U>>::flux_type(
            elt.face1.flux_key,
            minus,
            plus,
            &elt.face1.outward_normal_x,
            &elt.face1.outward_normal_y,
        )
    };
    let face2_flux = {
        let u_minus = elt_storage.u_face2_minus.borrow();
        let minus = Side {
            u: u_minus.deref(),
            f: &elt_storage.f_face2_minus,
        };
        let u_plus = elt_storage.u_face2_plus.borrow();
        let plus = Side {
            u: u_plus.deref(),
            f: &elt_storage.f_face2_plus,
        };
        <GS::FS as FluxScheme<GS::U>>::flux_type(
            elt.face2.flux_key,
            minus,
            plus,
            &elt.face2.outward_normal_x,
            &elt.face2.outward_normal_y,
        )
    };
    let face3_flux = {
        let u_minus = elt_storage.u_face3_minus.borrow();
        let minus = Side {
            u: u_minus.deref(),
            f: &elt_storage.f_face3_minus,
        };
        let u_plus = elt_storage.u_face3_plus.borrow();
        let plus = Side {
            u: u_plus.deref(),
            f: &elt_storage.f_face3_plus,
        };
        <GS::FS as FluxScheme<GS::U>>::flux_type(
            elt.face3.flux_key,
            minus,
            plus,
            &elt.face3.outward_normal_x,
            &elt.face3.outward_normal_y,
        )
    };

    (face1_flux, face2_flux, face3_flux)
}
