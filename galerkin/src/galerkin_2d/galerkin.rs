extern crate rulinalg;

use crate::galerkin_2d::flux::FluxScheme;
use crate::galerkin_2d::operators::FaceLiftable;
use crate::galerkin_2d::unknowns::Unknown;
use std::fmt::Debug;

pub trait GalerkinScheme {
    type U: Unknown + FaceLiftable + Debug;
    type FS: FluxScheme<Self::U>;
}
