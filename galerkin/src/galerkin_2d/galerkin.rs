extern crate rulinalg;

use std::fmt::Debug;

use crate::galerkin_2d::flux::FluxScheme;
use crate::galerkin_2d::operators::FaceLiftable;
use crate::galerkin_2d::unknowns::Unknown;

pub trait GalerkinScheme {
    type U: Unknown + FaceLiftable + Debug;
    type FS: FluxScheme<Self::U>;
}
