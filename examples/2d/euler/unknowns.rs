extern crate galerkin;
extern crate rulinalg;

use rulinalg::vector::Vector;
use std::ops::{Add, Div, Mul, Neg, Sub};
use galerkin::blas::{
    matrix_multiply,
    elemul,
    vector_add,
    vector_add_,
    vector_sub,
    vector_sub_,
    vector_scale,
    vector_scale_,
};

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct Q {
    pub rho: Vector<f64>,
    pub rho_u: Vector<f64>,
    pub rho_v: Vector<f64>,
    pub E: Vector<f64>,
}

macro_rules! operators {
    ($strct:ident, $($field:ident), *) => {
        impl Add for $strct {
            type Output = Self;

            fn add(self, rhs: $strct) -> $strct {
                $strct {
                $(
                    $field: vector_add_(&self.$field, rhs.$field),
                )*
                }
            }
        }
    }
}

operators!(Q, rho, rho_u, rho_v, E);
