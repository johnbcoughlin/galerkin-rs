extern crate galerkin;
extern crate rulinalg;

use galerkin::blas::matrix_multiply;
use galerkin::galerkin_2d::operators::{FaceLift, FaceLiftable};
use galerkin::galerkin_2d::reference_element::ReferenceElement;
use galerkin::galerkin_2d::unknowns::Unknown;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use std::ops::{Add, Div, Mul, Neg, Sub};

unknown_from_vector_fields!(Q, rho, rho_u, rho_v, E);

unknown_from_vector_fields!(F, a, b, c, d);

unknown_from_vector_fields!(G, a, b, c, d);

unknown_from_vector_fields!(UVP, u, v, p);

impl Q {
    pub const GAMMA: f64 = 1.4;

    #[allow(non_snake_case)]
    pub fn to_FG_UVP(&self) -> (F, G, UVP) {
        let u = self.rho_u.elediv(&self.rho);
        let v = self.rho_v.elediv(&self.rho);
        let p = (&self.E - (self.rho_u.elemul(&u) + self.rho_v.elemul(&v)) * 0.5) * (Q::GAMMA - 1.);

        let f = F {
            a: self.rho_u.clone(),
            b: self.rho_u.elemul(&u) + &p,
            c: self.rho_u.elemul(&v),
            d: u.elemul(&(&self.E + &p)),
        };
        let g = G {
            a: self.rho_v.clone(),
            b: self.rho_u.elemul(&v),
            c: self.rho_v.elemul(&v) + &p,
            d: v.elemul(&(&self.E + &p)),
        };
        (f, g, UVP { u, v, p })
    }

    // Computes m * q
    pub fn matrix_multiply(&self, m: &Matrix<f64>) -> Q {
        Q {
            rho: matrix_multiply(m, &self.rho),
            rho_u: matrix_multiply(m, &self.rho_u),
            rho_v: matrix_multiply(m, &self.rho_v),
            E: matrix_multiply(m, &self.E),
        }
    }
}

impl FaceLiftable for Q {
    fn lift_faces(face_lift: &FaceLift, face1: &Q, face2: &Q, face3: &Q) -> Q {
        let from_face1 = face1.matrix_multiply(&face_lift.face1);
        let from_face2 = face2.matrix_multiply(&face_lift.face2);
        let from_face3 = face3.matrix_multiply(&face_lift.face3);
        from_face1 + from_face2 + from_face3
    }
}

impl F {
    // Computes m * f
    pub fn matrix_multiply(&self, m: &Matrix<f64>) -> Q {
        Q {
            rho: matrix_multiply(m, &self.a),
            rho_u: matrix_multiply(m, &self.b),
            rho_v: matrix_multiply(m, &self.c),
            E: matrix_multiply(m, &self.d),
        }
    }
}

impl G {
    // Computes m * f
    pub fn matrix_multiply(&self, m: &Matrix<f64>) -> Q {
        Q {
            rho: matrix_multiply(m, &self.a),
            rho_u: matrix_multiply(m, &self.b),
            rho_v: matrix_multiply(m, &self.c),
            E: matrix_multiply(m, &self.d),
        }
    }
}

impl Add<G> for F {
    type Output = Q;

    fn add(self, rhs: G) -> Q {
        Q {
            rho: self.a + rhs.a,
            rho_u: self.b + rhs.b,
            rho_v: self.c + rhs.c,
            E: self.d + rhs.d,
        }
    }
}
