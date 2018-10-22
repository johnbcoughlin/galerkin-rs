extern crate galerkin;
extern crate rulinalg;

use galerkin::blas::matrix_multiply;
use galerkin::galerkin_2d::operators::FaceLift;
use galerkin::galerkin_2d::operators::FaceLiftable;
use galerkin::galerkin_2d::reference_element::ReferenceElement;
use galerkin::galerkin_2d::unknowns::Unknown;
use rulinalg::vector::Vector;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

unknown_from_vector_fields!(EH, Ez, Hx, Hy);

impl fmt::Display for EH {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        writeln!(f, "EH {{")?;
        writeln!(f, "  Ez: {}", self.Ez)?;
        writeln!(f, "  Hx: {}", self.Hx)?;
        writeln!(f, "  Hy: {}", self.Hy)?;
        writeln!(f, "}}")
    }
}

impl FaceLiftable for EH {
    fn lift_faces(
        face_lift: &FaceLift,
        face1: &<Self as Unknown>::Line,
        face2: &<Self as Unknown>::Line,
        face3: &<Self as Unknown>::Line,
    ) -> Self {
        let face1_lifted = EH {
            Ez: matrix_multiply(&face_lift.face1, &face1.Ez),
            Hx: matrix_multiply(&face_lift.face1, &face1.Hx),
            Hy: matrix_multiply(&face_lift.face1, &face1.Hy),
        };
        let face2_lifted = EH {
            Ez: matrix_multiply(&face_lift.face2, &face2.Ez),
            Hx: matrix_multiply(&face_lift.face2, &face2.Hx),
            Hy: matrix_multiply(&face_lift.face2, &face2.Hy),
        };
        let face3_lifted = EH {
            Ez: matrix_multiply(&face_lift.face3, &face3.Ez),
            Hx: matrix_multiply(&face_lift.face3, &face3.Hx),
            Hy: matrix_multiply(&face_lift.face3, &face3.Hy),
        };
        face1_lifted + face2_lifted + face3_lifted
    }
}
