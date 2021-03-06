extern crate rulinalg;

use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg};
use std::ops::Sub;

use rulinalg::vector::Vector;

use crate::galerkin_2d::galerkin::GalerkinScheme;
use crate::galerkin_2d::grid::{ElementStorage, FaceNumber, FaceType, Grid, SpatialVariable};
use crate::galerkin_2d::reference_element::ReferenceElement;

pub trait Unknown
where
    Self::Line: Neg<Output = Self::Line>
        + Add<Output = Self::Line>
        + Sub<Output = Self::Line>
        + Mul<f64, Output = Self::Line>
        + for<'a> Mul<&'a Vector<f64>, Output = Self::Line>
        + Div<f64, Output = Self::Line>
        + fmt::Debug,
{
    type Line;

    fn zero(reference_element: &ReferenceElement) -> Self;

    fn slice(&self, slice: &[usize]) -> Self::Line;

    fn edge_1(&self, reference_element: &ReferenceElement) -> Self::Line;

    fn edge_2(&self, reference_element: &ReferenceElement) -> Self::Line;

    fn edge_3(&self, reference_element: &ReferenceElement) -> Self::Line;

    fn face(&self, number: FaceNumber, reference_element: &ReferenceElement) -> Self::Line {
        match number {
            FaceNumber::One => self.edge_1(reference_element),
            FaceNumber::Two => self.edge_2(reference_element),
            FaceNumber::Three => self.edge_3(reference_element),
        }
    }

    fn face_flipped(&self, number: FaceNumber, reference_element: &ReferenceElement) -> Self::Line {
        let mut slice = reference_element.face(number).clone();
        let slice = slice.as_mut_slice();
        slice.reverse();
        self.slice(slice)
    }

    fn face1_zero(reference_element: &ReferenceElement) -> Self::Line;

    fn face2_zero(reference_element: &ReferenceElement) -> Self::Line;

    fn face3_zero(reference_element: &ReferenceElement) -> Self::Line;
}

pub fn initialize_storage<GS, Fx>(
    u_0: Fx,
    reference_element: &ReferenceElement,
    grid: &Grid<GS>,
) -> Vec<ElementStorage<GS>>
where
    GS: GalerkinScheme,
    Fx: Fn(&Vector<f64>, &Vector<f64>) -> GS::U,
{
    let mut result: Vec<ElementStorage<GS>> = vec![];
    for (_i, elt) in grid.elements.iter().enumerate() {
        let (f_face1_minus, f_face1_plus) = match elt.face1.face_type {
            FaceType::Interior(j, face_number) => (
                elt.spatial_parameters.edge_1(reference_element),
                grid.elements[j as usize]
                    .spatial_parameters
                    .face(face_number, reference_element),
            ),
            FaceType::Boundary(_, f) => (elt.spatial_parameters.edge_1(reference_element), f()),
        };
        let (f_face2_minus, f_face2_plus) = match elt.face2.face_type {
            FaceType::Interior(j, face_number) => (
                elt.spatial_parameters.edge_2(reference_element),
                grid.elements[j as usize]
                    .spatial_parameters
                    .face(face_number, reference_element),
            ),
            FaceType::Boundary(_, f) => (elt.spatial_parameters.edge_2(reference_element), f()),
        };
        let (f_face3_minus, f_face3_plus) = match elt.face3.face_type {
            FaceType::Interior(j, face_number) => (
                elt.spatial_parameters.edge_3(reference_element),
                grid.elements[j as usize]
                    .spatial_parameters
                    .face(face_number, reference_element),
            ),
            FaceType::Boundary(_, f) => (elt.spatial_parameters.edge_3(reference_element), f()),
        };
        result.push(ElementStorage {
            u_k: u_0(&elt.x_k, &elt.y_k),
            u_face1_minus: RefCell::new(GS::U::face1_zero(reference_element)),
            u_face1_plus: RefCell::new(GS::U::face1_zero(reference_element)),
            u_face2_minus: RefCell::new(GS::U::face2_zero(reference_element)),
            u_face2_plus: RefCell::new(GS::U::face2_zero(reference_element)),
            u_face3_minus: RefCell::new(GS::U::face3_zero(reference_element)),
            u_face3_plus: RefCell::new(GS::U::face3_zero(reference_element)),

            f_face1_minus: f_face1_minus,
            f_face1_plus: f_face1_plus,
            f_face2_minus: f_face2_minus,
            f_face2_plus: f_face2_plus,
            f_face3_minus: f_face3_minus,
            f_face3_plus: f_face3_plus,
        });
    }
    result
}

pub fn communicate<GS>(
    t: f64,
    reference_element: &ReferenceElement,
    grid: &Grid<GS>,
    storages: &mut Vec<ElementStorage<GS>>,
) where
    GS: GalerkinScheme,
{
    for (i, elt) in grid.elements.iter().enumerate() {
        let storage = &storages[i];
        let mut u_k: &GS::U = &storage.u_k;

        let face1 = u_k.edge_1(reference_element);
        let (face1_minus, face1_plus) = match elt.face1.face_type {
            FaceType::Interior(j, face_number) => {
                let u_k_neighbor: &GS::U = &storages[j as usize].u_k;
                // minus is interior, plus is neighbor
                (
                    face1,
                    u_k_neighbor.face_flipped(face_number, reference_element),
                )
            }
            FaceType::Boundary(bc, _) => {
                // minus is interior, plus is neighbor
                (
                    face1,
                    bc(
                        t,
                        &elt.face_x(FaceNumber::One, reference_element),
                        &elt.face_y(FaceNumber::One, reference_element),
                    ),
                )
            }
        };
        storage.u_face1_minus.replace(face1_minus);
        storage.u_face1_plus.replace(face1_plus);

        let face2 = u_k.edge_2(reference_element);
        let (face2_minus, face2_plus) = match elt.face2.face_type {
            FaceType::Interior(j, face_number) => {
                let u_k_neighbor: &GS::U = &storages[j as usize].u_k;
                // minus is interior, plus is neighbor
                (
                    face2,
                    u_k_neighbor.face_flipped(face_number, reference_element),
                )
            }
            FaceType::Boundary(bc, _) => {
                // minus is interior, plus is neighbor
                (
                    face2,
                    bc(
                        t,
                        &elt.face_x(FaceNumber::Two, reference_element),
                        &elt.face_y(FaceNumber::Two, reference_element),
                    ),
                )
            }
        };
        storage.u_face2_minus.replace(face2_minus);
        storage.u_face2_plus.replace(face2_plus);

        let face3 = u_k.edge_3(reference_element);
        let (face3_minus, face3_plus) = match elt.face3.face_type {
            FaceType::Interior(j, face_number) => {
                let u_k_neighbor: &GS::U = &storages[j as usize].u_k;
                // minus is interior, plus is neighbor
                (
                    face3,
                    u_k_neighbor.face_flipped(face_number, reference_element),
                )
            }
            FaceType::Boundary(bc, _) => {
                // minus is interior, plus is neighbor
                (
                    face3,
                    bc(
                        t,
                        &elt.face_x(FaceNumber::Three, reference_element),
                        &elt.face_y(FaceNumber::Three, reference_element),
                    ),
                )
            }
        };
        storage.u_face3_minus.replace(face3_minus);
        storage.u_face3_plus.replace(face3_plus);
    }
}

#[macro_export]
macro_rules! unknown_from_vector_fields {
    ($U:ident, $($field:ident),*) => {
        #[allow(non_snake_case)]
        #[derive(Debug)]
        pub struct $U { $(pub $field: Vector<f64>, )* }

        // Implement the Unknown trait
        impl Unknown for $U {
            type Line = $U;

            fn slice(&self, slice: &[usize]) -> Self::Line {
                $U { $($field: self.$field.select(slice), )* }
            }

            fn edge_1(&self, reference_element: &ReferenceElement) -> Self::Line {
                $U { $($field: self.$field.select(reference_element.face1.as_slice()), )* }
            }

            fn edge_2(&self, reference_element: &ReferenceElement) -> Self::Line {
                $U { $($field: self.$field.select(reference_element.face2.as_slice()), )* }
            }

            fn edge_3(&self, reference_element: &ReferenceElement) -> Self::Line {
                $U { $($field: self.$field.select(reference_element.face3.as_slice()), )* }
            }

            fn zero(reference_element: &ReferenceElement) -> Self {
                use rulinalg::vector::Vector;
                $U { $($field: Vector::zeros(reference_element.n_p), )* }
            }

            fn face1_zero(reference_element: &ReferenceElement) -> Self::Line {
                use rulinalg::vector::Vector;
                $U { $($field: Vector::zeros(reference_element.face1.len()), )* }
            }

            fn face2_zero(reference_element: &ReferenceElement) -> Self::Line {
                use rulinalg::vector::Vector;
                $U { $($field: Vector::zeros(reference_element.face2.len()), )* }
            }

            fn face3_zero(reference_element: &ReferenceElement) -> Self::Line {
                use rulinalg::vector::Vector;
                $U { $($field: Vector::zeros(reference_element.face3.len()), )* }
            }
        }

        // Implement arithmetic traits
        impl Neg for $U {
            type Output = $U;
            fn neg(self: $U) -> $U {
                $U { $($field: $crate::blas::vector_scale_(self.$field, -1.), )* }
            }
        }

        impl<'a> Neg for &'a $U {
            type Output = $U;
            fn neg(self: &'a $U) -> $U {
                $U { $($field: $crate::blas::vector_scale(&self.$field, -1.), )* }
            }
        }

        impl Add for $U {
            type Output = $U;
            fn add(self, rhs: $U) -> $U {
                $U { $($field: $crate::blas::vector_add_(&self.$field, rhs.$field), )* }
            }
        }

        impl<'a> Add for &'a $U {
            type Output = $U;
            fn add(self, rhs: &$U) -> $U {
                $U { $($field: $crate::blas::vector_add(&self.$field, &rhs.$field), )* }
            }
        }

        impl Sub for $U {
            type Output = $U;
            fn sub(self, rhs: $U) -> $U {
                $U { $($field: $crate::blas::vector_sub_(self.$field, &rhs.$field), )* }
            }
        }

        impl<'a> Sub for &'a $U {
            type Output = $U;
            fn sub(self, rhs: &$U) -> $U {
                $U { $($field: $crate::blas::vector_sub(&self.$field, &rhs.$field), )* }
            }
        }

        impl Mul<f64> for $U {
            type Output = $U;
            fn mul(self, rhs: f64) -> Self {
                $U { $($field: $crate::blas::vector_scale_(self.$field, rhs), )* }
            }
        }

        impl<'a> Mul<f64> for &'a $U {
            type Output = $U;
            fn mul(self, rhs: f64) -> $U {
                $U { $($field: $crate::blas::vector_scale(&self.$field, rhs), )* }
            }
        }

        impl<'a> Mul<&'a Vector<f64>> for $U {
            type Output = $U;
            fn mul(self, rhs: &Vector<f64>) -> $U {
                $U { $($field: $crate::blas::elemul(&self.$field, rhs), )* }
            }
        }

        impl Div<f64> for $U {
            type Output = $U;
            fn div(self, rhs: f64) -> Self {
                $U { $($field: self.$field / rhs, )* }
            }
        }

        impl<'a> Div<f64> for &'a $U {
            type Output = $U;
            fn div(self, rhs: f64) -> $U {
                $U { $($field: &self.$field / rhs, )* }
            }
        }
    }
}
