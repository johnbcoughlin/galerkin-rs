extern crate rulinalg;

use distmesh::mesh::{Mesh, Triangle};
use galerkin_2d::flux::FluxScheme;
use galerkin_2d::galerkin::GalerkinScheme;
use galerkin_2d::operators::Operators;
use galerkin_2d::reference_element::ReferenceElement;
use galerkin_2d::unknowns::Unknown;
use rulinalg::vector::Vector;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceNumber {
    One,
    Two,
    Three,
}

pub enum FaceType<'grid, GS: GalerkinScheme>
    where
        <GS::U as Unknown>::Line: 'grid,
        <GS::FS as FluxScheme<GS::U>>::F: 'grid,
{
    // An interior face with the index of the element on the other side.
    Interior(i32, FaceNumber),

    // A complex boundary condition which may depend on the other side of the boundary and on
    // the time parameter.
    Boundary(
        // the exterior value of the unknown, as a function of time
        &'grid Fn(f64, &Vector<f64>, &Vector<f64>) -> <GS::U as Unknown>::Line,
        // the exterior value of the spatial parameter
        &'grid Fn() -> <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
    ),
}

impl<'grid, GS: GalerkinScheme> Debug for FaceType<'grid, GS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            FaceType::Interior(i, face) => write!(f, "Interior({}, {:?})", i, face),
            FaceType::Boundary(_, _) => write!(f, "Boundary()"),
        }
    }
}

#[derive(Debug)]
pub struct Face<'grid, GS: GalerkinScheme>
    where
        <GS::U as Unknown>::Line: 'grid,
        <GS::FS as FluxScheme<GS::U>>::F: 'grid,
{
    pub face_type: FaceType<'grid, GS>,
    pub flux_key: <GS::FS as FluxScheme<GS::U>>::K,
    pub surface_jacobian: Vector<f64>,
    pub f_scale: Vector<f64>,
    pub outward_normal_x: Vector<f64>,
    pub outward_normal_y: Vector<f64>,
}

#[derive(Debug)]
pub struct LocalMetric {
    // Derivatives of the metric mapping at each point
    // dx/dr
    pub x_r: Vector<f64>,
    // dy/dr
    pub y_r: Vector<f64>,
    // dx/ds
    pub x_s: Vector<f64>,
    // dy/ds
    pub y_s: Vector<f64>,
    // The Jacobian, x_r * y_s - x_s * y_r
    pub jacobian: Vector<f64>,

    // derivatives in the other direction
    pub r_x: Vector<f64>,
    pub s_x: Vector<f64>,
    pub r_y: Vector<f64>,
    pub s_y: Vector<f64>,
}

#[derive(Debug)]
pub struct Element<'grid, GS: GalerkinScheme>
    where
        <GS::U as Unknown>::Line: 'grid,
        <GS::FS as FluxScheme<GS::U>>::F: 'grid,
{
    pub index: i32,
    pub x_k: Vector<f64>,
    pub y_k: Vector<f64>,

    pub local_metric: LocalMetric,

    pub spatial_parameters: <GS::FS as FluxScheme<GS::U>>::F,

    pub face1: Face<'grid, GS>,
    pub face2: Face<'grid, GS>,
    pub face3: Face<'grid, GS>,
}

impl<'grid, GS: GalerkinScheme> Element<'grid, GS> {
    pub fn face<'a>(&'a self, number: FaceNumber) -> &'a Face<'grid, GS> {
        match number {
            FaceNumber::One => &self.face1,
            FaceNumber::Two => &self.face2,
            FaceNumber::Three => &self.face3,
        }
    }

    pub fn face_x(&self, number: FaceNumber, reference_element: &ReferenceElement) -> Vector<f64> {
        match number {
            FaceNumber::One => self.x_k.select(reference_element.face1.as_slice()),
            FaceNumber::Two => self.x_k.select(reference_element.face2.as_slice()),
            FaceNumber::Three => self.x_k.select(reference_element.face3.as_slice()),
        }
    }

    pub fn face_y(&self, number: FaceNumber, reference_element: &ReferenceElement) -> Vector<f64> {
        match number {
            FaceNumber::One => self.y_k.select(reference_element.face1.as_slice()),
            FaceNumber::Two => self.y_k.select(reference_element.face2.as_slice()),
            FaceNumber::Three => self.y_k.select(reference_element.face3.as_slice()),
        }
    }
}

pub struct ElementStorage<GS>
    where
        GS: GalerkinScheme,
{
    pub u_k: GS::U,

    // minus is interior, plus is exterior
    pub u_face1_minus: RefCell<<GS::U as Unknown>::Line>,
    pub u_face1_plus: RefCell<<GS::U as Unknown>::Line>,
    pub u_face2_minus: RefCell<<GS::U as Unknown>::Line>,
    pub u_face2_plus: RefCell<<GS::U as Unknown>::Line>,
    pub u_face3_minus: RefCell<<GS::U as Unknown>::Line>,
    pub u_face3_plus: RefCell<<GS::U as Unknown>::Line>,

    pub f_face1_minus: <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
    pub f_face1_plus: <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
    pub f_face2_minus: <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
    pub f_face2_plus: <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
    pub f_face3_minus: <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
    pub f_face3_plus: <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
}

#[derive(Debug)]
pub struct Grid<'grid, GS: GalerkinScheme>
    where
        <GS::U as Unknown>::Line: 'grid,
        <GS::FS as FluxScheme<GS::U>>::F: 'grid,
{
    pub elements: Vec<Element<'grid, GS>>,
}

fn assemble_edges_to_triangle(triangles: &Vec<Triangle>) -> HashMap<Edge, EdgeType> {
    let mut edges_to_triangle: HashMap<Edge, EdgeType> = HashMap::new();
    for (i, ref triangle) in triangles.iter().enumerate() {
        let (e1, e2, e3) = triangle.edges();
        let modifier = |e: Edge, face_number: FaceNumber, map: &mut HashMap<Edge, EdgeType>| {
            let new_value = if map.contains_key(&e) {
                let existing = map.get(&e).expect("we just checked");
                existing.with_other_triangle(i as i32, face_number)
            } else {
                EdgeType::Exterior(i as i32, face_number)
            };
            map.insert(e, new_value);
        };
        modifier(e1, FaceNumber::One, &mut edges_to_triangle);
        modifier(e2, FaceNumber::Two, &mut edges_to_triangle);
        modifier(e3, FaceNumber::Three, &mut edges_to_triangle);
    }
    edges_to_triangle
}

#[allow(too_many_arguments, many_single_char_names)]
pub fn assemble_grid<'grid, GS, F, FExterior, FSP>(
    reference_element: &ReferenceElement,
    operators: &Operators,
    mesh: &Mesh,
    boundary_condition: &'grid F,
    exterior_boundary_spatial_parameter: &'grid FExterior,
    initial_spatial_parameter: FSP,
    interior_flux_key: <GS::FS as FluxScheme<GS::U>>::K,
    exterior_flux_key: <GS::FS as FluxScheme<GS::U>>::K,
) -> Grid<'grid, GS>
    where
        GS: GalerkinScheme,
        for<'r, 's> F: Fn(f64, &'r Vector<f64>, &'s Vector<f64>) -> <GS::U as Unknown>::Line + 'grid,
        FExterior: Fn() -> <<GS::FS as FluxScheme<GS::U>>::F as SpatialVariable>::Line,
        FSP: Fn(&Vector<f64>, &Vector<f64>) -> <GS::FS as FluxScheme<GS::U>>::F,
{
    let points = &mesh.points;
    let rs = &reference_element.rs;
    let ss = &reference_element.ss;

    let edges_to_triangle = assemble_edges_to_triangle(&mesh.triangles);

    let mut elements = Vec::new();

    for (i, ref triangle) in mesh.triangles.iter().enumerate() {
        let (ref a, ref b, ref c) = (
            &points[triangle.a as usize],
            &points[triangle.b as usize],
            &points[triangle.c as usize],
        );
        let x: Vector<f64> = (&(-rs - ss) * a.x + (rs + 1.) * b.x + (ss + 1.) * c.x) * 0.5;
        let y: Vector<f64> = (&(-rs - ss) * a.y + (rs + 1.) * b.y + (ss + 1.) * c.y) * 0.5;

        let x_r = &operators.d_r * &x;
        let x_s = &operators.d_s * &x;
        let y_r = &operators.d_r * &y;
        let y_s = &operators.d_s * &y;
        let jacobian = x_r.elemul(&y_s) - &(x_s.elemul(&y_r));

        let r_x = y_s.elediv(&jacobian);
        let s_x = -y_r.elediv(&jacobian);
        let r_y = -x_s.elediv(&jacobian);
        let s_y = x_r.elediv(&jacobian);

        let (e1, e2, e3) = triangle.edges();
        let edge_to_face_type = |e: &Edge| match edges_to_triangle.get(e) {
            Some(EdgeType::Interior(a, a_number, b, b_number)) => if *a == i as i32 {
                (FaceType::Interior(*b, *b_number), interior_flux_key)
            } else {
                (FaceType::Interior(*a, *a_number), interior_flux_key)
            },
            Some(EdgeType::Exterior(_, _)) => (
                FaceType::Boundary(boundary_condition, exterior_boundary_spatial_parameter),
                exterior_flux_key,
            ),
            None => panic!("edge_to_triangle did not contain {:?}", e),
        };
        let local_metric = LocalMetric {
            x_r,
            y_r,
            x_s,
            y_s,
            jacobian,
            r_x,
            s_x,
            r_y,
            s_y,
        };

        let spatial_parameters = initial_spatial_parameter(&x, &y);
        let ef1 = edge_to_face_type(&e1);
        let face1: Face<'grid, GS> = build_face(
            FaceNumber::One,
            ef1.0,
            ef1.1,
            reference_element,
            &local_metric,
        );
        let ef2 = edge_to_face_type(&e2);
        let face2: Face<'grid, GS> = build_face(
            FaceNumber::Two,
            ef2.0,
            ef2.1,
            reference_element,
            &local_metric,
        );
        let ef3 = edge_to_face_type(&e3);
        let face3: Face<'grid, GS> = build_face(
            FaceNumber::Three,
            ef3.0,
            ef3.1,
            reference_element,
            &local_metric,
        );

        elements.push(Element {
            index: i as i32,
            x_k: x,
            y_k: y,
            local_metric,
            spatial_parameters,
            face1,
            face2,
            face3,
        });
    }
    Grid { elements }
}

fn build_face<'grid, GS>(
    face_number: FaceNumber,
    face_type: FaceType<'grid, GS>,
    flux_key: <GS::FS as FluxScheme<GS::U>>::K,
    reference_element: &ReferenceElement,
    local_metric: &LocalMetric,
) -> Face<'grid, GS>
    where
        GS: GalerkinScheme,
{
    let slice = reference_element.face(face_number).as_slice();
    let x_r_face = local_metric.x_r.select(slice);
    let x_s_face = local_metric.x_s.select(slice);
    let y_r_face = local_metric.y_r.select(slice);
    let y_s_face = local_metric.y_s.select(slice);

    let nx = match face_number {
        FaceNumber::One => y_r_face.clone(),
        FaceNumber::Two => y_s_face.clone() - y_r_face.clone(),
        FaceNumber::Three => -y_s_face.clone(),
    };
    let ny = match face_number {
        FaceNumber::One => -x_r_face.clone(),
        FaceNumber::Two => -x_s_face.clone() + x_r_face.clone(),
        FaceNumber::Three => x_s_face.clone(),
    };
    let surface_jacobian: Vector<f64> = (&(nx.elemul(&nx)) + &(ny.elemul(&ny)))
        .iter()
        .map(|&f| f.sqrt())
        .collect();
    let f_scale: Vector<f64> = surface_jacobian.elediv(
        &local_metric
            .jacobian
            .select(reference_element.face(face_number).as_slice()),
    );
    let nx = nx.elediv(&surface_jacobian);
    let ny = ny.elediv(&surface_jacobian);
    Face {
        face_type,
        flux_key,
        surface_jacobian,
        f_scale,
        outward_normal_x: nx,
        outward_normal_y: ny,
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct Edge {
    n1: i32,
    n2: i32,
}

impl Edge {
    fn from(a: i32, b: i32) -> Edge {
        if a > b {
            Edge { n1: b, n2: a }
        } else {
            Edge { n1: a, n2: b }
        }
    }
}

impl Triangle {
    fn edges(&self) -> (Edge, Edge, Edge) {
        (
            Edge::from(self.a, self.b),
            Edge::from(self.b, self.c),
            Edge::from(self.c, self.a),
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeType {
    Exterior(i32, FaceNumber),
    Interior(i32, FaceNumber, i32, FaceNumber),
}

impl EdgeType {
    fn with_other_triangle(&self, triangle: i32, neighbors_face_number: FaceNumber) -> EdgeType {
        match self {
            EdgeType::Exterior(t1, t1_number) => {
                EdgeType::Interior(*t1, *t1_number, triangle, neighbors_face_number)
            }
            EdgeType::Interior(_, _, _, _) => panic!("found an edge with more than two faces"),
        }
    }
}

#[derive(Debug)]
pub struct XYTuple<T> {
    pub x: T,
    pub y: T,
}

pub type Vec2 = XYTuple<f64>;

pub trait SpatialVariable: Debug {
    type Line;

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

    fn face1_zero(reference_element: &ReferenceElement) -> Self::Line;

    fn face2_zero(reference_element: &ReferenceElement) -> Self::Line;

    fn face3_zero(reference_element: &ReferenceElement) -> Self::Line;
}

impl SpatialVariable for () {
    type Line = ();

    fn edge_1(&self, _reference_element: &ReferenceElement) -> () {
        ()
    }

    fn edge_2(&self, _reference_element: &ReferenceElement) -> () {
        ()
    }

    fn edge_3(&self, _reference_element: &ReferenceElement) -> () {
        ()
    }

    fn face1_zero(_reference_element: &ReferenceElement) -> () {
        ()
    }

    fn face2_zero(_reference_element: &ReferenceElement) -> () {
        ()
    }

    fn face3_zero(_reference_element: &ReferenceElement) -> () {
        ()
    }
}

#[cfg(test)]
mod assemble_edges_to_triangle_test {
    use super::{assemble_edges_to_triangle, Edge, EdgeType, FaceNumber::*};
    use distmesh::mesh::Triangle;
    #[macro_use]
    use testing::assertions::{self, Assertion};
    use std::collections::HashMap;

    #[test]
    fn test_no_triangles() {
        let result = assemble_edges_to_triangle(&vec![]);
        assert_that!(result).is_empty();
    }

    #[test]
    fn test_one_triangle() {
        let triangles = vec![Triangle { a: 0, b: 1, c: 2 }];
        let mut result = assemble_edges_to_triangle(&triangles);
        assert_that!(result).has_size(3);
        assert_that!(result.remove(&Edge { n1: 0, n2: 1 })).contains(EdgeType::Exterior(0, One));
        assert_that!(result.remove(&Edge { n1: 1, n2: 2 })).contains(EdgeType::Exterior(0, Two));
        assert_that!(result.remove(&Edge { n1: 0, n2: 2 })).contains(EdgeType::Exterior(0, Three));
    }

    #[test]
    fn test_two_triangles() {
        let triangles = vec![Triangle { a: 0, b: 1, c: 2 }, Triangle { a: 1, b: 2, c: 3 }];
        let mut result = assemble_edges_to_triangle(&triangles);
        assert_that!(result).has_size(5);
        assert_that!(remove_edge(&mut result, 0, 1)).contains(EdgeType::Exterior(0, One));
        assert_that!(remove_edge(&mut result, 1, 2)).contains(EdgeType::Interior(0, Two, 1, One));
        assert_that!(remove_edge(&mut result, 0, 2)).contains(EdgeType::Exterior(0, Three));
    }

    fn remove_edge(result: &mut HashMap<Edge, EdgeType>, n1: i32, n2: i32) -> Option<EdgeType> {
        result.remove(&Edge { n1, n2 })
    }
}
