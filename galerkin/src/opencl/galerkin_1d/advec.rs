extern crate ocl;
extern crate rulinalg;

use rulinalg::vector::Vector;
use std::f64::consts;
use opencl::galerkin_1d::galerkin::GalerkinScheme;
use opencl::galerkin_1d::grid::Grid;
use opencl::galerkin_1d::grid::ElementStorage;
use opencl::galerkin_1d::galerkin::initialize_storage;
use galerkin_1d::grid::ReferenceElement;
use galerkin_1d::operators::Operators;
use ocl::{ProQue, Buffer};
use ocl::Program;
use opencl::galerkin_1d::unknowns::{Unknown};
use opencl::galerkin_1d::galerkin::prepare_communication_kernels;
use opencl::galerkin_1d::unknowns::initialize_residuals;
use opencl::galerkin_1d::galerkin::communicate;
use functions::range_kutta::*;
use opencl::galerkin_1d::grid::Element;
use opencl::galerkin_1d::operators::OperatorsStorage;
use opencl::galerkin_1d::operators::store_operators;
use opencl::galerkin_1d::grid::SpatialFlux;

pub struct F(f32);

gen_unknown!(U, f32, u);

impl SpatialFlux for F {
    type Unit = f32;

    fn first(&self) -> <Self as SpatialFlux>::Unit {
        self.0
    }

    fn last(&self) -> <Self as SpatialFlux>::Unit {
        self.0
    }

    fn zero() -> <Self as SpatialFlux>::Unit {
        0.
    }
}

pub struct Scheme {}

impl GalerkinScheme for Scheme {
    type U = U;
    type F = F;
}

pub fn advec_1d<Fx>(
    reference_element: &ReferenceElement,
    operators: &Operators,
    grid: &Grid<Scheme>,
    u_0: Fx
)
where
Fx: Fn(&Vector<f64>) -> Vec<U>
{
    let final_time = 1.3;

    let cfl = 0.75;
    let x_scale = 0.01;
    let dt: f64 = 0.5 * cfl / (consts::PI * 2.) * x_scale;
    let n_t = (final_time / dt).ceil() as i32;
    let dt = final_time / n_t as f64;

    let mut t: f64 = 0.0;

    // TODO use spatial flux instead
    let a = 1.0;

    let mut program_builder = ProQue::builder();
    program_builder.src(APPLY_RESIDUAL_KERNEL)
        .src(AUGMENT_RESIDUAL_KERNEL)
        .src(FREEFLOW_BC_KERNEL)
        .src(SIN_BC_KERNEL);
    let communication_kernels = prepare_communication_kernels(
        &U::cl_struct_type(), &vec![
            String::from("freeflow_bc"),
            String::from("sin_bc"),
        ]
    );
    program_builder.src(communication_kernels);
    let pro_que: ProQue = program_builder.build().unwrap();

    let mut storage: Vec<ElementStorage<U>> =
        initialize_storage(u_0, reference_element.n_p, grid, operators, &pro_que);
    let operator_storage = store_operators(operators, &pro_que);
    let mut residuals = initialize_residuals(grid.elements.len(), reference_element.n_p, &pro_que);

    for epoch in 0..3 {
        for int_rk in 0..5 {
            let t = t + RKC[int_rk] * dt;

            for elt in &grid.elements {
                communicate(t, reference_element, elt, &grid.elements,
                    &storage[elt.index as usize], &storage, &pro_que);
            }

            for elt in (*grid).elements.iter() {
                let mut storage = &mut storage[elt.index as usize];
                let residuals_u = &(residuals[elt.index as usize]);

                apply_rhs_to_residual(int_rk, elt, storage, residuals_u, &operator_storage,
                    reference_element, &pro_que, dt, a);
            }
        }
    }
}

fn apply_rhs_to_residual(
    int_rk: usize,
    elt: &Element<Scheme>,
    elt_storage: &ElementStorage<U>,
    residual_u: &Buffer<U>,
    operators: &OperatorsStorage,
    reference_element: &ReferenceElement,
    pro_que: &ProQue,
    dt: f64,
    a: f64,
) {
    let rhs = rhs(reference_element, elt, elt_storage, operators, pro_que, a);

    let augment_residual = pro_que.kernel_builder("augment_residual")
        .arg_named("residual", residual_u)
        .arg_named("rhs_u", rhs)
        .arg_named("rka", RKA[int_rk])
        .arg_named("dt", dt)
        .global_work_size(reference_element.n_p)
        .build().unwrap();
    let apply_residual = pro_que.kernel_builder("apply_residual")
        .arg_named("residual", residual_u)
        .arg_named("u", &elt_storage.u_k)
        .arg_named("rkb", RKB[int_rk])
        .global_work_size(reference_element.n_p)
        .build().unwrap();

    unsafe {
        augment_residual.enq();
        apply_residual.enq();
    }
}

fn rhs<'a>(reference_element: &ReferenceElement,
       elt: &Element<Scheme>,
       elt_storage: &'a ElementStorage<U>,
       operators: &OperatorsStorage,
       pro_que: &ProQue,
       a: f64,
) -> &'a Buffer<U> {
    let left_flux_kernel = pro_que.kernel_builder("advec_flux")
        .arg_named("u_minus", &elt_storage.u_left_minus)
        .arg_named("u_plus", &elt_storage.u_left_plus)
        .arg_named("f_minus", &elt.f_left_minus)
        .arg_named("f_plus", &elt.f_left_plus)
        .arg_named("outward_normal", &elt.left_outward_normal)
        .arg_named("output", &elt_storage.du_left)
        .global_work_size(1)
        .build().unwrap();
    let right_flux_kernel = pro_que.kernel_builder("advec_flux")
        .arg_named("u_minus", &elt_storage.u_right_minus)
        .arg_named("u_plus", &elt_storage.u_right_plus)
        .arg_named("f_minus", &elt.f_right_minus)
        .arg_named("f_plus", &elt.f_right_plus)
        .arg_named("outward_normal", &elt.right_outward_normal)
        .arg_named("output", &elt_storage.du_right)
        .global_work_size(1)
        .build().unwrap();
    unsafe {
        left_flux_kernel.enq();
        right_flux_kernel.enq();
    }

    let rhs_kernel = pro_que.kernel_builder("advec_rhs")
        .arg_named("n_p", reference_element.n_p)
        .arg_named("n_f", 1)
        .arg_named("u", &elt_storage.u_k)
        .arg_named("du_left", &elt_storage.du_left)
        .arg_named("du_right", &elt_storage.du_right)
        .arg_named("d_r", &operators.d_r)
        .arg_named("r_x", &elt_storage.r_x)
        .arg_named("r_x_left", &elt.r_x_left)
        .arg_named("r_x_right", &elt.r_x_right)
        .arg_named("lift", &operators.lift)
        .arg_named("a", a as f32)
        .arg_named("output", &elt_storage.u_k_rhs)
        .global_work_size(reference_element.n_p)
        .build().unwrap();

    &elt_storage.u_k_rhs
}

pub const APPLY_RESIDUAL_KERNEL: &'static str = r#"
__kernel void apply_residual(
    __global float* residual,
    __global float* u,
    __global float rkb
) {
    int i = get_global_id();
    u[i] = u[i] + residual[i] * rkb;
}
"#;

pub const AUGMENT_RESIDUAL_KERNEL: &'static str = r#"
__kernel void augment_residual(
    __global float* residual,
    __global float* rhs_u,
    __global float rka,
    __global float dt
) {
    int i = get_global_id();
    residual[i] = residual[i] * rka + rhs_u[i] * dt;
}
"#;

pub const ADVEC_FLUX_KERNEL: &'static str = r#"
__kernel void advec_flux(
    __global float* u_minus,
    __global float* u_plus,
    __global float* f_minus,
    __global float* f_plus,
    __global float outward_normal,
    __global float* output
) {
    int i = get_global_id();

    float flux_minus = u_minus[i] * f_minus[i];
    float flux_plus = u_plus[i] * f_plus[i];
    float avg = flux_minus + flux_plus;
    float jump = flux_minus * outward_normal - flux_plus * outward_normal;

    float flux_numerical = avg + jump / 2.0;

    output[i] = (flux_minus - flux_numerical) * outward_normal
}
"#;

pub const ADVEC_RHS_KERNEL: &'static str = r#"
__kernel void advec_rhs(
    // the total number of nodes per element
    __global int n_p,
    // the number of nodes on a face
    __global int n_f,
    // the n_p dimensioned array of unknowns on this element
    __global float* u,
    __global float* du_left,
    __global float* du_right,
    // the n_p * n_p d_r matrix, in a row-major format
    __global float* d_r,
    __global float* r_x,
    __global float r_x_left,
    __global float r_x_right,
    __global float* lift,
    __global float a,

    __global float* output
) {
    int i = get_global_id();

    float d_r_u = 0.0;
    for (int j = 0; j < n_p, j++) {
        float d_r_ij = d_r[j + i * n_p];
        d_r_u += d_r_ij * u[j];
    }

    float a_rx = -r_x[i] * a;
    float rhs_u = a_rx * d_r_u;

    float scaled_du_left = r_x_left * du_left[0];
    float scaled_du_right = r_x_right * du_right[0];

    float lifted_flux = 0.0;
    lifted_flux += lift[2 * i] * du_left[0];
    lifted_flux += lift[2 * i] * du_right[0];

    output[i] = rhs_u + lifted_flux;
}
"#;

pub const FREEFLOW_BC_KERNEL: &'static str = r#"
__kernel float freeflow_bc(float t, float u) {
    return u;
}
"#;

pub const SIN_BC_KERNEL: &'static str = r#"
__kernel float sin_bc(float t, float u) {
    return -sin(a * t);
}
"#;

#[cfg(test)]
mod tests {
    use super::{advec_1d, U};
    use galerkin_1d::grid::ReferenceElement;
    use galerkin_1d::operators::{Operators, assemble_operators};
    use opencl::galerkin_1d::grid::{Grid, generate_grid, Face, FaceType};
    use opencl::galerkin_1d::galerkin::initialize_storage;
    use opencl::galerkin_1d::grid::BoundaryCondition;
    use opencl::galerkin_1d::advec::F;
    use rulinalg::vector::Vector;

    #[test]
    fn test() {
        let reference_element = ReferenceElement::legendre(5);
        let operators = assemble_operators(&reference_element);
        let left_boundary = Face {
            face_type: FaceType::Boundary(
                BoundaryCondition { function_name: "sin_bc".to_string()},
                1.,
            ),
        };
        let right_boundary = Face {
            face_type: FaceType::Boundary(
                BoundaryCondition { function_name: "freeflow_bc".to_string()},
                1.,
            ),
        };
        let grid = generate_grid(
            -1.0, 1.0, 8, &reference_element, &operators, left_boundary, right_boundary,
            move |_| F(1.),
        );

        advec_1d(
            &reference_element,
            &operators,
            &grid,
            &u_0,
        );
    }

    fn u_0(xs: &Vector<f64>) -> Vec<U> {
        xs.iter().map(|x: &f64| U { u: x.sin() as f32 }).collect()
    }
}
