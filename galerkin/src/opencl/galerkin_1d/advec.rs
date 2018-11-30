extern crate ocl;

use std::f64::consts;
use opencl::galerkin_1d::galerkin::GalerkinScheme;
use opencl::galerkin_1d::grid::Grid;
use opencl::galerkin_1d::grid::ElementStorage;
use opencl::galerkin_1d::galerkin::initialize_storage;
use galerkin_1d::grid::ReferenceElement;
use galerkin_1d::operators::Operators;
use ocl::ProQue;
use opencl::galerkin_1d::unknowns::{U, unknown};
use opencl::galerkin_1d::galerkin::prepare_communication_kernels;
use opencl::galerkin_1d::unknowns::initialize_residuals;
use opencl::galerkin_1d::galerkin::communicate;
use functions::range_kutta::RKC;
use opencl::galerkin_1d::grid::Element;
use opencl::galerkin_1d::operators::OperatorsStorage;

struct U {
    u: f32,
}

struct F {
    f: f32,
}

unknown!(U, f32, u);

struct Scheme {}

impl GalerkinScheme for Scheme {
    type U = U;
    type F = F;
}

pub fn advec_1d(
    reference_element: &ReferenceElement,
    operators: &Operators,
    grid: &Grid<Scheme>,
) {
    let final_time = 1.3;

    let cfl = 0.75;
    let x_scale = 0.01;
    let dt: f64 = 0.5 * cfl / (consts::PI * 2.) * x_scale;
    let n_t = (final_time / dt).ceil() as i32;
    let dt = final_time / n_t as f64;

    let mut t: f64 = 0.0;

    let boundary_condition_kernels =
    let communication_kernels = prepare_communication_kernels(
        U::cl_struct_type(), &vec![
            String::from("freeflow_bc"),
            String::from("sin_bc")
        ]
    );
    let all_src = format!("{}\n{}\n{}\n{}",
    ADVEC_FLUX_KERNEL, ADVEC_RHS_KERNEL, FREEFLOW_BC_KERNEL, SIN_BC_KERNEL);
    let pro_que: ProQue = prepare_program_queue();

    let mut storage: Vec<ElementStorage<GS>> =
        initialize_storage(grid, reference_element.n_p, grid, operators, &pro_que);
    let mut residuals = initialize_residuals(grid.elements.len(), reference_element.n_p, &pro_que);

    for epoch in 0..3 {
        for int_rk in 0..5 {
            let t = t + RKC[int_rk] * dt;

            for elt in &grid.elements {
                communicate(t, reference_element, elt, &grid.elements,
                    storage[elt.index as usize], storage, &pro_que);
            }

            for elt in (*grid).elements.iter() {
                let mut storage = &mut storage[elt.index as usize];

                let residuals_u = {
                    let residuals_u = &(residuals[elt.index as usize]);

                    let rhs_u = advec_rhs_1d(&elt, &storage, &operators, a);
                    residuals_u * RKA[int_rk] + rhs_u * dt
                };
            }
        }
    }
}

fn rhs(reference_element: &ReferenceElement,
       elt: &Element<GS>,
       elt_storage: &ElementStorage<U>,
       operators: &OperatorsStorage,
       pro_que: &ProQue) {
    pro_que.kernel_builder("advec_rhs")
        .arg_named("n_p", reference_element.n_p)
        .arg_named("n_f", 1)
        .arg_named("u", elt_storage.u_k)
        .arg_named("du_left", elt_storage)
}

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

    float scaled_du_left = r_x_left * du_left;
    float scaled_du_right = r_x_right * du_right;

    float lifted_flux = 0.0;
    lifted_flux += lift[2 * i] * du_left;
    lifted_flux += lift[2 * i] * du_right;

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

