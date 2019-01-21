#[macro_use]
extern crate galerkin;
extern crate ocl;
extern crate rulinalg;

use std::f64::consts;
use std::fmt::Debug;
use std::iter::repeat;

use ocl::{Buffer, Context, Program, ProQue};
use rulinalg::vector::Vector;

use galerkin::functions::range_kutta::*;
use galerkin::galerkin_1d::grid::ReferenceElement;
use galerkin::galerkin_1d::operators::{assemble_operators, Operators};
use galerkin::opencl::galerkin_1d::galerkin::communicate;
use galerkin::opencl::galerkin_1d::galerkin::GalerkinScheme;
use galerkin::opencl::galerkin_1d::galerkin::initialize_storage;
use galerkin::opencl::galerkin_1d::galerkin::prepare_communication_kernels;
use galerkin::opencl::galerkin_1d::grid::{Face, FaceType, generate_grid, Grid};
use galerkin::opencl::galerkin_1d::grid::BoundaryCondition;
use galerkin::opencl::galerkin_1d::grid::Element;
use galerkin::opencl::galerkin_1d::grid::ElementStorage;
use galerkin::opencl::galerkin_1d::grid::SpatialFlux;
use galerkin::opencl::galerkin_1d::operators::OperatorsStorage;
use galerkin::opencl::galerkin_1d::operators::store_operators;
use galerkin::opencl::galerkin_1d::unknowns::initialize_residuals;
use galerkin::opencl::galerkin_1d::unknowns::Unknown;

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

pub fn advec_1d<Fx, Fu>(
    reference_element: &ReferenceElement,
    operators: &Operators,
    grid: &Grid<Scheme>,
    u_0: Fx,
    _solution_callback: Fu,
) where
    Fx: Fn(&Vector<f64>) -> Vec<U>,
    Fu: FnMut(&Vec<U>),
{
    let final_time = 1.3;

    let cfl = 0.75;
    let x_scale = 0.01;
    let dt: f64 = 0.5 * cfl / (consts::PI * 2.) * x_scale;
    let n_t = (final_time / dt).ceil() as i32;
    let dt = final_time / n_t as f64;

    let t: f64 = 0.0;

    let a = 1.0;

    let _context = Context::builder()
        .build()
        .expect("could not create OpenCL context");
    let mut program_builder = Program::builder();
    program_builder
        .src(U::cl_struct_def())
        .src(APPLY_RESIDUAL_KERNEL)
        .src(AUGMENT_RESIDUAL_KERNEL)
        .src(FREEFLOW_BC_KERNEL)
        .src(SIN_BC_KERNEL)
        .src(ADVEC_FLUX_KERNEL)
        .src(ADVEC_RHS_KERNEL);
    let communication_kernels = prepare_communication_kernels(
        &U::cl_struct_type(),
        &vec![String::from("freeflow_bc"), String::from("sin_bc")],
    );
    program_builder.src(communication_kernels);
    let pro_que: ProQue = ProQue::builder()
        .prog_bldr(program_builder)
        .dims(reference_element.n_p)
        .build()
        .expect("could not build ProQue");

    let mut storage: Vec<ElementStorage<U>> =
        initialize_storage(u_0, reference_element.n_p, grid, operators, &pro_que);
    let operator_storage = store_operators(operators, &pro_que);
    let residuals = initialize_residuals(grid.elements.len(), reference_element.n_p, &pro_que);

    for epoch in 0..10 {
        println!("here! {}", epoch);
        for int_rk in 0..5 {
            let t = t + RKC[int_rk] * dt;

            for elt in &grid.elements {
                communicate(
                    t,
                    reference_element,
                    elt,
                    &grid.elements,
                    &storage[elt.index as usize],
                    &storage,
                    &pro_que,
                );
            }

            for elt in (*grid).elements.iter() {
                let mut storage = &mut storage[elt.index as usize];
                let residuals_u = &(residuals[elt.index as usize]);

                apply_rhs_to_residual(
                    int_rk,
                    elt,
                    storage,
                    residuals_u,
                    &operator_storage,
                    reference_element,
                    &pro_que,
                    dt,
                    a,
                );
            }
            &pro_que.finish().unwrap();
        }
        let mut u = vec![];
        for ref storage in storage.iter() {
            let mut slice = vec![U::default(); reference_element.n_p as usize];
            storage.u_k.read(&mut slice).enq().unwrap();
            u.extend_from_slice(slice.as_slice());
        }
        //        solution_callback(&u);
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

    debug_buffer(rhs);

    let augment_residual = pro_que
        .kernel_builder("augment_residual")
        .arg_named("residual", residual_u)
        .arg_named("rhs_u", rhs)
        .arg_named("rka", RKA[int_rk])
        .arg_named("dt", dt)
        .global_work_size(reference_element.n_p)
        .build()
        .unwrap();
    let apply_residual = pro_que
        .kernel_builder("apply_residual")
        .arg_named("residual", residual_u)
        .arg_named("u", &elt_storage.u_k)
        .arg_named("rkb", RKB[int_rk])
        .global_work_size(reference_element.n_p)
        .build()
        .unwrap();

    unsafe {
        augment_residual.enq().expect("kernel error");
        apply_residual.enq().expect("kernel error");
    }
}

fn rhs<'a>(
    reference_element: &ReferenceElement,
    elt: &Element<Scheme>,
    elt_storage: &'a ElementStorage<U>,
    operators: &OperatorsStorage,
    pro_que: &ProQue,
    a: f64,
) -> &'a Buffer<U> {
    let left_flux_kernel = pro_que
        .kernel_builder("advec_flux")
        .arg_named("u_minus", &elt_storage.u_left_minus)
        .arg_named("u_plus", &elt_storage.u_left_plus)
        .arg_named("f_minus", &elt.f_left_minus)
        .arg_named("f_plus", &elt.f_left_plus)
        .arg_named("outward_normal", elt.left_outward_normal as f32)
        .arg_named("output", &elt_storage.du_left)
        .global_work_size(1)
        .build()
        .unwrap();
    let right_flux_kernel = pro_que
        .kernel_builder("advec_flux")
        .arg_named("u_minus", &elt_storage.u_right_minus)
        .arg_named("u_plus", &elt_storage.u_right_plus)
        .arg_named("f_minus", &elt.f_right_minus)
        .arg_named("f_plus", &elt.f_right_plus)
        .arg_named("outward_normal", elt.right_outward_normal as f32)
        .arg_named("output", &elt_storage.du_right)
        .global_work_size(1)
        .build()
        .unwrap();
    unsafe {
        left_flux_kernel.enq().expect("kernel error");
        right_flux_kernel.enq().expect("kernel error");
    }

    let rhs_kernel = pro_que
        .kernel_builder("advec_rhs")
        .arg_named("n_p", reference_element.n_p + 1)
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
        .global_work_size(reference_element.n_p + 1)
        .build()
        .unwrap();
    unsafe {
        rhs_kernel.enq().expect("kernel error");
    }

    &elt_storage.u_k_rhs
}

pub const APPLY_RESIDUAL_KERNEL: &'static str = r#"
__kernel void apply_residual(
    __global float* residual,
    __global float* u,
    float rkb
) {
    int i = get_global_id(0);
    u[i] = u[i] + residual[i] * rkb;
}
"#;

pub const AUGMENT_RESIDUAL_KERNEL: &'static str = r#"
__kernel void augment_residual(
    __global cl_U* residual,
    __global cl_U* rhs_u,
    float rka,
    float dt
) {
    int i = get_global_id(0);
    residual[i] = (cl_U) {residual[i].u * rka + rhs_u[i].u * dt};
}
"#;

pub const ADVEC_FLUX_KERNEL: &'static str = r#"
__kernel void advec_flux(
    __global cl_U* u_minus,
    __global cl_U* u_plus,
    float f_minus,
    float f_plus,
    float outward_normal,
    __global cl_U* output
) {
    int i = get_global_id(0);

    float flux_minus = u_minus[i].u * f_minus;
    float flux_plus = u_plus[i].u * f_plus;
    float avg = flux_minus + flux_plus;
    float jump = flux_minus * outward_normal - flux_plus * outward_normal;

    float flux_numerical = avg + jump / 2.0;

    output[i] = (cl_U) {(flux_minus - flux_numerical) * outward_normal};
}
"#;

pub const ADVEC_RHS_KERNEL: &'static str = r#"
__kernel void advec_rhs(
    // the total number of nodes per element
    int n_p,
    // the number of nodes on a face
    int n_f,
    // the n_p dimensioned array of unknowns on this element
    __global float* u,
    __global cl_U* du_left,
    __global cl_U* du_right,
    // the n_p * n_p d_r matrix, in a row-major format
    __global float* d_r,
    __global float* r_x,
    float r_x_left,
    float r_x_right,
    __global float* lift,
    float a,

    __global float* output
) {
    int i = get_global_id(0);

    float d_r_u = 0.0;
    for (int j = 0; j < n_p; j++) {
        float d_r_ij = d_r[j + i * n_p];
        d_r_u += d_r_ij * u[j];
    }

    float a_rx = -r_x[i] * a;
    float rhs_u = a_rx * d_r_u;

    float scaled_du_left = r_x_left * du_left[0].u;
    float scaled_du_right = r_x_right * du_right[0].u;

    float lifted_flux = 0.0;
    lifted_flux += lift[2 * i] * du_left[0].u;
    lifted_flux += lift[2 * i + 1] * du_right[0].u;

    output[i] = rhs_u + lifted_flux;
}
"#;

pub const FREEFLOW_BC_KERNEL: &'static str = r#"
static cl_U freeflow_bc(float t, cl_U u) {
    return u;
}
"#;

pub const SIN_BC_KERNEL: &'static str = r#"
static cl_U sin_bc(float t, cl_U u) {
    return (cl_U) { -sin(1.0 * t) };
}
"#;

pub fn main() {
    let reference_element = ReferenceElement::legendre(5);
    let operators = assemble_operators(&reference_element);
    let left_boundary = Face {
        face_type: FaceType::Boundary(
            BoundaryCondition {
                function_name: "sin_bc".to_string(),
            },
            1.,
        ),
    };
    let right_boundary = Face {
        face_type: FaceType::Boundary(
            BoundaryCondition {
                function_name: "freeflow_bc".to_string(),
            },
            1.,
        ),
    };
    let grid = generate_grid(
        -1.0,
        1.0,
        8,
        &reference_element,
        &operators,
        left_boundary,
        right_boundary,
        move |_| F(1.),
    );

    advec_1d(&reference_element, &operators, &grid, &u_0, |u| {
        println!("{:?}", u)
    });
}

fn u_0(xs: &Vector<f64>) -> Vec<U> {
    xs.iter().map(|x: &f64| U { u: x.sin() as f32 }).collect()
}

fn debug_buffer<T: OclPrm>(buf: &Buffer<T>) {
    let mut u: Vec<T> = repeat(T::default()).take(buf.len()).collect();
    buf.read(&mut u).enq().unwrap();
    println!("{}: {:?}", buf.len(), u);
}
