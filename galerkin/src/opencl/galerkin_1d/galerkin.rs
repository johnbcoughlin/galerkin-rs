extern crate ocl;
extern crate string_builder;

use opencl::galerkin_1d::grid::SpatialFlux;
use opencl::galerkin_1d::grid::Grid;

use ocl::{ProQue, OclPrm};
use opencl::galerkin_1d::grid::generate_grid;
use opencl::galerkin_1d::grid::{ElementStorage, FaceType};
use galerkin_1d::operators::Operators;
use opencl::galerkin_1d::unknowns::Unknown;
use rulinalg::vector::Vector;
use opencl::galerkin_1d::grid::Element;
use ocl::{Kernel, Buffer};
use ocl::builders::KernelBuilder;
use galerkin_1d::grid::ReferenceElement;
use opencl::galerkin_1d::grid::BoundaryCondition;

pub trait GalerkinScheme {
    type U: Unknown;
    type F: SpatialFlux;
}

pub fn entry_point<GS>(
    grid: &Grid<GS>,
    src: &str,
)
    where
        GS: GalerkinScheme
{
    let pro_que = ProQue::builder()
        .src(src)
        .build();

//    let grid = generate_grid()
}

pub fn initialize_storage<GS, Fx>(
    u_0: Fx,
    n_p: i32,
    grid: &Grid<GS>,
    operators: &Operators,
    pro_que: &ProQue,
) -> Vec<ElementStorage<GS::U>>
    where
        GS: GalerkinScheme,
        Fx: Fn(&Vector<f64>) -> Vec<GS::U>,
{
    let mut result: Vec<ElementStorage<GS::U>> = vec![];
    for elt in &grid.elements {
        let u_k_host: Vec<GS::U> = u_0(&elt.x_k);
        let u_k = pro_que.buffer_builder()
            .copy_host_slice(&u_k_host)
            .build().unwrap();
        let endpoint_supplier = || pro_que.buffer_builder()
            .len(1)
            .build().unwrap();
        let (u_left_minus, u_right_minus, u_left_plus, u_right_plus,
            du_left, du_right) = (
            endpoint_supplier(),
            endpoint_supplier(),
            endpoint_supplier(),
            endpoint_supplier(),
            endpoint_supplier(),
            endpoint_supplier(),
        );
        let u_k_rhs = pro_que.buffer_builder()
            .len(u_k_host.len())
            .build().unwrap();
        result.push(ElementStorage {
            u_k,
            u_left_minus,
            u_right_minus,
            u_left_plus,
            u_right_plus,
            du_left,
            du_right,
            u_k_rhs,
        });
    }
    result
}

pub fn communicate<GS>(
    t: f64,
    reference_element: &ReferenceElement,
    elt: &Element<GS>,
    elements: &Vec<Element<GS>>,
    elt_storage: &mut ElementStorage<GS::U>,
    storages: &Vec<ElementStorage<GS::U>>,
    pro_que: &ProQue,
)
    where
        GS: GalerkinScheme
{
    let build_kernel = |face_type: &FaceType<GS>, destination| {
        match face_type {
            FaceType::Interior(i) => pro_que.kernel_builder("communicate_internal")
                .arg_named("destination", destination)
                .arg_named("index_in_destination", 0)
                .arg_named("neighbor", &storages[*i as usize].u_k)
                .arg_named("index_in_neighbor", reference_element.n_p)
                .build().unwrap(),
            FaceType::Boundary(BoundaryCondition { function_name }) =>
                pro_que.kernel_builder(format!("communicate_external__{}", function_name))
                    .arg_named("destination", destination)
                    .arg_named("index_in_destination", 0)
                    .arg_named("t", t as f32)
                    .arg_named("x", elt.x_k[0])
                    .build().unwrap(),
        }
    };
    let left_kernel: Kernel = build_kernel(&elt.left_face.face_type, &elt_storage.u_left_plus);
    let right_kernel: Kernel = build_kernel(&elt.right_face.face_type, &elt_storage.u_right_plus);
    unsafe {
        left_kernel.enq();
        right_kernel.enq();
    }
}


pub fn prepare_communication_kernels(u_ident: &str, bc_idents: &Vec<String>) -> String {
    let mut builder = string_builder::Builder::new(1_000);
    let interior = format!(r#"
__kernel void communicate_internal(
    __global {U}* destination,
    __global int index_in_destination,
    __global {U}* neighbor,
    __global int index_in_neighbor
) {{
    destination[index_in_destination] = neighbor[index_in_neighbor];
}}
"#, U = u_ident);
    builder.append(interior);
    builder.append("\n");
    for ref bc_ident in bc_idents {
        let exterior = format!(r#"
__kernel void communicate_external__{boundary_value_kernel}(
    __global {U}* destination,
    __global int index_in_destination,
    __local float t,
    __global {U}* existing_values,
    __global int existing_value_index
) {{
    {U} existing_value = existing_values[existing_value_index];
    {U} boundary_value = {boundary_value_kernel}(t, existing_value);
    destination[index_in_destination] = boundary_value;
}}
"#,
                               U = u_ident, boundary_value_kernel = bc_ident);
        builder.append(exterior);
        builder.append("\n");
    }
    builder.string().unwrap()
}

pub fn simulate<GS>(grid: &Grid<GS>)
    where
        GS: GalerkinScheme,
{}
