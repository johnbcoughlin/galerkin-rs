extern crate ocl;

use ocl::Buffer;

pub struct LaxFriedrichsFluxStorage {
    pub du_left: Buffer<f32>,
    pub du_right: Buffer<f32>,
}



pub const LAX_FRIEDRICHS_FLUX_KERNEL: &'static str = r#"
__kernel void advec(
    __global {U}* u_minus,
    __global {U}* u_plus,
    __global float* f_minus,
    __global float* f_plus,
    __global float* du_left,
    __global float* du_right
) {
    int i = get_global_id();


}
"#;
