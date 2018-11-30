pub const LAX_FRIEDRICHS_FLUX_KERNEL: &'static str = r#"
__kernel void advec(
    __global {U}* u_minus,
    __global {U}* u_plus,
    __global float* f_minus,
    __global float* f_plus
) {
    int i = get_global_id();

}
"#;
