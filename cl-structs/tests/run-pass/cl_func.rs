#[macro_use]
extern crate cl_structs;

use cl_structs::opencl_function;

#[opencl_function]
fn my_func() {
}

fn main() {
    my_func();
    my_func_src();
    my_func_kernel();
}
