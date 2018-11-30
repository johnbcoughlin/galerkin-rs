#![feature(custom_attribute)]

extern crate ocl_macro;

use ocl_macro::opencl_function;

#[opencl_function()]
fn my_func(q: Q) {
}
