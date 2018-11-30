extern crate ocl;
#[macro_use]
extern crate cl_structs;
extern crate static_assertions;

use ocl::OclPrm;
use static_assertions::assert_impl;

#[derive(Debug, Copy, Clone, PartialEq, Default, OclPrm)]
struct U {
    a: f32,
    b: f32,
}

fn main() {
    assert_impl!(U, OclPrm);
}

