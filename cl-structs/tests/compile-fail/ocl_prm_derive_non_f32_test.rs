extern crate ocl;
#[macro_use]
extern crate cl_structs;

use ocl::OclPrm;

//~ ERROR: some error here?
#[derive(Debug, Copy, Clone, PartialEq, Default, OclPrm)]
struct U {
    a: f64,
    b: i32,
}

fn main() {

}
