extern crate ocl;
#[macro_use]
extern crate cl_structs;

use ocl::OclPrm;

#[derive(Debug, Copy, Clone, PartialEq, Default, OclPrm)]
struct U {
    a: f64, //~ ERROR: All fields must be f32
    b: i32, //~ ERROR: All fields must be f32
}

#[derive(Debug, Copy, Clone, PartialEq, Default, OclPrm)]
struct V(f64); //~ ERROR: Struct fields must be named

#[derive(OclPrm)]
enum Foo { //~ ERROR: OclPrm may only be derived for structs
    Var1,
}

fn main() {

}
