#[macro_use]
extern crate cl_structs;

use cl_structs::opencl_function;

#[opencl_function]
struct S { //~ ERROR: opencl_function may only be applied to top-level functions
}
