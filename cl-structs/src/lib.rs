extern crate ocl;
extern crate cl_structs_derive;

pub use cl_structs_derive::*;

use ocl::OclPrm;

pub trait Unknown: OclPrm {
    fn cl_struct_type() -> String;

    fn cl_struct_def() -> String;
}
