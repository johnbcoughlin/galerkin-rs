#![feature(associated_type_defaults)]
#![feature(trace_macros)]

#[macro_use]
extern crate rulinalg;
extern crate gnuplot;
#[macro_use]
extern crate glium;
extern crate ocl;

#[macro_use]
pub mod testing;

pub mod blas;
pub mod distmesh;
pub mod functions;
pub mod galerkin_1d;
pub mod galerkin_2d;
pub mod opencl;
pub mod plot;
