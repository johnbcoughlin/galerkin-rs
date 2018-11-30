extern crate ocl;

use ocl::{Buffer, ProQue};
use ocl::flags::{MemFlags};
use galerkin_1d::operators::Operators;

pub struct OperatorsStorage {
    // The Vandermonde matrix, stored row-major.
    pub v: Buffer<f32>,

    // The D_r derivative matrix
    pub d_r: Buffer<f32>,

    // The matrix lifting [a, b] to [a, ..., 0, ..., b]  with length n_p, followed by the inverse
    // mass matrix.
    pub lift: Buffer<f32>,
}

pub fn store_operators(operators: &Operators, pro_que: &ProQue) -> OperatorsStorage {
    let v: Vec<f32> = operators.v.row_iter()
        .flat_map(|row| row.iter())
        .map(|x| x as f32)
        .collect();
    let v = pro_que.buffer_builder()
        .copy_host_slice(v.as_slice())
        .flags(MemFlags::new().read_only())
        .build();

    let d_r: Vec<f32> = operators.d_r.row_iter()
        .flat_map(|row| row.iter())
        .map(|x| x as f32)
        .collect();
    let d_r = pro_que.buffer_builder()
        .copy_host_slice(d_r.as_slice())
        .flags(MemFlags::new().read_only())
        .build();

    let lift: Vec<f32> = operators.lift.row_iter()
        .flat_map(|row| row.iter())
        .map(|x| x as f32)
        .collect();
    let lift = pro_que.buffer_builder()
        .copy_host_slice(lift.as_slice())
        .flags(MemFlags::new().read_only())
        .build();

    OperatorsStorage { v, d_r, lift }
}
