extern crate ocl;

use self::ocl::{Buffer, OclPrm};

pub struct CLMatrix<T: OclPrm> {
    nrows: usize,
    ncols: usize,
    buf: Buffer<T>
}

pub struct CLVector<T: OclPrm> {
    size: usize,
    buf: Buffer<T>,
}

impl<T: OclPrm> CLVector<T> {
    pub fn buf(&self) -> &Buffer<T> {
        &self.buf
    }
}
