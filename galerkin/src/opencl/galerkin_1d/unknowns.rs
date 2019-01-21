extern crate ocl;

use ocl::{Buffer, OclPrm, ProQue};
use std::iter::repeat;

pub trait Unknown: OclPrm {
    fn cl_struct_type() -> String;

    fn cl_struct_def() -> String;
}

#[macro_export]
macro_rules! cl_type (
    (f32) => {"float"};
);

#[macro_export]
macro_rules! gen_unknown {
    // Must capture T as tt because we will pass it to the cl_type macro.
    ($U:ident, $T:tt, $($field:ident),*) => {
        use ocl::OclPrm;

        #[repr(C)]
        #[derive(Debug, Default, PartialEq, Copy, Clone)]
        pub struct $U {
            $($field: $T,)*
        }

        unsafe impl OclPrm for $U {}

        impl Unknown for $U {
            fn cl_struct_type() -> String {
                format!("cl_{U}", U=stringify!($U))
            }

            fn cl_struct_def() -> String {
                let field_list = concat!(
                    $(concat!("\n\t", cl_type!($T), " ", stringify!($field), ";"),)*
                );
                format!("typedef struct {{{fields}\n}} {U};",
                fields=field_list, U=Self::cl_struct_type())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Unknown;

    gen_unknown!(Q, f32, a, b);

    #[test]
    fn test_unknown() {
        assert_eq!(
            Q::cl_struct_def(),
            r#"typedef struct {
	float a;
	float b;
} cl_Q;"#
        );
    }
}

pub fn initialize_residuals<T: OclPrm>(k: usize, n_p: i32, pro_que: &ProQue) -> Vec<Buffer<T>> {
    repeat(repeat(T::default()).take(n_p as usize).collect())
        .take(k)
        .map(|v: Vec<T>| {
            pro_que
                .buffer_builder()
                .len(v.len())
                .copy_host_slice(v.as_slice())
                .build()
                .unwrap()
        })
        .collect()
}
