extern crate ocl;

use ocl::OclPrm;

pub trait Unknown: OclPrm {
    fn cl_struct_type() -> String;

    fn cl_struct_def() -> String;
}

#[macro_export]
macro_rules! cl_type (
    (f32) => {"float"};
);

#[macro_export]
macro_rules! unknown {
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

    unknown!(Q, f32, a, b);

    #[test]
    fn test_unknown() {
        assert_eq!(Q::cl_struct_def(), r#"typedef struct {
	float a;
	float b;
} cl_Q;"#);
    }
}