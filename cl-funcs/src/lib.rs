#![feature(trace_macros)]
#![feature(concat_idents)]
#![feature(arbitrary_self_types)]
#![recursion_limit = "256"]

extern crate ocl;
#[macro_use]
extern crate mashup;

use self::Like::*;
use ocl::OclPrm;
use std::cell::RefCell;
use std::ops::*;
use std::rc::Rc;

#[derive(Debug)]
struct DryRunContext {
    name: String,
    params: RefCell<Vec<String>>,
    body: RefCell<String>,
    ident_count: RefCell<i32>,
}

impl DryRunContext {
    fn new(name: &str) -> Rc<Self> {
        let result = Rc::new(DryRunContext {
            name: String::from(name),
            ident_count: RefCell::new(0),
            body: RefCell::new("".to_string()),
            params: RefCell::new(vec![]),
        });
        result
            .clone()
            .append_stmt("int global_id = get_global_id(0)");
        result
    }

    pub fn src<F, T>(self: &Rc<Self>, f: F) -> String
    where
        F: FnOnce() -> Like<T>,
        Like<T>: ClTypeable,
    {
        let output: Like<T> = f();
        // output is always passed as a buffer so that we may set it
        self.params
            .borrow_mut()
            .push(format!("__global {}* output", output.cl_type()));
        self.append_stmt(&format!(
            "output[global_id] = {}",
            output.unwrap_dry_run().expr_name
        ));

        let mut src: String = String::from("");
        src.push_str(&format!("__kernel void {}(\n\t", self.name));
        src.push_str(&self.params.borrow().join(",\n\t"));
        src.push_str("\n) {\n");
        src.push_str(&self.body.borrow().deref().clone());
        src.push_str("}");
        src
    }

    pub fn append_stmt(self: &Rc<Self>, stmt: &str) {
        let mut body = self.body.borrow_mut();
        body.push_str("\t");
        body.push_str(stmt);
        body.push_str(";\n");
    }

    fn new_var<T>(self: &Rc<Self>, name: &str, is_param: bool) -> Like<T> {
        // output is a reserved parameter name
        assert_ne!(name, "output");
        // global_id is a reserved parameter name
        assert_ne!(name, "global_id");
        DryRun(Handle {
            expr_name: String::from(name),
            ctx: self.clone(),
            metadata: DryRunMetadata { is_param },
        })
    }

    fn param<T>(self: &Rc<Self>, name: &str) -> Like<T>
    where
        Like<T>: ClTypeable,
    {
        let result = self.new_var(name, true);
        self.params
            .borrow_mut()
            .push(format!("{} {}", result.cl_type(), name.to_string()));
        result
    }

    fn next_var<T, F>(self: &Rc<Self>, f: F) -> Like<T>
    where
        Like<T>: ClTypeable,
        F: FnOnce(&Like<T>) -> String,
    {
        let expr_name = {
            let mut tmp = self.ident_count.borrow_mut();
            let expr_name = format!("v{}", tmp);
            *tmp += 1;
            expr_name
        };
        let var = self.new_var(&expr_name, false);
        let stmt = f(&var);
        self.append_stmt(&stmt);
        var
    }
}

struct DryRunHandle<'run> {
    expr_name: String,
    ctx: &'run DryRunContext,
}

#[derive(Copy, Clone, Debug)]
struct DryRunMetadata {
    is_param: bool,
}

#[derive(Debug, Clone)]
pub struct Handle {
    expr_name: String,
    ctx: Rc<DryRunContext>,
    metadata: DryRunMetadata,
}

// Common traits

pub trait DryRunMul<RHS = Self> {
    fn dry_run_mul(lhs: &str, rhs: &str) -> String;
}

trait DryRunIndex {
    type Output;

    fn index(&self) -> Self::Output;
}

pub trait ClPrim: OclPrm {
    fn cl_prim_type() -> String;
}

pub trait ClTypeable {
    fn cl_type(&self) -> String;
}

impl<T: ClPrim> ClTypeable for T {
    fn cl_type(&self) -> String {
        Self::cl_prim_type()
    }
}

impl<T: ClPrim> ClTypeable for Like<Vec<T>> {
    fn cl_type(&self) -> String {
        match self {
            Actual(_) => unimplemented!(),
            DryRun(handle) => {
                if handle.metadata.is_param {
                    format!("__global {}*", T::cl_prim_type())
                } else {
                    T::cl_prim_type()
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum Like<T> {
    Actual(T),
    DryRun(Handle),
}

impl<T: ClPrim> ClTypeable for Like<T> {
    fn cl_type(&self) -> String {
        T::cl_prim_type()
    }
}

impl<T> Like<T> {
    fn unwrap_actual(self) -> T {
        match self {
            Actual(t) => t,
            DryRun(_) => panic!("Expected Like::Real"),
        }
    }

    fn unwrap_dry_run(self) -> Handle {
        match self {
            Actual(_) => panic!("Expected Like::DryRun"),
            DryRun(h) => h,
        }
    }

    fn expr_name(&self) -> String {
        match self {
            Actual(_) => panic!("Expected Like::DryRun"),
            DryRun(h) => h.expr_name.clone(),
        }
    }
}

impl<T: ClPrim> Mul for Like<T>
where
    T: Mul<T, Output = T> + DryRunMul,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match self {
            Actual(t) => Actual(t * rhs.unwrap_actual()),
            DryRun(handle) => handle.ctx.next_var(|result| {
                format!(
                    "{} {} = {}",
                    result.cl_type(),
                    result.expr_name(),
                    T::dry_run_mul(&handle.expr_name, &rhs.unwrap_dry_run().expr_name)
                )
            }),
        }
    }
}

impl<T: ClPrim> DryRunIndex for Like<Vec<T>> {
    type Output = Like<T>;

    fn index(&self) -> Like<T> {
        match self {
            Actual(_) => panic!("Indexing into an actual vector is not allowed"),
            DryRun(handle) => {
                if handle.metadata.is_param {
                    handle.ctx.next_var(|result| {
                        format!(
                            "{} {} = {}[global_id]",
                            result.cl_type(),
                            result.expr_name(),
                            handle.expr_name
                        )
                    })
                } else {
                    DryRun(handle.clone())
                }
            }
        }
    }
}

impl<T: ClPrim> Mul for Like<Vec<T>>
where
    T: Mul<T, Output = T> + DryRunMul,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match self {
            Actual(vec) => Actual(
                vec.into_iter()
                    .zip(rhs.unwrap_actual().into_iter())
                    .map(|(a, b)| a * b)
                    .collect(),
            ),
            DryRun(ref handle) => handle.ctx.next_var(|result| {
                format!(
                    "{} {} = {}",
                    result.cl_type(),
                    result.expr_name(),
                    T::dry_run_mul(&self.index().expr_name(), &rhs.index().expr_name())
                )
            }),
        }
    }
}

macro_rules! atom_accessors {
    ($S:ident, {$($field:ident: $T:ty),+}) => {
        mashup! {
            m["trait"] = $S like;
        }

        m! {
            pub trait "trait" {
                $(fn $field(& self) -> Like<$T>;)+

            }
        }

        m! {
            impl "trait" for Like<$S> {
                $(fn $field(&self) -> Like<$T> {
                    match self {
                        Like::Actual(t) => Like::Actual(t.$field),
                        Like::DryRun(handle) => {
                            handle.ctx.next_var(|result| {
                                format!("{} {} = {}.{}", result.cl_type(),
                                    result.expr_name(), handle.expr_name, stringify!($field))
                            })
                        },
                    }
                })*
            }
        }

        m! {
            impl $S {
                fn new($($field: Like<$T>,)+) -> Like<Self> {
                    let fields = vec![$(&$field,)+];
                    assert!(!fields.is_empty());
                    let ref arg1 = fields[0];
                    match arg1 {
                        Like::Actual(_) => Like::Actual($S {
                            $($field: Like::unwrap_actual($field),
                            )+
                        }),
                        Like::DryRun(handle) => {
                            handle.ctx.next_var(|result| {
                                let type_name = result.cl_type();
                                let field_names: String = fields.iter()
                                .map(|f| f.expr_name())
                                .collect::<Vec<String>>().join(", ");
                                format!("{} {} = ({}) {{ {} }}", type_name,
                                    result.expr_name(), type_name, field_names)
                            })
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[macro_use]
    use super::*;

    #[test]
    fn it_accepts_normal_arguments() {
        let _f3 = kernel(Actual(3.), Actual(4.));
    }

    #[test]
    fn it_writes_its_own_source_code() {
        let ctx = DryRunContext::new("kernel");
        let actual = ctx.src(|| kernel(ctx.param("f1"), ctx.param("f2")));
        println!("{}", actual);
        assert_eq!(
            actual,
            "__kernel void kernel(\n\tfloat f1,\n\tfloat f2,\n\t__global float* output\n) {
\tint global_id = get_global_id(0);
\tfloat v0 = f1 * f2;
\toutput[global_id] = v0;
}"
        );
    }

    fn kernel(f1: Like<f32>, f2: Like<f32>) -> Like<f32> {
        f1 * f2
    }

    impl ClPrim for f32 {
        fn cl_prim_type() -> String {
            "float".to_string()
        }
    }

    impl DryRunMul for f32 {
        fn dry_run_mul(lhs: &str, rhs: &str) -> String {
            format!("{} * {}", lhs, rhs)
        }
    }

    #[derive(Copy, Clone, Default, Debug, PartialEq, PartialOrd)]
    struct Point {
        x: f32,
        y: f32,
    }

    unsafe impl OclPrm for Point {}

    impl ClPrim for Point {
        fn cl_prim_type() -> String {
            String::from("cl_Point")
        }
    }

    atom_accessors!(Point, {x: f32, y: f32});

    #[test]
    fn it_destructures_structs() {
        fn point_kernel(p1: Like<Point>) -> Like<f32> {
            p1.x()
        }

        assert_eq!(
            3.0f32,
            point_kernel(Actual(Point {
                x: 3.0f32,
                y: 0.0f32
            }))
            .unwrap_actual()
        );

        let ctx = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| point_kernel(ctx.param("f1")));
        println!("{}", actual);
        assert!(actual.contains(
            "float v0 = f1.x;
	output[global_id] = v0;"
        ));
    }

    #[test]
    fn it_returns_structs() {
        fn point_kernel(p1: Like<Point>) -> Like<Point> {
            Point::new(p1.x(), p1.y())
        }
        let ctx = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| point_kernel(ctx.param("f1")));
        println!("{}", actual);
        assert!(actual.contains(
            "cl_Point v2 = (cl_Point) { v0, v1 };
	output[global_id] = v2;"
        ));
    }

    #[test]
    fn it_multiplies_vecs() {
        fn vec_kernel(a: Like<Vec<f32>>, b: Like<Vec<f32>>) -> Like<Vec<f32>> {
            a * b
        }
        let actual = vec_kernel(Actual(vec![1., 2., 3.]), Actual(vec![4., 5., 6.]));
        assert_eq!(actual.unwrap_actual(), vec![4., 10., 18.]);

        let ctx = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| vec_kernel(ctx.param("a"), ctx.param("b")));
        println!("{}", actual);
        assert_eq!(actual, "__kernel void point_kernel(
	__global float* a,
	__global float* b,
	__global float* output
) {
	int global_id = get_global_id(0);
	float v1 = a[global_id];
	float v2 = b[global_id];
	float v0 = v1 * v2;
	output[global_id] = v0;
}"
        );
    }
}
