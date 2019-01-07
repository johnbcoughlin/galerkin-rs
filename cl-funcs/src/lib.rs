#![feature(trace_macros)]
#![feature(concat_idents)]
#![feature(arbitrary_self_types)]
#![recursion_limit="256"]

extern crate ocl;
#[macro_use]
extern crate mashup;

use ocl::OclPrm;
use std::ops::*;
use std::cell::{RefCell};
use std::rc::Rc;
use self::Like::*;

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
        result.clone().append_stmt("int global_id = get_global_id(0)");
        result
    }

    pub fn src<F, T>(self: &Rc<Self>, f: F) -> String
        where
            F: FnOnce() -> Like<T>,
            T: ClPrim,
    {
        let output: Like<T> = f();
        // output is always passed as a buffer so that we may set it
        self.params.borrow_mut().push(
            format!("{}* output", <T as ClPrim>::cl_type()));
        self.append_stmt(&format!("output[global_id] = {}", output.unwrap_dry_run().expr_name));

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

    fn new_var<T: ClPrim>(self: &Rc<Self>, name: &str) -> Like<T> {
        // output is a reserved parameter name
        assert_ne!(name, "output");
        // global_id is a reserved parameter name
        assert_ne!(name, "global_id");
        DryRun(Handle {
            expr_name: String::from(name),
            ctx: self.clone(),
        })
    }

    fn param<T: ClPrim>(self: &Rc<Self>, name: &str) -> Like<T> {
        self.params.borrow_mut().push(
            format!("{} {}", <T as ClPrim>::cl_type(), name.to_string()));
        self.new_var(name)
    }

    fn next_var<T: ClPrim>(self: &Rc<Self>) -> Like<T> {
        let mut tmp = self.ident_count.borrow_mut();
        let ident_count_borrow = tmp.deref_mut();
        let expr_name = format!("v{}", *ident_count_borrow);
        *ident_count_borrow += 1;
        self.new_var(&expr_name)
    }
}

struct DryRunHandle<'run> {
    expr_name: String,
    ctx: &'run DryRunContext,
}

#[derive(Debug)]
pub struct Handle {
    expr_name: String,
    ctx: Rc<DryRunContext>,
}

// Common traits

pub trait DryRunMul<RHS = Self> {
    fn dry_run_mul(lhs: &str, rhs: &str) -> String;
}

pub trait ClPrim: OclPrm {
    fn cl_type() -> String;
}

#[derive(Debug)]
pub enum Like<T: ClPrim> {
    Actual(T),
    DryRun(Handle),
}

impl<T: ClPrim> Like<T> {
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
    T: Mul<T, Output=T> + DryRunMul {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match self {
            Actual(t) => Actual(t * rhs.unwrap_actual()),
            DryRun(handle) => {
                let result = handle.ctx.next_var();
                let stmt = format!("{} {} = {}", T::cl_type(), result.expr_name(),
                                   T::dry_run_mul(&handle.expr_name, &rhs.unwrap_dry_run().expr_name));
                handle.ctx.append_stmt(&stmt);
                result
            }
        }
    }
}

trait DryRunLValue<'run> {
    type ClType: ClPrim;

    fn new(ctx: &'run DryRunContext, name: &str) -> Self;

    fn next(ctx: &'run DryRunContext) -> Self;
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
                            let result = handle.ctx.next_var();
                            let stmt = format!("{} {} = {}.{}", <$T as ClPrim>::cl_type(),
                                result.expr_name(), handle.expr_name, stringify!($field));
                            handle.ctx.clone().append_stmt(&stmt);
                            result
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
                            let result = handle.ctx.next_var();
                            let type_name = <$S as ClPrim>::cl_type();
                            let field_names: String = fields.iter()
                                .map(|f| f.expr_name())
                                .collect::<Vec<String>>().join(", ");
                            let stmt = format!("{} {} = ({}) {{ {} }}", type_name,
                                result.expr_name(), type_name, field_names);
                            handle.ctx.append_stmt(&stmt);
                            result
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
        assert_eq!(actual, "__kernel void kernel(\n\tfloat f1,\n\tfloat f2,\n\tfloat* output\n) {
\tint global_id = get_global_id(0);
\tfloat v0 = f1 * f2;
\toutput[global_id] = v0;
}");
    }

    fn kernel(f1: Like<f32>, f2: Like<f32>) -> Like<f32> {
        f1 * f2
    }

    impl ClPrim for f32 {
        fn cl_type() -> String {
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
        fn cl_type() -> String {
            String::from("cl_Point")
        }
    }

    atom_accessors!(Point, {x: f32, y: f32});

    #[test]
    fn it_destructures_structs() {
        fn point_kernel(p1: Like<Point>) -> Like<f32> {
            p1.x()
        }

        assert_eq!(3.0f32, point_kernel(Actual(Point {x: 3.0f32, y: 0.0f32})).unwrap_actual());

        let ctx = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| point_kernel(ctx.param("f1")));
        println!("{}", actual);
        assert!(actual.contains("float v0 = f1.x;
	output[global_id] = v0;"));
    }

    #[test]
    fn it_returns_structs() {
        fn point_kernel(p1: Like<Point>) -> Like<Point> {
            Point::new(p1.x(), p1.y())
        }
        let ctx = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| point_kernel(ctx.param("f1")));
        println!("{}", actual);
        assert!(actual.contains("cl_Point v2 = (cl_Point) { v0, v1 };
	output[global_id] = v2;"));
    }
}
