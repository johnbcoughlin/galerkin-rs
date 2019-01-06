#![feature(trace_macros)]
#![feature(concat_idents)]
#![recursion_limit="256"]

extern crate ocl;
#[macro_use]
extern crate mashup;

use ocl::OclPrm;
use std::ops::*;
use std::marker::PhantomData;
use std::cell::{RefCell};

struct DryRunContext {
    name: String,
    params: RefCell<Vec<String>>,
    body: RefCell<String>,
    ident_count: RefCell<i32>,
}

impl DryRunContext {
    fn new(name: &str) -> Self {
        let result = DryRunContext {
            name: String::from(name),
            ident_count: RefCell::new(0),
            body: RefCell::new("".to_string()),
            params: RefCell::new(vec![]),
        };
        result.append_stmt("int global_id = get_global_id(0)");
        result
    }

    pub fn src<F, T>(&self, f: F) -> String
        where
            F: FnOnce() -> T,
            T: DryRun + DryRunRValue,
    {
        let output: T = f();
        // output is always passed as a buffer so that we may set it
        self.params.borrow_mut().push(
            format!("{}* output", <T::ExprType as ClPrim>::cl_type()));
        self.append_stmt(&format!("output[global_id] = {}", output.val()));

        let mut src: String = String::from("");
        src.push_str(&format!("__kernel void {}(\n\t", self.name));
        src.push_str(&self.params.borrow().join(",\n\t"));
        src.push_str("\n) {\n");
        src.push_str(&self.body.borrow().deref().clone());
        src.push_str("}");
        src
    }

    pub fn append_stmt(&self, stmt: &str) {
        let mut body = self.body.borrow_mut();
        body.push_str("\t");
        body.push_str(stmt);
        body.push_str(";\n");
    }

    pub fn var<'run, T: DryRunLValue<'run>>(&'run self, name: &str) -> T {
        // output is a reserved parameter name
        assert_ne!(name, "output");
        // global_id is a reserved parameter name
        assert_ne!(name, "global_id");
        self.params.borrow_mut().push(
            format!("{} {}", <T::ClType as ClPrim>::cl_type(), name.to_string()));
        T::new(self, name)
    }
}

struct DryRunHandle<'run> {
    expr_name: String,
    ctx: &'run DryRunContext,
}

// Common traits

trait DryRun {
    type ExprType: ClPrim;

    fn handle(&self) -> &DryRunHandle;
}

trait DryRunLValue<'run> {
    type ClType: ClPrim;

    fn new(ctx: &'run DryRunContext, name: &str) -> Self;
}

trait DryRunRValue {
    fn val(&self) -> String;
}

trait DryRunMul<RHS = Self> {
    fn mul(lhs: &str, rhs: &str) -> String;
}

trait ClPrim: OclPrm {
    fn cl_type() -> String;
}

// An opencl primitive value, like float
trait ClFuncScalar<T: ClPrim>: Sized + Mul<Output=Self> {}

// A marker trait for structs.
trait ClFuncAtom<T: ClPrim>: Sized {}

// Dry run value of a struct type
struct DryRunClAtom<'run, T: ClPrim> {
    _marker: PhantomData<T>,
    handle: DryRunHandle<'run>,
}

impl<'run, T: ClPrim> DryRun for DryRunClAtom<'run, T> {
    type ExprType = T;

    fn handle(&self) -> &DryRunHandle {
        &self.handle
    }
}

impl<'run, T: ClPrim> ClFuncAtom<T> for DryRunClAtom<'run, T> {}

impl<'run, T: ClPrim> DryRunLValue<'run> for DryRunClAtom<'run, T> {
    type ClType = T;

    fn new(ctx: &'run DryRunContext, name: &str) -> Self {
        DryRunClAtom {
            _marker: PhantomData,
            handle: DryRunHandle {
                expr_name: String::from(name),
                ctx
            }
        }
    }
}

macro_rules! atom_accessors (
    ($S:ident, {$($field:ident: $T:ty),*}) => {
        mashup! {
            m["accessorTrait"] = $S Accessor;
            m["creatorTrait"] = $S Creator;
            $(
                m["type" $field] = TypeOf_ $field;
            )*
        }

        m! {
            pub trait "accessorTrait" {
                $(type "type" $ field;
                )*
                $(fn $field(& self) -> Self::"type" $ field;)*
            }
        }

        m! {
            pub trait "creatorTrait" {
                $(type "type" $ field;
                )*
                fn new($($field: )*) -> Self;
            }
        }

        m! {
            impl "accessorTrait" for $S {
                $(type "type" $field = $T;
                )*

                $(fn $field(&self) -> Self::"outputType" $field {
                    self.$field
                })*
            }
        }

        m! {
            impl "creatorTrait" for $S {
                $(type "type" $field = $T;
                )*

                fn new($($field: "type" $field)*) -> Self {
                    S { $($field,)* }
                }
            }
        }

        m! {
            impl "accessorTrait" for Vec<$S> {
                $(
                    type "outputType" $field = Vec<$T>;
                )*
                $(fn $field(&self) -> Self::"outputType" $field {
                    self.iter()
                        .map(|s| s.$field())
                        .collect()
                })*
            }
        }

        m! {
            impl<'run> "accessorTrait" for DryRunClAtom<'run, $S> {
                $(type "type" $field = DryRunClScalar<'run, <$S as "traitName">::"outputType" $field>;
                )*
                $(fn $field(&self) -> Self::"outputType" $field {
                    let result = DryRunClScalar::next(self.handle.ctx);
                    let stmt = format!("{} {} = {}.{}", <$T as ClPrim>::cl_type(), result.handle.expr_name,
                        self.handle.expr_name, stringify!($field));
                    self.handle.ctx.append_stmt(&stmt);
                    result
                })*
            }
        }

        m! {
            impl<'run> "creatorTrait" for DryRunClAtom<'run, $S> {
                $(type "type" $field = DryRunS)
            }
        }
    }
);

impl<T: ClPrim + Mul<Output=T>> ClFuncScalar<T> for T {}

impl<T: ClPrim> ClFuncAtom<T> for T {}

struct DryRunClScalar<'run, T: ClPrim> {
    _marker: PhantomData<T>,
    handle: DryRunHandle<'run>,
}

impl<'run, T: ClPrim> DryRunClScalar<'run, T> {
    fn next(ctx: &'run DryRunContext) -> DryRunClScalar<'run, T> {
        let mut tmp = ctx.ident_count.borrow_mut();
        let ident_count_borrow = tmp.deref_mut();
        let expr_name = format!("v{}", *ident_count_borrow);
        *ident_count_borrow += 1;
        Self::new(ctx, &expr_name)
    }
}

impl<'run, T: ClPrim> DryRunLValue<'run> for DryRunClScalar<'run, T> {
    type ClType = T;

    fn new(ctx: &'run DryRunContext, name: &str) -> DryRunClScalar<'run, T> {
        DryRunClScalar {
            _marker: PhantomData,
            handle: DryRunHandle {
                expr_name: name.to_string(),
                ctx,
            },
        }
    }
}

impl<'run, T: ClPrim> DryRun for DryRunClScalar<'run, T> {
    type ExprType = T;

    fn handle(&self) -> &DryRunHandle {
        &self.handle
    }
}

impl<'run, T: ClPrim> DryRunRValue for DryRunClScalar<'run, T> {
    fn val(&self) -> String {
        self.handle.expr_name.clone()
    }
}

impl<'run, T> ClFuncScalar<T> for DryRunClScalar<'run, T>
    where T: ClPrim + DryRunMul {}

impl<'run, T> Mul<DryRunClScalar<'run, T>> for DryRunClScalar<'run, T>
    where
        T: ClPrim + DryRunMul {
    type Output = DryRunClScalar<'run, T>;

    fn mul(self, rhs: DryRunClScalar<'run, T>) -> DryRunClScalar<'run, T> {
        let result = DryRunClScalar::next(self.handle.ctx);
        let stmt = format!("{} {} = {}", T::cl_type(), result.handle.expr_name,
                           T::mul(&self.handle.expr_name, &rhs.handle.expr_name));
        self.handle.ctx.append_stmt(&stmt);
        result
    }
}


#[cfg(test)]
mod tests {
    #[macro_use]
    use super::*;

    #[test]
    fn it_accepts_normal_arguments() {
        let f3 = kernel(3., 4.);
    }

    #[test]
    fn it_writes_its_own_source_code() {
        let mut ctx: DryRunContext = DryRunContext::new("kernel");
        let actual = ctx.src(|| kernel::<DryRunClScalar<f32>>(ctx.var("f1"), ctx.var("f2")));
        println!("{}", actual);
        assert_eq!(actual, "__kernel void kernel(\n\tfloat f1,\n\tfloat f2,\n\tfloat* output\n) {
\tint global_id = get_global_id(0);
\tfloat v0 = f1 * f2;
\toutput[global_id] = v0;
}");
    }

    fn kernel<F: ClFuncScalar<f32>>(f1: F, f2: F) -> F {
        f1 * f2
    }

    impl ClPrim for f32 {
        fn cl_type() -> String {
            "float".to_string()
        }
    }

    impl DryRunMul for f32 {
        fn mul(lhs: &str, rhs: &str) -> String {
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
        fn point_kernel<
            P: ClFuncAtom<Point> + PointAccessor,
        >(p1: P) -> P::OutputTypeOf_x {
            p1.x()
        }

        assert_eq!(3.0f32, point_kernel(Point {x: 3.0f32, y: 0.0f32}));

        let mut ctx: DryRunContext = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| point_kernel::<
            DryRunClAtom<Point>,
        >(ctx.var("f1")));
        println!("{}", actual);
        assert!(actual.contains("float v0 = f1.x;
	output[global_id] = v0;"));
    }

    #[test]
    fn it_returns_structs() {
        fn point_kernel<
            P: ClFuncAtom<Point> + PointAccessor,
        >(p1: P) -> P::OutputTypeOf_x {
            P::new(p1.x(), p2.x())
        }
        let mut ctx: DryRunContext = DryRunContext::new("point_kernel");
        let actual: String = ctx.src(|| point_kernel::<
            DryRunClAtom<Point>,
        >(ctx.var("f1")));
        println!("{}", actual);
        assert!(actual.contains("float v0 = f1.x;
	output[global_id] = v0;"));
    }
}
