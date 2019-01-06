extern crate ocl;

use ocl::OclPrm;
use std::ops::*;
use std::marker::PhantomData;
use std::cell::{RefCell, Cell};

struct DryRunContext {
    name: String,
    params: RefCell<Vec<String>>,
    body: RefCell<String>,
    ident_count: RefCell<i32>,
}

impl DryRunContext {
    fn new(name: &str) -> Self {
        DryRunContext {
            name: String::from(name),
            ident_count: RefCell::new(0),
            body: RefCell::new("".to_string()),
            params: RefCell::new(vec![]),
        }
    }

    pub fn src(&self) -> String {
        let mut src: String = String::from("");
        src.push_str(&format!("__kernel void {}(\n", self.name));
        src.push_str(&self.params.borrow().join("\n,"));
        src.push_str("\n) {");
        src.push_str(&self.body.borrow().deref().clone());
        src
    }

    pub fn append_stmt(&self, stmt: &str) {
        self.body.borrow_mut().deref_mut().push_str(stmt);
    }

    pub fn var<T: ClPrim>(&self, name: &str) -> DryRunClPrim<T> {
        self.params.borrow_mut().push(format!("{} {}", T::cl_type(), name.to_string()));
        DryRunClPrim::new(self, name)
    }
}

struct DryRunHandle<'run> {
    expr_name: String,
    ctx: &'run DryRunContext,
}

trait DryRunMul<RHS = Self> {
    fn mul(lhs: &str, rhs: &str) -> String;
}

trait ClFuncScalar<T: ClPrim>: Sized + Mul<Output=Self> {}

impl<T: ClPrim + Mul<Output=T>> ClFuncScalar<T> for T {}

trait ClPrim: OclPrm + DryRunMul {
    fn cl_type() -> String;
}

//trait ClPrimWrapper<T: ClPrim> {}

struct DryRunClPrim<'run, T: ClPrim> {
    _marker: PhantomData<T>,
    handle: DryRunHandle<'run>,
}

impl<'run, T: ClPrim> DryRunClPrim<'run, T> {
    fn new(ctx: &'run DryRunContext, name: &str) -> DryRunClPrim<'run, T> {
        DryRunClPrim {
            _marker: PhantomData,
            handle: DryRunHandle {
                expr_name: name.to_string(),
                ctx,
            },
        }
    }

    fn next(ctx: &'run DryRunContext) -> DryRunClPrim<'run, T> {
        let mut tmp = ctx.ident_count.borrow_mut();
        let ident_count_borrow = tmp.deref_mut();
        let expr_name = format!("v{}", *ident_count_borrow);
        *ident_count_borrow += 1;
        Self::new(ctx, &expr_name)
    }
}

impl<'run, T> ClFuncScalar<T> for DryRunClPrim<'run, T> where T: ClPrim {}

impl<'run, T> Mul<DryRunClPrim<'run, T>> for DryRunClPrim<'run, T>
    where
        T: ClPrim {
    type Output = DryRunClPrim<'run, T>;

    fn mul(self, rhs: DryRunClPrim<'run, T>) -> DryRunClPrim<'run, T> {
        let result = DryRunClPrim::next(self.handle.ctx);
        let stmt = format!("{} {} = {}", T::cl_type(), result.handle.expr_name,
                           T::mul(&self.handle.expr_name, &rhs.handle.expr_name));
        self.handle.ctx.append_stmt(&stmt);
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_accepts_normal_arguments() {
        let f1 = ClFloat(3.);
        let f2 = ClFloat(4.);
        let f3 = kernel(ClFloat(3.), ClFloat(4.));
    }

    #[test]
    fn it_writes_its_own_source_code() {
        let mut ctx: DryRunContext = DryRunContext::new("kernel");
        let actual = kernel(ctx.var("f1"), ctx.var("f2"));
        assert_eq!("float v0 = f1 * f2", actual.handle.ctx.src());
    }

    fn kernel<F: ClFuncScalar<ClFloat>>(f1: F, f2: F) -> F {
        f1 * f2
    }

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    struct ClFloat(f32);

    unsafe impl OclPrm for ClFloat {}

    impl ClPrim for ClFloat {
        fn cl_type() -> String {
            "float".to_string()
        }
    }

    impl Mul for ClFloat {
        type Output = ClFloat;

        fn mul(self, rhs: ClFloat) -> ClFloat {
            ClFloat(self.0 * rhs.0)
        }
    }

    impl DryRunMul for ClFloat {
        fn mul(lhs: &str, rhs: &str) -> String {
            format!("{} * {}", lhs, rhs)
        }
    }
}
