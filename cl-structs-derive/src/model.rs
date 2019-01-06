extern crate syn;

pub struct Func {
    pub name: syn::Ident,
    pub params: Vec<Param>,
    pub return_type: ReturnType,
}

pub struct Param {
    pub ident: syn::Ident,
    pub param_type: ParamType,
}

pub enum OpenclFuncType {
    Scalar(syn::Ident),
    Array(syn::Ident),
    Matrix(syn::Ident),
}

impl OpenclFuncType {
    fn from_syn_type(ty: syn::Type) -> OpenclFuncType {
        match ty {
            syn::TypePath {path: syn::Path {
                segments
            }}
        }
    }
}

pub trait AsClType {
    fn as_cl_type(&self) -> &'static str;
}

impl AsClType for ParamType {
    fn as_cl_type(&self) -> &'static str {
        match self {
            OpenclFuncType::Scalar(ident) => ident.as_cl_type(),
            OpenclFuncType::Array(ident) | OpenclFuncType::Matrix(ident) =>
                format!("__global {}*", ident.as_cl_type()),
        }
    }
}

impl AsClType for syn::Ident {
    fn as_cl_type(&self) -> &'static str {
        match self.to_string().as_ref() {
            "f32" => "float",
            "i32" | "u32" => "int",
            e @ _ => format!("cl_{}", e),
        }
    }
}

pub enum ReturnType {
    Scalar,
    Array,
}