#![feature(custom_attribute)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
#![feature(proc_macro_span)]

extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;
extern crate quote;
extern crate ocl;

use proc_macro::{TokenStream, Span, Diagnostic, Level};
use syn::{parse_macro_input, Data, Type, TypePath, Path, PathSegment, Ident, DataStruct, DeriveInput, Field, Fields, FieldsNamed, GenericParam, Generics, Index};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::spanned::Spanned;
use quote::*;

mod model;

use model::*;

#[proc_macro_derive(ocl::OclPrm)]
pub fn ocl_prm_derive(input: proc_macro::TokenStream) -> TokenStream {
    println!("okay here we are");
    // Parse the input tokens into a syntax tree.
    let input = parse_macro_input!(input as DeriveInput);

    // Used in the quasi-quotation below as `#name`.
    let name = &input.ident;
    let input_span = input.span();

    match input.data {
        Data::Struct(ds) => {
            match ds.fields {
                Fields::Named(fields) => validate_fields(fields.named),
                _ => input_span.unstable()
                    .error("Struct fields must be named when deriving OclPrm")
                    .emit(),
            }
        },
        _ => {
            input_span.unstable()
                .error("OclPrm may only be derived for structs")
                .emit()
        },
    }

    let expanded = quote! {
        unsafe impl ::ocl::OclPrm for #name {}
    };

    TokenStream::from(expanded)
}

fn validate_fields(punctuated: Punctuated<Field, Comma>) {
    for ref field in punctuated.iter() {
        match field {
            Field { ident: Some(name), ty, .. } => {
                if is_of_type_f32(ty) {
                    println!("{:?}", ty);
                } else {
                    field.span().unstable()
                        .error("All fields must be f32")
                        .emit();
                }
            },
            Field { ident: None, .. } => field.span().unstable()
                .error("All fields must be named")
                .emit(),
        }
    }
}

fn is_of_type_f32(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { path: Path { segments, .. }, .. }) => {
            if segments.len() == 1 {
                match segments.first().unwrap().value() {
                    PathSegment { ident, .. } => ident.to_string() == "f32",
                    _ => false,
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

#[proc_macro_attribute]
pub fn opencl_kernel(metadata: TokenStream, input: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(input).expect("failed to parse input");

    match item {
        syn::Item::Fn(syn::ItemFn {decl}) => ,
        _ => item.span().unstable()
            .error("opencl_function may only be applied to top-level functions.")
            .emit(),
    }

    let expanded = quote! {
fn my_func() {}
fn my_func_src() {}
};

    TokenStream::from(expanded)
}

fn expand_opencl_function(itemfn: syn::ItemFn) -> OpenclFunctionExpansion {
}

fn validate_original_function(itemfn: syn::ItemFn) -> Option<Func> {
    match itemfn.decl.output {
        syn::ReturnType::Default => {
            decl.span().unstable()
                .error("Function must have a non-() return type")
                .emit();
            None
        },
        syn::ReturnType::Type(_, Box(ty)) => match ty {
            Type::Path(TypePath {})
        }
    }?;
}

fn generate_kernel_src(itemfn: syn::ItemFn) -> Option<TokenStream> {
    let statements = itemfn.block.stmts;
    itemfn.dec
}

fn generate_kernel_decl(func: Func) -> String {
    let mut result = String::new();
    result.push_str(&format!("__kernel void {}(", func.name));
    for param in &func.params {
        result.push_str(&format!(
            "{} {},\n", param.param_type.as_cl_type(), param.ident));
    }
    result.push_str(")");
    result
}

fn generate_kernel_body() -> Option<TokenStream> {

}

fn generate_kernel_invocation(itemfn: syn::ItemFn) -> Option<TokenStream> {

}

struct OpenclFunctionExpansion {
    original: TokenStream,
    src: TokenStream,
    kernel_invocation: TokenStream,
}
