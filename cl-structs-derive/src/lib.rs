#![feature(custom_attribute)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
#![feature(proc_macro_span)]
#![feature(use_extern_macros)]

extern crate proc_macro;
extern crate syn;
extern crate quote;
extern crate ocl;

use proc_macro::{TokenStream, Span, Diagnostic};
use syn::{parse_macro_input, parse_quote, Data, Type, TypePath, Path, PathSegment, Ident, DataStruct, DeriveInput, Field, Fields, FieldsNamed, GenericParam, Generics, Index};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::spanned::Spanned;
use quote::*;

#[proc_macro_derive(ocl::OclPrm)]
pub fn ocl_prm_derive(input: TokenStream) -> TokenStream {
    println!("okay here we are");
    // Parse the input tokens into a syntax tree.
    let input = parse_macro_input!(input as DeriveInput);

    // Used in the quasi-quotation below as `#name`.
    let name = input.ident;

    let errors = match input.data {
        Data::Struct(DataStruct {
                         fields: Fields::Named(FieldsNamed { named, .. }), ..
                     }) => validate_fields(named),
        _ => {
            vec![String::from("The trait OclPrm may only be derived for structs")]
        },
    };

    println!("errors: {:?}", errors);

    if !errors.is_empty() {
        let mut output = TokenStream::new();
        for err in errors {
            output.extend(TokenStream::from(quote! {
                compile_error!("The trait OclPrm may only be derived for structs")
            }));
        }
        return output
    }

    let expanded = quote! {
        unsafe impl ::ocl::OclPrm for #name {}
    };
    TokenStream::from(expanded)
}

fn validate_fields(punctuated: Punctuated<Field, Comma>) -> Vec<String> {
    let mut errors = vec![];
    for ref field in punctuated.iter() {
        match field {
            Field { ident: Some(name), ty, .. } => {
                if is_of_type_f32(ty) {
                    println!("{:?}", ty);
                } else {
                    errors.push(format!("`{}` must be of type `f32`", name));
                }
            }
            Field { ident: Some(name), .. } => errors.push(format!(
                "`{}` must be of type `f32`", name)),
            Field { ident: None, .. } => errors.push(String::from("All fields must be named.")),
        }
    }
    errors
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
pub fn opencl_function(metadata: TokenStream, input: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(input).expect("failed to parse input");

    match item {
        syn::Item::Struct(itemfn) => (),
        _ => println!("AAAAHHHH"),
    }

    let expanded = quote! {
fn foo() {}
};

    TokenStream::from(expanded)
}
