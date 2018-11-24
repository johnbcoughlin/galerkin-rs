#![feature(custom_attribute)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
#![feature(proc_macro_span)]
#![feature(use_extern_macros)]

extern crate proc_macro;
extern crate syn;
extern crate quote;

use proc_macro::{TokenStream, Span, Diagnostic};
use syn::spanned::Spanned;
use quote::*;

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
