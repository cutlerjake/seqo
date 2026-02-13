use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub fn derive_add_assign(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            _ => {
                return syn::Error::new_spanned(
                    name,
                    "AddAssign can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "AddAssign can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let add_assigns1 = fields.iter().map(|f| {
        let ident = f.ident.as_ref().unwrap();
        quote! {
            self.#ident += rhs.#ident;
        }
    });

    let add_assigns2 = add_assigns1.clone();

    let expanded = quote! {
        impl std::ops::AddAssign for #name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                #(#add_assigns1)*
            }
        }

        impl std::ops::AddAssign<&#name> for #name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: &Self) {
                #(#add_assigns2)*
            }
        }
    };

    expanded.into()
}
