use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, Fields, Ident, Type};

pub fn expand_block_delta(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;

    // Parse struct-level attributes for block type and optional context
    let (block_ty, ctx_ty) = parse_struct_block_attr(&input.attrs)?;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "BlockDelta only supports named structs",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "BlockDelta only supports structs",
            ));
        }
    };

    let mut field_exprs = Vec::new();
    for field in fields {
        let name = field.ident.as_ref().unwrap();
        if let Some(expr) = parse_field_block_delta(&field.attrs)? {
            field_exprs.push((name.clone(), expr));
        }
    }

    Ok(generate_impl(struct_name, &block_ty, &ctx_ty, &field_exprs))
}

// Parse struct-level attribute: #[block_delta(block = ..., context = ...)]
fn parse_struct_block_attr(attrs: &[Attribute]) -> syn::Result<(Type, Type)> {
    let mut block_ty: Option<Type> = None;
    let mut ctx_ty: Option<Type> = None;

    for attr in attrs {
        if attr.path().is_ident("block_delta") {
            let tokens = &attr.meta.require_list()?.tokens;
            let tokens_str = tokens.to_string();

            // crude parser: split on commas
            for token in tokens_str.split(',') {
                let token = token.trim();
                if token.starts_with("block") {
                    let parts: Vec<&str> = token.split('=').collect();
                    if parts.len() == 2 {
                        let ty: Type = syn::parse_str(parts[1].trim())?;
                        block_ty = Some(ty);
                    }
                } else if token.starts_with("context") {
                    let parts: Vec<&str> = token.split('=').collect();
                    if parts.len() == 2 {
                        let ty: Type = syn::parse_str(parts[1].trim())?;
                        ctx_ty = Some(ty);
                    }
                }
            }
        }
    }

    let block_ty = block_ty
        .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "Missing block type"))?;
    let ctx_ty = ctx_ty.unwrap_or_else(|| syn::parse_str("()").unwrap()); // default to ()
    Ok((block_ty, ctx_ty))
}

// Parse field-level attribute #[block_delta(<expr>)]
fn parse_field_block_delta(attrs: &[Attribute]) -> syn::Result<Option<Expr>> {
    for attr in attrs {
        if attr.path().is_ident("block_delta") {
            let expr = syn::parse2::<Expr>(attr.meta.require_list()?.tokens.clone())?;
            return Ok(Some(expr));
        }
    }
    Ok(None)
}

// Generate add_block / sub_block with ctx reference
fn generate_impl(
    struct_name: &Ident,
    block_ty: &Type,
    ctx_ty: &Type,
    field_exprs: &[(Ident, Expr)],
) -> proc_macro2::TokenStream {
    let add_stmts = field_exprs.iter().map(|(name, expr)| {
        quote! {
            self.#name += (#expr);
        }
    });

    let sub_stmts = field_exprs.iter().map(|(name, expr)| {
        quote! {
            self.#name -= (#expr);
        }
    });

    quote! {
        impl #struct_name {
            #[inline(always)]
            pub fn add_block(&mut self, block: &#block_ty, ctx: &#ctx_ty) {
                #(#add_stmts)*
            }

            #[inline(always)]
            pub fn sub_block(&mut self, block: &#block_ty, ctx: &#ctx_ty) {
                #(#sub_stmts)*
            }
        }
    }
}
