use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, Fields, Ident, Type};

pub fn expand_block_delta(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;

    let block_ty = parse_struct_block_attr(&input.attrs)?;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "BlockDelta only supports structs with named fields",
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

    // Collect (field_ident, Expr) for each #[block_delta(...)] field
    let mut field_exprs = Vec::new();
    for field in fields {
        let name = field.ident.as_ref().unwrap();
        if let Some(expr) = parse_field_block_delta(&field.attrs)? {
            field_exprs.push((name.clone(), expr));
        }
    }

    Ok(generate_impl(struct_name, &block_ty, &field_exprs))
}

// e.g. #[block_delta(block = SomeBlockType)]
fn parse_struct_block_attr(attrs: &[Attribute]) -> syn::Result<Type> {
    for attr in attrs {
        if attr.path().is_ident("block_delta") {
            // The tokens are everything inside the parentheses
            // e.g.: block = GoldBlock
            let tokens = &attr.meta.require_list()?.tokens;

            // We parse it as an assignment: left = right
            let assign: syn::ExprAssign = syn::parse2(tokens.clone())?;
            if let syn::Expr::Path(left_path) = *assign.left.clone()
                && left_path.path.is_ident("block")
            {
                // The right side must be a path (the block type)
                if let syn::Expr::Path(right_path) = *assign.right {
                    let ty = Type::Path(syn::TypePath {
                        qself: None,
                        path: right_path.path,
                    });
                    return Ok(ty);
                }
            }
        }
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "Expected #[block_delta(block = Type)]",
    ))
}

// e.g. #[block_delta(if block.grade() >= CUTOFF { ... } else { ... })]
fn parse_field_block_delta(attrs: &[Attribute]) -> syn::Result<Option<Expr>> {
    for attr in attrs {
        if attr.path().is_ident("block_delta") {
            let tokens = &attr.meta.require_list()?.tokens;
            // here we parse tokens directly as an expression
            // because the user provided a single expr in the attribute.
            let expr = syn::parse2::<Expr>(tokens.clone())?;
            return Ok(Some(expr));
        }
    }
    Ok(None)
}

fn generate_impl(
    struct_name: &Ident,
    block_ty: &Type,
    field_exprs: &[(Ident, Expr)],
) -> proc_macro2::TokenStream {
    let add_stmts = field_exprs.iter().map(|(name, expr)| {
        quote! {
            self.#name += #expr;
        }
    });

    let sub_stmts = field_exprs.iter().map(|(name, expr)| {
        quote! {
            self.#name -= #expr;
        }
    });

    quote! {
        impl #struct_name {
            #[inline(always)]
            pub fn add_block(&mut self, block: &#block_ty) {
                #(#add_stmts)*
            }

            #[inline(always)]
            pub fn sub_block(&mut self, block: &#block_ty) {
                #(#sub_stmts)*
            }
        }
    }
}
