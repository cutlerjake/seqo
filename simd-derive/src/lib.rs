use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, ExprClosure, Ident, Pat, Token, Type, parse_macro_input, parse_str};

// List of supported numeric types and SIMD lengths
const SIMD_TYPES: &[&str] = &[
    "f32x4", "f32x8", "f64x2", "f64x4", "i8x16", "i8x32", "i16x8", "i16x16", "i32x4", "i32x8",
    "i64x2", "i64x4", "u8x16", "u16x8", "u16x16", "u32x4", "u32x8", "u64x2", "u64x4",
];

/// The procedural macro for implementing binary operations
#[proc_macro]
pub fn impl_binary_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ImplOpInput);

    let ImplOpInput {
        op,
        simd_op,
        scalar_op,
        exclude_types,
    } = input;

    let op = reconstruct_type(op);

    let scalar_op = scalar_op.unwrap_or_else(|| simd_op.clone());

    // Generate code for each combination of type and length
    let mut expanded = quote! {};

    for &simd_type_str in SIMD_TYPES {
        if exclude_types.iter().any(|exc| simd_type_str.contains(exc)) {
            continue;
        }
        // Parse the SIMD type and scalar type
        let simd_ty_str = format!("{simd_type_str}");

        let simd_ty = syn::parse_str::<syn::Type>(format!("wide::{simd_type_str}").as_str())
            .expect("Invalid SIMD type");

        // The scalar type is the base type before the 'x' in the SIMD type (e.g., f32, i8, etc.)
        let scalar_type_str = simd_type_str.split('x').next().unwrap();
        let scalar_ty = syn::parse_str::<syn::Type>(scalar_type_str).expect("Invalid scalar type");

        // Extract the vector length from the SIMD type string (e.g., 4 from f32x4)
        let n: usize = simd_type_str
            .split('x')
            .nth(1)
            .unwrap()
            .parse()
            .expect("Invalid SIMD length");

        // Generate the implementation for each SIMD type
        // Generate the implementation for each combination of type and length
        let mut simd_op = simd_op.clone();
        expand_closure_with_type(
            &mut simd_op,
            format!("wide::{simd_ty_str}").as_str(),
            &simd_ty_str,
        );
        let mut scalar_op = scalar_op.clone();
        expand_closure_with_type(&mut scalar_op, scalar_type_str, scalar_type_str);

        let mut op = op.clone();
        expand_op_with_type(
            &mut op,
            format!("wide::{simd_ty_str}").as_str(),
            scalar_type_str,
        );
        expanded.extend(quote! {
            impl WideBinaryOp<#n, #scalar_ty> for #op {
                #[inline(always)]
                fn simd_apply(&self, a: &[#scalar_ty; #n], b: &[#scalar_ty; #n], out: &mut [#scalar_ty; #n]) {
                    let op = #simd_op;
                    let result = op(#simd_ty::from(*a), #simd_ty::from(*b));          // Perform SIMD operation
                    *out = result.to_array(); // Convert back to array
                }

                #[inline(always)]
                fn scalar_apply(&self, a: &#scalar_ty, b: &#scalar_ty, out: &mut #scalar_ty) {
                    let op = #scalar_op;
                    *out = op(*a, *b);  // Perform the scalar operation
                }
            }
        });
    }

    // Return the generated code as TokenStream
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn impl_binary_op_mut(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ImplOpInput);

    let ImplOpInput {
        op,
        simd_op,
        scalar_op,
        exclude_types,
    } = input;

    let op = reconstruct_type(op);

    let scalar_op = scalar_op.unwrap_or_else(|| simd_op.clone());

    // Generate code for each combination of type and length
    let mut expanded = quote! {};

    for &simd_type_str in SIMD_TYPES {
        if exclude_types.iter().any(|exc| simd_type_str.contains(exc)) {
            continue;
        }
        // Parse the SIMD type and scalar type
        let simd_ty_str = format!("{simd_type_str}");

        let simd_ty = syn::parse_str::<syn::Type>(format!("wide::{simd_type_str}").as_str())
            .expect("Invalid SIMD type");

        // The scalar type is the base type before the 'x' in the SIMD type (e.g., f32, i8, etc.)
        let scalar_type_str = simd_type_str.split('x').next().unwrap();
        let scalar_ty = syn::parse_str::<syn::Type>(scalar_type_str).expect("Invalid scalar type");

        // Extract the vector length from the SIMD type string (e.g., 4 from f32x4)
        let n: usize = simd_type_str
            .split('x')
            .nth(1)
            .unwrap()
            .parse()
            .expect("Invalid SIMD length");

        // Generate the implementation for each SIMD type
        // Generate the implementation for each combination of type and length
        let mut simd_op = simd_op.clone();
        expand_closure_with_type(
            &mut simd_op,
            format!("wide::{simd_ty_str}").as_str(),
            &simd_ty_str,
        );
        let mut scalar_op = scalar_op.clone();
        expand_closure_with_type(&mut scalar_op, scalar_type_str, scalar_type_str);

        let mut op = op.clone();
        expand_op_with_type(
            &mut op,
            format!("wide::{simd_ty_str}").as_str(),
            scalar_type_str,
        );
        expanded.extend(quote! {
            impl WideBinaryOpMut<#n, #scalar_ty> for #op {
                #[inline(always)]
                fn simd_apply(&self, a: &mut [#scalar_ty; #n], b: &[#scalar_ty; #n], out: &mut [#scalar_ty; #n]) {
                    let op = #simd_op;
                    let mut _a = #simd_ty::from(*a);
                    let result = op(&mut _a, #simd_ty::from(*b)); // Perform SIMD operation
                    *out = result.to_array(); // Convert back to array
                    *a = _a.to_array(); // Update mutated value
                }

                #[inline(always)]
                fn scalar_apply(&self, a: &mut #scalar_ty, b: &#scalar_ty, out: &mut #scalar_ty) {
                    let op = #scalar_op;
                    *out = op(a, *b);  // Perform the scalar operation
                }
            }
        });
    }

    // Return the generated code as TokenStream
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn impl_unary_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ImplOpInput);

    let ImplOpInput {
        op,
        simd_op,
        scalar_op,
        exclude_types,
    } = input;

    let op = reconstruct_type(op);

    let scalar_op = scalar_op.unwrap_or_else(|| simd_op.clone());

    // Generate code for each combination of type and length
    let mut expanded = quote! {};

    for &simd_type_str in SIMD_TYPES {
        if exclude_types.iter().any(|exc| simd_type_str.contains(exc)) {
            continue;
        }
        // Parse the SIMD type and scalar type
        let simd_ty_str = format!("{simd_type_str}");

        let simd_ty = syn::parse_str::<syn::Type>(format!("wide::{simd_type_str}").as_str())
            .expect("Invalid SIMD type");

        // The scalar type is the base type before the 'x' in the SIMD type (e.g., f32, i8, etc.)
        let scalar_type_str = simd_type_str.split('x').next().unwrap();
        let scalar_ty = syn::parse_str::<syn::Type>(scalar_type_str).expect("Invalid scalar type");

        // Extract the vector length from the SIMD type string (e.g., 4 from f32x4)
        let n: usize = simd_type_str
            .split('x')
            .nth(1)
            .unwrap()
            .parse()
            .expect("Invalid SIMD length");

        // Generate the implementation for each SIMD type
        // Generate the implementation for each combination of type and length
        let mut simd_op = simd_op.clone();
        expand_closure_with_type(
            &mut simd_op,
            format!("wide::{simd_ty_str}").as_str(),
            &simd_ty_str,
        );
        let mut scalar_op = scalar_op.clone();
        expand_closure_with_type(&mut scalar_op, scalar_type_str, scalar_type_str);

        let mut op = op.clone();
        expand_op_with_type(
            &mut op,
            format!("wide::{simd_ty_str}").as_str(),
            scalar_type_str,
        );
        expanded.extend(quote! {
            impl WideUnaryOp<#n, #scalar_ty> for #op {
                #[inline(always)]
                fn simd_apply(&self, a: &[#scalar_ty; #n], out: &mut [#scalar_ty; #n]) {
                    let op = #simd_op;
                    let result = op(#simd_ty::from(*a)); // Perform SIMD operation
                    *out = result.to_array(); // Convert back to array
                }

                #[inline(always)]
                fn scalar_apply(&self, a: & #scalar_ty, out: &mut #scalar_ty) {
                    let op = #scalar_op;
                    *out = op(*a);  // Perform the scalar operation
                }
            }
        });
    }

    // Return the generated code as TokenStream
    TokenStream::from(expanded)
}

fn expand_closure_with_type(
    closure: &mut ExprClosure,
    full_replacement: &str,
    short_replacement: &str,
) {
    // Convert the closure body to a string
    let body_str = closure.body.to_token_stream().to_string();

    // Replace all occurrences of `__T__` with the provided replacement string
    let modified_str = body_str
        .replace("__T__", short_replacement)
        .replace("__FT__", full_replacement);

    // Parse the modified string back into a closure body
    let modified_body: Expr = match parse_str(&modified_str) {
        Ok(expr) => expr,
        Err(e) => panic!("{:?}", e),
    };

    // Convert inputs to a string
    for pat in closure.inputs.iter_mut() {
        match pat {
            Pat::Type(pat) => {
                let pat_s = pat.ty.to_token_stream().to_string();
                let modified_str = pat_s
                    .replace("__T__", short_replacement)
                    .replace("__FT__", full_replacement);

                let modified_ident: Type = match parse_str(&modified_str) {
                    Ok(expr) => expr,
                    Err(e) => panic!("{:?}", e),
                };

                pat.ty = Box::new(modified_ident);
            }
            _ => (),
        }
    }

    // Replace the old body with the new one
    closure.body = Box::new(modified_body);
}

fn expand_op_with_type(
    op: &mut proc_macro2::TokenStream,
    wide_replacement: &str,
    scalar_replacement: &str,
) {
    // Convert the op to a string.
    let op_str = op.to_string();

    // Replace all occurrences of `__T__` with the provided replacement string
    let modified_str =
        op_str
            .replacen("__T__", wide_replacement, 1)
            .replacen("__T__", scalar_replacement, 1);

    // Parse the modified string back into a closure body
    let modified_ident: proc_macro2::TokenStream = match parse_str(&modified_str) {
        Ok(ident) => ident,
        Err(e) => panic!("{:?}", e),
    };

    *op = modified_ident;
}

// Struct for parsing the input
struct ImplOpInput {
    op: (Ident, Vec<Ident>), // op holds a first generic and a vector of additional generics
    simd_op: ExprClosure,
    scalar_op: Option<ExprClosure>,
    exclude_types: Vec<String>,
}

impl Parse for ImplOpInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let op: Ident = input.parse()?; // Parse the first identifier (generic)

        // Optionally parse the generics enclosed in `< >`
        let op = if input.peek(Token![<]) {
            input.parse::<Token![<]>()?;

            let mut generics = vec![];
            while !input.peek(Token![>]) {
                let generic: Ident = input.parse()?;
                generics.push(generic);

                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }

            input.parse::<Token![>]>()?; // Closing `>`

            (op, generics)
        } else {
            (op, Vec::new()) // No generics, just return the first `Ident`
        };

        input.parse::<Token![,]>()?;
        let simd_op: ExprClosure = input.parse()?;

        let scalar_op: Option<ExprClosure> = if input.is_empty() {
            None
        } else {
            let _ = input.parse::<Token![,]>();
            input.parse().ok()
        };

        let exclude_types = if input.is_empty() {
            Vec::new()
        } else {
            // Parse the exclusion list (e.g., `exclude_types = ["f32x4", "i8x16"]`)
            let _ = input.parse::<Token![,]>();
            let exclude_types: syn::ExprArray = input.parse()?;
            exclude_types
                .elems
                .into_iter()
                .map(|elem| match elem {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }) => lit_str.value(),
                    _ => panic!("Expected string literal for excluded types"),
                })
                .collect()
        };

        Ok(ImplOpInput {
            op,
            simd_op,
            scalar_op,
            exclude_types,
        })
    }
}

fn reconstruct_type(op: (Ident, Vec<Ident>)) -> proc_macro2::TokenStream {
    let (op_ident, generics) = op;

    // If no generics are provided, just return the base identifier.
    if generics.is_empty() {
        quote! { #op_ident }
    } else {
        // Otherwise, reconstruct the generics part.
        let generics = generics.iter().map(|g| quote! { #g });

        // Reconstruct the full type `op::<T, U, V>`
        quote! { #op_ident::<#(#generics),*> }
    }
}
