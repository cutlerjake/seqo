use core::panic;

use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Group, TokenStream as TS2, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{
    Block, Expr, Ident, Meta, Stmt, Type,
    parse::Parser,
    visit_mut::{self, VisitMut},
};

pub fn seqo_log(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let ts: proc_macro2::TokenStream = item.into();

    // Strip all `slog!(...)` calls and collect info.
    let (cleaned_ts, logs, markers) = strip_and_collect_logs(ts);

    let Ok(mut impl_block): Result<syn::ItemImpl, _> = syn::parse2(cleaned_ts.clone()) else {
        return cleaned_ts.into();
    };

    // Find state_value method
    let method = impl_block
        .items
        .iter_mut()
        .find_map(|it| match it {
            syn::ImplItem::Fn(f) if f.sig.ident == "state_value" => Some(f),
            _ => None,
        })
        .expect("seqo_log requires `state_value`");

    let logs = dedupe_logged(
        logs.into_iter()
            .map(|entry| (entry.ident, entry.ty))
            .collect(),
    );

    // ---- Generate struct fields ----
    let fields = logs.iter().map(|(ident, ty)| {
        quote! { #ident: Vec<#ty> }
    });

    let mut typed_inputs = method.sig.inputs.iter().filter_map(|arg| match arg {
        syn::FnArg::Typed(pt) => Some(pt.clone()),
        syn::FnArg::Receiver(_) => None,
    });

    let state_arg = typed_inputs
        .next()
        .expect("seqo_log requires `state_value(&self, state, workspace)`");
    let _workspace_arg = typed_inputs
        .next()
        .expect("seqo_log requires `state_value(&self, state, workspace)`");

    // ---- Clone function body for StateSummary::new ----
    let mut new_body = method.block.clone();
    strip_returns(&mut new_body);

    // Remove internal markers from the original user method body.
    strip_internal_markers(&mut method.block);

    // rename self -> of
    SelfToOf.visit_block_mut(&mut new_body);

    // inject log pushes where `slog!(...)` appeared
    let mut injector = PushInjector {
        markers,
        next_marker_idx: 0,
    };
    injector.visit_block_mut(&mut new_body);

    let self_ty = impl_block.self_ty.clone();

    let workspace_param = if let Some((_, trait_path, _)) = &impl_block.trait_ {
        quote! { _workspace: &mut <#self_ty as #trait_path>::WorkSpace }
    } else {
        quote! { #_workspace_arg }
    };

    let output = quote! {
        #impl_block

        #[derive(Debug, Clone, Default)]
        pub struct StateSummary {
            #(#fields,)*
        }

        impl StateSummary {
            pub fn new(
                of: &#self_ty,
                #state_arg,
                #workspace_param,
            ) -> Self {
                macro_rules! __seqo_log_marker {
                    () => {};
                    ($id:ident) => {};
                }

                let mut log = Self::default();

                #new_body

                log
            }
        }
    };

    output.into()
}

fn dedupe_logged(logged: Vec<(Ident, Type)>) -> Vec<(Ident, Type)> {
    let mut out = Vec::new();
    for (ident, ty) in logged {
        if !out.iter().any(|(existing, _)| existing == &ident) {
            out.push((ident, ty));
        }
    }
    out
}

pub struct LogEntry {
    pub ident: Ident,
    pub ty: Type,
}

#[derive(Clone)]
pub struct LogMarker {
    pub field: Ident,
    pub value: Expr,
}

struct ParsedLogCall {
    ty: Type,
    field: Option<Ident>,
    expr: Expr,
}

pub fn strip_and_collect_logs(input: TS2) -> (TS2, Vec<LogEntry>, Vec<LogMarker>) {
    let mut out = TS2::new();
    let mut logs = Vec::new();
    let mut markers = Vec::new();
    let mut iter = input.into_iter().peekable();

    while let Some(tt) = iter.next() {
        if let TokenTree::Ident(id) = &tt {
            if id == "slog" {
                let mut lookahead = iter.clone();
                if let Some(TokenTree::Punct(bang)) = lookahead.next() {
                    if bang.as_char() == '!' {
                        if let Some(TokenTree::Group(args)) = lookahead.next() {
                            if args.delimiter() == Delimiter::Parenthesis {
                                // Consume ! and (..)
                                let _ = iter.next();
                                let _ = iter.next();

                                let parsed = parse_log_call(args.stream());
                                let value_expr = parsed.expr;

                                let field_ident = parsed.field.unwrap_or_else(|| {
                                    ident_from_expr(&value_expr).unwrap_or_else(|| {
                                        panic!(
                                            "slog!(expr = ...) requires `field = name` when expr is not a simple identifier"
                                        )
                                    })
                                });

                                logs.push(LogEntry {
                                    ident: field_ident.clone(),
                                    ty: parsed.ty,
                                });

                                markers.push(LogMarker {
                                    field: field_ident,
                                    value: value_expr,
                                });

                                out.extend(quote! {
                                    __seqo_log_marker!();
                                });

                                // Optional trailing ';'
                                if let Some(TokenTree::Punct(p)) = iter.peek() {
                                    if p.as_char() == ';' {
                                        let _ = iter.next();
                                    }
                                }

                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Recurse into groups
        if let TokenTree::Group(g) = tt {
            let (new_stream, mut inner_logs, mut inner_markers) =
                strip_and_collect_logs(g.stream());
            logs.append(&mut inner_logs);
            markers.append(&mut inner_markers);
            out.extend([TokenTree::Group(Group::new(g.delimiter(), new_stream))]);
        } else {
            out.extend([tt]);
        }
    }

    (out, logs, markers)
}

fn ident_from_expr(expr: &Expr) -> Option<Ident> {
    match expr {
        Expr::Path(path) if path.path.segments.len() == 1 => {
            Some(path.path.segments[0].ident.clone())
        }
        _ => None,
    }
}
fn parse_log_call(tokens: TS2) -> ParsedLogCall {
    let parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    let metas = parser.parse2(tokens).expect("invalid slog!(...)");

    let mut ty: Option<Type> = None;
    let mut field: Option<Ident> = None;
    let mut expr: Option<Expr> = None;

    for meta in metas {
        if let Meta::NameValue(nv) = meta {
            if nv.path.is_ident("ty") {
                ty = Some(
                    syn::parse2::<Type>(nv.value.into_token_stream())
                        .expect("invalid type in slog!(ty=...)"),
                );
            } else if nv.path.is_ident("field") {
                field = ident_from_expr(&nv.value)
                    .or_else(|| syn::parse2::<Ident>(nv.value.into_token_stream()).ok())
                    .or_else(|| panic!("invalid field in slog!(field = ...), expected identifier"));
            } else if nv.path.is_ident("expr") {
                expr = Some(nv.value);
            }
        }
    }

    ParsedLogCall {
        ty: ty.unwrap_or_else(|| panic!("slog!(...) requires ty = Type")),
        field,
        expr: expr.unwrap_or_else(|| panic!("slog!(...) requires expr = ...")),
    }
}

struct PushInjector {
    markers: Vec<LogMarker>,
    next_marker_idx: usize,
}

fn is_marker(stmt: &Stmt) -> bool {
    if let Stmt::Macro(stmt_macro) = stmt {
        if stmt_macro.mac.path.is_ident("__seqo_log_marker") {
            return true;
        }
    }
    false
}

impl VisitMut for PushInjector {
    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        let mut new_stmts = Vec::new();

        for mut stmt in block.stmts.clone() {
            visit_mut::visit_stmt_mut(self, &mut stmt);

            if is_marker(&stmt) {
                let marker = self
                    .markers
                    .get(self.next_marker_idx)
                    .cloned()
                    .unwrap_or_else(|| panic!("internal seqo_log marker mismatch"));
                self.next_marker_idx += 1;

                let field = marker.field;
                let value = marker.value;
                new_stmts.push(syn::parse_quote! {
                    log.#field.push(#value);
                });
                continue;
            }

            new_stmts.push(stmt);
        }

        block.stmts = new_stmts;
    }
}

fn strip_internal_markers(block: &mut Block) {
    MarkerStripper.visit_block_mut(block);
}

struct MarkerStripper;

impl VisitMut for MarkerStripper {
    fn visit_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.visit_stmt_mut(stmt);
        }

        block.stmts.retain(|stmt| match stmt {
            Stmt::Macro(stmt_macro) => !stmt_macro.mac.path.is_ident("__seqo_log_marker"),
            Stmt::Expr(Expr::Macro(expr_macro), _) => {
                !expr_macro.mac.path.is_ident("__seqo_log_marker")
            }
            _ => true,
        });
    }
}

struct SelfToOf;

impl VisitMut for SelfToOf {
    fn visit_expr_path_mut(&mut self, path: &mut syn::ExprPath) {
        if path.path.is_ident("self") {
            path.path = format_ident!("of").into();
        }
    }
}

pub fn strip_returns(block: &mut Block) {
    // Remove explicit top-level `return ...;`.
    block.stmts.retain(|stmt| match stmt {
        Stmt::Expr(Expr::Return(_), _) => false,
        _ => true,
    });

    // Remove implicit top-level tail return (last stmt without semicolon).
    if let Some(last) = block.stmts.last() {
        if matches!(last, Stmt::Expr(_, None)) {
            block.stmts.pop();
        }
    }
}
