use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};
mod add_assign;
mod expand_block_delta;
mod state_logger;

use add_assign::derive_add_assign;
use expand_block_delta::expand_block_delta;

/// Implements `add_block` and `sub_block` methods for the annotated struct.
///
/// Reduces duplicate logic between `add_block` and `sub_block` by allowing
/// users to specify expressions for each field that determine how much to
/// add or subtract based on the provided block.
///
/// ## Example
/// ```rust
/// use seqo_derive::BlockDelta;
///
/// #[derive(Debug, Copy, Clone)]
/// struct GoldBlock {
///     idx: [i32; 3],
///     gold_grams: f64,
///     tonnage: f64,
/// }
/// impl GoldBlock {
///     fn grade(&self) -> f64 {
///         self.gold_grams / self.tonnage
///     }
/// }
///
/// struct Ctx {
///    cutoff_grade: f64,
/// }
///
/// #[derive(Debug, Copy, Clone, Default, BlockDelta)]
/// #[block_delta(block = GoldBlock, context = Ctx)]
/// struct GoldAggregateSummary {
///     #[block_delta(if block.grade() >= ctx.cutoff_grade { block.gold_grams } else { 0.0 })]
///     total_gold_grams_to_mill: f64,
///     #[block_delta(if block.grade() >= ctx.cutoff_grade { block.tonnage } else { 0.0 })]
///     total_tonnage_to_mill: f64,
///     #[block_delta(if block.grade() < ctx.cutoff_grade { block.tonnage } else { 0.0 })]
///     waste_tonnage: f64,
/// }
/// ```
///
/// Expands to:
/// ```rust
///  impl GoldAggregateSummary {
///     #[inline(always)]
///     pub fn add_block(&mut self, block: &GoldBlock, ctx: &Ctx) {
///         self.total_gold_grams_to_mill
///             += if block.grade() >= ctx.cutoff_grade { block.gold_grams } else { 0.0 };
///         self.total_tonnage_to_mill
///             += if block.grade() >= ctx.cutoff_grade { block.tonnage } else { 0.0 };
///         self.waste_tonnage += if block.grade() < ctx.cutoff_grade { block.tonnage } else { 0.0 };
///     }
///     #[inline(always)]
///     pub fn sub_block(&mut self, block: &GoldBlock, ctx: &Ctx) {
///         self.total_gold_grams_to_mill
///             -= if block.grade() >= ctx.cutoff_grade { block.gold_grams } else { 0.0 };
///         self.total_tonnage_to_mill
///             -= if block.grade() >= ctx.cutoff_grade { block.tonnage } else { 0.0 };
///         self.waste_tonnage -= if block.grade() < ctx.cutoff_grade { block.tonnage } else { 0.0 };
///     }
/// }
/// ```
#[proc_macro_derive(BlockDelta, attributes(block_delta))]
pub fn derive_block_delta(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_block_delta(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_derive(AddAssign)]
pub fn derive_add_assign_macro(input: TokenStream) -> TokenStream {
    derive_add_assign(input)
}

#[proc_macro_attribute]
pub fn seqo_log(attr: TokenStream, item: TokenStream) -> TokenStream {
    state_logger::seqo_log(attr, item)
}
