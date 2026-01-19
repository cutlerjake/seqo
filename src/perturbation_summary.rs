//! Perturbation summary trait accumulating block perturbation effects.

/// Perturbation summary trait for accumulating the effects of block perturbations.
pub trait PerturbationSummary {
    /// The type of block being summarized.
    type Block;
    /// The context needed for summarization.
    ///
    /// This could be any additional data required to compute the summary (cut-off grades to define ore/waste, etc.).
    type Context: Send + Sync;

    /// Create a new empty perturbation summary for the given period.
    fn new_empty(period: u8) -> Self;

    /// Add a block to the perturbation summary.
    fn add_block(&mut self, period: u8, block: &Self::Block, context: &Self::Context);

    /// Subtract a block from the perturbation summary.
    fn sub_block(&mut self, period: u8, block: &Self::Block, context: &Self::Context);

    /// Reset the perturbation summary to its initial empty state.
    fn reset(&mut self);
}

/// A no-op implementation of `PerturbationSummary` for the unit type `()`.
impl PerturbationSummary for () {
    type Block = ();
    type Context = ();

    #[inline(always)]
    fn new_empty(_period: u8) -> Self {}

    #[inline(always)]
    fn add_block(&mut self, _period: u8, _block: &Self::Block, _context: &Self::Context) {}

    #[inline(always)]
    fn sub_block(&mut self, _period: u8, _block: &Self::Block, _context: &Self::Context) {}

    #[inline(always)]
    fn reset(&mut self) {}
}
