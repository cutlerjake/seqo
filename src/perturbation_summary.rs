//! Perturbation summary trait accumulating block perturbation effects.

use std::ops::AddAssign;

use crate::prelude::GenericStateModifier;

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

/// A summary of a single period.
pub trait SinglePeriodSummary {
    type Ctx;
    type Block;

    fn add_block(&mut self, block: &Self::Block, ctx: &Self::Ctx);
    fn sub_block(&mut self, block: &Self::Block, ctx: &Self::Ctx);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrayState<const N: usize, T> {
    pub periods: [T; N],
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VecState<T> {
    pub periods: Vec<T>,
}

impl<const N: usize, T, U> GenericStateModifier<U> for ArrayState<N, T>
where
    U: for<'a> AddAssign<&'a ArrayState<N, T>>,
{
    fn add_delta_to_state(&self, state: &mut U) {
        *state += self;
    }
}

impl<T, U> GenericStateModifier<U> for VecState<T>
where
    U: for<'a> AddAssign<&'a VecState<T>>,
{
    fn add_delta_to_state(&self, state: &mut U) {
        *state += self;
    }
}

impl<const N: usize, T: for<'a> AddAssign<&'a T>> AddAssign<&ArrayState<N, T>>
    for ArrayState<N, T>
{
    fn add_assign(&mut self, rhs: &ArrayState<N, T>) {
        for (a, b) in self.periods.iter_mut().zip(rhs.periods.iter()) {
            *a += b;
        }
    }
}

impl<T: for<'a> AddAssign<&'a T>> AddAssign<&VecState<T>> for VecState<T> {
    fn add_assign(&mut self, rhs: &VecState<T>) {
        for (a, b) in self.periods.iter_mut().zip(rhs.periods.iter()) {
            *a += b;
        }
    }
}

impl<const N: usize, T> PerturbationSummary for ArrayState<N, T>
where
    T: SinglePeriodSummary + Default + Send + Sync + Copy,
    <T as SinglePeriodSummary>::Ctx: Send + Sync,
{
    type Block = T::Block;
    type Context = T::Ctx;

    fn new_empty(period: u8) -> Self {
        assert_eq!(period as usize, N);
        Self {
            periods: [T::default(); N],
        }
    }

    fn add_block(&mut self, period: u8, block: &Self::Block, context: &Self::Context) {
        self.periods[period as usize].add_block(block, context);
    }

    fn sub_block(&mut self, period: u8, block: &Self::Block, context: &Self::Context) {
        self.periods[period as usize].sub_block(block, context);
    }

    fn reset(&mut self) {
        for period_summary in self.periods.iter_mut() {
            *period_summary = T::default();
        }
    }
}

impl<T> PerturbationSummary for VecState<T>
where
    T: SinglePeriodSummary + Default + Send + Sync + Clone,
    <T as SinglePeriodSummary>::Ctx: Send + Sync,
{
    type Block = T::Block;
    type Context = T::Ctx;

    fn new_empty(period: u8) -> Self {
        Self {
            periods: vec![T::default(); period as usize],
        }
    }

    fn add_block(&mut self, period: u8, block: &Self::Block, context: &Self::Context) {
        self.periods[period as usize].add_block(block, context);
    }

    fn sub_block(&mut self, period: u8, block: &Self::Block, context: &Self::Context) {
        self.periods[period as usize].sub_block(block, context);
    }

    fn reset(&mut self) {
        for period_summary in self.periods.iter_mut() {
            *period_summary = T::default();
        }
    }
}
