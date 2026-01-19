//! Representation of a mine with blocks and their precedence relations.
use crate::relation_provider::RelationProvider;

/// Trait representing a mine with blocks and their precedence relations.
pub trait Mine {
    /// The type of block in the mine.
    type Block;

    /// Returns a reference to the block with the given index.
    fn get_block(&self, block: u32) -> &Self::Block;

    /// Returns a slice of all blocks in the mine.
    fn all_blocks(&self) -> &[Self::Block];

    /// The number of blocks in the mine.
    fn num_blocks(&self) -> usize {
        self.all_blocks().len()
    }

    /// Returns the predecessor relation provider.
    fn get_pred_provider(&self) -> &RelationProvider;

    /// Returns the successor relation provider.
    fn get_succ_provider(&self) -> &RelationProvider;

    /// Returns the predecessors of the given block.
    fn get_preds(&self, block: u32) -> &[u32];

    /// Returns the successors of the given block.
    fn get_succs(&self, block: u32) -> &[u32];

    /// Returns a reference to the block with the given index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `block` is a valid index.
    unsafe fn get_block_unchecked(&self, block: u32) -> &Self::Block;

    /// Returns the predecessors of the given block without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `block` is a valid index.
    unsafe fn get_preds_unchecked(&self, block: u32) -> &[u32];

    /// Returns the successors of the given block without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `block` is a valid index.
    unsafe fn get_succs_unchecked(&self, block: u32) -> &[u32];
}

/// A default implementation of the `Mine` trait using vectors.
pub struct DefaultMine<T> {
    blocks: Vec<T>,
    preds: RelationProvider,
    succs: RelationProvider,
}

impl<T> DefaultMine<T> {
    /// Creates a new `DefaultMine` instance.
    ///
    /// # Arguments
    /// * `blocks` - A vector of blocks in the mine.
    /// * `preds` - A relation provider for predecessor relations.
    /// * `succs` - A relation provider for successor relations.
    pub fn new(blocks: Vec<T>, preds: RelationProvider, succs: RelationProvider) -> Self {
        Self {
            blocks,
            preds,
            succs,
        }
    }
}

impl<T> Mine for DefaultMine<T> {
    type Block = T;

    #[inline(always)]
    fn all_blocks(&self) -> &[Self::Block] {
        &self.blocks
    }
    #[inline(always)]
    fn get_block(&self, block: u32) -> &Self::Block {
        &self.blocks[block as usize]
    }

    #[inline(always)]
    fn get_pred_provider(&self) -> &RelationProvider {
        &self.preds
    }

    #[inline(always)]
    fn get_preds(&self, block: u32) -> &[u32] {
        self.preds.relations(block as usize)
    }

    #[inline(always)]
    fn get_succ_provider(&self) -> &RelationProvider {
        &self.succs
    }

    #[inline(always)]
    fn get_succs(&self, block: u32) -> &[u32] {
        self.succs.relations(block as usize)
    }

    #[inline(always)]
    unsafe fn get_block_unchecked(&self, block: u32) -> &Self::Block {
        unsafe { self.blocks.get_unchecked(block as usize) }
    }

    #[inline(always)]
    unsafe fn get_preds_unchecked(&self, block: u32) -> &[u32] {
        unsafe { self.preds.relations_unchecked(block as usize) }
    }

    #[inline(always)]
    unsafe fn get_succs_unchecked(&self, block: u32) -> &[u32] {
        unsafe { self.succs.relations_unchecked(block as usize) }
    }
}
