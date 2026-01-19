//! Block perturbations applied in the scheduling process.
//!
//! [`BlockPerturbation`] represents a perturbation applied to a specific block,
//! tracking the change in state if the block (and its related blocks) were to
//! be moved to a different period.

use crate::{
    perturbation_summary::PerturbationSummary, relation_provider::RelationRangeProvider,
    walkers::FixDir,
};

/// The type of perturbation being applied to a block's scheduling period.
#[derive(Clone, Copy, Debug)]
pub enum PertType {
    /// Scheduling a block to an earlier period.
    DecreasePeriod,
    /// Scheduling a block to a later period.
    IncreasePeriod,
    /// No change in the scheduling period.
    NoOp,
}

impl PertType {
    /// Create a new `PertType` based on the current and new periods.
    #[inline(always)]
    pub fn new(curr_period: u8, new_period: u8) -> Self {
        match new_period.cmp(&curr_period) {
            std::cmp::Ordering::Less => PertType::DecreasePeriod,
            std::cmp::Ordering::Equal => PertType::NoOp,
            std::cmp::Ordering::Greater => PertType::IncreasePeriod,
        }
    }
}

/// Represents a perturbation applied to a specific block in the scheduling process.
///
/// Tracks the change in the state if the block were to be moved to a different period,
/// along with the number of affected blocks.
#[derive(Clone, Debug)]
pub struct BlockPerturbation<T> {
    block_index: u32,
    period: u8,
    pub(crate) delta_summary: T,
    pub(crate) block_cnt: usize,
}

impl<T: PerturbationSummary> BlockPerturbation<T> {
    /// Create a new `BlockPerturbation` for the specified block index and period.
    ///
    /// # Arguments
    /// * `block_index` - The index of the block being perturbed.
    /// * `period` - The target period for the perturbation.
    /// * `mine_life` - The mining life parameter for initializing the delta summary.
    #[inline(always)]
    pub fn new(block_index: u32, period: u8, mine_life: u8) -> Self {
        Self {
            block_index,
            period,
            delta_summary: T::new_empty(mine_life),
            block_cnt: 0,
        }
    }

    /// Get the number of blocks affected by this perturbation.
    #[inline(always)]
    pub fn block_cnt(&self) -> usize {
        self.block_cnt
    }

    /// Get the index of the block being perturbed.
    #[inline(always)]
    pub fn block_index(&self) -> u32 {
        self.block_index
    }

    /// Get the target period for the perturbation.
    #[inline(always)]
    pub fn period(&self) -> u8 {
        self.period
    }

    /// Get a reference to the delta summary of the perturbation.
    #[inline(always)]
    pub fn delta_summary(&self) -> &T {
        &self.delta_summary
    }
    /// Add a block to the perturbation summary.
    ///
    /// # Arguments
    /// * `period` - The period of the block being added.
    /// * `block` - A reference to the block being added.
    /// * `context` - The context for the perturbation summary update.
    #[inline(always)]
    pub fn add_block(&mut self, period: u8, block: &T::Block, context: &T::Context) {
        self.delta_summary.add_block(period, block, context);
    }

    /// Subtract a block from the perturbation summary.
    ///
    /// # Arguments
    /// * `period` - The period of the block being subtracted.
    /// * `block` - A reference to the block being subtracted.
    /// * `context` - The context for the perturbation summary update.
    #[inline(always)]
    #[track_caller]
    pub fn sub_block(&mut self, period: u8, block: &T::Block, context: &T::Context) {
        self.delta_summary.sub_block(period, block, context);
    }

    /// Walk through all blocks affected by this perturbation, applying the provided function.
    ///
    /// # Arguments
    /// * `sched` - A function that returns the period of a block given its index.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `func` - A mutable function to apply to each affected block and its period.
    #[inline(always)]
    pub fn walk_effected_blocks(
        &self,
        sched: impl Fn(u32) -> u8,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        mut func: impl FnMut(u32, u8),
    ) {
        let fix_dir = FixDir::new(sched(self.block_index), self.period);

        let relations = match fix_dir {
            FixDir::Preds(_) => flat_preds.relations(self.block_index() as usize),
            FixDir::Succs(_) => flat_succs.relations(self.block_index() as usize),
        };

        for dep in relations {
            let period = sched(dep);
            if !fix_dir.is_valid(period) {
                func(dep, period);
            }
        }
    }

    /// Get an iterator over all blocks affected by this perturbation.
    ///
    /// # Arguments
    /// * `sched` - A function that returns the period of a block given its index
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    #[inline(always)]
    pub fn effected_it<'a>(
        &'a self,
        sched: impl Fn(u32) -> u8 + 'a,
        flat_preds: &'a RelationRangeProvider,
        flat_succs: &'a RelationRangeProvider,
    ) -> impl Iterator<Item = (u32, u8)> + 'a {
        let fix_dir = FixDir::new(sched(self.block_index), self.period);

        let relations = match fix_dir {
            FixDir::Preds(_) => flat_preds.relations(self.block_index() as usize),
            FixDir::Succs(_) => flat_succs.relations(self.block_index() as usize),
        };

        relations.filter_map(move |dep| {
            let period = sched(dep);
            if !fix_dir.is_valid(period) {
                Some((dep, period))
            } else {
                None
            }
        })
    }

    /// Compute the perturbation summary from scratch based on the current scheduling.
    ///
    /// # Arguments
    /// * `blocks` - A slice of all blocks in the scheduling.
    /// * `sched` - A function that returns the period of a block given its index
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `context` - The context for the perturbation summary computation.
    #[inline(always)]
    pub fn compute_from_scratch(
        &mut self,
        blocks: &[T::Block],
        sched: impl Fn(u32) -> u8,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        context: &T::Context,
    ) {
        // 1. Reset the perturbation.
        self.delta_summary.reset();
        self.block_cnt = 0;

        // 2. Get the period of the perturbation target block.
        let current_period = sched(self.block_index());

        // 3. Build the walker to fix any violation that would result from applying this perturbation.
        let fix_dir = FixDir::new(current_period, self.period());

        let relations = match fix_dir {
            FixDir::Preds(_) => flat_preds.relations(self.block_index() as usize),
            FixDir::Succs(_) => flat_succs.relations(self.block_index() as usize),
        };

        for dep in relations {
            let from_period = sched(dep);

            if !fix_dir.is_valid(from_period) {
                let block = &blocks[dep as usize];
                self.sub_block(from_period, block, context);
                self.add_block(self.period(), block, context);
                self.block_cnt += 1;
            }
        }
    }

    /// Update the perturbation summary based on changes in scheduling.
    ///
    /// This method update the perturbation summary to be accurate for the current scheduling,
    /// taking into account changes from a previous scheduling. This method is more efficient than
    /// recomputing the summary from scratch.
    ///
    /// # Arguments
    /// * `blocks` - A slice of all blocks in the scheduling.
    /// * `curr_sched` - A function that returns the current period of a block given its index.
    /// * `prev_sched` - A function that returns the previous period of a block given its index.
    /// * `preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `context` - The context for the perturbation summary update.
    #[inline(always)]
    pub fn update(
        &mut self,
        blocks: &[T::Block],
        curr_sched: impl Fn(u32) -> u8,
        prev_sched: impl Fn(u32) -> u8,
        preds: &RelationRangeProvider,
        succs: &RelationRangeProvider,
        context: &T::Context,
    ) {
        // 2. Get the period of the perturbation target block.
        let current_period = curr_sched(self.block_index());
        let prev_period = prev_sched(self.block_index());

        // 3. Build the walker to fix any violation that would result from applying this perturbation.
        let curr_fix_dir = FixDir::new(current_period, self.period());
        let prev_fix_dir = FixDir::new(prev_period, self.period());

        match (curr_fix_dir, prev_fix_dir) {
            (FixDir::Preds(_), FixDir::Preds(_)) | (FixDir::Succs(_), FixDir::Succs(_)) => {
                let relations = match curr_fix_dir {
                    FixDir::Preds(_) => preds.relations(self.block_index() as usize),
                    FixDir::Succs(_) => succs.relations(self.block_index() as usize),
                };

                for dep in relations {
                    let curr_from_period = curr_sched(dep);
                    let prev_from_period = prev_sched(dep);

                    if curr_from_period == prev_from_period {
                        continue;
                    }

                    if !prev_fix_dir.is_valid(prev_from_period) {
                        let block = &blocks[dep as usize];
                        self.add_block(prev_from_period, block, context);
                        self.sub_block(self.period(), block, context);
                        self.block_cnt -= 1;
                    }

                    if !curr_fix_dir.is_valid(curr_from_period) {
                        let block = &blocks[dep as usize];
                        self.sub_block(curr_from_period, block, context);
                        self.add_block(self.period(), block, context);
                        self.block_cnt += 1;
                    }
                }
            }
            _ => {
                // 1. Reset the perturbation.
                self.delta_summary.reset();
                self.block_cnt = 0;

                let relations = match curr_fix_dir {
                    FixDir::Preds(_) => preds.relations(self.block_index() as usize),
                    FixDir::Succs(_) => succs.relations(self.block_index() as usize),
                };
                // assert!(relations[0] == self.block_index());

                for dep in relations {
                    let from_period = curr_sched(dep);

                    if !curr_fix_dir.is_valid(from_period) {
                        let block = &blocks[dep as usize];
                        self.sub_block(from_period, block, context);
                        self.add_block(self.period(), block, context);
                        self.block_cnt += 1;
                    }
                }
            }
        }
    }
}
