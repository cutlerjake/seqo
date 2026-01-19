//! Aggregate perturbations for scheduling.
//!
//! An `AggregatePerturbation` is a collection of unit block perturbations
//! that are applied together as a single perturbation. It maintains a summary
//! of the overall changes induced by its unit perturbations.

use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    block_perturbation::BlockPerturbation, perturbation_summary::PerturbationSummary,
    relation_provider::RelationRangeProvider,
};

/// A collection of unit block perturbations aggregated together, and applied as a single perturbation.
///
/// The aggregate perturbation maintains a summary of the overall changes induced by its unit perturbations.
pub struct AggregatePerturbation<T> {
    unit_perts: Vec<BlockPerturbation<()>>,
    pert_set: FxHashSet<(u32, u8)>,
    delta_summary: T,
    pub(crate) block_cnt: usize,
}

impl<T: PerturbationSummary> AggregatePerturbation<T> {
    /// Creates a new aggregate perturbation from a list of unit block perturbations.
    ///
    /// # Arguments
    /// * `unit_perts` - A vector of unit block perturbations to be aggregated
    ///   together. The inner type is `()` since no data is stored
    ///   in the unit perturbations for aggregate perturbations.
    /// * `mine_life` - The maximum number of periods in the schedule, used to
    ///   initialize the perturbation summary.
    pub fn new(unit_perts: Vec<BlockPerturbation<()>>, mine_life: u8) -> Self {
        let pert_set = unit_perts
            .iter()
            .map(|p| (p.block_index(), p.period()))
            .collect();
        Self {
            unit_perts,
            pert_set,
            delta_summary: T::new_empty(mine_life + 1),
            block_cnt: 0,
        }
    }

    /// Checks if the aggregate perturbation contains a unit perturbation for the specified block and period.
    #[inline(always)]
    pub fn contains_pert(&self, block: u32, period: u8) -> bool {
        self.pert_set.contains(&(block, period))
    }

    /// Returns a reference to the list of unit perturbations in this aggregate perturbation.
    #[inline(always)]
    pub fn unit_perts(&self) -> &[BlockPerturbation<()>] {
        &self.unit_perts
    }

    /// Returns a reference to the summary of changes induced by this aggregate perturbation.
    #[inline(always)]
    pub fn delta_summary(&self) -> &T {
        &self.delta_summary
    }

    /// Populates the provided change map with the blocks and their new periods affected by this aggregate perturbation.
    ///
    /// # Arguments
    /// * `sched` - The current schedule represented as a slice of periods.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `change_map` - A mutable reference to a hash map to be populated with block indices and their new periods.
    #[inline(always)]
    pub fn populate_change_map(
        &self,
        sched: &[u8],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        change_map: &mut FxHashMap<u32, u8>,
    ) {
        change_map.clear();
        for pert in &self.unit_perts {
            let get_period = |b| {
                if let Some(&period) = change_map.get(&b) {
                    period
                } else {
                    sched[b as usize]
                }
            };

            let mut tmp = FxHashMap::default();
            pert.walk_effected_blocks(
                get_period,
                flat_preds,
                flat_succs,
                |block_idx, _from_period| {
                    tmp.insert(block_idx, pert.period());
                },
            );

            change_map.extend(tmp.into_iter());
        }

        change_map.retain(|k, p| *p != sched[*k as usize]);
    }

    /// Generates and returns a change map of blocks and their new periods affected by this aggregate perturbation.
    ///
    /// # Arguments
    /// * `sched` - The current schedule represented as a slice of periods.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    #[inline(always)]
    pub fn get_change_map(
        &self,
        sched: &[u8],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) -> FxHashMap<u32, u8> {
        let mut change_map = FxHashMap::default();
        for pert in &self.unit_perts {
            let get_period = |b| {
                if let Some(&period) = change_map.get(&b) {
                    period
                } else {
                    sched[b as usize]
                }
            };

            let mut tmp = FxHashMap::default();
            pert.walk_effected_blocks(
                get_period,
                flat_preds,
                flat_succs,
                |block_idx, _from_period| {
                    tmp.insert(block_idx, pert.period());
                },
            );

            change_map.extend(tmp.into_iter());
        }

        change_map.retain(|k, p| *p != sched[*k as usize]);
        change_map
    }

    /// Computes and updates the summary of changes induced by this aggregate perturbation when applied to the current schedule.
    ///
    /// NOTE: This function assumes that `mappable_sched` is identical to `curr_sched` at the start of its execution.
    ///
    /// # Arguments
    /// * `blocks` - A slice of blocks in the schedule.
    /// * `curr_sched` - The current schedule represented as a slice of periods.
    /// * `mappable_sched` - A mutable slice representing the schedule that will be updated with the new periods.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `context` - The context used for perturbation summary calculations.
    /// * `buffer` - A mutable vector used as a temporary buffer to track changed blocks.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn compute(
        &mut self,
        blocks: &[T::Block],
        curr_sched: &[u8],
        mappable_sched: &mut [u8],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        context: &T::Context,
        buffer: &mut Vec<u32>,
    ) {
        // 1. Reset the perturbation summary.
        self.delta_summary.reset();
        // self.geometric_penalty = 0.0;
        self.block_cnt = 0;

        // 2. Apply each perturbation sequentially.
        let mut cnt = 0;
        for pert in &mut self.unit_perts {
            // a. For each perturbation, define a function to get the current period of a block.
            let curr_sched_fn = |b: u32| mappable_sched[b as usize];

            // b. Walk through the effected blocks and update the summary.
            pert.walk_effected_blocks(
                curr_sched_fn,
                flat_preds,
                flat_succs,
                |block_idx, from_period| {
                    // Get the block.
                    let block = &blocks[block_idx as usize];
                    // Remove the blocks contribution from its previous period.
                    self.delta_summary.sub_block(from_period, block, context);
                    // Add the blocks contribution to its new period.
                    self.delta_summary.add_block(pert.period(), block, context);
                    // Record the block index in the buffer.
                    buffer.push(block_idx);
                },
            );

            // c. Update the mappable schedule with the new periods for the effected blocks.
            for &block_idx in buffer[cnt..].iter() {
                mappable_sched[block_idx as usize] = pert.period();
            }
            cnt = buffer.len();
        }

        // 3. Clean up the buffer to retain only the blocks that have changed periods.
        buffer.retain(|i| mappable_sched[*i as usize] != curr_sched[*i as usize]);

        // 4. Update the block count.
        self.block_cnt = buffer.len();
    }

    /// Applies this aggregate perturbation to the provided schedule, updating the schedule in place.
    ///
    /// Also records the changed blocks and their new periods in the provided vectors.
    ///
    /// # Arguments
    /// * `sched` - A mutable slice representing the schedule to be updated.
    /// * `changed_blocks` - A mutable vector to record the indices of blocks that have changed periods.
    /// * `changed_periods` - A mutable vector to record the new periods of the changed blocks.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    #[inline(always)]
    pub fn apply_to_sched(
        &self,
        sched: &mut [u8],
        changed_blocks: &mut Vec<u32>,
        changed_periods: &mut Vec<u8>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        // 1. Get the change map induced by this aggregate perturbation.
        let change_map = self.get_change_map(sched, flat_preds, flat_succs);

        // 2. Apply the changes to the schedule and record the changed blocks and periods.
        change_map.iter().for_each(|(&block_idx, &period)| {
            changed_blocks.push(block_idx);
            changed_periods.push(period);
            sched[block_idx as usize] = period;
        });
    }
}

/// A collection of aggregate perturbations, with efficient access to perturbations affecting specific blocks and periods.
///
/// This structure maintains a mapping from (block_index, period) pairs to the indices of aggregate perturbations that include unit perturbations for those blocks and periods.
pub struct AggregatePerturbationCollection<T> {
    dep_map: FxHashMap<(u32, u8), Vec<u32>>,
    perturbations: Vec<Mutex<AggregatePerturbation<T>>>,
}

impl<T> AggregatePerturbationCollection<T> {
    /// Creates a new collection of aggregate perturbations from the provided list.
    pub fn new(perturbations: Vec<Mutex<AggregatePerturbation<T>>>) -> Self {
        let mut dep_map: FxHashMap<(u32, u8), Vec<u32>> = FxHashMap::default();
        for (i, pert) in perturbations.iter().enumerate() {
            for unit_pert in pert.lock().unit_perts.iter() {
                dep_map
                    .entry((unit_pert.block_index(), unit_pert.period()))
                    .or_default()
                    .push(i as u32);
            }
        }

        Self {
            dep_map,
            perturbations,
        }
    }

    /// Returns the number of aggregate perturbations in the collection.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.perturbations.len()
    }

    /// Returns true if the collection contains no aggregate perturbations.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.perturbations.is_empty()
    }

    /// Returns a reference to the aggregate perturbation at the specified index.
    #[inline(always)]
    pub fn perturbation(&self, idx: u32) -> &Mutex<AggregatePerturbation<T>> {
        &self.perturbations[idx as usize]
    }

    /// Returns a reference to the list of all aggregate perturbations in the collection.
    #[inline(always)]
    pub fn perturbations(&self) -> &[Mutex<AggregatePerturbation<T>>] {
        &self.perturbations
    }

    /// Adds a new aggregate perturbation to the collection, updating the dependency map accordingly.
    #[inline(always)]
    pub fn add_aggregate_perturbation(&mut self, pert: AggregatePerturbation<T>) {
        let idx = self.perturbations.len() as u32;
        for unit_pert in pert.unit_perts.iter() {
            self.dep_map
                .entry((unit_pert.block_index(), unit_pert.period()))
                .or_default()
                .push(idx);
        }
        self.perturbations.push(Mutex::new(pert));
    }

    /// Retrieves the list of aggregate perturbation indices that affect the specified block and period.
    #[inline(always)]
    pub fn deps(&self, block: u32, period: u8) -> Option<&[u32]> {
        self.dep_map.get(&(block, period)).map(|v| v.as_slice())
    }
}

impl<T: PerturbationSummary> AggregatePerturbationCollection<T> {
    /// Populates the change map for the aggregate perturbation at the specified index.
    ///
    /// # Arguments
    /// * `idx` - The index of the aggregate perturbation in the collection.
    /// * `sched` - The current schedule represented as a slice of periods.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `change_map` - A mutable reference to a hash map to be populated with block indices and their new periods.
    #[inline(always)]
    pub fn populate_change_map(
        &self,
        idx: u32,
        sched: &[u8],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        change_map: &mut FxHashMap<u32, u8>,
    ) {
        let pert = &self.perturbations[idx as usize];
        pert.lock()
            .populate_change_map(sched, flat_preds, flat_succs, change_map);
    }
}
