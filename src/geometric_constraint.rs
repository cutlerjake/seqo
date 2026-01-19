//! Geometric constraints for scheduling.
//!
//! This module defines the `GeometricConstraint` trait, which types must implement to represent
//! geometric constraints in the scheduling process. It also provide efficient implementations
//! for mining width constraints.
//!
//! Implementors of this trait should ensure that penalties are cached as, or cheap to evaluate,
//! as `GeometricConstraint::penalty_delta` is called for every perturbation for each iteration
//! of local search.

use core::panic;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicI32, AtomicU64, Ordering},
};

use indicatif::{ParallelProgressIterator, ProgressIterator};
use itertools::{Itertools, izip};
use parking_lot::Mutex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    aggregate_perturbation::AggregatePerturbationCollection,
    block_perturbation::BlockPerturbation,
    idx_period_map::IdxPeriodMap,
    local_search::PertIdx,
    perturbation_summary::PerturbationSummary,
    relation_provider::{RelationProvider, RelationRangeProvider, to_block_ranges},
    walkers::{DualWalkerWithPredicate, FixDir},
};
use uuid::Uuid;

/// Trait representing a geometric constraint in the scheduling process.
///
/// This trait defines methods for calculating penalty costs, initializing the constraint
/// based on the current schedule and perturbations, processing changes in the schedule,
/// and retrieving the current value of the constraint.
pub trait GeometricConstraint {
    /// Returns the penalty delta associated with a specific perturbation index.
    ///
    /// NOTE: This is not the global cost, only the delta in the penalty if
    /// the perturbation at `idx` were to be applied.
    fn penalty_delta(&self, idx: PertIdx) -> f64;

    /// Initializes the geometric constraint with the current schedule and perturbations.
    ///
    /// # Arguments
    /// * `sched` - The current schedule represented as a slice of periods.
    /// * `unit_perts` - A map of unit perturbations indexed by block and period.
    /// * `aggregate_perts` - A collection of aggregate perturbations.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    fn init<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perts: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    );

    /// Processes changes in the schedule and updates the constraint accordingly.
    ///
    /// # Arguments
    /// * `new_sched` - The new schedule after changes, represented as a slice of periods.
    /// * `old_sched` - The old schedule before changes, represented as a slice of periods.
    /// * `changed_blocks` - A slice of block indices that have changed in the schedule.
    /// * `flat_preds` - A relation range provider for predecessors (must contain all predecessors and self for each block).
    /// * `flat_succs` - A relation range provider for successors (must contain all successors and self for each block).
    /// * `unit_perts` - A map of unit perturbations indexed by block and period.
    /// * `aggregate_perts` - A collection of aggregate perturbations.
    #[allow(clippy::too_many_arguments)]
    fn process_change<T: PerturbationSummary + Send + Sync>(
        &mut self,
        new_sched: &[u8],
        old_sched: &[u8],
        changed_blocks: &[u32],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perts: &AggregatePerturbationCollection<T>,
    );

    /// Returns the current value of the geometric constraint based on the schedule.
    ///
    /// # Arguments
    /// * `sched` - The current schedule represented as a slice of periods.
    fn curr_val(&self, sched: &[u8]) -> f64;
}

/// A no-op implementation of the `GeometricConstraint` trait.
impl GeometricConstraint for () {
    /// Returns a penalty cost of zero for any perturbation index.
    #[inline(always)]
    fn penalty_delta(&self, _idx: PertIdx) -> f64 {
        0.0
    }

    /// Returns a current value of zero for the geometric constraint.
    #[inline(always)]
    fn curr_val(&self, _sched: &[u8]) -> f64 {
        0.0
    }

    /// No initialization needed for the no-op constraint.
    fn init<T: PerturbationSummary + Send + Sync>(
        &mut self,
        _sched: &[u8],
        _unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        _aggregate_perts: &AggregatePerturbationCollection<T>,
        _flat_preds: &RelationRangeProvider,
        _flat_succs: &RelationRangeProvider,
    ) {
    }

    /// No processing needed for changes in the schedule.
    fn process_change<T: PerturbationSummary + Send + Sync>(
        &mut self,
        _sched: &[u8],
        _old_sched: &[u8],
        _changed: &[u32],
        _flat_preds: &RelationRangeProvider,
        _flat_succs: &RelationRangeProvider,
        _unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        _aggregate_perts: &AggregatePerturbationCollection<T>,
    ) {
    }
}

/// A composite implementation of the `GeometricConstraint` trait that combines two constraints.
impl<A: GeometricConstraint, B: GeometricConstraint> GeometricConstraint for (A, B) {
    /// Returns the sum of the penalty costs from both constraints for a specific perturbation index.
    #[inline(always)]
    fn penalty_delta(&self, idx: PertIdx) -> f64 {
        self.0.penalty_delta(idx) + self.1.penalty_delta(idx)
    }

    /// Returns the sum of the current values from both constraints based on the schedule.
    #[inline(always)]
    fn curr_val(&self, sched: &[u8]) -> f64 {
        self.0.curr_val(sched) + self.1.curr_val(sched)
    }

    /// Initializes both constraints with the current schedule and perturbations.
    fn init<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perts: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        self.0
            .init(sched, unit_perts, aggregate_perts, flat_preds, flat_succs);
        self.1
            .init(sched, unit_perts, aggregate_perts, flat_preds, flat_succs);
    }

    /// Processes changes in the schedule and updates both constraints accordingly.
    fn process_change<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        old_sched: &[u8],
        changed: &[u32],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perts: &AggregatePerturbationCollection<T>,
    ) {
        self.0.process_change(
            sched,
            old_sched,
            changed,
            flat_preds,
            flat_succs,
            unit_perts,
            aggregate_perts,
        );
        self.1.process_change(
            sched,
            old_sched,
            changed,
            flat_preds,
            flat_succs,
            unit_perts,
            aggregate_perts,
        );
    }
}

/// A geometric constraint that accurately computes mining width penalties.
///
/// This constraint only assigns a penalty when there are no mining width `windows`
/// overlapping the block in question with all periods equal. If no such window exists,
/// the minimum penalty over all possible windows is assigned.
pub struct AccurateMiningWidthConstraint<'a> {
    penalty_cost: f64,
    mw_relations: &'a RelationProvider<(u32, f32)>,
    unit_penalties: IdxPeriodMap<Mutex<f64>>,
    agg_penalties: Vec<Mutex<f64>>,
    expanded_mw_relations: RelationRangeProvider,
    mine_life: u8,
    discounts: Vec<f64>,
}

impl<'a> AccurateMiningWidthConstraint<'a> {
    /// Create a new `AccurateMiningWidthConstraint`.
    pub fn new(
        penalty_cost: f64,
        mw_relations: &'a RelationProvider<(u32, f32)>,
        mine_life: u8,
        discount_rate: f64,
    ) -> Self {
        // Expand the mining width relatations to include indirect relations.
        //
        // This is the set of all possible blocks that may have their mining width
        // affected by a change in the target block.
        let mut expanded_mw_relations = vec![];
        for block in 0..mw_relations.num_blocks() {
            let mut tmp = FxHashSet::default();
            for (rel_block, weight) in mw_relations.relations(block) {
                // Zero weight relations are invalid.
                assert!(*weight != 0.0);
                for (mw_block, _mw_weight) in mw_relations.relations(*rel_block as usize) {
                    tmp.insert(*mw_block);
                }
            }
            // Must contain itself.
            assert!(tmp.contains(&(block as u32)));
            expanded_mw_relations.push(to_block_ranges(&tmp.into_iter().sorted().collect_vec()));
        }

        let expanded_mw_relations = RelationRangeProvider::from_linked_list(&expanded_mw_relations);

        let discounts = (0..mine_life + 1)
            .map(|i| (1.0f64 + discount_rate).powi(-(i as i32)))
            .collect::<Vec<_>>();

        Self {
            penalty_cost,
            mw_relations,
            expanded_mw_relations,
            unit_penalties: IdxPeriodMap::default(),
            agg_penalties: vec![],
            mine_life,
            discounts,
            // mw_expanded_all,
        }
    }
}

/// Represents the state of the mining width penalty computation.
pub enum MWState {
    /// The mining width is valid, with the specified center and relations.
    ///
    /// This state indicates that there exists a mining width window centered on `mw_center`
    /// where all related blocks have the same period as the target block, resulting in zero penalty.
    Valid { mw_center: u32 },
    /// The mining width is invalid, with the specified penalty value.
    Invalid { penalty: f64 },
    /// The mining width state is unchecked.
    Unchecked,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PackedIdxPeriod(u64);

impl PackedIdxPeriod {
    #[inline(always)]
    fn new(idx: u32, period: u8) -> Self {
        Self(((idx as u64) << 8) | (period as u64))
    }
}

impl AccurateMiningWidthConstraint<'_> {
    /// Returns the minimum mining width penalty for a target block in the schedule.
    ///
    /// If a valid mining width window is found, returns `MWState::Valid` with the center block.
    /// If no valid window is found, returns `MWState::Invalid` with the penalty value.
    /// If the target block is unscheduled, returns `MWState::Unchecked`.
    #[inline(always)]
    fn mw_penalty(
        &self,
        sched: &[u8],
        target_block: u32,
        cached: &mut FxHashMap<PackedIdxPeriod, f64>,
    ) -> MWState {
        let cmp_period = sched[target_block as usize];
        if cmp_period == self.mine_life {
            return MWState::Unchecked;
        }

        // Get all blocks that are related to the target block
        let to_check = self.mw_relations.relations(target_block as usize);
        let discounts = &self.discounts;

        // Find the minimum mining width penalty centered on one of the related blocks.
        // We can return early if we find a penalty of zero.
        let mut min_penalty = f64::MAX;
        for (block, _) in to_check {
            if sched[*block as usize] == self.mine_life {
                continue;
            }

            let penalty = *cached
                .entry(PackedIdxPeriod::new(*block, cmp_period))
                .or_insert_with(|| {
                    let mut penalty = 0.0;
                    for (i, weight) in self.mw_relations.relations(*block as usize) {
                        let period = sched[*i as usize];
                        let min_period = cmp_period.min(period);

                        penalty += if period != cmp_period {
                            *weight
                                * unsafe { *discounts.get_unchecked(min_period as usize) } as f32
                        } else {
                            0.0
                        };
                    }

                    penalty as f64
                });

            if penalty < min_penalty {
                // Update the minimum penalty
                min_penalty = penalty;

                if min_penalty.abs() < 1e-6 {
                    return MWState::Valid { mw_center: *block };
                }
            }
        }

        MWState::Invalid {
            penalty: min_penalty,
        }
    }

    /// Computes the mining width penalty delta for a target block given changes in the schedule.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn mw_point_delta_from_change(
        &self,
        curr_sched: &[u8],
        changed_sched: &[u8],
        target_block: u32,
        curr_checked: &mut FxHashSet<u32>,
        new_checked: &mut FxHashSet<u32>,
        cached_curr: &mut FxHashMap<PackedIdxPeriod, f64>,
        cached_new: &mut FxHashMap<PackedIdxPeriod, f64>,
    ) -> f64 {
        // 1. compute the penalty in the unchanged state.
        let curr_penalty = if curr_checked.contains(&target_block) {
            0.0
        } else {
            match self.mw_penalty(curr_sched, target_block, cached_curr) {
                MWState::Valid { mw_center } => {
                    curr_checked.extend(
                        unsafe { self.mw_relations.relations_unchecked(mw_center as usize) }
                            .iter()
                            .map(|(b, _)| *b),
                    );
                    0.0
                }
                MWState::Invalid { penalty } => penalty,
                MWState::Unchecked => 0.0,
            }
        };

        // 2. compute the penalty in the changed state.
        let new_penalty = if new_checked.contains(&target_block) {
            0.0
        } else {
            match self.mw_penalty(changed_sched, target_block, cached_new) {
                MWState::Valid { mw_center } => {
                    new_checked.extend(
                        unsafe { self.mw_relations.relations_unchecked(mw_center as usize) }
                            .iter()
                            .map(|(b, _)| *b),
                    );

                    0.0
                }
                MWState::Invalid { penalty } => penalty,
                MWState::Unchecked => 0.0,
            }
        };

        new_penalty - curr_penalty
    }

    /// Computes the total mining width penalty delta resulting from changes in the schedule.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn mw_delta_from_change(
        &self,
        curr_sched: &[u8],
        changed_sched: &[u8],
        changed: &FxHashMap<u32, u8>,
        curr_checked: &mut FxHashSet<u32>,
        new_checked: &mut FxHashSet<u32>,
        cached_curr: &mut FxHashMap<PackedIdxPeriod, f64>,
        cached_new: &mut FxHashMap<PackedIdxPeriod, f64>,
        visited: &mut [u32],
        id: u32, // deps: &mut FxHashMap<u32, i32>,
    ) -> f64 {
        let mut sum = 0.0;

        for key in changed.keys() {
            for block in self.expanded_mw_relations.relations(*key as usize) {
                if visited[block as usize] == id {
                    continue;
                }

                visited[block as usize] = id;

                let delta = self.mw_point_delta_from_change(
                    curr_sched,
                    changed_sched,
                    block,
                    curr_checked,
                    new_checked,
                    cached_curr,
                    cached_new,
                );

                sum += delta;
            }
        }

        sum
    }

    /// Computes the global mining width penalty for the entire schedule.
    fn global_mw_penalty(&self, sched: &[u8]) -> f64 {
        let mut violation_cnt = 0.0;
        let mut checked = ahash::HashSet::default();

        for i in 0..sched.len() {
            let penalty = if checked.contains(&(i as u32)) {
                0.0
            } else {
                match self.mw_penalty(sched, i as u32, &mut Default::default()) {
                    MWState::Valid { mw_center } => {
                        checked.extend(
                            unsafe { self.mw_relations.relations_unchecked(mw_center as usize) }
                                .iter()
                                .map(|(b, _)| *b),
                        );
                        0.0
                    }
                    MWState::Invalid { penalty } => penalty,
                    MWState::Unchecked => 0.0,
                }
            };
            violation_cnt += penalty;
        }

        violation_cnt
    }

    /// Checks the correctness of unit mining width penalties.
    #[allow(dead_code)]
    fn check_unit_mw_correctness<T: PerturbationSummary>(
        &self,
        sched: &[u8],
        perturbations: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        checked_blocks: &ahash::HashMap<u32, Vec<u8>>,
    ) {
        // 1. create a copy of the schedule
        let mut local_sched = sched.to_vec();

        // 2. Compute the global mining width penalty
        let global_penalty = self.global_mw_penalty(&local_sched);

        // 2. Iterate over all perturbations
        for ((idx, _period), pert) in perturbations.into_iter() {
            // 3. Get the perturbation.
            let pert = pert.lock();
            let period = pert.period();

            let fix_dir = FixDir::new(local_sched[pert.block_index() as usize], period);

            let relations = match fix_dir {
                FixDir::Preds(_) => flat_preds.relations(pert.block_index() as usize),
                FixDir::Succs(_) => flat_succs.relations(pert.block_index() as usize),
            };

            let mut changed = vec![];
            for block in relations {
                if !fix_dir.is_valid(local_sched[block as usize]) {
                    changed.push(block);
                }
            }

            // let changed = pert.effected_blocks().to_vec();

            // 4. Get required info to revert perturbation.
            let prev_periods = changed
                .iter()
                .map(|b| local_sched[*b as usize])
                .collect::<Vec<_>>();

            // 5. Apply pert.
            changed
                .iter()
                .for_each(|b| local_sched[*b as usize] = period);

            // 6. Compute new global mining width penalty.
            let new_mw_penalty = self.global_mw_penalty(&local_sched);

            // 7. Ensure delta between new and old matches computed delta
            let true_delta = new_mw_penalty - global_penalty;
            let computed_delta = *self.unit_penalties.at((pert.block_index(), period)).lock();

            if (computed_delta - true_delta).abs() > 1e-3 {
                println!("INCORRECT MW PENALTY");
                println!("Block: {idx}, period: {period}");
                println!("True {true_delta}, computed: {computed_delta}");
                println!("block checked: {}", checked_blocks.contains_key(&idx));
                println!(
                    "periods checked: {:?}",
                    checked_blocks
                        .get(&idx)
                        .map(|v| v.iter().contains(&period))
                        .unwrap_or_default()
                );
                println!("all checked periods: {:?}", checked_blocks.get(&idx));
                panic!();
            }

            // 8. Revert perturbation.
            changed.iter().zip(prev_periods.iter()).for_each(|(b, p)| {
                local_sched[*b as usize] = *p;
            })
        }

        println!("MW CORRECTNESS CHECK PASSED");
    }

    /// Checks the correctness of aggregate mining width penalties.
    #[allow(clippy::too_many_arguments)]
    fn check_agg_mw_correctness<T: PerturbationSummary>(
        &self,
        sched: &[u8],
        prev_sched: &[u8],
        changed_blocks: &[u32],
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        aggs_to_fix: &FxHashSet<u32>,
        total_set_vec: &FxHashSet<(u32, u8)>,
    ) {
        let curr = self.global_mw_penalty(sched);
        let mut mapped_sched = sched.to_vec();
        let mut visited = vec![0; sched.len()];

        for (idx, agg) in aggregate_perturbations.perturbations().iter().enumerate() {
            let agg = agg.lock();
            let change = agg.get_change_map(sched, flat_preds, flat_succs);
            for (block, period) in change.iter() {
                mapped_sched[*block as usize] = *period;
            }

            let new_global = self.global_mw_penalty(&mapped_sched);
            let true_delta = (new_global - curr) * self.penalty_cost;

            let diff = true_delta - self.penalty_cost * *self.agg_penalties[idx].lock();

            if diff.abs() > 1e-6 {
                let recomputed_penalty = self.penalty_cost
                    * self.mw_delta_from_change(
                        sched,
                        &mapped_sched,
                        &change,
                        &mut FxHashSet::default(),
                        &mut FxHashSet::default(),
                        &mut FxHashMap::default(),
                        &mut FxHashMap::default(),
                        &mut visited,
                        idx as u32 + 1,
                    );

                izip!(sched.iter(), prev_sched.iter())
                    .enumerate()
                    .for_each(|(x, (a, b))| {
                        if a != b && !changed_blocks.contains(&(x as u32)) {
                            panic!("Schedules differ during recompute!");
                        }
                    });

                let old_change = agg.get_change_map(prev_sched, flat_preds, flat_succs);
                assert_eq!(old_change, change);

                let unit_perts = agg
                    .unit_perts()
                    .iter()
                    .map(|p| (p.block_index(), p.period()))
                    .collect::<Vec<_>>();
                let flat_preds = changed_blocks
                    .iter()
                    .flat_map(|b| flat_preds.relations(*b as usize))
                    .collect::<FxHashSet<_>>();
                let flat_succs = changed_blocks
                    .iter()
                    .flat_map(|b| flat_succs.relations(*b as usize))
                    .collect::<FxHashSet<_>>();

                let mw_deps_changed = {
                    let mut set = FxHashSet::default();
                    for b in changed_blocks.iter() {
                        for dep in self.expanded_mw_relations.relations(*b as usize) {
                            set.insert(dep);
                        }
                    }
                    set
                };

                let mw_deps_agg = {
                    let mut set = FxHashSet::default();
                    for (b, _) in unit_perts.iter() {
                        for dep in self.expanded_mw_relations.relations(*b as usize) {
                            set.insert(dep);
                        }
                    }
                    set
                };

                println!("AGGREGATE PERTURBATION PENALTY MISMATCH");
                println!(
                    "True: {}, computed: {}, recomputed: {}",
                    true_delta,
                    self.penalty_cost * *self.agg_penalties[idx].lock(),
                    recomputed_penalty
                );
                println!("DIFF: {}", diff);
                println!("Checked: {}", aggs_to_fix.contains(&(idx as u32)));
                println!("Unit perts: {:?}", unit_perts);
                println!(
                    "Unit pert blocks: {:?}",
                    unit_perts.iter().map(|(b, _)| *b).collect::<Vec<_>>()
                );
                println!(
                    "Unit perts checked: {:?}",
                    unit_perts
                        .iter()
                        .any(|(b, p)| { total_set_vec.contains(&(*b, *p)) })
                );

                println!("Changed blocks: {:?}", changed_blocks);
                println!("flatpreds: {:?}", flat_preds);
                println!("flat succs: {:?}", flat_succs);

                println!(
                    "Unit perts in preds: {:?}",
                    unit_perts
                        .iter()
                        .filter(|(b, _)| flat_preds.contains(b))
                        .collect::<Vec<_>>()
                );
                println!(
                    "Unit perts in succs: {:?}",
                    unit_perts
                        .iter()
                        .filter(|(b, _)| flat_succs.contains(b))
                        .collect::<Vec<_>>()
                );
                println!("Unit perts mw deps: {:?}", mw_deps_changed);
                println!("Agg mw deps: {:?}", mw_deps_agg);
                println!("Agg change: {:?}", change.keys().collect::<Vec<_>>());

                // Save changed blocks to csv.
                // let mut wtr = csv::Writer::from_path("./changed_blocks.csv").unwrap();
                // wtr.write_record(&["x", "y", "z", "dx", "dy", "dz"]).unwrap();
                // for b in changed_blocks.iter() {
                //     let block =
                // }
                panic!();
            }

            for (b, _) in change.iter() {
                mapped_sched[*b as usize] = sched[*b as usize];
            }
        }
    }

    /// Checks the correctness of mining width penalties for both unit and aggregate perturbations.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn check_mw_correctness<T: PerturbationSummary>(
        &self,
        sched: &[u8],
        prev_sched: &[u8],
        changed_blocks: &[u32],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        let mut checked_blocks = ahash::HashMap::default();
        for ((idx, period), _pert) in unit_perts.into_iter() {
            checked_blocks
                .entry(idx)
                .or_insert_with(Vec::new)
                .push(period);
        }

        self.check_unit_mw_correctness(sched, unit_perts, flat_preds, flat_succs, &checked_blocks);

        let aggs_to_fix = checked_blocks.keys().copied().collect::<FxHashSet<_>>();
        let total_set_vec = checked_blocks
            .iter()
            .flat_map(|(b, periods)| periods.iter().map(move |p| (*b, *p)))
            .collect::<FxHashSet<_>>();

        self.check_agg_mw_correctness(
            sched,
            prev_sched,
            changed_blocks,
            aggregate_perturbations,
            flat_preds,
            flat_succs,
            &aggs_to_fix,
            &total_set_vec,
        );
    }
}

impl GeometricConstraint for AccurateMiningWidthConstraint<'_> {
    /// Returns the penalty cost associated with a specific perturbation index.
    #[inline(always)]
    fn penalty_delta(&self, idx: PertIdx) -> f64 {
        self.penalty_cost
            * match idx {
                PertIdx::Unit(block, period) => *self.unit_penalties.at((block, period)).lock(),
                PertIdx::Aggregate(i) => *self.agg_penalties[i].lock(),
            }
    }

    /// Returns the current value of the geometric constraint based on the schedule.
    #[inline(always)]
    fn curr_val(&self, sched: &[u8]) -> f64 {
        self.penalty_cost * self.global_mw_penalty(sched)
    }

    /// Initializes the geometric constraint with the current schedule and perturbations.
    fn init<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        self.unit_penalties = IdxPeriodMap::default();
        unit_perts
            .into_iter()
            .for_each(|(idx, _pert)| self.unit_penalties.insert(idx, Mutex::new(0.0)));

        self.agg_penalties = vec![];
        for _ in 0..aggregate_perturbations.perturbations().len() {
            self.agg_penalties.push(Mutex::new(0.0));
        }

        let len = unit_perts.len();
        unit_perts
            .par_iter()
            .progress_count(len as u64)
            .for_each_init(
                || {
                    (
                        FxHashMap::default(),
                        FxHashSet::default(),
                        FxHashSet::default(),
                        vec![0; sched.len()],
                        FxHashMap::default(),
                        FxHashMap::default(),
                        sched.to_vec(),
                    )
                },
                |(
                    change_map,
                    curr_checked,
                    new_checked,
                    visited,
                    chached_curr,
                    cached_new,
                    changed_sched,
                ),
                 ((node, period), pert)| {
                    let pert = pert.lock();

                    // previous_set.clear();
                    change_map.clear();
                    // buffer.clear();

                    pert.walk_effected_blocks(
                        |b| sched[b as usize],
                        flat_preds,
                        flat_succs,
                        |b, _p| {
                            change_map.insert(b, pert.period());
                        },
                    );

                    // Update changed_sched
                    for (block, period) in change_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    new_checked.clear();
                    cached_new.clear();
                    // curr_checked.clear();
                    let penalty = self.mw_delta_from_change(
                        sched,
                        changed_sched,
                        change_map,
                        curr_checked,
                        new_checked,
                        chached_curr,
                        cached_new,
                        visited,
                        (node * self.mine_life as u32) + period as u32 + 1,
                    );

                    *self
                        .unit_penalties
                        .at((node, period))
                        // .at((pert.block_index(), pert.period()))
                        .lock() = penalty;

                    // Revert changed_sched
                    for (block, _period) in change_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );

        let change_map_time = AtomicU64::new(0);
        let penalty_time = AtomicU64::new(0);
        let t1 = std::time::Instant::now();
        aggregate_perturbations
            .perturbations()
            .par_iter()
            .enumerate()
            .progress()
            .for_each_init(
                || {
                    (
                        FxHashMap::default(),
                        FxHashSet::default(),
                        FxHashSet::default(),
                        vec![0; sched.len()],
                        FxHashMap::default(),
                        FxHashMap::default(),
                        sched.to_vec(),
                    )
                },
                |(
                    change_map,
                    curr_checked,
                    new_checked,
                    visited,
                    cached_curr,
                    cached_new,
                    changed_sched,
                ),
                 (idx, agg_pert)| {
                    let t_start = std::time::Instant::now();
                    agg_pert
                        .lock()
                        .populate_change_map(sched, flat_preds, flat_succs, change_map);
                    let t_end = std::time::Instant::now();
                    change_map_time.fetch_add(
                        (t_end - t_start).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    // Update changed_sched
                    for (block, period) in change_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    // println!("CHANGED MAP SIZE: {}", changed_map.len());
                    new_checked.clear();
                    cached_new.clear();
                    let t_start = std::time::Instant::now();
                    let penalty = self.mw_delta_from_change(
                        sched,
                        changed_sched,
                        change_map,
                        curr_checked,
                        new_checked,
                        cached_curr,
                        cached_new,
                        visited,
                        idx as u32 + 1,
                    );

                    let t_end = std::time::Instant::now();
                    penalty_time.fetch_add(
                        (t_end - t_start).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    // agg_pert.lock().geometric_penalty = self.penalty_cost * penalty;
                    *self.agg_penalties[idx].lock() = penalty;

                    // Revert changed_sched
                    for (block, _period) in change_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );
        let t2 = std::time::Instant::now();
        println!(
            "AGGREGATE PERTURBATIONS MW TIME: {}",
            (t2 - t1).as_secs_f32()
        );

        println!(
            "CHANGE MAP TIME: {}s",
            change_map_time.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        );

        println!(
            "PENALTY TIME: {}s",
            penalty_time.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        );

        // Uncomment to check correctness
        // self.check_mw_correctness(
        //     sched,
        //     sched,
        //     &[],
        //     unit_perts,
        //     aggregate_perturbations,
        //     flat_preds,
        //     flat_succs,
        // );

        println!("MW INIT COMPLETE");
    }

    /// Processes changes in the schedule and updates the mining width penalties accordingly.
    fn process_change<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        prev_sched: &[u8],
        changed_blocks: &[u32],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
    ) {
        let t1 = std::time::Instant::now();
        // 1. Create set of blocks containing changed blocks
        //    with a mining width dep on the changed set
        //    These are all the mining widths that may have new penalties.
        let mut mw_set = FxHashSet::default();

        changed_blocks
            .iter()
            .copied()
            .flat_map(|block| self.expanded_mw_relations.relations(block as usize))
            .for_each(|b| {
                self.expanded_mw_relations
                    .relations(b as usize)
                    .for_each(|b| {
                        mw_set.insert(b);
                    });
            });

        // 2. Create set of blocks the depend on the blocks in `me_set`
        let mw_set = mw_set.into_iter().sorted().collect_vec();
        let min_period = mw_set
            .iter()
            .flat_map(|b| [sched[*b as usize], prev_sched[*b as usize]])
            .min()
            .unwrap();
        let max_period = mw_set
            .iter()
            .flat_map(|b| [sched[*b as usize], prev_sched[*b as usize]])
            .max()
            .unwrap();

        let mut total_set = FxHashSet::default();
        let mut all_nodes = FxHashSet::default();
        let mut preds = FxHashSet::default();
        let mut succs = FxHashSet::default();

        for mw in mw_set.iter() {
            for pred in flat_preds.relations(*mw as usize) {
                all_nodes.insert(pred);
                preds.insert(pred);
                for period in min_period..self.mine_life + 1 {
                    total_set.insert((pred, period));
                }
            }

            for succ in flat_succs.relations(*mw as usize) {
                all_nodes.insert(succ);
                succs.insert(succ);
                for period in 0..max_period + 1 {
                    total_set.insert((succ, period));
                }
            }
        }

        let len = total_set.len();

        total_set
            .par_iter()
            .progress_count(len as u64)
            .filter(|(idx, period)| unit_perts.contains((*idx, *period)))
            .for_each_init(
                || {
                    (
                        FxHashMap::default(),
                        FxHashSet::default(),
                        FxHashSet::default(),
                        vec![0; sched.len()],
                        FxHashMap::default(),
                        FxHashMap::default(),
                        sched.to_vec(),
                    )
                },
                |(
                    change_map,
                    curr_checked,
                    new_checked,
                    visited,
                    chached_curr,
                    cached_new,
                    changed_sched,
                ),
                 (node, period)| {
                    let pert = unit_perts.at((*node, *period)).lock();

                    // previous_set.clear();
                    change_map.clear();
                    // buffer.clear();

                    pert.walk_effected_blocks(
                        |b| sched[b as usize],
                        flat_preds,
                        flat_succs,
                        |b, _p| {
                            change_map.insert(b, pert.period());
                        },
                    );

                    // Update changed_sched
                    for (block, period) in change_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    new_checked.clear();
                    cached_new.clear();
                    // curr_checked.clear();
                    let penalty = self.mw_delta_from_change(
                        sched,
                        changed_sched,
                        change_map,
                        curr_checked,
                        new_checked,
                        chached_curr,
                        cached_new,
                        visited,
                        (*node * self.mine_life as u32) + *period as u32 + 1,
                    );

                    *self
                        .unit_penalties
                        .at((*node, *period))
                        // .at((pert.block_index(), pert.period()))
                        .lock() = penalty;

                    // Revert changed_sched
                    for (block, _period) in change_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );

        let mut aggs_to_fix = FxHashSet::default();
        for (node, period) in total_set.iter() {
            if let Some(deps) = aggregate_perturbations.deps(*node, *period) {
                for agg_idx in deps {
                    aggs_to_fix.insert(*agg_idx);
                }
            }
        }

        let len = aggs_to_fix.len();
        aggs_to_fix
            .par_iter()
            .progress_count(len as u64)
            .for_each_init(
                || {
                    (
                        FxHashMap::default(),
                        FxHashSet::default(),
                        FxHashSet::default(),
                        vec![0; sched.len()],
                        FxHashMap::default(),
                        FxHashMap::default(),
                        sched.to_vec(),
                    )
                },
                |(
                    changed_map,
                    curr_checked,
                    new_checked,
                    visited,
                    cached_curr,
                    cached_new,
                    changed_sched,
                ),
                 idx| {
                    aggregate_perturbations.populate_change_map(
                        *idx,
                        sched,
                        flat_preds,
                        flat_succs,
                        changed_map,
                    );

                    // Update changed_sched
                    for (block, period) in changed_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    // println!("CHANGED MAP SIZE: {}", changed_map.len());
                    // curr_checked.clear();
                    new_checked.clear();
                    cached_new.clear();
                    let penalty = self.mw_delta_from_change(
                        sched,
                        changed_sched,
                        changed_map,
                        curr_checked,
                        new_checked,
                        cached_curr,
                        cached_new,
                        visited,
                        *idx + 1,
                    );

                    *self.agg_penalties[*idx as usize].lock() = penalty;

                    // Revert changed_sched
                    for (block, _period) in changed_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );

        // Uncomment to check correctness
        // self.check_mw_correctness(
        //     sched,
        //     prev_sched,
        //     changed_blocks,
        //     unit_perts,
        //     aggregate_perturbations,
        //     flat_preds,
        //     flat_succs,
        // );

        let t2 = std::time::Instant::now();
        println!("MW PROC TIME: {}", (t2 - t1).as_secs_f32());
    }
}

/// A fast, but conservative implementation of the mining width constraint.
///
/// only checks the mining width centered on each block, without attempting
/// to find a covering mining width that minimizes the penalty.
pub struct FastMiningWidthConstraint<'a> {
    penalty_cost: f64,
    mw_relations: &'a RelationProvider<(u32, f32)>,
    unit_penalties: IdxPeriodMap<Mutex<f64>>,
    agg_penalties: Vec<Mutex<f64>>,
    mine_life: u8,
    discounts: Vec<f64>,
}

impl<'a> FastMiningWidthConstraint<'a> {
    pub fn new(
        penalty_cost: f64,
        mw_relations: &'a RelationProvider<(u32, f32)>,
        mine_life: u8,
        discount_rate: f64,
    ) -> Self {
        let discounts = (0..mine_life + 1)
            .map(|i| (1.0f64 + discount_rate).powi(-(i as i32)))
            .collect::<Vec<_>>();

        Self {
            penalty_cost,
            mw_relations,
            unit_penalties: IdxPeriodMap::default(),
            agg_penalties: vec![],
            mine_life,
            discounts,
        }
    }
}

impl FastMiningWidthConstraint<'_> {
    #[inline(always)]
    fn mw_penalty(&self, sched: &[u8], target_block: u32) -> f64 {
        let cmp_period = sched[target_block as usize];

        if cmp_period == self.mine_life {
            return 0.0;
        }

        let mut penalty = 0.0;
        let mut cnt = 0;
        let discounts = &self.discounts;
        let relations = &self.mw_relations.relations(target_block as usize);
        for (i, weight) in relations.iter() {
            let period = sched[*i as usize];
            let min_period = cmp_period.min(period);

            cnt += (period != cmp_period) as u32;

            penalty += if period != cmp_period {
                *weight * unsafe { *discounts.get_unchecked(min_period as usize) } as f32
            } else {
                0.0
            };
        }

        if cnt as f32 > 0.5 * relations.len() as f32 {
            penalty as f64
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn mw_point_delta_from_change2(
        &self,
        curr_sched: &[u8],
        changed_sched: &[u8],
        target_block: u32,
    ) -> f64 {
        // 1. compute the penalty in the unchanged state.
        let curr_penalty = self.mw_penalty(curr_sched, target_block);

        // 2. compute the penalty in the changed state.
        let new_penalty = self.mw_penalty(changed_sched, target_block);

        new_penalty - curr_penalty
    }

    #[inline(always)]
    fn mw_delta_from_change2(
        &self,
        curr_sched: &[u8],
        changed_sched: &[u8],
        changed: &FxHashMap<u32, u8>,
        visited: &mut [u32],
        id: u32, // deps: &mut FxHashMap<u32, i32>,
    ) -> f64 {
        // 1. Create a `HashSet` to keep track of all the blocks we've visited

        let mut sum = 0.0;

        // let mut _visited = FxHashSet::default();
        for key in changed.keys() {
            for (block, _) in self.mw_relations.relations(*key as usize) {
                if visited[*block as usize] == id {
                    continue;
                }
                visited[*block as usize] = id;
                // if !_visited.insert(*block) {
                //     continue;
                // }

                let delta = self.mw_point_delta_from_change2(curr_sched, changed_sched, *block);

                sum += delta;
            }
        }

        sum
    }

    fn global_mw_penalty(&self, sched: &[u8]) -> f64 {
        let mut violation_cnt = 0.0;

        for i in 0..sched.len() {
            let penalty = self.mw_penalty(sched, i as u32);
            violation_cnt += penalty;
        }

        violation_cnt
    }

    #[allow(dead_code)]
    fn check_unit_mw_correctness<T: PerturbationSummary>(
        &self,
        sched: &[u8],
        perturbations: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        checked_blocks: &ahash::HashMap<u32, Vec<u8>>,
    ) {
        // 1. create a copy of the schedule
        let mut local_sched = sched.to_vec();

        // 2. Compute the global mining width penalty
        let global_penalty = self.global_mw_penalty(&local_sched);

        // 2. Iterate over all perturbations
        for ((idx, _period), pert) in perturbations.into_iter() {
            // 3. Get the perturbation.
            let pert = pert.lock();
            let period = pert.period();

            let fix_dir = FixDir::new(local_sched[pert.block_index() as usize], period);

            let relations = match fix_dir {
                FixDir::Preds(_) => flat_preds.relations(pert.block_index() as usize),
                FixDir::Succs(_) => flat_succs.relations(pert.block_index() as usize),
            };

            let mut changed = vec![];
            for block in relations {
                if !fix_dir.is_valid(local_sched[block as usize]) {
                    changed.push(block);
                }
            }

            // let changed = pert.effected_blocks().to_vec();

            // 4. Get required info to revert perturbation.
            let prev_periods = changed
                .iter()
                .map(|b| local_sched[*b as usize])
                .collect::<Vec<_>>();

            // 5. Apply pert.
            changed
                .iter()
                .for_each(|b| local_sched[*b as usize] = period);

            // 6. Compute new global mining width penalty.
            let new_mw_penalty = self.global_mw_penalty(&local_sched);

            // 7. Ensure delta between new and old matches computed delta
            let true_delta = new_mw_penalty - global_penalty;
            let computed_delta = *self.unit_penalties.at((pert.block_index(), period)).lock();

            if (computed_delta - true_delta).abs() > 1e-3 {
                println!("INCORRECT MW PENALTY");
                println!("Block: {idx}, period: {period}");
                println!("True {true_delta}, computed: {computed_delta}");
                println!("block checked: {}", checked_blocks.contains_key(&idx));
                println!(
                    "periods checked: {:?}",
                    checked_blocks
                        .get(&idx)
                        .map(|v| v.iter().contains(&period))
                        .unwrap_or_default()
                );
                println!("all checked periods: {:?}", checked_blocks.get(&idx));

                panic!();
            }

            // 8. Revert perturbation.
            changed.iter().zip(prev_periods.iter()).for_each(|(b, p)| {
                local_sched[*b as usize] = *p;
            })
        }

        println!("MW CORRECTNESS CHECK PASSED");
    }

    #[allow(clippy::too_many_arguments)]
    fn check_agg_mw_correctness<T: PerturbationSummary>(
        &self,
        sched: &[u8],
        prev_sched: &[u8],
        changed_blocks: &[u32],
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        aggs_to_fix: &FxHashSet<u32>,
        total_set_vec: &FxHashSet<(u32, u8)>,
    ) {
        let curr = self.global_mw_penalty(sched);
        let mut mappable_sched = sched.to_vec();
        let mut visited = vec![0u32; sched.len()];
        for (idx, agg) in aggregate_perturbations.perturbations().iter().enumerate() {
            let agg = agg.lock();
            let change = agg.get_change_map(sched, flat_preds, flat_succs);

            for (block, period) in change.iter() {
                mappable_sched[*block as usize] = *period;
            }

            let new_mw = self.global_mw_penalty(&mappable_sched);
            let true_penalty = (new_mw - curr) * self.penalty_cost;

            let diff = true_penalty - self.penalty_cost * *self.agg_penalties[idx].lock();

            if diff.abs() > 1e-6 {
                let recomputed_penalty = self.penalty_cost
                    * self.mw_delta_from_change2(
                        sched,
                        &mappable_sched,
                        &change,
                        &mut visited,
                        idx as u32 + 1,
                    );

                izip!(sched.iter(), prev_sched.iter())
                    .enumerate()
                    .for_each(|(x, (a, b))| {
                        if a != b && !changed_blocks.contains(&(x as u32)) {
                            panic!("Schedules differ during recompute!");
                        }
                    });

                let old_change = agg.get_change_map(prev_sched, flat_preds, flat_succs);
                assert_eq!(old_change, change);

                let unit_perts = agg
                    .unit_perts()
                    .iter()
                    .map(|p| (p.block_index(), p.period()))
                    .collect::<Vec<_>>();

                println!("AGGREGATE PERTURBATION PENALTY MISMATCH");
                println!(
                    "True: {}, computed: {}, recomputed: {}",
                    true_penalty,
                    *self.agg_penalties[idx].lock(),
                    recomputed_penalty
                );
                println!("DIFF: {}", diff);
                println!("Checked: {}", aggs_to_fix.contains(&(idx as u32)));
                println!("Unit perts: {:?}", unit_perts);
                println!(
                    "Unit pert blocks: {:?}",
                    unit_perts.iter().map(|(b, _)| *b).collect::<Vec<_>>()
                );
                println!(
                    "Unit perts checked: {:?}",
                    unit_perts
                        .iter()
                        .any(|(b, p)| { total_set_vec.contains(&(*b, *p)) })
                );

                println!("Changed blocks: {:?}", changed_blocks);
                println!("flatpreds: {:?}", flat_preds);
                println!("flat succs: {:?}", flat_succs);
                println!("Agg change: {:?}", change.keys().collect::<Vec<_>>());

                // Save changed blocks to csv.
                // let mut wtr = csv::Writer::from_path("./changed_blocks.csv").unwrap();
                // wtr.write_record(&["x", "y", "z", "dx", "dy", "dz"]).unwrap();
                // for b in changed_blocks.iter() {
                //     let block =
                // }
                panic!();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn check_mw_correctness<T: PerturbationSummary>(
        &self,
        sched: &[u8],
        prev_sched: &[u8],
        changed_blocks: &[u32],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        let mut checked_blocks = ahash::HashMap::default();

        for ((idx, period), _pert) in unit_perts.into_iter() {
            checked_blocks
                .entry(idx)
                .or_insert_with(Vec::new)
                .push(period);
        }

        self.check_unit_mw_correctness(sched, unit_perts, flat_preds, flat_succs, &checked_blocks);

        let mut total_set = FxHashSet::default();
        for block in changed_blocks.iter() {
            for pred in flat_preds.relations(*block as usize) {
                for period in 0..=self.mine_life {
                    total_set.insert((pred, period));
                }
            }

            for succ in flat_succs.relations(*block as usize) {
                for period in 0..=self.mine_life {
                    total_set.insert((succ, period));
                }
            }
        }

        let mut aggs_to_fix = FxHashSet::default();
        for (node, period) in total_set.iter() {
            if let Some(deps) = aggregate_perturbations.deps(*node, *period) {
                for agg_idx in deps {
                    aggs_to_fix.insert(*agg_idx);
                }
            }
        }

        self.check_agg_mw_correctness(
            sched,
            prev_sched,
            changed_blocks,
            aggregate_perturbations,
            flat_preds,
            flat_succs,
            &aggs_to_fix,
            &total_set,
        );
    }
}

impl GeometricConstraint for FastMiningWidthConstraint<'_> {
    #[inline(always)]
    fn penalty_delta(&self, idx: PertIdx) -> f64 {
        match idx {
            PertIdx::Unit(idx, period) => {
                self.penalty_cost * *self.unit_penalties.at((idx, period)).lock()
            }
            PertIdx::Aggregate(idx) => *self.agg_penalties[idx].lock(),
        }
    }

    fn curr_val(&self, sched: &[u8]) -> f64 {
        self.penalty_cost * self.global_mw_penalty(sched)
    }

    fn init<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        println!("Init MW");
        self.unit_penalties = IdxPeriodMap::default();
        unit_perts
            .into_iter()
            .for_each(|(idx, _pert)| self.unit_penalties.insert(idx, Mutex::new(0.0)));

        self.agg_penalties = vec![];
        for _ in 0..aggregate_perturbations.perturbations().len() {
            self.agg_penalties.push(Mutex::new(0.0));
        }

        let len = unit_perts.len();
        unit_perts
            .par_iter()
            .progress_count(len as u64)
            .for_each_init(
                || (FxHashMap::default(), vec![0; sched.len()], sched.to_vec()),
                |(changed, visited, mappable_sched), ((block, period), perturbation)| {
                    let perturbation = unsafe { perturbation.make_guard_unchecked() };

                    changed.clear();

                    let fix_dir = FixDir::new(
                        sched[perturbation.block_index() as usize],
                        perturbation.period(),
                    );

                    let node_provider = match fix_dir {
                        FixDir::Preds(_) => flat_preds,
                        FixDir::Succs(_) => flat_succs,
                    };

                    for dep in node_provider.relations(perturbation.block_index() as usize) {
                        if !fix_dir.is_valid(sched[dep as usize]) {
                            changed.insert(dep, perturbation.period());
                            mappable_sched[dep as usize] = perturbation.period();
                        }
                    }

                    let mw_penalty_delta = self.mw_delta_from_change2(
                        sched,
                        mappable_sched,
                        changed,
                        visited,
                        block * self.mine_life as u32 + period as u32 + 1,
                    );

                    *unsafe {
                        self.unit_penalties
                            .at((perturbation.block_index(), perturbation.period()))
                            .make_guard_unchecked()
                    } = mw_penalty_delta;
                },
            );

        println!("Init AGGREGATE PERTURBATIONS MW");
        let change_map_time = AtomicU64::new(0);
        let penalty_time = AtomicU64::new(0);
        let cum_new_cnt = AtomicU64::new(0);
        let cum_curr_cnt = AtomicU64::new(0);
        let t1 = std::time::Instant::now();
        aggregate_perturbations
            .perturbations()
            .par_iter()
            .enumerate()
            .progress()
            .for_each_init(
                || (FxHashMap::default(), vec![0; sched.len()], sched.to_vec()),
                |(change_map, visited, changed_sched), (idx, agg_pert)| {
                    let t_start = std::time::Instant::now();

                    agg_pert
                        .lock()
                        .populate_change_map(sched, flat_preds, flat_succs, change_map);
                    let t_end = std::time::Instant::now();
                    change_map_time.fetch_add(
                        (t_end - t_start).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    // Update changed_sched
                    for (block, period) in change_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    // println!("CHANGED MAP SIZE: {}", changed_map.len());
                    let t_start = std::time::Instant::now();
                    let penalty = self.mw_delta_from_change2(
                        sched,
                        changed_sched,
                        change_map,
                        visited,
                        idx as u32 + 1,
                    );

                    let t_end = std::time::Instant::now();
                    penalty_time.fetch_add(
                        (t_end - t_start).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    *self.agg_penalties[idx].lock() = self.penalty_cost * penalty;

                    // Revert changed_sched
                    for (block, _period) in change_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );
        let t2 = std::time::Instant::now();
        println!(
            "AGGREGATE PERTURBATIONS MW TIME: {}",
            (t2 - t1).as_secs_f32()
        );

        println!(
            "CHANGE MAP TIME: {}s",
            change_map_time.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        );

        println!(
            "PENALTY TIME: {}s",
            penalty_time.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        );

        println!(
            "CUM NEW CNT: {}",
            cum_new_cnt.load(std::sync::atomic::Ordering::Relaxed)
        );

        println!(
            "CUM CURR CNT: {}",
            cum_curr_cnt.load(std::sync::atomic::Ordering::Relaxed)
        );
        // Uncomment to check correctness
        // self.check_mw_correctness(sched, &self.mw_relations, perts, flat_preds, flat_succs);
    }

    fn process_change<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        prev_sched: &[u8],
        changed_blocks: &[u32],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perturbations: &AggregatePerturbationCollection<T>,
    ) {
        let t1 = std::time::Instant::now();
        // 1. Create set of blocks containing changed blocks
        //    with a mining width dep on the changed set
        //    These are all the mining widths that may have new penalties.
        let mut mw_set = FxHashSet::default();

        changed_blocks
            .iter()
            .copied()
            // .flat_map(|block| self.expanded_mw_relations.relations(*block as usize))
            .flat_map(|block| self.mw_relations.relations(block as usize))
            // .map(|(b, _)| *b)
            .for_each(|&(b, _)| {
                self.mw_relations
                    .relations(b as usize)
                    .iter()
                    .for_each(|&(b, _)| {
                        mw_set.insert(b);
                    });

                // mw_set.insert(*b);
            });

        // 2. Create set of blocks the depend on the blocks in `me_set`
        let mw_set = mw_set.into_iter().sorted().collect_vec();
        let min_period = mw_set
            .iter()
            .flat_map(|b| [sched[*b as usize], prev_sched[*b as usize]])
            .min()
            .unwrap();
        let max_period = mw_set
            .iter()
            .flat_map(|b| [sched[*b as usize], prev_sched[*b as usize]])
            .max()
            .unwrap();

        let mut total_set = FxHashSet::default();
        let mut all_nodes = FxHashSet::default();
        let mut preds = FxHashSet::default();
        let mut succs = FxHashSet::default();

        for mw in mw_set.iter() {
            for pred in flat_preds.relations(*mw as usize) {
                all_nodes.insert(pred);
                preds.insert(pred);
                for period in min_period..self.mine_life + 1 {
                    // if unit_perts.contains((node_id, period)) {
                    total_set.insert((pred, period));
                    // }
                }
            }

            for succ in flat_succs.relations(*mw as usize) {
                all_nodes.insert(succ);
                succs.insert(succ);
                for period in 0..max_period + 1 {
                    // if unit_perts.contains((node_id, period)) {
                    total_set.insert((succ, period));
                    // }
                }
            }
        }

        println!("MW SET SIZE: {}", mw_set.len());
        println!("ALL NODES SIZE: {}", all_nodes.len());
        println!("TOTAL SET SIZE: {}", total_set.len());

        let total_set_vec = total_set.into_iter().collect_vec();

        println!("MW PERTURBATIONS TO PROCESS: {}", total_set_vec.len());
        total_set_vec
            .par_iter()
            // .with_min_len(1)
            // .with_max_len(100)
            .progress()
            .filter(|(idx, period)| unit_perts.contains((*idx, *period)))
            .for_each_init(
                || (FxHashMap::default(), vec![0; sched.len()], sched.to_vec()),
                |(change_map, visited, changed_sched), (node, period)| {
                    let pert = unit_perts.at((*node, *period)).lock();

                    // previous_set.clear();
                    change_map.clear();
                    // buffer.clear();

                    pert.walk_effected_blocks(
                        |b| sched[b as usize],
                        flat_preds,
                        flat_succs,
                        |b, _p| {
                            change_map.insert(b, pert.period());
                        },
                    );

                    // Update changed_sched
                    for (block, period) in change_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    // curr_checked.clear();
                    let penalty = self.mw_delta_from_change2(
                        sched,
                        changed_sched,
                        change_map,
                        visited,
                        (*node * self.mine_life as u32) + *period as u32 + 1,
                    );

                    *self
                        .unit_penalties
                        .at((*node, *period))
                        // .at((pert.block_index(), pert.period()))
                        .lock() = penalty;

                    // Revert changed_sched
                    for (block, _period) in change_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );

        let mut aggs_to_fix = FxHashSet::default();
        for (node, period) in total_set_vec.iter() {
            if let Some(deps) = aggregate_perturbations.deps(*node, *period) {
                for agg_idx in deps {
                    aggs_to_fix.insert(*agg_idx);
                }
            }
        }

        println!("AGGREGATE PERTURBATIONS TO FIX: {}", aggs_to_fix.len());
        let len = aggs_to_fix.len();
        aggs_to_fix
            .par_iter()
            .progress_count(len as u64)
            .for_each_init(
                || (FxHashMap::default(), vec![0; sched.len()], sched.to_vec()),
                |(changed_map, visited, changed_sched), idx| {
                    aggregate_perturbations.populate_change_map(
                        *idx,
                        sched,
                        flat_preds,
                        flat_succs,
                        changed_map,
                    );

                    // Update changed_sched
                    for (block, period) in changed_map.iter() {
                        changed_sched[*block as usize] = *period;
                    }

                    let penalty = self.mw_delta_from_change2(
                        sched,
                        changed_sched,
                        changed_map,
                        visited,
                        *idx + 1,
                    );

                    *self.agg_penalties[*idx as usize].lock() = self.penalty_cost * penalty;

                    // Revert changed_sched
                    for (block, _period) in changed_map.iter() {
                        changed_sched[*block as usize] = sched[*block as usize];
                    }
                },
            );

        //Uncomment to check correctness
        // self.check_mw_correctness(
        //     sched,
        //     prev_sched,
        //     changed_blocks,
        //     unit_perts,
        //     aggregate_perturbations,
        //     flat_preds,
        //     flat_succs,
        // );

        let t2 = std::time::Instant::now();
        println!("MW PROC TIME: {}", (t2 - t1).as_secs_f32());
    }
}

#[derive(Debug, Clone, Default)]
struct ConnectedComponent {
    nodes: ahash::HashSet<u32>,
    neighbors: ahash::HashSet<Uuid>,
    min_z: i32,
    max_z: i32,
    period: u8,
}

#[derive(Debug, Clone)]
struct ConnectedComponents {
    components: ahash::HashMap<Uuid, ConnectedComponent>,
    node_map: ahash::HashMap<u32, Uuid>,
    mine_life: u8,
}

impl ConnectedComponents {
    pub fn new(mine_life: u8) -> Self {
        ConnectedComponents {
            components: ahash::HashMap::default(),
            node_map: ahash::HashMap::default(),
            mine_life,
        }
    }

    pub fn build(
        &mut self,
        sched: &[u8],
        preds: &RelationProvider,
        succs: &RelationProvider,
        locs: &[[i32; 3]],
    ) {
        self.components.clear();
        self.node_map.clear();

        let mut curr_comp_id = Uuid::new_v4();
        let mut discovered = ahash::HashSet::default();
        let mut stack = VecDeque::new();

        sched.iter().enumerate().progress().for_each(|(b, period)| {
            // 1. Check if the block is already assigned to a component.
            if self.node_map.contains_key(&(b as u32)) || *period == self.mine_life {
                return;
            }

            // 2. Create a walker.
            let mut walker =
                DualWalkerWithPredicate::new(b as u32, &mut stack, &mut discovered, |i: u32| {
                    let should_include = sched[i as usize] == *period;

                    if should_include && self.node_map.contains_key(&i) {
                        panic!("Block already assigned to a component");
                    }

                    should_include
                });

            // 3. Collect the nodes in the component.
            let mut component_nodes = ahash::HashSet::default();
            while let Some(node) = walker.next(preds, succs) {
                component_nodes.insert(node);
            }

            let mut min_z = i32::MAX;
            let mut max_z = i32::MIN;
            for node in &component_nodes {
                self.node_map.insert(*node, curr_comp_id);
                let z = locs[*node as usize][2];
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }

            self.components.insert(
                curr_comp_id,
                ConnectedComponent {
                    nodes: component_nodes,
                    neighbors: Default::default(),
                    min_z,
                    max_z,
                    period: *period,
                },
            );
            curr_comp_id = Uuid::new_v4();
        });

        // 4. Compute edges between components.
        let mut neighbors = ahash::HashSet::default();
        for (comp_id, component) in self.components.iter() {
            let starts = component.nodes.iter().copied().collect_vec();
            let mut stack = VecDeque::new();
            let mut discovered = ahash::HashSet::default();
            let mut walker = DualWalkerWithPredicate::new_with_many(
                &starts,
                &mut stack,
                &mut discovered,
                |i: u32| {
                    let should_continue = sched[i as usize] == component.period;

                    if sched[i as usize] != component.period {
                        // Check outgoing edges
                        let Some(other_comp_id) = self.node_map.get(&i) else {
                            return should_continue;
                        };
                        neighbors.insert((*comp_id, *other_comp_id));
                    }

                    should_continue
                },
            );

            while let Some(_node) = walker.next(preds, succs) {
                // Do nothing
            }
        }

        for (comp_a, comp_b) in neighbors.iter() {
            self.components
                .get_mut(comp_a)
                .unwrap()
                .neighbors
                .insert(*comp_b);
            self.components
                .get_mut(comp_b)
                .unwrap()
                .neighbors
                .insert(*comp_a);
        }

        // println!("Connected components built: {}", self.components.len());
        //println!("Nodes in components: {}", self.node_map.len());
    }

    #[track_caller]
    fn process_change(
        &mut self,
        changes: &ahash::HashSet<u32>,
        new_sched: &impl Fn(usize) -> u8,
        old_sched: &impl Fn(usize) -> u8,
        preds: &RelationProvider,
        succs: &RelationProvider,
        locs: &[[i32; 3]],
    ) -> Vec<u32> {
        // 1. Find all components that may be affected by the change.
        let mut affected_components = ahash::HashSet::default();
        let mut affected_blocks = ahash::HashSet::default();
        for block in changes.iter() {
            let period = new_sched(*block as usize);
            affected_blocks.insert(*block);
            let Some(comp_id) = self.node_map.get(block) else {
                continue;
            };
            affected_components.insert(*comp_id);
            let component = &self.components[comp_id];
            for neighbor in component.neighbors.iter() {
                let neighbor_comp = &self.components[neighbor];
                if neighbor_comp.period == period {
                    affected_components.insert(*neighbor);
                }
            }
        }

        let changed_periods = changes
            .iter()
            .flat_map(|i| [old_sched(*i as usize), new_sched(*i as usize)])
            .collect::<ahash::HashSet<_>>();
        self.components.iter().for_each(|(comp_id, component)| {
            if changed_periods.contains(&component.period) {
                affected_components.insert(*comp_id);
            }
        });

        // 2. Remove edges between affected components.
        let all_neighbors = affected_components
            .iter()
            .flat_map(|comp_id| {
                let component = &self.components[comp_id];
                component.neighbors.iter().copied()
            })
            .collect_vec();
        all_neighbors.iter().for_each(|comp_id| {
            let component = self.components.get_mut(comp_id).unwrap();
            component
                .neighbors
                .retain(|neighbor_id| !affected_components.contains(neighbor_id));
        });

        // 3. Collect all blocks in affected components.
        affected_blocks.extend(affected_components.iter().flat_map(|comp_id| {
            let component = &self.components[comp_id];
            component.nodes.iter().copied()
        }));

        let old_min_max_z = affected_blocks
            .iter()
            .filter_map(|block| {
                let comp_id = self.node_map.get(block)?;
                let component = &self.components[comp_id];
                Some((*block, (component.min_z, component.max_z)))
            })
            .collect::<ahash::AHashMap<u32, (i32, i32)>>();

        // 4. Remove affected components.
        affected_components.iter().for_each(|comp_id| {
            self.components.remove(comp_id);
        });

        // 5. Rebuild affected components.
        affected_blocks.iter().for_each(|block| {
            self.node_map.remove(block);
        });

        let mut curr_comp_id = Uuid::new_v4();
        let mut new_comp_ids = vec![];
        affected_blocks.iter().for_each(|block| {
            // If the block is already assigned to a component, skip it.
            if self.node_map.contains_key(block) || new_sched(*block as usize) == self.mine_life {
                return;
            }

            // Create a walker.
            let mut discovered = ahash::HashSet::default();
            let mut stack = VecDeque::new();
            let mut walker =
                DualWalkerWithPredicate::new(*block, &mut stack, &mut discovered, |i: u32| {
                    let should_include = new_sched(i as usize) == new_sched(*block as usize);

                    if should_include && self.node_map.contains_key(&i) {
                        panic!("Block already assigned to a component");
                    }

                    should_include
                });

            // Collect the nodes in the component.
            let mut component_nodes = ahash::HashSet::default();
            while let Some(node) = walker.next(preds, succs) {
                component_nodes.insert(node);
            }

            let mut min_z = i32::MAX;
            let mut max_z = i32::MIN;

            for node in &component_nodes {
                self.node_map.insert(*node, curr_comp_id);
                let z = locs[*node as usize][2];
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }

            self.components.insert(
                curr_comp_id,
                ConnectedComponent {
                    nodes: component_nodes,
                    neighbors: Default::default(),
                    min_z,
                    max_z,
                    period: new_sched(*block as usize),
                },
            );
            new_comp_ids.push(curr_comp_id);

            curr_comp_id = Uuid::new_v4();
        });

        // 6. Recompute edges between components.
        let mut neighbors = ahash::HashSet::default();
        for comp_id in new_comp_ids.iter() {
            let component = &self.components[comp_id];
            let starts = component.nodes.iter().copied().collect_vec();
            let mut stack = VecDeque::new();
            let mut discovered = ahash::HashSet::default();
            let mut walker = DualWalkerWithPredicate::new_with_many(
                &starts,
                &mut stack,
                &mut discovered,
                |i: u32| {
                    let should_continue = new_sched(i as usize) == component.period;

                    if new_sched(i as usize) != component.period {
                        // Check outgoing edges
                        let Some(other_comp_id) = self.node_map.get(&i) else {
                            return should_continue;
                        };
                        neighbors.insert((*comp_id, other_comp_id));
                    }

                    should_continue
                },
            );

            while let Some(_node) = walker.next(preds, succs) {
                // Do nothing
            }
        }

        for (comp_a, comp_b) in neighbors.iter() {
            self.components
                .get_mut(comp_a)
                .unwrap()
                .neighbors
                .insert(**comp_b);
            self.components
                .get_mut(*comp_b)
                .unwrap()
                .neighbors
                .insert(*comp_a);
        }

        // 7. Return affected blocks that had min/max z changes.
        let mut out = affected_blocks.into_iter().collect_vec();
        out.retain(|block| {
            let Some(comp_id) = self.node_map.get(block) else {
                return true;
            };
            let component = &self.components[comp_id];
            let Some((old_min_z, old_max_z)) = old_min_max_z.get(block) else {
                return true;
            };

            let old_delta = old_max_z - old_min_z;
            let new_delta = component.max_z - component.min_z;
            old_delta != new_delta
        });

        out
    }
}

/// WIP: Sink Rate Constraint
#[allow(dead_code)]
struct SinkRateConstraint<'a> {
    penalty_cost: f64,
    sink_rate: u32,
    preds: &'a RelationProvider,
    succs: &'a RelationProvider,
    unit_penalties: IdxPeriodMap<AtomicI32>,
    agg_penalties: Vec<AtomicI32>,
    locs: &'a [[i32; 3]],
    mine_life: u8,
    connected_components: ConnectedComponents,
}

#[allow(dead_code)]
impl<'a> SinkRateConstraint<'a> {
    pub fn new(
        penalty_cost: f64,
        sink_rate: u32,
        preds: &'a RelationProvider,
        succs: &'a RelationProvider,
        locs: &'a [[i32; 3]],
        mine_life: u8,
    ) -> SinkRateConstraint<'a> {
        SinkRateConstraint {
            penalty_cost,
            sink_rate,
            preds,
            succs,
            unit_penalties: Default::default(),
            agg_penalties: vec![],
            locs,
            mine_life,
            connected_components: ConnectedComponents::new(mine_life),
        }
    }
}

impl<'a> GeometricConstraint for SinkRateConstraint<'a> {
    fn penalty_delta(&self, idx: PertIdx) -> f64 {
        match idx {
            PertIdx::Aggregate(i) => {
                self.penalty_cost * self.agg_penalties[i].load(Ordering::Relaxed) as f64
            }
            PertIdx::Unit(idx, period) => {
                self.penalty_cost
                    * self
                        .unit_penalties
                        .at((idx, period))
                        .load(Ordering::Relaxed) as f64
            }
        }
    }

    #[inline(always)]
    fn curr_val(&self, _sched: &[u8]) -> f64 {
        self.connected_components
            .components
            .values()
            .map(|comp| {
                let delta = comp.max_z - comp.min_z;
                if delta > self.sink_rate as i32 && comp.period != self.mine_life {
                    (delta - self.sink_rate as i32) * comp.nodes.len() as i32
                } else {
                    0
                }
            })
            .sum::<i32>() as f64
    }

    fn init<T: PerturbationSummary + Send + Sync>(
        &mut self,
        sched: &[u8],
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        aggregate_perts: &AggregatePerturbationCollection<T>,
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
    ) {
        println!("Init Sink Rate");

        self.unit_penalties = IdxPeriodMap::default();
        unit_perts.into_iter().for_each(|(idx, _pert)| {
            self.unit_penalties.insert(idx, AtomicI32::new(0));
        });

        self.agg_penalties = vec![];
        for _ in 0..aggregate_perts.len() {
            self.agg_penalties.push(AtomicI32::new(0));
        }

        // 1. Compute connected components.
        self.connected_components
            .build(sched, self.preds, self.succs, self.locs);

        println!("Current penalty: {}", self.curr_val(sched));

        let old = self
            .connected_components
            .components
            .values()
            .map(|comp| {
                let delta = comp.max_z - comp.min_z;
                if delta > self.sink_rate as i32 && comp.period != self.mine_life {
                    (delta - self.sink_rate as i32) * comp.nodes.len() as i32
                } else {
                    0
                }
            })
            .sum::<i32>();

        // 2. For each perturbation, compute the sink rate penalty delta.
        let len = unit_perts.len();
        unit_perts
            .par_iter()
            .progress_count(len as u64)
            .for_each_with(
                self.connected_components.clone(),
                |cc, ((_node, _period), perturbation)| {
                    let perturbation = perturbation.lock();

                    let changes = perturbation
                        .effected_it(|i| sched[i as usize], flat_preds, flat_succs)
                        .map(|(i, _)| i)
                        .collect::<ahash::HashSet<_>>();

                    let sched_fn = |i: usize| {
                        if changes.contains(&(i as u32)) {
                            perturbation.period()
                        } else {
                            sched[i]
                        }
                    };
                    cc.process_change(
                        &changes,
                        &sched_fn,
                        &|i| sched[i],
                        self.preds,
                        self.succs,
                        self.locs,
                    );

                    let new = cc
                        .components
                        .values()
                        .map(|comp| {
                            let delta = comp.max_z - comp.min_z;
                            if delta > self.sink_rate as i32 && comp.period != self.mine_life {
                                (delta - self.sink_rate as i32) * comp.nodes.len() as i32
                            } else {
                                0
                            }
                        })
                        .sum::<i32>();

                    cc.process_change(
                        &changes,
                        &|i| sched[i],
                        &sched_fn,
                        self.preds,
                        self.succs,
                        self.locs,
                    );

                    self.unit_penalties
                        .at((perturbation.block_index(), perturbation.period()))
                        .store(new - old, Ordering::Relaxed);
                },
            );
    }
    fn process_change<T: PerturbationSummary + Send + Sync>(
        &mut self,
        new_sched: &[u8],
        old_sched: &[u8],
        changed_blocks: &[u32],
        flat_preds: &RelationRangeProvider,
        flat_succs: &RelationRangeProvider,
        unit_perts: &IdxPeriodMap<Mutex<BlockPerturbation<T>>>,
        _aggregate_perts: &AggregatePerturbationCollection<T>,
    ) {
        println!("Process Sink Rate Change");
        let effected_blocks = self.connected_components.process_change(
            &changed_blocks.iter().copied().collect(),
            &|i| new_sched[i],
            &|i| old_sched[i],
            self.preds,
            self.succs,
            self.locs,
        );

        let old = self
            .connected_components
            .components
            .values()
            .map(|comp| {
                let delta = comp.max_z - comp.min_z;
                if delta > self.sink_rate as i32 && comp.period != self.mine_life {
                    (delta - self.sink_rate as i32) * comp.nodes.len() as i32
                } else {
                    0
                }
            })
            .sum::<i32>();

        println!("Old sink rate penalty: {}", old);

        let to_check = effected_blocks
            .iter()
            .flat_map(|b| (0..self.mine_life + 1).map(move |p| (*b, p)))
            .filter(|(b, p)| unit_perts.contains((*b, *p)))
            .collect::<Vec<_>>();

        to_check.par_iter().progress().for_each_with(
            self.connected_components.clone(),
            |cc, (node, period)| {
                let perturbation = unit_perts.at((*node, *period)).lock();

                let changes = perturbation
                    .effected_it(|i| new_sched[i as usize], flat_preds, flat_succs)
                    .map(|(i, _)| i)
                    .collect::<ahash::HashSet<_>>();

                let sched_fn = |i: usize| {
                    if changes.contains(&(i as u32)) {
                        perturbation.period()
                    } else {
                        new_sched[i]
                    }
                };

                cc.process_change(
                    &changes,
                    &sched_fn,
                    &|i| new_sched[i],
                    self.preds,
                    self.succs,
                    self.locs,
                );

                let new = cc
                    .components
                    .values()
                    .map(|comp| {
                        let delta = comp.max_z - comp.min_z;
                        if delta > self.sink_rate as i32 && comp.period != self.mine_life {
                            (delta - self.sink_rate as i32) * comp.nodes.len() as i32
                        } else {
                            0
                        }
                    })
                    .sum::<i32>();

                cc.process_change(
                    &changes,
                    &|i| new_sched[i],
                    &sched_fn,
                    self.preds,
                    self.succs,
                    self.locs,
                );

                self.unit_penalties
                    .at((perturbation.block_index(), perturbation.period()))
                    .store(new - old, Ordering::Relaxed);
            },
        );
    }
}
