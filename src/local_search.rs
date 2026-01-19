//! Cached local search for precedence constrained production scheduling.
//!
//! Each perturbation's influce on the objective function is precomputed and cached.
//! During local search, only the perturbations affected by a schedule change are updated.

use core::f64;
use std::{
    cmp::{max, min},
    fmt::Debug,
    fs::File,
    io::{self, Write},
};

use dashmap::DashSet;
use indicatif::ParallelProgressIterator;
use itertools::izip;
use log::info;
use parking_lot::Mutex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    aggregate_perturbation::{AggregatePerturbation, AggregatePerturbationCollection},
    block_perturbation::BlockPerturbation,
    geometric_constraint::GeometricConstraint,
    idx_period_map::{IdxPeriodMap, make_key},
    mine::Mine,
    objective_function::GenericObjectiveFunction,
    perturbation_summary::PerturbationSummary,
    relation_provider::RelationRangeProvider,
    state_summary::{GenericState, GenericStateModifier},
};

/// The scaling method used when evaluating perturbations.
pub enum PerturbationScaling {
    /// No scaling applied to perturbation values.
    Raw,
    /// Average the perturbation value by the number of affected blocks.
    PerBlock,
}

impl PerturbationScaling {
    /// Scale the perturbation value based on the selected scaling method.
    #[inline(always)]
    pub fn scale_value(&self, raw_value: f64, block_cnt: usize) -> f64 {
        match self {
            // No scaling applied to perturbation values.
            PerturbationScaling::Raw => raw_value,
            // Average the perturbation value by the number of affected blocks.
            PerturbationScaling::PerBlock => {
                // Avoid division by zero.
                let div = (block_cnt as f64).max(1.0);
                raw_value / div
            }
        }
    }
}

/// A single mine (of potentially many).
pub struct SingleMine<'a, M, P, G, C> {
    mine: &'a M,
    perturbations: IdxPeriodMap<Mutex<BlockPerturbation<P>>>,
    flattened_preds: RelationRangeProvider,
    flattened_succs: RelationRangeProvider,
    sched: Vec<u8>,
    prev_sched: Vec<u8>,
    mine_life: u8,
    geometric_constraints: G,
    aggregate_perturbations: AggregatePerturbationCollection<P>,
    scaling: PerturbationScaling,
    context: C,
}

impl<'a, M, P, G> SingleMine<'a, M, P, G, P::Context>
where
    M: Mine + Send + Sync,
    P: PerturbationSummary<Block = M::Block> + Sync + Send + PartialEq + Debug,
    G: GeometricConstraint + Send + Sync,
    P::Context: Send + Sync,
{
    /// Create a new mine without a schedule.
    pub fn new(
        mine: &'a M,
        mine_life: u8,
        geometric_constraints: G,
        perturbation_it: impl Iterator<Item = (u32, u8)>,
        context: P::Context,
        scaling: PerturbationScaling,
    ) -> Self {
        info!("Flattening preds");
        let flattened_preds = mine.get_pred_provider().flatten_to_ranges();
        for block in 0..flattened_preds.num_blocks() {
            let mut check = FxHashSet::default();
            assert!(
                flattened_preds.relations(block).all(|p| check.insert(p)),
                "Duplicate predecessor detected for block {}",
                block
            );
            assert!(flattened_preds.relations(block).any(|p| p == block as u32));
        }
        info!("Flattening succs");
        let flattened_succs = mine.get_succ_provider().flatten_to_ranges();
        for block in 0..flattened_succs.num_blocks() {
            let mut check = FxHashSet::default();
            assert!(
                flattened_succs.relations(block).all(|s| check.insert(s)),
                "Duplicate successor detected for block {}",
                block
            );

            assert!(flattened_succs.relations(block).any(|s| s == block as u32));
        }

        let mut perturbations = IdxPeriodMap::default();
        for (idx, period) in perturbation_it {
            perturbations.insert(
                (idx, period),
                Mutex::new(BlockPerturbation::new(idx, period, mine_life + 1)),
            );
        }

        Self {
            mine,
            perturbations,
            sched: Default::default(),
            prev_sched: Default::default(),
            mine_life,
            geometric_constraints,
            flattened_preds,
            flattened_succs,
            aggregate_perturbations: AggregatePerturbationCollection::new(vec![]),
            context,
            scaling,
        }
    }

    #[inline(always)]
    pub fn sched(&self) -> &[u8] {
        &self.sched
    }

    pub fn add_aggregate_perturbation(&mut self, perturbation: AggregatePerturbation<P>) {
        self.aggregate_perturbations
            .add_aggregate_perturbation(perturbation);
    }

    /// Initiliaze without a schedule.
    ///
    /// Schedule is initialized with all blocks unmined.
    pub fn init_no_sched<S: Send + Sync>(&mut self, state: &mut S, mine_life: u8)
    where
        P: GenericStateModifier<S>,
    {
        info!("Initializing with no schedule.");
        let sched = vec![mine_life; self.mine.num_blocks()];
        self.init_with_sched(sched, state, mine_life);
    }

    /// Initialize with a schedule.
    pub fn init_with_sched<S: Send + Sync>(&mut self, sched: Vec<u8>, state: &mut S, _mine_life: u8)
    where
        P: GenericStateModifier<S>,
    {
        info!("Initializing with schedule.");

        // 1. Add blocks to state.
        sched.iter().enumerate().for_each(|(block, period)| {
            let block = self.mine.get_block(block as u32);
            let mut p = P::new_empty(self.mine_life + 1);

            p.add_block(*period, block, &self.context);

            p.add_delta_to_state(state);
        });

        // 2. Store schedule.
        self.sched = sched;
        self.prev_sched = self.sched.clone();

        // 3. Initialize unit and aggregate perturbations.
        self.init_perturbations();
        self.init_aggregate_perturbations();

        // 4. Initialize geometric constraints.
        self.geometric_constraints.init(
            &self.sched,
            &self.perturbations,
            &self.aggregate_perturbations,
            &self.flattened_preds,
            &self.flattened_succs,
        );

        info!("Finished initialization");
    }

    /// Initializes preturbations.
    ///
    /// Computes and stores the state delta if perturbation applied.
    fn init_perturbations(&mut self) {
        info!("Computing unit perturbations.");
        let len = self.perturbations.len();
        self.perturbations
            .par_iter_mut()
            .progress_count(len as u64)
            .for_each(|(_, pert)| {
                pert.get_mut().compute_from_scratch(
                    self.mine.all_blocks(),
                    |b| self.sched[b as usize],
                    &self.flattened_preds,
                    &self.flattened_succs,
                    &self.context,
                );
            });
    }

    fn init_aggregate_perturbations(&mut self) {
        info!("Computing aggregate perturbations.");
        self.aggregate_perturbations
            .perturbations()
            .par_iter()
            .progress()
            .for_each_with(
                (self.sched.to_vec(), vec![]),
                |(mappable_sched, buffer), pert| {
                    pert.lock().compute(
                        self.mine.all_blocks(),
                        &self.sched,
                        mappable_sched,
                        &self.flattened_preds,
                        &self.flattened_succs,
                        &self.context,
                        buffer,
                    );

                    for idx in buffer.drain(..) {
                        mappable_sched[idx as usize] = self.sched[idx as usize];
                    }
                },
            );
    }

    /// Find the perturbation that causes the greatest positive change
    /// in the objective function.
    fn find_best_pert<S, OF>(
        &self,
        state_summary: &S,
        objective_function: &OF,
        curr_state_value: f64,
        curr_geom_penalty: f64,
    ) -> Option<(PertIdx, f64, f64)>
    where
        S: Clone + Send + Sync,
        OF: GenericObjectiveFunction<S> + Send + Sync,
        P: GenericStateModifier<S>,
    {
        info!("Finding best perturbation.");
        self.perturbations
            .par_iter()
            .map_init(
                || {
                    (
                        <OF as GenericObjectiveFunction<S>>::WorkSpace::default(),
                        state_summary.clone(),
                    )
                },
                |(work_space, state), ((idx, period), pert)| {
                    // 1. Get the perturbation.
                    let p = pert.lock();

                    // 2. Copmute the geometric penalty of the encumbent solution with the perturbation applied.
                    let geom_penalty = curr_geom_penalty
                        + self
                            .geometric_constraints
                            .penalty_delta(PertIdx::Unit(idx, period));

                    // 3. Compute the value of the incumbent solution with the perturbation applied.
                    p.delta_summary().add_delta_to_state(state); // Apply perturbation
                    let raw_value = objective_function.state_value(state, work_space);
                    state.clone_from(state_summary);

                    let raw_value_with_geom = raw_value - geom_penalty;

                    let scaled_value = self.scaling.scale_value(
                        raw_value_with_geom - (curr_state_value - curr_geom_penalty),
                        p.block_cnt,
                    );

                    (
                        PertIdx::Unit(idx, period),
                        raw_value,
                        geom_penalty,
                        scaled_value,
                    )
                },
            )
            .chain(
                self.aggregate_perturbations
                    .perturbations()
                    .par_iter()
                    .enumerate()
                    .map_init(
                        || {
                            (
                                <OF as GenericObjectiveFunction<S>>::WorkSpace::default(),
                                state_summary.clone(),
                            )
                        },
                        |(work_space, state), (idx, pert)| {
                            // 1. Get the perturbation.
                            let p = pert.lock();

                            // 2. Copmute the geometric penalty of the encumbent solution with the perturbation applied.
                            let geom_penalty = curr_geom_penalty
                                + self
                                    .geometric_constraints
                                    .penalty_delta(PertIdx::Aggregate(idx));

                            // 3. Compute the value of the incumbent solution with the perturbation applied.
                            p.delta_summary().add_delta_to_state(state); // Apply perturbation
                            let raw_value = objective_function.state_value(state, work_space);

                            state.clone_from(state_summary);
                            // p.delta_summary().sub_delta_from_state(state); // Remove perturbation

                            let raw_value_with_geom = raw_value - geom_penalty;

                            let scaled_value = self.scaling.scale_value(
                                raw_value_with_geom - (curr_state_value - curr_geom_penalty),
                                p.block_cnt,
                            );

                            (
                                PertIdx::Aggregate(idx),
                                raw_value,
                                geom_penalty,
                                scaled_value,
                            )
                        },
                    ),
            )
            .max_by(|(_, _, _, scaled_a), (_, _, _, scaled_b)| scaled_a.total_cmp(scaled_b))
            .map(|(pert_idx, raw_value, geom_penalty, _scaled_value)| {
                info!(
                    "Best pert: {:?} with value {}, geom_penalty: {}, scaled: {}",
                    pert_idx, raw_value, geom_penalty, _scaled_value
                );

                (pert_idx, raw_value, geom_penalty)
            })
    }

    /// Helper function for checking the validity of a state.
    ///
    /// Useful for teting algorithmic changes.
    /// Tests that deltas and states are correctly computed.
    #[allow(dead_code)]
    fn check_perts(&self) {
        for ((_idx, _period), unit_pert) in (&self.perturbations).into_iter() {
            let p = unit_pert.lock();
            let mut check_p: BlockPerturbation<P> =
                BlockPerturbation::new(p.block_index(), p.period(), self.mine_life + 1);

            check_p.compute_from_scratch(
                self.mine.all_blocks(),
                |b| self.sched[b as usize],
                &self.flattened_preds,
                &self.flattened_succs,
                &self.context,
            );

            if p.delta_summary() != check_p.delta_summary() {
                info!("--- CHECK ---");
                info!("{:#?}", check_p.delta_summary());

                info!("--- CACHED ---");
                info!("{:#?}", p.delta_summary());
                panic!()
            }

            if p.block_cnt != check_p.block_cnt {
                info!(
                    "Block cnt mismatch: cached: {}, computed: {}",
                    p.block_cnt, check_p.block_cnt
                );
                panic!()
            }
        }

        let mut mappable_sched = self.sched.to_vec();
        let mut buffer = vec![];
        for agg in self.aggregate_perturbations.perturbations().iter() {
            let p = agg.lock();
            let mut check_p =
                AggregatePerturbation::new(p.unit_perts().to_vec(), self.mine_life + 1);
            check_p.compute(
                self.mine.all_blocks(),
                &self.sched,
                &mut mappable_sched,
                &self.flattened_preds,
                &self.flattened_succs,
                &self.context,
                &mut buffer,
            );

            for idx in buffer.drain(..) {
                mappable_sched[idx as usize] = self.sched[idx as usize];
            }

            if p.delta_summary() != check_p.delta_summary() {
                info!("--- CHECK ---");
                info!("{:#?}", check_p.delta_summary());

                info!("--- CACHED ---");
                info!("{:#?}", p.delta_summary());
                panic!()
            }

            if p.block_cnt != check_p.block_cnt {
                info!(
                    "Block cnt mismatch: cached: {}, computed: {}",
                    p.block_cnt, check_p.block_cnt
                );
                panic!()
            }
        }
    }

    #[inline]
    fn apply_pert<S>(&mut self, perturbation: PertIdx, state_summary: &mut S) -> (Vec<u32>, Vec<u8>)
    where
        S: Send + Sync,
        P: GenericStateModifier<S>,
    {
        info!("Applying perturbation: {:?}", perturbation);
        match perturbation {
            PertIdx::Unit(idx, period) => {
                // Here we apply a unit perturbation, only changing the period of a single block.
                // a. Get the perturbation.
                let perturbation = self.perturbations.at((idx, period)).lock();
                let target_period = perturbation.period();

                // b. Apply the perturbation delta to the summary.
                perturbation
                    .delta_summary()
                    .add_delta_to_state(state_summary);

                // c. Walk the effected blocks to find all blocks that need to be updated.
                let mut changed_blocks = vec![];

                perturbation.walk_effected_blocks(
                    |i| self.sched[i as usize],
                    &self.flattened_preds,
                    &self.flattened_succs,
                    |block, _period| {
                        changed_blocks.push(block);
                    },
                );

                // d. drop the lock.
                drop(perturbation);

                // e. Update the schedule of all effected blocks.
                for block in &changed_blocks {
                    self.sched[*block as usize] = target_period;
                }

                // f. Sort the blocks.
                // TODO: I am node sure this is required anymore.
                // changed_blocks.sort();
                let changed_periods = vec![target_period; changed_blocks.len()];

                (changed_blocks, changed_periods)
            }
            PertIdx::Aggregate(idx) => {
                //Here we apply an aggregate perturbation, changing the period of multiple blocks.

                // a. Allocate vecs to track the state change.
                let mut changed_blocks = vec![];
                let mut changed_periods = vec![];

                // b. Get the aggregate perturbation.
                let perturbation = self.aggregate_perturbations.perturbations()[idx].lock();

                // c. Apply the aggregate perturbation to the state.
                perturbation
                    .delta_summary()
                    .add_delta_to_state(state_summary);

                // d. Apply the aggregate perturbation to the schedule.
                perturbation.apply_to_sched(
                    &mut self.sched,
                    &mut changed_blocks,
                    &mut changed_periods,
                    &self.flattened_preds,
                    &self.flattened_succs,
                );
                (changed_blocks, changed_periods)
            }
        }
    }

    #[inline]
    fn fix_cached_perts(&mut self, changed_blocks: &[u32]) {
        info!("Fixing cached perturbations.");

        // 1. Get all successors of the changed blocks.
        let mut succs = vec![];
        let mut succ_mask = FxHashSet::default();
        for block in changed_blocks {
            if succ_mask.contains(block) {
                continue;
            }
            for succ in self.flattened_succs.relations(*block as usize) {
                if !succ_mask.insert(succ) {
                    continue;
                }
                succs.push(succ);
            }
        }

        // 2. Get all predecessors of the changed blocks.
        let mut preds = vec![];
        let mut pred_mask = FxHashSet::default();
        for block in changed_blocks {
            if pred_mask.contains(block) {
                continue;
            }
            for pred in self.flattened_preds.relations(*block as usize) {
                if !pred_mask.insert(pred) {
                    continue;
                }
                preds.push(pred);
            }
        }

        // 3. Update all perturbations in successor set.
        let fixed_perts = DashSet::new();
        let aggs_to_recompute = DashSet::new();
        succs.par_iter().progress().for_each(|succ_id| {
            for p in 0..max(
                self.prev_sched[*succ_id as usize],
                self.sched[*succ_id as usize],
            ) {
                let key = make_key(*succ_id, p);

                if !fixed_perts.contains(&key) {
                    fixed_perts.insert(key);

                    static EMPTY: &[u32] = &[];
                    for agg_index in self
                        .aggregate_perturbations
                        .deps(*succ_id, p)
                        .unwrap_or(EMPTY)
                    {
                        if self.aggregate_perturbations.perturbations()[*agg_index as usize]
                            .lock()
                            .contains_pert(*succ_id, p)
                        {
                            aggs_to_recompute.insert(*agg_index);
                        }
                    }

                    let Some(mut pert) = self.perturbations.get((*succ_id, p)).map(|p| p.lock())
                    else {
                        continue;
                    };
                    pert.update(
                        self.mine.all_blocks(),
                        |b| self.sched[b as usize],
                        |b| self.prev_sched[b as usize],
                        &self.flattened_preds,
                        &self.flattened_succs,
                        &self.context,
                    );
                }
            }
        });

        // 4. Update all perturbation in predecessor set.
        preds.par_iter().for_each(|pred_id| {
            for p in min(
                self.prev_sched[*pred_id as usize],
                self.sched[*pred_id as usize],
            ) + 1..=self.mine_life
            {
                let key = make_key(*pred_id, p);
                if !fixed_perts.contains(&key) {
                    fixed_perts.insert(make_key(*pred_id, p));

                    static EMPTY: &[u32] = &[];
                    for agg_index in self
                        .aggregate_perturbations
                        .deps(*pred_id, p)
                        .unwrap_or(EMPTY)
                    {
                        if self.aggregate_perturbations.perturbations()[*agg_index as usize]
                            .lock()
                            .contains_pert(*pred_id, p)
                        {
                            aggs_to_recompute.insert(*agg_index);
                        }
                    }

                    let Some(mut pert) = self.perturbations.get((*pred_id, p)).map(|p| p.lock())
                    else {
                        continue;
                    };

                    pert.update(
                        self.mine.all_blocks(),
                        |b| self.sched[b as usize],
                        |b| self.prev_sched[b as usize],
                        &self.flattened_preds,
                        &self.flattened_succs,
                        &self.context,
                    );
                }
            }
        });

        // 5. Recompute all aggregate perturbations that were marked for recomputation.
        let len = aggs_to_recompute.len();
        aggs_to_recompute
            .par_iter()
            .progress_count(len as u64)
            .for_each_with(
                (self.sched.to_vec(), vec![]),
                |(mappable_sched, buffer), agg_index| {
                    let mut agg = self.aggregate_perturbations.perturbation(*agg_index).lock();

                    agg.compute(
                        self.mine.all_blocks(),
                        &self.sched,
                        mappable_sched,
                        &self.flattened_preds,
                        &self.flattened_succs,
                        &self.context,
                        buffer,
                    );
                    for idx in buffer.drain(..) {
                        mappable_sched[idx as usize] = self.sched[idx as usize];
                    }
                },
            );
    }

    #[inline]
    fn fix_cached_geometric_constraints(&mut self, changed_blocks: &[u32]) {
        info!("Fixing cached geometric constraints.");
        self.geometric_constraints.process_change(
            &self.sched,
            &self.prev_sched,
            changed_blocks,
            &self.flattened_preds,
            &self.flattened_succs,
            &self.perturbations,
            &self.aggregate_perturbations,
        );
    }

    #[inline(always)]
    fn sync_prev_sched(&mut self, changed_blocks: &[u32], changed_periods: &[u8]) {
        izip!(changed_blocks.iter(), changed_periods.iter())
            .for_each(|(b, p)| self.prev_sched[*b as usize] = *p);
    }

    /// Applies the selected perturbation.
    ///
    /// Updates the global state.
    /// Updates effected perturbation to account for change in global state.
    /// Updates geometric constraints as required by G.
    fn process_pert<S>(&mut self, perturbation: PertIdx, state_summary: &mut S)
    where
        S: Send + Sync,
        P: GenericStateModifier<S>,
    {
        // 1. Apply the perturbation to the state.
        let (changed_blocks, changed_periods) = self.apply_pert(perturbation, state_summary);

        if changed_blocks.is_empty() {
            match perturbation {
                PertIdx::Unit(block, period) => {
                    let p = self.perturbations.at((block, period)).lock();

                    println!("DELTA: {:?}", p.delta_summary())
                }
                PertIdx::Aggregate(idx) => {
                    let p = self.aggregate_perturbations.perturbation(idx as u32).lock();

                    println!("DELTA: {:?}", p.delta_summary())
                }
            };

            panic!("NO OP PERT");
        }

        // 2. Fix the cached perturbations.
        self.fix_cached_perts(&changed_blocks);
        // self.check_perts();

        // 3. Fix the cached geometric constaints.
        self.fix_cached_geometric_constraints(&changed_blocks);

        // 4. Sync the previous schedule
        self.sync_prev_sched(&changed_blocks, &changed_periods);
    }
}
/// Indexer for the perturbations of a [`SingleMine`].
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum PertIdx {
    /// Idx of a unit perturbation.
    Unit(u32, u8),
    /// Idx of an Aggregate perturbation.
    Aggregate(usize),
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct MultiMinePertIndex {
    mine: u8,
    pub pert: PertIdx,
    state_value: f64,
    geom_penalty: f64,
}

impl MultiMinePertIndex {
    pub fn total_value(&self) -> f64 {
        self.state_value - self.geom_penalty
    }
}
pub trait MultiMine<S, OF> {
    fn max_blocks(&self) -> usize;

    // Initialize all perturbations
    fn init(&mut self, state_summary: &mut S, mine_life: u8);

    // Get the current value of the state
    fn current_state_value(&self, state_summary: &S, of: &OF) -> f64;

    // Get the current geometric penalty of the state
    fn current_geometric_penalty(&self) -> f64;

    // Find perturbation with greatest value
    fn find_best_pert(
        &self,
        state_summary: &S,
        of: &OF,
        curr_state_value: f64,
        curr_geom_penalty: f64,
    ) -> MultiMinePertIndex;

    fn apply_pert(&mut self, pert_index: MultiMinePertIndex, state_summary: &mut S);
}

pub struct MultiMineScheduleOptimizer<M, S, OF> {
    multi_mine: M,
    state_summary: S,
    objective_function: OF,
}

impl<M, S, OF> MultiMineScheduleOptimizer<M, S, OF>
where
    M: MultiMine<S, OF> + Send + Sync,
    S: GenericState,
{
    pub fn new(mut multi_mine: M, of: OF, mine_life: u8) -> Self {
        let mut state_summary = S::new(mine_life + 1);
        multi_mine.init(&mut state_summary, mine_life);
        Self {
            multi_mine,
            state_summary,
            objective_function: of,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn optimize(
        &mut self,
        n_iters: usize,
        mut cb: Option<&mut dyn FnMut(usize, &S, f64, f64)>,
    ) {
        // let mut curr_val = self.objective_function.state_value(&self.state_summary);
        let t1 = std::time::Instant::now();
        println!("num threads: {}", rayon::current_num_threads());

        println!("Computing current value...");
        let mut curr_val = self
            .multi_mine
            .current_state_value(&self.state_summary, &self.objective_function);

        let mut curr_geom = self.multi_mine.current_geometric_penalty();

        let mut curr_total = curr_val - curr_geom;

        let mut perts = vec![];

        for i in 0..n_iters {
            println!("--- Iteration {} ---", i);
            let tf = std::time::Instant::now();
            let pert_index = self.multi_mine.find_best_pert(
                &self.state_summary,
                &self.objective_function,
                curr_val,
                curr_geom,
            );

            println!("find time: {}", tf.elapsed().as_secs_f64());

            println!("Value delta: {}", pert_index.state_value - curr_val);
            if pert_index.total_value() > curr_total {
                perts.push(pert_index);
                curr_val = pert_index.state_value;
                curr_geom = pert_index.geom_penalty;
                curr_total = pert_index.total_value();
                self.multi_mine
                    .apply_pert(pert_index, &mut self.state_summary);

                if i % 10 == 0
                    && let Some(cb) = cb.as_mut()
                {
                    cb(i, &self.state_summary, curr_val, curr_geom);
                }

                println!("{},{}", t1.elapsed().as_secs_f32(), curr_val);
            } else {
                break;
            }
        }

        let t2 = std::time::Instant::now();
        println!("Opt time: {}", (t2 - t1).as_secs_f32());

        write_vec_json("./pertidxs.json", &perts)
            .expect("Failed to write perturbation indices to JSON");
    }

    pub fn state_summary(&self) -> &S {
        &self.state_summary
    }
}

pub fn write_vec_json<T: Serialize>(path: &str, values: &Vec<T>) -> io::Result<()> {
    let json = serde_json::to_string_pretty(values).map_err(io::Error::other)?;

    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

pub fn read_vec_json<T: DeserializeOwned>(path: &str) -> io::Result<Vec<T>> {
    let file = File::open(path)?;
    let values = serde_json::from_reader(file).map_err(io::Error::other)?;
    Ok(values)
}

impl<'b, 'a, A, AB, AG, S, OF> MultiMine<S, OF> for (&'a mut SingleMine<'b, A, AB, AG, AB::Context>,)
where
    A: Mine + Send + Sync,
    AB: PerturbationSummary
        + Send
        + Sync
        + PerturbationSummary<Block = <A as Mine>::Block>
        + GenericStateModifier<S>
        + PartialEq
        + Debug,
    AG: GeometricConstraint + Send + Sync,
    OF: GenericObjectiveFunction<S> + Send + Sync,
    S: Clone + Send + Sync,
{
    fn max_blocks(&self) -> usize {
        self.0.mine.num_blocks()
    }

    fn init(&mut self, state_summary: &mut S, mine_life: u8) {
        self.0.init_no_sched(state_summary, mine_life);
    }

    fn find_best_pert(
        &self,
        state_summary: &S,
        of: &OF,
        curr_state_value: f64,
        curr_geom_penalty: f64,
    ) -> MultiMinePertIndex {
        let (idx, state_value, geom_penalty) = self
            .0
            .find_best_pert(state_summary, of, curr_state_value, curr_geom_penalty)
            .unwrap();
        MultiMinePertIndex {
            mine: 0,
            pert: idx,
            state_value,
            geom_penalty,
        }
    }

    fn current_state_value(&self, state_summary: &S, of: &OF) -> f64 {
        of.state_value(
            state_summary,
            &mut <OF as GenericObjectiveFunction<S>>::WorkSpace::default(),
        )
    }

    fn current_geometric_penalty(&self) -> f64 {
        self.0.geometric_constraints.curr_val(&self.0.sched)
    }

    fn apply_pert(&mut self, pert_index: MultiMinePertIndex, state_summary: &mut S) {
        self.0.process_pert(pert_index.pert, state_summary);
    }
}

impl<'a_mine, 'a, 'b_mine, 'b, A, AB, AG, B, BB, BG, S, OF> MultiMine<S, OF>
    for (
        &'a_mine mut SingleMine<'a, A, AB, AG, AB::Context>,
        &'b_mine mut SingleMine<'b, B, BB, BG, BB::Context>,
    )
where
    A: Mine + Send + Sync,
    AB: PerturbationSummary
        + Send
        + Sync
        + PerturbationSummary<Block = <A as Mine>::Block>
        + GenericStateModifier<S>
        + PartialEq
        + Debug,
    AG: GeometricConstraint + Send + Sync,
    B: Mine + Send + Sync,
    BB: PerturbationSummary
        + Send
        + Sync
        + PerturbationSummary<Block = <B as Mine>::Block>
        + GenericStateModifier<S>
        + PartialEq
        + Debug,
    BG: GeometricConstraint + Send + Sync,
    OF: GenericObjectiveFunction<S> + GenericObjectiveFunction<S> + Send + Sync,
    S: Clone + Send + Sync,
{
    fn max_blocks(&self) -> usize {
        self.0.mine.num_blocks().max(self.1.mine.num_blocks())
    }

    fn init(&mut self, state_summary: &mut S, mine_life: u8) {
        println!("Initializing Mine 1");
        self.0.init_no_sched(state_summary, mine_life);
        println!("Initializing Mine 2");
        self.1.init_no_sched(state_summary, mine_life);
    }

    fn find_best_pert(
        &self,
        state_summary: &S,
        of: &OF,
        curr_state_value: f64,
        curr_geom_penalty: f64,
    ) -> MultiMinePertIndex {
        println!("Finding best pert for mine 1");
        let t1 = std::time::Instant::now();
        let p0 = self
            .0
            .find_best_pert(state_summary, of, curr_state_value, curr_geom_penalty)
            .unwrap();
        let p0 = MultiMinePertIndex {
            mine: 0,
            pert: p0.0,
            state_value: p0.1,
            geom_penalty: p0.2,
        };
        println!(
            "Mine 1 found in {}s",
            (std::time::Instant::now() - t1).as_secs_f32()
        );
        println!("Finding best pert for mine 2");
        let t2 = std::time::Instant::now();
        let p1 = self
            .1
            .find_best_pert(state_summary, of, curr_state_value, curr_geom_penalty)
            .unwrap();
        let p1 = MultiMinePertIndex {
            mine: 1,
            pert: p1.0,
            state_value: p1.1,
            geom_penalty: p1.2,
        };
        println!(
            "Mine 2 found in {}s",
            (std::time::Instant::now() - t2).as_secs_f32()
        );

        [p0, p1]
            .into_iter()
            .max_by(|p1, p2| p1.state_value.partial_cmp(&p2.state_value).unwrap())
            .unwrap()
    }

    fn current_state_value(&self, state_summary: &S, of: &OF) -> f64 {
        <OF as GenericObjectiveFunction<S>>::state_value(
            of,
            state_summary,
            &mut <OF as GenericObjectiveFunction<S>>::WorkSpace::default(),
        )
    }

    fn current_geometric_penalty(&self) -> f64 {
        self.0.geometric_constraints.curr_val(&self.0.sched)
            + self.1.geometric_constraints.curr_val(&self.1.sched)
    }

    fn apply_pert(&mut self, pert_index: MultiMinePertIndex, state_summary: &mut S) {
        let MultiMinePertIndex {
            mine,
            pert,
            state_value: value,
            geom_penalty,
        } = pert_index;

        println!(
            "Applying pert {pert:?} to mine {mine} with value {}, and geom {}",
            value, geom_penalty
        );
        match mine {
            0 => self.0.process_pert(pert, state_summary),
            1 => self.1.process_pert(pert, state_summary),
            _ => unreachable!(),
        }

        // println!(
        //     "Apply time: {}s",
        //     (std::time::Instant::now() - t1).as_secs_f32()
        // );
        // println!("Value: {value}");
    }
}
