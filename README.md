# SeqO
 Library for modelling and optimizing precedence constrained production
 scheduling problems, considering multiple mines, geometric constraints,
 uncertainty, and non-linear transfer functions.

 The optimizer, `MultiMineScheduleOptimizer`, caches the effect each
 perturbation has on the current state, allowing for efficient evaluation
 of all perturbations during local search. After a perturbation is applied,
 only invalidated cached values are recomputed. The library leverages
 parallelism via the `rayon` crate to accelerate perturbation evaluation, and
 cache updates.

 Geometric constraints, such as mining width constraints, are implemented
 via the `GeometricConstraint` trait. These constraints, like perturbations,
 cache their effect on the state to allow for efficient evaluation during
 local search. When a perturbation is applied, only invalidated cached values
  are recomputed. The library provides efficient implementations of common
 geometric constraints used in production scheduling.

 # Getting started

 User of this library are required to define and implement the following items.

 - **Block Type:**
   This should be a simple structure composed of the data stored in each block.

 ```rust
 #[derive(Debug, Copy, Clone)]
 pub struct GoldBlock {
     // Index of block.
     pub idx: [i32; 3],
     // The amount of gold in grams contained in the block.
     pub gold_grams: f64,
     // The tonnage of the block.
     pub tonnage: f64,
 }
 ```

 - **Aggregate Summary:** A structure summarizing a flow of material for one period.

 ```rust
 #[derive(Debug, Copy, Clone, Default)]
 pub struct GoldAggregateSummary {
     // Total gold in grams to be sent to the mill.
     pub total_gold_grams_to_mill: f64,
     // Total tonnage to be sent to the mill.
     pub total_tonnage_to_mill: f64,
     // Total waste tonnage.
     pub waste_tonnage: f64,
 }
 ```

 - **Period Summary:** A structure summarizing the state of the mine for multiple periods.
     - This structure must implement `BlockPerturbationSummary`, `GenericState`, and `GenericStateModifier` traits.

 ```rust
 #[derive(Debug, Clone)]
 pub struct GoldPeriodSummary {
     // Summaries for each period.
     pub aggregates: Vec<GoldAggregateSummary>,
 }

 impl GoldPeriodSummary {
     // Cut-off grade for milling.
     //
     // Blocks with a grade below this value are considered waste.
     const CUT_OFF_GRADE: f64 = 0.5; // grams per ton
 }

 impl BlockPerturbationSummary for GoldPeriodSummary {
     // Associated block type.
     type Block = GoldBlock;
     // Associated context type.
     // We don't have any complex context in this example, so we use the unit type.
     type Context = ();

     // Creates a new empty `GoldPeriodSummary` with the specified number of periods.
     // We add one to `num_periods` because material not mined is stored it index `num_periods`.
     fn new_empty(num_periods: u8) -> Self {
        Self {
            aggregates: vec![GoldAggregateSummary::default(); num_periods as usize + 1],
        }
     }

     // Adds a block to the summary for the specified period.
     #[inline(always)]
     fn add_block(&mut self, period: u8, block: &Self::Block, _ctx: &Self::Context) {
         // Get the aggregate summary for the specified period.
         let agg = &mut self.aggregates[period as usize];
         // Calculate the grade of the block.
         let grade = block.gold_grams / block.tonnage;
         // Update the aggregate summary based on the block's grade.
         if grade >= Self::CUT_OFF_GRADE {
             agg.total_gold_grams_to_mill += block.gold_grams;
             agg.total_tonnage_to_mill += block.tonnage;
         } else {
             agg.waste_tonnage += block.tonnage;
         }
     }

     // Removes a block from the summary for the specified period.
     #[inline(always)]
     fn sub_block(&mut self, period: u8, block: &Self::Block, _ctx: &Self::Context) {
         // Get the aggregate summary for the specified period.
         let agg = &mut self.aggregates[period as usize];
         // Calculate the grade of the block.
         let grade = block.gold_grams / block.tonnage;
         // Update the aggregate summary based on the block's grade.
         if grade >= Self::CUT_OFF_GRADE {
             agg.total_gold_grams_to_mill -= block.gold_grams;
             agg.total_tonnage_to_mill -= block.tonnage;
         } else {
             agg.waste_tonnage -= block.tonnage;
         }
     }
 }

 impl GenericState for GoldPeriodSummary {
     // Creates a new empty `GoldPeriodSummary` with the specified number of periods.
     fn new(mine_num_periods: u8) -> Self {
         Self::new_empty(mine_num_periods)
     }
}

 impl GenericStateModifier<GoldPeriodSummary> for GoldPeriodSummary {
     // Adds the delta from `self` to the provided `state`.
     fn add_delta_to_state(&self, state: &mut Self) {
         // Iterate over each aggregate summary and add the values to the state.
         for (idx, agg) in self.aggregates.iter().enumerate() {
             let state_agg = &mut state.aggregates[idx];
             state_agg.total_gold_grams_to_mill += agg.total_gold_grams_to_mill;
             state_agg.total_tonnage_to_mill += agg.total_tonnage_to_mill;
             state_agg.waste_tonnage += agg.waste_tonnage;
         }
     }

     // Subtracts the delta from `self` from the provided `state`.
     fn sub_delta_from_state(&self, state: &mut Self) {
         // Iterate over each aggregate summary and subtract the values from the state.
         for (idx, agg) in self.aggregates.iter().enumerate() {
             let state_agg = &mut state.aggregates[idx];
             state_agg.total_gold_grams_to_mill -= agg.total_gold_grams_to_mill;
             state_agg.total_tonnage_to_mill -= agg.total_tonnage_to_mill;
             state_agg.waste_tonnage -= agg.waste_tonnage;
         }
     }
 }
 ```

 - **Objective Function:** A structure implementing the `ObjectiveFunction` trait to evaluate
   the value of a provided state.

 ```rust
 pub struct GoldObjectiveFunction {
     // Price of gold per gram.
     pub gold_price_per_gram: f64,
     // Recovery of gold as a fraction (0.0 to 1.0).
     pub recovery: f64,
     // Cost per ton milled.
     pub cost_per_ton_milled: f64,
     // Cost per ton of waste.
     pub cost_per_ton_waste: f64,
     // Period discounts.
     pub period_discounts: Vec<f64>,
 }

 impl ObjectiveFunction<GoldPeriodSummary> for GoldObjectiveFunction {
     // Associated workspace type.
     //
     // In this simple example, we don't need any additional workspace, so we use the unit type.
     type WorkSpace = ();

     // Evaluates the objective function for the given state.
     fn evaluate(&self, state: &GoldPeriodSummary, _ws: &mut Self::WorkSpace) -> f64 {
         let mut total_value = 0.0;
         for (period_idx, agg) in state.aggregates.iter().enumerate() {
             // Get the discount for the current period.
             let discount = *self.period_discounts[period_idx];
            // Calculate recovered gold, profits, and costs.
             let recovered_gold = agg.total_gold_grams_to_mill * self.recovery;
             let udsic_gold_profit = recovered_gold * self.gold_price_per_gram;
             let udsic_milling_cost = agg.total_tonnage_to_mill * self.cost_per_ton_milled;
             let udsic_waste_cost = agg.waste_tonnage * self.cost_per_ton_waste;
             // Calculate the value for the current period.
             let period_value = discount * (udsic_gold_profit - udsic_milling_cost - udsic_waste_cost);
             // Accumulate the total value.
             total_value += period_value;
         }
         total_value
     }
 }
 ```

 We are now ready to **optimize** a schedule using the `MultiMineScheduleOptimizer`.

 ```rust
 // 1. Get the blocks (read from file, database, etc.)
 let blocks: Vec<GoldBlock> = vec![/* ... populate blocks ... */];

 // 2. Collect the block indexes for predecessor and successor relations.
 let block_idxs: Vec<[i32; 3]> = blocks.iter().map(|b| b.idx).collect();

 // 3. Create relation providers for predecessors and successors.
 let block_size = [10, 10, 10]; // Example block size in each dimension.
 let preds = Circle::build_pred_relations(&block_idxs, &block_size, 5);
 let succs = preds.invert();

 // 4. Create a mine instance.
 let mine = DefaultMine::new(blocks, preds, succs);

 // 5. Create objective function instance.
 let objective_function = GoldObjectiveFunction {
    ... // Initialize parameters
 };

 // 6. Define the perturbations to consider.
 let mine_life = 10; // Example mine life in periods.
 let mut perturbations = vec![];
 for block_idx in 0..mine.num_blocks() {
     for period in 0..mine_life {
         perturbations.push((block_idx as u32, period as u8));
     }
 }

 // 7. Create the optimizer instance.
 let mut ls_mine = SingleMine::new(
     mine,
     mine_life as u8
     (), // Not geometric constraints in this example.
     perturbations.iter(),
     (), // No context needed.
     PerturbationScaling::Raw
 );

 let mut optimizer = MultiMineScheduleOptimizer::new(
     (ls_mine,),
     objective_function,
     mine_life as u8,
 );

 // 8. Optimize the schedule.
 optimizer.optimize_schedule(1000); // Run for 1000 iterations or until convergence.
