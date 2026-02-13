use log::info;
use mimalloc::MiMalloc;
use rand::Rng;
use seqo::{init_opt_logger, prelude::*};
use seqo_derive::{AddAssign, BlockDelta, seqo_log};

/// Number of periods in the McLaughlin mine scheduling problem.
const MINE_LIFE: u8 = 8;

/// Use mimalloc as the global allocator for better performance.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// A simple structure representing a block in the McLaughlin mine model.
#[derive(Copy, Clone, Default, Debug)]
pub struct MclaughlinBlock {
    /// Tonnage of ore in the block.
    pub ore_tonnage: i32,
    /// Total tonnage of the block.
    pub total_tonnage: i32,
    /// Undiscounted value of the block.
    pub udisc_value: i32,
}

/// A summary of the material mined in a single period.
///
/// This structure uses the `BlockDelta` derive macro to automatically
/// generate methods for adding and subtracting block contributions.
#[derive(Copy, Clone, Default, PartialEq, Debug, BlockDelta, AddAssign)]
#[block_delta(block = MclaughlinBlock)]
pub struct MclaughlinAggregateSummary {
    /// Total ore tonnage mined in period.
    #[block_delta(if block.udisc_value > 0 { block.ore_tonnage } else { 0 })]
    pub ore_tonnage: i32,
    /// Total tonnage mined in period.
    #[block_delta(block.total_tonnage)]
    pub total_tonnage: i32,
    /// Total undiscounted value mined in period.
    #[block_delta(block.udisc_value)]
    pub udisc_value: i32,
}

/// In this example, we use an array-based state summary since the mine life is known and small.
///
/// We add one extra period to the state summary because unmined material is stored in period `MINE_LIFE`.
type MclaughlinPeriodSummary = ArrayState<{ MINE_LIFE as usize + 1 }, MclaughlinAggregateSummary>;

/// Type alias for the McLaughlin mine using the default mine structure.
type MclaughlinMine = DefaultMine<MclaughlinBlock>;

/// Objective function for the McLaughlin mine scheduling problem.
#[derive(Clone)]
pub struct MclaughlinObjectiveFunction {
    /// Lower bound on ore tonnage per period.
    pub ore_lb: i64,
    /// Upper bound on ore tonnage per period.
    pub ore_ub: i64,
    /// Penalty per unit of ore tonnage outside the bounds.
    pub ore_delta_penalty: i64,

    /// Lower bound on total tonnage per period.
    pub total_lb: i64,
    /// Upper bound on total tonnage per period.
    pub total_ub: i64,
    /// Penalty per unit of total tonnage outside the bounds.
    pub total_delta_penalty: i64,

    /// Discount factors for each period.
    pub discounts: Vec<f64>,
}

/// Helper function to compute lower bound penalty.
#[inline(always)]
fn lb_penalty(value: i64, lb: i64, penalty_per_unit: i64) -> i64 {
    if value < lb {
        (lb - value) * penalty_per_unit
    } else {
        0
    }
}

/// Helper function to compute upper bound penalty.
#[inline(always)]
fn ub_penalty(value: i64, ub: i64, penalty_per_unit: i64) -> i64 {
    if value > ub {
        (value - ub) * penalty_per_unit
    } else {
        0
    }
}

/// Implement the objective function trait for the McLaughlin objective function.
///
/// This implementation calculates the total objective value based on the period summaries,
/// applying penalties for ore and total tonnage that fall outside specified bounds.
///
/// `#[seqo_log]` attribute is used to automatically generate logging for the state value computation.
#[seqo_log]
impl GenericObjectiveFunction<MclaughlinPeriodSummary> for MclaughlinObjectiveFunction {
    /// This example doesn't require any workspace.
    type WorkSpace = ();

    fn state_value(
        &self,
        state: &MclaughlinPeriodSummary,
        _workspace: &mut Self::WorkSpace,
    ) -> f64 {
        // 1. Initialize output value.
        let mut out = 0;

        // 2. Iterate over each period to compute contributions to the objective value.
        for i in 0..MINE_LIFE {
            // a. Retrieve the summary for the current period.
            let summary = &state.periods[i as usize];

            // b. Get the discount factor for the current period.
            let discount = self.discounts[i as usize];

            slog!(
                ty = i64,
                field = ore_tonnage,
                expr = summary.ore_tonnage as i64
            );
            slog!(
                ty = i64,
                field = total_tonnage,
                expr = summary.total_tonnage as i64
            );

            // c. Calculate penalties for ore and total tonnage bounds.
            let ore_lb_penalty = lb_penalty(
                summary.ore_tonnage as i64,
                self.ore_lb,
                self.ore_delta_penalty,
            );
            slog!(ty = i64, expr = ore_lb_penalty);

            let ore_ub_penalty = ub_penalty(
                summary.ore_tonnage as i64,
                self.ore_ub,
                self.ore_delta_penalty,
            );
            slog!(ty = i64, expr = ore_ub_penalty);

            let total_lb_penalty = lb_penalty(
                summary.total_tonnage as i64,
                self.total_lb,
                self.total_delta_penalty,
            );
            slog!(ty = i64, expr = total_lb_penalty);

            let total_ub_penalty = ub_penalty(
                summary.total_tonnage as i64,
                self.total_ub,
                self.total_delta_penalty,
            );
            slog!(ty = i64, expr = total_ub_penalty);

            // d. Update the output value with the discounted contribution from the current period.
            out += ((summary.udisc_value as i64
                - ore_lb_penalty
                - ore_ub_penalty
                - total_lb_penalty
                - total_ub_penalty) as f64
                * discount) as i64;
            slog!(ty = i64, expr = out);
        }

        // 3. Return the final computed objective value.
        out as f64
    }
}

fn main() {
    // 1. Initialize logger for progress output.
    init_opt_logger();

    // 2. Read in blocks from the data file.
    info!("Reading blocks...");
    let mut raw_blocks = Vec::new();
    let mut locs = vec![];
    let mut block_rdr = csv::ReaderBuilder::new()
        .delimiter(b' ')
        .from_path(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/data/mclaughlin_limit.blocks"),
        )
        .unwrap();

    for record in block_rdr.records() {
        let record = record.unwrap();
        let x = record[1].parse::<i64>().unwrap();
        let y = record[2].parse::<i64>().unwrap();
        let z = record[3].parse::<i64>().unwrap();

        locs.push([x as i32, y as i32, z as i32]);
        let udisc_value = record[4].parse::<f64>().unwrap() as i32;

        let total_tonnage = record[5].parse::<f64>().unwrap() as i32;
        let ore_tonnage = if udisc_value > 0 { total_tonnage } else { 0 };

        raw_blocks.push(MclaughlinBlock {
            udisc_value,
            ore_tonnage,
            total_tonnage,
        })
    }

    // 3. Build relations based on block locations.
    info!("Building relations...");
    let preds = Circle::build_pred_relations(&locs, [25.0, 25.0, 20.0], 45.0f32.to_radians(), 8);
    let succs = preds.invert();

    // 4. Create the mine.
    let mine = MclaughlinMine::new(raw_blocks.clone(), preds.clone(), succs.clone());

    // 5. Generate perturbations.
    //
    // We randomly sample 10% of (block, period) pairs to create perturbations for.
    info!("Generating perturbations...");
    let perts = (0..raw_blocks.len())
        .flat_map(|b| {
            let mut rng = rand::rng();
            (0..MINE_LIFE).filter_map(move |p| rng.random_bool(0.1).then_some((b as u32, p)))
        })
        .collect::<Vec<_>>();

    // 6. Define the objective function.
    info!("Defining objective function...");
    let of = MclaughlinObjectiveFunction {
        ore_lb: 0,
        ore_ub: 3_300_000,
        ore_delta_penalty: 1_000,
        total_lb: 0,
        total_ub: 100_000_000,
        total_delta_penalty: 1_000,
        discounts: (0..MINE_LIFE)
            .map(|i| (1.0f64 + 0.1).powi(-(i as i32)))
            .collect::<Vec<_>>(),
    };

    // 7. Create the single mine wrapper for scheduling.
    info!("Creating single-mine wrapper...");
    let mut gmine = SingleMine::<_, MclaughlinPeriodSummary, _, _>::new(
        &mine,
        MINE_LIFE,
        (), // No geometric constraints in example.
        perts.iter().copied(),
        (),                       // No extra context needed in example.
        PerturbationScaling::Raw, // No perturbation scaling in example.
    );

    // 8. Create the multi-mine scheduler.
    info!("Creating scheduler...");
    let mut grasp_scheduler = MultiMineScheduleOptimizer::new((&mut gmine,), of.clone(), MINE_LIFE);

    // 9. Optimize the schedule.
    info!("Optimizing schedule...");
    grasp_scheduler.optimize(10000, None);

    let logged_final_state = StateSummary::new(&of, grasp_scheduler.state_summary(), &mut ());
    println!("{:#?}", logged_final_state);
}
