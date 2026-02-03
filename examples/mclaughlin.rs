use log::info;
use mimalloc::MiMalloc;
use rand::Rng;
use seqo::{init_opt_logger, prelude::*};
use seqo_derive::{AddAssign, BlockDelta};

const MINE_LIFE: u8 = 9;

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

type MclaughlinPeriodSummary = ArrayState<{ MINE_LIFE as usize }, MclaughlinAggregateSummary>;
type MclaughlinMine = DefaultMine<MclaughlinBlock>;

#[derive(Clone)]
pub struct MclaughlinObjectiveFunction {
    pub ore_lb: i64,
    pub ore_ub: i64,
    pub ore_delta_penalty: i64,

    pub total_lb: i64,
    pub total_ub: i64,
    pub total_delta_penalty: i64,

    pub discounts: Vec<f64>,
}

#[inline(always)]
fn lb_penalty(value: i64, lb: i64, penalty_per_unit: i64) -> i64 {
    if value < lb {
        (lb - value) * penalty_per_unit
    } else {
        0
    }
}

#[inline(always)]
fn ub_penalty(value: i64, ub: i64, penalty_per_unit: i64) -> i64 {
    if value > ub {
        (value - ub) * penalty_per_unit
    } else {
        0
    }
}

impl GenericObjectiveFunction<MclaughlinPeriodSummary> for MclaughlinObjectiveFunction {
    type WorkSpace = ();

    fn state_value(
        &self,
        state: &MclaughlinPeriodSummary,
        _workspace: &mut Self::WorkSpace,
    ) -> f64 {
        let mut out = 0;

        for i in 0..MINE_LIFE - 1 {
            let summary = &state.periods[i as usize];
            let discount = self.discounts[i as usize];

            let ore_lb_penalty = lb_penalty(
                summary.ore_tonnage as i64,
                self.ore_lb,
                self.ore_delta_penalty,
            );

            let ore_ub_penalty = ub_penalty(
                summary.ore_tonnage as i64,
                self.ore_ub,
                self.ore_delta_penalty,
            );

            let total_lb_penalty = lb_penalty(
                summary.total_tonnage as i64,
                self.total_lb,
                self.total_delta_penalty,
            );

            let total_ub_penalty = ub_penalty(
                summary.total_tonnage as i64,
                self.total_ub,
                self.total_delta_penalty,
            );

            out += ((summary.udisc_value as i64
                - ore_lb_penalty
                - ore_ub_penalty
                - total_lb_penalty
                - total_ub_penalty) as f64
                * discount) as i64
        }

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
        MINE_LIFE - 1,
        (), // No geometric constraints in example.
        perts.iter().copied(),
        (),                       // No extra context needed in example.
        PerturbationScaling::Raw, // No perturbation scaling in example.
    );

    // 8. Create the multi-mine scheduler.
    info!("Creating scheduler...");
    let mut grasp_scheduler =
        MultiMineScheduleOptimizer::new((&mut gmine,), of.clone(), MINE_LIFE - 1);

    // 9. Optimize the schedule.
    info!("Optimizing schedule...");
    grasp_scheduler.optimize(10000, None);
}
