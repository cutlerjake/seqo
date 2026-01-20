/// Expand the block delta derive macro
use seqo_derive::BlockDelta;

#[derive(Debug, Copy, Clone)]
#[allow(unused)]
struct GoldBlock {
    // Index of block.
    idx: [i32; 3],
    // The amount of gold in grams contained in the block.
    gold_grams: f64,
    // The tonnage of the block.
    tonnage: f64,
}

impl GoldBlock {
    #[allow(unused)]
    fn grade(&self) -> f64 {
        self.gold_grams / self.tonnage
    }
}

#[derive(Debug, Copy, Clone, Default, BlockDelta)]
#[block_delta(block = GoldBlock)]
#[allow(unused)]
struct GoldAggregateSummary {
    // Total gold in grams to be sent to the mill.
    #[block_delta(if block.grade() >= 0.5 { block.gold_grams } else { 0.0 })]
    total_gold_grams_to_mill: f64,
    // Total tonnage to be sent to the mill.
    #[block_delta(if block.grade() >= 0.5 { block.tonnage } else { 0.0 })]
    total_tonnage_to_mill: f64,
    // Total waste tonnage.
    #[block_delta(if block.grade() < 0.5 { block.tonnage } else { 0.0 })]
    waste_tonnage: f64,
}
