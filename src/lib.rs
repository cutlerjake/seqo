#![doc = include_str!("../README.md")]

pub mod aggregate_perturbation;
pub mod block_perturbation;
pub mod circle;
pub mod geometric_constraint;
pub mod helpers;
mod idx_period_map;
pub mod local_search;
pub mod mine;
pub mod objective_function;
pub mod perturbation_summary;
pub mod recovery_curve;
pub mod relation_provider;
pub mod simd;
pub mod state_summary;
pub mod stockpile;
pub mod walkers;

pub use rayon;
pub use seqo_derive;
