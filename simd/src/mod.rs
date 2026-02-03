use std::{
    marker::PhantomData,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use itertools::izip;
use proc::{impl_binary_op, impl_binary_op_mut, impl_unary_op};
use pulp::Simd;
use wide::{CmpGt, CmpLt};

pub mod cross;
pub mod mask;
pub mod scaled_add;
pub use wide;
