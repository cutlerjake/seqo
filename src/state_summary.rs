use crate::prelude::{ArrayState, VecState};

pub trait GenericState {
    fn new(mine_life: u8) -> Self;
}

pub trait GenericStateModifier<T> {
    fn add_delta_to_state(&self, state: &mut T);
}

impl<const N: usize, T> GenericState for ArrayState<N, T>
where
    T: Default + Copy,
{
    fn new(mine_life: u8) -> Self {
        assert_eq!(N as u8, mine_life);
        Self {
            periods: [T::default(); N],
        }
    }
}

impl<T> GenericState for VecState<T>
where
    T: Default + Clone,
{
    fn new(mine_life: u8) -> Self {
        Self {
            periods: vec![T::default(); mine_life as usize],
        }
    }
}
