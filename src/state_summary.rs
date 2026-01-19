pub trait GenericState {
    fn new(mine_life: u8) -> Self;
}

pub trait GenericStateModifier<T> {
    fn add_delta_to_state(&self, state: &mut T);
}
