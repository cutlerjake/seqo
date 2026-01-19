//! Objective function trait for evaluating states.
pub trait GenericObjectiveFunction<S>: Clone {
    type WorkSpace: Default + Send + Sync;
    fn state_value(&self, state: &S, workspace: &mut Self::WorkSpace) -> f64;
}
