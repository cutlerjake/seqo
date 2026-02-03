//! Objective function trait for evaluating states.
pub trait GenericObjectiveFunction<S>: Clone {
    type WorkSpace: Default + Send + Sync;
    fn state_value(&self, state: &S, workspace: &mut Self::WorkSpace) -> f64;
}

pub trait SimplifiedWorkSpace {
    /// Prep the workspace for a new evaluation.
    fn prep_for_new_eval(&mut self);

    /// Prep the workspace to evaluate the next period.
    ///
    /// This is called at the end of each period iteration and
    /// should retain things like quantities in stockpiles.
    fn prep_for_next_period(&mut self);
}

pub trait SimplifiedObjectiveFunction<S>: Clone {
    type WorkSpace: SimplifiedWorkSpace + Default + Send + Sync;

    fn evaluate_period(&self, state: &S, workspace: &mut Self::WorkSpace, period: u8) -> f64;
    fn periods(&self) -> u8;
}

impl<T, S> GenericObjectiveFunction<S> for T
where
    T: SimplifiedObjectiveFunction<S>,
    S: Clone,
{
    type WorkSpace = <T as SimplifiedObjectiveFunction<S>>::WorkSpace;

    fn state_value(&self, state: &S, workspace: &mut Self::WorkSpace) -> f64 {
        let mut out = 0.0;
        workspace.prep_for_new_eval();

        for period in 0..self.periods() {
            out += SimplifiedObjectiveFunction::evaluate_period(self, state, workspace, period);
            workspace.prep_for_next_period();
        }

        out
    }
}
