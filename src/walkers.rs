//! Walkers over the graph with external buffers for stack and discovered nodes.
//!
//! These walkers do not allocate any memory on their own, making them suitable for
//! performance-critical paths where allocations are undesirable.

use std::collections::VecDeque;

use crate::relation_provider::RelationProvider;

/// Direction for fixing periods during a perturbation walk.
#[derive(Copy, Clone)]
pub enum FixDir {
    ///  Fix to predecessors (i.e., new period must be less than or equal to current).
    Preds(u8),

    /// Fix to successors (i.e., new period must be greater than or equal to current).
    Succs(u8),
}

impl FixDir {
    /// Create a new `FixDir` based on the current and new periods.
    #[inline(always)]
    pub fn new(curr_period: u8, new_period: u8) -> Self {
        if new_period < curr_period {
            Self::Preds(new_period)
        } else {
            Self::Succs(new_period)
        }
    }

    /// Check if the current period is valid according to the fixing direction.
    #[inline(always)]
    pub fn is_valid(&self, period: u8) -> bool {
        match self {
            FixDir::Preds(curr) => period <= *curr,
            FixDir::Succs(curr) => period >= *curr,
        }
    }

    /// Check validity of multiple periods in a chunked manner.
    #[inline(always)]
    pub fn chunked_is_valid(&self, periods: [u8; 8]) -> [bool; 8] {
        match *self {
            FixDir::Preds(curr) => [
                periods[0] <= curr,
                periods[1] <= curr,
                periods[2] <= curr,
                periods[3] <= curr,
                periods[4] <= curr,
                periods[5] <= curr,
                periods[6] <= curr,
                periods[7] <= curr,
            ],
            FixDir::Succs(curr) => [
                periods[0] >= curr,
                periods[1] >= curr,
                periods[2] >= curr,
                periods[3] >= curr,
                periods[4] >= curr,
                periods[5] >= curr,
                periods[6] >= curr,
                periods[7] >= curr,
            ],
        }
    }
}

/// A walker that uses external buffers for stack and discovered nodes.
pub struct WalkerWithBuffer<'a> {
    /// The queue of nodes to visit
    pub stack: &'a mut VecDeque<u32>,
    /// The map of discovered nodes
    pub discovered: &'a mut Vec<bool>,
}

impl<'a> WalkerWithBuffer<'a> {
    /// Create a new `WalkerWithBuffer`.
    ///
    /// # Arguments
    /// * `start` - The starting node for the walk.
    /// * `stack` - External buffer for the stack of nodes to visit.
    /// * `discovered` - External buffer for tracking discovered nodes.
    pub fn new(start: u32, stack: &'a mut VecDeque<u32>, discovered: &'a mut Vec<bool>) -> Self {
        stack.clear();
        stack.push_back(start);

        discovered.iter_mut().for_each(|v| *v = false);
        discovered[start as usize] = true;

        Self { stack, discovered }
    }

    /// Create a new `WalkerWithBuffer` with multiple starting nodes.
    ///
    /// # Arguments
    /// * `starts` - The starting nodes for the walk.
    /// * `stack` - External buffer for the stack of nodes to visit.
    /// * `discovered` - External buffer for tracking discovered nodes.
    pub fn new_with_many(
        starts: &[u32],
        stack: &'a mut VecDeque<u32>,
        discovered: &'a mut Vec<bool>,
    ) -> Self {
        stack.clear();

        discovered.iter_mut().for_each(|v| *v = false);
        for start in starts.iter() {
            stack.push_back(*start);
            discovered[*start as usize] = true;
        }

        Self { stack, discovered }
    }

    /// Get the next node in the walk.
    ///
    /// # Arguments
    /// * `node_provider` - The relation provider for accessing node relations.
    pub fn next(&mut self, node_provider: &RelationProvider) -> Option<u32> {
        if let Some(node) = self.stack.pop_front() {
            for node in node_provider.relations(node as usize) {
                let discovered = unsafe { self.discovered.get_unchecked_mut(*node as usize) };
                if !*discovered {
                    *discovered = true;

                    self.stack.push_back(*node);
                }
            }

            return Some(node);
        }
        None
    }
}

/// A dual walker that uses external buffers for stack and discovered nodes.
pub struct DualWalkerWithPredicate<'a, F> {
    /// The queue of nodes to visit
    pub stack: &'a mut VecDeque<u32>,
    /// The map of discovered nodes
    pub discovered: &'a mut ahash::HashSet<u32>,

    predicate: F,
}

impl<'a, F: FnMut(u32) -> bool> DualWalkerWithPredicate<'a, F> {
    /// Create a new `DualWalkerWithPredicate`.
    ///
    /// # Arguments
    /// * `start` - The starting node for the walk.
    /// * `stack` - External buffer for the stack of nodes to visit.
    /// * `discovered` - External buffer for tracking discovered nodes.
    /// * `predicate` - A predicate function to filter nodes.
    pub fn new(
        start: u32,
        stack: &'a mut VecDeque<u32>,
        discovered: &'a mut ahash::HashSet<u32>,
        predicate: F,
    ) -> Self {
        stack.clear();

        stack.push_back(start);

        discovered.clear();
        discovered.insert(start);

        Self {
            stack,
            discovered,
            predicate,
        }
    }

    /// Create a new `DualWalkerWithPredicate` with multiple starting nodes.
    ///
    /// # Arguments
    /// * `starts` - The starting nodes for the walk.
    /// * `stack` - External buffer for the stack of nodes to visit.
    /// * `discovered` - External buffer for tracking discovered nodes.
    /// * `predicate` - A predicate function to filter nodes.
    pub fn new_with_many(
        starts: &[u32],
        stack: &'a mut VecDeque<u32>,
        discovered: &'a mut ahash::HashSet<u32>,
        predicate: F,
    ) -> Self {
        stack.clear();

        discovered.clear();

        for start in starts.iter() {
            stack.push_back(*start);
            discovered.insert(*start);
        }

        Self {
            stack,
            discovered,
            predicate,
        }
    }

    /// Get the next node in the walk.
    ///
    /// # Arguments
    /// * `pred_provider` - The relation provider for accessing predecessor relations.
    /// * `succ_provider` - The relation provider for accessing successor relations.
    pub fn next(
        &mut self,
        pred_provider: &RelationProvider,
        succ_provider: &RelationProvider,
    ) -> Option<u32> {
        if let Some(node) = self.stack.pop_front() {
            for node in pred_provider
                .relations(node as usize)
                .iter()
                .chain(succ_provider.relations(node as usize))
            {
                if self.discovered.insert(*node) && (self.predicate)(*node) {
                    self.stack.push_back(*node);
                }
            }

            return Some(node);
        }
        None
    }
}
