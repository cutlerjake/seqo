//! Relation provider for managing block relations.
//!
//! This module defines a `RelationProvider` struct that efficiently stores and retrieves
//! block relations (e.g., predecessors and successors, mining width) using a compact representation.

use std::collections::VecDeque;

use indicatif::ParallelProgressIterator;
use rayon::iter::IntoParallelIterator;

use crate::walkers::WalkerWithBuffer;

/// A provider for block relations, such as predecessors or successors.
#[derive(Debug, Clone)]
pub struct RelationProvider<T = u32> {
    relations_starts: Vec<u32>,
    relations: Vec<T>,
}

impl<T> Default for RelationProvider<T> {
    fn default() -> Self {
        Self {
            relations_starts: Vec::new(),
            relations: Vec::new(),
        }
    }
}

impl<T: Copy + PartialOrd> RelationProvider<T> {
    /// Add a precedence constraint for a block.
    pub fn add_relations(&mut self, mut precedences: Vec<T>) {
        if self.relations_starts.is_empty() {
            self.relations_starts.push(0);
        }
        precedences.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        self.relations.extend(precedences);
        self.relations_starts.push(self.relations.len() as u32);
    }

    pub fn from_linked_list(linked_list: &[Vec<T>]) -> Self {
        let mut out = Self::default();
        for relations in linked_list {
            let mut relations = relations.clone();
            relations.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            out.add_relations(relations);
        }

        out
    }
}

impl<T: Copy> RelationProvider<T> {
    /// Convert the relation provider to a linked list representation.
    pub fn to_linked_list(&self) -> Vec<Vec<T>> {
        let mut linked_list = vec![vec![]; self.num_blocks()];

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.num_blocks() {
            linked_list[i].extend(self.relations(i));
        }

        linked_list
    }
}

impl<T> RelationProvider<T> {
    /// Returns a slice of all relations.
    #[inline(always)]
    pub fn all_relations(&self) -> &[T] {
        &self.relations
    }

    /// Returns the average number of relations per block.
    #[inline(always)]
    pub fn avg_relation_len(&self) -> u32 {
        (self.total_relations() as f64 / self.num_blocks() as f64).round() as u32
    }

    /// Returns the relations for a given block index.
    #[inline(always)]
    pub fn relations(&self, block_index: usize) -> &[T] {
        let start = self.relations_starts[block_index] as usize;
        let end = self.relations_starts[block_index + 1] as usize;
        self.relations[start..end].as_ref()
    }

    /// Returns the relations for a given block index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `block_index` is a valid index.
    #[inline(always)]
    pub unsafe fn relations_unchecked(&self, block_index: usize) -> &[T] {
        unsafe {
            let start = *self.relations_starts.get_unchecked(block_index) as usize;
            let end = *self.relations_starts.get_unchecked(block_index + 1) as usize;
            self.relations.get_unchecked(start..end)
        }
    }

    /// Returns the range of relations for a given block index.
    pub fn relation_range(&self, block_index: usize) -> (usize, usize) {
        let start = self.relations_starts[block_index] as usize;
        let end = self.relations_starts[block_index + 1] as usize;

        (start, end)
    }

    /// Returns the number of relations for a given block index.
    pub fn num_relations(&self, block_index: usize) -> usize {
        let start = self.relations_starts[block_index];
        let end = self.relations_starts[block_index + 1];
        (end - start) as usize
    }

    /// Returns the total number of relations.
    pub fn total_relations(&self) -> usize {
        self.relations.len()
    }

    /// Returns the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.relations_starts.len() - 1
    }

    /// Applies a function to each range of relations.
    pub fn map_range_data(&mut self, f: impl Fn(&mut [T])) {
        for block in 0..self.num_blocks() {
            let (start, end) = self.relation_range(block);
            f(&mut self.relations[start..end]);
        }
    }
}

impl RelationProvider {
    /// Inverts the relation provider, swapping predecessors and successors.
    pub fn invert(&self) -> Self {
        let mut inverted_relations = vec![vec![]; self.num_blocks()];

        for (block, start) in self.relations_starts[0..self.relations_starts.len() - 1]
            .iter()
            .enumerate()
        {
            let end = self.relations_starts[block + 1];
            for relation in &self.relations[*start as usize..end as usize] {
                inverted_relations[*relation as usize].push(block as u32);
            }
        }

        let relations_starts = inverted_relations.iter().fold(vec![0], |mut acc, v| {
            acc.push(acc.last().unwrap() + v.len() as u32);
            acc
        });

        inverted_relations
            .iter_mut()
            .for_each(|v| v.sort_unstable());

        Self {
            relations_starts,
            relations: inverted_relations.into_iter().flatten().collect(),
        }
    }

    /// Combines two relation providers into one.
    pub fn combine(mut self, other: Self) -> Self {
        let offset = self.num_blocks() as u32;
        for block in 0..other.num_blocks() {
            let relations = other
                .relations(block)
                .iter()
                .map(|i| i + offset)
                .collect::<Vec<_>>();

            self.add_relations(relations);
        }

        self
    }

    /// Flattens the relations using
    ///
    /// outputs a [`RelationProvider`] where each block's relations contains all reachable blocks and itself.
    pub fn flatten(&self) -> Self {
        use rayon::iter::ParallelIterator;
        let flattened = (0..self.num_blocks())
            .into_par_iter()
            .progress()
            .map_init(
                || (VecDeque::new(), vec![false; self.num_blocks()]),
                |(stack, mask), block| {
                    let mut walker = WalkerWithBuffer::new(block as u32, stack, mask);

                    let mut tmp = vec![];
                    while let Some(node_id) = walker.next(self) {
                        tmp.push(node_id);
                    }
                    tmp.sort();

                    tmp
                },
            )
            .collect::<Vec<_>>();

        RelationProvider::from_linked_list(&flattened)
    }

    /// Flattens the relations into ranges using
    ///
    /// outputs a [`RelationRangeProvider`] where each block's relations contains all reachable blocks and itself.
    pub fn flatten_to_ranges(&self) -> RelationRangeProvider {
        use rayon::iter::ParallelIterator;
        let flattened = (0..self.num_blocks())
            .into_par_iter()
            .progress()
            .map_init(
                || (VecDeque::new(), vec![false; self.num_blocks()]),
                |(stack, mask), block| {
                    let mut walker = WalkerWithBuffer::new(block as u32, stack, mask);

                    let mut tmp = vec![];
                    while let Some(node_id) = walker.next(self) {
                        tmp.push(node_id);
                    }
                    tmp.sort();
                    to_block_ranges(&tmp)
                },
            )
            .collect::<Vec<_>>();

        RelationRangeProvider::from_linked_list(&flattened)
    }
}

/// A range of blocks, represented by a start (inclusive) and end (exclusive).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Default)]
pub struct BlockRange {
    pub start: u32,
    pub end: u32,
}

/// A provider for block relations using ranges.
#[derive(Clone, Debug, Default)]
pub struct RelationRangeProvider {
    ranges: Vec<BlockRange>,
    relations_starts: Vec<u32>,
}

impl RelationRangeProvider {
    /// Add a precedence constraint for a block.
    pub fn add_relations(&mut self, precedences: Vec<BlockRange>) {
        if self.ranges.is_empty() {
            self.relations_starts.push(0);
        }
        // precedences.sort_unstable();
        self.ranges.extend(precedences);
        self.relations_starts.push(self.ranges.len() as u32);
    }

    /// Create a [`RelationRangeProvider`] from a linked list of block ranges.
    pub fn from_linked_list(linked_list: &[Vec<BlockRange>]) -> Self {
        let mut out = Self::default();
        for relations in linked_list {
            let relations = relations.clone();
            out.add_relations(relations);
        }

        out
    }

    /// Returns the relations for a given block index as an iterator.
    #[inline(always)]
    pub fn relations(&self, block_index: usize) -> BlockRangeIter<'_> {
        let start = self.relations_starts[block_index] as usize;
        let end = self.relations_starts[block_index + 1] as usize;
        BlockRangeIter::new(&self.ranges[start..end])
    }

    /// Returns the number of blocks.
    #[inline(always)]
    pub fn num_blocks(&self) -> usize {
        self.relations_starts.len() - 1
    }
}

/// Converts a sorted list of block indices into a list of `BlockRange`s.
pub fn to_block_ranges(nums: &[u32]) -> Vec<BlockRange> {
    let mut ranges = Vec::new();
    let mut iter = nums.iter().peekable();

    while let Some(&start) = iter.next() {
        let mut end = start;

        while let Some(&&next) = iter.peek() {
            if next == end + 1 {
                end = next;
                iter.next();
            } else {
                break;
            }
        }

        // Exclusive range: end + 1
        ranges.push(BlockRange {
            start,
            end: (end + 1),
        });
    }

    ranges
}

/// An iterator over block indices defined by a list of `BlockRange`s.
pub struct BlockRangeIter<'a> {
    ranges: &'a [BlockRange],
    outer_idx: usize,
    inner_val: u32,
    current_end: u32,
}

impl<'a> BlockRangeIter<'a> {
    /// Creates a new `BlockRangeIter` from a slice of `BlockRange`s.
    #[inline(always)]
    pub fn new(ranges: &'a [BlockRange]) -> Self {
        let mut iter = Self {
            ranges,
            outer_idx: 0,
            inner_val: 0, // will be set on first `next`
            current_end: 0,
        };

        // Initialize inner_val to first start (if any)
        if let Some(first) = ranges.first() {
            iter.inner_val = first.start;
            iter.current_end = first.end;
        }

        iter
    }
}

impl<'a> Iterator for BlockRangeIter<'a> {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.inner_val < self.current_end {
            let val = self.inner_val;
            self.inner_val += 1;
            return Some(val);
        }

        // Move to next range
        self.outer_idx += 1;
        if self.outer_idx < self.ranges.len() {
            let BlockRange { start, end } = unsafe { *self.ranges.get_unchecked(self.outer_idx) };
            self.inner_val = start;
            self.current_end = end;

            // Could this just be a jump?
            if self.inner_val < self.current_end {
                let val = self.inner_val;
                self.inner_val += 1;
                return Some(val);
            }
        }

        None
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut remaining = 0;
        if self.outer_idx < self.ranges.len() {
            let BlockRange { end, .. } = self.ranges[self.outer_idx];
            remaining += (end - self.inner_val) as usize;

            for r in &self.ranges[self.outer_idx + 1..] {
                remaining += (r.end - r.start) as usize;
            }
        }
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for BlockRangeIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_precedence() {
        let mut provider = RelationProvider::default();
        provider.add_relations(vec![1, 2]);
        provider.add_relations(vec![2]);
        provider.add_relations(vec![]);
        provider.add_relations(vec![0, 1, 2]);
        assert_eq!(provider.relations(0), &[1, 2]);
        assert_eq!(provider.relations(1), &[2]);
        assert_eq!(provider.relations(2), &Vec::<u32>::new());
        assert_eq!(provider.relations(3), &[0, 1, 2]);
        assert_eq!(provider.num_blocks(), 4);
    }

    #[test]
    fn invert_precedence() {
        let mut provider = RelationProvider::default();
        provider.add_relations(vec![1, 2]);
        provider.add_relations(vec![2]);
        provider.add_relations(vec![]);
        provider.add_relations(vec![0, 1, 2]);
        let inverted = provider.invert();
        assert_eq!(inverted.relations(0), &[3]);
        assert_eq!(inverted.relations(1), &[0, 3]);
        assert_eq!(inverted.relations(2), &[0, 1, 3]);
        assert_eq!(inverted.relations(3), &Vec::<u32>::new());
        assert_eq!(inverted.num_blocks(), 4);
    }
}
