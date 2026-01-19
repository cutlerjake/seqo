//! A specialized map structure that uses a combination of an index and a period
//! as the key. This is particularly useful in scheduling contexts where each task
//! or block can have multiple periods, and we need to efficiently map
//! (index, period) pairs to specific values.
use std::ops::Deref;

use parking_lot::Mutex;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rustc_hash::FxHashMap;

type IdxPeriod = (u32, u8);

#[inline(always)]
pub fn make_key(idx: u32, period: u8) -> u64 {
    ((idx as u64) << 8) | (period as u64)
}

#[allow(dead_code)]
#[inline(always)]
pub fn make_idx_period(key: u64) -> IdxPeriod {
    let idx = (key >> 8) as u32;
    let period = (key & 0xFF) as u8;
    (idx, period)
}

#[derive(Debug, Clone)]
pub struct IdxPeriodMap<T>(FxHashMap<u64, T>);

impl<T> Default for IdxPeriodMap<T> {
    fn default() -> Self {
        Self(FxHashMap::default())
    }
}

impl<T> IdxPeriodMap<T> {
    #[inline(always)]
    pub fn make_key(idx: u32, period: u8) -> u64 {
        ((idx as u64) << 8) | (period as u64)
    }

    #[inline(always)]
    pub fn make_idx_period(key: u64) -> IdxPeriod {
        let idx = (key >> 8) as u32;
        let period = (key & 0xFF) as u8;
        (idx, period)
    }

    #[inline(always)]
    pub fn insert(&mut self, (idx, period): IdxPeriod, value: T) {
        self.0.insert(Self::make_key(idx, period), value);
    }

    #[inline(always)]
    pub fn get(&self, (idx, period): IdxPeriod) -> Option<&T> {
        self.0.get(&Self::make_key(idx, period))
    }

    #[inline(always)]
    #[track_caller]
    pub fn at(&self, (idx, period): IdxPeriod) -> &T {
        &self.0[&Self::make_key(idx, period)]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, (idx, period): IdxPeriod) -> Option<&mut T> {
        self.0.get_mut(&Self::make_key(idx, period))
    }

    #[inline(always)]
    pub fn at_mut(&mut self, (idx, period): IdxPeriod) -> &mut T {
        self.0.get_mut(&Self::make_key(idx, period)).unwrap()
    }

    #[inline(always)]
    pub fn entry(&mut self, key: IdxPeriod) -> std::collections::hash_map::Entry<'_, u64, T> {
        self.0.entry(Self::make_key(key.0, key.1))
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn contains(&self, (idx, period): IdxPeriod) -> bool {
        self.0.contains_key(&Self::make_key(idx, period))
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T> IntoIterator for IdxPeriodMap<T> {
    type Item = (IdxPeriod, T);
    type IntoIter = std::iter::Map<
        std::collections::hash_map::IntoIter<u64, T>,
        fn((u64, T)) -> (IdxPeriod, T),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.0
            .into_iter()
            .map(|(key, value)| (Self::make_idx_period(key), value))
    }
}

impl<'a, T> IntoIterator for &'a IdxPeriodMap<T> {
    type Item = (IdxPeriod, &'a T);
    type IntoIter = std::iter::Map<
        std::collections::hash_map::Iter<'a, u64, T>,
        fn((&'a u64, &'a T)) -> (IdxPeriod, &'a T),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.0
            .iter()
            .map(|(&key, value)| (IdxPeriodMap::<T>::make_idx_period(key), value))
    }
}

impl<'a, T> IntoIterator for &'a mut IdxPeriodMap<T> {
    type Item = (IdxPeriod, &'a mut T);
    type IntoIter = std::iter::Map<
        std::collections::hash_map::IterMut<'a, u64, T>,
        fn((&'a u64, &'a mut T)) -> (IdxPeriod, &'a mut T),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.0
            .iter_mut()
            .map(|(&key, value)| (IdxPeriodMap::<T>::make_idx_period(key), value))
    }
}

impl<T: Send> IntoParallelIterator for IdxPeriodMap<T> {
    type Item = (IdxPeriod, T);
    type Iter = rayon::iter::Map<
        rayon::collections::hash_map::IntoIter<u64, T>,
        fn((u64, T)) -> (IdxPeriod, T),
    >;

    fn into_par_iter(self) -> Self::Iter {
        self.0
            .into_par_iter()
            .map(|(key, value)| (Self::make_idx_period(key), value))
    }
}

impl<'a, T: Send + Sync> IntoParallelIterator for &'a IdxPeriodMap<T>
where
    &'a T: Send,
{
    type Item = (IdxPeriod, &'a T);
    type Iter = rayon::iter::Map<
        rayon::collections::hash_map::Iter<'a, u64, T>,
        fn((&'a u64, &'a T)) -> (IdxPeriod, &'a T),
    >;

    fn into_par_iter(self) -> Self::Iter {
        self.0
            .par_iter()
            .map(|(key, value)| (IdxPeriodMap::<T>::make_idx_period(*key), value))
    }
}

impl<'a, T: Send + Sync> IntoParallelIterator for &'a mut IdxPeriodMap<T>
where
    &'a T: Send,
{
    type Item = (IdxPeriod, &'a mut T);
    type Iter = rayon::iter::Map<
        rayon::collections::hash_map::IterMut<'a, u64, T>,
        fn((&'a u64, &'a mut T)) -> (IdxPeriod, &'a mut T),
    >;

    fn into_par_iter(self) -> Self::Iter {
        self.0
            .par_iter_mut()
            .map(|(key, value)| (IdxPeriodMap::<T>::make_idx_period(*key), value))
    }
}

impl Deref for IdxPeriodMap<Mutex<u32>> {
    type Target = FxHashMap<u64, Mutex<u32>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
