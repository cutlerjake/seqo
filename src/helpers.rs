//! Helper functions and traits for numerical operations and SIMD optimizations.

use pulp::{Arch, Simd, WithSimd};

/// Computes a penalty for exceeding an upper bound.
#[inline(always)]
pub fn scalar_ub_penalty(val: f64, ub: f64, cost: f64) -> f64 {
    if val > ub { (val - ub) * cost } else { 0.0 }
}

/// Computes a penalty for falling below a lower bound.
#[inline(always)]
pub fn scalar_lb_penalty(val: f64, lb: f64, cost: f64) -> f64 {
    if val < lb { (lb - val) * cost } else { 0.0 }
}

/// Performs a cross operation between two slices using the provided binary operation,
/// storing the results in the output buffer.
///
/// The output buffer must be pre-allocated and have a size equal to `a.len() * b.len()`.
#[inline(always)]
pub fn cross_op(a: &[f64], b: &[f64], out_buffer: &mut [f64], op: impl Fn(f64, f64) -> f64) {
    let size = a.len() * b.len();
    let mut filler_it = a.iter().flat_map(|a| b.iter().map(|b| op(*a, *b)));
    out_buffer[0..size].fill_with(|| filler_it.next().unwrap());
}

/// Performs an in-place scaled addition of slice `b` to slice `a`, scaled by `scalar`.
#[inline(always)]
pub fn slice_scaled_add_in_place(a: &mut [f64], b: &[f64], scalar: f64) {
    struct InPlaceAdder<'a>(&'a mut [f64], &'a [f64], f64);
    impl<'a> WithSimd for InPlaceAdder<'a> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;
            let scaler = self.2;

            let (a_head, a_tail) = S::as_mut_simd_f64s(a);
            let (b_head, b_tail) = S::as_simd_f64s(b);
            let scaler_splat = simd.splat_f64s(scaler);

            for (a, b) in a_head.iter_mut().zip(b_head) {
                *a = simd.mul_add_f64s(*b, scaler_splat, *a)
            }

            for (a, b) in a_tail.iter_mut().zip(b_tail) {
                *a += b * scaler;
            }
        }
    }

    let arch = Arch::new();

    let adder = InPlaceAdder(a, b, scalar);
    arch.dispatch(adder);
}

/// Performs an in-place subtraction of slice `b` from slice `a`.
#[inline(always)]
pub fn slice_sub_in_place(a: &mut [f64], b: &[f64]) {
    struct InPlaceSubber<'a>(&'a mut [f64], &'a [f64]);
    impl<'a> WithSimd for InPlaceSubber<'a> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;

            let (a_head, a_tail) = S::as_mut_simd_f64s(a);
            let (b_head, b_tail) = S::as_simd_f64s(b);

            for (a, b) in a_head.iter_mut().zip(b_head) {
                *a = simd.sub_f64s(*a, *b)
            }

            for (a, b) in a_tail.iter_mut().zip(b_tail) {
                *a -= *b;
            }
        }
    }

    let arch = Arch::new();

    let adder = InPlaceSubber(a, b);
    arch.dispatch(adder);
}

/// Performs a masked selection between two slices based on the comparison of two other slices.
#[inline(always)]
pub fn slice_ge_mask(a: &[f64], b: &[f64], c: &[f64], fill: f64, out_buffer: &mut [f64]) {
    struct GeMask<'a>(&'a [f64], &'a [f64], &'a [f64], f64, &'a mut [f64]);

    impl<'a> WithSimd for GeMask<'a> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;
            let c = self.2;
            let fill = self.3;
            let out = self.4;

            let (a_head, a_tail) = S::as_simd_f64s(a);
            let (b_head, b_tail) = S::as_simd_f64s(b);
            let (c_head, c_tail) = S::as_simd_f64s(c);
            let (out_head, out_tail) = S::as_mut_simd_f64s(out);

            let fill_splat = simd.splat_f64s(fill);

            for (((a, b), c), out) in a_head
                .iter()
                .zip(b_head)
                .zip(c_head)
                .zip(out_head.iter_mut())
            {
                let mask = simd.greater_than_or_equal_f64s(*a, *b);

                *out = simd.select_f64s_m64s(mask, *c, fill_splat);
            }

            for (((a, b), c), out) in a_tail
                .iter()
                .zip(b_tail)
                .zip(c_tail)
                .zip(out_tail.iter_mut())
            {
                if a >= b {
                    *out = *c;
                } else {
                    *out = 0.0;
                }
            }
        }
    }

    let arch = Arch::new();

    let ge_mask = GeMask(a, b, c, fill, out_buffer);
    arch.dispatch(ge_mask);
}

/// Performs a masked selection between two slices based on the comparison of two other slices.
#[inline(always)]
pub fn slice_gt_mask(a: &[f64], b: &[f64], c: &[f64], fill: f64, out_buffer: &mut [f64]) {
    struct GtMask<'a>(&'a [f64], &'a [f64], &'a [f64], f64, &'a mut [f64]);

    impl<'a> WithSimd for GtMask<'a> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;
            let c = self.2;
            let fill = self.3;
            let out = self.4;

            let (a_head, a_tail) = S::as_simd_f64s(a);
            let (b_head, b_tail) = S::as_simd_f64s(b);
            let (c_head, c_tail) = S::as_simd_f64s(c);
            let (out_head, out_tail) = S::as_mut_simd_f64s(out);

            let fill_splat = simd.splat_f64s(fill);

            for (((a, b), c), out) in a_head
                .iter()
                .zip(b_head)
                .zip(c_head)
                .zip(out_head.iter_mut())
            {
                let mask = simd.greater_than_f64s(*a, *b);

                *out = simd.select_f64s_m64s(mask, *c, fill_splat);
            }

            for (((a, b), c), out) in a_tail
                .iter()
                .zip(b_tail)
                .zip(c_tail)
                .zip(out_tail.iter_mut())
            {
                if a > b {
                    *out = *c;
                } else {
                    *out = 0.0;
                }
            }
        }
    }

    let arch = Arch::new();

    let ge_mask = GtMask(a, b, c, fill, out_buffer);
    arch.dispatch(ge_mask);
}

/// Trait for element-wise addition and subtraction.
pub trait ElementAddSub {
    fn element_add(&mut self, other: &Self);
    fn element_sub(&mut self, other: &Self);
}

macro_rules! impl_addsub_scalar {
    ($($t:ty),*) => {
        $(
            impl ElementAddSub for $t {
                #[inline(always)]
                fn element_add(&mut self, other: &$t) {
                    *self += other;
                }

                #[inline(always)]
                fn element_sub(&mut self, other: &$t) {
                    *self -= other;
                }
            }
        )*
    };
}

macro_rules! impl_addsub_array {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> ElementAddSub for [$t; N] {

                #[inline(always)]
                fn element_add(&mut self, other: &[$t; N]) {
                    for i in 0..N {
                        self[i] += other[i];
                    }
                }

                #[inline(always)]
                fn element_sub(&mut self, other: &[$t; N]) {
                    for i in 0..N {
                        self[i] -= other[i];
                    }
                }
            }
        )*
    };
}

macro_rules! impl_addsub_vec {
    ($($t:ty),*) => {
        $(
            impl ElementAddSub for Vec<$t> {

                #[inline(always)]
                fn element_add(&mut self, other: &Vec<$t>) {
                    assert_eq!(self.len(), other.len(), "Vector lengths must match");
                    for (a, b) in self.iter_mut().zip(other.iter()) {
                        *a += b;
                    }
                }

                #[inline(always)]
                fn element_sub(&mut self, other: &Vec<$t>) {
                    assert_eq!(self.len(), other.len(), "Vector lengths must match");
                    for (a, b) in self.iter_mut().zip(other.iter()) {
                        *a -= b;
                    }
                }
            }
        )*
    };
}

impl_addsub_scalar!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
impl_addsub_array!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
impl_addsub_vec!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);

/// Trait for setting values to zero.
pub trait SetZero {
    fn set_zero(&mut self);
}

impl<T> SetZero for Vec<T>
where
    T: Default,
{
    #[inline(always)]
    fn set_zero(&mut self) {
        self.iter_mut().for_each(|v| *v = T::default());
    }
}

impl<T> SetZero for [T]
where
    T: Default,
{
    #[inline(always)]
    fn set_zero(&mut self) {
        self.iter_mut().for_each(|v| *v = T::default());
    }
}

impl<T, const N: usize> SetZero for [T; N]
where
    T: Default,
{
    #[inline(always)]
    fn set_zero(&mut self) {
        self.iter_mut().for_each(|v| *v = T::default());
    }
}

macro_rules! impl_set_zero_scalar {
    ($($t:ty),*) => {
        $(
            impl SetZero for $t {
                #[inline(always)]
                fn set_zero(&mut self) {
                   *self = 0 as $t;
                }
            }
        )*
    };
}

impl_set_zero_scalar!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
