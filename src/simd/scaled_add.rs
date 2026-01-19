use itertools::izip;
use pulp::{Arch, Simd, WithSimd};
use wide::f64x4;

use super::{BinaryOp, WideBinaryOp};

#[inline(always)]
pub fn slice_scaled_add(a: &[f64], b: &[f64], scalar: f64, out: &mut [f64]) {
    struct Adder<'a>(&'a [f64], &'a [f64], f64, &'a mut [f64]);
    impl<'a> WithSimd for Adder<'a> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;
            let scaler = self.2;
            let out = self.3;

            let (a_head, a_tail) = S::as_simd_f64s(a);
            let (b_head, b_tail) = S::as_simd_f64s(b);
            let scaler_splat = simd.splat_f64s(scaler);
            let (out_head, out_tail) = S::as_mut_simd_f64s(out);

            for (a, b, out) in izip!(a_head.iter(), b_head.iter(), out_head.iter_mut()) {
                *out = simd.mul_add_f64s(*b, scaler_splat, *a)
            }

            for (a, b, out) in izip!(a_tail.iter(), b_tail.iter(), out_tail.iter_mut()) {
                *out = a + b * scaler;
            }
        }
    }

    let arch = Arch::new();

    let adder = Adder(a, b, scalar, out);
    arch.dispatch(adder);
}

pub struct SimdAdder;
impl BinaryOp for SimdAdder {
    fn apply<S: Simd>(&self, a: &[f64], b: &[f64], out: &mut [f64], _simd: &S) {
        slice_scaled_add(a, b, 1.0, out);
    }
}

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

pub struct WideAdd;

impl WideBinaryOp<4, f64> for WideAdd {
    fn simd_apply(&self, a: &[f64; 4], b: &[f64; 4], out: &mut [f64; 4]) {
        *out = (f64x4::from(*a) + f64x4::from(*b)).to_array();
    }

    fn scalar_apply(&self, a: &f64, b: &f64, out: &mut f64) {
        *out = a + b
    }
}
