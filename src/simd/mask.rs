use itertools::izip;
use pulp::{Arch, Simd, WithSimd};

pub trait MaskOp {
    fn masks_iter<'b, S: Simd>(
        &'b self,
        s: &'b S,
    ) -> (
        impl Iterator<Item = <S as Simd>::m64s> + 'b,
        impl Iterator<Item = bool> + 'b,
    );
}

pub trait StatelessBinaryMaskOp {
    fn head_mask<S: Simd>(
        &self,
        a: <S as Simd>::f64s,
        b: <S as Simd>::f64s,
        simd: &S,
    ) -> <S as Simd>::m64s;
    fn tail_mask(&self, a: f64, b: f64) -> bool;
}

pub struct GTMaskOp;
impl StatelessBinaryMaskOp for GTMaskOp {
    fn head_mask<S: Simd>(
        &self,
        a: <S as Simd>::f64s,
        b: <S as Simd>::f64s,
        simd: &S,
    ) -> <S as Simd>::m64s {
        simd.greater_than_f64s(a, b)
    }

    fn tail_mask(&self, a: f64, b: f64) -> bool {
        a > b
    }
}

pub struct GEMaskOp;
impl StatelessBinaryMaskOp for GEMaskOp {
    fn head_mask<S: Simd>(
        &self,
        a: <S as Simd>::f64s,
        b: <S as Simd>::f64s,
        simd: &S,
    ) -> <S as Simd>::m64s {
        simd.greater_than_or_equal_f64s(a, b)
    }

    fn tail_mask(&self, a: f64, b: f64) -> bool {
        a >= b
    }
}

pub struct LTMaskOp;
impl StatelessBinaryMaskOp for LTMaskOp {
    fn head_mask<S: Simd>(
        &self,
        a: <S as Simd>::f64s,
        b: <S as Simd>::f64s,
        simd: &S,
    ) -> <S as Simd>::m64s {
        simd.less_than_f64s(a, b)
    }

    fn tail_mask(&self, a: f64, b: f64) -> bool {
        a < b
    }
}

pub struct LEMaskOp;
impl StatelessBinaryMaskOp for LEMaskOp {
    fn head_mask<S: Simd>(
        &self,
        a: <S as Simd>::f64s,
        b: <S as Simd>::f64s,
        simd: &S,
    ) -> <S as Simd>::m64s {
        simd.less_than_or_equal_f64s(a, b)
    }

    fn tail_mask(&self, a: f64, b: f64) -> bool {
        a <= b
    }
}

pub struct StatefullBinaryMaskOp<'a, OP> {
    pub a: &'a [f64],
    pub b: &'a [f64],
    pub op: OP,
}

impl<'a, OP: StatelessBinaryMaskOp> MaskOp for StatefullBinaryMaskOp<'a, OP> {
    fn masks_iter<'b, S: Simd>(
        &'b self,
        s: &'b S,
    ) -> (
        impl Iterator<Item = <S as Simd>::m64s> + 'b,
        impl Iterator<Item = bool> + 'b,
    ) {
        let (a_head, a_tail) = S::as_simd_f64s(self.a);
        let (b_head, b_tail) = S::as_simd_f64s(self.b);

        let head_iter =
            izip!(a_head.iter(), b_head.iter()).map(|(a, b)| self.op.head_mask(*a, *b, s));

        let tail_iter = izip!(a_tail.iter(), b_tail.iter()).map(|(a, b)| self.op.tail_mask(*a, *b));

        (head_iter, tail_iter)
    }
}

#[inline(always)]
pub fn select_mask(fill_1: &[f64], fill_2: &[f64], mask_op: impl MaskOp, out_buffer: &mut [f64]) {
    struct GeMask<'a, MO: MaskOp>(&'a [f64], &'a [f64], MO, &'a mut [f64]);

    impl<'a, MO: MaskOp> WithSimd for GeMask<'a, MO> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;
            let mo = self.2;
            let out = self.3;

            let (a_head, a_tail) = S::as_simd_f64s(a);
            let (b_head, b_tail) = S::as_simd_f64s(b);
            let (mask_head, mask_tail) = mo.masks_iter(&simd);

            let (out_head, out_tail) = S::as_mut_simd_f64s(out);

            for (a, b, mask, out) in
                izip!(a_head.iter(), b_head.iter(), mask_head, out_head.iter_mut())
            {
                *out = simd.select_f64s_m64s(mask, *a, *b);
            }

            for (a, b, mask, out) in
                izip!(a_tail.iter(), b_tail.iter(), mask_tail, out_tail.iter_mut())
            {
                if mask {
                    *out = *a;
                } else {
                    *out = *b;
                }
            }
        }
    }

    let arch = Arch::new();

    let ge_mask = GeMask(fill_1, fill_2, mask_op, out_buffer);
    arch.dispatch(ge_mask);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gt() {
        let a = [1.0; 100];
        let b = [2.0; 100];
        let mut out_buffer = [0.0; 100];

        let mask_op = StatefullBinaryMaskOp {
            a: &a,
            b: &b,
            op: LEMaskOp,
        };

        select_mask(&a, &b, mask_op, &mut out_buffer);

        println!("out buffer: {out_buffer:?}")
    }
}
