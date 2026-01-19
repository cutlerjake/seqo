use pulp::{Arch, Simd, WithSimd};

use super::BinaryOp;

pub fn cross_with_op<OP: BinaryOp>(
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    op: OP,
    workspace: &mut Vec<f64>,
) {
    struct Crosser<'a, OP: BinaryOp>(&'a [f64], &'a [f64], &'a mut [f64], OP, &'a mut [f64]);
    impl<'a, OP: BinaryOp> WithSimd for Crosser<'a, OP> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let a = self.0;
            let b = self.1;
            let out = self.2;
            let op = self.3;
            let workspace = self.4;

            for (i, a) in a.iter().enumerate() {
                // Fill workspace
                workspace.fill(*a);

                let out_slice = &mut out[i * b.len()..(i + 1) * b.len()];
                op.apply(workspace, b, out_slice, &simd);
            }
        }
    }

    let arch = Arch::new();

    workspace.resize(b.len(), 0.0);

    let adder = Crosser(a, b, out, op, workspace.as_mut_slice());
    arch.dispatch(adder);
}

#[cfg(test)]
mod test {
    use crate::simd::scaled_add::SimdAdder;

    use super::*;

    #[test]
    fn cross_add() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];

        let mut workspace = vec![0.0; 4];
        let mut out = [0.0; 16];

        cross_with_op(&a, &b, &mut out, SimdAdder, &mut workspace);

        println!("out: {out:?}");
    }
}
