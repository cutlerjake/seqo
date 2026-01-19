use std::{
    marker::PhantomData,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use itertools::izip;
use proc::{impl_binary_op, impl_binary_op_mut, impl_unary_op};
use pulp::Simd;
use wide::{CmpGt, CmpLt};

pub mod cross;
pub mod mask;
pub mod scaled_add;
pub use wide;

#[inline(always)]
pub fn chunk<const N: usize, T>(slice: &[T]) -> (&[[T; N]], &[T]) {
    assert!(N > 0);

    let len = slice.len();
    let data = slice.as_ptr();

    let div = len / N;
    let rem = len % N;

    unsafe {
        (
            from_raw_parts(data as *const [T; N], div),
            from_raw_parts(data.add(len - rem), rem),
        )
    }
}

#[inline(always)]
pub fn chunk_mut<const N: usize, T>(slice: &mut [T]) -> (&mut [[T; N]], &mut [T]) {
    assert!(N > 0);

    let len = slice.len();
    let data = slice.as_mut_ptr();

    let div = len / N;
    let rem = len % N;

    unsafe {
        (
            from_raw_parts_mut(data as *mut [T; N], div),
            from_raw_parts_mut(data.add(len - rem), rem),
        )
    }
}

pub trait BinaryOp {
    fn apply<S: Simd>(&self, a: &[f64], b: &[f64], out: &mut [f64], simd: &S);
}

pub trait WideUnaryOp<const N: usize, T> {
    fn simd_apply(&self, a: &[T; N], out: &mut [T; N]);
    fn scalar_apply(&self, a: &T, out: &mut T);
}

pub trait WideBinaryOp<const N: usize, T> {
    fn simd_apply(&self, a: &[T; N], b: &[T; N], out: &mut [T; N]);
    fn scalar_apply(&self, a: &T, b: &T, out: &mut T);
}

pub trait WideBinaryOpMut<const N: usize, T> {
    fn simd_apply(&self, a: &mut [T; N], b: &[T; N], out: &mut [T; N]);
    fn scalar_apply(&self, a: &mut T, b: &T, out: &mut T);
}

#[inline(always)]
pub fn wide_cross_op<const N: usize, T: Copy>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    op: impl WideBinaryOp<N, T>,
) {
    let (b_head, b_tail) = chunk::<N, _>(b);

    let mut idx = 0;
    let mut a_wide = [a[0]; N];
    for a in a {
        a_wide.fill(*a);

        let (out_head, out_tail) = chunk_mut::<N, _>(&mut out[idx..idx + b.len()]);
        for (b, out) in izip!(b_head.iter(), out_head.iter_mut()) {
            op.simd_apply(&a_wide, b, out);
        }

        for (b, out) in izip!(b_tail.iter(), out_tail.iter_mut()) {
            op.scalar_apply(a, b, out);
        }

        idx += b.len();
    }
}

#[inline(always)]
pub fn wide_binary_op<const N: usize, T>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    op: impl WideBinaryOp<N, T>,
) {
    let (a_head, a_tail) = chunk::<N, _>(a);
    let (b_head, b_tail) = chunk::<N, _>(b);
    let (out_head, out_tail) = chunk_mut::<N, _>(out);

    for (a, b, out) in izip!(a_head.iter(), b_head.iter(), out_head.iter_mut()) {
        op.simd_apply(a, b, out);
    }

    for (a, b, out) in izip!(a_tail.iter(), b_tail.iter(), out_tail.iter_mut()) {
        op.scalar_apply(a, b, out);
    }
}

#[inline(always)]
pub fn wide_binary_mut_op<const N: usize, T>(
    a: &mut [T],
    b: &[T],
    out: &mut [T],
    op: impl WideBinaryOpMut<N, T>,
) {
    let (a_head, a_tail) = chunk_mut::<N, _>(a);
    let (b_head, b_tail) = chunk::<N, _>(b);
    let (out_head, out_tail) = chunk_mut::<N, _>(out);

    for (a, b, out) in izip!(a_head.iter_mut(), b_head.iter(), out_head.iter_mut()) {
        op.simd_apply(a, b, out);
    }

    for (a, b, out) in izip!(a_tail.iter_mut(), b_tail.iter(), out_tail.iter_mut()) {
        op.scalar_apply(a, b, out);
    }
}

#[inline(always)]
pub fn wide_unary_op<const N: usize, T>(a: &[T], out: &mut [T], op: impl WideUnaryOp<N, T>) {
    let (a_head, a_tail) = chunk::<N, _>(a);
    let (out_head, out_tail) = chunk_mut::<N, _>(out);

    for (a, out) in izip!(a_head.iter(), out_head.iter_mut()) {
        op.simd_apply(a, out);
    }

    for (a, out) in izip!(a_tail.iter(), out_tail.iter_mut()) {
        op.scalar_apply(a, out);
    }
}

pub struct WideAdd;
impl_binary_op!(WideAdd, |a, b| a + b);

pub struct WideSub;
impl_binary_op!(WideSub, |a, b| a - b);

pub struct WideMul;
impl_binary_op!(WideMul, |a, b| a * b, ["i8", "u8"]);

pub struct WideDiv;
impl_binary_op!(WideDiv, |a, b| a / b, ["i", "u"]);

pub struct WidePow;
impl_binary_op!(
    WidePow,
    |a: __FT__, b: __FT__| a.pow___T__(b),
    |a: __FT__, b: __FT__| a.powf(b),
    ["i", "u"]
);

pub struct WideSPStep<T, V> {
    rehandle: V,

    _marker: PhantomData<T>,
}

impl<T, V> WideSPStep<T, V> {
    pub fn new(rehandle: V) -> Self {
        Self {
            rehandle,
            _marker: PhantomData,
        }
    }
}

impl_binary_op_mut!(
    WideSPStep<__T__,__T__>,
    |a:&mut __FT__, b:__FT__| {
        // Add incoming quantity to stockpiled quantity.
        *a += b;

        // Compute rehandled quantity.
        let rehandle = *a * __FT__::from(self.rehandle);

        // Subtract rehandled quantity.
        *a -= rehandle;

        // Return rehandled quantity
        rehandle
    },

    ["i", "u"]
);

pub struct WideUB<T, V> {
    ub: V,
    _marker: PhantomData<T>,
}
impl<T, V> WideUB<T, V> {
    pub fn new(ub: V) -> Self {
        Self {
            ub,
            _marker: PhantomData,
        }
    }
}
impl_unary_op!(WideUB<__T__,__T__>,
|a: __FT__| {
    let ub = __FT__::from(self.ub);
    let mask = a.cmp_gt(ub);
    (a-ub)&mask

},
|a: __FT__| {
    if a > self.ub {
        a - self.ub
    }else {
    0.0
    }
}
, ["i", "u"]);

pub struct WideLB<T, V> {
    lb: V,
    _marker: PhantomData<T>,
}
impl<T, V> WideLB<T, V> {
    pub fn new(lb: V) -> Self {
        Self {
            lb,
            _marker: PhantomData,
        }
    }
}
impl_unary_op!(WideLB<__T__,__T__>,
|a: __FT__| {
    let lb = __FT__::from(self.lb);
    let mask = a.cmp_lt(lb);
    (lb - a)&mask

},
|a: __FT__| {
    if a < self.lb {
        self.lb - a
    } else {
        0.0
    }
}
, ["i", "u"]);
