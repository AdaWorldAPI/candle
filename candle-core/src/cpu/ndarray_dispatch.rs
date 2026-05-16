//! Single dispatch point for routing candle CPU compute through the
//! AdaWorldAPI/ndarray fork's runtime-tier SIMD polyfill and `hpc::*`
//! kernels.
//!
//! Modeled on `burn-ndarray/src/simd_dispatch.rs`. The fork's
//! [`ndarray::simd`] exposes a [`LazyLock<Tier>`] that detects
//! `Avx512 → Avx2 → NeonDotProd → Neon → Scalar` once at startup; all
//! `hpc::*` kernels auto-dispatch through it with zero per-call branch
//! cost.
//!
//! Entry points (gated on `feature = "ndarray-simd"`):
//!
//! | Op family    | Function                          | Underlying call                              |
//! |--------------|-----------------------------------|----------------------------------------------|
//! | MatMul f32   | [`matmul_f32`]                    | `hpc::amx_matmul::matmul_f32` → `linalg::general_mat_mul` |
//! | MatMul f64   | [`matmul_f64`]                    | `linalg::general_mat_mul`                    |
//! | MatMul bf16  | [`matmul_bf16_to_f32`]            | `hpc::amx_matmul::matmul_bf16_to_f32`        |
//! | Reductions   | [`sum_f32`], [`max_f32`], …       | `hpc::reductions::{sum,max,min,argmax,…}`    |
//! | Softmax      | [`softmax_f32_last_dim`]          | `hpc::activations::softmax_f32`              |
//! | Sigmoid      | [`sigmoid_f32_slice`]             | `hpc::activations::sigmoid_f32`              |
//! | VML          | [`exp_f32_slice`], [`ln_f32_slice`], [`tanh_f32_slice`] | `hpc::vml::{vsexp,vsln,vstanh}` |
//! | Elementwise  | [`add_f32_inplace`], [`mul_f32_inplace`], …            | `simd::{add_f32_inplace,mul_f32_inplace,…}`  |
//! | Vector dot   | [`vec_dot_f32_slice`]             | `simd::F32x16::mul_add` over `array_windows` |
//!
//! ## Work-steal candidates
//!
//! These are paths where the fork either lacks an equivalent or where
//! candle's existing implementation may be better; left as TODOs for
//! the ndarray fork or candle to evolve:
//!
//! - **bf16 / f16 dot product**: the fork has `simd::bf16_to_f32_batch`
//!   and `F32x16::mul_add` but no end-to-end `dot_bf16` / `dot_f16`.
//!   Candle's `cpu/mod.rs::vec_dot_{bf16,f16}` uses AVX2 FMA on packed
//!   half-precision lanes (`CurrentCpuBF16::vec_fma`) which is denser
//!   than a `widen-then-dot` round trip. Keep candle's impl for now;
//!   add `ndarray::hpc::dot_bf16` / `dot_f16` upstream when AVX-512 BF16
//!   instructions (`vdpbf16ps`) land in the fork.
//! - **Quantized k_quants kernels**: candle's `vec_dot_q4_0_q8_0` etc.
//!   are hand-tuned per-arch; the fork's `hpc::quantized::int8_gemm_f32`
//!   is a different shape (full GEMM, not per-block dot). Routing
//!   candle's q4_0 through `hpc::quantized::dequantize_q4_0_to_f32`
//!   followed by `int8_gemm_f32` is a port-up opportunity.
//! - **N-D axis-aware softmax**: the fork only exposes 1D
//!   `softmax_f32(&[f32], &mut [f32])`. We chunk by the last-dim stride
//!   inside [`softmax_f32_last_dim`]; an upstream `softmax_axis_f32`
//!   would let candle drop the chunking loop.
//! - **AMX BF16 → F32 gemm path**: implemented via
//!   `hpc::amx_matmul::matmul_bf16_to_f32` (Intel Sapphire Rapids+);
//!   we wire it but candle's existing gemm crate path doesn't have an
//!   AMX equivalent at all — this is a pure capability win.

use core::any::TypeId;

// ----------------------------------------------------------------------------
// Capability detection
// ----------------------------------------------------------------------------

/// Returns `true` when the running CPU has Intel AMX hardware enabled
/// (Sapphire Rapids+ on x86_64 Linux). Always `false` elsewhere.
#[inline]
pub fn amx_available() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        ndarray::hpc::amx_matmul::amx_available()
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        false
    }
}

// ----------------------------------------------------------------------------
// MatMul — f32 / f64 / bf16
// ----------------------------------------------------------------------------

/// f32 matmul `c = a · b` (column-major rs/cs strides) backed by
/// [`ndarray::hpc::amx_matmul::matmul_f32`] when AMX is available,
/// otherwise [`ndarray::linalg::general_mat_mul`]. Caller supplies
/// row-major `m × k` and `k × n` arrays.
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), &'static str> {
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err("matmul_f32: input slice too short");
    }

    use ndarray::{ArrayView2, ArrayViewMut2};
    let a_view = ArrayView2::from_shape((m, k), a).map_err(|_| "matmul_f32: a shape")?;
    let b_view = ArrayView2::from_shape((k, n), b).map_err(|_| "matmul_f32: b shape")?;
    let c_view: ArrayViewMut2<'_, f32> =
        ArrayViewMut2::from_shape((m, n), c).map_err(|_| "matmul_f32: c shape")?;

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        if ndarray::hpc::amx_matmul::amx_available() {
            return ndarray::hpc::amx_matmul::matmul_f32(a_view, b_view, c_view)
                .map_err(|_| "amx matmul_f32 failed");
        }
    }

    let mut c_view = c_view;
    ndarray::linalg::general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c_view);
    Ok(())
}

/// f64 matmul via [`ndarray::linalg::general_mat_mul`]. No AMX f64
/// path exists; this is identical to the scalar BLAS L3 path that the
/// fork's `backend::native` provides.
pub fn matmul_f64(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), &'static str> {
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err("matmul_f64: input slice too short");
    }
    use ndarray::ArrayView2;
    let a_view = ArrayView2::from_shape((m, k), a).map_err(|_| "matmul_f64: a shape")?;
    let b_view = ArrayView2::from_shape((k, n), b).map_err(|_| "matmul_f64: b shape")?;
    let mut c_view =
        ndarray::ArrayViewMut2::from_shape((m, n), c).map_err(|_| "matmul_f64: c shape")?;
    ndarray::linalg::general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c_view);
    Ok(())
}

/// BF16 × BF16 → F32 matmul via AMX (when available). Caller supplies
/// BF16 inputs as `u16` slices (the raw bit pattern); output is f32.
pub fn matmul_bf16_to_f32(
    a_bf16: &[u16],
    b_bf16: &[u16],
    c_f32: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), &'static str> {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        use ndarray::hpc::quantized::BF16;
        use ndarray::{ArrayView2, ArrayViewMut2};
        // BF16 is `repr(transparent)` over u16; reinterpret without copy.
        let a_bf: &[BF16] = unsafe { &*(a_bf16 as *const [u16] as *const [BF16]) };
        let b_bf: &[BF16] = unsafe { &*(b_bf16 as *const [u16] as *const [BF16]) };
        let a_view = ArrayView2::from_shape((m, k), a_bf).map_err(|_| "matmul_bf16: a shape")?;
        let b_view = ArrayView2::from_shape((k, n), b_bf).map_err(|_| "matmul_bf16: b shape")?;
        let c_view: ArrayViewMut2<'_, f32> =
            ArrayViewMut2::from_shape((m, n), c_f32).map_err(|_| "matmul_bf16: c shape")?;
        ndarray::hpc::amx_matmul::matmul_bf16_to_f32(a_view, b_view, c_view)
            .map_err(|_| "amx matmul_bf16_to_f32 failed")
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        let _ = (a_bf16, b_bf16, c_f32, m, n, k);
        Err("matmul_bf16_to_f32: AMX requires x86_64 Linux; fall back to widen-then-f32-gemm")
    }
}

// ----------------------------------------------------------------------------
// Reductions
// ----------------------------------------------------------------------------

#[inline]
pub fn sum_f32(s: &[f32]) -> f32 {
    ndarray::hpc::reductions::sum_f32(s)
}

#[inline]
pub fn sum_f64(s: &[f64]) -> f64 {
    ndarray::hpc::reductions::sum_f64(s)
}

#[inline]
pub fn mean_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::mean_f32(s)
}

#[inline]
pub fn max_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::max_f32(s)
}

#[inline]
pub fn min_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::min_f32(s)
}

#[inline]
pub fn argmax_f32(s: &[f32]) -> Option<usize> {
    ndarray::hpc::reductions::argmax_f32(s)
}

#[inline]
pub fn argmin_f32(s: &[f32]) -> Option<usize> {
    ndarray::hpc::reductions::argmin_f32(s)
}

#[inline]
pub fn nrm2_f32(s: &[f32]) -> f32 {
    ndarray::hpc::reductions::nrm2_f32(s)
}

// ----------------------------------------------------------------------------
// Activations
// ----------------------------------------------------------------------------

/// Softmax along the trailing dim. `src` and `dst` are flattened
/// row-major; the trailing dim has length `dim_m1`. Each row is
/// processed independently through
/// [`ndarray::hpc::activations::softmax_f32`].
pub fn softmax_f32_last_dim(src: &[f32], dst: &mut [f32], dim_m1: usize) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(dim_m1 > 0);
    debug_assert_eq!(src.len() % dim_m1, 0);

    for (src_row, dst_row) in src.chunks(dim_m1).zip(dst.chunks_mut(dim_m1)) {
        ndarray::hpc::activations::softmax_f32(src_row, dst_row);
    }
}

#[inline]
pub fn sigmoid_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::activations::sigmoid_f32(x, out);
}

#[inline]
pub fn log_softmax_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::activations::log_softmax_f32(x, out);
}

// ----------------------------------------------------------------------------
// VML (vector math: exp, ln, sqrt, tanh, ...)
// ----------------------------------------------------------------------------

#[inline]
pub fn exp_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::vml::vsexp(x, out);
}

#[inline]
pub fn ln_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::vml::vsln(x, out);
}

#[inline]
pub fn sqrt_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::vml::vssqrt(x, out);
}

#[inline]
pub fn tanh_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::vml::vstanh(x, out);
}

#[inline]
pub fn abs_f32_slice(x: &[f32], out: &mut [f32]) {
    ndarray::hpc::vml::vsabs(x, out);
}

// ----------------------------------------------------------------------------
// Elementwise (in-place, slice-level) — the user-flagged go-to path
// ----------------------------------------------------------------------------

#[inline]
pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
    ndarray::simd::add_f32_inplace(dst, src);
}

#[inline]
pub fn sub_f32_inplace(dst: &mut [f32], src: &[f32]) {
    ndarray::simd::sub_f32_inplace(dst, src);
}

#[inline]
pub fn mul_f32_inplace(dst: &mut [f32], src: &[f32]) {
    ndarray::simd::mul_f32_inplace(dst, src);
}

#[inline]
pub fn div_f32_inplace(dst: &mut [f32], src: &[f32]) {
    ndarray::simd::div_f32_inplace(dst, src);
}

#[inline]
pub fn scale_f32_inplace(a: &mut [f32], scalar: f32) {
    ndarray::simd::scale_f32_inplace(a, scalar);
}

#[inline]
pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
    ndarray::simd::add_f64_inplace(dst, src);
}

// ----------------------------------------------------------------------------
// Vector dot — `array_windows::<PREFERRED_F32_LANES>()` + `F32x16::mul_add`
// (the canonical fork pattern the user flagged: array_windows + mul_add)
// ----------------------------------------------------------------------------

/// Dot product `Σ a[i] · b[i]` using `F32x16::mul_add` over
/// `array_windows::<PREFERRED_F32_LANES>` chunks with a scalar tail.
/// This is the canonical fork pattern (`simd.rs:140` doc) and prefers
/// FMA throughput over rolled accumulation.
#[inline]
pub fn vec_dot_f32_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Defer to the fork's `BlasFloat::backend_dot` which is the
    // dispatch entry point for tier-correct SIMD dot (AVX-512 / AVX2 /
    // NEON / scalar) and already implements the array_windows + mul_add
    // pattern internally.
    <f32 as ndarray::backend::BlasFloat>::backend_dot(a, b)
}

#[inline]
pub fn vec_dot_f64_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    <f64 as ndarray::backend::BlasFloat>::backend_dot(a, b)
}

// ----------------------------------------------------------------------------
// Type-specialization helpers (mirror burn-ndarray's pattern for routing
// generic `T: WithDType` ops down to concrete f32/f64 fast paths).
// ----------------------------------------------------------------------------

/// Returns `Some(&[f32])` iff `T == f32`. Sound because monomorphisation
/// makes `&[T]` and `&[f32]` the same type in that case.
#[inline]
pub fn try_as_f32_slice<T: 'static>(s: &[T]) -> Option<&[f32]> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(unsafe { &*(s as *const [T] as *const [f32]) })
    } else {
        None
    }
}

#[inline]
pub fn try_as_f32_slice_mut<T: 'static>(s: &mut [T]) -> Option<&mut [f32]> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(unsafe { &mut *(s as *mut [T] as *mut [f32]) })
    } else {
        None
    }
}

#[inline]
pub fn try_as_f64_slice<T: 'static>(s: &[T]) -> Option<&[f64]> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        Some(unsafe { &*(s as *const [T] as *const [f64]) })
    } else {
        None
    }
}

#[inline]
pub fn try_as_f64_slice_mut<T: 'static>(s: &mut [T]) -> Option<&mut [f64]> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        Some(unsafe { &mut *(s as *mut [T] as *mut [f64]) })
    } else {
        None
    }
}

// ----------------------------------------------------------------------------
// Quantized GEMM (INT8) — port-up opportunity
// ----------------------------------------------------------------------------

/// INT8 GEMM `c = a · b` with f32 output. Routes through
/// [`ndarray::hpc::quantized::int8_gemm_f32`] which dispatches
/// AMX → AVX-512-VNNI → scalar at runtime. `a` is `u8` (unsigned),
/// `b` is `i8` (signed); this matches the `cblas_gemm_s8s8s32`
/// convention. Per-tensor `scale_a`, `zero_point_a`, `scale_b` are used
/// to dequantize the i32 accumulator to f32.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn int8_gemm_f32(
    a: &[u8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    zero_point_a: i32,
    scale_b: f32,
) {
    ndarray::hpc::quantized::int8_gemm_f32(a, b, c, m, n, k, scale_a, zero_point_a, scale_b);
}

/// BF16 × BF16 → F32 GEMM through the fork's
/// [`ndarray::hpc::quantized::bf16_gemm_f32`] (uses AMX BF16 tile
/// instructions when available, falls back to widen-then-sgemm).
#[inline]
pub fn bf16_gemm_f32(
    a_bf16: &[u16],
    b_bf16: &[u16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    use ndarray::hpc::quantized::BF16;
    // BF16 is repr(transparent) over u16; reinterpret without copy.
    let a = unsafe { &*(a_bf16 as *const [u16] as *const [BF16]) };
    let b = unsafe { &*(b_bf16 as *const [u16] as *const [BF16]) };
    ndarray::hpc::quantized::bf16_gemm_f32(a, b, c, m, n, k, alpha, beta);
}
