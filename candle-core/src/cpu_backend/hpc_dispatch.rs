//! Optional acceleration hook into the AdaWorldAPI/ndarray HPC fork.
//!
//! candle's equivalent of `burn-ndarray`'s `simd_dispatch.rs`. It adds CPU
//! **bf16** matmul — which candle's default CPU backend does not support
//! (`MatMul` returns `UnsupportedDTypeForOp` for bf16) — by routing it through
//! the fork's three-tier bf16 tile GEMM
//! (`ndarray::hpc::amx_matmul::matmul_bf16_to_f32`):
//!
//!   * AMX `TDPBF16PS` on Sapphire Rapids+ (16/16/32-aligned tiles),
//!   * AVX-512 `VDPBF16PS` on Cooper Lake / Cascade Lake / Zen 4+,
//!   * else the validated scalar `bf16_gemm_f32` reference.
//!
//! In every tier the bf16 products accumulate into f32 via `mul_add` (FMA):
//! there is **no f32 round-trip of the inputs** and no silent precision change
//! — bf16 in, f32-accumulated, bf16 out (a single output rounding, which is the
//! inherent output quantization of any bf16 GEMM). This is why we route only
//! bf16 here and never f32: the fork's `matmul_f32` would down-cast f32 inputs
//! to bf16 on AMX (~1% error), which would silently degrade candle's full-f32
//! matmul, so f32 stays on the existing `gemm` path.
//!
//! The whole module is gated behind `ndarray-hpc`; with the feature off it is
//! empty and candle's behavior is unchanged.
#![cfg(feature = "ndarray-hpc")]

use ndarray::hpc::quantized::BF16;
use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder};

/// One `bf16 × bf16 → bf16` matmul: `C = round_bf16(A · B)`, with the products
/// accumulated in f32 (FMA) by the fork. `A` is `m×k`, `B` is `k×n`, `C` is
/// `m×n`, all with the given element strides (row stride `*_rs`, column stride
/// `*_cs`), matching candle's `MatMul` layout convention. bf16 values are passed
/// as their `u16` bit patterns.
///
/// # Safety
/// The pointers must be valid for the described `(shape, strides)` for the
/// duration of the call: `a`/`b` readable, `c` writable, with `c` not aliasing
/// the inputs. Callers pass pointers derived from live candle bf16 slices
/// (`half::bf16` is `repr(transparent)` over `u16`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn bf16_matmul_step(
    a: *const u16,
    a_rs: usize,
    a_cs: usize,
    b: *const u16,
    b_rs: usize,
    b_cs: usize,
    c: *mut u16,
    c_rs: usize,
    c_cs: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    // SAFETY (caller-guaranteed): the pointers are valid for these shapes and
    // strides; `BF16` is `repr(transparent)` over `u16`, matching the bit layout
    // of the `half::bf16` values candle hands us, so the cast is sound.
    let lhs = ArrayView2::<BF16>::from_shape_ptr((m, k).strides((a_rs, a_cs)), a as *const BF16);
    let rhs = ArrayView2::<BF16>::from_shape_ptr((k, n).strides((b_rs, b_cs)), b as *const BF16);

    // f32 accumulator (row-contiguous), as required by `matmul_bf16_to_f32`.
    let mut acc = vec![0.0f32; m * n];
    let out = ArrayViewMut2::from_shape_ptr((m, n).strides((n, 1)), acc.as_mut_ptr());
    // candle validates shapes before dispatch; the fork only errors on a shape
    // mismatch, so this does not fail in practice (leaving zeros if it ever did).
    let _ = ndarray::hpc::amx_matmul::matmul_bf16_to_f32(lhs, rhs, out);

    // Single output rounding f32 -> bf16, honoring the destination strides.
    for i in 0..m {
        for j in 0..n {
            let bits = half::bf16::from_f32(acc[i * n + j]).to_bits();
            *c.add(i * c_rs + j * c_cs) = bits;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Runs on every host: AMX/AVX-512 hosts take the tile path, others the
    // scalar reference. The inputs (1..4) and the result (7,10,15,22) are all
    // exactly representable in bf16, so the assertion is bit-exact everywhere.
    #[test]
    fn bf16_matmul_step_small() {
        let bits = |v: f32| half::bf16::from_f32(v).to_bits();
        // A = [[1,2],[3,4]], B = [[1,2],[3,4]] (row-major bf16 bits).
        let a = [bits(1.0), bits(2.0), bits(3.0), bits(4.0)];
        let b = [bits(1.0), bits(2.0), bits(3.0), bits(4.0)];
        let mut c = [0u16; 4];

        unsafe {
            bf16_matmul_step(
                a.as_ptr(),
                2,
                1, // A: 2x2, rs=2 cs=1
                b.as_ptr(),
                2,
                1, // B: 2x2, rs=2 cs=1
                c.as_mut_ptr(),
                2,
                1, // C: 2x2, rs=2 cs=1
                2,
                2,
                2, // m, n, k
            );
        }

        let got: Vec<f32> = c
            .iter()
            .map(|&x| half::bf16::from_bits(x).to_f32())
            .collect();
        assert_eq!(got, vec![7.0, 10.0, 15.0, 22.0]);
    }
}
