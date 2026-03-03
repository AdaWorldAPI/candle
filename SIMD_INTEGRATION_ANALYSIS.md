# How Burn, Candle, and ORT Can Consume rustynum CPU SIMD Acceleration

> Analysis: March 2026 — Branch `claude/compare-rustynum-ndarray-5ePRn`

## rustynum SIMD Surface Area

rustynum-core exposes a comprehensive CPU SIMD acceleration layer on **stable Rust 1.93+** (no nightly):

| Module | SIMD ISA | Operations |
|--------|----------|------------|
| `simd_compat` | AVX-512 (F32x16, F64x8, U8x64, I32x16) + AVX2 (F32x8, F64x4) | Wrapper types with full operator overloads, `splat/from_slice/reduce_sum/mul_add` |
| `simd` / `simd_avx2` | AVX-512 / AVX2 | `dot_f32`, `dot_f64`, `axpy_f32/f64`, `scal_f32/f64`, `asum_f32/f64`, `nrm2_f32/f64`, `hamming_distance`, `hamming_batch` |
| `rustyblas::level1` | Via simd | CBLAS L1: `sdot/ddot/saxpy/daxpy/sscal/dscal/snrm2/dnrm2/sasum/dasum/isamax/idamax/scopy/dcopy/sswap/dswap` |
| `rustyblas::level2` | Via simd | CBLAS L2: `sgemv/dgemv/sger/dger/ssymv/dsymv/strmv/dtrmv/strsv/dtrsv` |
| `rustyblas::level3` | Via simd | CBLAS L3: `sgemm/dgemm/ssymm/dsymm/ssyrk/dsyrk/strsm/dtrsm` — 6x16 microkernel, 3-level cache blocking |
| `rustyblas::bf16_gemm` | AVX-512 BF16 | Mixed-precision BF16→FP32 GEMM |
| `rustyblas::int8_gemm` | AVX-512 VNNI | INT8 GEMM with `VPDPBUSD`, quantization utils |
| `compute` | CPUID | Runtime detection: AVX-512F/BW/VNNI/BF16/VPOPCNTDQ/BITALG, AVX2, AMX (detect only) |
| `blackboard` | — | 64-byte aligned shared memory arena, zero-copy borrow |

**Feature flags**: `avx512` (default), `avx2`, `mkl` (MKL FFI), `libxsmm` (JIT GEMM).

---

## 1. Burn — Already Integrated (burn-rustynum)

### Current State

`burn-rustynum` (at `crates/burn-rustynum/`) is a full Burn `Backend` implementation that already consumes rustynum SIMD:

```toml
# burn-rustynum/Cargo.toml
[dependencies]
rustynum-core = { path = "../../../rustynum/rustynum-core", features = ["avx512"] }
rustyblas = { path = "../../../rustynum/rustyblas", features = ["avx512"] }
```

**SIMD-accelerated hot paths** (gated by `#[cfg(feature = "simd")]`):

| Burn Op | rustynum Call | SIMD Width |
|---------|---------------|------------|
| `float_add` | `rustynum_core::simd::add_f32_vec` | f32x16 (AVX-512) |
| `float_sub` | `rustynum_core::simd::sub_f32_vec` | f32x16 |
| `float_mul` | `rustynum_core::simd::mul_f32_vec` | f32x16 |
| `float_div` | `rustynum_core::simd::div_f32_vec` | f32x16 |
| `float_matmul` | `rustyblas::level3::sgemm` | 6x16 cache-blocked GEMM |
| `float_add_scalar` | `rustynum_core::simd::add_f32_scalar` | f32x16 |
| `float_mul_scalar` | `rustynum_core::simd::mul_f32_scalar` | f32x16 |

### Further Integration Opportunities

1. **Reduction ops** — `float_sum`, `float_mean`, `float_max` currently use scalar loops; could use `rustynum_core::simd::asum_f32` / custom reduce kernels
2. **Norm ops** — `float_powf` with p=2 could dispatch to `nrm2_f32`
3. **BF16 support** — `dtype_usage` currently returns empty for BF16; `rustyblas::bf16_gemm` could enable BF16 tensors
4. **INT8 quantized inference** — `QuantizedTensorOps` is stubbed; `rustyblas::int8_gemm` could provide quantized matmul
5. **Blackboard integration** — Use `Blackboard` as the tensor storage arena for zero-copy across ops

---

## 2. Candle — Drop-In SIMD Kernel Replacement

### Architecture Analysis

Candle's CPU backend uses three layers:

```
candle-core/src/cpu/mod.rs    → Cpu<N> / CpuF16<N> / CpuBF16<N> traits (SIMD primitives)
candle-core/src/cpu/avx.rs    → CurrentCpu impl using __m256 (AVX2, 8 lanes)
candle-core/src/cpu_backend/  → CpuStorage dispatch (matmul → gemm crate or MKL)
```

**Key trait** (`cpu/mod.rs:7-21`):
```rust
trait Cpu<const ARR: usize> {
    type Unit;               // e.g., __m256 for AVX2
    type Array;              // [__m256; ARR]
    const STEP: usize;       // 32 (4 × 8 lanes)
    const EPR: usize;        // 8 (elements per register)
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    // ...
}
```

**Current SIMD**: AVX2 only (`__m256`, 8 f32 lanes). No AVX-512 path at all.

**MatMul**: Uses `gemm` crate (by `sarah-ek`) when neither MKL nor Accelerate is enabled.

### How rustynum Can Be Consumed

#### Option A: Replace `Cpu` Trait Implementation with AVX-512

Create `candle-core/src/cpu/avx512.rs` implementing `Cpu<ARR>` with `rustynum_core::simd_compat::F32x16`:

```rust
use rustynum_core::simd_compat::{f32x16, f64x8};

pub struct CurrentCpu {}
const STEP: usize = 64;  // 4 × 16 lanes
const EPR: usize = 16;   // f32x16
const ARR: usize = STEP / EPR;  // 4

impl Cpu<ARR> for CurrentCpu {
    type Unit = __m512;    // or use f32x16 wrapper
    type Array = [__m512; ARR];
    const STEP: usize = STEP;
    const EPR: usize = EPR;

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        _mm512_loadu_ps(mem_addr)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        _mm512_fmadd_ps(b, c, a)
    }
    // ...
}
```

This doubles the SIMD width for `vec_dot_f32`, `vec_sum`, `vec_dot_f16`, `vec_dot_bf16` — all hot paths.

#### Option B: Replace `gemm` Crate with rustyblas for MatMul

In `cpu_backend/mod.rs:1357-1438`, candle uses `gemm::gemm()`. This can be replaced:

```rust
#[cfg(feature = "rustynum")]
{
    rustyblas::level3::sgemm(
        Layout::RowMajor,
        if lhs_rs < lhs_cs { Transpose::Trans } else { Transpose::NoTrans },
        if rhs_rs < rhs_cs { Transpose::Trans } else { Transpose::NoTrans },
        m, n, k, 1.0, lhs_p, lda, rhs_p, ldb, 0.0, dst_p, n,
    );
}
```

Benefits:
- rustyblas SGEMM uses AVX-512 6x16 microkernels (vs gemm's AVX2 path)
- 3-level cache blocking tuned for Sapphire Rapids / EPYC
- BF16 GEMM support (`bf16_gemm_f32`) for candle's BF16 tensors
- INT8 GEMM for quantized inference

#### Option C: Feature-Gated Hybrid

Add a `rustynum` feature to candle-core:
```toml
[features]
rustynum = ["dep:rustynum-core", "dep:rustyblas"]
```

Gate the dispatch in `cpu_backend/mod.rs`:
```rust
#[cfg(feature = "rustynum")]
fn matmul_f32(...) { rustyblas::level3::sgemm(...) }

#[cfg(all(not(feature = "rustynum"), not(feature = "mkl"), not(feature = "accelerate")))]
fn matmul_f32(...) { gemm::gemm(...) }
```

### Impact Assessment

| Operation | Current (AVX2/gemm) | With rustynum (AVX-512) | Speedup |
|-----------|---------------------|-------------------------|---------|
| vec_dot_f32 | 8 lanes × 4 accum | 16 lanes × 4 accum | ~2x |
| SGEMM (1024×1024) | gemm AVX2 | rustyblas AVX-512 6x16 | ~1.5-2x |
| BF16 GEMM | Not available | bf16_gemm_f32 | New capability |
| INT8 quantized | Not available | int8_gemm_f32 (VNNI) | New capability |

---

## 3. ORT — Custom Operator Domain

### Architecture Analysis

ORT is a Rust wrapper around ONNX Runtime's C API. It does **not** expose the internal MLAS SIMD kernels. Instead, it provides a **custom operator system**:

```rust
// Operator trait (ort/src/operator/mod.rs:35)
pub trait Operator: Send {
    fn name(&self) -> &str;
    fn execution_provider_type(&self) -> Option<&str> { None } // None = CPU
    fn inputs(&self) -> Vec<OperatorInput>;
    fn outputs(&self) -> Vec<OperatorOutput>;
    fn create_kernel(&self, attrs: &KernelAttributes) -> Result<Box<dyn Kernel>>;
}

// Kernel trait (ort/src/operator/kernel.rs:19)
pub trait Kernel {
    fn compute(&mut self, ctx: &KernelContext) -> Result<()>;
}
```

The `KernelContext` provides:
- `ctx.input(idx)` → `ValueRef` → `try_extract_tensor::<f32>()` → `&[f32]` slice
- `ctx.output(idx, shape)` → `ValueRefMut` → `try_extract_tensor_mut::<f32>()` → `&mut [f32]` slice
- `ctx.par_for(total, batches, f)` → parallel execution within ONNX Runtime's thread pool

### How rustynum Can Be Consumed

Create a custom operator domain `"rustynum.simd"` with SIMD-accelerated kernels:

```rust
use ort::operator::*;
use rustynum_core::simd;
use rustyblas::level3;

struct RustyNumMatMul;

impl Operator for RustyNumMatMul {
    fn name(&self) -> &str { "MatMul" }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(|ctx: &KernelContext| {
            let a = ctx.input(0)?.unwrap();
            let b = ctx.input(1)?.unwrap();
            let (a_shape, a_data) = a.try_extract_raw_tensor::<f32>()?;
            let (b_shape, b_data) = b.try_extract_raw_tensor::<f32>()?;

            let m = a_shape[0] as usize;
            let k = a_shape[1] as usize;
            let n = b_shape[1] as usize;

            let mut out = ctx.output(0, vec![m as i64, n as i64])?.unwrap();
            let (_, out_data) = out.try_extract_raw_tensor_mut::<f32>()?;

            // AVX-512 SGEMM via rustyblas
            level3::sgemm(
                Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                m, n, k, 1.0, a_data, k, b_data, n, 0.0, out_data, n,
            );

            Ok(())
        }))
    }
}

// Register with session:
let domain = OperatorDomain::new("rustynum.simd")?
    .add(RustyNumMatMul)?
    .add(RustyNumDotProduct)?
    .add(RustyNumBF16Gemm)?;

let session = Session::builder()?
    .with_operators(domain)?
    .commit_from_file("model.onnx")?;
```

### Practical Integration Patterns

| Pattern | Description | Complexity |
|---------|-------------|------------|
| **Custom op domain** | Register `rustynum.simd:MatMul`, `rustynum.simd:DotProduct`, etc. Models must use custom op names. | Medium |
| **Pre/post processing** | Use rustynum SIMD for feature extraction (BF16 Hamming, HDC distances) before/after ONNX inference. | Low |
| **Hybrid pipeline** | ONNX Runtime for graph execution + rustynum for custom distance/scoring kernels that ORT doesn't optimize. | Low |
| **INT8 quantized ops** | Custom INT8 GEMM op using `rustyblas::int8_gemm` for quantized layers ONNX RT doesn't optimize. | Medium |
| **BF16 mixed precision** | Custom BF16 GEMM op where ORT's MLAS doesn't have BF16 path on x86. | Medium |

### Key Limitation

ORT custom operators require the ONNX model to reference the custom op domain. For standard models (e.g., `MatMul` in default ONNX domain), you **cannot** override the built-in MLAS kernel — only add operators in custom domains.

**Workaround**: Use ONNX graph transformation tools to replace standard ops with custom domain equivalents before inference.

---

## Summary: Integration Feasibility Matrix

| Framework | Integration Path | Effort | SIMD Benefit |
|-----------|-----------------|--------|--------------|
| **Burn** | Already done (`burn-rustynum` backend) — extend with BF16/INT8/reduction | Low (incremental) | High — direct kernel replacement |
| **Candle** | Replace `Cpu` trait + `gemm` crate with rustynum-core + rustyblas | Medium (new feature flag) | High — AVX-512 doubles SIMD width, adds BF16/INT8 |
| **ORT** | Custom operator domain for non-standard ops; pre/post-processing | Medium (model changes needed) | Moderate — only for custom ops, can't replace built-in MLAS |

### Recommended Priority

1. **Candle** — highest impact: candle has no AVX-512 path; rustynum adds 2x SIMD width + BF16/INT8
2. **Burn** — incremental: extend existing backend with BF16 GEMM, INT8 quantized, reduction SIMD
3. **ORT** — targeted: custom ops for HDC/BF16 distance, pre/post processing; standard ops already optimized by MLAS
