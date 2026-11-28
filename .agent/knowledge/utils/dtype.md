# DTYPE — the cross-layer data-type enum

**In one line**: native operators (CCE and host) pass dtypes as `sgl::kernel_npu::utils::DTYPE`; the Torch adapter, the host launcher and the device kernel all read the same header. Do not pass raw `uint8_t` codes across these layers.

**Scope**: repository `main`; device `Ascend910_9382` (arch 2201 / `dav-2201`); CANN 9.2.1; bisheng 15.0.5; torch 2.13.0+cpu / torch_npu 2.13.0. Re-verify with section 7 after changing device or CANN version.

## 1. Definition

- Header: `csrc/utils/dtype.h`
- Namespace: `sgl::kernel_npu::utils`
- Enum (`csrc/utils/dtype.h:24`):

```cpp
enum class DTYPE : uint8_t {
    FP32 = 0,  // 32-bit floating point
    FP16 = 1,  // IEEE 16-bit floating point
    BF16 = 2,  // bfloat16
};
```

- Count: `inline constexpr uint8_t kDTypeCount = 3;` (`csrc/utils/dtype.h:41`)
- The header depends only on `<cstdint>`, so it can be included by bisheng-compiled device code and by g++-compiled host code alike.

## 2. Design constraints

1. **`uint8_t` underlying type**: the `extern "C"` launcher takes `DTYPE` directly; both translation units see the same declaration, so the argument is passed as its underlying type.
2. **Dense values, append only**: add new entries at the end and bump `kDTypeCount`; never renumber values that have shipped, because both sides of a build must agree.
3. **No persistence**: enum values are not written to files or configs, so a renumber only affects one build. The append-only rule still applies to avoid mixing artifacts built from different header versions.
4. **Device-code safe**: only foundational headers such as `<cstdint>` are allowed; no STL containers and no ATen.

## 3. Helpers

| Function | Semantics | Use when |
| --- | --- | --- |
| `dtype_to_code(DTYPE)` | enum → stable `uint8_t` code | an integer code is needed: ABI tables, logging, kernel metadata |
| `is_valid_dtype_code(uint8_t)` | `code < kDTypeCount` | validating an integer code coming from outside |
| `is_16bit_dtype(DTYPE)` | FP16 / BF16 → true | operators that share the 16-bit packed layout (two elements per 32-bit word) |

All three are `constexpr`, so they evaluate at compile time. The silu_and_mul operator passes the enum directly and does not call the code helpers; they are kept as the extension surface.

## 4. Cross-layer flow (silu_and_mul as the reference)

```text
Python  swiglu(x, impl="cce")
  └─ torch.ops.cce.silu_and_mul                       torch_adapter.cpp:24-31
       at::kHalf → DTYPE::FP16, at::kBFloat16 → DTYPE::BF16
  └─ sgl_launch_silu_and_mul(..., DTYPE, ...)         op_host/silu_and_mul.cpp:12-26
       support-set validation (BF16/FP16) + block_dim computation
  └─ cce_launch_silu_and_mul(..., DTYPE, ...)         op_kernel/silu_and_mul.cce:202-217
       dispatch to silu_and_mul_fp16 / silu_and_mul_bf16
  └─ silu_rows<DT, TT, RT>                            template parameter DT selects storage type and conversions
```

Notes:

- The declaration in `op_host` and the definition in the `.cce` must stay byte-for-byte compatible in signature.
- Do not add a `uint8_t` bridge "for safety": two representations in one build invite inconsistency.
- The device kernel uses a template parameter `DT` (0/1) for the storage type; that value is internal and unrelated to the enum values.

## 5. Measured dav-2201 support matrix

| dtype | Storage | Vector arithmetic | Conclusion |
| --- | --- | --- | --- |
| FP32 | supported | **native** (`vexp`/`vrec`/`vmul` accept f32) | can be instantiated directly, no conversion stage |
| FP16 | supported | **native** (`half` is accepted) | can compute natively; this repository widens to f32 for accuracy |
| BF16 | supported | **not available** (CCE unary/binary intrinsics reject `bfloat16_t`) | must widen with `vconv_bf162f32`, compute in f32, and narrow with `vconv_f322bf16r` (RNE) |

Evidence:

- **FP32**: a full FP32 silu-and-mul probe (`vmins/vexp/vadds/vrec/Newton/vmul` plus f32 GM↔UB copies, 3×2048) matched a double reference with `worst_rel = 7.5e-06`, passing a 1e-5 threshold.
- **FP16**: a probe compiled `vmul`/`vexp` on `half` buffers without errors.
- **BF16**: the same arithmetic probe on `bfloat16_t` fails to compile with
  `error: the 1st parameter maybe need a type '__ubuf__ half *'`.
- The probes were disposable diagnostics under the task workspace (`build/cce-work/`, gitignored); section 7 describes how to reproduce the checks.

## 6. Extending with a new dtype

1. `csrc/utils/dtype.h`: append the value, bump `kDTypeCount`, and add it to `is_16bit_dtype` if it is a 16-bit storage type.
2. Host launcher (`op_host/*.cpp`): update the supported-set check and error messages; adjust block/tiling decisions if needed.
3. Torch adapter (`torch_adapter.cpp`): add the `at::ScalarType` → `DTYPE` mapping.
4. Device kernel (`.cce`): add the storage type and conversion path (see `vconv_bf162f32` / `vconv_f322f16r`); compute in f32 after conversion.
5. Public Python backend: relax/tighten the dtype check in `operators/<domain>/<op>.py`.
6. Tests: parameterize by dtype and add edge cases (saturation, zero, non-aligned widths).
7. Docs: bilingual API docs plus the dtype table in the operator README.
8. Verification: static asserts → device probe → wheel build → full pytest run.

## 7. How to re-verify

```bash
# 1) Header semantics (host compiler is enough)
cat > /tmp/dtype_check.cpp <<'EOF'
#include "csrc/utils/dtype.h"
using namespace sgl::kernel_npu::utils;
static_assert(dtype_to_code(DTYPE::FP32) == 0);
static_assert(dtype_to_code(DTYPE::FP16) == 1);
static_assert(dtype_to_code(DTYPE::BF16) == 2);
static_assert(is_valid_dtype_code(kDTypeCount - 1) && !is_valid_dtype_code(kDTypeCount));
static_assert(is_16bit_dtype(DTYPE::FP16) && is_16bit_dtype(DTYPE::BF16));
int main() { return 0; }
EOF
clang++ -std=c++17 -Wall -Wextra -Werror -I<repo> -fsyntax-only /tmp/dtype_check.cpp

# 2) Device level: can one dtype be instantiated? (inside the target container)
bisheng --npu-arch=dav-2201 -std=c++17 -O2 -c probe.cce -I"$ASCEND_HOME_PATH/include" -o /tmp/probe.o

# 3) End to end
python3 setup.py --arch-version 2201 --ops <op_id> --without-third-party -j 16
PYTHONPATH=<site> python -m pytest tests/python/operators/<domain>/test_<op>.py \
  --impl=cce --impl-ref=torch_native -m "" -q --npu-device=<id>
```

Reference result (2026-11-28, `Ascend910_9382`, CANN 9.2.1): `test_swiglu.py` passes all 29 cases (BF16/FP16 accuracy, saturation, rejected inputs, performance guard).

## 8. Pitfalls

- **Gaps in the enum**: `is_valid_dtype_code` relies on `code < kDTypeCount`, so a gap lets an invalid code through. Renumber and bump `kDTypeCount` when removing a type.
- **Renumbering shipped values**: breaks the cross-TU ABI; both sides of one build must use the same header version.
- **Assuming BF16 arithmetic works**: this architecture rejects it at compile time; do not bypass with `reinterpret_cast`.
- **Confusing HF32 with this enum**: `USE_HF32_MODE` in `csrc/attention/chunk_kda_fwd` is a Cube precision policy macro; HF32 has been removed from `DTYPE`.
- **Verifying only one build path**: the refactor `.cce` build tracks `dtype.h` through the dependency glob at `csrc/cmake/cce_kernels.cmake:74`, while the legacy build does not compile `csrc/cce/`; check both paths separately.

## 9. Change log

- 2026-11-28: created `dtype.h` (FP32/HF32/FP16/BF16) and wired it into silu_and_mul; then removed HF32, renumbered to FP32=0/FP16=1/BF16=2 with `kDTypeCount=3`, switched the launcher to take `DTYPE` directly, and recorded the measured dav-2201 support matrix.

## 10. Related files

| File | Relationship |
| --- | --- |
| `csrc/utils/dtype.h` | definition |
| `csrc/cce/silu_and_mul/torch_adapter.cpp` | Torch dtype → DTYPE mapping |
| `csrc/cce/silu_and_mul/op_host/silu_and_mul.cpp` | host-side validation and forwarding |
| `csrc/cce/silu_and_mul/op_kernel/silu_and_mul.cce` | launcher dispatch by DTYPE; template selects storage type |
| `csrc/cmake/cce_kernels.cmake:74` | dependency tracking of `csrc/utils/*.h` for CCE compilations |
| `csrc/cce/silu_and_mul/README.md`, `README.zh.md` | operator-level dtype support notes |
