# batch_matmul_transpose

## Hardware support

| Implementation | arch22 (A2/A3, 2201) | arch35 (A5, 3510) |
| --- | --- | --- |
| `ascendc` | 910C only | Unsupported |
| `torch_native` | Supported | Supported |

The AscendC operator supports only 910C (`ascend910_93`); other arch22 platforms
are unsupported.

## Signature

```python
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

def batch_matmul_transpose(
    tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None, *, impl="auto"
):
    ...
```

`auto` selects `ascendc`, with no fallback. The function writes `tensor_c` and
returns **None**, preserving the legacy `torch.ops.npu.batch_matmul_transpose`
contract. The legacy build and import path remain available separately.

## Description

Compute a batched matrix product with interleaved token and batch axes:

$$
C_{m,b,n} = \sum_{k=0}^{K-1} A_{m,b,k} W_{b,k,n}.
$$

Here $M$ is the token count, $B$ the independent matrix/head count, $K$ the
reduction dimension, and $N$ the output feature dimension. Both implementations
accumulate in FP32 and store the result in the input dtype.

For explicit NZ weights, the supplied four-dimensional ND tensor contains
$W^{NZ}_{b,j,k,r}=W_{b,k,16j+r}$. This is a physical packing convention, not a
request to convert an opaque `torch_npu` storage format. For example, pack ND
weights with `w.reshape(B, K, N // 16, 16).permute(0, 2, 1, 3).contiguous()`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `tensor_a` | Tensor | `[M, B, K]` | FP16 / BF16 | Required | Contiguous; `1 <= M <= 1024`. |
| `tensor_b` | Tensor | ND: `[B, K, N]`; NZ: `[B, N/16, K, 16]` | Same as A | Required | Contiguous; NZ requires K and N divisible by 16. |
| `tensor_c` | Tensor | `[M, B, N]` | Same as A | Required | Caller-allocated, contiguous output, overwritten in place; must not overlap A or B. |
| `format_mode` | str / None | — | — | `None` | `None` means `"ND"`; accepts `"ND"` and `"NZ"`. |
| `quant_mode` | str / None | — | — | `None` | Accepts `per_channel_symm`, `per_channel_asymm`, `per_token_symm`; None means the first. Legacy compatibility selector; **does not enable quantization** for these floating-point kernels. |
| `impl` | str | — | — | `"auto"` | Keyword-only; `auto`, `ascendc`, or `torch_native`. |

| Return value | Type | Shape | Dtype | Side effects |
| --- | --- | --- | --- | --- |
| None | NoneType | — | — | Writes C; A and B remain unchanged. |

All dimensions must be positive. Each tensor has at most `UINT32_MAX` elements.
All tensors must have the same dtype and device. `ascendc` requires an NPU;
directly loading the torch-native implementation also permits CPU algebra checks.
The public package itself requires its normal NPU/native runtime prerequisites.
Passing unsupported layouts, dtypes, shapes or mode names raises an error.

## ascendc backend

The arch22 implementation preserves the legacy cube kernel, tiling cost model,
swizzle selection and packed tiling layout. Host scheduling runs through a CANN
Tiling callback, retaining the original `HardwareInfo()` constructor and
`PlatformInfo::Instance()` hardware lookup. CANN owns the tiling
storage; the old process-global 1024-slot tiling tensor and synchronous host-to-
device copies are not used. The original `M <= 1024` interface limit is retained.
The Torch adapter uses the bundled ACLNN implementation and current NPU stream.
The kernel's shared helpers still come from `csrc/mla_preprocess` and `csrc/utils`.

The current SGLang MLA callers allocate `[M, B, N]` output tensors with the
required dtype before invoking the Torch operator. This integration uses the
ACLNN single-operator path and registers no handwritten shape/type inference.
The original private file layout is retained: `op_host/batch_matmul_transpose.cpp`
contains OpDef and the CANN Tiling callback; `op_host/tiling/tiling_data.h` holds
the shared declarations and packed data, and `tiling_data.cpp` implements
scheduling and directly reuses `csrc/utils/common_tiling.h`. The original shared
scheduling interfaces are retained; Torch checks are conditional on Torch builds.
No scheduling helpers are copied into the operator.
Both Host sources participate in codegen and Tiling compilation. The device
source keeps its original name, `op_kernel/batch_matmul_transpose_kernel.cpp`.
Only CANN entry/registration and relocated include paths require adaptation;
the computation and scheduling formulas are retained.
See the [Ascend C operator prototype guide](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/design_and_implementation/operator_prototype_definition.html)
for the distinction between ACLNN calls and graph inference registration.

## torch_native backend

Uses independent PyTorch FP32 `bmm`, unpacks NZ explicitly, casts to the output
dtype and copies into C. It does not call a custom operator or another backend.
Different accumulation orders can cause numerical differences.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

a = torch.randn(3, 2, 32, device="npu", dtype=torch.float16)
b = torch.randn(2, 32, 48, device="npu", dtype=a.dtype)
c = torch.empty(3, 2, 48, device="npu", dtype=a.dtype)
assert batch_matmul_transpose(a, b, c, impl="ascendc") is None
reference = torch.empty_like(c)
batch_matmul_transpose(a, b, reference, impl="torch_native")
torch.testing.assert_close(c, reference, atol=1e-3, rtol=1e-3)
```

Build from the repository root with
`python3 setup.py --arch-version 2201 --ops all`.
The current build verifies artifacts for both arch22 CANN targets, so build the
complete migrated operator set when testing this 910C-only operator. This does
not extend this operator's hardware support beyond 910C.
Install the produced wheel before testing the migrated ACLNN path:
`pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl`.
Run accuracy tests with
`python -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py --enable-perf=false -v -rs`.
Omit `--enable-perf` to profile the selected BF16 MLA-shaped cases. Inputs follow
the SGLang MLA post-attention layout; model dimensions and representative token
loads are labelled in the test source, rather than claimed as captured traces.

## Model-shaped tests

The caller analysis uses local SGLang commit
`2456c9111a121a02e41e625e3930e1eb143fe9bf`; checkpoint parameters were checked
on 2026-11-05. Tests construct tensors and call this operator directly. They do
not start models or exercise NPU Graph, weight loading, attention backends or
the complete inference pipeline.

The two callers are in
[`deepseek_v2_attention_mla_npu.py`](https://github.com/sgl-project/sglang/blob/43af9fcbb68d2d34a5549635dfcc7b46360e6fc6/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py#L290):

- Ordinary MLA, `forward_mla_core_npu`: this operator is selected only when
  `M >= 65536`, `B*K >= 65536`, or `N >= 65536`. Given the ordinary MLA models
  below and the existing operator limit `M <= 1024`, only effective attention
  TP=1 satisfies this condition. TP=2/4/8 selects
  `torch_npu.npu_transpose_batchmatmul`.
- DSA, `forward_dsa_core_npu`: on 910C, decode, target verification and draft
  extend v2 select this operator; ordinary extend selects `torch_npu`.
  Attention TP=1/2/4/8 can therefore be covered. See
  [`is_deepseek_dsa`](https://github.com/sgl-project/sglang/blob/43af9fcbb68d2d34a5549635dfcc7b46360e6fc6/python/sglang/srt/configs/model_config.py#L187)
  and [`handle_attention_ascend`](https://github.com/sgl-project/sglang/blob/43af9fcbb68d2d34a5549635dfcc7b46360e6fc6/python/sglang/srt/models/deepseek_common/attention_backend_handler.py#L75)
  for model detection and dispatch.

Model shapes are `(M, B, K, N)`, where
`B = num_attention_heads / effective_attention_tp`, `K = kv_lora_rank`, and
`N = v_head_dim`. Use attention TP at the call site; it can differ from
whole-model TP. The following verified checkpoints supply representative tests,
rather than an exhaustive list of derived checkpoints:
For identical shapes within a model family, keep one or two recent models.
DeepSeek-V3.1/V3.2 represent ordinary MLA and DSA respectively; GLM retains
5.2/5.3-Flash.

| Shape ID prefix / checkpoint configuration | Total heads | K | N | Tested attention TP | Caller |
| --- | --- | --- | --- | --- | --- |
| [`deepseek-v3.1`](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/config.json) | 128 | 512 | 128 | 1 | Ordinary MLA |
| [`mistral-large-3`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16/blob/main/params.json) | 128 | 512 | 128 | 1 | Ordinary MLA |
| [`deepseek-v3.2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) | 128 | 512 | 128 | 1/2/4/8 | DSA |
| [`glm-5.2`](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) | 64 | 512 | 256 | 1/2/4/8 | DSA |
| [`glm-5.3-flash`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) | 64 | 512 | 256 | 1/2/4/8 | DSA layers |
| [`longcat-flash-lite-sparse`](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite-Sparse/blob/main/config.json) | 32 | 512 | 128 | 1/2/4/8 | DSA |

Mistral-Large-3 uses `n_heads` for total heads and inherits the DeepSeek model
implementation. GLM-5.3-Flash dimensions come from `text_config`; only its DSA
layers reuse this path. SGLang normalizes LongCat's `LongcatCausalLM`
architecture to `LongcatFlashForCausalLM`, then detects DSA through `index_topk`.
These observations select direct operator inputs; they do not establish that
each complete model has been validated on 910C.

Sharing DeepSeek attention alone does not establish that a model calls this
operator. For example,
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)
has 64 total heads, K=512, N=128 and no DSA configuration, so it does not meet
the ordinary MLA condition within the supported M range. The caller's
`M >= 65536` branch exceeds the operator's original 1024 limit; this work
neither treats it as a valid input nor changes that limit.

Each model/TP combination tests `M = 1, 8, 36, 128, 1024`. M counts flattened
input tokens, potentially including multiple verification candidates. These
are representative loads, not fixed model batch sizes or captured requests.
DeepSeek-V3.2 and GLM-5.2 additionally test M=15/16/17 at attention TP=8, covering
the tile boundary with both real output widths. Every shape covers FP16/BF16
and random/zero inputs, checking output mutation and input preservation. No
checkpoint weights are needed; tests use the operator's floating-point inputs
even when the original checkpoint is quantized.

For example, `glm-5.2-attn-tp8-tokens36` means A=`[36,8,512]`, W=`[8,512,256]`,
C=`[36,8,256]`. Both SGLang callers use default ND and quant arguments. NZ
cases with model prefixes extend direct API format coverage; they do not imply
an NZ SGLang caller. Invalid-input tests remain `extra_checks`, excluded by
default.

After building and loading the migrated package, run direct tests on 910C:

```bash
python -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py \
  --impl=ascendc --impl-ref=torch_native --enable-perf=false -v -rs
```

Append `-k 'glm-5'` for GLM model cases or `-k 'longcat-flash-lite-sparse'` for
LongCat cases. These filters select operator shapes without starting a model.
