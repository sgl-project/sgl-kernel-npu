# MoE gating Top-K

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `torch_npu` | supported | supported |
| `cann_9_1` | supported | supported |
| `torch_native` | supported | supported |

CANN 9.1 on A3 does not support `norm_type=2` (sqrtsoftplus). The model-mode test skips this unsupported combination instead of substituting softmax or sigmoid.

## Signature

```python
from sgl_kernel_npu.operators.moe import moe_gating_top_k

def moe_gating_top_k(x, k, *, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

FP32/FP16/BF16 [T,E] input, optional grouped or hash routing. A3 sqrt-softplus uses group_count=1.

Score with softmax/sigmoid/sqrt-softplus; bias affects selection but not gathered weights. Select experts, optionally renormalize and multiply routed_scaling_factor. Hash input_ids and tid2eid must be supplied together.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| k | Tensor / int | required | Indexer: paged [pages,page_size,1,D]; MoE: requested expert count |
| bias | Tensor / None | None | Optional correction bias [E], affects selection only |
| input_ids | Tensor / None | None | Optional token hash IDs, paired with tid2eid |
| tid2eid | Tensor / None | None | Token-to-expert hash mapping |
| k_group | int | 1 | Number of selected groups |
| group_count | int | 1 | Number of expert groups |
| group_select_mode | int | 0 | Vendor group score selection mode |
| renorm | int / bool | 0 | Whether selected weights are renormalized |
| norm_type | int | 0 | 0 softmax, 1 sigmoid, 2 sqrt-softplus |
| out_flag | bool | False | Whether to return full normalized scores |
| routed_scaling_factor | float | 1.0 | Multiplier after optional renormalization |
| eps | float | 1e-20 | Numerical stability epsilon |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| weights | Tensor | [T,k], x.dtype. |
| indices | int32 Tensor | [T,k] expert IDs. |
| scores | FP32 Tensor | [T,E] if out_flag=True, otherwise empty. |

## torch_npu

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding.

## cann_9_1

Uses the CANN 9.1 binding and performs input conversion in this operator module. It requires the initialized CANN 9.1 SDK and never retries a CANN 9.2 binding.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.moe import moe_gating_top_k

x = torch.randn(8, 256, device="npu", dtype=torch.float32)
weights, ids, scores = moe_gating_top_k(x, 8, norm_type=0, renorm=1, impl="auto")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/moe/test_moe_gating_top_k.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream torch_npu schemas](https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml)
