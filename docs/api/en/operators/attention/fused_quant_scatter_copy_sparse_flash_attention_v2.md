# fused_quant_scatter_copy_sparse_flash_attention_v2

Stage packed C8 KV for Ascend950 sparse attention. The operator updates the HBM buffer, attention output, and optional resident extents in place. Each 656-byte KV record contains FP8 NoPE data, BF16 RoPE data, and four FP32 scales. The route reads the caller's destination slots through the HBM block table.

`auto` selects `ascendc`. `torch_native` applies the complete request copy plan and evaluates attention on CPU. It reads `dram_kv_bytes` directly. The six optional resident-staging tensors must be supplied together.

See the [operator contract](../../../../../csrc/attention/fused_quant_scatter_copy_sparse_flash_attention/README.md).

## Usage example

```python
import torch

from sgl_kernel_npu.operators.attention.fused_quant_scatter_copy_sparse_flash_attention_v2 import (
    fused_quant_scatter_copy_sparse_flash_attention_v2,
)

impl = "torch_native"  # Use "ascendc" with device="npu" to run the kernel.
device = "cpu"
def ids(values):
    return torch.tensor(values, dtype=torch.int32, device=device)

query = torch.zeros((1, 8, 512), dtype=torch.bfloat16, device=device)
rope = torch.zeros((1, 8, 64), dtype=query.dtype, device=device)
hbm = torch.zeros((1, 128, 1, 656), dtype=torch.int8, device=device)
host = hbm.clone()
slots = ids([[[0] + [-1] * 2047]])
empty_plan = ids([[-1] * 32768])
result = torch.empty_like(query)
fused_quant_scatter_copy_sparse_flash_attention_v2(
    query, rope, hbm, host, host.data_ptr(), ids([[0]]), ids([[0]]),
    slots, slots, ids([0]), ids([1]), ids([1]), ids([0]),
    empty_plan, empty_plan, ids([0]), 576**-0.5, result,
    impl=impl,
)
assert result.shape == (1, 8, 512)
```
