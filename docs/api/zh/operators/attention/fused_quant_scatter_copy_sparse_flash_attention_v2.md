# fused_quant_scatter_copy_sparse_flash_attention_v2

在 Ascend950 上为稀疏注意力准备 packed C8 KV。算子原地更新 HBM 缓冲区、注意力输出和可选的 resident 长度。每条 656 字节的 KV 记录包含 FP8 NoPE、BF16 RoPE 和四个 FP32 scale。每条 query route 通过 HBM 页表读取调用者提供的目标槽位。

`auto` 选择 `ascendc`。`torch_native` 在 CPU 上执行整份请求拷贝计划和注意力计算，直接读取 `dram_kv_bytes`。六个可选的 resident staging 张量需要一起传入。

完整参数约定见[算子说明](../../../../../csrc/attention/fused_quant_scatter_copy_sparse_flash_attention/README.md)。

## 使用示例

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
