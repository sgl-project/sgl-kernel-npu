# fused_quant_lightning_indexer_manage_full_visible

在 Ascend950 上使用 FP8 索引键计算因果 TopK，更新 resident slot map，并为每个请求生成一份拷贝计划。查询采用 TND 布局，索引键每页 128 个 token。算子原地写入 slot map、TopK 和 miss 输出。

`auto` 选择 `ascendc`。`torch_native` 在 CPU 上执行 eager PyTorch 参考实现。分数相同时有确定的选择顺序，补齐 token 和替换槽位可以与设备实现不同。拷贝计划和 slot map 对应同一份分配结果。

完整参数约定见[算子说明](../../../../../csrc/attention/fused_quant_lightning_indexer_manage/README.md)。

## 使用示例

```python
import torch

from sgl_kernel_npu.operators.indexer.fused_quant_lightning_indexer_manage_full_visible import (
    fused_quant_lightning_indexer_manage_full_visible,
)

impl = "torch_native"  # Use "ascendc" with device="npu" to run the kernel.
device = "cpu"
def ids(values):
    return torch.tensor(values, dtype=torch.int32, device=device)

query = torch.ones((1, 32, 128), device=device).to(torch.float8_e4m3fn)
weights = torch.ones((1, 32), dtype=torch.bfloat16, device=device)
keys = torch.ones((1, 128, 1, 128), device=device).to(torch.float8_e4m3fn)
query_scale = torch.ones((1, 32), device=device)
key_scale = torch.ones((1, 128, 1), device=device)
pool = ids([[-1] * 128] * 3)
topk_sources = ids([[-1] * 2048])
topk_slots = torch.empty_like(topk_sources)
route_misses = ids([0])
miss_sources = ids([[-1] * 32768])
miss_slots = torch.empty_like(miss_sources)
miss_counts = ids([0])
fused_quant_lightning_indexer_manage_full_visible(
    query, weights, keys, query_scale, key_scale, ids([[0]]),
    ids([1]), ids([16]), ids([0]), ids([-3]), ids([1]), pool,
    topk_sources, topk_slots, route_misses, miss_sources, miss_slots, miss_counts,
    impl=impl,
)
assert set(topk_slots[0, :16].tolist()) == set(range(16))
```
