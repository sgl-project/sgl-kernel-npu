# fused_quant_lightning_indexer_manage_full_visible

Compute causal TopK from FP8 index keys, update the resident slot map, and publish one copy plan per request on Ascend950. Inputs use TND query rows and page-128 index keys. The operator writes the slot map, TopK routes, and miss outputs in place.

`auto` selects `ascendc`. `torch_native` runs an eager PyTorch reference on CPU. It uses a deterministic order for equal scores and can choose different filler tokens or replacement slots from the kernel. The copy plan and slot map describe the same resulting allocation.

See the [operator contract](../../../../../csrc/attention/fused_quant_lightning_indexer_manage/README.md).

## Usage example

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
