# fused_lightning_indexer_manage

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 已在 Ascend910_9362 验证 | 不支持 |

## 接口

```python
from sgl_kernel_npu.operators.indexer import fused_lightning_indexer_manage
```

```python
def fused_lightning_indexer_manage(
    index_weights, query, index_key_cache, index_block_table,
    actual_seq_lengths_query, actual_seq_lengths_key, offload_seq_lengths_key,
    num_cache_tokens, request_state, req_pool_entries, cache_slots_pool,
    topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids,
    miss_dst_slots, miss_counts, full_visible=False, *, impl="auto",
) -> None:
    ...
```

## 说明

算子计算索引 TopK，原地更新常驻槽映射和 miss 搬运计划。ARCH22 安装包含该
算子的 wheel 后，`impl="auto"` 选择 `"ascendc"`。

## 参数及返回值

| 参数 | 约定 |
| --- | --- |
| `index_weights`、`query`、`index_key_cache` | FP16 或 BF16 张量，形状分别为 `[T, H]`、`[T, H, 128]`、`[P, 128, 1, 128]`。 |
| `index_block_table` | 索引 K 的页表，int32 `[B, L]`。 |
| `actual_seq_lengths_query`、`actual_seq_lengths_key`、`offload_seq_lengths_key` | int32 `[B]`，分别是累计查询结束位置、可见键长度和上轮 offload 长度。 |
| `num_cache_tokens`、`request_state`、`req_pool_entries` | int32 `[B]`，分别是常驻容量、请求状态和持久槽映射的行号。 |
| `cache_slots_pool` | 原地更新的 int32 `[R, S]` 编码槽映射。 |
| `topk_src_ids`、`topk_dst_slots`、`topk_miss_counts` | 原地写入的查询计划，形状为 `[T, 2048]`、`[T, 2048]`、`[T]`，均为 int32。 |
| `miss_src_ids`、`miss_dst_slots`、`miss_counts` | 原地写入的请求搬运计划，形状为 `[B, 32768]`、`[B, 32768]`、`[B]`，均为 int32。 |
| `full_visible` | TopK 覆盖当前查询因果可见的所有位置，默认 `False`。 |

`T` 是打包后的查询 token 数，`H` 是索引头数，`P` 是索引 K 页数，`B` 是请求数。
张量位于同一 NPU 且连续。返回值为 `None`。算子约定与测试命令见 [native 文档](../../../../../csrc/indexer/fused_lightning_indexer_manage/README.md)。

## ascendc 后端

直接调用 wheel 的 native LIM 绑定。精度测试用独立的 CPU 热点集合和缓存模型核对槽位计划。
CPU 模型用于核对选择与驻留结果。公开接口通过 `auto` 和 `ascendc` 执行融合的设备原地更新。

## 可运行示例

```python
import torch
import torch_npu
import sgl_kernel_npu
from sgl_kernel_npu.operators.indexer import fused_lightning_indexer_manage

def main():
    torch.npu.set_device(0)
    device, dtype = "npu:0", torch.bfloat16
    batch, heads, source_capacity = 1, 32, 8320
    weights = torch.full((batch, heads), 1 / heads, dtype=dtype, device=device)
    query = torch.zeros((batch, heads, 128), dtype=dtype, device=device)
    query[..., 0] = 1
    keys = torch.zeros((source_capacity // 128, 128, 1, 128), dtype=dtype, device=device)
    keys.view(-1, 128)[:, 0] = -8
    keys.view(-1, 128)[:2048, 0] = 8
    table = torch.arange(source_capacity // 128, dtype=torch.int32, device=device).view(1, -1)
    def vec(value):
        return torch.tensor([value], dtype=torch.int32, device=device)
    pool = torch.full((batch, source_capacity), -1, dtype=torch.int32, device=device)
    ids = torch.empty((batch, 2048), dtype=torch.int32, device=device)
    slots = torch.empty_like(ids)
    miss_src = torch.empty((batch, 32768), dtype=torch.int32, device=device)
    miss_dst = torch.empty_like(miss_src)
    miss_counts = vec(0)
    fused_lightning_indexer_manage(
        weights, query, keys, table, vec(1), vec(8193), vec(8192), vec(4096),
        vec(-2), vec(0), pool, ids, slots, vec(0), miss_src, miss_dst, miss_counts,
    )
    torch.npu.synchronize()
    assert torch.equal(ids.cpu().sort().values[0], torch.arange(2048, dtype=torch.int32))
    assert int(miss_counts.cpu()[0]) == 4096  # Cold fill populates the full resident capacity.

if __name__ == "__main__":
    main()
```
