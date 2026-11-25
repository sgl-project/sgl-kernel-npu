# fused_lightning_indexer_manage

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 未验证 | 不支持 |

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

## 可运行示例

```bash
python -m pytest tests/python/sgl_kernel_npu/test_fused_lightning_indexer_manage.py -q
```
