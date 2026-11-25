# fused_scatter_copy_sparse_flash_attention_split

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 未验证 | 不支持 |

## 接口

```python
from sgl_kernel_npu.operators.attention import (
    fused_scatter_copy_sparse_flash_attention_split,
)
```

```python
def fused_scatter_copy_sparse_flash_attention_split(
    query, query_rope, hbm_ckv, hbm_rope, actual_seq_lengths_query,
    actual_seq_lengths_kv, num_cache_tokens, topk_src_ids, topk_dst_slots,
    topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts,
    hbm_block_table, dram_block_table, dram_kv_dev_ptr, attention_out,
    scale_value, *, full_visible=False, resident_refresh_tail=False,
    current=None, positions=None, valid=None, request_slots=None,
    req_to_token=None, resident_end=None, host_token_capacity=0,
    dram_rope_dev_ptr=0, impl="auto",
) -> None:
    ...
```

## 说明

算子消费 LIM 搬运计划，把缺失的 KV 写入分离的 HBM 缓存并计算稀疏 attention。
ARCH22 安装包含该算子的 wheel 后，`impl="auto"` 选择 `"ascendc"`。

## 参数及返回值

| 参数 | 约定 |
| --- | --- |
| `query`、`query_rope`、`attention_out` | FP16 或 BF16 TND 张量，形状为 `[T,H,512]`、`[T,H,64]`、`[T,H,512]`。 |
| `hbm_ckv`、`hbm_rope` | 原地写入的分页 HBM 缓存，形状为 `[P,128,1,512]`、`[P,128,1,64]`，类型同 query。 |
| `actual_seq_lengths_query`、`actual_seq_lengths_kv`、`num_cache_tokens` | int32 `[B]`，分别是累计查询结束位置、请求内 KV 长度和缓存容量。 |
| `topk_src_ids`、`topk_dst_slots`、`topk_miss_counts` | int32 查询计划，形状为 `[T,2048]`、`[T,2048]`、`[T]`。 |
| `miss_src_ids`、`miss_dst_slots`、`miss_counts` | int32 请求搬运计划，形状为 `[B,M]`、`[B,M]`、`[B]`。 |
| `hbm_block_table`、`dram_block_table` | int32 页表，每个请求一行。 |
| `dram_kv_dev_ptr`、`dram_rope_dev_ptr` | 注册后 NPU 可访问的 host CKV、KPE 地址。 |
| `scale_value` | attention 缩放系数。 |
| `full_visible`、`resident_refresh_tail` | 是否启用全可见模式、是否允许 LIM refresh 后缀与计算重叠，默认均为 `False`。 |
| `current`、`positions`、`valid`、`request_slots`、`req_to_token`、`resident_end`、`host_token_capacity` | 可选的全可见常驻写入参数。形状和边界见 native 文档。 |

返回值为 `None`。算子原地更新 HBM 缓存、`attention_out` 和可选的 `resident_end`。
详细约定与验证用例见 [native 文档](../../../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/README.md)
和 [独立的 attention 及 replay 测试](../../../../../tests/test_copy_sfa_split_hbm_resident.py)。

## ascendc 后端

直接调用 wheel 的 native split Copy-SFA 绑定。精度测试分配已注册的 host 缓存，并用独立的 PyTorch attention 核对输出。

## 可运行示例

```bash
python -m pytest tests/test_copy_sfa_split_hbm_resident.py -q
```
