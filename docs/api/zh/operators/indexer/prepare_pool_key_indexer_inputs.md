# prepare_pool_key_indexer_inputs

为 CANN 因果 `TND` / `PA_BBND` 索引（`mask_mode=3`）准备非空输入。
`auto` 和 `triton` 选择同一份 Triton 实现，`torch_native` 选择 PyTorch 对照实现。

```python
from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

query_padded, weights_padded, ends_padded, lengths_padded, tails_padded, table_padded = (
    prepare_pool_key_indexer_inputs(query, weights, ends, lengths, tails, table)
)
```

| 输入 | 形状 | 类型 | 输出处理 |
| --- | --- | --- | --- |
| `query` | `[T,H,D]` | BF16 | 追加一行零。 |
| `weights` | `[T,H]` | BF16 | 追加一行零。 |
| `ends` | `[R]` | int64 | 追加 `T+1`。 |
| `lengths` | `[R]` | int64 | 下限置为一，并追加一。 |
| `tails` | `[R]` | int64 | 追加零。 |
| `table` | `[M,C]` | int32 | 追加一行零。 |

所有输入位于 query 所在 NPU，要求 `M >= R`、`C >= 1`。Ends 单调不减，
位于 `[0,T]`，不计入图补齐 token；重复 end 表示没有 query 的请求。
若 `M > R`，包括第 `R` 行在内的未使用页表行必须已有有效块号。
物理缓存块零、每个空历史请求的首块都必须已经分配且可读。

非连续输入先调用 `contiguous()`。六个输出保持各输入类型，均连续，
不修改输入，每次调用独立分配；图捕获后回放复用捕获的分配，不引入设备到主机同步。

用准备后的输入调用 CANN，将输出裁为 `indices[:T]`，再使用原始 positions
和有效 query metadata 执行 packing，排除人工历史及图补齐候选。

Kernel 原样迁自 vllm-ascend 的
`vllm_ascend/ops/triton/prepare_nonempty_keypool_inputs.py`，来源提交
`e0e8dc91ce5a5a03f5cd18176bc6f4ec5fd0e4a8`；仅适配导入和公开函数名，
保留 Apache-2.0 声明。本次迁移尚未完成目标 NPU 验证。
