# pack_pool_key_indexer_indices

整理 CANN 因果池化索引并恢复每个 query 的未压缩 tail。
`auto` 选择 `triton`，`torch_native` 是独立 PyTorch 对照实现。

```python
from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

output = pack_pool_key_indexer_indices(
    indices, positions, query_ends, topk, pool_size, output_buffer=None
)
```

输入为 int32 `indices[T,C]`、int64 `positions[T]` 和 int64 `query_ends[B]`，
位于同一 NPU，末维连续，要求 `C >= topk + pool_size - 1`。
Query ends 是不含 sentinel 的原始真实 query 累积边界，单调不减；
有效 query 构成长为 `query_ends[-1]` 的前缀，剩余行是图补齐。
非空输入至少需要一个 query end。`topk`、`pool_size` 与生成输入的 CANN 调用一致。

位置 `p` 的完整历史边界为 `tail_start = (p+1) // pool_size * pool_size`。
保留 CANN 输出的前 `min(tail_start,topk)` 个历史索引，再追加
`[tail_start,p+1)` 中的原始 token 位置，其余列及 padding 行填 `-1`。
与 vllm-ascend 原 kernel 一致，不对历史候选额外按值过滤；短请求的完整历史
数量为零，因此不会保留人工池化历史。

返回 int32 `[T,topk+pool_size-1]`。默认原地覆盖 `indices`。
可提供容量充足、末维连续的 int32 输出缓冲，允许额外行和更宽列；
前 `T` 行未使用的宽列也填为 `-1`。与 vLLM wrapper 不同，本接口返回二维张量。

Triton kernel 原样迁自 vllm-ascend
`e0e8dc91ce5a5a03f5cd18176bc6f4ec5fd0e4a8` 的
`vllm_ascend/ops/triton/glm5_next_cann_keypool.py::_pack_indices_kernel`。
公开接口以 `query_ends` 替换原 token 有效掩码，两仓必须同步更新。
本次迁移尚待目标 NPU 验证。
