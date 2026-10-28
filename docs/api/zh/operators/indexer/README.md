# indexer

生成因果长度、pool 索引及 decode indexer 输入。

公共入口：`sgl_kernel_npu.operators.indexer`.

后端按 `impl` 显式选择；`auto` 是页面所述后端的别名，无运行时回退。导入公共 registry 需要 NPU 架构检测，后端实现按首次调用延迟加载。

- [expand_causal_seqlens](_expand_causal_seqlens.md)
- [expand_pool_topk_and_append_tail_batched_npu](_expand_pool_topk_and_append_tail_batched_npu.md)
- [fused_decode_prepare_indexer](_fused_decode_prepare_indexer.md)

三个 API 的 `auto` 均选择 `triton`，并提供独立 `torch_native`。`expand_causal_seqlens` 支持可选原地输出缓冲复用；其余两个 API 分配输出。
