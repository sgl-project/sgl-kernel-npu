# attentions

Pooled-key selection through the optional ACLNN attentions extension.

Public import: `sgl_kernel_npu.operators.attentions`.

Select backends with `impl`; `auto` aliases the backend listed on each API page, with no runtime fallback. Registry import requires NPU architecture detection; implementations load lazily on first execution.

- [pool_key_indexer](_pool_key_indexer.md)

Only `aclnn` is available; install the separate attentions extension and vendor operator dependencies. Hardware acceptance is pending.
