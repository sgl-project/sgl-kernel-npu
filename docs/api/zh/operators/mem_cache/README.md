# mem_cache

更新 KPool cache/tail 状态。

公共入口：`sgl_kernel_npu.operators.mem_cache`.

后端按 `impl` 显式选择；`auto` 是页面所述后端的别名，无运行时回退。导入公共 registry 需要 NPU 架构检测，后端实现按首次调用延迟加载。

- [kpool_assemble_softmax_rotate_write_cache](_kpool_assemble_softmax_rotate_write_cache.md)
- [kpool_decode_update_and_maybe_write_cache](_kpool_decode_update_and_maybe_write_cache.md)
- [kpool_write_tail_and_maybe_compress](_kpool_write_tail_and_maybe_compress.md)
- [scatter_kpool_tail_updates](_scatter_kpool_tail_updates.md)

四个 API 均为原地操作，返回 `None`；调用方需隔离并发写入并初始化历史状态。pool 页表构造属于 [attention](../attention/README.md)。
