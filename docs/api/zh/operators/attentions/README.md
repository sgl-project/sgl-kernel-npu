# attentions

通过可选 ACLNN attentions 扩展执行 pooled-key 选择。

公共入口：`sgl_kernel_npu.operators.attentions`.

后端按 `impl` 显式选择；`auto` 是页面所述后端的别名，无运行时回退。导入公共 registry 需要 NPU 架构检测，后端实现按首次调用延迟加载。

- [pool_key_indexer](_pool_key_indexer.md)

仅提供 `aclnn`；须单独安装 attentions 扩展及 vendor 算子依赖。当前未完成硬件验收。
