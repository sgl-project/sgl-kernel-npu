# mamba

为线性及树状投机候选执行 target verify 卷积。

公共入口：`sgl_kernel_npu.operators.mamba`.

后端按 `impl` 显式选择；`auto` 是页面所述后端的别名，无运行时回退。导入公共 registry 需要 NPU 架构检测，后端实现按首次调用延迟加载。

- [causal_conv1d_target_verify_npu](_causal_conv1d_target_verify_npu.md)

线性模式原地覆盖 `x`；树模式分配新输出。两者均保持 committed `conv_state` 只读，并按候选写 scratch；详见 API 的副作用契约。
