# API 索引

| 接口名称 | 支持的后端 |
| --- | --- |
| [rmsnorm_bias](operators/norm/rmsnorm_bias.md) | `triton`、`torch_native` |
| [l1_norm](operators/norm/l1_norm.md) | `triton`、`torch_native` |
| [l2norm_fwd](operators/norm/l2norm_fwd.md) | `triton`、`torch_native` |
| [layer_norm_fwd_npu](operators/norm/layer_norm_fwd_npu.md) | `triton`、`torch_native` |

以上接口的 `impl` 默认值均为 `"auto"`，当前与 `"triton"` 使用同一注册入口，不会在执行失败后自动切换实现。输入约束和各代际验证状态请参阅对应接口文档。
