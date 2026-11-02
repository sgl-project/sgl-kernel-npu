# mega_moe

调用外部 CANN MegaMoE provider，融合 token 分发、expert FFN 计算和结果合并。本仓库负责接口注册与参数转发；通信 buffer 由调用方管理。

```python
from sgl_kernel_npu.operators.moe import mega_moe

mega_moe(*args, impl="auto", **kwargs)
```

`auto` 和 `cann` 指向同一个实现，显式注册支持 `Generation.ARCH22`、`Generation.ARCH35`。其他代际未声明支持。导入本包需要正常的 Ascend NPU 运行环境；首次调用后端时才加载 `cann_ops_transformer.ops`。请安装兼容的 provider 和 JIT 编译依赖（包括 `ninja`）。provider 导入或执行失败会直接向调用方传播，不进行回退。

除 `impl` 外，所有位置参数和关键字参数原样传入 `cann_ops_transformer.ops.mega_moe`。形状、dtype、量化模式、激活参数及返回值遵循所安装 provider 的契约。注册支持某个代际，不代表任意 provider 版本支持所有输入组合。

现有集成传入 `x`、`topk_ids`、`topk_weights`、每个 expert 的 `l1_weights` 和 `l2_weights`，以及持久化的 `sym_buffer`。可选参数包括权重 scale、bias、`x_active_mask`、`activation` 和 `activation_params`。provider 返回 `(output, expert_token_nums)`。

通过 [get_symm_buffer_for_mega_moe](get_symm_buffer_for_mega_moe.md) 创建通信 buffer。调用方负责 buffer 生命周期、必要的 token padding 和输出切片；注册层不会自动完成这些操作。

```python
from sgl_kernel_npu.operators import get_impl, list_impls
from sgl_kernel_npu.operators.moe import mega_moe

assert list_impls(mega_moe) == ["auto", "cann"]
assert get_impl(mega_moe, "auto") is get_impl(mega_moe, "cann")
# output, counts = mega_moe(..., sym_buffer=buffer, impl="cann")
```
