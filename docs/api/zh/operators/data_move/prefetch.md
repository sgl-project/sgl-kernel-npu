# prefetch

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `runtime` | 支持 | 不支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `runtime`。

```python
from sgl_kernel_npu.operators.data_move import prefetch


def prefetch(input, dependency, max_size, offset=0, *, impl="auto"): ...
```

## 功能说明

对输入存储的一段区域发起异步 L2 预取。可通过依赖 Tensor 表达与其他算子的执行顺序。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | contiguous NPU Tensor | 必传 | 待预取的连续 NPU Tensor。 |
| `dependency` | Tensor or None | same NPU | 必传 | 可选依赖 Tensor。 |
| `max_size` | int | bytes | 必传 | 最大预取字节数。 |
| `offset` | int | bytes | `0` | 相对输入存储起始位置的字节偏移。 |
| `impl` | str | - | `"auto"` | `auto`, `runtime`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `result` | None | - | 该调用提供缓存提示，无 Tensor 返回值。 |

## runtime 后端

调用 `torch_npu.npu_prefetch`，由 CANN Runtime 的 `aclrtCmoAsync` 执行预取。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_prefetch.md) [CANN Runtime 源码](https://gitcode.com/cann/runtime/blob/be5219e84b0af2b95165b44d9231871fa921d5fa/include/external/acl/acl_rt.h)

预取是设备缓存提示，接口返回 `None` 并保持输入数值。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.data_move import prefetch

x = torch.ones(2048, dtype=torch.float16, device="npu")
prefetch(x, x, x.numel() * x.element_size(), impl="runtime")
```
