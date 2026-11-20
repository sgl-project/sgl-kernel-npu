# attention_update

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-transformer`。

```python
from sgl_kernel_npu.operators.attention import attention_update


def attention_update(lse, local_out, update_type, *, impl="auto"): ...
```

## 功能说明

合并互不相交的分区注意力。设各分区的 log-sum-exp 为 $L_i$、归一化结果为 $O_i$，合并后 $L=\log\sum_i e^{L_i}$，$O=\sum_i e^{L_i-L}O_i$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `lse` | sequence[Tensor] | P tensors of [N], FP32 | 必传 | 分区 LSE，N 合并了 token 和头轴。 |
| `local_out` | sequence[Tensor] | P tensors of [N, D], same floating dtype | 必传 | 各分区的归一化输出，与 LSE 同卡。 |
| `update_type` | int | - | 必传 | 0 计算合并输出，1 还计算合并 LSE。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `output`, `lse` | tuple[Tensor, Tensor] | [N, D], [N] | 合并注意力和 SDK LSE 缓冲区。模式 1 才计算可用的合并 LSE。 |

## ops-transformer 后端

调用 `torch_npu.npu_attention_update`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_attention_update.md) [CANN ops-transformer 源码](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/attention_update)

## torch_native 后端

使用 PyTorch 按各分片的 softmax 权重合并输出，模式 1 同时返回合并后的 LSE。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import attention_update

lse_parts = [torch.zeros((64,), dtype=torch.float32, device="npu") for _ in range(2)]
values = [torch.ones((64, 512), dtype=torch.float32, device="npu") for _ in range(2)]
out, lse = attention_update(lse_parts, values, 1, impl="ops-transformer")
```
