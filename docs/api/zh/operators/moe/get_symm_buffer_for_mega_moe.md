# get_symm_buffer_for_mega_moe

## 硬件支持

| impl | arch22（测试目标 A3） | arch35 (A5) |
| --- | --- | --- |
| `auto` | 已注册，硬件测试待运行 | 已注册，硬件测试待运行 |
| `cann_ops_transformer` | 已注册，硬件测试待运行 | 已注册，硬件测试待运行 |

仅声明支持 ARCH22 和 ARCH35，并非支持所有代际；硬件验证仍待完成。

依赖 CANN 9.2.0、匹配的 PyTorch/torch_npu 和提供 MegaMoE 的 `cann_ops_transformer`（含 JIT 编译依赖 `ninja`）。注册声明不代表所有 provider 版本与输入模式均经过验证。

## 函数签名

```python
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe
get_symm_buffer_for_mega_moe(*args, impl="auto", **kwargs)
```

`auto` 与 `cann_ops_transformer` 是同一实现的别名，不自动回退。首次调用导入 provider，后续缓存函数。除 `impl` 外，参数、对象身份及返回值均原样转发；导入和执行异常原样传播。未知 impl 在导入 provider 前失败。

## 功能描述

调用方管理生命周期：相同配置可复用，配置变化需新建 buffer；计算完成并同步后调用 `destroy()`，最后销毁 process group。包装层不缓存、补齐或销毁 buffer。所有 rank 必须以一致顺序执行集合通信。此接口是资源创建接口，不要求第二种计算实现。

## 参数与返回值

表中默认值来自下文链接的固定版本 provider。公开签名仍为 `(*args, impl="auto", **kwargs)`，仅自行定义 `impl` 默认值；其他 provider 选项原样透传。其他可选参数见原始文档。

| 参数 | 默认值 | 类型、说明与约束 |
| --- | --- | --- |
| 第一个位置参数 group | 必填 | `torch.distributed.ProcessGroup`，已初始化 HCCL EP group。 |
| `num_experts` | 必填 | int，全局专家数 E，正数且可被 group 大小整除。 |
| `num_max_tokens_per_rank` | 必填 | int，每 rank 容量上限，所有 rank 使用兼容配置。 |
| `num_topk` | 必填 | int，每 token 的专家数 K。 |
| `hidden`, `intermediate_hidden` | 必填 | int，模型 H 与专家 I；满足所用 provider 模式的对齐要求。 |
| `max_recv_token_num` | `0` | int，接收容量配置；测试显式使用 0，由 provider 处理。 |
| `dispatch_quant_mode` | `0` | int；测试使用 0（BF16）、2（INT8）或 4（MXFP）。 |
| `dispatch_quant_out_dtype` | `None` | dtype 或 provider FP4 枚举，与量化模式匹配；BF16 测试显式传 None。 |
| `impl` | `"auto"` | 仅关键字，默认 `"auto"`，可选 `"cann_ops_transformer"`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `buffer` | provider `SymmBuffer` | 有效的 EP 通信资源；调用 `destroy()` 前必须同步。 |

## cann_ops_transformer 后端

原始算子文档：[ops-transformer MegaMoE](https://gitcode.com/cann/ops-transformer/blob/be9c0b2766e827b523c4a2b8728e738d470b5a36/mc2/mega_moe/docs/torchapi_mega_moe.md)。该固定版本文档包含 provider 的完整签名、默认值、量化公式与布局约束；实际调用时应使用与安装版本匹配的文档。

A3 覆盖 BF16、W8A8、W4A8；A5 覆盖 MXFP8 E4M3/E5M2、MXFP4 E2M1 和 MXFP8-W4。A5 使用独立的平台权重布局、ND E8M0 scales；FP4 存储需按 provider 要求打包，A8W4-FP 使用 FRACTAL_NZ_C0_32。不能将 A3 权重直接用于 A5。

此接口保留 `*args/**kwargs`，不重新定义外部 provider 的默认参数或限制。详细限制以匹配版本的 provider 为准。

## 示例

完整的四卡 A3 输入构造、计算与清理示例见 [mega_moe](mega_moe.md).

```python
# Save as buffer_example.py; run: torchrun --standalone --nproc-per-node=4 buffer_example.py
import os
import torch
import torch.distributed as dist
import torch_npu
from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe

device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group("hccl")
buffer = None
try:
    dist.all_reduce(torch.ones(1, device=device))
    buffer = get_symm_buffer_for_mega_moe(
        dist.group.WORLD, num_experts=8, num_max_tokens_per_rank=32,
        num_topk=4, hidden=2048, intermediate_hidden=1024,
        impl="cann_ops_transformer",
    )
finally:
    if buffer is not None:
        torch.npu.synchronize()
        buffer.destroy()
    dist.destroy_process_group()
```
