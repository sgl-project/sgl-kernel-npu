# mega_moe

## 芯片支持

| impl | arch22（测试目标 A3） | arch35 (A5) |
| --- | --- | --- |
| `auto` | 已注册，硬件测试待运行 | 已注册，硬件测试待运行 |
| `cann` | 已注册，硬件测试待运行 | 已注册，硬件测试待运行 |

依赖 CANN 9.2.0、匹配的 PyTorch/torch_npu 和提供 MegaMoE 的 `cann_ops_transformer`（含 JIT 编译依赖 `ninja`）。注册声明不代表所有 provider 版本与输入模式均经过验证。

## 函数原型

```python
from sgl_kernel_npu.operators.moe import mega_moe
mega_moe(*args, impl="auto", **kwargs)
```

`auto` 与 `cann` 是同一实现的别名，不自动回退。首次调用导入 provider，后续缓存函数。除 `impl` 外，参数、对象身份及返回值均原样转发；导入和执行异常原样传播。未知 impl 在导入 provider 前失败。

## 功能说明

将 token 分发到专家所在 rank，计算 FFN，再加权合并。SwiGLU 非量化语义如下（g/u 为第一层投影的两半）；量化转换、舍入和饱和遵循安装的 provider，包装层不做数值转换。

`y[t] = sum_k routing[t,k] * (silu(g[t,e]) * u[t,e]) @ W2[e]`, `e = ids[t,k]`, `[g,u] = x[t] @ W1[e]`.

## 参数与返回值

| 参数 | 类型、shape 与约束 |
| --- | --- |
| `x` | NPU Tensor `[T,H]`；测试的输入为 BF16。A3 的 T 为 buffer 容量，调用方负责补齐；A5 使用实际 token 数。 |
| `topk_ids` | NPU int32 `[T,K]`，全局专家编号；E 必须可被 EP world size 整除。 |
| `topk_weights` | NPU `[T,K]`；A3 测试 FP32，A5 测试 BF16/FP32。 |
| `l1_weights`, `l2_weights` | 本 rank 的专家权重 TensorList；A3 BF16 `[H,2I]` / `[I,H]` NZ；量化与 A5 的布局见后端说明。 |
| `sym_buffer` | 同一 EP group 创建的有效 provider buffer；不可跨不同配置复用。 |
| `l1_weights_sf`, `l2_weights_sf` | 可选量化 scale TensorList；BF16 路径不传或 None，量化模式必须与权重格式匹配。 |
| `l1_bias`, `l2_bias` | 可选补偿 bias；A3 W4A8 使用，A5 必须 None。 |
| `x_active_mask` | A3 可选 int8 `[T]`，标记有效 token；A5 不传或 None。 |
| `activation` | 测试覆盖 `swiglu`、`situglu`（A5 为 SiTU）。 |
| `activation_params` | 可选 dict，如 `beta`、`linear_beta`；具体默认值遵循 provider。 |
| `weight1_type`, `weight2_type` | A5 FP4 权重的 provider 类型枚举；非 FP4 不传或 None。 |
| `impl` | 仅关键字，默认 `"auto"`，可选 `"cann"`。 |

| 返回值 | 说明 |
| --- | --- |
| `output` | NPU `[T,H]`；BF16 测试与 x 同 dtype；调用方截取 A3 有效 token。 |
| `expert_token_nums` | 本 rank 专家的接收 token 统计；计数/累积格式遵循 provider 环境配置，测试固定 `MOE_EXPERT_TOKEN_NUMS_TYPE=1`。 |

## cann

A3 覆盖 BF16、W8A8、W4A8；A5 覆盖 MXFP8 E4M3/E5M2、MXFP4 E2M1 和 MXFP8-W4。A5 使用独立的平台权重布局、ND E8M0 scales；FP4 存储需按 provider 要求打包，A8W4-FP 使用 FRACTAL_NZ_C0_32。不能将 A3 权重直接用于 A5。

此接口保留 `*args/**kwargs`，不重新定义外部 provider 的默认参数或限制。详细限制以匹配版本的 provider 为准。

## 调用示例

```python
# Save as example.py; run: torchrun --standalone --nproc-per-node=4 example.py
import os
import torch
import torch.distributed as dist
import torch_npu
from sgl_kernel_npu.operators.moe import mega_moe, get_symm_buffer_for_mega_moe

torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("hccl")
T, H, I, E, K = 32, 2048, 1024, 8, 4  # A3 BF16 example
local_experts = E // dist.get_world_size()
x = torch.randn(T, H, device="npu", dtype=torch.bfloat16)
ids = torch.arange(K, device="npu", dtype=torch.int32).expand(T, K).contiguous()
routing = torch.full((T, K), 1 / K, device="npu", dtype=torch.float32)
w1 = [torch_npu.npu_format_cast(torch.randn(H, 2*I, device="npu",
      dtype=torch.bfloat16) / H**0.5, 29) for _ in range(local_experts)]
w2 = [torch_npu.npu_format_cast(torch.randn(I, H, device="npu",
      dtype=torch.bfloat16) / I**0.5, 29) for _ in range(local_experts)]
buffer = None
try:
    buffer = get_symm_buffer_for_mega_moe(
        dist.group.WORLD, num_experts=E, num_max_tokens_per_rank=T,
        num_topk=K, hidden=H, intermediate_hidden=I,
        max_recv_token_num=0, dispatch_quant_mode=0,
        dispatch_quant_out_dtype=None, impl="cann",
    )
    output, counts = mega_moe(
        x=x, topk_ids=ids, topk_weights=routing, l1_weights=w1, l2_weights=w2,
        sym_buffer=buffer, activation="swiglu", impl="cann",
    )
    torch.npu.synchronize()
finally:
    if buffer is not None:
        torch.npu.synchronize()
        buffer.destroy()
    dist.destroy_process_group()
```

分布式测试、独立参考实现的例外说明与性能命令见 [MoE test README](../../../../../tests/python/operators/moe/README.md).
