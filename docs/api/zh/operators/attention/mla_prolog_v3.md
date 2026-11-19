# mla_prolog_v3

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.attention import mla_prolog_v3


def mla_prolog_v3(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    kv_cache,
    kr_cache,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能说明

执行 V3 Multi-head Latent Attention 前处理。将 token 激活投影到 query 和 KV 路径，应用 RMS 归一化和 RoPE，再将压缩潜变量和旋转键写入指定缓存位置。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `token_x` | Tensor | [B, S, He] 或 SDK 打包 variant | 必传 | 输入 token 激活，示例采用 BF16。 |
| `weight_dq` | Tensor | [He, Hcq] | 必传 | query 降维投影矩阵，示例采用 BF16 FRACTAL_NZ。 |
| `weight_uq_qr` | Tensor | [Hcq, H * (D + Dr)] | 必传 | query 升维投影矩阵，包含旋转宽度，示例采用 BF16 FRACTAL_NZ。 |
| `weight_uk` | Tensor | [H, D, Hckv] | 必传 | 各头的潜变量 query 投影，示例采用 BF16 ND。 |
| `weight_dkv_kr` | Tensor | [He, Hckv + Dr] | 必传 | KV 降维投影与旋转键的合并矩阵，示例采用 BF16 FRACTAL_NZ。 |
| `rmsnorm_gamma_cq`, `rmsnorm_gamma_ckv` | Tensor | [Hcq], [Hckv], BF16 在示例中 | 必传 | 两个潜变量路径的归一化权重。 |
| `rope_sin`, `rope_cos` | Tensor | [B, S, Dr], BF16 在示例中 | 必传 | 正弦和余弦系数。 |
| `kv_cache`, `kr_cache` | Tensor | [P, L, 1, Hckv] 与 [P, L, 1, Dr] 在 PA_BSND 布局下 | 必传 | 由调用方分配的潜变量和旋转键缓存，原地更新。 |
| `cache_index`, `cache_mode` | Tensor or None, str | [B, S], INT64 在示例中 | `None` / `"PA_BSND"` | 可选缓存槽位和 SDK 布局，示例采用 PA_BSND。 |
| `rmsnorm_epsilon_cq`, `rmsnorm_epsilon_ckv` | float | - | `1e-05` | 可选归一化稳定项，SDK 默认均为 1e-5。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `query`, `query_rope`, `dequant_scale_q_nope`, `query_norm`, `dequant_scale_q_norm` | tuple of five tensors | Shapes 与 dtypes follow projection 与 quantization modes | 投影后的 query、旋转 query、反量化尺度、可选归一化 query 及其尺度。两个缓存另行原地更新。 |

## torch_npu 后端

调用 `torch_npu.npu_mla_prolog_v3`。形状、dtype 和布局组合以已安装的 SDK 为准。 V3 返回元组和缓存写入遵循已安装 TorchNPU 的 ABI。其他量化属性可通过 SDK 关键字参数传入。示例采用 A3 的 BF16 PA_BSND 配置。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import mla_prolog_v3

b, seq, h, hidden, cq, ckv, d, dr = 8, 2, 32, 7168, 1536, 512, 128, 64


def bf16(*shape):
    return torch.randn(shape, dtype=torch.bfloat16, device="npu")


def nz(*shape):
    return torch_npu.npu_format_cast(bf16(*shape), 29)


x = bf16(b, seq, hidden)
sin = torch.zeros_like(bf16(b, seq, dr))
cos = torch.ones_like(sin)
kv_cache = bf16(64, 128, 1, ckv)
kr_cache = bf16(64, 128, 1, dr)
slots = torch.arange(b * seq, dtype=torch.int64, device=x.device).reshape(b, seq)
out = mla_prolog_v3(
    x,
    nz(hidden, cq),
    nz(cq, h * (d + dr)),
    bf16(h, d, ckv),
    nz(hidden, ckv + dr),
    bf16(cq),
    bf16(ckv),
    sin,
    cos,
    kv_cache,
    kr_cache,
    cache_index=slots,
    cache_mode="PA_BSND",
    impl="torch_npu",
)
```
