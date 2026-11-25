# kv_rmsnorm_rope_cache

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-transformer`。

```python
from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache


def kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    *,
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    v=None,
    epsilon=1e-05,
    cache_mode="Norm",
    is_output_kv=False,
    impl="auto"
): ...
```

## 功能说明

将 KV 分成潜变量与旋转分量。对潜变量计算 $g\odot x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$，再旋转其余坐标，并将两路结果写入指定槽位。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `kv` | Tensor | [B, 1, S, D + Dr], FP16/BF16 | 必传 | 输入先存 D 维潜变量，再存 Dr 维旋转分量。 |
| `gamma` | Tensor | [D], 输入 dtype，用于 example | 必传 | 潜变量归一化权重。 |
| `cos`, `sin` | Tensor | [B, 1, S, Dr], 输入 dtype，用于 example | 必传 | 旋转系数。 |
| `index` | Tensor | [B] for one token per batch, INT64，用于 example | 必传 | PA_BNSD 下的物理槽位，具体解释取决于 cache_mode。 |
| `k_cache`, `ckv_cache` | Tensor | [P, L, 1, Dr] 与 [P, L, 1, D] for PA_BNSD | 必传 | 旋转和潜变量缓存缓冲区，原地更新。 |
| `k_rope_scale`, `c_kv_scale` | Tensor or None | SDK quantization scale shapes | `None` | 可选缓存量化系数。 |
| `k_rope_offset`, `c_kv_offset` | Tensor or None | SDK quantization offset shapes | `None` | 可选缓存量化偏移。 |
| `v` | Tensor or None | SDK-specific | `None` | 可选 SDK 参数，须核对所安装版本。 |
| `epsilon` | float | - | `1e-05` | RMS 归一化的稳定项。 |
| `cache_mode` | str | - | `'Norm'` | 厂商存储布局和槽位解释方式。 |
| `is_output_kv` | bool | - | `False` | 为 true 时返回旋转和归一化 KV 中间结果。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `k_cache`, `ckv_cache`, `rotated`, `normalized` | tuple of four tensors | cache shapes 与 intermediate shapes | 前两项为更新后的旋转与潜变量缓存。is_output_kv=True 时后两项有效。 |

## ops-transformer 后端

调用 `torch_npu.npu_kv_rmsnorm_rope_cache`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_kv_rmsnorm_rope_cache.md) [CANN ops-transformer 源码](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/posembedding/kv_rms_norm_rope_cache)

## torch_native 后端

在 Norm、PA 或 PA_BNSD 模式下为未量化 ND 缓存计算 RMSNorm 和 RoPE。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache

kv = torch.randn((2, 1, 1, 576), dtype=torch.float16, device="npu")
gamma = torch.ones((512,), dtype=kv.dtype, device=kv.device)
cos = torch.ones((2, 1, 1, 64), dtype=kv.dtype, device=kv.device)
sin = torch.zeros_like(cos)
slots = torch.tensor([19, 0], dtype=torch.int64, device=kv.device)
rotary_cache = torch.zeros((2, 16, 1, 64), dtype=kv.dtype, device=kv.device)
latent_cache = torch.zeros((2, 16, 1, 512), dtype=kv.dtype, device=kv.device)
results = kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    slots,
    rotary_cache,
    latent_cache,
    cache_mode="PA_BNSD",
    is_output_kv=True,
    impl="ops-transformer",
)
```
