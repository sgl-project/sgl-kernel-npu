# kv_compress_epilog_v2

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_recipes_infer` | 不支持 | 支持（CANN 9.2.x） |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog_v2

def kv_compress_epilog_v2(
    cache, x, slot_mapping, *,
    quant_group_size=32, quant_mode="mxfp8_bf16",
    round_scale=True, x_scale=1.0, impl="auto",
): ...
```

默认的 `auto` 选择 `cann_recipes_infer`（bundled CANN 9.2.0 recipes kernel，仅 arch35）。

## 功能说明

整行量化 BF16 KV 行并按 `slot_mapping` 原地写入 paged KV cache，供 DeepSeek-V4.1 在 A5 上的全宽（非 448+64 打包）KV 写路径使用。每个 `quant_group_size` 列一组：MXFP8 模式 scale 取 `max(amax, 1e-4) / FP8 上限` 并在 `round_scale=True` 时向上舍入到 2 的幂，值除以 scale 后饱和转换为 FP8；MXFP4 模式按 OCP MX 约定取 scale = 2^(floor(log2 amax) − 2)，值饱和转换为 e2m1（每字节打包两个半字节，低位在前的偶元素）。cache 行布局为 `[量化值 | uint16 scale]`：layout 1（2D cache）每行按 32 字节对齐补零；layout 2（4D cache）块页内紧凑存放，块间距（`cache.stride(0)`）须 32 字节对齐。槽位为负或超出 cache 行数时跳过该行，未写入区域保留原内容。

## 参数与返回值

所有输入和输出 Tensor 均使用同一张 NPU（`torch_native` 亦可在 CPU 上做正确性参考）。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `cache` | `torch.Tensor` | `[N, C]` 或 `[B, S, 1, C]` | `float8_e4m3fn` / `float8_e5m2`（MXFP8）或 `uint8`（MXFP4） | 必填 | 调用方分配的缓存。2D 要求连续且 `C ≥ RoundUp32(值区 + scale 区)`；4D 要求 dim2 为 1、末维连续、行宽 `C ≥ 值区 + scale 区` 且块间距 32 字节对齐。 |
| `x` | `torch.Tensor` | `[T, D]` | `bfloat16` | 必填 | 连续输入，`0 < D ≤ 8192` 且 `D % quant_group_size == 0`。 |
| `slot_mapping` | `torch.Tensor` | `[T]` | `int32` / `int64` | 必填 | 各输入行的目标槽位；负值或越界跳过。 |
| `quant_group_size` | `int` | — | — | `32` | MXFP8 固定 32；MXFP4 可选 16 或 32。 |
| `quant_mode` | `str` | — | — | `"mxfp8_bf16"` | `"mxfp8_bf16"` 或 `"mxfp4_bf16"`（大小写不敏感，忽略首尾空白）。 |
| `round_scale` | `bool` | — | — | `True` | MXFP8 的 scale 是否向上舍入到 2 的幂；MXFP4 忽略。 |
| `x_scale` | `float` | — | — | `1.0` | 预留参数，必须为 1.0。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字，可选 `"auto"`, `"cann_recipes_infer"`, `"torch_native"`。 |

无返回值；`cache` 原地更新。

## cann_recipes_infer 后端

仅 arch35，要求初始化的 CANN SDK 为 9.2.x；kernel 由 wheel 内捆绑的 cann-recipes-infer vendor 提供（`torch.ops.sgl_cann_recipes_infer.kv_compress_epilog_v2`），缺失时给出明确错误，不回退其他后端。接口校验由该绑定完成；cache 行宽与块步长由 host tiling 校验。

## torch_native 后端

独立 eager PyTorch 参考，按 kernel 位级语义（E8M0 式 scale、RINT 舍入的 FP8/FP4 转换、分页寻址）在输入设备上重写相同区域；仅用于正确性检查，不自动替代生产实现，不保证图捕获。非有限输入的行为未与硬件逐位对齐。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog_v2

x = torch.randn(128, 512, device="npu", dtype=torch.bfloat16)
slots = torch.arange(128, device="npu", dtype=torch.int32)
slots[-1] = -1
# MXFP4-g16 全宽行：值区 256 字节 + 32 组 x 2 字节 scale = 320 字节行宽。
cache = torch.full((1, 128, 1, 320), 0, device="npu", dtype=torch.uint8)
kv_compress_epilog_v2(cache, x, slots, quant_group_size=16,
                      quant_mode="mxfp4_bf16", impl="auto")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog_v2.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

该测试只在 CANN 9.2.x + arch35 上执行：版本不匹配时整文件跳过，arch22 上由注册表判定后跳过。

[上游算子仓库](https://gitcode.com/cann/cann-recipes-infer)
