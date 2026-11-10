# indexer_prologue_k

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | 不支持 | 支持 |
| torch_native | 不支持 | 支持 |

## 函数原型

`from sgl_kernel_npu.operators.indexer import indexer_prologue_k`

```python
import torch

def indexer_prologue_k(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    *,
    cache_index: torch.Tensor,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
    impl="auto"
): ...
```

## 功能说明

Indexer K 前处理：BF16 投影、RMSNorm、RoPE、MXFP4 E2M1 量化，并写入分页 K cache。

输入 `latent` 为 BF16 `[T,H]`，`wk` 为 BF16 `[D,H]` 的 FRACTAL_NZ 权重；`norm_weight`、`rope_sin`、`rope_cos` 为 FP32。`cache_index` 为 INT64 `[T]`，值 `-1` 跳过写入。`k_cache` 为 UINT8；`storage_mode=0` 时使用独立 `k_scale_cache`，`storage_mode=1` 时按 `combined_block_size` 合并存储。`D` 是 32 的倍数，RoPE 维度为偶数。

计算语义（量化边界使用 BF16）：

```text
p = BF16(latent @ wk.T); k = RoPE(BF16(p / sqrt(mean(p**2) + norm_eps) * norm_weight)); cache = MXFP4(k)
```

每 32 个值共用一个 2 的整数次幂 scale：`s = 2**ceil(log2(max(amax, 6 * 2**-126) / 6))`。幅值映射到 E2M1 的 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，无 zero point；中点取较小幅值，超过范围饱和到 6。每字节低半字节存第一个值，高半字节存第二个值。E8M0 scale 字节保存 `log2(s) + 127`。

## 参数

T 为查询行数，B 为请求数，P 为物理页数，S 为每页槽位数。

| 参数 | 默认值 | 类型、形状与含义 |
| --- | --- | --- |
| `latent` | 必填 | BF16 [T,H]，RoPE 前的压缩 latent。 |
| `wk` | 必填 | BF16 [D,H]，投影权重；DSL 路径使用 FRACTAL_NZ。 |
| `norm_weight` | 必填 | FP32 [D]，RMSNorm 权重。 |
| `rope_sin` | 必填 | FP32 [T,Dr]，最后 Dr 维的 RoPE 正弦系数。 |
| `rope_cos` | 必填 | FP32 [T,Dr]，最后 Dr 维的 RoPE 余弦系数。 |
| `k_cache` | 必填 | UINT8 可写缓存：模式 0 为 [P,S,1,D/2]；模式 1 为 [P,S/G,1,G*(D/2+D/32)]。 |
| `k_scale_cache` | `None` | 可选 UINT8 [P,S,1,D/32]，模式 0 的独立 E8M0 scale。 |
| `cache_index` | 必填 | INT64 [T]，展平的缓存槽位；-1 跳过该行。 |
| `storage_mode` | 必填 | int：0 分离存储数据与 scale，1 合并存储。 |
| `norm_eps` | 必填 | float，RMSNorm 的 epsilon。 |
| `combined_block_size` | `-1` | int G，每条合并记录的行数；模式 1 要求为 S 的正因数。 |
| `impl` | `'auto'` | str，auto 选择 cannbot_dsl；torch_native 是独立计算参考。 |

## 返回值

| 值 | 含义 |
| --- | --- |
| k_cache | 返回原地更新的同一个张量；scale cache 也会更新。 |

## cannbot_dsl

`auto` 默认选择此后端。需要 A5 和环境中安装的兼容 CANNBot-DSL 运行时，算子入口不强制检查版本。输入在同一设备，首次调用 JIT 编译；图捕获前预热实际形状。

## torch_native

独立的 CPU Torch 精度参考，结果复制回输入设备，保留原地输出语义。无需 DSL；不适合性能执行或 NPU graph capture。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import indexer_prologue_k

x = torch.randn(1, 512, device="npu", dtype=torch.bfloat16)
w = torch_npu.npu_format_cast(
    torch.randn(128, 512, device="npu", dtype=torch.bfloat16),
    torch_npu.Format.FRACTAL_NZ,
)
cache = torch.zeros(1, 64, 1, 64, device="npu", dtype=torch.uint8)
scale = torch.zeros(1, 64, 1, 4, device="npu", dtype=torch.uint8)
indexer_prologue_k(
    x,
    w,
    torch.ones(128, device="npu"),
    torch.zeros(1, 64, device="npu"),
    torch.ones(1, 64, device="npu"),
    cache,
    scale,
    cache_index=torch.tensor([0], device="npu", dtype=torch.int64),
    storage_mode=0,
    norm_eps=1e-6,
)
```

## 设备测试

```bash
python -m pytest tests/python/operators/indexer/test_indexer_prologue_k.py --npu-device=0 --enable-perf=false -v -s
```
