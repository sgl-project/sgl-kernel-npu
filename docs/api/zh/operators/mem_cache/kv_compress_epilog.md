# kv_compress_epilog

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(
    kv_compress_cache, x, slot_mapping, quant_group_size, quant_mode,
    round_scale_flag, layout, *, impl="auto",
): ...
```

默认的 `auto` 选择 `ascendc`。

## 功能说明

量化 KV Cache 输入行，并根据 slot_mapping 写入指定缓存槽位，用于 DeepSeek-V4 的 NPU 缓存路径。每行前 D − 64 列参与量化，最后 64 列 BF16 RoPE 数据按字节保留。模式 1 使用 FP8 量化，每 128 个值对应一个 FP32 scale；模式 2 使用 MXFP8 量化，每 64 或 128 个值对应一个 E8M0 指数 scale。布局 1 按槽位存放打包记录，布局 2 在每个缓存块内分别存放值区和 scale 区。槽位为 -1 时跳过该行，未写入槽位保留原内容；函数原地修改 cache 并返回 None。

## 参数与返回值

所有输入和输出 Tensor 均使用同一张 NPU。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `kv_compress_cache` | `torch.Tensor` | 见布局说明 | `float8_e4m3fn` / `float8_e5m2` | 必填 | 调用方分配的打包缓存。布局 1 按槽位分行，也支持 [slots, 1, width]；布局 2 使用 [blocks, block_size, width] 或 [blocks, block_size, 1, width]。 |
| `x` | `torch.Tensor` | `[N, D]` | `bfloat16` | 必填 | 稠密输入，N 大于零，末尾 64 列为 RoPE 数据。 |
| `slot_mapping` | `torch.Tensor` | `[N]` | `int32` / `int64` | 必填 | 各输入行的目标槽位。-1 表示跳过该行。 |
| `quant_group_size` | `int` | — | — | 必填 | MXFP8 的分组宽度，可选 64 或 128。FP8 固定每 128 列一组。 |
| `quant_mode` | `int` | — | — | 必填 | 1 表示 FP8，scale 为 FP32；2 表示 MXFP8，scale 为 E8M0 指数字节。 |
| `round_scale_flag` | `bool` | — | — | 必填 | MXFP8 模式下，设为 True 时将每个非零的带偏置 scale 指数加一。 |
| `layout` | `int` | — | — | 必填 | 1 按 cache 的行间距逐槽存储。FP8 顺序为量化值、RoPE、scale；MXFP8 顺序为 RoPE、量化值、scale。2 仅支持 MXFP8，每个 block 先放所有槽位的值区，再放所有槽位的 scale 区。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字，可选 `"auto"`, `"ascendc"`, `"torch_native"`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | `NoneType` | 原地更新传入的缓存，未写入槽位保留原值。 |

## ascendc 后端

支持 arch35。布局 1 的打包行宽按 128 字节向上取整。布局 2 每个槽位的 scale 区按 8 字节向上取整，已写槽位仅有效分组的指数字节有规定值。

## torch_native 后端

在输入设备上按指定布局打包。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

x = torch.ones((2, 192), device="npu", dtype=torch.bfloat16)
slots = torch.tensor([1, -1], device="npu", dtype=torch.int32)
cache = torch.zeros((2, 384), device="npu", dtype=torch.uint8).view(torch.float8_e4m3fn)
kv_compress_epilog(cache, x, slots, quant_group_size=64, quant_mode=2,
                   round_scale_flag=True, layout=1, impl="ascendc")
```

## 按 SDK 选择 DSV4 打包缓存格式

### 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 不支持 | 支持 |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 不支持 | 支持 |

### 接口

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(kv_compress_cache, x, slot_mapping, *, quant_group_size=64, quant_mode=None, round_scale=True, x_scale=1.0, impl='auto'):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

### 说明

仅 A5；64 元素量化组，输入宽度 448+64，先通过 get_npu_kv_quant_path() 确定分配格式。

分组量化 NoPE 并散写打包行，RoPE 尾部保持未量化；-1 槽跳过，0 槽有效。比较有效语义字节，不比较对齐填充。

### 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| cache | Tensor | 必填 | Mutable paged cache; packed KV uint8/FP8 bytes, indexer FP8 payload |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| slot_mapping | Tensor | 必填 | int32 [T] destination slot; -1 skips a write |
| quant_group_size | int | 64 | 64 for migrated packed KV |
| quant_mode | int / str | None | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
| round_scale | bool | True | Scale rounding; CANN 9.1 normal FP8 ignores it |
| x_scale | float | 1.0 | Additional vendor quantization scale; migrated packed KV requires 1.0 |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| cache | Tensor | The same mutated packed input cache. |

### cann_ops_transformers_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。

### ascendc

Source: [sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp). The C++ wrapper directly launches the checked-in Ascend C kernel; this is not inferred from the Torch namespace. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

### torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

### 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

from sgl_kernel_npu.operators.mem_cache import get_npu_kv_quant_path
path = get_npu_kv_quant_path()  # Resolve before allocation; A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
x = torch.randn(1, 512, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
kv_compress_epilog(cache, x, slots, impl="auto")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
