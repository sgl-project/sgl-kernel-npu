# torch.ops.catlass_matmul_basic


## Function Description | 功能描述

### English:
This is the catlass version matmul kernel  `catlass_matmul_basic` kernel function

### 中文:
这是调用catlass模板库实现的矩阵乘法运算 `catlass_matmul_basic`内核方法

参考/Refs: [CATLSS](https://gitcode.com/cann/catlass)


## Interface Prototype | 接口原型

### Python Binding Definition
```python
import sgl_kernel_npu

torch.ops.npu.catlass_matmul_basic(
    input_a: torch.Tensor,        # bf16/fp16/fp32, [m, k]
    input_b: torch.Tensor,        # bf16/fp16/fp32, [k, n]
    output_c: torch.Tensor,       # bf16/fp16/fp32, [m, n]
    format_mode: str = None       # string "ND"/"NZ"
) -> None
```

### Kernel Definition | 核函数定义
```C++
extern "C" __global__ __aicore__ void catlass_matmul_basic(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
                                                           GM_ADDR gmWorkspace, GM_ADDR gmTiling)
```

## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description                          | 说明            |
|:----------------------|:----------------|:-------------------------------------|:--------------|
| `input_a`         | `torch.Tensor`  | input left tensor with shape (m, k)  | 左输入矩阵，（m,k)大小 |
| `input_b`      | `torch.Tensor`  | input right tensor with shape (k, n) | 右输入矩阵，（k,n)大小 |
| `format_mode`  | `[optional]string`                                            | weight format ND/NZ, default ND        | 权重格式ND/NZ, 默认为 ND


## Output Description | 输出说明

| Parameter Name (参数名称)  | DataType (数据类型) | Description                     | 说明            |
|:-----------------------|:----------------|:--------------------------------|:--------------|
| `output_c`    | `torch.Tensor`  | output tensor with shape (m, n) | 输出矩阵，（m, n）大小 |


## Constraints | 约束说明

### English:
`format_mode = "NZ"` is not implemented

### 中文:
`format_mode = "NZ"` 暂未实现

## Example | 调用示例

```python
import math
import sgl_kernel_npu
import torch
import torch_npu

device = torch.device('npu:0')

dtypes = [torch.float16, torch.bfloat16, torch.float32]
m, k, n = 128, 256, 256

for dtype in dtypes:
    a = torch.rand(m, k, dtype=dtype, device="npu")
    b = torch.rand(k, n, dtype=dtype, device="npu")
    res = torch.empty((m, n), dtype=dtype, device="npu")

    torch.ops.npu.catlass_matmul_basic(a, b, res)
```

# torch.ops.npu.fp8_w8a16_matmul

## Function Description | 功能描述

### English:
Catlass-based soft-FP8 (W8A16) matmul. Activations are BF16; weights are FP8 bits stored as int8 and are dequantized to BF16 with per-block scales, then GEMM is executed and BF16 output is produced.

### 中文:
基于 Catlass 的 soft-FP8（W8A16）矩阵乘：输入激活为 BF16；权重以 int8 形式存储 FP8 bit，按照块用 scale 做反量化，转换为 BF16，再进行 GEMM，输出为 BF16。

参考/Refs: Catlass (submodule)

---

## Interface Prototype | 接口原型

### Python Binding Definition
```python
import sgl_kernel_npu
import torch

torch.ops.npu.fp8_w8a16_matmul(
    mat1: torch.Tensor,     # bf16, [m, k]
    mat2: torch.Tensor,     # int8 (FP8 bits), [k, n]
    scale: torch.Tensor,    # fp32, per-block scale
    outDType: str = "bf16"  # output dtype string
) -> torch.Tensor           # [m, n]
```

### Kernel Definition | 核函数定义
```C++
extern "C" __global__ __aicore__ void catlass_fp8w8a16_matmul_bfloat16_t(
    GM_ADDR deviceA, GM_ADDR deviceB, GM_ADDR deviceScale, GM_ADDR deviceC,
    GM_ADDR deviceWorkspace, uint32_t m, uint32_t n, uint32_t k
);
```

---

## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description | 说明 |
|:--|:--|:--|:--|
| mat1 | torch.Tensor | BF16 activation matrix, shape (m, k) | BF16 左矩阵，形状 (m, k) |
| mat2 | torch.Tensor | FP8(E4M3) bits stored in int8, shape (k, n) | 以 int8 存储的 FP8 bit 权重，形状 (k, n) |
| scale | torch.Tensor | FP32 scale table for dequant (block-wise) | 反量化 scale 表（按 block） |
| outDType | str | output dtype string (now only support "bf16") | 输出 dtype 字符串（当前仅支持 "bf16"） |

---

## Output Description | 输出说明

| Output | DataType | Description | 说明 |
|:--|:--|:--|:--|
| returned tensor | torch.Tensor | output matrix with shape (m, n) | 返回输出矩阵 (m, n) |

---

## Constraints | 约束说明

### English:
- mat1 must be BF16 and 2D: [m, k].
- mat2 must be 2D: [k, n].
- Currently only supports ND format and RowMajor layout (i.e., standard contiguous row-major tensors). Other formats (e.g., NZ) are not supported.
- Scale is block-wise; for the common setting groupSize = 128, a typical scale table shape is:
  - scale_rows = ceil(k / 128)
  - scale_cols = ceil(n / 128)
- This op is built only when BUILD_CATLASS_MODULE=ON.

### 中文:
- mat1 必须是 BF16 且为 2D：[m, k]。
- mat2 必须为 2D：[k, n]。
- 当前仅支持 ND 与 RowMajor（行主序、连续内存）。其它格式（如 NZ）暂不支持。
- scale 为按 block 的布局；常见 groupSize = 128 时，典型 scale 形状为：
  - scale_rows = ceil(k / 128)
  - scale_cols = ceil(n / 128)
- 仅在 BUILD_CATLASS_MODULE=ON 时参与构建。

---

## Example | 调用示例
```python
import torch
import torch_npu
import sgl_kernel_npu

device = "npu"
m, k, n = 128, 256, 256
groupSize = 128
scale_rows = (k + groupSize - 1) // groupSize
scale_cols = (n + groupSize - 1) // groupSize

a = torch.randn(m, k, dtype=torch.bfloat16, device=device)

# NOTE: mat2 expects FP8 bits stored as int8; here is a placeholder.
b = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)

scale = torch.ones(scale_rows, scale_cols, dtype=torch.float32, device=device)

c = torch.ops.npu.fp8_w8a16_matmul(a, b, scale, "bf16")
print(c.shape, c.dtype)  # (m, n), bf16
```

---

# torch.ops.npu.fp8_w8a16_grouped_matmul

## Function Description | 功能描述

### English:
Catlass-based soft-FP8 (W8A16) grouped-matmul (GMM). Activations are BF16 [M, K], weights are [g, K, N] (FP8 bits stored as int8), and groupList describes how the M dimension is partitioned across groups.

### 中文:
基于 Catlass 的 soft-FP8（W8A16）分组矩阵乘（GMM）：输入激活 BF16 [M, K]，权重 [g, K, N]（FP8 bit 以 int8 存储），groupList 描述 M 维如何按组切分。

---

## Interface Prototype | 接口原型

### Python Binding Definition
```python
import sgl_kernel_npu
import torch

torch.ops.npu.fp8_w8a16_grouped_matmul(
    mat1: torch.Tensor,       # bf16, [M, K]
    mat2: torch.Tensor,       # int8 (FP8 bits), [g, K, N]
    scale: torch.Tensor,      # fp32, per-block scale
    groupList: torch.Tensor,  # prefix-sum describing groups (see "Constraints")
    outDType: str = "bf16"
) -> torch.Tensor             # [M, N]
```

### Kernel Definition | 核函数定义
```C++
extern "C" __global__ __aicore__ void catlass_fp8w8a16_gmm_bfloat16_t(
    GM_ADDR deviceA, GM_ADDR deviceW, GM_ADDR deviceScale, GM_ADDR deviceGroupList,
    GM_ADDR deviceC, GM_ADDR deviceWorkspace,
    uint32_t g, uint32_t m, uint32_t n, uint32_t k
);
```

---

## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description | 说明 |
|:--|:--|:--|:--|
| mat1 | torch.Tensor | BF16 activation, shape (M, K) | BF16 输入矩阵 (M, K) |
| mat2 | torch.Tensor | grouped weights, shape (g, K, N) | 分组权重 (g, K, N) |
| scale | torch.Tensor | FP32 scale table for dequant (block-wise) | 反量化 scale 表（按 block） |
| groupList | torch.Tensor | prefix-sum group descriptor over M | M 维分组的前缀和描述 |
| outDType | str | output dtype string (now only support "bf16") | 输出 dtype 字符串（当前仅支持 "bf16"） |

---

## Output Description | 输出说明

| Output | DataType | Description | 说明 |
|:--|:--|:--|:--|
| returned tensor | torch.Tensor | output matrix (M, N) | 返回输出矩阵 (M, N) |

---

## Constraints | 约束说明

### English:
- mat1 must be BF16 and 2D: [M, K].
- mat2 must be 3D: [g, K, N].
- groupList must be a prefix-sum representation of the group sizes along M:
  - 1D tensor of length g
- Currently only supports ND format and RowMajor layout (standard contiguous row-major tensors). Other formats (e.g., NZ) are not supported.
- Scale is block-wise (commonly groupSize=128), and must match the kernel's dequant packing strategy.
- This op is built only when BUILD_CATLASS_MODULE=ON.

### 中文:
- mat1 必须是 BF16 且为 2D：[M, K]。
- mat2 必须为 3D：[g, K, N]。
- groupList 必须用前缀和形式表示每组的 M 大小：
  - 1D 张量，长度为 g
- 当前仅支持 ND 与 RowMajor（行主序、连续内存）。其它格式（如 NZ）暂不支持。
- scale 为按 block 的布局（常见 groupSize=128），需要与 kernel 的权重打包/反量化策略一致。
- 仅在 BUILD_CATLASS_MODULE=ON 时参与构建。

---

## Example | 调用示例
```python
import torch
import torch_npu
import sgl_kernel_npu

device = "npu"
g = 4
K, N = 256, 256

Ms = [256, 128, 512, 64]
groupList = [0]
for mi in Ms:
    groupList.append(groupList[-1] + mi)
M = groupList[-1]

a = torch.randn(M, K, dtype=torch.bfloat16, device=device)

# NOTE: placeholder FP8-bits in int8
w = torch.randint(-128, 127, (g, K, N), dtype=torch.int8, device=device)

groupList_t = torch.tensor(groupList, dtype=torch.int32, device=device)

# scale layout depends on packing strategy; here is a placeholder
scale = torch.ones(1, dtype=torch.float32, device=device)

c = torch.ops.npu.fp8_w8a16_grouped_matmul(a, w, scale, groupList_t, "bf16")
print(c.shape, c.dtype)  # (M, N), bf16
```

---

# torch.ops.npu.fp8_w8a16_batch_matmul

## Function Description | 功能描述

### English:
Catlass-based soft-FP8 (W8A16) batch-matmul (BMM). mat1 is BF16 [b, M, K]; mat2 is [b, K, N] (FP8 bits stored as int8). Weights are dequantized using block-wise scales, then bf16 BMM is executed.

### 中文:
基于 Catlass 的 soft-FP8（W8A16）批量矩阵乘（BMM）：mat1 为 BF16 [b, M, K]；mat2 为 [b, K, N]（FP8 bit 以 int8 存储）。按块用 scale 做权重反量化，再执行bf16 BMM。

---

## Interface Prototype | 接口原型

### Python Binding Definition
```python
import sgl_kernel_npu
import torch

torch.ops.npu.fp8_w8a16_batch_matmul(
    mat1: torch.Tensor,     # bf16, [b, M, K]
    mat2: torch.Tensor,     # int8 (FP8 bits), [b, K, N]
    scale: torch.Tensor,    # fp32, per-block scale
    outDType: str = "bf16"
) -> torch.Tensor           # [b, M, N]
```

### Kernel Definition | 核函数定义
```C++
extern "C" __global__ __aicore__ void catlass_fp8w8a16_bmm_bfloat16_t(
    GM_ADDR deviceA, GM_ADDR deviceB, GM_ADDR deviceScale, GM_ADDR deviceC,
    GM_ADDR deviceWorkspace, uint32_t g, uint32_t m, uint32_t n, uint32_t k
);
```

---

## Parameter Description | 参数说明

| Parameter Name (参数名称) | DataType (数据类型) | Description | 说明 |
|:--|:--|:--|:--|
| mat1 | torch.Tensor | BF16 activations, shape (b, M, K) | BF16 输入 (b, M, K) |
| mat2 | torch.Tensor | weights (b, K, N), FP8 bits stored as int8 | 权重 (b, K, N)，FP8 bit 以 int8 承载 |
| scale | torch.Tensor | FP32 scale table (block-wise) | FP32 scale 表（按 block） |
| outDType | str | output dtype string (now only support "bf16") | 输出 dtype 字符串（当前仅支持 "bf16"） |

---

## Output Description | 输出说明

| Output | DataType | Description | 说明 |
|:--|:--|:--|:--|
| returned tensor | torch.Tensor | output tensor with shape (b, M, N) | 返回输出 (b, M, N) |

---

## Constraints | 约束说明

### English:
- mat1 must be BF16 and 3D: [b, M, K].
- mat2 must be 3D: [b, K, N].
- Currently only supports ND format and RowMajor layout (standard contiguous row-major tensors). Other formats (e.g., NZ) are not supported.
- Scale is block-wise; for the common groupSize = 128, a typical scale table shape is:
  - scale_rows = ceil(k / 128)
  - scale_cols = ceil(n / 128)
- This op is built only when BUILD_CATLASS_MODULE=ON.

### 中文:
- mat1 必须是 BF16 且为 3D：[b, M, K]。
- mat2 必须为 3D：[b, K, N]。
- 当前仅支持 ND 与 RowMajor（行主序、连续内存）。其它格式（如 NZ）暂不支持。
- scale 为按 block 的布局；常见 groupSize = 128 时，典型 scale 形状为：
  - scale_rows = ceil(k / 128)
  - scale_cols = ceil(n / 128)
- 仅在 BUILD_CATLASS_MODULE=ON 时参与构建。

---

## Example | 调用示例
```python
import torch
import torch_npu
import sgl_kernel_npu

device = "npu"
b, M, K, N = 4, 128, 256, 256
groupSize = 128
scale_rows = (K + groupSize - 1) // groupSize
scale_cols = (N + groupSize - 1) // groupSize

a = torch.randn(b, M, K, dtype=torch.bfloat16, device=device)

# NOTE: placeholder FP8-bits in int8
b_w = torch.randint(-128, 127, (b, K, N), dtype=torch.int8, device=device)

scale = torch.ones(scale_rows, scale_cols, dtype=torch.float32, device=device)

c = torch.ops.npu.fp8_w8a16_batch_matmul(a, b_w, scale, "bf16")
print(c.shape, c.dtype)  # (b, M, N), bf16
```
