# qsa_sparse_attention

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.attention import qsa_sparse_attention

def qsa_sparse_attention(q, k, v, slots, softmax_scale=None, *, impl="auto"):
    ...
```

## 说明

按物理槽位 slots[R,S] 计算稀疏 GQA。BF16 Q[R,Hq,256]、K/V[N,Hkv,256]；支持本地头数 24:2、12:1、6:1、3:1。S=(B+1)*c-1+t，其中 B=512/2048、c>=2，t=0 或 3<=t<=c；例如 2051/2054/2055。输出与 Q 同形状且连续；全 -1 行输出零。softmax_scale=None 或 0 保留默认 1/16。

所有张量位于同一 NPU，输入不修改；零行返回空输出。

对每个 query 头选择对应 GQA 键值头和 slots 指定的键值，再计算 `softmax(scale * QK.T) @ V`。独立 oracle 在 CPU FP64 中计算选中键值的 attention，直接舍入到 BF16。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [R,Hq,256], BF16 | required | NPU；最后一维连续，允许行和头之间有间距。 |
| `k / v` | Tensor | [N,Hkv,256], BF16 | required | q 同设备连续缓存；允许存储偏移。 |
| `slots` | Tensor | [R,S], int32 | required | 同设备、列连续；负数为 padding，非负值必须小于 N。 |
| `softmax_scale` | float or None | — | None | None 和数值零均使用 1/16。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | q.shape | BF16 | 输入设备；新分配的连续张量。 |

## triton

默认实现。启动前检查输入元数据，支持 graph capture，以及原地更新输入后的 replay。

## torch_native

独立 CPU FP64 参考实现，结果返回输入设备。仅在图外使用，通过 `impl="torch_native"` 显式选择。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import qsa_sparse_attention

q = torch.randn(1, 3, 256, device="npu", dtype=torch.bfloat16)
k = torch.randn(16, 1, 256, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
slots = torch.full((1, 2051), -1, device="npu", dtype=torch.int32)
slots[:, :16] = torch.arange(16, device="npu", dtype=torch.int32)
y = qsa_sparse_attention(q, k, v, slots, impl="auto")
```
