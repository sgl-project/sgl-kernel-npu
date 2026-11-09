# sparse_block_estimate

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import sparse_block_estimate

def sparse_block_estimate(
    query, key, actual_seq_lengths=None, actual_seq_lengths_kv=None,
    input_layout="BNSD", stride=8, sparse_size=128, num_heads=1,
    num_key_value_heads=1, scale_value=1, threshold=1, causal=True,
    keep_sink=True, keep_recent=True, row_sparse=1, *, impl="auto",
):
    ...
```

`auto` 与 `ascendc` 指向同一入口，没有自动回退。`import sgl_kernel_npu` 后可直接调用 `torch.ops.npu.sparse_block_estimate`；其业务参数和默认值一致，没有 `impl`。

## 功能说明

估计稀疏注意力块，返回块选择 mask 及每行保留块数。对于连续 BNSD 输入，令采样跨度为 $s$、头维为 $D$，Q 每个跨度内反序，与 K 的连续跨度向量做 Matmul：

$$
A_{ij}=\sum_{r=0}^{s-1}\sum_{d=0}^{D-1}Q_{is+s-1-r,d}K_{js+r,d},\qquad Z_{ij}=\mathrm{scale\_value}\,A_{ij}.
$$

令 $R=\mathrm{sparse\_size}/s$，在线 softmax 的指数结果先转为 FP16，按 $R$ 个列采样归约，再用 FP16 重标定及行归约形成块分数。原 `ScoreCompute` 的排序、累计阈值、row_sparse 和强制保留分支产生 $M_{uv}\in\{0,1\}$，计数为 $C_u=\sum_v M_{uv}$。实际舍入和分块顺序以实现为准，不等价于对稠密 attention 直接执行标准 Top-k。

## 参数与返回值

公开 wrapper 不增加输入检查。保留旧入口对 num_heads 和 sparse_size 的非零检查，以及旧 Tensor 转换 helper 对映射不支持 dtype 的拒绝检查；其他失败由 Tensor 操作、ACLNN 或旧 Tiling 触发。

| 参数 | 类型 / shape / dtype | 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | NPU Tensor；BNSD `[B,Nq,Sq,D]`、BSND `[B,Sq,Nq,D]`、BSH `[B,Sq,Nq*D]`；FP16/BF16 | 必填 | 保留旧存储处理，未新增连续性检查 |
| `key` | NPU Tensor；对应布局，以 Nkv、Sk 代替 Nq、Sq；FP16/BF16 | 必填 | 常规安全调用使用与 query 相同的长度和 dtype |
| `actual_seq_lengths` | `list[int]` 或 `None` | `None` | Q 实际长度；空/None 使用完整长度，单元素广播，多元素按 batch 取值 |
| `actual_seq_lengths_kv` | `list[int]` 或 `None` | `None` | KV 实际长度，规则同上；不是 Tensor |
| `input_layout` | `str` | `"BNSD"` | BNSD、BSND、BSH；旧 Torch wrapper 未定义 TND 输出 |
| `stride` | `int` | `8` | 常规用例取正数并整除 sparse_size，不新增校验 |
| `sparse_size` | `int` | `128` | 稀疏块大小，保留非零检查 |
| `num_heads` | `int` | `1` | Host 调度的 Q head 数，调用方按实际 shape 提供 |
| `num_key_value_heads` | `int` | `1` | KV head 数，用于原 GQA 映射 |
| `scale_value` | `float` | `1` | Matmul 后缩放，显式 0 保持 0 |
| `threshold` | `float` | `1` | 原累计分数阈值 |
| `causal` | `bool` | `True` | 原 causal 模板分支，未改为逐 token 三角 mask |
| `keep_sink` | `bool` | `True` | 原首列处理开关，最终强制首列行为仍按旧实现 |
| `keep_recent` | `bool` | `True` | 原末列开关，非 causal 分支也会强制末列 |
| `row_sparse` | `float` | `1` | 有效列数至少为 10 时按旧条件应用保留比例，不裁剪取值 |
| `impl` | 仅限关键字 `str` | `"auto"` | auto、ascendc、torch_native |

令 $L=\lceil Sq/\mathrm{sparse\_size}\rceil$，$L_{32}=32\lceil L/32\rceil$：

| 返回值 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `sparse_mask` | `[B,Nq,L,L32]` | int8 | 新建输出，与 query 同设备，保留旧 NPU format 分配 |
| `sparse_count_table` | `[B,Nq,L]` | int32 | 每行保留块数，与 query 同设备 |

实际 Q 长度之外的行未保证写入。mask 按 **Q 长度**分配，Kernel 写入跨度按 **K 长度**计算；不等长输入保留旧限制，不扩展为通用 cross-attention。输入不应被修改。

## ascendc 后端

调用随 wheel 打包的 SglSparseBlockEstimate ACLNN/Host/Kernel，不依赖旧 attentions 包；A2/A3 共用源码。Torch 默认 causal=True，CANN OpDef 默认 false；adapter 显式传入 Torch 的值。

计算类保持不变，包括 BSND/BSH 原始字节拷贝 stride、Matmul 地址关系、首末列特殊分支、可变全局 Tiling 基数及未写输出区域。调用顺序可能影响分块。

## torch_native 后端

独立 PyTorch 实现，按旧源码表达原始布局取数、Q 反序、在线 max/sum、FP16 两级归约、排序和阈值边界；包含 keep_sink=False 时也最终强制首列的旧行为。读取当前 NPU 的 Vector core 数确定分块，保留自身 Tiling 调用历史。

共享 check_accuracy 对两个整数输出要求精确相等。CPU 算术检查不代表设备精度通过，专用 Matmul/Softmax 的数值对齐及新旧设备对照仍待实机验证。比较时两套后端需重放相同调用序列。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import sparse_block_estimate

q = torch.randn(1, 1, 2048, 64, device="npu", dtype=torch.float16) * 0.1
k = torch.randn_like(q)
mask, count = sparse_block_estimate(q, k, threshold=0.8, impl="ascendc")
mask_ref, count_ref = sparse_block_estimate(q, k, threshold=0.8, impl="torch_native")
torch.testing.assert_close(mask, mask_ref, atol=0, rtol=0)
torch.testing.assert_close(count, count_ref, atol=0, rtol=0)
```
