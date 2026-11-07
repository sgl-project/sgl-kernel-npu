# compressor

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

```python
from sgl_kernel_npu.operators.attention import compressor

compressor(
    x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
    state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    rope_head_dim=64, cmp_ratio=4, coff=1, norm_eps=1e-6,
    rotary_mode=2, cache_mode=1, state_cache_stride_dim0=0, *, impl="auto",
)
```

`auto` 选择 `ascendc`，不会自动回退。`rotary_mode` 默认值为 `2`（交错），
这是唯一已编译的模式。旧默认值为 `1`；现在省略该参数会使用模式 2。
显式传入模式 1 会在启动前报错。

模式 1 的报错方式有一项明确差异：对于其他参数合法的非空输入，旧入口没有
匹配的计算分支，可能触发设备端 `Trap`；新入口在启动前拒绝调用。
对于通过原有检查的空输入，旧路径即使使用模式 1，也可能直接返回而不计算；
新入口对空输入仍然拒绝模式 1。模式 2 的计算实现保持不变。

## 功能说明

用于 DeepSeek-V4 KV 压缩：对投影后的 token 分组聚合，执行 RMSNorm
和交错 RoPE，并原地更新 KV/score 状态缓存。
令 `r=cmp_ratio`、`c=coff`、`D=len(norm_weight)`，以 FP32 计算：

\[
 K_t=x_t W_{kv}^{T},\qquad
 G_t=x_t W_{gate}^{T}+APE_{p_t\bmod r}.
\]

对于结束于绝对位置 `e` 的完整压缩组（不含 `e`，且 `e` 为 `r` 的倍数），
重叠分量 `j=0,...,c-1` 使用位置
`e-(c-j)r,...,e-(c-j-1)r-1` 以及特征区间 `jD:(j+1)D`。
每个特征在这些 `c*r` 个分数上做 softmax，用所得权重聚合 KV。
负位置的 KV 为零、分数为负无穷；本次调用起始位置之前的历史从状态缓存读取。

\[
 u_d=\sum_{j,t}\operatorname{softmax}_{j,t}(G_{t,jD+d})K_{t,jD+d},\quad
 v_d=\gamma_d u_d/\sqrt{D^{-1}\sum_i u_i^2+\epsilon}.
\]

最后 64 个特征按相邻元素对执行交错 RoPE，使用调用者提供的正弦、余弦，
随后转换为 `x.dtype`。起始位置 `p`、有效长度 `n` 对应的有效输出行数为
`floor((p+n)/r)-floor(p/r)`。BSH 的有效行位于各请求输出的开头，
TH 按请求顺序紧密排列所有有效行。剩余输出行内容未定义，不应读取。

分页模式保存新 KV/score，但不写入虚拟页 0。模式 2 保存当前原始 token 的尾部。
令 `E=p+n`、`q=floor(E/r)*r`、`keep=max(q-(c-1)*r,0)`。
arch22 和 arch35 NORMAL 实现还保留验证所需的尾部：当 `L>c*r` 时，
令 `pad=L-c*r+2`，将 `keep` 降为 `min(keep,max(E-pad,0))`。
arch35 短 BSH 的 FULL_LOAD 路径（`S<=4` 且 `B*S<=256`）使用原始 `keep`，不加该 pad。
绝对位置不小于 `keep` 的新行会被写入，允许从压缩块内部开始；其他缓存行保留原值。

## 参数与返回值

所有张量须连续且位于同一 NPU。`H` 为 `[1024,10240]` 内 512 的倍数，
`D` 为 128 或 512。令 `W=c*D`；BSH 的 `Sr=ceil(S/r)`，
TH 的 `Sr=min(T,T//r+B)`。

| 参数 | 类型 | 含义与约束 |
| --- | --- | --- |
| `x` | Tensor | FP16/BF16，形状 `[B,S,H]`（BSH）或 `[T,H]`（TH）；token 相关维度可为零。 |
| `wkv` | Tensor | 与 `x` 同 dtype，`[W,H]`，KV 投影权重。 |
| `wgate` | Tensor | 与 `x` 同 dtype，`[W,H]`，分数投影权重。 |
| `state_cache` | Tensor | 可变 FP32 `[P,L,2*W]`，`L>=1`；先 KV 后 scores。起始位置非零时须提供有效历史。 |
| `ape` | Tensor | FP32 `[r,W]`，位置分数偏置。 |
| `norm_weight` | Tensor | FP32 `[D]`，RMSNorm 缩放权重。 |
| `rope_sin` | Tensor | FP32，BSH 为 `[B,Sr,64]`，TH 为 `[Sr,64]`，与输出行布局对应。 |
| `rope_cos` | Tensor | FP32，与 `rope_sin` 同形状。 |
| `state_block_table` | Tensor | INT32，虽然默认 `None`，实际必需。模式 1 为 `[B,M]` 页号；arch22 模式 2 为 `[B,c*r+capacity]` 的展平 token 位置表，绝对位置 `t` 使用列 `c*r+t-p`；arch35 模式 2 为 `[B]` 请求 bank 号，位置为 `bank*L+t%L`，要求 `P>=B`。也兼容 arch22 模式 1 的单页 `[B]` 表；arch35 模式 2 的连续 `[B,M]` 表取展平后的前 B 项作为 bank 号（包括 `[B,1]`）。所有访问位置须落在缓存内，并提供足够的历史和当前 token 列。 |
| `cu_seqlens` | Tensor 或 None | INT32 `[B+1]`，默认 `None`。TH 必需，BSH 禁止；非递减、首项 0、末项 T，相邻差为输入容量；`T=0` 时 `[0]` 表示零请求。 |
| `seqused` | Tensor 或 None | INT32 `[B]`，默认 `None` 使用完整容量；各项位于 `[0,capacity]`。 |
| `start_pos` | Tensor 或 None | INT32 `[B]`，非负绝对起始位置；默认 `None` 表示零。 |
| `rope_head_dim` | int | 必须为 64，默认 64。 |
| `cmp_ratio` | int | 2、4、8、16、32、64 或 128，默认 4。 |
| `coff` | int | 1 或 2，默认 1；2 表示包含一个前置组。 |
| `norm_eps` | float | RMSNorm epsilon，默认 `1e-6`。 |
| `rotary_mode` | int | 默认 2（交错），仅支持此模式。 |
| `cache_mode` | int | 1（分页）或 2（显式位置/ring），默认 1。 |
| `state_cache_stride_dim0` | int | 默认 0 推导为 `L*2*W`；显式值须等于该值且不超过有符号 INT32 范围。 |
| `impl` | str | `auto`（默认）、`ascendc` 或 `torch_native`。 |

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| 压缩后的 KV | Tensor | 连续，与 `x` 同设备和 dtype，形状 `[B,Sr,D]` 或 `[Sr,D]`。仅上述有效行有定义；`state_cache` 同时被原地更新。 |

## ascendc 后端

通过 CANN/ACLNN 调用原有的架构专用 Ascend C 计算实现，支持 FP16/BF16、
TH/BSH、重叠压缩和分页/显式位置/ring 缓存。
模式 2 的缓存表格式随架构不同，不能互换。
输入值和表项须描述合法且无写入冲突的请求；元数据检查不会读取设备索引值。

Python 入口负责选择后端并兼容旧缓存表的视图。Torch 适配层检查设备、
存储布局、dtype，以及分配输出必需的元数据。参数取值范围和张量形状关系
继续由原有 Host tiling 通过 CANN 检查。空输入也走 ACLNN 查询与执行流程。
缺省可选张量使用 CANN 的缺省输入表示，避免旧 helper 对 `None` 返回非空空形状的问题。

## torch_native 后端

参考实现检查自身支持的模式和必需元数据，不重复原生后端的硬件及 tiling 约束。

独立参考实现使用 CPU FP32 PyTorch 运算，将输出和更新后的缓存复制回输入设备。
用于精度对照，不支持 NPU graph capture。矩阵乘法和归约可能与 Ascend C
产生小幅浮点误差。

## 使用示例

分页模式示例不依赖两种架构模式 2 的缓存表差异。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import compressor

device = "npu"
B, S, H, D, r = 1, 4, 1024, 128, 4
x = torch.randn(B, S, H, device=device, dtype=torch.bfloat16) * 0.02
wkv = torch.randn(D, H, device=device, dtype=x.dtype) * 0.02
wgate = torch.randn_like(wkv) * 0.02
state = torch.zeros(2, 16, 2 * D, device=device)
out = compressor(
    x, wkv, wgate, state,
    torch.zeros(r, D, device=device), torch.ones(D, device=device),
    torch.zeros(B, 1, 64, device=device), torch.ones(B, 1, 64, device=device),
    state_block_table=torch.ones(B, 1, device=device, dtype=torch.int32),
    cmp_ratio=r, rotary_mode=2, cache_mode=1, impl="ascendc",
)
print(out.shape)  # [1, 1, 128]；state 同时被原地更新。
```
