# split_qkv_rmsnorm_rope_pos_cache_half_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    rope_dim=None,
    cast_norm_to_bf16=True,
    wide_grid_min_tokens=1024,
    *,
    impl="auto",
):
    ...
```

`auto` 与 `triton` 指向相同的延迟导入入口，不自动回退。实现声明支持全部已知代际，但实际硬件执行仍待验证。

## 功能说明

拆分 Q/K/V，可选地对 Q/K 按 head 做 RMSNorm。从 positions 选中的 packed [cos_half,sin_half] 缓存行读取 NeoX 旋转频率。索引先转 int32，再在设备上截断到合法范围。归一化结果可选在 RoPE 前经过 BF16 舍入；最终 Q/K 的所有特征（包括未旋转尾部）均先经 BF16 舍入，再转为输入 dtype；V 保持原值和原 dtype。

$$
p=\mathrm{clamp}(\mathrm{int32}(\mathrm{positions}),0,M-1),\quad A=\mathrm{optionalRMSNorm}(X),\quad C=\begin{cases}\mathrm{BF16}(A),&\mathrm{cast\_norm\_to\_bf16},\\A,&\text{otherwise},\end{cases}\quad Y_{q,k}=\mathrm{cast}_{\mathrm{input.dtype}}(\mathrm{BF16}(\mathrm{RoPE}_R(C))).
$$


## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input_tensor` | Tensor | [N,Q+2K] | floating | required | 连续拼接的 Q/K/V。 |
| `positions` | Tensor | N values | int32/int64 | required | 每 token 一个索引，与输入同设备；先转 int32 再截断。 |
| `cos_sin_cache` | Tensor | [M,R], M>0 | floating | required | 先 cos_half 后 sin_half 的 packed 缓存，与输入同设备，R 等于 rope_dim。 |
| `q_hidden_size` | int | — | — | required | 正整数 Q，必须能被 K 整除。 |
| `kv_hidden_size` | int | — | — | required | 正整数 K，必须能被 head_dim 整除。 |
| `head_dim` | int | — | — | required | 每个 head 的正整数维度 D，不接受 bool。 |
| `eps` | int/float/None | — | — | None | None 跳过归一化，否则为有限正 Python 标量。 |
| `q_weight` | Tensor/None | at least D values | floating | None | 启用归一化时必填，连续且与输入同设备。 |
| `k_weight` | Tensor/None | at least D values | floating | None | 启用归一化时必填，连续且与输入同设备。 |
| `q_bias` | Tensor/None | at least D values | floating | None | 可选，须与 k_bias 成对提供；仅在归一化时应用。 |
| `k_bias` | Tensor/None | at least D values | floating | None | 可选，须与 q_bias 成对提供；连续且与输入同设备。 |
| `rope_dim` | int/None | — | — | None | 默认 D；须为正偶数且不大于 D。 |
| `cast_norm_to_bf16` | bool | — | — | True | 在 RoPE 前将 Q/K 经 BF16 舍入，跳过归一化时也生效。 |
| `wide_grid_min_tokens` | int | — | — | 1024 | 选择 Triton wide grid 的正整数 token 阈值。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字参数：auto、triton、torch_native。 |

floating 指 float16、bfloat16 或 float32；参与运算的 tensor 须与输入同设备，归一化向量连续且可使用不同浮点 dtype。N 可以为零。精度比较使用有限输入。正整数选项不接受 bool。

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [N,Q] | input.dtype | 与输入同设备的新建连续输出。 |
| `k` | Tensor | [N,K] | input.dtype | 与输入同设备的新建连续输出。 |
| `v` | Tensor | [N,K] | input.dtype | 与输入同设备的新建连续输出。 |

不修改输入、权重或 bias。两后端分别使用各自 norm/norm_utils.py 检查元信息，不共享校验依赖。Triton 额外要求 NPU；非法元信息在启动前抛出 AssertionError，检查不将设备数值回读主机。

## triton 后端

保留原 kernel、RoPE 前确定性 BF16 RNE helper 和 grid 选择策略。低于 wide_grid_min_tokens 时按 KV head 分列；达到阈值且 Q 宽度不超过 8192 时使用 wide grid，超过该宽度仍按 head 分列。head_dim 必须为 2 的幂。允许非连续 positions/cache，wrapper 会转为连续存储。索引值不回读主机，保留图捕获能力。空 token 输入不启动 kernel。

原 `sgl_kernel_npu.norm` 脚本保持不变；这里的输入检查属于迁移后的实现。

## torch_native 后端

使用独立的 PyTorch FP32 归一化及旋转，并显式匹配 kernel 的 BF16 转换边界，而非照搬旧 golden 的低精度分步路径。不再额外给权重加 1：MiniMax 已传入 Gemma 调整后的乘数。wide_grid_min_tokens 只校验合法性，不改变数学计算。旧 golden 未处理 kernel 的索引截断，这里一并保留。

不调用 Triton kernel，可在 CPU 或 NPU 上独立使用。归约顺序、rsqrt 近似和乘加融合可能造成数值差异；按 dtype 容差通过不代表与框架分步路径逐位等价。

CPU 运算支持不改变包初始化要求：常规公开接口导入仍依赖 torch_npu、原生动态库及 ACL 硬件识别。

FP16/FP32 存储类型的可选精度用例使用 BF16 容差，因为 Q/K 仍经过 BF16 量化。
另有精确值用例独立检查 RoPE 前舍入开关、最终 Q/K 舍入、V 原值复制及输入不变性。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

x = torch.randn(4, 256, device='npu', dtype=torch.bfloat16)
positions = torch.arange(4, device='npu', dtype=torch.int64)
cache = torch.cat((torch.ones(64, 16, device='npu'), torch.zeros(64, 16, device='npu')), dim=-1)
wq = torch.ones(64, device='npu')
wk = torch.ones_like(wq)
args = (x, positions, cache, 128, 64, 64)
options = dict(eps=1e-6, q_weight=wq, k_weight=wk, rope_dim=32)
outputs = split_qkv_rmsnorm_rope_pos_cache_half_npu(*args, **options)
reference = split_qkv_rmsnorm_rope_pos_cache_half_npu(*args, impl="torch_native", **options)
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
