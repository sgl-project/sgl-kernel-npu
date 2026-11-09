# la

LaserAttention 稠密注意力算子。

## 硬件支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | FP16 | — |
| `torch_native` | FP16 / BF16 | FP16 / BF16 |

## 签名

```python
from sgl_kernel_npu.operators.attention import la

def la(
    query,
    key,
    value,
    *,
    atten_mask=None,
    alibi_mask=None,
    drop_mask=None,
    scale_value=1.0,
    head_num=2,
    input_layout="BNSD",
    keep_prob=1.0,
    pre_tokens=2147483647,
    next_tokens=1,
    is_highPrecision=True,
    impl="auto",
):
    ...
```

`auto` 在 arch22 上选择 `ascendc`；`torch_native` 使用 PyTorch 实现。
在 arch35 上请求 `ascendc` 会直接报错，不会回退。

## 功能说明

对 BNSD 布局的 Q/K/V 计算稠密（非因果）注意力。分数、softmax 与 value 乘
都在 FP32 中完成，Kernel 直接输出 FP32：

```text
scores = scale_value * (query @ key^T)
probs  = softmax(scores)
out    = probs @ value
```

支持 GQA：`key`/`value` 的头数可以少于 `query`，每个 KV 头服务
`query_heads / kv_heads` 个查询头。

## 参数与返回值

`B` 为 batch，`Nq`/`Nkv` 为查询/KV 头数，`S` 为序列长度，`D` 为 head 宽度。

| 参数 | 形状 | 类型 / 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | `[B, Nq, S, D]` | FP16 / BF16 | 必须为 4 维。 |
| `key` | `[B, Nkv, S, D]` | 同 `query` | `Nq % Nkv == 0`。 |
| `value` | `[B, Nkv, S, D]` | 同 `query` | value 张量。 |
| `atten_mask` / `alibi_mask` | — | `None` | 为保持接口兼容而保留；稠密 Kernel 不使用。 |
| `drop_mask` | — | `None` | 为保持接口兼容而保留；dropout 关闭。 |
| `scale_value` | — | float，1.0 | 分数缩放系数。 |
| `head_num` | — | int，2 | 查询头数，应等于 `Nq`。 |
| `input_layout` | — | str，`"BNSD"` | 仅支持 BNSD。 |
| `keep_prob` | — | float，1.0 | 必须为 1.0。 |
| `pre_tokens` | — | int，2147483647 | 窗口边界；稠密路径使用默认值。 |
| `next_tokens` | — | int，1 | 窗口边界。 |
| `is_highPrecision` | — | bool，`True` | 高精度累加。 |
| `impl` | — | `"auto"` | 仅关键字：`auto`、`ascendc`、`torch_native`。 |

| 返回 | 形状 | 类型 |
| --- | --- | --- |
| `softmax_log_max_sum` | `[B, Nq, S]` | FP32 |
| `attention_out` | `[B, Nq, S, D]` | FP32 |

## ascendc 后端

- 要求所有张量位于同一 NPU；非连续输入会在内部转为连续工作张量。
- `softmax_log_max_sum` 仅为保持接口兼容而分配，当前 Kernel 从不写入，
  内容未定义；精度校验只应比较 `attention_out`。
- Kernel 以 128 为基本块寻址（`S1/128`、`S2/128`），且按两行块成对分配核
  （`ky=2`），要求 `S` 为 256 的倍数、`D == 128`；其他形状不会报错，但结果
  错误，`S < 128` 时还会因 AIC/AIV 握手指令对不齐而死锁（原算子即如此）。
- Kernel 仅实例化 FP16 计算路径：算子声明虽含 BF16，但 Host 为 BF16 设置的
  tilingKey=1 没有对应 Kernel 入口，BF16 调用会在 Kernel 启动阶段报 funcEntry
  错误（原算子即如此，本次重构保持原行为，暂不提供 BF16 支持）。

## torch_native 后端

- 仅支持稠密路径：`input_layout="BNSD"`、不传 mask、`keep_prob=1.0`；
  其他情况会显式报错，且不会使用 `pre_tokens` 和 `next_tokens`。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import la

batch, heads, seq, dim = 1, 2, 256, 128
query = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
key = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
value = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)

softmax_log_max_sum, out = la(
    query, key, value,
    scale_value=dim**-0.5,
    head_num=heads,
)
print(out.shape)  # [1, 2, 256, 128]
```
