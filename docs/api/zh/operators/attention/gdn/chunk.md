# chunk_gated_delta_rule_npu

此接口将旧 `fla.chunk` 的 sequence-first GDN 外层编排纳入公共算子框架：

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

out, final_state, h = chunk_gated_delta_rule_npu(
    q, k, v, g, beta,
    initial_state=state,
    cu_seqlens=cu_seqlens,
    impl="auto",
)
```

`auto` 与 `triton` 选择同一延迟加载实现。实现直接组合迁移后的 cumsum、KKT、
solve_tril、WY、state update 和 output implementation，不在内部重复经过公共分发。
本外层接口没有注册 `torch_native`；需要 Triton-Ascend 和 Ascend NPU。

## 参数与返回

- `q/k` 为 `[B,T,Hg,K]`，`v` 为 `[B,T,H,V]`，`H % Hg == 0`。
- `g/beta` 为 `[B,T,H]`；g 是逐 token 的 log decay，编排内部执行 chunk cumsum。
- q/k/v 为相同的 FP16/BF16 dtype。普通分支 chunk size=64；整链回归采用模型
  head dimension `K=V=128`。现有 state/output kernel 使用 128 宽 key tile 和最多
  两个 64 宽 value tile。
- `initial_state=None` 或 `[N,H,K,V]`；N 为序列数。
- `cu_seqlens=None` 或同设备连续 INT32/INT64 `[N+1]`；packed 要求 B=1，
  边界从 0 到 T、长度为正。不要原地修改已缓存的边界张量。
- `scale=None` 使用 `K**-0.5`；显式 0 不会改成默认值。
- `use_qk_l2norm_in_kernel=False`；开启后使用平方和加 `1e-6` 的 L2 归一化，
  保留输入 dtype。输入 guard 保留连续化与 NPU device context 行为。
- `head_first=False`；True 仍抛出 `DeprecationWarning`。
- 返回 `(out, final_state, h)`：out 的 shape/dtype 与 v 相同；final_state 为
  `[N,H,K,V]` FP32，`output_final_state=False` 时为 None，默认值仍为 True；
  h 是 key dtype 的进入 chunk 前状态 `[B,NT,H,K,V]`。packed 的 NT 为各序列
  chunk 数之和。

保留 `GDN_USE_MEGA_GDN=1` 分支，使用可选的 mega native 算子、chunk size=128
和原有 FP16 中间转换，因此 h 的 chunk 数与普通路径不同。
`GDN_RECOMPUTE_SUPPRESS_LEVEL` 仍在导入时读取；外层返回仍包含 h。

## 既有约束

数值回归覆盖 fixed B=1 和 packed B=1 多序列。原有 state/output kernel 在
fixed B>1 时的 gate batch 偏移有问题；mega 也假设 B=1。本次保留接线范围，
未修复这些既有底层问题，也不宣称 B>1 数值正确。

不要与已有 `gdn.chunk_gated_delta_rule` 混用。后者是独立融合 CANN 接口，使用
query/key/value、actual_seq_lengths、chunk_state 参数及两个返回值，state 方向
也不同。本迁移不改变主仓标准 Ascend 融合 prefill 的路径选择。

## 测试

```bash
python -m pytest tests/python/operators/attention/gdn/test_chunk.py \
  --enable-perf=false -v -rs
```

独立逐 token recurrence 验证 output、final state 与进入 chunk 前的状态，覆盖
FP16/BF16、packed 边界、chunk 尾部、初始状态、非连续输入、归一化及零 scale。
mega case 仅在缺少可选 native 算子时跳过。
