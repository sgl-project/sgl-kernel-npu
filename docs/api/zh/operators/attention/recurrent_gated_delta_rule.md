# recurrent_gated_delta_rule

## 芯片与实现

| 实现 | arch22（2201） | arch35（3510） |
| --- | --- | --- |
| `ascendc` / `auto` | 已声明，待 NPU 验证 | 已声明，待 NPU 验证 |
| `torch_native` | 独立 PyTorch 参考实现 | 独立 PyTorch 参考实现 |

原生实现保留旧 BF16 kernel 及其内部 arch35 分支。CPU 参考检查不代表 NPU
验证通过。原生实现不可用时报错，不自动切换后端。

## 函数原型

```python
from sgl_kernel_npu.operators.attention import recurrent_gated_delta_rule

def recurrent_gated_delta_rule(
    mix_qkv, recurrent_state, beta, scale, actual_seq_lengths,
    ssm_state_indices, nk, nv, intermediate_state=None,
    cache_indices=None, num_accepted_tokens=None, g=None, gk=None,
    *, impl="auto",
):
    ...
```

旧接口名 `torch.ops.npu.recurrent_gated_delta_rule` 保留。
新公共接口的 `auto` / `ascendc` 通过该名称调用当前安装包的实现；
`torch_native` 使用独立递推公式。

## 参数与返回值

| 参数 | Shape / Dtype | 说明 |
| --- | --- | --- |
| `mix_qkv` | `[B,S,2*Nk*Dk+Nv*Dv]`，BF16 | 按 Q、K、V 顺序拼接。 |
| `recurrent_state` | `[N,Nv,Dv,Dk]`，BF16 | 普通模式下读写的状态池；提供中间池时作为 recurrent 种子池。 |
| `beta` | `[B,S,Nv]`，BF16 | sigmoid 之前的值。 |
| `scale` | 必填 float | Q 的缩放系数，保留 `0.0`，沿用旧 Host 的 FP32 窄化。 |
| `actual_seq_lengths` | `[B]`，int32 | 每条序列的长度，不是累积边界；序列起点固定为 `b*S`。 |
| `ssm_state_indices` | `[B,S]`，int32 | 初始状态选择及各 token 的回写槽位。 |
| `nk`、`nv` | 正整数 | Q/K 与 value 头数，要求 `Nv % Nk == 0`。 |
| `intermediate_state` | 完整的 `[Nv,Dv,Dk]` 槽位，BF16 / `None` | 可选工作池，通常为 `[N*S,Nv,Dv,Dk]`。 |
| `cache_indices` | 至少 `B` 个元素 / `None` | 每条序列的 recurrent 种子槽；提供中间池时必填；转为连续 int32。 |
| `num_accepted_tokens` | 至少 `B` 个元素 / `None` | 选择初始列 `accepted[b]-1`，转为连续 int32；不缩短本次计算长度。 |
| `g` | 至少 `B*S*Nv` 个元素 / `None` | log decay，转为连续 FP32 后取 exp；缺省时衰减系数为 1。 |
| `gk` | 可选 Tensor | 为兼容旧接口转为连续 FP32；旧计算逻辑不使用其数值。 |
| `impl` | `auto`、`ascendc`、`torch_native` | 仅限关键字传入。 |

只返回 BF16 `out`，形状为 `[B,S,Nv,Dv]`。状态通过调用方传入的池回写，不增加
返回值。每条序列有效长度以后的输出未定义。维度必须为正，`1 <= S <= 8`，有效长度
在 `[1,S]` 内。`ascendc` 要求所有张量位于同一 NPU；`torch_native` 要求同一设备。
槽位必须在对应池的范围内；不同序列不能发生冲突的状态读写。原生校验只检查元数据，
不把设备张量中的索引、长度值同步回 Host 做检查。

## 递推与状态协议

value 头 `h` 使用 Q/K 头 `h // (Nv // Nk)`：

```text
q = scale * q_raw / sqrt(sum(q_raw*q_raw) + 1e-6)
k = k_raw / sqrt(sum(k_raw*k_raw) + 1e-6)
D_t = exp(g_t) * H_(t-1)                 # g=None 时 D_t = H_(t-1)
delta_t = sigmoid(beta_t) * (v_t - D_t @ k_t)
H_t = D_t + outer(delta_t, k_t)
out_t = H_t @ q_t
```

状态矩阵布局为 `[Dv,Dk]`。连续 token 之间保留 FP32 状态；每个 token 写出 BF16
快照，下一步不重新读入舍入后的快照。

1. 没有中间池时，读取并回写 `recurrent_state`。
2. 有中间池时，读取并回写中间工作池；两个池不重叠时，原 recurrent 池只提供种子，
   保持不变。
3. 有 accepted 计数时初始槽为 `indices[b,accepted[b]-1]`，否则为 `indices[b,0]`。
   accepted 计数必须在 `[1,S]` 内。
4. 中间池模式下，所选槽能被 `S` 整除时，从 `recurrent_state[cache_indices[b]]`
   初始化；否则从所选中间槽初始化。第 `t` 个有效 token 写入工作池的 `indices[b,t]`。

必填的 QKV、beta、lengths、indices、recurrent_state 保留旧 `data_ptr()` 语义：
从 storage offset 开始按物理连续区间读取，不按 Tensor 的逻辑 stride 重排，
也不新增 contiguous 拷贝；底层 storage 必须覆盖该区间。可选索引、gate 按旧行为
转换 dtype 并连续化。连续中间池直接回写；非连续中间池使用临时副本，**不拷回原池**，
与旧 Host 一致。需要观察状态更新时应传连续状态池。

## 验证范围

新 pytest 保留以下八个旧输入族，均为 BF16：

| 输入族 | `(B,S,Nk,Nv,Dk,Dv)` | Accepted 设置 | 默认测性能 |
| --- | --- | --- | --- |
| compatible | `(2,1,8,16,128,128)` | 原随机计数 | 否 |
| compatible | `(64,2,8,16,128,128)` | 原随机计数 | 否 |
| compatible | `(32,8,8,16,128,128)` | 原随机计数 | 否 |
| compatible | `(32,8,16,32,128,128)` | 原随机计数 | 否 |
| benchmark | `(2,1,8,16,128,128)` | `None`，seed 42 | 是 |
| benchmark | `(4,4,8,16,128,128)` | `None`，seed 42 | 是 |
| benchmark | `(8,8,8,16,128,128)` | `None`，seed 42 | 是 |
| benchmark | `(2,4,8,16,64,32)` | `None`，seed 42 | 是 |

检查有效输出、整个状态池、未写入槽位、只读种子和其他只读输入。使用公共精度标准，
同时保留来源更严格的阈值：compatible 为 `atol=rtol=2**-8`；benchmark 为
`atol=0.01, rtol=0`。非方形配置保持原样，需在目标架构上建立旧包、新包对照结果。

先加载服务器平时使用的 CANN/PyTorch 环境，在仓库根目录设置绝对根路径和实际架构：

```bash
export SGL_RGDR_ROOT="$(pwd -P)"
export SGL_RGDR_ARCH=YOUR_ARCH  # 替换为 2201 或 3510。
python "$SGL_RGDR_ROOT/setup.py" --arch-version "$SGL_RGDR_ARCH" \
  --ops recurrent_gated_delta_rule --artifact wheel --without-third-party
python -m pip install --force-reinstall --no-deps "$SGL_RGDR_ROOT"/output/sgl_kernel_npu-*.whl
python -m pytest "$SGL_RGDR_ROOT/tests/python/operators/attention/test_recurrent_gated_delta_rule.py" \
  -q --impl=ascendc --impl-ref=torch_native --skip-acc-check=false --enable-perf=false
```

构建、安装和测试使用同一个 Python。setup 会输出准确的 `ARTIFACT` 路径。
测四个原基准配置的性能时，去掉禁用性能的选项：

```bash
python -m pytest "$SGL_RGDR_ROOT/tests/python/operators/attention/test_recurrent_gated_delta_rule.py" \
  -q --impl=ascendc --impl-ref=torch_native --skip-acc-check=false \
  --prof-dir="$SGL_RGDR_ROOT/profilings/rgdr-new" --csv-path="$SGL_RGDR_ROOT/rgdr-new.csv"
```

公共 profiler 统计设备任务耗时：先执行一次不计时调用，再预热 5 次、采样 5 次，
每次使用独立的初始工作池，状态复制安排在 profiling 之前。显式
`--enable-perf=true` 会测全部八个输入族，显著增加内存占用：最大工作池为 520 MiB，
性能入口预先准备 11 份工作副本。

## 旧版、新版验收

旧内核、新内核和 `torch_native` 的同输入三方对照属于独立的本地验收：
保存同一份输入，检查新旧输出及整个状态池逐位一致，再分别与独立参考比较。
上面的正式单算子测试覆盖公开接口；旧版捕获、回放工具作为本地开发材料保留。

旧版、新版的 NPU 回放和同设备性能回归仍待执行。验收时应记录并对齐设备、CANN、
驱动、编译器、PyTorch 和 torch_npu 版本；上面的命令本身不表示这些验收已经通过。
