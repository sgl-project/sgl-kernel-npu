# causal_conv1d

## 签名

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d

causal_conv1d(
    x, weight, conv_states,
    bias=None, query_start_loc=None, cache_indices=None,
    has_initial_state=None, num_accepted_tokens=None,
    activation_mode=0, pad_slot_id=-1, run_mode=0,
    *, impl="auto",
) -> Tensor
```

`auto` 明确选择 `ascendc`，失败时不自动切换后端；`torch_native` 是独立的
PyTorch 实现。新 Ascend C 绑定为 `torch.ops.npu.sgl_causal_conv1d`，旧绑定
`torch.ops.npu.causal_conv1d` 与旧源码保持不变。

## 硬件声明与验证状态

| 实现 | arch22（2201） | arch35（3510） |
| --- | --- | --- |
| `ascendc` / `auto` | 声明 BF16 / FP16 | 声明 BF16 / FP16 |
| `torch_native` | 已注册 | 已注册 |

此表是源码与注册声明，不表示已完成设备验证。真实 CANN 接口生成、编译、出包和
NPU 执行需要分别验收。PyTorch 实现能够计算 CPU Tensor，但公开包和注册表的导入
仍要求其 NPU 环境。

## 参数与返回值

`B` 为序列数，`L` 为共同序列长度，`T` 为拼接后的 token 数，`D` 为独立通道数，
`K` 为卷积宽度，`S` 为每个缓存槽的状态容量。本接口采用 **token 在前的布局**。
旧 Python Mamba 包装的 `[B,D,L]` 输入、`[D,K]` 权重属于不同接口。

| 参数 | shape | 约定 |
| --- | --- | --- |
| `x` | `[T,D]` 或 `[B,L,D]` | 连续 BF16 或 FP16。 |
| `weight` | `[K,D]` | 连续，dtype 与 `x` 一致；`K` 为 2、3 或 4。 |
| `conv_states` | `[slots,S,D]` | 连续，dtype 与 `x` 一致；**原地修改**。 |
| `bias` | `[D]` 或 `None` | 可选通道偏置；有效输入使用与 `x` 相同 dtype。 |
| `query_start_loc` | `[B+1]` 或 `None` | 拼接序列的边界。2D 输入始终要求它，包括 update；3D 输入忽略。转换为 int64。 |
| `cache_indices` | `[B]` 或 `None` | 序列到状态槽的映射，转换为 int64；省略时序列 `b` 使用槽 `b`。 |
| `has_initial_state` | `[B]` 或 `None` | prefill 对转换为 int64 后非零的序列读取历史；省略时不读历史。update 忽略该标志。 |
| `num_accepted_tokens` | `[B]` 或 `None` | 转换为 int32；仅在 `run_mode == 1` 且 `K == 4` 时控制推测更新的历史起点。 |
| `activation_mode` | 整数，默认 `0` | 任意非零值应用 SiLU。 |
| `pad_slot_id` | 整数，默认 `-1` | 显式 cache index 等于此值时跳过该序列。 |
| `run_mode` | 整数，默认 `0` | 恰好为 `1` 时采用集成 update；其他值都采用 prefill。 |
| `impl` | 仅关键字字符串 | `auto`、`ascendc`、`torch_native`。 |
| 返回值 `y` | shape/dtype 与 `x` 相同 | 新分配输出；state 原地写回，不作为第二个 Tensor 返回。 |

空的可选 Tensor 按缺省处理，但空或缺省的 `query_start_loc` 不满足 2D 输入要求。
`ascendc` 输入 Tensor 应位于同一个 NPU。边界和索引必须有效，使用的状态槽至少
容纳 `K-1` 行历史。旧 Host 没有完整检查元数据值和状态容量，本次重构不重新定义
非法输入。多个序列同时写入同一状态槽的执行顺序没有既定契约，不属于独立参考的
精度保证范围。

旧 Kernel 对可选缓冲按 `data_ptr` 指向的物理顺序读取，在元数据类型转换后忽略
stride。新 adapter 与参考保留该行为，不把非连续可选 Tensor 归一化为逻辑元素
顺序。调用方应提供容量充足的连续缓冲；任意 stride view 不是已验证契约。

pad 序列不修改 state，其 `y` 行保持**未初始化**。不能把这些行当作零，也不应纳入
数值比较。省略 cache indices 时的默认槽映射不检查 `pad_slot_id`。

## 计算与状态

对长度为 `L` 的序列 `X`，将 `K-1` 行历史 `H` 与 `X` 按时间顺序拼接为 `Z`。
最旧的历史对应第零行权重：

```text
u[t,d] = sum(Z[t+j,d] * weight[j,d], j=0..K-1) + bias[d]
y[t,d] = 转为 x 的 dtype(u[t,d])                         activation_mode == 0
         转为 x 的 dtype(u[t,d] / (1 + exp(-u[t,d])))     其他情况
```

没有 bias 时省略偏置项。独立 PyTorch 计算将原始历史、token 和权重转为 FP32，
每行权重沿 token 维广播相乘，按 tap 递增累加，再处理 bias、SiLU 和最终类型转换。
这给出数学参考；硬件乘加和 SiLU 的舍入可能不同，不承诺逐位一致。

| 模式 | 读取历史 | 写回 state |
| --- | --- | --- |
| prefill（`run_mode != 1`） | 初始状态标志非零时读 `state[slot,:K-1]`，否则使用零。 | 将 `Z` 最后 `K-1` 行原始值写入 state 前 `K-1` 行。 |
| 普通 update（`run_mode == 1`） | 无论初始状态标志如何，始终读 `state[slot,:K-1]`。 | 同普通写回。 |
| update、accepted 非空、`K == 4` | `o = min(max(accepted[b]-1,0),S-3)`，读旧 `state[slot,o:o+3]`。 | 若 `S >= 2+L`，将 `[旧state[o+1],旧state[o+2],X]` 写入前 `2+L` 行；否则退回普通 `K-1` 行写回。 |

偏移公式以有效容量 `S >= 3` 为前提。accepted 数量不会截断本次需要计算的 token。
`K == 2/3` 即使传入 accepted 也采用普通 update。未写入的状态行和未使用槽保持
原值；state 保存原始输入/历史，不保存卷积或激活后的输出。独立参考先保存旧状态
快照，再执行写入，避免覆盖计算所需的历史。

这些 update 分支属于集成 `causal_conv1d`，不调用或依赖独立的
`csrc/causal_conv1d_update` 算子；后者不在本轮迁移范围。

## 精度覆盖

正式测试保留旧脚本全部九个 prefill 配置、CPU 随机输入生成顺序和 seed `20260326`。
每个实现获得独立状态存储。有效 `y` 行通过公共 `assert_accuracy` 使用
`DEFAULT_STANDARDS`：BF16 的 `atol/rtol` 均为 `5e-2`，FP16 均为 `1e-3`。
**整个** state 池精确比较，包括未使用槽和未覆盖行；同时检查只读输入没有改变。

九个普通 update 用例原样复用上述输入，仅将 `run_mode` 改为 1。另外复用旧 update
测试中已确认的两组配置：FP16 的 `(B,L,D,K,S)=(1,4,1024,4,6)`，带 bias 和 SiLU；
BF16 的 `(32,4,12288,4,6)`，无 bias、带 SiLU。accepted 全部为 4，cache 槽为
`0..B-1`。FP16 state 前部为零、末尾三行为随机历史；BF16 state 全部随机。这两组
输入在 CPU 上用 seed 42 生成。全部十一个用例经本算子的 `run_mode=1` 执行，
不调用独立 update 算子或旧 Triton 包装。

同一测试文件另提供互斥的 `--capture-legacy PATH` 与 `--compare-legacy PATH`
命令行模式。前者在旧包环境保存原始输入及旧输出/state；后者在新包环境加载完全
相同的输入，分别比较旧/新、新/参考、旧/参考。不要求同一进程加载两个包或同时
注册两套扩展。回放文件缺失或格式不兼容会报错。默认 pytest 的新/参考比较通过，
不能替代新旧直接对照。

录制时增加 `--include-update`，可用新版回放格式保存全部 20 个 prefill/update
用例。比较时自动识别文件中的用例集，仍兼容原九个 prefill 用例的回放文件。

## 性能测试

独立的 `perf_shape` 列表选择全部 20 个配置。先检查精度，再通过公共性能框架
仅测量 `impl`，预热和正式采样均为 5 次。每次调用使用独立的原始 state 副本；
全部副本在计时前的首次调用中准备，复制任务不进入计时区间，各次调用不会累积
历史状态。性能测试不缩小输入或调整 dtype。

在仓库根目录、已安装新版 wheel 的 Python 环境中运行：

```bash
SGL_TEST_ROOT="$(pwd -P)"
python3 -m pytest "$SGL_TEST_ROOT/tests/python/operators/mamba/test_causal_conv1d.py" \
  --impl ascendc --impl-ref torch_native --skip-acc-check=false \
  --enable-perf=true --csv-path "$SGL_TEST_ROOT/test-results/causal_conv1d.csv" \
  --prof-dir "$SGL_TEST_ROOT/test-results/profilings" -sv -ra
```

`--enable-perf=false` 仅验精度；`--skip-acc-check=true` 不调用参考实现，性能结果
标记为 `accuracy_checked=False`。CSV 记录平均 NPU 设备任务耗时，不是主机端到端
延迟。新增 update 的设备精度、新旧设备回放、实际性能采样和同设备性能回归对照
仍需上机完成，接入这些测试路径不表示硬件结果已经通过。
