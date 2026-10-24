# DSpark Markov sample：A3 精度修复与预生成噪声

2026-10-24。本次修复位于独立分支 `fix/npu-dspark-sampling-precomputed`，基于新增五算子的 `77250dc`。尚未集成 SGLang `origin/main`。

## 改动与精度契约

1. 修复原 Philox 路径中 CANN `libdevice.log1p` 对极小 uniform 的消差问题，保留 Philox4x32-10、key、row、step、offset、vocab_start 的映射。
2. 增加可选的 FP32 `exp_noise` 输入。调用方可用 NPU 原生 `exponential_()` 在选择 kernel 之前生成 Exp(1) 噪声，kernel 跳过 Triton Philox 和 log1p，只计算 `value / temperature - log(exp_noise)` 并选 token。
3. 预生成路径使用至多4096词的tile；词表很小时缩到相应2次幂。原Philox路径仍使用1024，控制随机数计算的UB工作集。
4. 将原测试的“随机token至少95%一致”收紧为逐token精确一致，增加数值、采样、分片和图回归；新增可独立复现的benchmark。

预生成Exp(1)与 `-log1p(-Uniform)` 的分布相同，但原生NPU RNG不承诺与原Philox具有相同随机序列。因此它是显式可选输入；不传时保持原Philox计数器语义。对预生成路径的精度比较使用同一份噪声，要求token和保留logits精确一致，不用“随机性”解释差异。

本次不更改其他四个新增算子的数值契约。特别是Softmax原有容差检查不等于与Torch逐位一致，Markov greedy仍遵循分支已有的Torch原生投影舍入契约。

## log1p 修复

原失败输入的同一行，参考选择46701，旧实现选择128206。Philox随机整数和uniform完全相同，差异发生在噪声变换：

| 候选token | 128206 | 46701 |
|---|---:|---:|
| Philox整数 | 883 | 103978 |
| uniform | 2.0570587367e-7 | 2.4209381081e-5 |
| CPU参考噪声 | 2.0570588788e-7 | 2.4209673938e-5 |
| 旧CANN libdevice噪声 | 1.7881394854e-7 | 2.4199778636e-5 |
| 修复后噪声 | 2.0570588788e-7 | 2.4209673938e-5 |

修复显式补偿 `1-u` 的FP32舍入：

```text
s = float32(1-u)
delta = (1-s)-u
noise = -log(s)-delta/s
```

对于小u，不再丢掉 `1-u` 中被舍入的部分；u=0和u=1分别显式返回0和正无穷。无额外CPU同步。误差性质与数值修复效果来自本机输出对比，而不是假设所有CANN版本具有相同问题。

2097157个uniform通过检查：包含最低1048576个Philox word映射值、1048576个固定种子的随机word、失败word以及0/1端点。相对FP64 `-torch.log1p(-u)` 舍入到FP32的参考，最大误差1 ULP，无NaN，Inf位置一致。有限值中1759732个为0 ULP、337423个为1 ULP。这里不宣称所有内部浮点运算逐位等于CPU库；实际API输出token与kept logits在回归中要求精确一致。

已有完整失败输入重新加载后修复通过；另外添加了不依赖外部文件的两候选回归：固定key=12345、row=36、step=2、offset=7，旧实现失败、新实现通过。

## NPU预生成路径与性能

设备：`192.168.25.216`，容器`lph-101500`，空闲逻辑NPU15，`Ascend910_9362`，运行时20 Cube/40 Vector。Torch 2.10.0+cpu、torch-npu 2.10.0.post6、Triton-Ascend 3.2.0，原容器CANN路径 `/usr/local/Ascend/cann-9.1.0`。没有A5或真实CUDA GPU实测。

原Philox主kernel在8行输入约2491μs，Scalar ratio约95.4%。分段测试确认仅Philox随机位生成约2360μs，而int64→FP32转换、log1p、log分别仅约15–20μs。先生成原生Exp(1)噪声能绕过这个Scalar瓶颈。

数据为BF16 base/bias/step logits，FP32噪声、temperature=0.7、词表129280，保留logits。以下为20次预热、每次30调用、3次采样的中位数。编译和一次性输入/噪声buffer分配不计时；正常API内部临时分配保留。顺序交替测量。

图重放摊销时间（μs）：

| batch | 旧Philox | 预生成后选择 | 原生生成 | 生成+选择整体 | 整体加速 |
|---:|---:|---:|---:|---:|---:|
| 1 | 403.02 | 20.37 | 107.44 | 109.09 | 3.69× |
| 8 | 2510.84 | 43.79 | 177.32 | 213.51 | 11.76× |
| 64 | 19537.13 | 182.87 | 948.33 | 1126.51 | 17.34× |

各阶段单独测量含各自提交成本，因此总耗时不要求等于两个单项之和。图只用于测量设备路径，不在产品API内建图；图重放时间也不等于纯kernel耗时之和。

独立eager调用时间（μs）：

| batch | 旧Philox | 预生成后选择 | 生成+选择整体 | 整体加速 |
|---:|---:|---:|---:|---:|
| 1 | 403.37 | 196.33 | 321.88 | 1.25× |
| 8 | 2508.76 | 197.18 | 314.18 | 7.98× |
| 64 | 19539.57 | 199.04 | 1113.93 | 17.54× |

小batch的Python/JIT分发与RNG启动开销仍明显；不将64行的加速比外推到所有batch。只修精度而保留内嵌Philox时，64行仍约19.8ms，性能收益来自显式预生成路径，旧调用点不会自动变快。

## Tile与profiler证据

固定相同噪声扫描1024/2048/4096/8192：64行选择时间约390/241/181μs；8192编译失败，需要208KiB UB，而该配置可用192KiB。因此选择4096，不照搬GPU或假设越大越好。最终独立benchmark再次测得约183μs。

最终8行的20次纯kernel中位数：旧Philox主kernel约2491μs，补偿精度后的Philox主kernel约2528μs，预生成路径主kernel约28μs。Scalar ratio从约95.4%降到约32.5%，Vector ratio从约5.8%升到约61.2%。流水线比例可重叠，不相加、不称作FLOPs利用率。

## 正确性验证

五个算子合并回归41项全部通过，无跳过。Markov sample包括：

- FP32/FP16/BF16、混合greedy/sample行、非对齐词表和行stride，token与Torch CPU参考精确一致，kept logits精确一致。
- 原Philox路径的小uniform反例、key/row/step/offset变化、NaN/Inf、并列取最小索引、padding和空shard。
- 预生成噪声的词表分片、candidate gather、kept logits gather。这里是单进程模拟collective，尚未进行真实多NPU/HCCL验证。
- 图重放读取更新后的噪声；原生RNG在图重放中确实更新，输出与同一份噪声的参考精确一致。
- 两种采样路径在temperature=0.5/1/2时的频率符合目标softmax（沿用统计测试的0.01阈值；不能用它替代逐token检查）。
- 拒绝噪声shape、device、dtype、列stride不符合接口的输入。

修改文件的pre-commit检查通过。原先一个新图测试在capture内创建offset CPU→NPU拷贝，属于测试准备错误；已移到capture外，最终回归通过，没有用放宽捕获模式绕过问题。

## 使用及后续集成约束

噪声buffer可在循环外分配，每个Markov step刷新：

```python
noise = torch.empty(base_logits.shape, dtype=torch.float32, device=base_logits.device)
noise.exponential_()
tokens = markov_sample_step(**sample_kwargs, exp_noise=noise)
```

- noise应为正的FP32 Exp(1)样本、列stride=1，shape与base_logits一致；gather完整kept logits时与logits_out一致。
- 外部模式由调用方管理RNG，key/offset/step不生成它。需要与完整词表采样一致时，分片必须使用同一份全局噪声对应的切片，不能各rank用相同seed生成相同的局部序列。
- 每个step应使用新噪声；可提前生成多个step的buffer，但需要计入分配和显存预算。64×129280的单步FP32噪声约31.56MiB，8步约252.5MiB。
- 默认模式继续用原Philox序列；若将来要求与GPU的Philox随机位序完全一致，不能直接用原生exponential_替换，需要外部生成兼容序列后提供同一噪声。
- 目前只完成单算子修复与测试，尚未进行SGLang整网、实际TP/DP、接受率或数据集验证。origin/main集成应在这些验证之后进行。

## 复现与证据

源码和单测为Python改动。可在容器隔离目录用当前源文件替换模块做验证，不覆盖已有wheel。此次目录为 `/tmp/dspark_sampling_fix_20261024`；原始失败输入位于前次审计目录。

```bash
export ASCEND_RT_VISIBLE_DEVICES=15 OMP_NUM_THREADS=1
cd /tmp/dspark_sampling_fix_20261024
python run.py tests-all
python run.py regression
python run.py transform-stats
python benchmark.py --source candidate.py --baseline-source baseline.py --batch 1 8 64
python run.py profile
```

独立benchmark的仓库位置：`tests/python/sgl_kernel_npu/bench_markov_sample.py`，支持`--source`直接验证Python候选源码，`--baseline-source`对照旧源码。正式安装对应源码后，可直接运行 `python tests/python/sgl_kernel_npu/test_markov_sample.py`（或按仓库测试导入方式执行）。复测前重新确认设备空闲。

机器可读结果见[dspark_markov_sampling_npu_results.json](dspark_markov_sampling_npu_results.json)。完整原始JSON、41项测试日志和kernel_details.csv保存在隔离目录及本地审计副本。使用vector-triton-ascend-ops-optimizer与ascendc-precision-debug技能进行优化和回归；不将经验值当成硬件实测。
