# 仓库性能测试工具与指标

为 CCE 算子添加精度和性能回归时查阅。统一行为以[测试说明](<../../../../../tests/python/README.md>)及[performance.py](<../../../../../tests/python/utils/performance.py>)为准。不要复制 profiler 实现或增加独立计时基础设施。

## 通常的测试结构

下面是结构示意，`example_op`、shape、参数、dtype 和参考后端需替换为实际算子。生产形状注明真实调用来源，借用模型维度要明确说明。

```python
import pytest
import torch
from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(16, 4096)]

def generate_data(shape, dtype):
    return (torch.randn(shape, dtype=dtype),)

@pytest.mark.parametrize("shape", [
    pytest.param((16, 4096), id="representative-scenario"),
    pytest.param((7, 1025), id="edge-tail"),
])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_example_op(shape, dtype):
    from sgl_kernel_npu.operators.norm import example_op
    args = generate_data(shape, dtype)
    executor = check_accuracy(example_op, args=args)
    check_performance(executor, enable_perf=shape in perf_shape)
```

正常通过 CLI 选择 `--impl=cce --impl-ref=torch_native`，不要默认把同一个函数的 auto 与 cce 比较。参考实现按算子实际可用路径选择。仓库 conftest 配置设备和测试上下文，输入生成沿用它，不再定义选卡 fixture。

- 默认先检查精度，再按 `perf_shape` 测性能。`--enable-perf` 与 `--skip-acc-check` 按仓库 CLI 覆盖规则处理；跳过精度不能记录为已通过。
- 输出路径用 `--prof-dir`、`--csv-path`；具体保留路径见[临时文件管理](../workspace-artifacts.md)。仓库当前不默认忽略 `results/` 或 `profilings/`，不要假定工具会自动清理或忽略它们。串行采集，避免多 worker 争卡或同时写 CSV。
- 补充测试按规范使用 `extra_checks`。SFA 的固定实现对照示例见[对应测试](<../../../../../tests/python/operators/attention/test_sparse_flash_attention.py>)，不将其固定 head/shape 扩散到新算子。
- 原地与状态算子使用独立输入、`assert_accuracy` 和 `make_executor`，处理性能重复执行的状态重置。只在真实比较通过后标记 `accuracy_checked=True`。
- 特殊准确性流程可直接用 `assert_accuracy`；确需自定义性能流程时调用仓库 `benchmark_npu`，说明例外。当前 helper 要求新建、不存在的采集目录，warmup/active 为正整数。

## 指标解释

公共结果 `average_device_task_us` 是 NPU 任务时长之和除以采样次数，包含被测 callable 的全部设备计算阶段；不是端到端延迟，也不是可重叠任务的时间线跨度。golden、数据生成与 H2D 在采样外，native 的完整计算在采样内。

需要额外调优分析时可记录同步调用时延、原始样本和分位数、PipeUtilization。不要改变公共 helper 来满足单个算子的临时分析。

Matmul 有效 FLOPs 为 `2*M*N*K`，TFLOPS 为 `effective_flops / (device_us*1e-6) / 1e12`。MFU 分母必须来自相同精度、核数和目标设备的权威峰值，缺失时只报吞吐。padding 等额外计算不计有效 FLOPs。由理论最少字节计算的是有效带宽，不能标成已测 HBM 流量。MX 量化数据量示例见[MX 参考](mx-quantization.md)。
