# 公共框架 UT（NPU 环境）

本目录验证公共算子框架及公共精度测试工具：注册、查询、分发、导入隔离、代际管理、
精度标准、输出比较和 `check_accuracy` 执行流程。
当前基线：`Todobe:pr/tests-refactor`，提交 `5476535720de9fce53c625b7d5584d57d87c3bc7`。
算子框架原有 70 条 UT 从 `codex/framework-ut-npu@ca965bb` 保留，新增精度工具 106 条。
实现目录使用基线中的 `implementation/backend/module`。
使用基线自带的公共精度工具和上层 conftest；operators 目录的自动上下文不作用于本目录，
因此精度流程 UT 显式设置并恢复上下文。
本目录不包含生产 kernel 的数值精度、性能、具体 backend 的 Tensor 转换或真实 AscendC 接入验收。

## 环境和运行

使用 Python 3.10+、pytest、匹配的 PyTorch/torch_npu/CANN 和可用的 Ascend NPU。
环境中需要可正常导入的 `sgl_kernel_npu`（含兼容的原生 `.so`），然后在仓库根目录执行。
Python 框架统一从当前 checkout 加载；已安装包可以尚未包含新 operators，
不要求其 Python 文件与本分支逐字节相同，也无需仅为此次 Python UT 更新而重编原生库：

```bash
python -m pytest tests/python/framework -v -rs -x --npu-device=0
```

只运行本次新增的精度工具 UT：

```bash
python -m pytest tests/python/framework/test_standards.py \
  tests/python/framework/test_accuracy.py \
  tests/python/framework/test_check_accuracy.py -v -rs -x --npu-device=0
```

这些 UT 自行构造配置场景，不用 `--impl`、`--impl-ref` 或 `--skip-acc-check` 改写用例。
输入 Tensor 显式放在当前 NPU；设备不一致负例仅额外创建一个 CPU Tensor，不要求第二张卡。

`--npu-device` 是可见设备的逻辑索引；省略时使用当前设备。请单进程串行运行，
不要设置多个 xdist worker。首次报错停下用 `-x`；收集完整失败信息时去掉 `-x`。

保存机器可读报告：

```bash
mkdir -p results/framework-ut
python -m pytest tests/python/framework -v -rs --npu-device=0 \
  --junitxml=results/framework-ut/junit.xml
```

仅查看用例清单（不执行 fixture、硬件识别或测试断言）：

```bash
python -m pytest tests/python/framework --collect-only -q
```

这里的环境缺失、包导入失败、`.so` 加载失败、未知 SoC 都应报告为错误，不以整批 skip
掩盖环境问题。框架测试使用当前 checkout 的源码，不从 site-packages 导入 operators，
也不改写已安装包的搜索路径。

## 用例组成

| 文件 | 参数化后用例数 | 覆盖内容 |
| --- | ---: | --- |
| test_registry.py | 20 | 公共算子/实现注册、重复与冲突、精确查询、别名、枚举及状态保护 |
| test_dispatch.py | 15 | 真实 RMSNorm 公共入口、默认/显式 impl、参数透传、错误不回退、新实现动态接入、双向切换时逐次检查缓存路由、参数和返回值 |
| test_generation.py | 10 | 真实硬件识别、A2/A3/A5 型号映射、未知型号、成功缓存和失败重试 |
| test_generation_registry.py | 18 | 支持/排除集合、空集合、类型校验、当前代际分类、别名范围一致性和注册原子性 |
| test_imports.py | 7 | checkout 公共导出、模块导入注册、不提前导入 kernel、检测失败、注册隔离、缺依赖与恢复、旧安装包兼容 |
| test_standards.py | 20 | 六种默认标准、局部覆盖与回退、非法容差/配置、未知 dtype |
| test_accuracy.py | 49 | 真实 Tensor 的绝对/相对容差边界、整数/bool、元信息、嵌套结构和路径、Python 标量、NaN、缺标准及未知输出类型 |
| test_check_accuracy.py | 37 | ref→DUT 顺序、参数绑定、executor、真实精度失败、标准传递、配置优先级、skip、别名拒绝、实现缺失/代际异常、异常传播和上下文恢复 |
| 合计 | 176 | 70 个测试函数；数量来自 pytest collection，不表示已在 NPU 执行通过 |

不同 backend 的路由、参数方式和异常分别测试，没有把所有维度做笛卡尔乘积。
`test_backend` 是人为注册的替身，用于验证扩展机制，不是一个真实算子实现。

## fixture 与 mock 边界

1. session fixture 真实检查 NPU，选择设备，正常导入安装包及其原生库，读取实际设备名称。
   代际映射和注册判断在当前 checkout 的 generation/registry 中执行。
2. 每条 UT 在私有命名空间加载当前 checkout 的真实 framework 源码，隔离注册表和入口缓存。
   这一步只避免测试注册污染正常包；不提供无 NPU 运行模式。
3. 一般注册/分发用例使用实际硬件识别；代际规则用例在导入私有 registry 前替换
   `get_hardware()` 返回值。因此只需要一台受支持的 NPU 机器，就能验证 A2/A3/A5 的规则。
   这不证明 kernel 在其他代际硬件上能运行，也不模拟生产进程运行中切换代际。
4. 硬件识别单元测试保留真实 `get_hardware()`，仅在指定用例中替换
   `torch.npu.get_device_name()`，验证成功缓存、失败原因和重试；结束后恢复。
5. 分发测试保持真实公共函数和薄适配函数，只替换 implementation 叶子模块的 kernel。
   使用不同 sentinel 参数和返回对象检查目标、调用次数、参数和结果，不启动 Triton 编译或 NPU kernel。
6. 临时模块、导入钩子和硬件替身在测试结束时清理，session 结束恢复显式切换前的设备。
7. `accuracy_tools` 正常导入 checkout 的真实精度工具。`accuracy_case` 将 common 内绝对导入的
   canonical registry 名称临时绑定到本条用例的同一份真实隔离 registry，结束后由 monkeypatch
   恢复；不替换注册查询函数。注册两个独立轻量 Python backend，用小 NPU Tensor 算术和列表
   记录调用，保留真实 `check_accuracy`、`assert_accuracy` 和 `torch.testing.assert_close`。
8. 代际不支持用 `supported_generations=[]` 构造，无需模拟另一代芯片；预期 skip 在 UT 中捕获并
   核对原因，因此用例通过意味着断言执行过，不是 UT 自身直接被跳过。
9. 比较层临时移除 FP32 默认标准的场景由 monkeypatch 自动恢复；配置上下文由 `use_impls` 恢复，
   新算子注册随私有模块删除，不污染安装包的真实注册。

本批不修改公共工具或算子生产实现。容差边界使用的 0.125 等合成阈值仅验证比较逻辑，
不是新的生产精度标准。性能、profiler、CSV、CLI parser、设备恢复 UT 留待后续批次。
真正的 kernel 精度与性能仍通过 operators 测试目录验证。
