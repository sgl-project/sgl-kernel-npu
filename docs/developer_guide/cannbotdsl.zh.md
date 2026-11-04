# DeepSeek-V4.1 DSL 算子迁入

本次迁入七个入口：`indexer_prologue_k`，以及
`mixed_quant_sparse_flash_mla`、`quant_lightning_indexer`、
`quant_sparse_lightning_indexer` 各自的计算和 metadata 函数。
源码固定到 `cann/cannbot-dsl@71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`，
原始路径、哈希与许可证随代码保存在 `implementation/cannbot-dsl/`。
七份实现保持上游字节内容，公共接口另外增加 `impl="auto"`。

**目标组合：cannbot-dsl 0.8.0 + CANN 9.2.0，仅 A5 / ARCH 3510。**
当前没有 A5 设备验证；编译、精度、图重放与性能均待实测。
上游 master 标注 CANN 9.2.1，本次不会自动升级目标环境，也不能将本地
接口测试通过解释成 9.2.0 兼容认证。

## 接口与职责

五个 indexer 入口从 `sgl_kernel_npu.operators.indexer` 导入，两个 MLA
入口从 `sgl_kernel_npu.operators.attention` 导入。用它们替换 SGLang
`_ds41_ops` 中对应的七个 `ops.*` 导入即可；compressor 等其他算子仍依赖原有包。

- `indexer_prologue_k` 原地写 K cache；mode 0 分开存 K/scale，mode 1 合并记录。
- QLI 返回索引、值、候选块索引与候选长度；QSLI 消费候选块后返回索引和值。
- MLA 保留显式 `out`、`lse` 参数；调用方负责分配输出。
- 三个 metadata 是 AICPU 调度算子，必须与对应计算实现成对更新。
- `auto` 与 `cannbot_dsl` 是同一实现，均限定 A5；独立精度参考保留在上游迁入的测试中。

## 构建与运行

普通 wheel 像现有 Triton 路径一样包含 Python 源码。目标 Linux 环境安装
`cannbot-dsl==0.8.0` 后可走上游 JIT；首次编译应在图捕获前完成。
DSL 0.8.0 需要 Python 3.10–3.12 及匹配的 torch/torch_npu。

需要预编译交付时，在仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install 'cannbot-dsl==0.8.0'
python3 setup.py --arch-version 3510 --without-third-party \
  --with-cannbot-dsl --plan --cann-version 9.2.0
python3 setup.py --arch-version 3510 --without-third-party --with-cannbot-dsl -j8
```

`--without-third-party` 用于单独验证本次迁入；完整部署有其他第三方算子需求时，
另行选取现有 profile。此入口只构建新构建系统已迁入的原生算子，不代表完整旧版算子库。
`--with-cannbot-dsl` 只适用于新 wheel 构建。实际编译读取安装的 SDK 版本，
`--cann-version` 只供 `--plan` 使用。

构建会生成 AICore Native 资源和三个 AICPU `.so`，一起装入 wheel；每个 metadata
使用单独编译目录。打包前核对资源和源码哈希，wheel 记录编译版本、profile 哈希，
并声明精确 DSL 依赖。源码运行、Native 资源命中、设备数值正确是不同验收项。

默认使用上游完整静态导出配置，indexer 编译量较大。需要限制配置时，通过
`CANNBOTDSL_DS41_PROFILES` 指定 JSON，字段以固定版本的 `export_*` 为准。
`-j` 不控制 DSL 编译器并发；配置裁剪后必须确认实际模型参数覆盖。

## 验证与交接

可通过 [A5 单卡 CI 环境](cannbotdsl-a5-ci.md)接入 Ascend/sglang 的测试 runner；
其中记录了已核实的 CANN 9.2.0 镜像和 PR 触发条件。

```bash
# CPU 离线接口/资源测试，不执行 NPU kernel。
python3 -m pytest tests/unit/test_cannbotdsl_packaging.py tests/unit/test_cannbotdsl_runtime.py -q

# A5 上安装本次构建的 wheel，再执行独立精度参考与 metadata 图重放。
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
export CANNBOTDSL_RUN_SMLA_TESTS=1
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl -q

# 对预编译交付检查真实 Native 命中，避免 JIT 掩盖缺少二进制。
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT="$PWD/native-hits.json"
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl -q
```

上游边界测试可能需要额外导出 profile；Native 未命中先按静态覆盖问题定位。
require 模式下本封装也检查三个 AICPU 资源完整性，避免 metadata 临时编译掩盖遗漏。
测试保留上游 CPU golden，覆盖量化边界、TopK、候选块、缓存跨度和页尾。
`test_metadata_graph.py` 检查三类调度算子在长度变化后的重放结果。
完整计算图、P/D 缓存迁移、模型精度和性能仍需拿到 A5 后验证。

完整英文说明及构建限制见 [cannbotdsl.md](cannbotdsl.md)。
