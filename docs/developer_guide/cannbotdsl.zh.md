# DeepSeek-V4.1 CANNBot-DSL 算子

七个 attention/indexer 入口与 Triton 实现一样按 Python 源码随包分发。
`implementation/cannbot-dsl/UPSTREAM.json` 统一记录来源 commit、原路径和
SHA256，同目录保留 CANN 许可证。`auto` 与 `cannbot_dsl` 选择同一个实现。

验证目标为 **A5 / ARCH 3510、CANN 9.2.0、cannbot-dsl 0.8.0**。
上游文档使用 CANN 9.2.1，当前目标组合仍需完成设备编译、精度和图回放验证。

## 构建与使用

在仓库根目录构建普通 kernel wheel：

```bash
python3 setup.py --arch-version 3510 --without-third-party -j8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pip install 'cannbot-dsl==0.8.0'
```

`--without-third-party` 用于单独验证迁入源码。完整部署需选择模型所需的
第三方 profile；上述命令不保证包含 SGLang 所需的所有其他算子。

DSL 函数按需导入和 JIT 编译，图捕获前应预热实际 shape。本次构建不生成
预编译 DSL bundle。wheel 的可选 `cannbot_dsl` extra 可安装 DSL 依赖，
其他算子不强制依赖 DSL。使用匹配的 Linux/Python、torch/torch_npu 环境；
当前 A5 验证环境为 Python 3.12、torch/torch_npu 2.13.0。

K prologue、QLI/QSLI 及其 metadata 从 `sgl_kernel_npu.operators.indexer`
导入；MLA 及 metadata 从 `sgl_kernel_npu.operators.attention` 导入。
接口沿用迁入源码签名，额外提供 `impl="auto"`。MLA 写入调用者的 `out/lse`，
K prologue 原地更新 cache，输入需位于同一 A5 设备。SGLang 的独立 AscendC
compressor 继续依赖 `cann_ops_transformer`。

## A5 验证

```bash
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl --npu-device 0 -q
```

测试保留独立数值参考；`test_metadata_graph.py` 验证三种 metadata 在长度
变化后的图回放。`scripts/check_cannbotdsl_ci.py` 核对固定环境、实际安装
产物和核心 JUnit 用例，跳过用例不计为通过。完整计算图、模型/PD 集成和
性能仍需单独在 A5 验证，离线及 fixture wheel 测试不能代替硬件结果。

`--npu-device` 指定可见卡号，数值参考仍在 CPU 上计算。
无 NPU 环境可检查源码来源和公开接口转发：

```bash
python3 -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```
