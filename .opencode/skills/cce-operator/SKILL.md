---
name: cce-operator
description: 在 sgl-kernel-npu 中开发、移植和优化直接使用 CCE Intrinsic 的 NPU 算子，接入统一构建、公开后端注册、正确性与性能测试及贡献流程。
---

# sgl-kernel-npu CCE 算子开发

Device 侧直接使用 CCE Intrinsic 编写 `.cce`。Ascend C 源码可用于查实现，交付的设备源码及依赖避免 Ascend C API 和 `kernel_operator.h`。最终产物是仓库统一构建与公开接口下的 `cce` 后端。Host 侧使用 ACL Runtime 提交到调用方的当前 NPU 流。

本 Skill 在 sgl-kernel-npu 仓库内使用，依赖当前分支的文档、构建工具、注册器及测试工具。执行流程见 [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md)。

## 先读仓库规范

- 新建或扩展算子前读[贡献指南](<../../../docs/developer_guide/contribution_guide_zh.md>)，构建时查[构建安装说明](<../../../python/sgl_kernel_npu/README.md>)，测试时查[测试规范](<../../../tests/python/README.md>)。
- 完成后按 [PR review Skill](<../pr-review/SKILL.md>)检查适用的贡献要求。该文件的审查输出格式只用于审查任务，不代替本 Skill 的开发交付说明。
- [Legacy 指南](<../../../docs/developer_guide/contribution_guide_legacy.md>)用于理解历史代码，新增 CCE 接入以当前构建、注册与测试代码为准。架构快照不能证明当前代码行为。
- 常规算子开发复用公共构建、注册、测试工具。确需修改公共设施时，先说明依赖、影响和验证范围，交由维护者审查，不顺带重构。

## 开发流程

1. **确认契约与接口。** 查仓库是否已有语义一致的公开接口，优先新增后端。明确公式、shape、dtype、stride、物理布局、精度、状态修改和支持范围。提供独立 `torch_native` 参考或说明替代验证方式；正确性 golden 与性能基线分别确定。
2. **设计并核对接口。** 按目标设备和 SDK 列出 GM/UB/L1/L0A/B/C 布局、容量、对齐和读写依赖。遇到 `(...)` 按[核对方法](references/intrinsic-verification.md)查证，无法确认必需参数时向用户列明缺失项。
3. **实现并逐步验证。** 从单核或单组单 tile 开始，按算子需要扩展核内循环、多核、尾块与步长。显式保护数据就绪和缓冲复用；用非对称输入和哨兵验证布局与边界。独立 ACL 程序可作诊断工具，不是必需交付物。
4. **接入仓库。** 在 `csrc/cce/<op>/` 声明 CCE 算子及编译变体，接入统一库；在公开 Python 接口注册 `cce` 和架构范围。`op_host/` 存放 Host 侧代码，文件名称和职责按实际需要组织，不限定为 dispatch。用户通过公开接口及 `impl="cce"` 调用，测试不手动加载独立算子 `.so`。
5. **统一测试与调优。** 使用公开接口和仓库的 `generate_data → check_accuracy → check_performance` 流程。声明生产与边缘用例及性能看护配置。保持数值、状态与计时口径一致，逐项验证优化收益；诊断需要时补充 profiler 指标、同步时延或 MFU，不复制公共计时器。
6. **交付与自检。** 更新双语 API 详情及必要索引，给出默认后端、架构范围和约束。用当前代码构建安装后跑测试和 pre-commit。性能、环境和验证证据放 PR 说明或结果附件，API 文档不放执行流水账。未运行的验证明确标注。

## 按问题查知识库

| 场景 | 查阅 |
|---|---|
| CMake、Host/Torch、公开接口和 SFA 全链路 | [仓库接入与 SFA 导航](references/repository-integration.md) |
| 架构、数据流、tile 预算 | [NPU 架构](references/npu-architecture.md)（按实际目标章节） |
| A5 C/V、随路搬运 | [A5 C/V 流水](references/a5-3510-cv-pipeline.md) |
| Intrinsic 签名、占位声明与 SDK 读码 | [核对方法](references/intrinsic-verification.md)、[资料索引](references/resource-index.md) |
| 编译、布局和同步的历史经验 | [开发经验](references/dev-experience.md) |
| 临时文件、profiling 与证据保留 | [临时文件管理](references/workspace-artifacts.md) |
| 统一计时与状态复位 | [测试工具使用](references/extracted/benchmark-template.md)、[性能与状态验证](references/performance-validation.md) |
| BF16 lane 与舍入 | [BF16 转换片段](references/extracted/bf16-conversion.md) |
| Matmul 多核与缓冲演进 | [历史演进经验](references/extracted/matmul-evolution.md) |
| MX 分组 scale 与指标口径 | [MX 量化参考](references/extracted/mx-quantization.md) |

SFA 是仓库完整接入参考，不将其固定尺寸、scale、分块、诊断宏或支持范围套用到其他算子。当前 CCE 构建映射与 SFA 支持范围以代码为准，A3 知识不代表已具备 A3 的统一 CCE 构建支持。

## 执行与文件边界

- 沿用用户授权的设备、工作目录和容器。共享服务器测试前检查授权卡占用；繁忙且没有共享授权时等待，不抢卡、不自动换卡、不终止他人进程或复位设备。测试使用仓库设备配置入口，不额外强制用户设置可见卡变量。
- 修改限制在当前任务；保留未跟踪文件和其他会话改动。产物、日志、profiling 和临时实验按[工作区约定](references/workspace-artifacts.md)留在本地或授权服务器，不把 SDK 安装树和机器专用信息加入交付。
- 设备、CANN、TorchNPU 的适用范围按证据记录，历史测试不能作为当前分支已通过的结论。当前 SDK 可用 `scripts/inspect_cann_sdk.py` 只读盘点。
