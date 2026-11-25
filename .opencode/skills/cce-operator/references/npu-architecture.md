# 昇腾 NPU 架构参考

## 识别目标

在目标运行环境查询 SoC 和架构版本。SDK 支持时，通过 `aclrtGetDeviceInfo` 的 `ACL_DEV_ATTR_NPU_ARCH` 查询架构，结合已安装的 CANN SDK 和毕昇编译选项确定目标。核数和可用资源按实际设备查询。型号与架构对照可在文档中搜索 `ACL_DEV_ATTR_NPU_ARCH`。 [华为 aclrtGetDeviceInfo 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1876.html)

## A3 · 架构 2201

- **设计 AI Core 数据流**。在 CANN 9.1.X 的 `Basic Architecture` 中找版本 220 的架构图，查 AIC/AIV 分工、GM、L1、L0A/L0B/L0C、UB 及搬运流水。设计 CV 交接前，按图核对中间结果从哪一级内存写出、Vector 从哪里读入。 [华为 AI Core 基本架构](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)
- **选择 Matmul tile**。查 `baseM`、计算访存比和大包搬运。官方案例使用 Atlas A2 的大矩阵，可参考分析方法。A3 算子应按实际形状、可用混合组、缓冲区和写回约束重新评估候选。 [华为 Matmul 性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240816-1)
- **设计 L1 双缓冲**。查「自主管理L1缓存」或 `Ping`。画出每个槽位的复用依赖，分别跟踪 MTE1 何时读完 L1、Cube 何时读完 L0A/L0B，再用 profiler 检查下一块搬运与当前块计算的重叠。 [华为 FlashAttention 双缓冲案例](https://www.hiascend.com/developer/techArticles/20250625-1)

## A5 · 已找到的 3510 资料

华为 Runtime 文档将 Ascend 950PR 和 950DT 列为 3510。下列资料适用于实际查到 3510 的目标。设计片上交接、C/V流水或随路格式转换时，继续查 [A5 C/V流水与随路搬运](a5-3510-cv-pipeline.md)。

- **从 2201 移植时**，在 CANN 9.1.X 文档中查「抽象硬件架构」「架构版本差异」和「NPU架构版本3510」。按实际指令核对 Regbase 向量计算、支持的搬运通路、片上存储及对齐约束。 [华为 2201 与 3510 架构差异和约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C/%E7%AE%97%E5%AD%90%E7%BC%96%E8%AF%91/%E7%BA%A6%E6%9D%9F%E8%AF%B4%E6%98%8E.md)
- **排查 UB bank 冲突时**，查 CANN 9.1.X 的 `Avoiding Bank Conflicts (NPU Architecture Version 3510)`。用它理解 UB bank 划分、访问冲突以及如何对比不同的访问布局。 [华为 3510 UB bank 冲突说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md)

若章节链接退回版本首页，用文档版本、章节标题和关键词重新定位。上述资料用于理解硬件行为。实际调用的 CCE Intrinsic 签名、事件机制和适用范围用目标 SDK、最小编译程序及设备验证确认。

## FixPipe 搬出时的 ReLU

评估矩阵结果能否在写回时完成激活，可查 CANN 9.1.X 的 `Real-Time ReLU`，搜索 `reluPre` 或「实时 ReLU」。先核对目标架构、累加和输出类型、量化模式及尾块写回约束，再用最小程序验证。对 CV 算子，这也可以作为评估 GM 中转和核间同步开销的性能对照。 [华为 FixPipe 实时 ReLU 说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_TensorAPI/cube_store_key_features/relu_pre.md)

## UB → GM 多段搬运

Vector 算完一个 tile，需要将 UB 中连续的多行写到 GM 中有行间距的输出时，查 CANN 9.0.0-beta.2 的 `copy_ubuf_to_gm_align`。用 `copy_ubuf_to_gm_align_b32`、`nBurst`、`lenBurst`、`srcGap`、`dstGap` 定位接口和参数。特别核对 burst 长度与源、目的间距的单位，以及 UB 实际读取量是否向 32 字节取整。短尾行、非对齐 GM 行首和相邻 tile 的并行写回要在目标 SDK 和设备上用边界哨兵验证。 [华为 UB → GM 多段搬运接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/cceintrinsicapi/cceapi_0094.html)
