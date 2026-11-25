# CCE 算子开发资料索引

按"查什么"组织昇腾 CCE 算子开发的全部外部与本地资料。先定位资料类别，再按目标架构（A5/3510）和 SDK 版本筛选。所有在线链接用于理解硬件行为与查找底层实现；**实际调用的 CCE Intrinsic 签名、事件机制和适用范围以目标 SDK 头文件、最小编译程序和上板验证为准**。

## 资料类别速查

| 想查什么 | 先看哪里 | 备选 |
|---|---|---|
| 目标 SDK 与历史读码资料 | cann-cce-sdk.md + cann-cce-intrinsics.md | scripts/inspect_cann_sdk.py |
| Intrinsic 函数签名 / 参数顺序 | 本地 SDK 头文件（见下） | clangd-ascend stubs、asc.gitcode.com API 页 |
| 算子参考实现（Ascend C 写法） | hicann/ops-transformer、ops-nn | FlashMLA、DeepGEMM-Ascend |
| 硬件架构 / 数据流 / 容量 | npu-architecture.md + asc.gitcode.com guide | hiascend.com 稳定文档 |
| 3510 新特性 / 2201→3510 差异 | a5-3510-cv-pipeline.md | asc.gitcode.com cross_gen_migration |
| 性能调优 / 流水分析 | msprof / npu-compute / simulator | hiascend techArticles |
| 反汇编 / 指令到 intrinsic 反查 | npu-objdump + 本地 CANN 头文件 | asc.gitcode.com API 页 |
| 算子接口契约（aclnn） | ops-transformer docs/ | CANN 接口文档 |
| 实时性能基线 / 原地状态 / 实际频率 | performance-validation.md | 当前接口schema、同场验证 |
| 已踩过的坑 / 经验 | dev-experience.md | 当前 PR 的验证证据与必要源码说明 |

---

## 一、官方在线文档

### 1.1 昇腾社区稳定文档（hiascend.com）

版本化、按 CANN 版本归档，链接退回首页时用文档版本 + 章节标题重新定位。

| 主题 | 链接 | 用途 |
|---|---|---|
| CANN 文档中心 | https://www.hiascend.com/document | 入口，按版本/语言筛选 |
| AI Core 基本架构 | https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md | AIC/AIV 分工、GM/L1/L0/UB 数据流 |
| SIMD 高维切分 | https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/SIMD_compute/high_dimension_slicing.md | DataBlock/repeat/stride 单位换算 |
| 向量计算与步长示例 | https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0022.html | 步长配置 `{1,1,1,8,8,8}` 含义 |
| FixPipe 实时 ReLU | https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_TensorAPI/cube_store_key_features/relu_pre.md | 写回时随路激活（`reluPre`） |
| UB → GM 多段搬运 | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/cceintrinsicapi/cceapi_0094.html | `copy_ubuf_to_gm_align_b32` burst/gap |
| 3510 UB bank 冲突 | https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md | UB bank 划分与冲突规避 |
| 2201→3510 架构差异 | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/编程指南/编译与运行/算子编译/约束说明.md | 跨架构移植约束 |
| Matmul 性能最佳实践 | https://www.hiascend.com/developer/techArticles/20240816-1 | tile 选型、计算访存比 |
| FlashAttention 双缓冲 | https://www.hiascend.com/developer/techArticles/20250625-1 | L1 双缓冲案例 |
| aclrtGetDeviceInfo | https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1876.html | 查 SoC/架构（`ACL_DEV_ATTR_NPU_ARCH`） |

### 1.2 asc.gitcode.com（持续开发分支）

asc-devkit 文档，页面可能含未提交修改，**不能代替安装版本 SDK 与上板验证**。稳定文档从 hiascend.com 按相同章节标题定位。

| 主题 | 链接 | 用途 |
|---|---|---|
| 3510 新特性 | https://asc.gitcode.com/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.html | 新增搬运/同步/Regbase/SSBuffer |
| 2201→3510 变更 | https://asc.gitcode.com/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.html | 删除的通路、L0A 分形、UB 结构变化 |
| 3510 硬件约束 | https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/hardware_constraints/npu_arch_3510.html | 对齐、分形、容量、步长 |
| L1/L0C ↔ UB 搬运 | https://asc.gitcode.com/api/SIMD-API/c_api/data_move_guide/l1_or_l0c_ub_data_move.html | C→V、V→C 片上交接同步 |
| L1→L0B 随路转置 | https://asc.gitcode.com/api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_trans_arch_3510.html | BF16 分形转置、src/dst gap |
| Fixpipe L0C→UB | https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToUB.html | 格式转换/Cast/量化/UnitFlag |
| GM→UB NDDMA | https://asc.gitcode.com/api/SIMD-API/c_api/vector_datamove/asc_ndim_copy_gm2ub.html | 多维 stride/切片/padding 合并搬运 |

---

## 二、参考实现仓库（GitHub / GitCode）

### 2.1 算子参考实现（Ascend C 写法，用于查底层调用和数据流）

> **重要**：这些仓库的 device 源码多用 Ascend C API（`AscendC::`、`kernel_operator.h`）。本项目 device 源码**禁用** Ascend C API，只用原生 CCE Intrinsic。参考时聚焦底层 CCE 调用、数据流布局和 tiling 思路，不照搬封装层。

| 仓库 | 路径 | 价值 | 查什么 |
|---|---|---|---|
| **hicann/ops-transformer** | https://github.com/hicann/ops-transformer | ★★★ Attention 类算子官方实现 | `attention/sparse_flash_attention/`、`attention/flash_attention/` 的 `op_kernel/arch22/`、`op_host/*_tiling.cpp`、`docs/aclnn*.md` |
| **hicann/ops-nn** | https://github.com/hicann/ops-nn | ★★★ NN 基础算子（Matmul/Normalization/Activation） | `matmul/`、`layernorm/`、`activation/` 的 kernel 与 tiling |
| **Ascend/asc-cce** | https://gitcode.com/org/Ascend/repos（搜 asc-cce / cce） | ★★ CCE 官方样例 | 最小 CCE kernel 范式、intrinsic 调用样例 |
| **Ascend/cann-ops** | https://gitcode.com/org/Ascend/repos（搜 cann-ops） | ★★ 官方算子库 | aclnn 接口定义、op_host tiling 参考 |
| **deepseek-ai/clangd-ascend** | https://github.com/deepseek-ai/clangd-ascend | ★★★ Intrinsic stubs 与调用样例 | `include/cce_stubs/cce_intrinsic_stubs.h`（`(...)` 占位声明，用于查符号名）、`tests/ascend.hpp`（`copy_gm_to_ubuf_align_v2` 等）、`tests/bf16_gemm.asc`（`set_flag`/`wait_flag`） |

### 2.2 模型/框架参考实现（A5 原生 kernel）

| 仓库 | 路径 | 价值 | 查什么 |
|---|---|---|---|
| **deepseek-ai/FlashMLA** | `csrc/ascend_kernels/prefill/sparse/` | ★★★ A5 原生 kernel 参考 | `kernel.h`/`kernel_body.h`/`config.h`/`instantiations/*.asc`；h_q=64/128 + topk + sink，与 SFA 场景几乎一致 |
| **deepseek-ai/DeepGEMM-Ascend** | — | ★★ host/device 共享头技巧 | `__CCE__` 条件编译、`gm_ptr` 共享；不含 .asc 主体 |
| **deepseek-ai/DeepEP-Ascend** | — | ★★ 同上 | host/device 参数结构体共享 |
| **sgl-project/sglang** | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | ★★ 模型调用方 | `forward_sparse`、`npu_sparse_flash_attention` 调用契约、scale/索引扩维 |
| **sgl-project/sglang** | `python/sglang/srt/models/deepseek_v2.py` | ★★ 模型缩放 | YaRN scaling、MLA-absorb 投影 |

### 2.3 GitCode 组织入口

| 组织 | 入口 | 内容 |
|---|---|---|
| **cann** | https://gitcode.com/org/cann/repos | CANN 官方仓库集合（toolkit、ops、文档） |
| **Ascend** | https://gitcode.com/org/Ascend/repos | 昇腾官方仓库集合（asc-cce、cann-ops、samples、pytorch、modelzoo） |
| **asc-devkit** | https://asc.gitcode.com | 文档站（见 §1.2） |

> GitCode 组织页可能因权限/网络无法直接列出全部 repo。已知仓库名（ops-transformer、ops-nn、asc-cce、cann-ops、pytorch、modelzoo、samples）直接拼 URL 访问更可靠。

---

## 三、本地 SDK 资料

### 3.1 CANN Toolkit 安装目录

通过 `ASCEND_HOME_PATH` 或 `${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}` 定位。

| 路径（相对 toolkit 根） | 内容 | 查什么 |
|---|---|---|
| `include/acl/acl.h` | ACL Runtime API | `aclrtSetDevice`/`aclrtMalloc`/`aclrtMemcpy`/`aclrtSynchronizeStream`/`aclrtLaunchKernel*` |
| `lib64/libascendcl.so`、`lib64/libruntime.so` | 运行时库 | 链接目标 |
| `tools/bisheng_compiler/lib/clang/<ver>/include/` | 毕昇资源头 | **CCE Intrinsic 签名核对的首选来源** |
| `.../__clang_cce_vector_intrinsics.h` | Vector intrinsic C++ 包装 | `vci` 等 Reg 指令的 C++ 封装（不能只看 builtin 名） |
| `.../__clang_cce_vector_intrinsics_legacy.h` | Legacy vector intrinsic | 旧版接口 |
| `.../__clang_cce_aicore_functions.h` | AICore 函数 | 搬运/同步/系统变量 |
| `.../cce_aicore_intrinsics.h` | AICore intrinsic 声明 | `(...)` 占位，需配合文档/反汇编 |
| `tools/tikcpp/tikcfw/impl/dav_3510/` | A5 Ascend C 实现源 | `kernel_operator_data_copy_impl.h`（搬运实现，查硬通路分支） |
| `tools/tikcpp/tikcfw/impl/dav_2201/` | A3 Ascend C 实现源 | 同上，A3 版本 |
| `x86_64-linux/asc/include/utils/base/sys_constants.h` | 系统常量 | `ASC_IS_AIV` 等 constexpr 定义 |
| `bin/bisheng` | 毕昇编译器 | `bisheng --npu-arch=dav-3510` |

定位编译器资源目录：`bisheng -print-resource-dir`。

### 3.2 反汇编与 .o 文件分析

| 工具/方法 | 用途 |
|---|---|
| `npu-objdump -d <kernel.o>` | 反汇编 device 侧 AICore 指令，查实际生成的助记符 |
| `readelf -p .strtab <kernel.o>` | 查 kernel 符号名、`.Kernel_Type_*`、`.AIV_Kernel_Type_*` |
| `readelf -S <kernel.o>` | 查 `__CCE_KernelArgSize`、`__CCE_KernelMetaData`、`.ascend.meta.*` NOTE 段 |

**典型流程**：写最小 CCE 调用 → `bisheng -c --aicore-only` 生成 `.o` → `npu-objdump -d` 看实际指令 → 回到 CANN 头文件核对签名 → 上板验证。

---

## 四、性能分析与调优工具

| 工具 | 用途 | 触发场景 |
|---|---|---|
| **msprof**（`torch_npu.profiler`） | 算子级 Task Duration、流水利用率、op_summary | 性能基线、CCE vs 基线对比、瓶颈定位 |
| **npu-compute** | 带宽、流水线、存储访问、资源冲突指标 | memory-bound vs compute-bound 判定、指令级分析 |
| **CANN Simulator**（`npusim`） | 无硬件环境下流水线仿真、精度仿真 | 无 NPU 时验证功能、分析性能瓶颈、定位精度问题 |
| **npu-smi** | 设备列表、状态、资源占用、卡占用检查 | 共享服务器选卡、确认设备空闲 |
| **asys** | 故障信息收集、coredump 解析、堆栈导出 | 卡死/崩溃/内存错误诊断 |
| **msaicerr** | AI Core Error 解析、Dump 解析 | `507015` 等运行时错误定位 |

### profiler 配置与输出

正常精度与性能回归使用仓库[测试工具](../../../../tests/python/utils/performance.py)，配置、CSV 解析和汇总由公共实现管理，使用方法见[测试参考](extracted/benchmark-template.md)。无需在算子测试中再创建一份 profiler 配置或计时器。

诊断需要时按目标版本核对 msprof 参数及输出，不能把历史 TorchNPU 导出格式当成所有环境约定。独立同步调用时延与额外流水分析按任务需要补充，和公共指标分开报告。

### 性能指标口径

- **device tasks**：profiler 记录的 NPU 任务时长之和 ÷ 采样次数（不含 host 开销）
- **synchronized**：Python 发起 → `torch.npu.synchronize()` 返回（含接口/提交/等待）
- **MFU**：有效 FLOPs（FMA 计 2）/ (kernel 耗时 × 峰值算力)。峰值算力按 dtype 与架构查官方规格，不明时不猜测
- **带宽利用率**：实际搬运字节 / (kernel 耗时 × 峰值带宽)

---

## 五、按开发阶段的资料查询路线

### 阶段 1：定义算子契约
- 模型调用方：sglang `ascend_backend.py`、`deepseek_v2.py`
- 官方接口文档：ops-transformer `docs/aclnn*.md`
- 模型配置：HuggingFace `config.json`

### 阶段 2：设计数据流 + 核对 Intrinsic
- 架构/数据流：`references/npu-architecture.md` + hiascend AI Core 基本架构
- 3510 特性：`references/a5-3510-cv-pipeline.md` + asc.gitcode.com 3510 页
- Intrinsic 签名：本地 SDK 头文件（§3.1）→ clangd-ascend stubs → asc.gitcode.com API 页
- 调用样例：clangd-ascend `tests/`、ops-transformer `op_kernel/`

### 阶段 3：最小 CCE 路径
- 编译范式：`references/dev-experience.md`（bisheng vs ccec、入口修饰、UB 分配）
- 已验证样例：[仓库 SFA 完整链路](repository-integration.md)（CCE、Host、Torch、公开接口、测试与文档）
- 踩坑规避：`references/dev-experience.md`

### 阶段 4：PyTorch 接入
- 构建与原生绑定：[SFA 仓库接入导航](repository-integration.md)
- 流语义：`references/dev-experience.md`（DeviceGuard、当前流、allocator）

### 阶段 5：性能调优
- 采集：msprof（§四）、npu-compute
- 分析：流水利用率、bound 诊断、tile/缓冲/调度迭代
- 反汇编验证：npu-objdump -d 看实际生成的指令，回 CANN 头文件核对签名
- 经验：[开发经验](dev-experience.md)；当前验证证据放任务结果及 PR 描述，参考当前分支规范

---

## 六、链接维护

- 在线链接失效时，用**文档版本 + 章节标题 + 关键词**在 hiascend.com 或 gitcode 重新定位。
- 新增资料时，按"查什么"归类，注明适用架构、SDK 版本和验证状态。
- 本地 SDK 路径以 `ASCEND_HOME_PATH` 为准，不硬编码用户名或安装目录名。
- 参考实现仓库的 commit hash 在引用具体代码行时一并记录（如 ops-transformer `32cfaec`）。
