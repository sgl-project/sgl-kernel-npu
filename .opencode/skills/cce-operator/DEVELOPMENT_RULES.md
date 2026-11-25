# CCE 算子开发与交付检查清单

与 [SKILL.md](SKILL.md) 配套。执行前读取当前分支的[贡献指南](<../../../docs/developer_guide/contribution_guide_zh.md>)，详细接入路径见[仓库与 SFA 导航](references/repository-integration.md)。仓库实际代码用于核实构建、注册和测试行为；本文补充 CCE 专有检查，不另建一套公共基础设施。

## 1. 契约与支持范围

- [ ] 查已有公开接口与调用方。语义一致时新增 `cce` 实现，保持签名、参数含义和返回值；新增接口补齐模块导出。
- [ ] 明确数学公式、累加与转换精度、shape、dtype、layout、stride、对齐、标量、动态维、空输入、非法索引及原地状态。
- [ ] 后端特有限制在启动前检查；不能忽略会影响结果的公开参数。
- [ ] 提供独立参考实现，通常放 `implementation/torch_native/<module>/`。同一函数的 auto/后端别名不算两种独立实现。
- [ ] 选定性能基线并记录实现路径；CPU golden 不能充当 NPU 性能基线。

## 2. 数据流与 Intrinsic

- [ ] 核对目标 SoC、SDK、编译目标和设备核数，设计 GM、UB、L1、L0A/B/C 布局、对齐与容量预算，计入多缓冲活跃区。
- [ ] 标明每份缓冲的最后读者和复用时机。区分核内 flag、组内握手及 stream 同步，不能以 `PIPE_ALL` 代替跨核屏障。
- [ ] Vector 逐阶段验证 GM → UB → 计算 → GM；Cube 验证装载/转置、首次 MAD 初始化、后续累加和 Fixpipe 写回。
- [ ] 按[Intrinsic 核对方法](references/intrinsic-verification.md)查证实参、单位、地址空间和架构支持，最小编译后再验证结果。必要参数缺资料时向用户提问。
- [ ] 设计说明放适当源码注释或任务设计记录，API 文档只包含用户需要理解的行为和约束。

## 3. Kernel 与统一构建

交付位置按实际算子组织，以下为目录职责，不要求固定 Host 文件名：

```text
csrc/cce/<op>/
├── CMakeLists.txt
├── op_kernel/                  CCE 源码和必要辅助文件
├── op_host/                    Host 侧代码，按需要组织文件和职责
└── torch_adapter.cpp           原生 Torch 绑定
```

- [ ] 由单 tile 推进到需要的核内循环、多核、尾块和步长。覆盖任务少于组数、不整除和每组多任务；这些检查按算子实际调度适用。
- [ ] 非对称输入验证布局，输出/workspace 哨兵检查越界，尾块补零不读旧数据；Host 不为 NPU Tensor 做逐元素打包作为最终路径。
- [ ] 通过 `sgl_register_operator(... EXECUTION_UNIT CCE ...)` 声明 `CPP_SOURCES`、`TORCH_ADAPTER`、`SUPPORTED_ARCHS`，通过 `sgl_register_cce_kernel` 声明编译源与必要变体。
- [ ] 多个编译变体的 Kernel 和 launcher 符号唯一。仅需一个实现时不照搬 SFA 的四变体分发。
- [ ] 使用仓库统一 CMake/build_tools 构建安装。选择 CCE 算子时，RUN 包含 `libsgl_cce_operators.a`；wheel 包含已链接 CCE 实现及所选 Torch adapter 的 `libsgl_kernel_npu.so`，不要求单独交付 archive。安装 wheel 后验证 Torch 注册、公开接口和实际执行。无需交付独立 `build_torch.sh` 或手动 load_library 流程。
- [ ] 核对[构建映射](<../../../csrc/cmake/cce_kernels.cmake>)。当前仅有 3510 CCE 映射；若新增架构需要改公共构建，说明依赖和影响，交维护者审查，不能仅改声明声称支持。
- [ ] 临时 ACL 测试或候选实现可用于诊断。最终保留实际构建依赖、必要来源和许可证，清理无关实验及生成物。

## 4. Torch 绑定与公开接口

- [ ] adapter 按统一库选择机制编译，核对 schema 和实现签名；与现有 `cce` 命名空间共存时使用 `TORCH_LIBRARY_FRAGMENT`，避免重复主注册。
- [ ] 在启动前检查设备、dtype、shape、物理格式、stride、对齐、标量及前向/反向范围。
- [ ] 输入设备 guard、当前 NPU stream、输入/输出/workspace 生命周期正确；按 allocator 语义记录跨流使用。绑定异步，不能靠 Host 同步掩盖依赖错误。
- [ ] `operators/<module>/<op>.py` 注册 `cce` 和公开函数，通过 `Generation` 声明实际架构范围，与 CMake 保持一致。
- [ ] 公开接口使用 keyword-only `impl="auto"`，说明实际默认后端。新增 CCE 后端不自动改变已有 auto；别名的架构声明保持一致。
- [ ] 需要 lazy import 时复用仓库缓存惯例，遵守依赖方向，不让实现反向导入 operators 或其他后端。

## 5. 统一正确性与性能测试

- [ ] 测试位于 `tests/python/operators/<module>/test_<op>.py`，导入公开接口，独立数据生成、pytest 参数化；每组 shape 有描述性 ID。
- [ ] 仓库自维护算子含真实生产场景或明确说明借用维度的用例，以及有价值的边缘用例。声明性能看护配置，控制默认组合规模；补充用例按规范标记 `extra_checks`。
- [ ] 通常采用 `generate_data → check_accuracy → check_performance`，复用仓库 `--impl`、`--impl-ref`、精度/性能开关和输出路径。
- [ ] 默认流、非默认 producer/consumer 流、临时输入释放及显存压力、非法输入拒绝按实际接口覆盖；补充回归适当标记，不测试未修改的公共基础设施。
- [ ] 原地或状态算子用独立输入及 `assert_accuracy`，再用 `make_executor`；只有比较通过才设 `accuracy_checked=True`。处理重复执行的状态复位并说明计时是否包含其成本。
- [ ] 性能基线同场实际运行，记录输入、dtype、精度、设备、软件及调用路径。完整基线计算纳入采样，正确性和输入准备放在采样外。
- [ ] 公共指标为每次调用设备任务时长之和，不能冒充端到端或重叠任务跨度；需要时额外分析同步时延、分位数、流水指标和 MFU，注明来源与口径，不强制所有算子提供无意义指标。
- [ ] 优化逐项验证，同口径比较首版、候选和基线；无收益或正确性失败则回退，不放宽精度门限。性能工具使用见[测试参考](references/extracted/benchmark-template.md)。

从仓库根运行，路径和实现名替换为目标算子：

```bash
python -m pytest tests/python/operators/<module>/test_<op>.py \
  --impl=cce --impl-ref=torch_native --skip-acc-check=false --enable-perf=false -v -s -x -rs
python -m pytest tests/python/operators/<module>/test_<op>.py \
  --impl=cce --impl-ref=torch_native --skip-acc-check=false \
  --prof-dir=build/cce-work/<op>/<run-id>/profilings \
  --csv-path=build/cce-work/<op>/<run-id>/performance.csv -v -s
```

`<run-id>` 每次取唯一值，输出及保留规则见[临时文件管理](references/workspace-artifacts.md)。

省略 `--enable-perf` 使用用例内性能选择，显式 true 会覆盖所有用例。参考实现需按实际可用后端选择，某些固定实现对照测试不接受 CLI 替换，先读用例。

## 6. 文档、审查与最终验收

- [ ] 双语 API 详情说明架构、签名及默认实现、公式、参数/返回值、每个后端支持范围和调用示例。新接口或有用户可见变化时更新必要的双语索引；内部变更不制造无意义索引编辑。
- [ ] 文档中硬件表使用支持/不支持/未验证三态，英文对应 Supported/Unsupported/Unverified，不把缺少实现写成硬件未验证。
- [ ] 任务设计/优化记录按需要保留；测试命令、环境、正确性与性能证据放 PR 描述或附件。API 文档不放 Agent 日志、机器信息或开发流水账。
- [ ] 测试确实加载当前构建的包，核对公开导出、后端可选性和限制拒绝；未运行项明确说明。
- [ ] 按仓库 pre-commit 配置检查，检查自动格式化后的 diff，并按 [PR review Skill](<../pr-review/SKILL.md>)审查适用要求。
- [ ] 无失效引用、个人路径或凭据，构建依赖及许可证齐全，生成物不入库。限定任务范围，提交和推送遵守当前用户授权。

## 7. 临时文件与证据

- [ ] 按[临时文件管理](references/workspace-artifacts.md)选择本任务工作区，写入前用 `git check-ignore` 确认仓库内输出被忽略。
- [ ] 正式源码和正式测试进入仓库约定目录；诊断源、候选副本、profiler、日志、SDK inventory 留在任务工作区或授权服务器。
- [ ] 不修改公共忽略规则来掩盖杂项，不清理其他会话的文件；临时数据不成为源码构建或 Skill 的必需依赖。
- [ ] 清理本任务可再生成的中转文件；验证证据先保留到报告完成及不再需要复验时，再明确清理范围。
