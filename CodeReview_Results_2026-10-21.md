# DSPark Triton-Ascend 接口适配审查

## 1. 结论与范围

审查对象是 `feat/npu-dspark-triton-ops` 分支中的 14 个新增 DSPark NPU
模块及其框架调用接口。依据用户提供的 `D:/Github/triton-ascend` 源码：

- Commit：`a176ab28b11b63d427760c15f6cd63c14ac296f1`。
- `python/triton/__init__.py:2` 声明版本 `3.6.0`。
- 该编译器工作区没有已跟踪文件的修改；本次未修改编译器。
- 当前状态：**发现并修正一项确定的编译接口阻断；其余已用 API 在该版本
  找到实现或测试依据。设备编译、数值与资源使用尚未验证。**

之前 Markov 验证节点使用的是 Triton 3.2.x / triton-ascend 3.2.2。
本报告不能证明这些新算子兼容那个旧版本，也不能用 Markov-only 的 GPQA
结果替代本次完整迁移的验证。

## 2. 已修正的问题

### 循环内 `continue`：确定的编译阻断

`attention/dspark_kv_write.py` 将 GPU 二维 grid 转为 persistent loop 后，
曾用 `continue` 跳过未提交或无效 cache location。当前前端没有
`visit_Continue` / `visit_Break`；
`python/triton/compiler/code_generator.py:1552` 的 `generic_visit` 会抛出
`unsupported AST node type`。

已经改为：计算 `write_row`，用 `if write_row & (loc >= 0)` 包裹写入主体。
不使用循环内 `return`、`break` 或 `continue`。host 回归检查也覆盖这三种
控制流，避免后续再次引入。

### FP32 projection：采用当前编译器已支持的 Cube dot

初稿对 FP32 projection 使用了向量乘加。源码表明当前 Ascend 已支持 FP32
`tl.dot`，因此现改为和 FP16/BF16 一样调用 `tl.dot`，显式指定
`input_precision="ieee"`，FP32 累加，并使用 AI core 数量配置 grid。
这消除了不必要的向量 matmul 分支；没有据此声明性能提升。

## 3. API 差异核对

下列行号均对应上述编译器 commit。

| 项目 | 当前源码证据 | 本次处理 |
| --- | --- | --- |
| 动态 for / if / while | `python/triton/compiler/code_generator.py:898,1057,1131` | 保留有界 persistent loop 和条件分支 |
| `num_warps` | `third_party/ascend/backend/compiler.py:1234,1491` | 当前选项接受；使用 1，不把它当 CUDA warp 数来推断 Vector/Cube 性能 |
| `num_stages` | `compiler.py:1387-1390` 默认 2；`:752-763` 控制 multibuffer | 不能说此选项无效；未在无测量情况下统一修改，UB 待编译验证 |
| FP32 dot 与精度 | `compiler.py:1257-1258`；`unittest/pytest_ut/test_dot.py:126-160` | 显式 IEEE；不沿用 GPU TF32 调优假设 |
| 核类型 | `compiler.py:598,1567`；`backend/runtime/dsl_analysis/kernel_classifier.py:26,71` | 核类型由 IR/lowering 决定；普通 JIT 不要求额外传 `kernel_type="cube"` |
| `tl.cumsum` | `third_party/ascend/unittest/pytest_ut/test_cumsum.py`、`test_associative_scan.py` | 保留 int32 scan |
| `tl.cumprod` | `third_party/ascend/unittest/pytest_ut/test_cumprod.py:181` 等 | 保留已提升到 FP32 的累计乘法 |
| Tensor 下标 | `python/triton/language/core.py:1088-1102` | 当前只用 `[:]` 和 `None` 扩维，没有整数索引或有限区间切片 |
| Scalar load/store | `third_party/ascend/unittest/pytest_ut/test_scalarPointer.py:91` | 有界标量访存是合法接口；向量尾块仍保留 mask |
| `tl.rand` | `python/triton/language/random.py:127-156`；`backend/op_dtype_config.py:1861-1879` | offsets 为整数；标准 rand 先 bitcast 到 signed 再折叠负半区，无需复制自定义 Markov 的 unsigned 转换补丁 |
| int64 地址转指针 | `third_party/ascend/unittest/pytest_ut/test_indirect_load_pointer_cast_precise_size.py:24-28` | 保留 meta 地址到 BF16 buffer 指针的转换；实际地址与 dtype 仍由 KV UT 验证 |
| int64 select | `python/triton/language/semantic.py:1668-1682`；`lib/TritonToLinalg/TritonOpConverter.cpp:550` | 未发现当前前端无条件经 FP32 转换的证据；token assembly 仍用位运算选择，并保留大整数 UT |
| FP8 load/cast | `compiler.py:1255` 声明 dtype；`test_dot_scaled_fp4_fp8.py:92-99` 有平台限制 | 不能由 FP8 scaled-dot 限制推断所有 FP8 cast 都不支持；新增完整 bit-pattern 和 scale 边界 UT，待 A3/A5 验证 |
| libdevice | NPU Markov 模块使用 CANN libdevice | 不重新导入 GPU generic libdevice；其他新增模块主要使用标准 tl 数学 API |

## 4. 保留的风险与支持边界

- Python 接口存在和编译器自带 UT 有覆盖，不等于本次完整 kernel 已编译成功。
  本机没有可用的 Ascend 编译/运行环境。
- `MarkovGreedyStep` 保留 GPU 的向量 GEMV/归约算法。是否改用 Cube 需要保持
  数值语义的独立设计和性能对照，不属于 API 缺失。
- Scan 的 batch 维、metadata 的 window 维、schedule 的比较矩阵，以及
  默认 multibuffer 都会影响实际 UB 占用。当前 grid/block 调整只能作为设计
  检查，不能宣称资源验证通过。
- KV projection 的 INT8、packed/MXFP8 scale 格式没有新增支持，不能用旧
  native Linear 静默替代。NPU block verification 也仍未支持。
- 新 API 未发布到官方固定 wheel，框架集成发布前需使用实际包含这些算子的
  wheel 更新 Docker/CI 版本；本分支没有虚构未来版本号。

## 5. 测试与证据

- 迁移保留了 CUDA parity suite 的全部 20 个 `_case_*`。已比较 AST，所有
  case body、参数、断言和容差一致；测试辅助入口和设备设置允许适配。
- 另迁移 3 个 simulated-bonus 用例，新增 NPU 边界测试，原有 Markov 与
  chain-sampling UT 保留。
- GPU kernel 目录相对本分支 HEAD 没有修改。
- Host launch/interface 检查通过 8 项。这些检查不执行设备代码。
- SGLang adapter 的 3 项行为测试通过独立 host harness 执行；标准 pytest
  在 Windows 的导入被现有 Linux `resource` 依赖阻止，不应宣称完整 SGLang
  测试环境已通过。
- NPU 测试在本机 skip；没有本次完整迁移的 NPU 编译、数值、GPQA 或性能结果。

## 6. 后续验证

安装包含本分支代码的 NPU wheel，在与目标部署匹配的编译器版本运行
`docs/developer_guide/dspark_triton.md` 中的 UT。重点检查 FP32 dot、FP8 cast、
masked KV write、NaN/tie 语义、graph replay 和 UB。再执行完整 DSPark 端到端
测试；最后才做相同配置下的性能对照。
