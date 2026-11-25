# 仓库接入与 SFA 导航

新增 CCE 算子的声明、编译、绑定、公开接口、测试和文档时按需查阅。本页直接引用当前仓库文件，不复制另一份工程；示例只用于说明集成方式，不定义新算子的支持范围。

## 规范与入口

| 目的 | 当前入口 |
|---|---|
| 贡献流程与双语 API 结构 | [贡献指南](<../../../../docs/developer_guide/contribution_guide_zh.md>) |
| 构建选项、包安装 | [构建安装说明](<../../../../python/sgl_kernel_npu/README.md>)、[根 setup.py](<../../../../setup.py>) |
| 注册和架构选择 | [registry](<../../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/registry.py>), [generation](<../../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/utils/generation.py>) |
| 测试 CLI、精度、性能约定 | [测试说明](<../../../../tests/python/README.md>)、[测试工具目录](<../../../../tests/python/utils>) |
| 审查准则 | [PR review Skill](<../../pr-review/SKILL.md>) |
| 旧代码背景 | [Legacy 指南](<../../../../docs/developer_guide/contribution_guide_legacy.md>)，其中集中式绑定和 unittest 描述不作为新贡献模板 |

## SFA 的完整链路

| 阶段 | 参考文件 | 要提取的模式 |
|---|---|---|
| CCE Kernel | [sfa.cce](<../../../../csrc/cce/sparse_flash_attention/op_kernel/sfa.cce>) | GM/L1/L0/UB 路径、Cube/Vector 子核、分块循环、事件与复用 |
| Host 侧 | [op_host/](<../../../../csrc/cce/sparse_flash_attention/op_host>) | Host 逻辑的位置；此算子使用运行时变体选择，其他算子按需要组织 |
| Torch adapter | [torch_adapter.cpp](<../../../../csrc/cce/sparse_flash_attention/torch_adapter.cpp>) | schema、输入检查、guard、当前 stream、recordStream、设备核数查询、异步 launcher |
| 编译声明 | [CMakeLists.txt](<../../../../csrc/cce/sparse_flash_attention/CMakeLists.txt>) | EXECUTION_UNIT CCE、编译定义、符号重命名和多个变体 |
| Python 公开接口 | [sparse_flash_attention.py](<../../../../python/sgl_kernel_npu/sgl_kernel_npu/operators/attention/sparse_flash_attention.py>) | cce 注册、Generation、限制检查与原生绑定调用；auto 仍选择 ascendc |
| 独立参考 | [torch_native 实现](<../../../../python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_native/attention/sparse_flash_attention.py>) | 与公开语义一致的独立计算 |
| 测试 | [公开接口测试](<../../../../tests/python/operators/attention/test_sparse_flash_attention.py>) | 固定 CCE 支持范围的精度与性能对照；其他后端的生产和边缘用例 |
| 双语文档 | [中文](<../../../../docs/api/zh/operators/attention/sparse_flash_attention.md>), [English](<../../../../docs/api/en/operators/attention/sparse_flash_attention.md>) | 签名、后端/架构范围、参数和示例 |

SFA 的 CCE 路径面向 arch35、固定 BSND、BF16、共享 K/V、特定 head 数和 top-k/scale。新算子根据自身契约设计，不继承这些限制，不照搬 timing-only/diagnostic 宏，不把 SFA 已有测试当作新实现已验证。

SFA 当前使用 `TORCH_LIBRARY(cce, ...)` 建立命名空间；继续向同一共享库添加算子时使用 `TORCH_LIBRARY_FRAGMENT` 或仓库后续统一形式，避免第二个主注册。实现的约束检查和默认后端需要逐项核对，样例不能取代当前贡献准则。

## 构建链路与扩展边界

1. [domain 发现](<../../../../csrc/cmake/domains.cmake>)和 [CCE domain](<../../../../csrc/cce/CMakeLists.txt>)发现算子声明。
2. [operators.cmake](<../../../../csrc/cmake/operators.cmake>)登记算子和 CCE 编译变体。
3. [cce_kernels.cmake](<../../../../csrc/cmake/cce_kernels.cmake>)调用 bisheng 编译 `.cce` 对象；[CCE target](<../../../../csrc/cmake/cce/CMakeLists.txt>)与 Host 源一起构建 `libsgl_cce_operators.a`。
4. [统一 Torch target](<../../../../csrc/aclnn_torch_adapter/CMakeLists.txt>)编译选中 adapter，并链接 CCE archive 和 Runtime。[wheel 构建](<../../../../python/sgl_kernel_npu/build_tools/torch_wheel.py>)完成打包。
5. [包初始化](<../../../../python/sgl_kernel_npu/sgl_kernel_npu/__init__.py>)加载 `libsgl_kernel_npu.so`，公开接口按 impl 分派。

以 SFA 为例，从仓库根目录构建 wheel，先核对当前分支的 setup.py --help 与安装说明：

```bash
python setup.py --arch-version 3510 --ops cce_sparse_flash_attention --without-third-party
```

默认 artifact、产物位置与安装命令以构建安装说明为准。要同时对照原生 CCE 与仓库 Ascend C SFA，构建选择必须包含两者所需算子或使用确实包含它们的包；仅构建 CCE 不代表 ascendc 绑定可用。

当前 CCE 映射仅实现 dav-3510。架构声明使用 3510/2201，Python 文档与注册使用 arch35/arch22，这些名字不能混用。添加架构映射、修改公共构建/注册/测试设施需说明必要性并交维护者审查，普通算子复用现有设施即可。
