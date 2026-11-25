# CCE Intrinsic 调用核对方法

目标 SDK 定位方法及历史安装的头文件和工具入口见[SDK导航](cann-cce-sdk.md)，已读取的搬运/SIMD/兼容封装行为见[读码笔记](cann-cce-intrinsics.md)。先用`../scripts/inspect_cann_sdk.py`定位实际SDK并保存哈希，不把历史安装路径作为当前依据。


遇到 `(...)` 声明、参数语义不清或需要跨架构移植时，按以下顺序核对。记录目标 SoC、架构和 CANN SDK 版本，查证结果限定在相应目标。

1. **定位接口。** 在目标 SDK 的毕昇资源头文件中查同名 intrinsic 和 builtin 别名。`bisheng -print-resource-dir` 可定位编译器资源目录。也可以用 [clangd-ascend 的 intrinsic 声明](https://github.com/deepseek-ai/clangd-ascend/blob/master/include/cce_stubs/cce_intrinsic_stubs.h) 搜索候选名称。`(...)` 只表明符号存在，无法确定实参顺序、类型或目标支持范围。
2. **找到调用并核对语义。** 优先查同架构、同版本的官方 CCE Intrinsic 文档、目标 SDK 实现和实际调用样例。SDK 声明仍为 `(...)` 时，搜索该 intrinsic 的调用点，沿封装传参追到配置字段、类型转换和常量。核对参数顺序、地址空间、长度和步长单位、对齐、布局及流水依赖。可借鉴 Ascend C 实现里的底层调用，Device 源码直接使用 CCE Intrinsic。
3. **最小编译，再上板。** 用目标毕昇编译选项验证该调用是否接受指针类型、枚举、立即数和目标特性。再用最小设备程序验证结果，视接口选用非对称输入、尾块、非默认步长、输出哨兵或多轮缓冲区复用。编译诊断只能约束调用形式，参数单位、布局和同步语义还需运行验证。
4. **记录结论。** 记录来源、适用架构和版本、验证过的参数组合、编译与上板结果。若公开资料和可访问 SDK 都无法确认必需参数，向用户列出接口名、目标架构、SDK 版本和具体缺失项，再请用户提供资料。

## 查证实例

- **向量加法与 GM/UB 搬运**。用生成的 CCE 源码核对搬运的参数顺序，再用官方资料核对长度单位。用 Add 的设备输出和 CPU 结果逐元素比对。 [公开生成代码示例](https://gitee.com/ascend/modelzoo/issues/I4E3MM) · [华为搬运参数说明](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/tbeaicpudevg/atlasopdev_10_0068.html)
- **Cube 装载与写回**。从目标 SDK 实现追踪 `load_cbuf_to_ca`、`load_cbuf_to_cb`、`mad` 和 `copy_matrix_cc_to_gm` 的实参。用非对称矩阵和跨多个分形的 tile 验证布局与步长。
- **跨核同步**。从目标 SDK 实现核对 `ffts_cross_core_sync` 和 `wait_flag_dev` 的消息编码、初始化和参与核。先验证一次交接，再测试多个 tile 连续复用同一槽位。
