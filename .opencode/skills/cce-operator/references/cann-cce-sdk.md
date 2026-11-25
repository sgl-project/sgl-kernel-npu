# CANN SDK 资料定位与历史验证快照

## 范围与版本

以下为迁入知识库的 2026-11-21 历史验证环境，不代表当前仓库或开发设备配置：

| 项目 | 实测 |
|---|---|
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-9.1.0` |
| CANN compiler package | `Version=9.1.0`，`timestamp=20260730_231653901` |
| bisheng | clang 15.0.5，`clang-5c68a1cb1231` |
| CCE resource dir | `$ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/15.0.5` |
| 本次验证编译目标 | `--npu-arch=dav-3510` |
| Python/TorchNPU | Python3.11，TorchNPU2.7.1.post8 |

这些是本次安装快照，不是所有环境的固定路径。其他版本的文档不能代替当前 SDK。每次切换环境先运行：

```bash
command -v bisheng
bisheng --version
bisheng -print-resource-dir
python .opencode/skills/cce-operator/scripts/inspect_cann_sdk.py
```

脚本默认将 JSON 输出到 stdout；需要保存时传入 `--output <任务工作区>/cann-cce-sdk.json`，路径按[临时文件约定](workspace-artifacts.md)选择。脚本仅查询编译器、目录及文件哈希，不初始化 NPU、不选卡、不修改 SDK。

## 按问题定位

| 想查什么 | 安装中的首选入口 | 作用与边界 |
|---|---|---|
| 编译器自动引入哪些头 | resource `include/__clang_cce_runtime_wrapper.h`、`__clang_cce_runtime.h` | CCE编译上下文与运行时封装；通常无需手动include |
| 地址空间、目标宏、属性 | `__clang_cce_defines.h`、`__clang_cce_types.h` | `__gm__`/`__ubuf__`及目标条件等，沿定义继续查 |
| Cube/scalar intrinsic与枚举 | `__clang_cce_aicore_intrinsics.h`、`cce_aicore_intrinsics.h` | builtin桥接与生成的类型枚举；不是每个builtin都能找到普通C++函数原型 |
| scalar兼容封装与搬运配置 | `__clang_cce_aicore_functions.h` | 多种旧/新重载、宏展开、config字段打包；需连同条件编译读取 |
| 核/组索引等编译器变量 | `__clang_cce_aicore_builtin_vars.h` | `get_block_idx`等内建入口，按目标宏核对 |
| 向量类型和寄存器宽度 | `__clang_cce_vector_types.h` | 3510分支`CCE_VL=256`字节；不能把元素数与字节数混淆 |
| Vector intrinsic重载 | `__clang_cce_vector_intrinsics.h` | `vlds/vsts`、算术、`vcvt`、`vdintlv`等模板与builtin实参 |
| SIMT | `__clang_cce_simt*.h`、`__clang_cce_simt_dav3510_intrinsic.h` | 另一执行模型；存在不代表可直接混用到现有SIMD VF中 |
| Host设备、流、内存与属性 | SDK `include/acl/acl_rt.h`及`pkg_inc/runtime/runtime/rt.h` | ACL/Runtime声明，和Device intrinsic分开 |
| 官方算子两段式接口 | `include/aclnnop/`，如`aclnn_fused_causal_conv1d.h` | Workspace/executor/launch签名；dtype与属性限制还要实际调用验证 |
| 官方高层API如何下沉 | SDK `asc/include/`与`asc/impl/` | 只读追踪参数到intrinsic；交付的 CCE Device 代码不依赖 Ascend C API |
| 算子包与编译后内核 | SDK `opp/` | 包版本、已安装算子与内核信息；不保证随包交付全部源码 |
| 反汇编、profile、sanitizer | SDK `bin/`与`tools/` | 按后文核实实际工具，不照搬旧版本名称 |

`include`和工具路径包含符号链接。普通`find`可能只看到链接本身；定位后先`readlink -f`，或用有深度限制的`find -L`。不要递归扫描整个`/usr/local/Ascend`并输出所有二进制。

## 定位具体符号

```bash
resource_dir=$(bisheng -print-resource-dir)
# 有rg优先使用rg；本次容器没有rg，可用grep。
grep -n 'copy_gm_to_ubuf_align_v2' "$resource_dir/include/__clang_cce_aicore_functions.h"
grep -n 'vlds\|vcvt\|vdintlv' "$resource_dir/include/__clang_cce_vector_intrinsics.h"
grep -n 'CCE_VL\|3510' "$resource_dir/include/__clang_cce_vector_types.h"
```

阅读命中前后的`#if`、宏定义和宏实例化。区分：原始builtin、CCE兼容包装、AscendC高层包装。若符号由编译器直接识别，查找不到C++声明并不等于目标不支持；最小`.cce`编译、反汇编及上板是独立证据。实参复核见[调用核对方法](intrinsic-verification.md)，已核对细节见[CCE搬运与SIMD笔记](cann-cce-intrinsics.md)。

## AscendC 源码作为只读参考

本次安装包含：

- `asc/include/c_api/asc_simd.h`：C API入口。
- `asc/include/c_api/vector_datamove/`、`cube_datamove/`、`sync/sync.h`：分类接口。
- `asc/include/basic_api/`、`interface/`：Tensor/同步等封装接口。
- `asc/impl/tensor_api/arch/vector/gm_to_ub/copy_impl/data_copy.h`、`ub_to_gm/`、`ub_to_l1/`：分架构搬运实现。
- `asc/impl/tensor_api/arch/cube/l0c_to_ub/`、`l1_to_ub/`、`l0c_to_gm/`：矩阵搬出路径。
- `asc/impl/adv_api/`：高级算子实现及参数检查。

从公开接口沿调用追到架构实现、配置字段、目标intrinsic，记录同版本路径与哈希。不能因为某个分支使用一种burst/gap单位，就推广到另一个同名或旧版重载。未经阅读与验证，不将私有实现当稳定API。

## 编译与分析工具

| 本次实际存在的入口 | 用途 |
|---|---|
| `$ASCEND_HOME_PATH/bin/bisheng` | 历史 Host+Device 编译入口，仓库正式构建由 CMake 调用 |
| `tools/bisheng_compiler/bin/llvm-objcopy`、`llvm-objdump` | 对支持的ELF提取/查看section与汇编 |
| `$ASCEND_HOME_PATH/tools/msobjdump/msobjdump` | 检查/提取Host共享库中嵌入的Device ELF |
| `$ASCEND_HOME_PATH/bin/msprof`、`tools/profiler/` | 采集、分析与导出实际任务；默认采集项以`--help`和输出配置为准 |
| `tools/msopprof/` | 算子级分析入口，按安装版本帮助使用 |
| `tools/msdebug/`、`tools/mssanitizer/`、`tools/msaicerr/` | 调试、内存检查和错误分析；存在不等于本项目已验证 |
| `tools/simulator/bin/cannsim-0.1.0-py3-none-any.whl` | 本次只确认安装包存在，未据此宣称仿真已可运行 |

实际核对过：

```bash
"$ASCEND_HOME_PATH/tools/msobjdump/msobjdump" --help
# 帮助中有--list-elf、--dump-elf、--extract-elf、--out-dir。
# 用提取后的Device ELF反汇编，不能把Host x86汇编当NPU指令。
msprof --help
```

历史操作验证过以下提取形式。复验时先按[工作区约定](workspace-artifacts.md)设置 `CCE_WORK_DIR`，将实际待分析库放入该目录或替换为当前构建库的路径：

```bash
"$ASCEND_HOME_PATH/tools/msobjdump/msobjdump" -l "$CCE_WORK_DIR/kernel_library.so"
"$ASCEND_HOME_PATH/tools/msobjdump/msobjdump" -e "$CCE_WORK_DIR/kernel_library.so" --out-dir "$CCE_WORK_DIR/device-elf"
```

历史诊断曾从候选库提取 Device ELF（`elf64-hiipu`）。但本次PATH中的`llvm-objdump -d`对该ELF只显示`<not available>`，因此尚未得到可用的NPU指令反汇编；不能据此判定没有指令、没有spill或断言某条硬件指令已生成。应另核对支持该目标的解码器/编译器汇编输出。

某些旧记录写`npu-objdump`；本次SDK PATH实际看到的是`llvm-objdump`与`msobjdump`，使用前`command -v`核实。不要用ccec与bisheng的选项互换而假定等价。

## 最小编译与验证顺序

1. 确认SDK、目标SoC和资源目录，保存inventory哈希。
2. 核对声明/宏、地址空间、单位与架构条件，只保留一个待测操作。
3. 使用与目标仓库构建相同的编译器与架构参数，参考 `csrc/cmake/cce_kernels.cmake`；历史读码目标为 dav-3510。
4. 查看Device代码生成；在已授权设备上验证非对称输入、对齐、尾块与哨兵。
5. 限定范围记录结果，再用于完整算子；性能还需同库、同状态、同频率的当前基线。

SDK头只能证明声明与编译器封装，不能证明所有形状的硬件语义。工具存在、编译通过、设备精度通过、性能有效是四种不同结论。

## 本次目录差异核实

盘点脚本最初假设`include/runtime/rt.h`，实际不存在；本次安装真实Runtime头为`pkg_inc/runtime/runtime/rt.h`（另有x86_64-linux链接），已修正脚本。12个关键CCE资源头均存在。

TorchNPU2.7.1的Level0采集实际导出`ASCEND_PROFILER_OUTPUT/kernel_details.csv`，列名为`Name`和`Duration(us)`；不保证同时导出`op_summary*.csv`。Level2/独立msprof的`Op Name`、`Task Duration(us)`是另一布局。按实际导出选择其中一种，不能合并两种CSV而重复计数；核对每次调用的任务数后才归一化。
