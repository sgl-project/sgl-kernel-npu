> 适用范围：本页保留特定 dav-3510 工具链的历史编程经验。裸编译命令仅用于局部诊断；仓库构建、测试与交付以[接入导航](repository-integration.md)和当前规范为准。历史测试及性能数字不代表本分支已验证。

# A5 / dav-3510 CCE 开发经验

以下提炼已验证算子开发中的可复用经验，按踩坑类型和有效模式组织。Matmul 的阶段推进和实现变更见[演进经验](extracted/matmul-evolution.md)，当前仓库接入以[SFA 导航](repository-integration.md)为准。

适用范围：Ascend 950（`__NPU_ARCH__ == 3510`，编译目标 `dav-3510`），CANN 9.1.0–9.3.0，毕昇 Baize clang 15.0.5。其他架构/版本需重新核对。

---

## 一、编译与启动范式（最重要）

### 1.1 必须用 `bisheng`，不能用 `ccec`

| 错误路径（`507015` 崩溃） | 正确路径（PASS） |
|---|---|
| 编译器 `ccec` | 编译器 `bisheng` |
| `--cce-aicore-arch=dav-c310-vec` | `--npu-arch=dav-3510` |
| `__aicore__` 入口 | `[aicore] __vector__` 入口 |
| 裸 UB 地址 `(__ubuf__ T*)0` | `get_imm(offset)` 分配 UB |
| `aclrtLaunchKernelWithHostArgs` | `<<<blocks, nullptr, stream>>>` 启动 |
| `aclrtCreateBinary` + `aclrtBinaryLoad` | bisheng 直接生成 .so / 可执行文件 |
| `libascendc_runtime.a`（重依赖） | 只需 `-lascendcl -lruntime` |

**根因**：`ccec` + `dav-c310-vec` 链路 UB 映射不建立，任何 UB 访问触发 `507015`。`bisheng` + `dav-3510` 是 A5 正确范式。这是特定编译链的历史调试观察，迁移时需按实际编译目标复验。

### 1.2 局部诊断编译

正式交付使用仓库统一构建，见[接入导航](repository-integration.md)。只需确认 intrinsic 调用形式时，可对临时最小源文件执行；先按[工作区约定](workspace-artifacts.md)设置 `CCE_WORK_DIR` 并放入 `minimal.cce`：

```bash
bisheng --npu-arch=dav-3510 -O2 -Werror -c --aicore-only "$CCE_WORK_DIR/minimal.cce" -o "$CCE_WORK_DIR/minimal.o"
```

这只验证设备编译，不证明 Host launcher、链接或实际计算正确；还需目标环境运行验证。避免把独立设备 `.so` 和第二套 Torch 构建脚本带入最终贡献。

### 1.3 启动语法

```cpp
// ✅ 正确：bisheng <<<>>> 语法
kernel<<<blockDim, nullptr, stream>>>(args...);

// ❌ 错误：第二参数是 nullptr，不是 0
kernel<<<blockDim, 0, stream>>>(args...);  // 编译过但运行异常
```

`.o` 加载模式用 `aclrtLaunchKernelWithHostArgs`，但 `hostArgs` 是**连续参数数据块**（指针值本身），不是指针数组：

```cpp
void* hostArgs[6] = { dK, dV, dKc, dVc, dSlot, &dTiling };  // ❌ 传了指针的地址
void* hostArgs[6] = { dK, dV, dKc, dVc, dSlot, dTiling };    // ✅ 传指针值本身
```

---

## 二、Kernel 写法要点

### 2.1 入口修饰

```cpp
// 纯 Vector 算子
extern "C" __global__ [aicore] __vector__ void kernel(...) { }

// 纯 Cube 算子
extern "C" __global__ [aicore] __cube__ void kernel(...) { }

// 混合 Cube+Vector（1C:2V）
extern "C" __global__ [aicore] __mix__(1, 2) void kernel(...) { }
```

- 按实际计算单元选择纯 Vector、纯 Cube 或混合入口。仓库 SFA 使用 `__mix__(1,2)`，当前 CCE 构建与统一链接规则未显式要求 `libascendc_runtime.a`；不能仅凭混合入口增加该依赖。遇到链接错误时核对目标 SDK、实际编译/链接命令和未解析符号，再定位原因，不能将某次编译路径的问题推广为通用要求。
- 公共设备函数只保留 `[aicore]`，核类型属性放在 Kernel 入口。

### 2.2 UB 分配

```cpp
// ✅ 正确：通过 get_imm(offset) 分配
__ubuf__ float* buf = (__ubuf__ float*)get_imm(0);
__ubuf__ float* buf2 = (__ubuf__ float*)get_imm(kBytes);

// ❌ 错误：裸 UB 地址
__ubuf__ float* buf = (__ubuf__ float*)0;  // 507015
```

UB 容量上限按 253952 字节核算（扣除 VF 栈和保留区），不能视作所有编译模式的保证。

### 2.3 向量操作必须用 `__simd_vf__` 包装

```cpp
// ✅ 正确：向量操作在 __simd_vf__ inline 函数内
__simd_vf__ inline void add_vectors(__ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* out) {
    vector_f32 left, right, sum;
    vector_bool mask = pset_b32(__cce_simd::PAT_ALL);
    vlds(left, a, 0, __cce_simd::NORM);
    vlds(right, b, 0, __cce_simd::NORM);
    vadd(sum, left, right, mask);
    vsts(sum, out, 0, __cce_simd::NORM_B32, mask);
}

// ❌ 错误：向量操作直接写在 [aicore] __vector__ kernel 内
//   触发 "Do not know how to split the result of this operator!"
```

### 2.4 流水同步

```cpp
// GM → UB → Vector → UB → GM 的标准同步
copy_gm_to_ubuf_align_v2(ub, gm, 0, 1, bytes, 0, 0, false, 0, 0, 0);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
// ... vector 计算 ...
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
copy_ubuf_to_gm_align_v2(gm, ub, 0, 1, bytes, 0, 0, 0);
pipe_barrier(PIPE_ALL);
```

### 2.5 混合 Kernel 的 FFTS 同步

混合 Kernel（Cube+Vector）首次运行常返回 `507015`，设备日志指向 FFTS 无效地址。修复：

```cpp
// Host 侧先获取控制地址
void* syncAddr;
aclrtGetHardwareSyncAddr(stream, &syncAddr);

// Device 侧 Cube 和 Vector 都用 set_ffts_base_addr 初始化后再发通知/等待
set_ffts_base_addr(syncAddr);
ffts_cross_core_sync(...);
```

多 tile 用例中，**两个 Vector 核共享同一个回执事件**（分别分配会超时）。

---

## 三、已知硬件缺陷

### 3.1 BF16转换：限定调用形式的历史失败，不视为架构级禁用

fused_causal_conv1d历史记录在特定工具链/寄存器布局下观察到`vcvt(...PART_ODD)`输出零。该记录缺少可复跑的完整输入布局与编译证据，不足以推出所有3510 BF16转换均有硬件缺陷。

需要BF16时，先核对目标SDK的向量宽度、有效lane、打包/解交织及PART语义，以最小测试覆盖非对称输入、两半lane和往返转换。失败时记录编译器、指令与布局，不直接把接口改为FP32后仍声称与BF16基线同精度。

### 3.2 `ASC_IS_AIV` 是 `constexpr(...)` 表达式

```cpp
// ❌ 错误：不能加 ! 或用在布尔表达式
if (!ASC_IS_AIV) { }

// ✅ 正确：用 if-else 语句形式
if (ASC_IS_AIV) { } else { }
```

### 3.3 bisheng (clang 15) 与系统 GCC 12 的 STL 不兼容

避免 `#include <vector>` 等 STL 头文件在 bisheng 编译的 device 代码中。Host 代码用 g++ 编译则无此问题。需要 STL 时 host 代码单独编译并显式链接 `libstdc++`。

---

## 四、数据搬运与布局

### 4.1 GM ↔ UB 搬运（11/8 参数 v2 版）

```cpp
// GM → UB（11 参数）
copy_gm_to_ubuf_align_v2(dst_ub, src_gm, 0, 1, bytes, 0, 0, false, 0, 0, 0);
// UB → GM（8 参数）
copy_ubuf_to_gm_align_v2(dst_gm, src_ub, 0, 1, bytes, 0, 0, 0);
```

短尾行、非对齐 GM 行首和相邻 tile 的并行写回要用**边界哨兵**验证。

### 4.2 Cube 装载与 ND2NZ

```cpp
// GM → L1 时转换布局（ND → NZ 分形）
copy_gm_to_cbuf_multi_nd2nz_b16(l1, gm, ...);
```

- ND2NZ 只覆盖有效区域。尾块直接加载时 L1 旧值污染填充区域 → **先清零相关 L1 tile**，再用 `pipe_barrier(PIPE_MTE2)` 隔开两次重叠写入。
- L1 → L0B 转置参数用**编译期常量**。

### 4.3 Fixpipe 随路能力

- L0C → GM/L1/UB，支持受约束的随路 NZ→ND、Cast、量化和激活（ReLU）。
- **不是任意组合**：双目标支持按 M 或 N 拆分，与 NZ2ND/NZ2DN、量化的组合受接口限制，不能把单项支持拼成未经验证的组合。
- `reluPre` 可在写回时完成 ReLU，省一次 GM 中转 + Vector 启动。来源：matmul_relu（Fixpipe 随路 ReLU）。

### 4.4 UB → L1 不等于免费 ND→NZ

CANN9.1 的 `DataCopyUB2L1ND2NZImpl` 硬通路分支仍先执行 `TransND2NZ` 再连续搬运。减少 GM 中转时，要把 Vector 重排、DMA 描述符、额外 UB 槽及同步开销算进去。

---

## 五、多核与任务切分

### 5.1 按设备核数自动选组数

在 Torch adapter 内，参考[当前 SFA 的查询方式](../../../../csrc/cce/sparse_flash_attention/torch_adapter.cpp)，先初始化结果并检查返回值及有效范围：

```cpp
// device 为输入 Tensor 的 c10::Device；使用 adapter 已引入的 ACL/Torch 头文件。
int64_t cubeCores = 0;
const aclError status = aclrtGetDeviceInfo(
    device.index(), ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeCores);
TORCH_CHECK(status == ACL_SUCCESS && cubeCores > 0,
            "could not query Cube core count: ", status);
```

`aclrtGetDeviceInfo` 的属性名、输出指针类型及平台支持范围仍需按目标 SDK 头文件核实。若调度还需要 Vector 核数，另行核对目标 SDK 的 Vector 属性并检查查询结果，不能沿用未经核实的 `ACL_DEV_ATTR_AIC_CORE/AIV_CORE` 或用 `size_t*` 替代要求的 `int64_t*`。

对于每组使用一个 Cube 核的方案，按有效任务数与可用 Cube 核数选择组数；1C:2V 的混合配置还需确认平台的子核配比和启动约束，不能把任意核数相除直接当成可用组数。历史某次设备查询得到 24 Cube、48 Vector，只用于解释当时的分工，不作为 A5 的固定规格。

### 5.2 任务划分

```cpp
// 输出 tile 编成一维任务，第 g 组处理 g, g+groupCount, g+2*groupCount, ...
uint32_t tile = get_block_idx();
while (tile < totalTiles) {
    // ... 处理 tile ...
    tile += get_block_num();
}
```

- 每组独占 workspace 槽位，避免跨组竞争。
- 空闲组（无 tile）跳过循环和核间等待，用哨兵检查不改写 workspace。
- Vector 用 `get_subblockid()` 分工（如两个 V 核各处理半个 tile）。

### 5.3 尾块处理

- K 尾块：补零，对齐缓冲区预先清零，缺失元素不贡献累加。
- M/N 尾块：只访问有效区域，用 `copy_ubuf_to_gm_align_b32` 逐行搬运有效字节。
- 三维同时有尾块（如 `M=19,N=53,K=35`）必须覆盖。

---

## 六、PyTorch 接入

### 6.1 统一构建

最终交付复用仓库 CCE 编译、静态 archive 和统一 Torch 共享库链路，见[接入导航](repository-integration.md)。Torch 头文件和 C++ ABI 由当前构建环境确定，不再交付独立 `build_torch.sh` 或第二套 Torch `.so` 加载流程。前面的裸 bisheng 命令仅用于隔离诊断。

### 6.2 Torch 绑定范式

```cpp
TORCH_LIBRARY_FRAGMENT(cce, m) {
    m.def("op(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(cce, PrivateUse1, m) {
    m.impl("op", &op_npu);
}

at::Tensor op_npu(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "...");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "...");
    // ... 契约检查（dtype/shape/连续性/对齐）...
    const c10::DeviceGuard guard(a.device());
    auto output = at::empty({M, N}, a.options());
    auto stream = c10_npu::getCurrentNPUStream(a.get_device()).stream();
    cce_launch_op(a.data_ptr(), output.data_ptr(), stream);
    return output;
}
```

- 用 `TORCH_LIBRARY_FRAGMENT` 与已有算子共享 `cce` 命名空间。
- 绑定保持**异步**，Host 同步留给测试或调用方。
- 输入可能跨分配流使用时按 TorchNPU allocator 语义登记其使用流。

### 6.3 流语义测试

必须覆盖：
- 默认流和非默认流
- 上游生成与下游消费（同一流上 PyTorch 操作消费输出）
- 提交后释放临时输入并施加显存分配压力（异步生命周期）
- 错误 dtype/shape/stride/device 被拒绝

---

## 七、性能调优有效模式

### 7.1 已验证有效的优化

| 优化 | 适用场景 | 收益 | 来源 |
|---|---|---|---|
| K 分块从 16 增至 32 | Cube matmul | 11–28% 用时下降 | matmul_relu |
| Fixpipe 随路 ReLU | Cube + 激活融合 | 省一次 GM 中转 + Vector 启动 | matmul_relu |

### 7.2 已验证无效或退化的尝试

| 尝试 | 结果 | 原因 |
|---|---|---|
| BN32（直接多段 UB→L1） | 退化（~70.5ms vs 51.6ms） | 跨块交接开销 |
| 独立同步 64-head tile（H64/H128） | 未通过编译（需 C++17） | 合入三槽通用版本未验收 |
| bf16 路径（fused_causal_conv1d，历史） | 某调用形式PART_ODD输出异常 | 版本与布局待最小测试复核（§3.1） |
| 逐 head Softmax（vs 批量） | 慢 | Softmax 开销未摊薄 |

### 7.3 性能测量纪律

- **正确性与性能入口分开**。性能测试默认先验证数值。
- 固定相同输入、dtype、设备、精度设置，预热与采样次数可配置并写入结果。
- 分别报告 **device tasks**（profiler Task Duration）和 **synchronized**（含 host 开销）。
- `torch_native` 等多 kernel 基线须覆盖每次调用的**全部阶段**，不能只取其中一个阶段。
- 流水时间可交叠，**不能相加**或仅凭某项下降认定因果。
- 报告 MFU 时写明有效 FLOPs 公式（FMA 计 2）、峰值口径和权威来源；padding/因果不可见区域不计入有效 FLOPs。
- 峰值不明时**不猜测**利用率。

---

## 八、调试经验

### 8.1 `507015`（UB 越界 / FFTS 无效地址）

最常见错误。排查顺序：
1. 编译器是否 `bisheng`（不是 `ccec`）？
2. `--npu-arch=dav-3510`（不是 `--cce-aicore-arch=dav-c310-vec`）？
3. 入口是否 `[aicore] __vector__`（不是 `__aicore__`）？
4. UB 是否 `get_imm(offset)` 分配（不是裸地址）？
5. 启动是否 `<<<blocks, nullptr, stream>>>`（第二参数是 `nullptr`）？
6. 混合 Kernel 是否 `set_ffts_base_addr` 初始化？
7. 向量操作是否在 `__simd_vf__` 包装内？

### 8.2 设备授权与频率

沿用会话授权的单卡及逻辑映射。用户允许共享时记录占用，不额外强制独占或必须设置可见卡环境变量。遵守当前任务明确的选卡范围，不终止他人进程、不复位设备。

`npu-smi info -t common -i <physical>`区分标称`Freq`与实际`curFreq`；空闲低频不能证明负载低频。性能异常需在持续负载期间只读采样，保存频率、利用率、温度与功耗；没有授权不修改频率。详见[性能与状态验证](performance-validation.md)。

### 8.3 哨兵检查

- 输出和 workspace 边界放置哨兵值（如 256 个 FP32），验证后保持原值。
- 检查设备 tile 填充区域是否为零。
- 检查每个活跃组最后一个 workspace tile。

---

## 九、文件与目录约定

交付结构见[开发清单](../DEVELOPMENT_RULES.md)及[仓库接入](repository-integration.md)。

- CCE 代码、Host 侧代码和 adapter 按仓库目录组织，必要 `.h`/`.inc` 是源码依赖，随调用方放入合理的源码目录并入库，不要求保存在 `experiments/`。
- 生成物、原始日志、profiling 与临时实验使用仓库忽略的本地或授权服务器目录，不入库。
- 双语 API 文档描述公开行为。开发设计按任务需要记录，详细正确性及性能证据放 PR 描述或附件，不强制每个算子增加 DEVELOPMENT/OPTIMIZATION 文档。
- SDK 安装树、个人路径和机器专用配置不进入贡献内容。
