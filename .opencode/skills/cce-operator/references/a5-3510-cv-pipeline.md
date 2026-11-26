# A5 / 3510：C/V 流水与随路搬运参考

查阅日期：2026-11-19。适用于已确认目标为 dav-3510 的 Ascend 950PR/950DT。本项目设备代码使用原生 CCE Intrinsic；下列 Ascend C API 文档仅用于理解硬件和查找底层实现，不作为设备源码依赖。

## 资料索引与查询时机

| 资料 | 何时查询 |
|---|---|
| [3510 新特性](https://asc.gitcode.com/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.html) | 迁移架构，选择新增搬运、同步、Regbase、SSBuffer能力 |
| [2201→3510 架构变更](https://asc.gitcode.com/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.html) | 核对被删除的通路、L0A分形和UB结构变化 |
| [3510 硬件约束](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/hardware_constraints/npu_arch_3510.html) | 核对地址对齐、分形切分、容量和步长 |
| [L1/L0C 与 UB 数据搬运](https://asc.gitcode.com/api/SIMD-API/c_api/data_move_guide/l1_or_l0c_ub_data_move.html) | 设计 C→V、V→C 片上交接及生产者/消费者同步 |
| [L1→L0B 随路转置](https://asc.gitcode.com/api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_trans_arch_3510.html) | 复用一份L1矩阵，或核对BF16分形转置、src/dst gap |
| [Fixpipe L0C→UB](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToUB.html) | 核对格式转换、双目标、Cast、量化、UnitFlag组合约束 |
| [GM→UB NDDMA](https://asc.gitcode.com/api/SIMD-API/c_api/vector_datamove/asc_ndim_copy_gm2ub.html) | 将规则多维stride、切片、padding合并为少量搬运 |

这些 asc-devkit 页面来自持续开发分支，页面可能包含未提交修改，不能代替安装版本的SDK与上板验证。稳定文档可从[昇腾社区](https://www.hiascend.com/)按相同章节标题定位。

## 硬件通路及资源

- 1C:2V Mix 中，一个C核与两个V核协作；融合不自动意味着有重叠，流水槽和事件决定实际调度。
- MTE2：GM→L1、GM→UB。3510没有通用GM→L0A/B直达通路。
- MTE1：L1→L0A/B、L1→UB等；转置装载在此阶段执行。
- MTE3：UB→GM、UB→L1；连续UB→L1保持原内容和布局。
- FIX：L0C→GM/L1/UB，支持受约束的随路布局转换、Cast、量化和激活。
- 3510没有通用UB→L0C或L1→GM直达搬运。不要根据指针地址空间猜测通路。
- 名义容量：L1 512 KiB，L0A/B各64 KiB，L0C 256 KiB，UB 256 KiB。UB需扣除实际VF栈和保留区；本工程实验按253952字节上限核算，不能视作所有编译模式的保证。

## 设计时容易误判的地方

### 随路转置及数据复用

BF16二维转置以16×16、512B分形工作。明确转置的是分形内部、分形排列，还是整个矩阵，逐项检查起始索引、重复次数、源步长与目标间隔。L1中同一份K可以尝试供QK和PV使用，但必须用非对称、可精确表达的数据验证两种装载，不能只看公式推断布局正确。

### UB→L1 不等于免费 ND→NZ

服务器CANN9.1的 `tools/tikcpp/tikcfw/impl/dav_3510/kernel_operator_data_copy_impl.h` 提供直接参照：

- `CopyUbufToCbuf` 落到 `copy_ubuf_to_cbuf`，burst长度与gap以32B计。
- `DataCopyUB2L1ND2NZImpl` 的硬通路分支仍先执行 `TransND2NZ`，再连续搬运。
- `DataCopyUB2L1ND2NZImplV2` 按行或列循环发行多条拷贝，再做MTE3屏障。

所以减少GM中转时，要把Vector重排、DMA描述符、额外UB槽及同步开销算进去。更高版本可能改变实现，应重新核对。Ascend C 的 `ENABLE_CV_COMM_VIA_SSBUF` 是封装层选择硬通路的宏，不应当作原生CCE程序必须设置的通用开关。

### Fixpipe 的随路能力不是任意组合

L0C结果可以直接写一个或两个V核UB。双目标支持按M或N拆分，具体尺寸约束与NZ2ND/NZ2DN、量化的组合受接口限制，不能把单项支持拼成未经验证的组合。PV最终FP32→BF16可评估搬出时完成；QK喂Softmax时先保留FP32基准。

UnitFlag提供MAD与Fixpipe的细粒度交叠，但不能让消费者读取尚未完成全部K累加的结果；先核对最后一次累加与搬出的合法配合，再做最小实验。随路scale/量化不等价于任意精度的FP32乘法，也不自动支持按head的归一化系数。

### 细粒度同步

为每个槽记录：谁写、哪个流水写完、谁读、哪个流水最后读完、谁允许覆盖。V→C需要保证MTE3完成并通知真正的C侧消费者；C→V需要保证FIX/MTE1搬出完成。复用环形槽还需要反向消费完成通知，单向ready不够。

模式4支持C与单个V分别同步；C确实依赖两个V的完整矩阵时仍需等待两者。Mutex提供核内流水同步机制，但换成锁不自动减少依赖。SSBuffer为3 KiB控制/标量通信存储，不应作KV/P大块缓存；需要检查安装SDK的访问与同步要求。

### NDDMA 的适用边界

NDDMA最多五维，可配置固定stride与padding。它适合连续段、规则分页片段及布局变换，不能直接用一条固定stride指令实现任意TopK索引Gather。对随机索引，先评估分组成本、指令数量和缓存行为。共享GM被其他核改写时还需核对NDDMA Cache一致性。

## 从机制到算子

先验证装载/转换机制，再比较仅改变一个变量的重读与复用；随后接入完整Softmax/PV，最后做槽级流水。QK与PV都用同一个C核，调度时交错，不能当作能同时执行的两个计算单元。跨Query或head组重叠可降低C等待V的空档，但拆head会增加KV重读，双缓冲会减少可用tile容量。

本参考记录硬件与接口核对方法。具体实验的收益和失败应结合其架构、工具链与输入条件判断。缺少完整原型时先查安装SDK，仍无法确认则向用户询问。

## 原生VF参数核对补充

查询 `vci` 等Reg指令时，继续读毕昇资源头 `__clang_cce_vector_intrinsics.h` 的C++包装，不能只看builtin名字。CANN9.1的递增索引调用使用 `vci(vector_s16, int16_t{0}, INC_ORDER)`：第三项是编译期常量封装，直接传裸 `Order` 枚举不符合该包装。Cube/Vector编译域可见类型也不同，不应在无条件全局声明里依赖仅Vector域存在的枚举。此形式已在A5概率打包实验中编译并通过精度对照；其他dtype和工具链仍需核对。
