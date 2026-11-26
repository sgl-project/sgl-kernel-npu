# CANN 9.1.0 CCE 搬运、同步与 SIMD 读码笔记

配套[SDK导航](cann-cce-sdk.md)。以下基于迁入的2026-11-21历史bisheng15.0.5资源头读码记录，目标dav-3510；同名的旧兼容接口、其他架构或其他CANN版本需重新核对。全文是定位与复验方法，不是完整Intrinsic API手册。

## 1. 关键文件与读码证据

相对`$(bisheng -print-resource-dir)/include/`：

| 文件 | 本次读取位置 | 可确认的事实 |
|---|---|---|
| `__clang_cce_vector_types.h` | 文件头，`CCE_VL`条件定义 | 3510分支向量寄存器256字节；FP32/u32为64 lane，BF16/u16为128 lane |
| `__clang_cce_vector_intrinsics.h` | `__VF_VLDS`，约1548行起 | `vlds`的offset注明元素单位，builtin接收`offset*sizeof(LT)`；dist是受静态检查的类型，不是任意整数 |
| 同上 | `vdintlv`，约681/718行附近 | 不同架构有不同实现/重载，必须看目标条件，不能只复制第一个命中 |
| 同上 | `vsts/vsti`，约1999行起 | 多种元素/寄存器重载，最终写出布局受dist及mask控制 |
| `__clang_cce_aicore_functions.h` | `MOV_UB_TO_OUT_ALIGN`/`MOV_OUT_TO_UB_ALIGN`，约485–615行 | 旧align包装会由gap重建stride，和直接v2的显式stride形式不同 |
| 同上 | `__cce_set_flag`/`__cce_wait_flag`，约3445行起 | 本版本兼容函数忽略传入p/tp，实际固定向`PIPE_M→PIPE_V`发送/等待；不用于任意pipe同步 |
| 同上 | 紧邻的`__cce_pipe_barrier` | 该兼容包装同样忽略传入p并固定调用；使用项目已验证的原始`pipe_barrier`形式 |

行号只用于这次安装定位；升级后以符号和文件哈希重新定位。资源头宏可含CRLF，搜索工具显示`\r`不代表参数本身多了字符。

## 2. GM ↔ UB：直接 v2 与兼容包装分开

历史dav-3510测试中已编译并上板的常用显式形式：

```cpp
copy_gm_to_ubuf_align_v2(dst_ub, src_gm,
    sid, nBurst, lenBurst, leftPadding, rightPadding, data_select,
    l2_cache_mode, burst_src_stride, burst_dst_stride);
copy_ubuf_to_gm_align_v2(dst_gm, src_ub,
    sid, nBurst, lenBurst, l2_cache_mode,
    burst_dst_stride, burst_src_stride);
```

- 一共11参/8参，两者stride顺序相反；先核对地址空间和参数位置，再解释目标特性错误。
- `lenBurst`是字节数，显式stride是相邻burst起点间距，包含本burst长度。不要把`stride=0`当作“连续多行”。
- 本次常用输入LP/RP均为0，`data_select=false`，cache mode为0；这不等于已验证其他padding/cache组合。
- 同名旧`copy_*_align`包装接收的gap可能由字节与32B单位混合构造；读取宏中的`ceil32`、`sizeof(ATYPE)`及config打包，不能把旧gap语义套给v2。
- `copy_gm_to_ubuf`、`copy_ubuf_to_cbuf`等非align-v2接口另有32B burst/gap规则；SFA使用的NZ散射参数仅在其形状/布局下成立。

例如FP32连续D列，UB按D列稠密存放，可用一个`nBurst=1, lenBurst=rows*D*4`搬整块。UB行距固定为2048列时则需要显式源行距`D*4`、目标行距`2048*4`。前者更少描述符，但只有物理连续时才等价。

## 3. Vector load/store 与 dtype 变换

`__simd_vf__`中执行向量指令，可调用合适的`__simd_callee__`辅助函数。Host指针、GM指针与UB指针不能互换。

- `vlds` offset按指针元素计数；NORM整寄存器读涉及256B，mask用于计算/写出不自动约束之前的整寄存器读取。
- `vector_f32`是64个FP32，`vector_u16`是128个16-bit；`PART_EVEN/ODD`、`PK_B32`、解交织等共同决定转换布局。
- 输入128个BF16时，不能仅看到64个FP32输出就认为其对应原输入前64个元素。必须标记lane序号验证两半和回程。
- 浮点`vcvt`某一形式失败不足以判定整个芯片不支持BF16；先检查输入是否按目标重载要求打包、mask是否匹配，以及编译目标/实际指令。

### dav-3510 RMSNorm 中已验证的连续BF16加载

2026-11-23，同一CANN9.1.0/bisheng15.0.5环境，RMSNorm有限值回归验证了以下组合：`vlds(vector_bf16, bf16_ptr, 0, UNPK_B16)`后，`vcvt(vector_f32, loaded, mask, PART_EVEN)`得到64个连续FP32元素；输出经`vcvt(...PART_EVEN)`及`vsts(...PK_B32)`写回64个BF16。128列需两次64元素加载/转换。资源头中的模式名是`UNPK_B16`，不是`UNPACK_B16`。

本次NORM加载后直接用PART_EVEN/PART_ODD拆两半的候选未通过精度，不能照搬其布局假设。以上验证尚未覆盖所有位型。

### 本次可复验的 BF16 整数位路径

卷积新增候选中，按128个BF16读取成64个u32：每个word低16位对应偶数元素，高16位对应奇数元素。将低半左移16位得到偶数FP32位型，高半按`0xffff0000`屏蔽得到奇数FP32位型；位重解释不做数值整型转换。

FP32转BF16按RNE：对FP32位型`bits`先加`0x7fff + ((bits>>16)&1)`，再右移16位。偶/奇结果分别放入u32低/高半并或合，直接存128个BF16。该路径需保留原来的FP32乘加顺序；不能顺手换FMA后仍承诺逐位一致。

该整数路径已在当前工具链编译，decode、L<C、多batch、残差和prefill样例对FP32计算后BF16舍入参考逐位一致；decode样例与安装的官方BF16接口一致。进一步验证的native打包候选用两次`vcvt(...PART_EVEN)`得到偶/奇结果，再用u32低半mask、左移和或操作合并，不使用PART_ODD。该候选通过61项FP32/BF16回归（包括C=32、额外state slot、非默认流和输出哨兵）。这证明历史PART_ODD失败不能推广为所有BF16转换均失效；不保证任意NaN payload或跨版本行为。独立片段及使用边界见[BF16 转换参考](extracted/bf16-conversion.md)。

## 4. 同步的四种范围

| 机制 | 范围/用途 | 不应假设 |
|---|---|---|
| `set_flag(src,dst,event)`/`wait_flag(...)` | 核内不同流水的生产/消费与缓冲复用 | 不能当跨核全局屏障 |
| `pipe_barrier(...)` | 当前核支持流水的排序 | `PIPE_ALL`不是所有核完成 |
| `set_intra_block`/`wait_intra_block` | 混合C/V配置中的组内交接 | 需核对子核编号、通知/回执次数与实际SDK初始化要求 |
| Runtime stream同步 | Host等待提交队列完成 | 不应放入异步PyTorch算子绑定作为修复数据竞争的方法 |

事件要同时保护“数据已就绪”和“旧消费者已读完”。核内flag发收配对，不能在循环里重复覆盖未消费信号。MTE3写回UB缓冲的读取结束前，下一轮MTE2不能覆盖同一UB。当前版本`__cce_*`兼容封装行为已由源码确认，使用原始已验证`set_flag/wait_flag`。

对于原地状态，先明确唯一写者和所有读者。因果卷积只有第一个时间tile读旧state时，可由该tile所有者在读完后更新；其余tile只读x。若划分改变、其他核也读旧state，这个论证就失效，必须重新设计所有权或同步。

## 5. Cube、Fixpipe 与 FFTS 查找词

在aicore头和同版本`asc/impl/tensor_api/arch/`按以下符号追踪：

- `load_cbuf_to_ca`、`load_cbuf_to_cb`、`mad`：L1→L0A/B布局、转置与初始化/累加。
- `copy_matrix_cc_to_ub`、`copy_matrix_cc_to_gm`、`set_loop3_para`：Fixpipe输出形状、stride、分形与子块分工。
- `copy_gm_to_cbuf_multi_nd2nz`、`set_mte2_nz_para`：ND→NZ的目标行距与多维参数。
- `set_ffts_base_addr`、`aclrtGetHardwareSyncAddr`、`set_intra_block`：混合kernel的硬件同步地址与握手。

这些符号用于查找目标 SDK 的调用实现。具体参数组合、布局和累加行为需用目标环境的最小编译和设备测试确认。
