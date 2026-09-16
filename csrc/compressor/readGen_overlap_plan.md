# readGen 轮询与输入载入重叠 —— 实施计划（⑤）

> 目标文件：`csrc/compressor/op_kernel/arch35/compressor_kernel.h` 与
> `csrc/compressor/op_kernel/arch35/compressor_block_cube.h`。
> 只针对 **arch35 (A5) + `cache_mode=2 (CYCLE)`**。

## 0. 已完成（①③，本仓已改）

`csrc/compressor/op_kernel/arch35/compressor_kernel.h`：

- **① slot-major**：`readGen[dbIdx][aiv]`
  - AIC 读：`readGenBase + (info.cubeDbIdx * aivNum + a)`
  - AIV 写：`readGenGm + (info.c1v1DbIdx * aivNum + GetBlockIdx())`
  - host 归零区域不变（仍是 `aivNum*dbRatio` 个 uint32）。
- **③ API 名**：`ReadGmByPassDCache` / `WriteGmByPassDCache`
  → `ReadGmBypassDCache` / `WriteGmBypassDCache`（同操作，仅去废弃告警）。

### ② 已撤销（原"跳过本核 AIV"）

原打算跳过本核 2 个 AIV 的 GM 轮询（认为 `SYNC_MODE2` flag 已覆盖）。
**撤销原因**：README `:184-190` 明确 "pure flags cannot express a generation"
（same-flag reordering，消费者可能被提前放行）。这个论证**对本核 AIV 同样成立**——
`CrossCoreWaitFlag<SYNC_MODE2>(SYNC_V1_C1_FLAG + cubeDbIdx)` 是"按槽 flag"，
跨圈复用会有 same-flag reordering，**不能替代本核 AIV 的圈数检查**。
所以本核 2 个 AIV **必须照常轮询**。

## 1. 目的

现在 `readGen` 轮询（56 次标量 GM 读）在 `ComputeMm1` **入口**、`blockCube_.ComputeMm1`
**之前**，与后续的输入载入 **完全串行**。把它挪到 **x 载入指令发射之后、等待载入之前**，
让轮询与 MTE2 搬运并行，把轮询延迟藏进载入。

## 2. 文档依据（asc-devkit）

- `guide/technical_appendix/concepts_and_terms/glossary.md:391`：
  **Scalar** 执行地址计算/循环控制，并把矢量/矩阵/搬运/同步指令**发射**给对应单元。
- `guide/programming_guide/programming_model/ai_core_simd_programming/abstract_hardware_architecture.md:27`：同上。
- `.../cpp_tensor_programming/cube_matrix_computation.md:355`：
  AI Core 内部执行单元（MTE2/Vector 等）**异步并行**；Cube 四步 = `PIPE_MTE2`/`PIPE_MTE1`/`PIPE_M`/`PIPE_FIX`。

**推论**：MTE2 载入**一旦发射**就在后台跑；标量随后去自旋**不影响已发射的 MTE2**。
所以"先发射载入、再轮询、再等载入"能重叠。

## 3. 当前 `block_cube.h::ComputeMm1` 结构（:344-438）

```
for (h = 0; h < hSize; h += K_L1_BASE) {           // K 循环
    WaitFlag<MTE1_MTE2>(X_EVENT0 + xBufId);
    CopyXGmToL1(...);                              // ← 发射 x 载入（MTE2）
    SetFlag<MTE2_MTE1>(X_EVENT0 + xBufId);
    WaitFlag<MTE2_MTE1>(X_EVENT0 + xBufId);        // ← 等 x 载入
    for (coffId) {
        CopyWeightGmToL1(...);                     // ← 发射 w 载入（MTE2）
        ...
        for (mL0) for (nL0) {
            LoadAToL0/LoadBToL0; MatrixMmad(...);  // 不写槽
            if (isLast) CopyOutMm1Res(...);        // ★ 首次写槽（readGen 保护点）
        }
    }
}
```

## 4. 怎么改

### 方案 A（推荐，改动小）：把轮询逻辑做成回调/参数，传进 `ComputeMm1`

- 在 `CompressorBlockCube` 增加一个成员/参数，保存：
  `__gm__ uint32_t* readGenBase`、`uint32_t gen`、`uint32_t cubeDbIdx`、`uint32_t aivNum`、`uint32_t local0`；
- 在 `ComputeMm1` 的 K 循环里，`SetFlag<MTE2_MTE1>(X_EVENT0 + xBufId)` **之后**、
  `WaitFlag<MTE2_MTE1>(X_EVENT0 + xBufId)` **之前**，插入轮询循环：

```cpp
WaitFlag<MTE1_MTE2>(X_EVENT0 + xBufId);
CopyXGmToL1(info, xL1Tensor, hStart + hIdx, kSize);   // 发射 MTE2
SetFlag<MTE2_MTE1>(X_EVENT0 + xBufId);
// ★ 轮询（只在第一次 K 迭代需要；后续已保证）
if (needPollReadGen) {
    for (uint32_t a = 0; a < aivNum; ++a) {
        if (a == local0 || a == local0 + 1) continue;
        while (AscendC::ReadGmBypassDCache(readGenBase + (cubeDbIdx * aivNum + a)) < gen) {}
    }
    needPollReadGen = false;
}
WaitFlag<MTE2_MTE1>(X_EVENT0 + xBufId);
```

- `compressor_kernel.h::ComputeMm1` 里去掉原来的轮询，改为把参数传给 `blockCube_`
  并调用 `blockCube_.ComputeMm1(info)`。

### 方案 B（改动大）：拆函数

```
blockCube_.PreloadMm1(info);            // 只发起 x/w 载入（不 Wait）
<readGen 轮询>
blockCube_.ComputeMm1AfterLoad(info);   // 等载入 + matmul + 写槽
```

## 5. 红线（违反=静默算错）

1. **轮询必须在首次 `CopyOutMm1Res`（写槽）之前**；
2. **轮询必须在 `CopyXGmToL1` 发射之后**（否则没有重叠窗口，等于没改）；
3. 只轮询一次（第一次 K 迭代），后续 K 迭代不必再轮询（同一个槽）。

## 6. 风险

- **不重叠**：若编译器把 `WaitFlag(MTE2_MTE1)` 提前，或标量自旋反而挡住载入发射 → 白改；
- **挪过头**：挪到 `CopyOutMm1Res` 之后 → 在 AIV 读完前覆盖 → 静默错；
- **接口改动**：`ComputeMm1` 需要拿到 `readGenGm/gen/cubeDbIdx/aivNum`（都在 kernel）。

## 7. 验证

1. **精度**：`tests/python/sgl_kernel_npu/test_compressor.py`（含
   `test_ring_real_c4_batch256_multi_round`、MTP、graph capture/replay）；
2. **性能**：sglang profiler 比 compressor op 耗时（before/after），
   或内核内 `AscendC::GetSystemCycle()` 量轮询 cycle；
3. **确认真的重叠**：对比"轮询在 `WaitFlag` 前 vs 后"的耗时差。

## 8. 不建议做的（④）

"只轮询覆盖该 group 的行"：读者集合由运行时 `dealedSeqCnt`（数据相关）决定，
静态收窄不成立，猜错=静默错。**先别做。**
