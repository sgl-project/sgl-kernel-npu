# readGen 优化 —— 测试指南

改动文件（都在 `csrc/compressor/op_kernel/arch35/`）：

| 文件 | 改动 |
|---|---|
| `compressor_kernel.h` | ① slot-major readGen 布局；③ 非废弃 API；⑤ 把轮询交给 `blockCube_` |
| `compressor_block_cube.h` | ⑤ 新增 `SetReadGenPoll` + 在 x 载入窗口内轮询 |

只影响 **arch35 (A5) + `cache_mode=2 (CYCLE)`** 路径。

---

## 0. 改动要点（便于对照）

- **① slot-major**：`readGen[dbIdx][aiv]`（原 `[aiv][dbIdx]`）。
  AIC 读 `readGenBase + (cubeDbIdx * aivNum + a)`；
  AIV 写 `readGenGm + (c1v1DbIdx * aivNum + GetBlockIdx())`。
  **host 归零区域不变**。
- **③**：`ReadGmByPassDCache`/`WriteGmByPassDCache` → `ReadGmBypassDCache`/`WriteGmBypassDCache`。
- **⑤**：`readGen` 轮询从 `ComputeMm1` 入口挪到 `block_cube.h::ComputeMm1` 的
  **x 载入发射之后、`WaitFlag(MTE2_MTE1)` 之前**，与 MTE2 搬运并行。
  - 轮询参数经 `blockCube_.SetReadGenPoll(base, gen, dbIdx, aivNum)` 传入；
  - 轮询**只做一次**（`needReadGenPoll_`），且在 `isNeedExcute` 分支内。

---

## 1. 编译

```bash
cd <你的 sgl-kernel-npu>
./build.sh -a kernels Ascend950PR_9599      # A5
# 或按你平时的方式；关键是重编 arch35 的 compressor
```

> 只改了 2 个头文件，**增量编译即可**（别跑会 `rm -rf build` 的全量）。

---

## 2. 精度测试（必做，兜底正确性）

### 2.1 单算子测试

```bash
cd <你的 sgl-kernel-npu>
python -m pytest tests/python/sgl_kernel_npu/test_compressor.py -x -q
```

**重点用例**：
- `test_ring_real_c4_batch256_multi_round` —— **256 batch 多轮 ring**（当初就是它逼出 readGen 的）；
- MTP 部分接受（speculative verify）；
- graph capture/replay。

**必须全过**。任一失败 → **回滚**。

### 2.2 sglang 端到端

```bash
# 按你平时起服务的方式，跑一段 prefill + decode（含 MTP）
# 用固定的 prompt/参数，方便前后对比
```
- 看**输出是否与改动前一致**（同 prompt 的 logits/greedy token 一致）；
- 建议连续跑**多轮**（让 ring 跨圈复用充分）。

---

## 3. 性能测试（决定"留不留"）

### 3.1 用 sglang profiler 比 compressor op 耗时

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/prof_after
# 起服务 → 触发 profiling → 导出 trace
# trace 里找 compressor op（npu::compressor）的耗时
```

**对照**：用**改动前**的版本同样跑一次（`/tmp/prof_before`），比 compressor op 的耗时（多轮取中位）。

- 如果 ⑤ 的收益真的存在，`after` 应 ≤ `before`；
- **差值 ≈ 0 或更差 → 回滚 ⑤**（收益没兑现）。

### 3.2 （可选）内核内量自旋占比

在 `compressor_block_cube.h` 的轮询前后加 `AscendC::GetSystemCycle()`，累计写到 GM（或直接打印），
看 `自旋 cycle / 总 cycle`。这是最直接的证据。

---

## 4. 判定与回滚

| 精度 | 性能 | 结论 |
|---|---|---|
| 过 | after ≤ before | **保留 ①③⑤** |
| 过 | after ≈ before（无收益）| **保留 ①③，回滚 ⑤** |
| 挂 | — | **回滚 ⑤**；若仍挂 → 回滚 ①③ |

### 回滚

```bash
cd <你的 sgl-kernel-npu>
git checkout -- csrc/compressor/op_kernel/arch35/compressor_block_cube.h \
                csrc/compressor/op_kernel/arch35/compressor_kernel.h
```

或只回滚 ⑤：把 `compressor_kernel.h::ComputeMm1` 里的 `SetReadGenPoll` 换回原来的直接轮询，
并把 `block_cube.h` 的 `SetReadGenPoll`/成员/轮询删掉。

---

## 5. 红线（改的时候别踩）

1. **轮询必须在首次 `CopyOutMm1Res`（写槽）之前** —— 现在写槽在**最后一次 K 迭代**，
   轮询在**第一次 K 迭代**，余量很大；
2. **轮询必须在 `CopyXGmToL1` 发射之后**（否则没重叠）；
3. **只轮询一次**（`needReadGenPoll_`）；
4. **本核 2 个 AIV 必须照常轮询**（flag 表达不了 generation，别省）。

---

## 6. 备注

- ⑤ 的**正确性**依赖"标量按序发射 + 编译器不跨 `while` 重排"；逻辑上成立，
  **用 §2 的精度测试兜底**。
- ⑤ 的**收益**上限是"轮询被 x 载入掩盖"，量级几十 ns/槽；**若 profile 显示轮询占比本来就小，则 ⑤ 无意义，回滚即可。**
