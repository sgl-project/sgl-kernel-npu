# A5 / CANN 9.2 图捕获后清缓存报 107000：排查与修复说明

更新日期：2026-11-10。

本文记录 kernel PR [#939](https://github.com/sgl-project/sgl-kernel-npu/pull/939) 中 compressor、SwiGLU 适配器的图捕获问题。代码对照基线为 `8ccf5bebd3`，修复版本为 `3182a4f34f`，运行环境报告为 torch / torch-npu 2.13.0、A5、CANN 9.2。

## 结论及证据边界

当前最有力的根因解释是：迁移后的适配器让后台队列回调持有 Tensor；回调在后台线程销毁时，可能触发与捕获线程不同的内存回收路径。核对的 torch-npu 2.13 源码依据释放线程的当前 stream 判断捕获状态，却依据内存记录的 stream 插入回收事件。两者不一致时，存在将捕获 stream 上的事件按普通事件处理的路径。

需要区分三个层面的证据：

- **源码已确认**：回调持有 Tensor、独立释放线程、线程局部的 current stream、分配器的捕获判断与事件插入方式。
- **运行对照已报告**：修复前开启任务队列失败；关闭队列通过；加入两个修复提交、恢复队列后通过。
- **尚未直接确认**：具体哪个 Tensor 对应第一条失败事件，以及服务器安装的 torch-npu 二进制是否与本文核对的公开源码完全一致。

因此，这是对迁移后接入路径的兼容问题定位，不是“third-party 打包必然有问题”或“所有 torch-npu 2.13 环境必然失败”的结论。

## 1. 原始现象

服务在目标图捕获后的显存统计阶段调用 `empty_cache()`，错误路径为：

```text
capture_decode_graph
  -> get_available_gpu_memory
  -> torch.npu.empty_cache
  -> NPUCachingAllocator::synchronize_and_free_events
  -> aclrtSynchronizeEvent
  -> error 107000
```

驱动报告的具体原因是：

```text
The event (event_id=-1) in the stream bound to the model is not supported.
```

这个错误说明同步了不受该接口支持的图绑定事件。仅凭 `event_id=-1` 不能断言是 Tensor 指针损坏；报错位置在清缓存，也不等于算子计算在这里出错。

## 2. 为什么迁移前能启动，迁移后却出现问题

这轮演进包含 third-party 打包，也包含原生算子接入 CANN/ACLNN 框架。它们不能混为同一个变更。

旧 compressor 和 SwiGLU 使用 `EXEC_KERNEL_CMD`。该宏也会调用 `RunOpApi`，所以差别不是“旧实现没有任务队列”。[旧启动宏](../../utils/torch_helper.h)先用 `data_ptr()` 将 Tensor 转成裸地址，再让回调保存转换后的地址与标量参数。

新的 [compressor 适配器](torch_adapter.cpp)和 [SwiGLU 适配器](../../activation/swiglu_group_quant/torch_adapter.cpp)则显式调用 `record_stream`，并让回调按值持有 Tensor、workspace 和 ACL 描述符。

| 行为 | 旧启动宏 | 新适配器，修复前 |
| --- | --- | --- |
| 算子提交入口 | `ACLRT_LAUNCH_KERNEL` | ACLNN query / execute |
| 回调保存的 Tensor 参数 | 转换后的裸地址 | Tensor 对象的一份引用 |
| 是否因此增加 Tensor 引用计数 | 否 | 是 |
| 回调销毁的影响 | 丢弃地址和参数 | 释放 Tensor 引用，可能触发内存回收 |

保存 Tensor 引用不会复制整块 NPU 数据。它会延长存储的生命周期，避免后台任务尚未提交时内存已经失效。但最后一份引用也可能因此转移到后台线程释放。

workspace 是算子运行的临时工作区，在适配器中用 Tensor 管理；它不等同于持久的模型权重或 KV cache。

这两份适配器分别来自主仓的 compressor [#982](https://github.com/sgl-project/sgl-kernel-npu/pull/982) 和 SwiGLU [#929](https://github.com/sgl-project/sgl-kernel-npu/pull/929)。在修复前的 `8ccf5bebd3`，两份适配器与当时 PR 主仓基线 `bef12c05d2` 对应文件一致；迁移提交 `0dea5e8c70` 开始复用新框架路径。这个对照只针对适配器文件，不表示整个 PR 的所有 kernel 文件都与主仓相同。

旧实现能运行，不能证明新的提交和释放方式在同一环境中也必然兼容；裸指针本身也不保证内存生命周期，不能简单通过删除新适配器的 Tensor 引用来修复。

## 3. 从后台回调释放到 107000 的可疑路径

假设捕获线程 A 的当前 stream 是 S，S 正在捕获图。

1. 适配器分配 workspace，调用 `workspace.record_stream(S)`，然后把持有 workspace 引用的回调送进任务队列。
2. 后台执行线程将 ACLNN 调用提交到 S。随后，回调交由独立的后台释放线程 C 处理。
3. C 清空回调，释放其中的 Tensor 引用。如果这是最后一份引用，就调用内存分配器的 `free()`。
4. 分配器先检查 C 的当前 stream。它可能是非捕获的默认 stream D，而不是 A 的 S。
5. 判断结果为非捕获时，分配器进入 `insert_events(block)`，但事件记录到的是 `block->stream_uses` 中的 S。
6. 如果事件在 S 捕获结束前实际提交，就可能成为图绑定事件。之后 `empty_cache()` 按普通回收事件调用 `aclrtSynchronizeEvent`，触发上述错误。

这里存在一个时间窗口，不是每次小规模调用都一定触发。若回调销毁时捕获已经结束，或者捕获线程仍持有最后一份引用，就可能走另一条路径。

对应源码证据：

- [后台释放线程](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/core/npu/NPUQueue.cpp#L973)调用释放队列；[回调的 Release](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/framework/NPUDefine.cpp#L39)清空 `customHandler`。
- [current stream 是线程局部状态](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/core/npu/NPUStream.cpp#L86)。
- [free 的分支选择](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L1551)依赖 [is_capture_context](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2350)，后者查询调用线程的当前 stream。
- [insert_events](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L3409)使用内存记录的 stream。后续事件提交虽然会切换到目标 stream，但已经晚于前面的捕获状态判断。
- [synchronize_and_free_events](https://github.com/Ascend/pytorch/blob/v2.13.0-26.2.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L3342)直接同步待回收事件，其报错行与提供的调用栈对应。

正常的捕获释放路径会先将 block 放入延迟列表，之后移除仅在图内记录的 stream 使用关系，再处理其余回收事件。可疑路径绕过了这一步。

## 4. torch-npu 版本差异的具体来源

对照的公开版本为 `v2.10.0` 和 `v2.13.0-26.2.0`。不能据此推定所有带相同版本号的定制 wheel 都具有完全相同的实现。

在 [上游提交 796d887076fe64a005585039060646fff7a9e86e](https://github.com/Ascend/pytorch/commit/796d887076fe64a005585039060646fff7a9e86e) 中，分配器将内存池分流作用域与实际图捕获状态解耦，并改变了 `free()` 的判断：

```diff
- if (C10_UNLIKELY(!captures_underway.empty())) {
+ if (C10_UNLIKELY(is_capture_context())) {
```

旧判断更保守：设备上存在相应作用域就延迟处理。新判断会进一步查看当前线程的 stream，允许其他 stream 上的普通操作继续回收。这有正常的内存池管理目的，但对“在另一个线程释放捕获资源”的场景引出了上述检查对象不一致的条件。

该版本差异是问题可能对运行栈敏感的源码依据，不是服务器历史运行环境已被完整核实的证据。

## 5. 已有运行日志能证明什么

一次失败报告共有 122 块待释放内存，总计 2,252,322,816 字节，分配调用栈归属如下：

| 分配调用位置 | 块数 | 大小 |
| --- | ---: | ---: |
| compressor | 41 | 1704.58 MiB |
| SwiGLU activation | 54 | 435.81 MiB |
| grouped matmul | 27 | 7.59 MiB |

这些块都属于同一个图池 `[0,1]`。重点候选包括适配器的 workspace，以及 SwiGLU 回调持有的上游 Tensor。

但 `active_pending_free` 只表示已经请求释放、尚未完成回收，不能单独证明泄漏或非法事件。日志中的 allocation 调用栈也不能直接当成具体 Tensor 变量名。该报告未保存失败 event 与 block 地址的一一对应关系，因此无法从这张汇总表恢复第一条失败事件的确切归属。

运行者报告的整网对照为：

| 代码与设置 | 结果 |
| --- | --- |
| 修复前，`TASK_QUEUE_ENABLE=1` | 清缓存时报 107000 |
| 修复前，仅改为 `TASK_QUEUE_ENABLE=0` | 启动通过 |
| 包含两个修复提交，恢复 `TASK_QUEUE_ENABLE=1` | 启动通过 |

这支持问题与后台提交及资源释放的时序相关。两个适配器同时修改，尚不能由此证明它们各自都可以独立触发错误。启动通过也不等于已经完成长期稳定性、全部模型精度或性能验证。

## 6. 两个修复提交分别解决什么

### ec33b52：捕获期间在捕获线程提交 ACLNN

提交：[ec33b52b4bad93bd4870d030d041540a9067b128](https://github.com/lilDss/sgl-kernel-npu/commit/ec33b52b4bad93bd4870d030d041540a9067b128)。

捕获期间先排空已经入队的主机提交，再直接调用 `launch()`；普通执行继续使用 `RunOpApi`。这样可以保留输入生产、workspace 清零等操作的提交顺序，并使适配器回调持有的引用在捕获线程释放。

排空的是主机提交队列，不是等待整个 NPU 计算完成。算子仍提交到原来的 stream，`record_stream` 和 ACLNN 错误检查继续保留。这个提交没有修改 kernel 计算公式或 ring 历史读写逻辑。

对应的单算子补充测试增加了重复捕获调用，并在首次 replay 前执行 `empty_cache()`，之后更新输入、replay 并检查结果，用于覆盖服务失败时的阶段。

### 3182a4f：修正捕获查询接口的二进制兼容性

提交：[3182a4f34f6583c50631946ed9bff926de00b186](https://github.com/lilDss/sgl-kernel-npu/commit/3182a4f34f6583c50631946ed9bff926de00b186)。

上一提交使用的内联包装 `currentStreamCaptureStatus()` 会引用 `IsContextInitialized()`。服务器的 torch-npu 动态库未导出该符号，导致新适配库加载失败。

两个调用点改用显式导出的 `currentStreamCaptureStatusMayInitCtx()`。执行到这里时，适配器已经设置设备、分配 NPU Tensor 并取得当前 stream，具备查询所需的上下文。图捕获状态检查仍然存在。

完整修复后的核心形态如下：

```cpp
if (c10_npu::currentStreamCaptureStatusMayInitCtx() ==
    c10_npu::CaptureStatus::Active) {
    TORCH_CHECK(stream.stream() != nullptr, "failed to drain the host task queue");
    launch();
} else {
    at_npu::native::OpCommand::RunOpApi("operator_name", launch);
}
```

构建过程的 wheel 内容校验只验证产物与本次暂存内容一致，不会证明所有动态符号都能在目标环境解析。加载错误与前述 107000 是两个不同阶段的问题。

## 7. 对照版本和仍需补齐的证据

| 对照分支 | 固定提交 | 用途 |
| --- | --- | --- |
| `test/pr939-before-graph-capture-fix` | `8ccf5bebd3` | 不含这两个修复，进行环境对照 |
| `test/pr939-graph-capture-fix` | `3182a4f34f` | 包含两个修复，进行代码对照 |

切换源码分支后还需安装对应构建产物；只切 Git 分支不会改变已安装的 wheel。对照时应保持同一份 SGLang、模型、启动参数和任务队列设置，并记录运行栈版本。

目前缺少的直接证据是：

1. 服务器安装的 torch-npu 对应的源码 commit。可只读获取，无需启动模型：

   ```bash
   python3 - <<'PY'
   import importlib
   version = importlib.import_module("torch_npu.version")
   print("torch_npu:", version.__version__)
   print("git_version:", version.git_version)
   PY
   ```

2. 失败 event、对应 block 地址、最后释放该 block 的线程及其 current stream 的运行时关联。现有汇总日志不足以补出这一关系。

本记录的运行结果来自服务器反馈；源码检查和本地主机测试不能替代 NPU 侧的逐事件验证。目前可以定位出明确的可疑源码路径，并有整网对照支持修复有效，但不能将其表述为已经完成每一个失败事件的直接取证。
