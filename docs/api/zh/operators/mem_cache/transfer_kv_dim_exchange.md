# transfer_kv_dim_exchange

HiCache 使用此接口在按页排列的主机内存和按层排列的 NPU 缓存之间拷贝选定的 KV 页。公开入口为 `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`，拷贝提交到当前 NPU 流。原 K-only PR 与 queued-copy 原型分别在昇腾 910B4（A2）及 CANN 9.1.0 上测量过；最新的原型迁移集成尚未重新实机验证。

## 函数签名

```python
transfer_kv_dim_exchange(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None, device_index_k_scale=None,
    host_index_k_scale=None, page_size=128, direction=TransferDirection.H2D,
    flags=None,
)
```

## 参数

| 参数 | 类型和约束 |
| --- | --- |
| `device_indices`, `host_indices` | 展平的 int64 token 索引。两端均按传输顺序提供同等页大小的完整页。 |
| `device_k`, `host_k` | 设备 K 的布局为 `[层数, 设备页数, 页大小, 头数, 宽度]`，锁页主机 K 的布局为 `[主机页数, 层数, 页大小, 头数, 宽度]`。两端使用相同的数据类型。 |
| `device_v`, `host_v` | 相同布局的 V 张量。仅传输 K 时传入两个空张量。 |
| `device_index_k`, `host_index_k`, `device_index_k_scale`, `host_index_k_scale` | 可选的索引和缩放系数张量对。设备端四维 scale 在拷贝前补一个大小为 1 的末维。 |
| `page_size`, `direction` | 每页 token 数必须为正，默认 128。方向默认为 `TransferDirection.H2D`，使用 `TransferDirection.D2H` 回写主机。 |
| `flags` | 显式传入的 `TransferFlag` 优先于环境变量。 |

## 返回值

返回 `None`，原地写入目标页。在主机读取结果前先同步流。

## 昇腾原生后端

`flags=None` 时默认使用 `FAST2D`。在导入前设置 `SGL_KERNEL_NPU_COALESCE_KV_H2D=1` 可让 H2D 选用 `FAST2D_COALESCED_H2D`。当索引位于 CPU 且为 int64 时，包装层将 K、V、index 和 scale 组件统一提交给一个 native 队列调用。非空组件使用五维布局，四维组件补一个大小为 1 的末维。页内 payload 必须连续；外层 layer/page stride 可以包含间隔，例如 packed host pool 的层切片。每对 host/device 的 dtype 和 payload 维度必须匹配；不同组件对可以使用不同 dtype、宽度和层数。

只有 host 与 device 页号同时递增的连续段才合并描述符。重复目的页保留输入覆盖顺序。规划使用实际字节 stride，在提交前校验页对齐、完整 token 页、边界和支持的 pitch。队列路径会拒绝不支持的页内 stride，不会隐式打包。

原生调用通过 Torch-NPU `OpCommand::RunOpApi` 使用当前流，避免在取流时清空主机任务队列。C++ 闭包持有张量到提交结束；Python helper 进一步持有组件视图和索引，直到同一线程后续调用确认完成事件。调用线程应保持存活到复制结束，源缓冲区在流完成前不得修改；完成事件只保障生命周期，不解决并发写入。

显式 `FAST2D` 与 D2H 保持原实现。非 CPU 索引或不支持的张量维数沿用 #947 原有路径。无需 GLM 私有环境变量、动态扩展加载器或 SGLang monkey patch。

## 迁移来源与验证状态

队列实现来自已实测的 GLM-5.2 扩展：`glm52_cache_copy.cpp` 的 SHA256 为 `e19368569eb484d605b57920d57ce677c46814ceb3e82fcc2e2503d7d72a6960`；实测二进制的 SHA256 为 `c0bb68872171ccd5f41b4d9719f846feb6d002670138ff89d0a89c8c88664f71`。

历史原型在任务队列模式 0 和 2 下分别通过 21 个组合、5 组件层切片全字节检查，以及连续 6 代的生命周期/流顺序检查。两轮诊断模型回放均完成 36/36 请求。这些属于历史原型结果，不是最新 PR 集成的验证或性能承诺。native 注册、包装层路由、完成事件的设备选择和允许重复目的页是此次接口适配；本次迁移没有重新执行 NPU 实验。

原 #947 的页顺序/scale 测试继续保留。原型的层切片、生命周期和非法索引用例已移入 `tests/python/sgl_kernel_npu/test_queued_h2d.py`，供后续分别使用 `TASK_QUEUE_ENABLE=0` 和 `TASK_QUEUE_ENABLE=2` 验证。

## 示例

```python
import torch
from sgl_kernel_npu.kvcacheio import TransferDirection, TransferFlag, transfer_kv_dim_exchange

page_size = 128
indices = torch.arange(2 * page_size, dtype=torch.int64)
host_k = torch.ones((4, 1, page_size, 1, 4), dtype=torch.int8).pin_memory()
device_k = torch.zeros((1, 4, page_size, 1, 4), dtype=torch.int8, device="npu")
no_v = torch.empty((0, 0, 0, 0, 0))
transfer_kv_dim_exchange(
    indices, indices, device_k, host_k, no_v, no_v,
    page_size=page_size, direction=TransferDirection.H2D,
    flags=TransferFlag.FAST2D_COALESCED_H2D,
)
torch.npu.synchronize()
assert torch.equal(device_k[:, :2].cpu(), host_k[:2].transpose(0, 1))
```
