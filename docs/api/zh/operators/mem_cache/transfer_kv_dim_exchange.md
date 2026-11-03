# transfer_kv_dim_exchange

HiCache 使用此接口在按页排列的主机内存和按层排列的 NPU 缓存之间拷贝选定的 KV 页。公开入口为 `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`，拷贝提交到当前 NPU 流。已在昇腾 910B4（A2）及 CANN 9.1.0 上验证。

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

`flags=None` 时默认使用 `FAST2D`。在导入前设置 `SGL_KERNEL_NPU_COALESCE_KV_H2D=1` 可让 H2D 选用 `FAST2D_COALESCED_H2D`。两端页号同时连续时，原生实现可合并稠密的纯 K 拷贝。显式 `FAST2D`、K/V 对及 D2H 均沿用逐页拷贝。

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
