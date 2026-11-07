# Q4 LIM 尾块修复与验证状态（2026-10-29）

**交付状态：已通过算子正确性回归；真实整网精度待验证；整网性能未验证。**

本说明仅覆盖 Q4 淘汰候选尾块导致的 slot0 重复映射问题。
不能据此宣称完整 GLM-5.2、MTP rollback 或端到端性能已经通过。

| 检查项目 | 状态 | 覆盖范围与结果 |
| --- | --- | --- |
| 原始故障复现的修复 | 已通过 | B1 eager、B16 eager、B16 NPU Graph，均退出 0 |
| 既有算子正确性测试 | 已通过 | 15 passed，0 failed，0 skipped |
| 新增尾块回归 | 已通过 | 固定 SO 的 eager、Graph 共 2 passed |
| 新回归对旧实现的区分能力 | 已确认 | 旧 SO 的 eager 在第 5 步出现 12289 个 resident 条目，预期 12288；退出 1 |
| 合成 TP1 manager + CopySFA 检查 | 已通过限定范围 | cold、warm、改变 query 的 warm，抽样 host/HBM payload 逐位一致 |
| 真实模型长序列数值 | 待验证 | 尚无完成的真实模型数值 capture |
| 完整 MTP rollback | 待验证 | 原始三组复现没有真正的长度回退 |
| 整网 TPOT、吞吐与 profiling | 未验证 | 本修复尚无正式端到端性能结果 |
| 本修复是否保持性能 | 未验证 | 不从正确性回归推断性能无劣化 |

## 原因与调用链

```text
LI 打分和 TopK
  → Q4 miss union 与淘汰候选构造
  → CompareScalar / Select 屏蔽无效 slot 与受保护 source
  → 尾部屏蔽、候选排序
  → 淘汰旧 owner、分配 slot、更新 resident map
  → metadata 与真实 KV 搬运
  → CopySFA
```

故障发生在 Q4 淘汰候选构造的 partial chunk。现场长度 65537 时，
最后一个 512 元素块只有 1 个有效元素。安装的 CANN 9.1 接口要求相关
CompareScalar 的 count 对应 256 字节对齐；原代码将 valid=1 用作向量
Compare/Select 的 count，不满足这一要求。

诊断中，一个 slot 为 INT32_MIN 的无效候选 source65536 在屏蔽后仍保留
有效排序 key，并被解码为 slot0 的 victim。随后真正的 slot0 owner 未被
清除，形成两个 source 共用 slot0，resident 条目从 12288 增至 12289。

修复沿用相邻通用 route 路径的做法：partial chunk 先初始化完整 scratch，
向量计算和 Compare/Select 处理完整 512 元素；GM 仍只读取 valid 元素，
排序前仍通过 PadEvictKeys 排除逻辑尾部。未修改 LI 打分、TopK、淘汰策略、
算子 ABI、workspace 大小，也未移除正确性断言或增加 kernel launch。

实现见 [Q4 淘汰候选构造](../csrc/fused_lightning_indexer_manage/op_kernel/upstream/fused_lightning_indexer_manage_union.h)，
新增回归见 [test_lim_mtp_full_visible.py](../tests/test_lim_mtp_full_visible.py)
中的 `test_lim_q4_partial_chunk_preserves_live_batch_slot_ownership`。

## 实测范围

验证环境为 Ascend A3、CANN 9.1.0、Python 3.12.13、torch_npu
2.10.0.post6。各算子用例在独占的单个 die 上执行，B16 表示算子 batch，
不代表 TP16 整网已经通过。

新增回归使用 Q4、C12288、K2048、BF16，17 行 map（保留 row0），
source capacity 66944，实际长度 65536/65537。覆盖 cold、warm、混合
cold/warm、query 改变、尾部淘汰、resident refresh，以及 eager/Graph。
检查 map 的 slot 唯一性、TopK 因果可见性和 miss source/slot 计划的一致性。

既有测试的实测结果包括：

- LIM 选择的 source/slot payload mismatch 为 0，future violation 为 0。
- 与 vendor LI 对账的长度为 16385 和 64000，每个长度四路 TopK 均为
  2048/2048 集合重合。
- CopySFA refresh 对 CPU FP32 参考的 max_abs 为 0.0009821057，
  RMS 为 0.0001291162；精确 K 用例 max_abs 为 0.000627473，padding 误差为 0。
- 合成 TP1 真实 manager 检查的抽样 payload bit mismatch 为 0，
  attention max_abs 最大为 0.0006703734。该用例的 index pool 为 stub、
  rank 为 mock，两层被安排复用同一 plan，不代表真实模型拓扑。

在已分配设备、初始化 CANN 环境并加载重新构建的固定版本后，可运行：

```bash
python3 -m pytest -q -s -ra \
  tests/test_lim_mtp_full_visible.py \
  tests/test_lim_mtp_vendor_parity.py
```

## 可追溯版本

| 产物 | SHA256 |
| --- | --- |
| 固定版本 union 头文件 | `751e8d799b852858214e12065dfb8796b2fa620ce1500ef0298e4059b3531f2d` |
| 含新增回归的测试文件（LF） | `04bd9c27aba0cae15c28f1eec9fc56bb041997990be4b22379878a7a038e9f81` |
| 实际验证过的固定 SO | `3a0d3e4cba06c7b4dfb327eb309ea261efdada5bdbcf39fe9ab897c2e5bcd0e4` |
| 用于故障对照的旧 SO | `61c08679342c8dd0b2a7acf33d89f5a1230701d9baffd4f5b764b28c198bbd85` |
| 固定 SO wheel | `4187e23c5fe28f81013e5c9b4771570200397750b0344bc632f818580d39d2dd` |

验证进程检查了实际加载的 SO 路径与 SHA256。源码的推送不代表该预构建
wheel 已发布，也不代表其他环境重新构建后的二进制哈希相同。
真实模型的旧正确性尝试被宿主重启中断，没有完成结果；因此本次提交的
正确标注是“算子正确性已验证，真实整网精度待验证，整网性能未验证”。

本次源码分支基于 `b8b6ac20ef342736e218f23282c0a25c2c57a599`，上述
修复头文件和新增测试与实测文件逐字节一致。本次推送没有重新构建这个
Git commit，也没有重新运行 NPU 测试；表中结果引用已冻结的固定 SO。
与原始构建的冻结输入比较，其余 925 个 csrc/include/cmake 文件中，
以下四个既有 C++ 文件的字节哈希不同，尚未取得原始内容完成差异归因：

- `csrc/fused_lightning_indexer_manage/op_host/fused_lightning_indexer_manage.cpp`
- `csrc/fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage.cpp`
- `csrc/fused_scatter_copy_sparse_flash_attention/op_kernel/fused_scatter_copy_sparse_flash_attention_combined.cpp`
- `csrc/pytorch_extensions.cpp`

这四个文件属于既有分支基线，本提交未修改；不能把头文件相同解释为
整个新 checkout 与已测试二进制的所有构建输入相同。
