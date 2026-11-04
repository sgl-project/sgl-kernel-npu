# PR #939：A5 + CANN 9.1 测试核对清单

核对日期：2026-11-04。范围为此次迁移的 13 个公共 API，不代表整个仓库的所有算子。

## 选择依据

- PR #47438 的 `_sparse_attn_ops()`：A5 走 KV-quant shared-KV，A3 走普通 shared-KV。
- 用户 A5 SDK 的符号扫描：`custom_transformer/libcust_opapi.so` 有 `aclnnKvQuantSparseAttnSharedkv` 及 metadata/WorkspaceSize；未找到普通 `aclnnSparseAttnSharedkvMetadata`。
- CANN 9.1 experimental KvQuantSparseAttnSharedkv 文档：BF16 Q、FP8 KV、640B 行、64 RoPE 特征、64 元素量化组、E8M0 scale，TND/PA_ND。
- CANN 9.1 experimental QuantLightningIndexer 文档：A5 FP8，权重及反量化 scale 为 FP32；此处固定 64 query heads、1 KV head、128 head dim、512 Top-K。
- 旧版 MoE 部署实际报错：`renorm only supports 0`。测试与参考均使用 renorm=0；A5 保留 norm_type=0/1/2。官方 9.1 文档的 norm_type=2 产品约束为 Ascend 950 系列。
- Compressor 沿用 A5 `custom.compressor` 的 fused RMSNorm/RoPE 契约；测试本次限定 TH，涵盖两种 cache mode、三个压缩形状、批处理和流式状态。BSH 明确不属于本次验证范围。

## API 与测试

| API | 9.1 A5 实际生产绑定 | 本次处理 |
|---|---|---|
| mhc_pre | custom.npu_hc_pre | 纳入；保留原生输入适配 |
| mhc_post | custom.npu_hc_post | 纳入；A5 包装增加 batch 维后调用 |
| inplace_partial_rotary_mul | custom.inplace_partial_rotary_mul | 纳入；独立输入缓冲区比较 |
| mixed_quant_sparse_flash_mla | custom.npu_kv_quant_sparse_attn_sharedkv | 纳入；640B/E8M0，匹配 writer |
| mixed_quant_sparse_flash_mla_metadata | custom.npu_kv_quant_sparse_attn_sharedkv_metadata | 纳入；通过独立参考 attention 验证 opaque tiling |
| kv_compress_epilog | npu.kv_compress_epilog，来自本仓库 libsgl_kernel_npu.so | 纳入；quant_mode=2、640B，比较有效字节 |
| quant_lightning_indexer | custom.npu_quant_lightning_indexer | 纳入；FP8 数据在 CPU 转换后按字节传入 NPU |
| quant_lightning_indexer_metadata | custom.npu_quant_lightning_indexer_metadata | 纳入；通过配套 Indexer 验证 |
| indexer_quant_cache | custom.indexer_compress_epilog | 纳入；FP8+FP32 scale，CPU 独立参考 |
| compressor | custom.compressor | 纳入 TH；输出及整个状态缓存比较；BSH 显示 EXCLUDED |
| moe_gating_top_k | custom.npu_moe_gating_top_k | 纳入；legacy renorm=0；保留 A5 支持的三个 norm_type |
| sparse_flash_mla | custom.npu_sparse_attn_sharedkv | EXCLUDED：A5 模型路径使用量化版本 |
| sparse_flash_mla_metadata | custom.npu_sparse_attn_sharedkv_metadata | EXCLUDED：当前 A5 9.1 配套库缺少此符号 |

“纳入”表示分支和输入契约核对后选择进行验证，不表示已经在用户服务器通过。Python schema 存在只证明绑定注册，底层 aclnn 库、dtype、输入约束与数值结果仍以实际运行结果为准。SDK 自带的 `aclnnSparseFlashMla` 与旧 shared-KV 符号不能因名字相近直接替换。

## 运行与结果

```bash
# 在最新 PR 检出仓库中安装包含 FP8 reference 修复的 wheel 后：
python3 tests/python/operators/run_pr939_a5_cann91.py . --device 0
```

也可以将 `test_cann91_a5.py` 复制到服务器：

```bash
python3 test_cann91_a5.py /home/x00937188/sgl-kernel-npu --device 0
```

默认只测精度，禁止跳过精度。每个测试文件独立进程；不使用 -x；默认每文件 900 秒上限，60 秒打印等待栈。超时会结束对应进程组并继续下一个文件。测试进程正常错误退出也继续后续文件。

```bash
python3 test_cann91_a5.py . --device 0 --timeout 1800 --output ./my-a5-results
```

输出目录包含：

- `environment.log`：Python、CANN、芯片、实际安装包路径、每个绑定的 schema/缺失信息。
- 每组 `.log` 和 `.xml`：原始日志与 JUnit 结果。
- 每组 `.jsonl`：实时记录，进程崩溃/超时后仍能还原已完成用例。
- `summary.txt`：每个 test node 最终状态，终端也会显示。
- `summary.tsv` / `summary.json`：可筛选的详细结果和失败原因。

状态包括 PASS、FAIL、ERROR、SKIP、TIMEOUT、NOT_RUN、EXCLUDED。只有实际完成精度用例的 PASS 算通过。文件中尚未执行的用例在超时后显示 NOT_RUN，不会被算作 PASS；全部跳过也不会给出整体成功。

旧检出版本的两个 Attention API 可能在一个文件中完成配套验证；最新版已拆成独立文件，runner 会自动选择。测试文件数量与 API 数量、参数化用例数量分别统计。

## 本地证据

macOS 无 NPU。检查范围仅包含 CPU 参考回归、数据准备、脚本状态汇总和格式，不能表述为 A5 原生精度/性能通过。

- CPU 参考/契约与数据检查：189 passed，9 skipped。
- Summary runner 实际 pytest 进程验证：PASS、FAIL、SKIP、setup ERROR、teardown ERROR、TIMEOUT、NOT_RUN 均按期望归类。

## 来源

- https://github.com/sgl-project/sglang/pull/47438
- https://gitcode.com/cann/ops-transformer/blob/9.1.0/experimental/attention/kv_quant_sparse_attn_sharedkv/README.md
- https://gitcode.com/cann/ops-transformer/blob/9.1.0/experimental/attention/quant_lightning_indexer/README.md
- https://gitcode.com/cann/ops-transformer/blob/9.1.0/experimental/attention/compressor/README.md
- https://gitcode.com/cann/ops-transformer/blob/9.1.0/moe/moe_gating_top_k/README.md
