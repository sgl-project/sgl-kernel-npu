# DeepSeek-V4.1 CANNBot-DSL 算子

七个源码入口位于 `implementation/cannbot-dsl/`：K、QLI/QSLI 及其 metadata
放在 `indexer/`，MLA 及其 metadata 放在 `attention/`。`UPSTREAM.json` 记录
上游路径、哈希和回移补丁，同目录保留 CANN 许可证。K 投影尾块修复继续保留：
backing 按 16 行对齐，逻辑视图仍只有 T 行。

## 安装与执行

从干净源码树使用仓库根目录现有构建入口。构建会打包 Python implementation
目录，包含 DSL 源码和许可证，无需新增 CMake 目标或修改公共构建脚本。
不包含第三方 provider 的 wheel 示例：

```bash
python3 setup.py --arch-version 3510 --without-third-party -j8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pip install 'cannbot-dsl==0.8.0'
```

部署完整 SGLang 时，按需要选择第三方 provider。独立的 compressor 仍依赖
`cann_ops_transformer`，上面的示例不提供它。DSL runtime 显式安装；本次迁入
不新增 wheel extra，也不在打包阶段预编译七个 JIT 算子。

目标组合是 A5 / ARCH 3510、CANN 9.2.0、DSL 0.8.0 及匹配的 Linux torch/torch_npu。
上游源码标注 CANN 9.2.1。固定源码的 DSL API 并不兼容所有更新版本，升级时需要
同时核对源码和 runtime。`auto` 选择 `cannbot_dsl`，图捕获前预热实际形状。

四个计算入口提供 `torch_native`：独立的 CPU Torch 参考，结果复制回输入设备，
保持缓存和输出的原地更新语义，用于精度核对，不用于性能执行或图捕获。
三个 metadata 是设备调度计划，没有唯一的数学答案，因此不注册伪 native 别名。

## A5 精度与性能测试

测试沿用公共接口框架，覆盖 decode T=1、DSPARK 静态验证 T=6 和必要边界。
输入按模型维度生成，不是现场张量 dump。K 的压缩 latent 宽度为 512，index
宽度 128、RoPE 宽度 64；QLI/QSLI 使用 32 头、4096 个 K、TopK 512；MLA 使用
64 头、头维度 512，原始/压缩稀疏长度为 128/512。

```bash
python3 -m pytest \
  tests/python/operators/indexer/test_indexer_prologue_k.py \
  tests/python/operators/indexer/test_quant_lightning_indexer.py \
  tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py \
  tests/python/operators/indexer/test_quant_sparse_lightning_indexer.py \
  tests/python/operators/indexer/test_quant_sparse_lightning_indexer_metadata.py \
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py \
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py \
  --npu-device=0 --skip-acc-check=false \
  --prof-dir=results/dsv41 --csv-path=results/dsv41.csv -v -s -x -rs
```

默认只对代表形状采集性能；加 `--enable-perf=false` 仅跑精度。现有工具记录设备
任务耗时，单位微秒。这是算子耗时，不代表服务 TPOT 或已测得的加速比。

K 和 MLA 使用独立分配的可写输出做比较。QLI/QSLI 检查分数、索引有效范围和
TopK 边界，允许同分项排序不同。保留 T=1/T=17 的 K 分配守卫，防止尾块越界
回归。metadata 测试原地更新长度后比较 graph replay 与新生成的 eager 调度；
配套计算算子的独立参考负责检查数值语义。

CPU 参考检查不能证明 NPU 精度、图执行兼容性或性能。修改后需在 A5 上运行以上
测试并保存设备结果。
