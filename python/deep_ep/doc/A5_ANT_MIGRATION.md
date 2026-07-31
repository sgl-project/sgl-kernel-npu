# A5 蚂蚁搬家（Normal Long Sequence）

## 实现范围

A5 Normal 模式支持把长序列切成多轮 Dispatch/Combine。每轮最多处理
`per_round_tokens` 个本地 token，Notify 的中间数据和八类输出按批次在 UB
中计算，不再按总轮数一次性分配 UB。

配置约束：

- `DEEPEP_NORMAL_LONG_SEQ_ROUND`：`[1, 256]`
- `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS`：`[32, 8192]`
- 两者必须同时设置，乘积不得超过 `131072`
- 本地 token 数不得超过 `round * per_round_tokens`
- `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ`：只能设置为 `0` 或 `1`

## A5 独立实现

A5 不复用 A3 Kernel 类：

- `NotifyDispatchA5` 按专家规模选择 `batchRounds=1/8/16/32`，并在
  `AssembleSendData`、`BuildTotalRecvTokens`、`BuildRecvCount`、
  `BuildRecvOffset`、`BuildMaxBs`、`BuildRecvTokenPerExp`、
  `BuildExpGlobalOffset`、`BuildsrcRankInExpOffset` 和
  `BuildRInSrcrankOffset` 中分批处理轮次。
- `CamMoeDispatchNormalA5` 每轮只加载当前轮的 `rInSrcrankOffset`，
  并用 A5 round-state slot 做跨 Rank 同步。
- `CamMoeCombineNormalA5MultiRound` 由 TilingKey `15001` 选择，每轮只处理
  一个 token 切片，使用独立的 invocation/round ping-pong 状态。
- A5 Host 容量校验按完整 HCCL window 减去 4 MiB MTE 状态区后的可用数据
  window 执行；Kernel 收到完整 window 大小后使用同一 4 MiB 预留量。

Dispatch 使用从 `450 KiB` 开始的两个 round-state slot，Combine 使用从
`458 KiB` 开始的两个 slot。Notify、Dispatch、Combine 的 invocation
magic 使用彼此独立的 A5 状态区偏移。

## 构建

```bash
bash build.sh -a deepep Ascend950
```

## A5 真机回归

默认回归包含单轮、整轮、尾轮不满、跨 32 轮 UB batch，以及真实执行 256
轮的边界用例：

```bash
bash tests/python/deepep/run_a5_ant_migration.sh
```

增加量化模式：

```bash
A5_ANT_RUN_ALL_QUANT=1 \
  bash tests/python/deepep/run_a5_ant_migration.sh
```

增加 Notify `batchRounds=32/16/8` 的专家规模矩阵：

```bash
A5_ANT_RUN_EXPERT_MATRIX=1 \
  bash tests/python/deepep/run_a5_ant_migration.sh
```

可以按机器拓扑覆盖进程数和 HCCL buffer：

```bash
A5_ANT_NUM_PROCESSES=8 HCCL_BUFFSIZE=2300 \
  bash tests/python/deepep/run_a5_ant_migration.sh
```

仅做快速冒烟测试时，可跳过真实 256 轮边界：

```bash
A5_ANT_RUN_BOUNDARY=0 \
  bash tests/python/deepep/run_a5_ant_migration.sh
```

所有 Rank 必须使用相同的三个长序列环境变量。Dispatch 和 Combine 应配套
开启多轮，除非测试目标仅为 Dispatch 向后兼容。
