# A5 单卡 CI 环境

2026-11-04 核查：七个 DSL 入口的设备验证可使用 Ascend/sglang 的
`linux-amd64-a5-1` runner。该标签的单卡含义由环境提供方确认；
它是否向其他仓库开放，不能由标签名称推断。

## 镜像

```yaml
runs-on: linux-amd64-a5-1
container:
  # CANN 9.2.0 / Ascend 950 / Ubuntu 22.04 / Python 3.12
  image: quay.io/ascend/cann:9.2.0-950-ubuntu22.04-py3.12@sha256:5c4fa2bb96c678d865eda4e50c8c4310fb013c3452c7a081afcb9c86ed318730
```

该 tag 已出现在上游
[发布 workflow](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/build_and_release.yml)
的 CANN/hardware 矩阵中。已读取 Quay Registry 的实际 manifest，含 Linux
amd64 和 arm64；上面的 digest 固定此次核查的多架构索引。
amd64 子镜像 digest 为
`sha256:ff154f82fe5d5252719fd853edb2e9b4376d3656221bc88f93c6fbbbf27e4b82`。

这是 CANN 基础镜像。上游
[依赖安装脚本](https://github.com/sgl-project/sgl-kernel-npu/blob/main/scripts/npu_ci_install_dependency.sh)
为 CANN 9.2.0 选择 PyTorch 2.13.0 和 torch_npu 2.13.0 / Python 3.12。
DSL 依赖另外固定为 `cannbot-dsl==0.8.0`。这个版本选择来源于上游发布配置，
尚未证明它与迁入的七个算子组合通过设备验证。

已确认 tag/manifest 可查询，未在 runner 中拉取镜像或执行 NPU 算子。

## 对接 Single Test workflow

参考 [Ascend/sglang PR #1374](https://github.com/Ascend/sglang/pull/1374/files)，
所查 head 为 `6f74f08352b0b1ac6ff9431d92200a9bbfc373a4`。
其 `.github/workflows/single-test-npu.yml` 使用 `pull_request` 事件，
目标分支包含 `testcases`、`testcases-261015`，路径过滤要求修改这个 workflow 文件。
这是 PR 的 base 分支条件，不是提交者的功能分支名称。
该文件没有 `workflow_dispatch`，不能据此直接使用 `gh workflow run`。

原示例使用四卡、CANN 9.1.0 镜像和整模型 GPQA 测试，且明确加载镜像内的
SGLang。迁入验证应换成上面的单卡镜像，并确保执行自己构建的 kernel wheel：

1. checkout SGLang 测试分支，并单独 checkout 含本次迁入的 sgl-kernel-npu
   **已推送 commit SHA**。当前本地未提交源码不能直接被远端 runner 读取。
2. 安装匹配的 torch/torch_npu、DSL 和构建依赖。先检查 `npu-smi`、SDK 的实际
   版本、Python 版本、`torch.npu.is_available()`，任何不匹配都先报错。
3. 在 kernel 仓库执行[迁入说明](cannbotdsl.zh.md)中的构建、安装与测试命令。
   首轮先构建普通源码 wheel 并验证 JIT 精度/metadata 图重放；Native 全量静态
   导出的编译量较大，应另一个 job 验证，避免混淆编译时间与 kernel 执行时间。
4. 打印安装后的 `sgl_kernel_npu.__file__` 和 build manifest，确认不是镜像旧包。
   显式开启 `CANNBOTDSL_RUN_SMLA_TESTS=1`，保存 pytest JUnit、环境信息及编译日志。
   设备验证不能把缺少 torch_npu 导致的跳过当成功。

这些单算子和 metadata 测试不需要模型权重，也不要求多卡通信。
四卡 runner 留给后续整模型或多卡联调。仓库 runner 权限、PR 首次执行审批及
镜像网络可达性仍以 Ascend CI 的实际配置为准。

`scripts/check_cannbotdsl_ci.py` 提供三个 CI 检查阶段：`environment` 对齐
checkout SHA、CANN、DSL、torch/torch_npu 和 ARCH 3510；`installed` 比较已安装
源码、原生库和本次构建产物；`results` 检查 JUnit 必要用例实际通过。
任意用例跳过、失败或缺失必要覆盖均不能通过最后一关。首轮覆盖两种 K cache
模式、分页/TND indexer、混合/原始 KV MLA 和三个 metadata 图重放；完整边界矩阵
及 Native require 构建验收仍需后续任务。
