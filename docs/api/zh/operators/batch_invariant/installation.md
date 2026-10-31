# batch invariant 算子安装

batch invariant 算子是外部包的轻量封装。Python 模块随 `sgl-kernel-npu`
wheel 一起发布，但需要先安装两个外部组件：

| 组件 | 提供内容 | 作用 |
| --- | --- | --- |
| CANN 算子包（`.run`） | `aclnnReduceSumBatchInvariant` 等 aclnn kernel | 每次调用都依赖 |
| torch extension（`.zip`） | `batch_invariant_ops` 包与 `batch_invariant_ops_lib` | 提供 `torch.ops.batch_invariant_ops.*` |

## 下载地址

CANN 算子包，按设备和 CPU 架构选择其一：

- Ascend 910B，x86_64：<https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run>
- Ascend 910B，aarch64：<https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-910b-1.0.0-linux.aarch64.run>
- A3，aarch64：<https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run>
- A3，x86_64：<https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-A3-1.0.0-linux.x86_64.run>
- Ascend 950，x86_64：<https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-950-1.0.0-linux.x86_64.run>
- Ascend 950，aarch64：<https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-950-1.0.0-linux.aarch64.run>

torch extension 源码包：

- <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/batch_invariant-torch_ops_extension-1.0.0.zip>

## 安装 CANN 算子包

下文的包名仅为示例，请以对应版本的包名为准。

```bash
chmod +x cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run
./cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run --help     # 确认安装参数
./cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run --quiet    # 以对应版本包名为准
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 运行时需要能解析 vendor 下的 op_api 动态库。
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant/op_api/lib:${LD_LIBRARY_PATH}
```

`--quiet` 表示静默安装、不交互。库路径中的 `cann-9.1.0` 需按机器上实际安装的
CANN 版本目录调整，并且每个运行这些算子的 shell 都要设置该变量。

## 编译并安装 torch extension

解压 zip 后，目录中包含 `setup.py` 与 `build_and_install.sh`。

```bash
cd batch_invariant_ops
bash build_and_install.sh
```

脚本会编译 `batch_invariant_ops_lib`、打包 wheel 并用 pip 安装。前置条件：
`torch_npu` 可导入、`ninja`、C++17 编译器、Python 开发头文件。wheel 与 Python
版本和 CPU 架构绑定，需在目标机器上编译。

## 验证

```bash
python -c "import batch_invariant_ops, torch; print(torch.ops.batch_invariant_ops.npu_mm_batch_invariant)"
python -c "from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant; print(mm_batch_invariant)"
cd examples && python test_npu_mm_batch_invariant.py
```

## 说明

- 算子模块采用懒加载：未安装扩展时导入
  `sgl_kernel_npu.operators.batch_invariant` 仍可成功，首次调用时才报错。
- 需使用包含 batch invariant 模块的 `sgl-kernel-npu` wheel；旧版本 wheel 中没有
  `sgl_kernel_npu.operators.batch_invariant` 包。
- 必须安装与设备、架构匹配的 `.run` 包，装错会在调用时报 aclnn 查找失败。
- CANN 安装在其他路径时，请相应调整 `set_env.sh` 的位置。
- 不要跨架构复制 wheel。
