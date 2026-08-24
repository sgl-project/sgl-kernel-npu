# SGL-Kernel-NPU 出包规则说明

本文档依据 `.github/workflows/build_and_release.yml` 整理，说明编译出包范围、构建矩阵、三个 job 的职责、各产物打包流程及命名规则。

## 1. 总览

工作流共包含 3 个独立 job，并行构建 3 类产物：

| Job | 产物 | 包内容 | 职责 |
| --- | --- | --- | --- |
| `build-and-release` | sgl-kernel-npu | 全部 `.whl` | 编译本仓库自身的 NPU 算子 |
| `build-ops-transformer` | ops-transformer | 全部 `.run` | 编译昇腾官方 ops-transformer 中项目依赖的算子 |
| `build-cuntom-ops` | custom-ops | `.whl` + `.run` | 编译 cann-recipes-infer 的自定义算子 |

> 注意：第三个 job 的 `build-cuntom-ops` 为原文件中的拼写（cuntom），产物名实际为 `custom-ops`。

## 2. 编译出包范围（目录树）

> 目录树刷新时间：2026-08-21（依据仓库实际目录整理）

```
sgl-kernel-npu/                              ← GitHub 工作区（本仓库）
├── .github/workflows/
│   └── build_and_release.yml                ← 出包工作流定义
├── build.sh                                 ← 构建入口：-a kernels / attentions / memory-saver / deepep
├── config.ini                               ← 随 sgl_kernel_npu wheel 一起打入
├── CMakeLists.txt                           ← 顶层 CMake 构建入口
├── cmake/                                   ← CMake 配置（config_*.cmake）
├── include/                                 ← 算子头文件（sgl_kenel_npu_ops.h）
├── scripts/
│   └── npu_ci_install_dependency.sh         ← 构建依赖安装
├── csrc/                                    ← C++ / AscendC 算子源码
│   ├── attentions/                          ← attentions 算子
│   ├── deepep/                              ← DeepEP ops / ops2（按硬件二选一）
│   └── <其他算子目录>                        ← kernels 对应的 AscendC 算子
├── python/
│   ├── sgl_kernel_npu/                      ← -a kernels 打的 wheel
│   ├── attentions/                          ← -a attentions 打的 wheel
│   └── deep_ep/                             ← -a deepep / deepep2 打的 wheel
├── contrib/
│   ├── torch_memory_saver/                  ← -a memory-saver 打的 wheel
│   └── patch/Adapt-custom-ops-A2.patch      ← 910b + aarch64 自定义算子补丁
├── third_party/
│   └── pto-isa/                             ← 子模块，mega_chunk_gdn 算子编译时引用
└── output/                                  ← 构建产物目录：*.whl → *.zip（构建时生成）

（以下两个目录为构建时克隆到工作区上一级的外部仓库，不属于本仓库）
../ops-transformer/                          ← Job 2：官方仓库（commit bcc6304）
└── build_out/                               ← 产物 *.run，打包成 *.zip

../cann-recipes-infer/                       ← Job 3：官方仓库（commit 35a476c）
├── ops/ascendc/
│   ├── torch_ops_extension/dist/            ← custom_ops-*.whl
│   └── output/                              ← CANN-custom_ops-*.run
└── output/                                  ← 打包输出：*.whl + *.run → *.zip
```

> 不参与本次出包：`csrc/catlass/` 与 `third_party/catlass/`（仅 `BUILD_CATLASS_MODULE=ON` 时编译，出包流程默认关闭）；`.github/workflows/scripts/wheel/`（属于另一个工作流 `release_deep_ep_wheel.yml`）。

## 3. 构建矩阵（三个 job 共用）

| 参数 | 取值 |
| --- | --- |
| `arch` | `x86_64`、`aarch64` |
| `npu_hardware` | `910b`、`a3`、`950` |
| `cann_version` | `9.0.0`、`9.1.0` |
| `torch_version` | `2.10.0`（固定） |
| 排除组合 | `950` + `cann 9.0.0` 不构建 |
| 运行器 | `x86_64` → `ubuntu-24.04`；`aarch64` → `ubuntu-24.04-arm` |

组合数量：2 × 3 × 2 − 2（排除 950 + CANN 9.0.0 的两种架构）= **10 种配置**，每个 job 各构建 10 个包，一次 Release 共 30 个包。

构建容器镜像统一为：

```
quay.io/ascend/cann:<cann_version>-<npu_hardware>-ubuntu22.04-py<版本>
```

其中 Python 版本随 CANN 版本变化：

| CANN 版本 | Python 版本 |
| --- | --- |
| 9.0.0 | 3.11 |
| 9.1.0 | 3.12 |

### 构建组合清单

torch 固定为 `2.10.0`；运行器按架构区分：`x86_64` → `ubuntu-24.04`，`aarch64` → `ubuntu-24.04-arm`。✓ 表示构建，✗ 表示排除。

| 架构 | 硬件 | CANN 9.0.0（py3.11） | CANN 9.1.0（py3.12） |
| --- | --- | --- | --- |
| x86_64 | 910b | ✓ | ✓ |
| x86_64 | a3 | ✓ | ✓ |
| x86_64 | 950 | ✗ | ✓ |
| aarch64 | 910b | ✓ | ✓ |
| aarch64 | a3 | ✓ | ✓ |
| aarch64 | 950 | ✗ | ✓ |

每个组合由 3 个 job 各产出一个包，共 3 个；10 种组合 × 3 个 job = 一次 Release 共 30 个包。

## 4. 公共流程

每个 job 都按以下顺序执行：

1. **Checkout 仓库**（`build-and-release` 额外启用 `submodules: recursive`）
2. **标记仓库为安全目录**：`git config --system --add safe.directory ${GITHUB_WORKSPACE}`
3. **安装依赖**：`bash scripts/npu_ci_install_dependency.sh --cann-version <版本>`
4. **按硬件构建**（各 job 详见下文）
5. **打包**（zip）
6. **上传产物**（仅 Release 触发，通过 `softprops/action-gh-release@v2` + `GITHUB_TOKEN`，需要 `contents: write` 权限）

## 5. Job：build-and-release（SGL-Kernel-NPU）

**用途：** 编译本仓库自身的算子，产出项目自己的 wheel 包，包含核心 kernels、attention 算子、memory-saver，以及按硬件区分的 DeepEP 通信库。

构建顺序固定为：kernels → attentions → memory-saver → 对应硬件的 DeepEP。

| 步骤 | 命令 | 适用硬件 |
| --- | --- | --- |
| kernels | `./build.sh -a kernels` | 全部 |
| attentions | `./build.sh -a attentions` | 全部 |
| memory-saver | `./build.sh -a memory-saver` | 全部 |
| DeepEP | `./build.sh -a deepep2` | 910b |
| DeepEP | `./build.sh -a deepep` | a3 |
| DeepEP | `./build.sh -a deepep Ascend950` | 950 |

构建时统一设置 `LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/runtime/lib64/stub`。

**打包规则：**

- 打包前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- torch / python 版本在运行时自动探测，而非使用矩阵固定值
- 打包 `output/` 目录下所有 `.whl`

**包命名格式：**

```
sgl-kernel-npu-<tag>-torch<torch版本>-py<python版本>-cann<cann版本>-<硬件>-<架构>.zip
```

示例：`sgl-kernel-npu-v1.0.0-torch2.10.0-py311-cann9.0.0-910b-x86_64.zip`

- torch 版本：`torch.__version__` 去掉 `+` 后缀
- python 版本：取前两位合并，如 `3.11` → `py311`

## 6. Job：build-ops-transformer

**用途：** 从昇腾官方 ops-transformer 仓库（固定 commit `bcc6304`）挑选项目依赖的稀疏注意力、量化索引、压缩等算子，编译成 CANN 的 `.run` 安装包。

从 `https://gitcode.com/cann/ops-transformer.git` 克隆并固定到 commit `bcc6304`，按硬件执行：

```
bash build.sh --pkg --experimental --soc=<soc> --ops="<算子列表>"
```

| 硬件 | `--soc` | 构建算子 |
| --- | --- | --- |
| a3 | `ascend910_93` | `sparse_attn_sharedkv`、`sparse_attn_sharedkv_metadata`、`quant_lightning_indexer`、`quant_lightning_indexer_metadata`、`compressor` |
| 910b | `ascend910b` | 同上 5 个算子 |
| 950 | `ascend950` | `quant_lightning_indexer`、`quant_lightning_indexer_metadata`、`compressor`、`kv_quant_sparse_attn_sharedkv`、`kv_quant_sparse_attn_sharedkv_metadata`、`kv-compressor_epilog`、`hc_post`、`inplace_partial_rotary_mul`、`indexer_compress_epilog`、`moe_gating_top_k` |

**打包规则：**

- 打包 `../ops-transformer/build_out/` 下所有 `.run`

**包命名格式：**

```
ops-transformer-<tag>-torch2.10.0-cann<cann版本>-<硬件>-<架构>.zip
```

示例：`ops-transformer-v1.0.0-torch2.10.0-cann9.1.0-a3-aarch64.zip`

## 7. Job：build-cuntom-ops（custom-ops）

**用途：** 从 cann-recipes-infer 仓库（固定 commit `35a476c`）构建自定义算子，同时产出 torch 扩展（`.whl`）和 CANN 自定义算子（`.run`）两种格式。

从 `https://gitcode.com/cann/cann-recipes-infer.git` 克隆并固定到 commit `35a476c`，构建两部分：

1. `ops/ascendc/torch_ops_extension`：先 `pip install ninja`，再执行 `build_and_install.sh`
2. `ops/ascendc`：执行 `build.sh -c <soc>`

| 硬件 | `build.sh -c` 的 soc | 特殊处理 |
| --- | --- | --- |
| a3 | `ascend910_93` | 无 |
| 910b | `ascend910b` | `aarch64` 架构需先应用补丁 `../sgl-kernel-npu/contrib/patch/Adapt-custom-ops-A2.patch` |
| 950 | `ascend950` | 无 |

**打包规则：**

- 将 `custom_ops-*.whl` 与 `CANN-custom_ops-*.run` 拷贝到 `output/` 目录
- 打包 `output/` 下所有 `.run` 和 `.whl`

**包命名格式：**

```
custom-ops-<tag>-torch2.10.0-cann<cann版本>-<硬件>-<架构>.zip
```

示例：`custom-ops-v1.0.0-torch2.10.0-cann9.0.0-950-x86_64.zip`

## 8. 关键规则速查

- **950 不支持 CANN 9.0.0**，该组合直接排除，不进入构建
- **torch 版本固定为 2.10.0**（仅 job 1 的包名中 torch/python 版本为运行时探测）
- 版本号统一取自 Release 的 `tag_name`
- 包名统一结构：`<产品>-<tag>-torch<torch>-cann<cann>-<硬件>-<架构>.zip`
- 所有构建均在昇腾 CANN 容器内进行，shell 固定为 `bash`
- 三个 job 相互独立，可并行执行