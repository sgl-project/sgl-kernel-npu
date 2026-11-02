# mlp_lightning_indexer A5 单算子验证

本分支基于 [sgl-kernel-npu #579](https://github.com/sgl-project/sgl-kernel-npu/pull/579)
的 `5230a0e226354788bc404be05e96cdf893a345ff`，交付 A3 → A5 的最小适配与
单算子测试入口。配套 [SGLang #47917](https://github.com/sgl-project/sglang/pull/47917)
的框架接入不在本分支修改。代码准备完成不代表 A5 编译或精度已经通过。

## 运行

在目标 A5 Linux 主机的实际 PyTorch/torch_npu 容器或虚拟环境里执行。已知目标为
Ascend950DT、CANN 9.2.0、aarch64；脚本按 `3510 → ascend950` 构建。
先选一张可用于测试的卡，`--device` 是当前进程可见设备的逻辑编号。

```bash
git clone --branch light_indexer_a5 --single-branch https://github.com/qybnb/sgl-kernel-npu.git
cd sgl-kernel-npu
bash scripts/run_light_indexer_a5.sh --device 0
```

如果当前 `python3` 不是运行 SGLang 的解释器，可以指定：

```bash
bash scripts/run_light_indexer_a5.sh --python /path/to/venv/bin/python --device 0 --jobs 8
```

选择的 Python 必须已有匹配的 `torch`、`torch_npu`、`pytest`、`numpy`、
`setuptools`、`wheel`、`packaging` 和 venv 支持。开发环境还需 CANN SDK、
CMake >= 3.20、C++17 编译器与 Python 开发头文件。脚本检查缺失依赖并报错，
不安装或升级这些依赖。若宿主机 `pip show torch torch-npu` 找不到包，
应进入实际容器或选择正确解释器。

脚本保留已有 `ASCEND_HOME_PATH`；未设置时尝试 source
`/usr/local/Ascend/cann-9.2.0/set_env.sh`，再尝试常见安装位置。
其他路径可先手动 source，或设置 `CANN_ENV=/path/to/set_env.sh`。

脚本依次执行：

1. 记录源码 commit/diff/status、设备、Torch/torch_npu、CMake、编译器与 SDK 头文件摘录。
2. 创建 `--system-site-packages` 测试 venv，复用所选 Python 的 Torch 依赖。
3. 清理本 checkout 的新构建缓存，并只编译本算子：

   ```bash
   python setup.py --make_clean --arch-version 3510 \
     --ops mlp_lightning_indexer --artifact wheel --jobs 8
   ```

4. 将生成的 wheel 安装到测试 venv（`--no-index --no-deps`），核验包来自该 venv、
   manifest 为 `3510` 且仅含本算子、原生 binding 与三个实现均可解析。
5. 只运行 `test_mlp_lightning_indexer.py`，显式选择 `ascendc` 对照 `torch_native`，
   开启精度比较，关闭性能采集。pytest 用例为 eager，不运行 NPUGraph 或整模型。

脚本不替换原环境的 sgl-kernel-npu，也不操作已有服务。测试 venv 和日志默认位于
`output/light_indexer_a5/<timestamp>/`，可用 `--output-dir /new/path` 指定新目录。
编译仍使用 checkout 下的 `build/` 与 `output/`，因此使用独立 checkout，
不要在同一 checkout 并发运行其他构建。

## 本次代码修改

| 位置 | 改动 | 原因 |
| --- | --- | --- |
| CMake、OpDef、host SoC 检查、Python registry、kernel 入口 | 新增 3510 / ascend950 / ARCH35，取消对 `__CCE_AICORE__ == 310` 的空入口 | 使 A5 能选中、构建并执行算子 |
| `LIMatmul::LoadQueryToL0a` | 仅 A5 调用 `SetLoadDataRepeat`，单次搬运，`dstStride=ceil(M_tile/16)` | A5 L0A 为 NZ，需要显式配置 K 方向上相邻分形列的目的间隔 |
| `LIMatmul::AllocEventID/FreeEventID` | 为每个缓冲槽申请独立 TPipe event，等待完成后释放；不假定 ID 连续 | 取代固定编号，保留原有缓冲复用与等待关系；此项也作用于 arch22 |
| eager 测试 | 增加 heads=64、M 分块、分页/尾块、长 K、Top-K 4096/8192、两种 return_value | 检查布局、缓冲复用和结果契约 |

数学功能与公共签名保持原样：`Q·K → ReLU → 按 head 权重求和 → block 归约/窗口优先级 → Top-K`。
现有 K→L0B 的 `LoadData2DParams`、`Mmad`、FP32/ReLU/NZ2ND 的 `Fixp()`、
`SetMMLayoutTransform`、`unitFlag=3`、Cube/Vector 通知协议与 Vector 算法继续保留。
不扩展 BSND、不改变输入 dtype/长度契约，也不把参考实现作为自动 fallback。

## 公开文档依据与待验证项

[LoadData 3D 文档](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/LoadData_3D.html)
列出 A5 对 V2 接口的支持，并说明 A5 L0A 为 NZ、L0B 为 ZN。
[SetLoadDataRepeat](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_load_aux_config/SetLoadDataRepeat.html)
要求 A5 调用 LoadData 3D 时配置 `dstStride`，单位为 512B 分形。本实现输出 tile
为 `[M_tile,128]`，FP16/BF16 分形是 16×16，因此 K 方向相邻分形列间隔为
`ceil(M_tile/16)` 个分形；每个 L0 tile 单独设置，避免复用上一个 tile 的步长。

[LoadData 2D](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/LoadData_2D.html)
支持 A5 的 L1→L0B；本算子 K 在 L0B 使用 ZN，不因 L0A 的变化强制改成 V2。
[DataCopy L0C→GM](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/DataCopy_L0CToGM.html)
列出 A5 对现有 `DataCopyCO12DstParams`、FP32 输出、ReLU 与 NZ2ND 的支持，
目前没有依据要求整段替换成另一套 Fixpipe 接口。

上述页面是官方 asc-devkit master 的持续开发预览，不能替代已安装的 CANN 9.2.0。
`sdk-headers.log` 保存实装头文件的相关声明（找不到时明确记录），
`build.log` 和 `build/cann/` 的日志确认实际编译器是否接受这些调用。
A5 上的搬运布局、FP32 写回、unitFlag、事件配对最终仍由实机精度测试确认；
arch22 的事件改动也需要 A3 回归。NPUGraph、性能和整模型尚未验收。

## 结果与反馈

全部用例成功且无 skip 才记录 `summary.json` 的 `status: PASS`。
同分项允许不同索引，但校验索引范围、唯一性、候选数和选中分数；
仅 `return_value=True` 比较 values，`False` 时 values 内容无定义。
FP16/BF16 的二进制分数输入可精确表示，用精确 oracle 分数检查 Top-K 排名，
避免 BF16 舍入掩盖排布错误。这不替代后续实际模型分布的精度与性能验收。

若失败，先回传 `summary.json` 及对应日志：

- `environment.log`：解释器/设备/依赖问题。
- `sdk-headers.log`、`build.log`：SDK 接口与编译问题。
- `installed-build.log`：加载到了哪个包、架构和实现是否匹配。
- `pytest.log`、`pytest.xml`：具体失败 shape/dtype 和断言。

不要用“编译成功”“pytest 全 skip”或 A3 旧结果代替 A5 单算子精度通过。
