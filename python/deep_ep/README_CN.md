<h2 align="left">
DeepEP-Ascend
</h2>

<p align="left">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>



## 介绍
DeepEP的ascend实现


## 软硬件配套说明
硬件型号支持：Atlas A2 和 A3 系列产品
平台：aarch64/x86
配套软件
- 驱动 Ascend HDK 25.0.RC1.1、CANN社区版8.2.RC1.alpha003及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装CANN开发套件包以及配套固件和驱动）
- 安装CANN软件前需安装相关[依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/softwareinst/instg/instg_0045.html)
- Python >= 3.9
- PyTorch >= 2.5.1, torch-npu >= 2.5.1-7.0.0

## 快速上手
DeepEP-Ascend支持A2和A3，需要在A2和A3上分别生成包。
### 编译执行
1、准备CANN的环境变量（根据安装路径修改）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、构建项目
执行工程构建脚本 build.sh前，根据CANN安装路径，修改`build.sh:line7`的`_ASCEND_INSTALL_PATH`。
- A3
    ```bash
    # Building Project
    bash build.sh -a deepep
    ```
- A2
    ```bash
    # Building Project
    bash build.sh -a deepep2
    ```

### 安装
1、执行pip安装命令，将`.whl`安装到你的python环境下
```bash
pip install output/deep_ep*.whl

# 设置deep_ep_cpp*.so的软链接
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -

# （可选）确认是否可以成功导入
python -c "import deep_ep; print(deep_ep.__path__)"
```

2、执行CANN的环境变量（根据安装路径修改）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
3、在python工程中导入`deep_ep`

### 特性
1. A2 的 `low_latency_dispatch` 和 `low_latency_combine` 算子支持两种内部算子类型：不分层和分层。
 在分层算子的实现中，节点内通信使用 HCCS，节点间通信使用 RDMA。在不分层算子的实现中，节点内和节点间通信均使用纯 RDMA。
 默认情况下，执行的是非层次化算子。如果配置了环境变量 `HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0`，则将执行分层算子。
 A3 无需分层，节点内和节点间通信均使用纯 HCCS 通信。

### 测试
执行deepep相关测试脚本
```bash
python3 tests/python/deepep/test_fused_deep_moe.py
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py

# 在A2双机下执行，测试internode (需要先设置run_test_internode.sh中的主节点IP)
bash run_test_internode.sh
```

### 常见问题
1、如果安装`.whl`后，在工程中`import deep_ep`出现找不到`deep_ep`库，则检查是否正确安装到当前Python环境的`site-packages`目录下；
查看安装路径：
```
pip show deep-ep
```

2、如果安装`.whl`后，出现找不到`deep_ep_cpp`，则需要将`site-packages/deep_ep`目录下的`deep_ep_cpp*.so`文件软链接到`site-packages`目录下；
在`site-packages`目录下执行：
```
ln -s deep_ep/deep_ep_cpp*.so
```
