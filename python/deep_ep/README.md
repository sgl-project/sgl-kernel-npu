<h2 align="left">
DeepEP-Ascend
</h2>

<p align="left">
<a><b>English</b></a> | <a href="README_CN.md"><b>中文</b></a>
</p>


## Introduction
Ascend Implementation of DeepEP

## Software and hardware
Supported Hardware Models: Atlas A3 Series Products
Platform: aarch64/x86
Supporting Software
- Driver Ascend HDK 25.0.RC1.1, CANN Community Edition 8.2.RC1.alpha003 and later versions (refer to the "[CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)" to install the CANN development kit package, as well as the supporting firmware and drivers)
- Before installing CANN software, you need to install the relevant [dependency list](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/softwareinst/instg/instg_0045.html)
- Python >= 3.9
- PyTorch >= 2.5.1, torch-npu >= 2.5.1-7.0.0

## Quick Start
DeepEP-Ascend supports both A2 and A3 and needs to generate packages separately on A2 and A3.
### Compile and Run
1. Prepare the CANN environment variables (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Build the project
Before executing the engineering build script build.sh, modify `_ASCEND_INSTALL_PATH` on line 7 of build.sh according to the CANN installation path.
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

### Installation
1. Pip install the `.whl` file into your Python environment
```bash
pip install output/deep_ep*.whl

# Link to the deep_ep_cpp.*.so file
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -

# (Optional) Confirm whether the import can be successfully
python -c "import deep_ep; print(deep_ep.__path__)"
```

2. Execute the environment variables for CANN (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
3. In the Python project, import `deep_ep`.

### Features
1. The A2 `low_latency_dispatch` and `low_latency_combine` operators support two types of internal operators: non-hierarchical and hierarchical.

    In the implementation of hierarchical operators, intra-node communication uses HCCS, while inter-node communication uses RDMA. In the implementation of non-hierarchical operators, both intra-node and inter-node communications use pure RDMA.

    By default, the non-hierarchical operator is executed. If the environment variables `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0` are configured, the hierarchical operator will be executed instead.

    A3 has only a non-hierarchical kernel implementation. Intra-node and inter-node communication uses pure HCCS communication.

2. In the A2 `dispatch_low_latency` **hierarchical** implementation, an additional parameter `topk_weights` needs to be passed. In addition, an extra 1D Tensor `expand_scales` with shape (A,) will be returned. `expand_scales` will replace `topk_weights` as the weight parameter for the internal kernel in `low_latency_combine`. A2 non-hierarchical kernels and A3 do not require passing topk_weights in dispatch.
> - For shared experts, $A$ must satisfy the condition: $ A = Bs * epWorldSize *  sharedExpertNum / sharedExpertRankNum $.
> - For MoE experts, when $globalBs$ is 0, $A$ must satisfy the condition: $A >= Bs * epWorldSize * min(localExpertNum, K)$; when $globalBs$ is not 0, A must > > satisfy the condition: $A >= globalBs * min(localExpertNum, K)$.

### Test
Execute deepep-related test scripts
```bash
python3 tests/python/deepep/test_fused_deep_moe.py
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py
```

### FAQ
1. If installing the `.whl` file results in the inability to import `deep_ep` in the project, check whether it is correctly installed in the `site-packages` directory of the current Python environment;
View installation path:
```
pip show deep-ep
```

2. If after installing the `.whl`, you encounter an issue where `deep_ep_cpp` is not found, you need to create a symbolic link of the `deep_ep_cpp*.so` files from the `site-packages/deep_ep` directory to the `site-packages` directory;
Execute the following command in the `site-packages` directory:
```
ln -s deep_ep/deep_ep_cpp*.so
```
