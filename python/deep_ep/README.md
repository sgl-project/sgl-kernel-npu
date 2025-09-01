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
### Compile and Run
1. Prepare the CANN environment variables (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Build the project
Before executing the engineering build script build.sh, modify `_ASCEND_INSTALL_PATH` on line 7 of build.sh according to the CANN installation path.
```bash
# Building Project
bash build.sh
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

### Test
Execute deepep-related test scripts
```bash
bash tests/python/deepep/run_test.sh
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
