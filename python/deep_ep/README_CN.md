<h2 align="left">
DeepEP-Ascend
</h2>

<p align="left">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>



## 介绍
DeepEP的ascend实现


## 软硬件配套说明
硬件型号支持：Atlas A3 系列产品
平台：aarch64/x86
配套软件
- 驱动 Ascend HDK 25.0.RC1.1、CANN社区版8.2.RC1.alpha003及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装CANN开发套件包以及配套固件和驱动）
- 安装CANN软件前需安装相关[依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/softwareinst/instg/instg_0045.html)
- Python >= 3.9
- PyTorch >= 2.5.1, torch-npu >= 2.5.1-7.0.0

## 快速上手
### 编译执行
1、准备CANN的环境变量（根据安装路径修改）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、构建项目
执行工程构建脚本 build.sh前，根据CANN安装路径，修改`build.sh:line7`的`_ASCEND_INSTALL_PATH`。
```bash
# 构建项目
bash build.sh
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

### 测试
执行deepep相关测试脚本
```bash
bash tests/python/deepep/run_test.sh
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
