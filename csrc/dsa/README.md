## 概述

此项目是基于昇腾Atlas A3的融合算子库，当前项目中包括[SparseFlashAttention](./docs/custom-npu_sparse_flash_attention.md)和[LightningIndexer](./docs/custom-npu_lightning_indexer.md)两个算子。

## 目录结构说明

融合算子代码目录结构如下：

  ```
  ├── cmake                                     # 项目工程编译目录
  ├── docs                                      # 算子使用说明和资料
  ├── examples                                  # 算子的使用示例代码
  ├── src                                       # 算子的源代码
  |   ├── sparse_flash_attention                # 推理SparseFlashAttention（简称sfa）算子示例代码
  |   |   ├── op_host                           # 算子信息库、Tiling、InferShape相关实现目录
  |   |   ├── op_kernel                         # 算子Kernel目录
  |   ├── lightning_indexer                     # 推理LightningIndexer（简称li）算子示例代码
  |   |   ├── op_host                           # 算子信息库、Tiling、InferShape相关实现目录
  |   |   ├── op_kernel                         # 算子Kernel目录
  |
  ├── torch_ops_extension                       # torch_ops_extension目录
      ├── custom_ops
      │   ├── csrc                              # 自定义算子适配层c++代码目录
      │   └── converter                         # 自定义算子包python侧converter代码
      ├── setup.py                              # wheel包编译文件
      ├── build_and_install.sh                  # 自定义算子wheel包编译与安装脚本
  |
  ├── build.sh                                  # 项目工程编译脚本
  ├── CMakeList.txt                             # 项目工程编译配置文件
  ├── README.md
  ├── version.info                              # 项目版本信息
  ```
昇腾社区Ascend C自定义算子开发资料：[Ascend C自定义算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)


## 环境准备<a name="1"></a>
### 下载源码

  执行如下命令下载 cann-recipes-infer 源码。
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone git@gitcode.com:cann/cann-recipes-infer.git
  cd cann-recipes-infer
  ```

### 获取 docker 镜像

  从[ARM镜像地址](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann8.3.rc1.alpha002/pt2.5.1/aarch/ascendc/cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image.tar)中下载 docker 镜像，然后上传到A3服务器上，并通过命令导入镜像 `docker load -i cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image.tar`。

### 拉起 docker 容器

  通过如下脚本拉起容器，默认容器名为 cann_recipes_infer。
  ```
  docker run -u root -itd --name cann_recipes_infer --ulimit nproc=65535:65535 --ipc=host \
      --device=/dev/davinci0     --device=/dev/davinci1 \
      --device=/dev/davinci2     --device=/dev/davinci3 \
      --device=/dev/davinci4     --device=/dev/davinci5 \
      --device=/dev/davinci6     --device=/dev/davinci7 \
      --device=/dev/davinci8     --device=/dev/davinci9 \
      --device=/dev/davinci10    --device=/dev/davinci11 \
      --device=/dev/davinci12    --device=/dev/davinci13 \
      --device=/dev/davinci14    --device=/dev/davinci15 \
      --device=/dev/davinci_manager --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      -v /home/:/home \
      -v /data:/data \
      -v /etc/localtime:/etc/localtime \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
      -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
      -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
      -v /usr/bin/hostname:/usr/bin/hostname \
      --net=host \
      --shm-size=128g \
      --privileged \
      cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image:v0.1 /bin/bash
  ```
  通过如下命令进入容器：
  ```
  docker attach cann_recipes_infer
  ```

### 设置环境变量

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

## 编译执行

### 自定义融合算子编译

执行如下命令编译所有自定义算子：

  ```bash
  cd /home/code/cann-recipes-infer/ops/ascendc
  bash build.sh
  ```

**说明：**

若提示如下信息，则说明编译成功。

  ```
  Self-extractable archive "CANN-custom_ops-<cann_version>-linux.<arch>.run" successfully created.
  ```

编译成功后在 `output` 目录生成自定义算子包：`CANN-custom_ops-<cann_version>-linux.<arch>.run`。其中，\<cann_version>表示软件版本号，\<arch>表示操作系统架构。

### 自定义融合算子安装

安装前，需确保所安装的自定义算子包与所安装CANN开发套件包CPU架构一致，安装命令如下：

  ```bash
  cd /home/code/cann-recipes-infer/ops/ascendc/output
  chmod +x CANN-custom_ops-<cann_version>-linux.<arch>.run
  ./CANN-custom_ops-<cann_version>-linux.<arch>.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
  source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
  ```

执行上述命令后，自定义融合算子对应的run包会安装到对应的CANN软件包目录:`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/`

### torch_ops_extension算子包编译与安装
编译与安装命令如下：
  ```shell
  cd /home/code/cann-recipes-infer/ops/ascendc/torch_ops_extension
  bash build_and_install.sh
  ```

编译成功后在 `dist` 目录生成自定义custom-ops算子包：`custom_ops-1.0-<python_version>-<python_version>-<arch>.whl`。其中，\<python_version>表示python版本号，\<arch>表示操作系统架构。


### examples用例运行
examples用例运行命令如下：
  ```shell
  cd /home/code/cann-recipes-infer/ops/ascendc/examples
  python3 test_npu_lightning_indexer.py
  python3 test_npu_sparse_flash_attention.py
  ```
