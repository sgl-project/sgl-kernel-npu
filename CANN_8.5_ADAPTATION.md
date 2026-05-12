# CANN 8.5适配问题解决方案

## 问题1：CANN版本不支持

### 问题描述
编译时报错：
```
#error "CANN version not supported or platform_infos_def.h not found. Check CANN_VERSION_MACRO definition."
```

### 原因
代码只支持CANN 8.2和8.3版本，但使用了CANN 8.5。

### 解决方案
已修改以下文件：
1. `cmake/config_ascend.cmake` - 添加CANN版本自动检测
2. `csrc/deepep/ops2/op_host/CMakeLists.txt` - 使用动态宏定义
3. 源文件（6个） - 支持`USE_CANN85_PATH`

CANN 8.5及更高版本使用CANN 8.3的API（向后兼容）。

---

## 问题2：找不到ASC包配置文件

### 问题描述
编译时报错：
```
CMake Error: Could not find a package configuration file provided by "ASC"
```

### 原因
CMake无法在`ascendc_kernel_cmake`目录中找到`ASCConfig.cmake`文件。

### 解决方案

#### 步骤1：运行诊断脚本
```bash
bash diagnose_cann.sh
```

这个脚本会检查：
- 环境变量设置
- ascendc_kernel_cmake目录位置
- ASC配置文件的实际位置
- CANN版本信息

#### 步骤2：根据诊断结果调整

已修改`csrc/deepep/ops/CMakeLists.txt`和`csrc/deepep/ops2/CMakeLists.txt`：
- 使用`find_package(ASC QUIET)`代替`find_package(ASC REQUIRED)`
- 如果找不到ASC包，手动配置必要的变量
- 直接include `ascendc.cmake`文件

#### 步骤3：手动设置ASC_DIR（如果需要）
如果诊断脚本找到了ASCConfig.cmake但不在默认路径，可以设置：

```bash
export ASC_DIR=/path/to/ascendc_kernel_cmake
```

或者在CMake命令中：
```bash
cmake -DASC_DIR=/path/to/ascendc_kernel_cmake ...
```

---

## 编译步骤

### 1. 设置环境变量
```bash
# 从系统配置获取CANN路径
_CANN_TOOLKIT_INSTALL_PATH=$(cat /etc/Ascend/ascend_cann_install.info | grep "Toolkit_InstallPath" | awk -F'=' '{print $2}')
source ${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh
```

### 2. 验证环境
```bash
echo $ASCEND_HOME_PATH
echo $ASCEND_TOOLKIT_HOME
bash diagnose_cann.sh
```

### 3. 运行编译
```bash
bash build.sh
```

---

## 已知的CANN 8.5路径变化

CANN 8.5可能包含以下变化：
- `ASCConfig.cmake`可能不存在或名称变化
- `ascendc_kernel_cmake`目录路径可能变化
- 某些头文件路径可能重新组织

如果遇到其他问题，请查看诊断脚本的输出并相应调整。

---

## 联系和支持

如果问题仍然存在，请：
1. 运行`diagnose_cann.sh`并保存输出
2. 检查CANN 8.5的官方文档
3. 在GitHub提交issue并附上诊断输出