# PlatFormInfos兼容性修复指南

## 问题分析
在CANN 8.5中，`fe::PlatFormInfos` 类可能API变化或头文件位置改变，导致编译错误：
```
error: invalid use of incomplete type 'class fe::PlatFormInfos'
```

## 已完成的修复

### 1. 创建兼容头文件
创建了 `csrc/deepep/ops2/op_host/soc_version_compat.h`，提供：
- CANN 8.2/8.3: 使用 `fe::PlatFormInfos`
- CANN 8.5+: 使用 `platform_ascendc::PlatformAscendCManager`

### 2. 已修改的文件
- ✓ `csrc/deepep/ops2/op_host/dispatch_normal_a2_tiling.cpp` - 使用兼容API
- ✓ `csrc/deepep/ops2/op_host/dispatch_layout_tiling.cc` - 使用兼容API
- ✓ `csrc/deepep/ops/op_host/mc2_tiling_utils.h` - GetSocVersion兼容函数

### 3. 需要手动修改的文件
剩余以下文件需要添加 `#include "soc_version_compat.h"` 并修改相关函数：

#### moe_distribute_combine_v2_tiling.cc
在第1416-1420行左右，找到类似代码：
```cpp
fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
fe::PlatFormInfos &platformInfo = *platformInfoPtr;
std::string socVersion;
(void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
```

修改为：
```cpp
#if defined(USE_CANN83_PATH) || defined(USE_CANN82_PATH)
fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
fe::PlatFormInfos &platformInfo = *platformInfoPtr;
std::string socVersion;
(void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
#else
std::string socVersion = Cam::GetSocVersionCompat();
#endif
```

#### moe_distribute_dispatch_v2_tiling.cc
类似修改，在第1342-1347行左右

#### notify_dispatch_tiling_a2.cc
类似修改

### 4. CMakeLists修改
已完成：
- ✓ `csrc/deepep/ops2/op_host/CMakeLists.txt` - 添加 `-D${CANN_VERSION_MACRO}` 到编译选项
- ✓ `csrc/deepep/ops2/CMakeLists.txt` - CANN版本检测
- ✓ `csrc/deepep/ops/CMakeLists.txt` - CANN版本检测

## 统一的修改模式

对于所有使用 `fe::PlatFormInfos` 的地方，按照以下模式修改：

1. 添加头文件：
```cpp
#include "soc_version_compat.h"
```

2. 修改函数逻辑：
```cpp
#if defined(USE_CANN83_PATH) || defined(USE_CANN82_PATH)
    // CANN 8.2/8.3的原有代码
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    ...
#else
    // CANN 8.5+使用新API
    std::string socVersion = Cam::GetSocVersionCompat();
#endif
```

## 编译测试

修改完成后，执行：
```bash
bash build.sh
```

如果仍有问题，运行诊断：
```bash
bash diagnose_cann.sh
```

## 常见问题

### Q: 编译时仍然报#error怎么办？
A: 确保 `CANN_VERSION_MACRO` 被正确传递。检查编译输出中是否有：
```
op_host using CANN_VERSION_MACRO: USE_CANN83_PATH
```

### Q: PlatFormInfos仍然不完整怎么办？
A: 这表示CANN 8.5中确实不存在或API完全不同，使用我们提供的 `soc_version_compat.h` 中的兼容函数。

### Q: platform_ascendc API也不工作怎么办？
A: 可能是CANN 8.5中该API也有变化，请检查CANN 8.5的官方文档或联系华为技术支持。