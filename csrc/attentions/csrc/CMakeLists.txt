cmake_minimum_required(VERSION 3.10)

project(PTAExtensionOPS)

execute_process(
  COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE python_site_packages_path
)
string(STRIP "${python_site_packages_path}" python_site_packages_path)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fpie -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")

set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wl,-Bsymbolic -rdynamic -Wl,--no-undefined ${CMAKE_CXX_FLAGS}")
set(PYTORCH_INSTALL_PATH ${python_site_packages_path}/torch)
set(PYTORCH_NPU_INSTALL_PATH ${python_site_packages_path}/torch_npu)

set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)
link_directories($ENV{ASCEND_HOME_PATH}/lib64)

if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH $ENV{ASCEND_TOOLKIT_HOME})
endif()
set(ACL_INCLUDE_DIRS ${ASCEND_PATH}/include)
include_directories(
  ${ACL_INCLUDE_DIRS}
  ${ACL_INCLUDE_DIRS}/aclnn
  )

add_library(PTAExtensionOPS SHARED
    ./plugin/register_ops.cpp
    ./plugin/la.cpp
    ./plugin/find_op_path.cpp
    ./plugin/rainfusionattention.cpp
    ./plugin/block_sparse_attention.cpp
    ./plugin/sparse_block_estimate.cpp
    ./plugin/layernorm.cpp)

target_compile_features(PTAExtensionOPS PRIVATE cxx_std_17)
if(DEFINED ENV{USER_ABI_VERSION})
    string(STRIP "$ENV{USER_ABI_VERSION}" ABI)
    if(NOT (ABI STREQUAL "0" OR ABI STREQUAL "1"))
        message(FATAL_ERROR "The value of USER_ABI_VERSION must be 0 or 1, but current value is '${ABI}'")
    endif()
else()
    set(ABI 0)
endif()
target_compile_options(PTAExtensionOPS PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${ABI})

include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/hccl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)

execute_process(
  COMMAND which python3
  OUTPUT_VARIABLE Python3_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE Python3_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
  OUTPUT_VARIABLE Python3_LIBRARY_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))"
  OUTPUT_VARIABLE Python3_LIBRARY_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Python3_LIBRARIES ${Python3_LIBRARY_DIRS}/${Python3_LIBRARY_NAME})
if(NOT EXISTS "${Python3_LIBRARIES}")
    message(STATUS "Python library ${Python3_LIBRARIES} not found, trying alternate suffix...")
    string(REGEX MATCH "(\\.[^.]+)$" PYLIB_NAME_EXT "${Python3_LIBRARY_NAME}")
    string(REGEX REPLACE "(\\.[^.]+)$" "" PYLIB_NAME_WE "${Python3_LIBRARY_NAME}")
    
    if(PYLIB_NAME_EXT STREQUAL ".so")
        set(Python3_LIBRARIES "${Python3_LIBRARY_DIRS}/${PYLIB_NAME_WE}.a")
    elseif(PYLIB_NAME_EXT STREQUAL ".a")
        set(Python3_LIBRARIES "${Python3_LIBRARY_DIRS}/${PYLIB_NAME_WE}.so")
    else()
        message(FATAL_ERROR "Unknown Python library extension: ${Python3_LIBRARY_NAME}")
    endif()

    if(NOT EXISTS "${Python3_LIBRARIES}")
        message(FATAL_ERROR "Python library not found after checking .so and .a: ${Python3_LIBRARIES}")
    else()
        message(STATUS "Found alternate Python library: ${Python3_LIBRARIES}")
    endif()
else()
    message(STATUS "Found Python library: ${Python3_LIBRARIES}")
endif()
include_directories(${Python3_INCLUDE_DIRS})  
link_directories(${Python3_LIBRARY_DIRS})   

target_link_libraries(PTAExtensionOPS PUBLIC 
  c10 
  torch 
  torch_cpu 
  torch_npu 
  ascendcl 
  ${Python3_LIBRARIES}
  ${ASCEND_PATH}/lib64/libascendcl.so
  ${ASCEND_PATH}/lib64/libnnopbase.so
  ${ASCEND_PATH}/lib64/libopapi.so
  )
