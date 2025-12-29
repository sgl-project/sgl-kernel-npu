cmake_minimum_required(VERSION 3.16)

set(SOC_VERSION "Ascend910_9382" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()


add_executable(ascendc_reduce_scatter ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_scatter.cpp)

target_compile_options(ascendc_reduce_scatter PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_include_directories(ascendc_reduce_scatter PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    ${ZCCL_TEST_ROOT}/utils
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/aclnn/opdev
    ${ZCCL_INCLUDE_DIR}
)

target_link_directories(ascendc_reduce_scatter PRIVATE
    ${SHMEM_HOME_PATH}/shmem/lib/
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    ${ASCEND_HOME_PATH}/lib64
    ${ZCCL_LIB_DIR}
)

target_link_libraries(ascendc_reduce_scatter PRIVATE
    ascendcl
    nnopbase
    zccl
    shmem
    mf_smem
    mf_hybm_core
)

install(TARGETS ascendc_reduce_scatter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
