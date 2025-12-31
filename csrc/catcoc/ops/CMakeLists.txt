cmake_minimum_required(VERSION 3.16.0)
project(catcoc_kernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_SGL_KERNEL_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(NOT DEFINED ASCEND_HOME_PATH AND NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED SHMEM_HOME_PATH AND NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED SHMEM_HOME_PATH)
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

set(CMAKE_BISHENG_COMPILER  ${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/bisheng)

set(THIRD_PARTY_HOME_PATH ${CMAKE_SGL_KERNEL_DIRECTORY}/3rdparty)
message(${THIRD_PARTY_HOME_PATH})

# check 3rdparty
if(NOT EXISTS "${THIRD_PARTY_HOME_PATH}/catlass")
    message(FATAL_ERROR "catlass not found in: ${THIRD_PARTY_HOME_PATH}/catlass")
endif()
if(NOT EXISTS "${THIRD_PARTY_HOME_PATH}/catcoc")
    message(FATAL_ERROR "catcoc not found in: ${THIRD_PARTY_HOME_PATH}/catcoc")
endif()

# use --cce-enable-print if you want debug
set(BISHENG_COMPILER_OPTIONS
        -O2 -std=c++17 -xcce -fPIC
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -DL2_CACHE_HINT
        -DTILING_KEY_VAR
        -I${ASCEND_HOME_PATH}/compiler/tikcpp
        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
        -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
        -I${ASCEND_HOME_PATH}/include
        -I${ASCEND_HOME_PATH}/include/experiment/runtime
        -I${ASCEND_HOME_PATH}/include/experiment/msprof
        -I${THIRD_PARTY_HOME_PATH}/catlass/include
        -I${THIRD_PARTY_HOME_PATH}/catcoc/include
        -I${SHMEM_HOME_PATH}/shmem/include
        -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
        -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
        -I${CMAKE_SOURCE_DIR}/../include
        -Wno-macro-redefined -Wno-ignored-attributes
)

set(BISHENG_LINK_OPTIONS
        --cce-fatobj-link
        -L${ASCEND_HOME_PATH}/lib64
        -L${SHMEM_HOME_PATH}/memfabric_hybrid/lib
        -L${SHMEM_HOME_PATH}/shmem/lib
        -lmf_smem -lmf_hybm_core -lshmem
        -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

if(DEFINED PROF)
    list(APPEND ${BISHENG_COMPILER_OPTIONS} -lprofapi)
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/obj)

function(catlass_add_kernel NAME ARCH)
    set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/obj)
    set(OUTPUT_NAME ${NAME}.o)
    set(OUTPUT_FULL_NAME ${OUTPUT_DIR}/${OUTPUT_NAME})

    add_custom_command(
            OUTPUT ${OUTPUT_FULL_NAME}
            COMMAND ${CMAKE_BISHENG_COMPILER} --cce-aicore-arch=${ARCH} ${BISHENG_COMPILER_OPTIONS} -c ${ARGN} -o ${OUTPUT_FULL_NAME}
            DEPENDS ${ARGN} ${HOST_SRC} ${CATLASS_INCLUDE_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Compiling single kernel obj ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${OUTPUT_FULL_NAME})

    list(APPEND KERNEL_OBJ_FILES ${OUTPUT_FULL_NAME})
    set(KERNEL_OBJ_FILES "${KERNEL_OBJ_FILES}" PARENT_SCOPE)

    list(APPEND ALL_KERNEL_TARGETS ${NAME})
    set(ALL_KERNEL_TARGETS "${ALL_KERNEL_TARGETS}" PARENT_SCOPE)
endfunction()

# FIXME if aicore-arch is different, maybe crash, to find the solution
catlass_add_kernel(catcoc_allgather_matmul dav-c220 ${CMAKE_CURRENT_SOURCE_DIR}/op_host/catcoc_allgather_matmul_kernel.cpp)
catlass_add_kernel(catcoc_matmul_allreduce dav-c220 ${CMAKE_CURRENT_SOURCE_DIR}/op_host/catcoc_matmul_allreduce_kernel.cpp)

message("Kernel Object Files: ${KERNEL_OBJ_FILES}")

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/libcatcoc_kernel.so
        COMMAND ${CMAKE_BISHENG_COMPILER} ${BISHENG_LINK_OPTIONS} ${KERNEL_OBJ_FILES} --shared -o ${CMAKE_BINARY_DIR}/libcatcoc_kernel.so
        DEPENDS ${KERNEL_OBJ_FILES}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Building shared library libcatcoc_kernel.so"
)

add_custom_target(catcoc_kernel ALL DEPENDS
        ${CMAKE_BINARY_DIR}/libcatcoc_kernel.so
)

install(FILES ${CMAKE_BINARY_DIR}/libcatcoc_kernel.so
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
)
