cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(sgl-kernel-npu LANGUAGES CXX)

option(BUILD_PYTHON "build python SDK" ON)
option(BUILD_TESTS  "build test or not" OFF)

option(BUILD_DEEPEP_MODULE "build deepep" ON)
option(BUILD_KERNELS_MODULE  "build kernels" ON)
option(BUILD_CATLASS_MODULE  "build catlass ops within kernels" OFF)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-hno-unused-parameter -lno-unused-function -Wunused-value -Wcast-align)
add_compile_options(-Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wextra)

if (BUILD_CATLASS_MODULE)
    add_compile_definitions(BUILD_CATLASS_MODULE)
    set(CATLASS_DIR "${PROJECT_SOURCE_DIR}/3rdparty/catlass")  # specific your catlass path here
    message(STATUS "[CATLASS] ${CATLASS_DIR}")
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -fPIE -fPIC -ftrapv -s)
    message(STATUS "build type set to RELEASE")
else ()
    add_compile_options(-g -rdynamic)
endif ()

if(DEFINED ENV{DEBUG_MODE})
    if("$ENV{DEBUG_MODE}" STREQUAL "ON")
        add_compile_definitions(DEBUG_MODE)
        message(STATUS "Debug logging enabled from environment")
    endif()
endif()

set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR}/csrc)
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)

include(cmake/config_git_last_commit.cmake)
include(cmake/config_envs.cmake)
include(cmake/config_ascend.cmake)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found, proceeding with normal compilation")
endif()

if (BUILD_KERNELS_MODULE)
    add_subdirectory(csrc)
endif ()

if (BUILD_DEEPEP_MODULE)
    add_subdirectory(csrc/deepep)
endif ()
