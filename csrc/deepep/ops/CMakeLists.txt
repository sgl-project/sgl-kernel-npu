cmake_minimum_required(VERSION 3.16.0)
project(opp)

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_HOME_PATH "${ASCEND_CANN_PACKAGE_PATH}")
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()

# Detect CANN version
if(EXISTS ${ASCEND_HOME_PATH}/version.info)
    file(READ ${ASCEND_HOME_PATH}/version.info CANN_VERSION_FILE)
    string(REGEX MATCH "Version=([0-9]+\\.[0-9]+)" _ ${CANN_VERSION_FILE})
    set(CANN_VERSION ${CMAKE_MATCH_1})
    message(STATUS "Detected CANN version: ${CANN_VERSION}")
elseif(EXISTS ${ASCEND_HOME_PATH}/../version.info)
    file(READ ${ASCEND_HOME_PATH}/../version.info CANN_VERSION_FILE)
    string(REGEX MATCH "Version=([0-9]+\\.[0-9]+)" _ ${CANN_VERSION_FILE})
    set(CANN_VERSION ${CMAKE_MATCH_1})
    message(STATUS "Detected CANN version: ${CANN_VERSION}")
else()
    # Try to get version from path
    string(REGEX MATCH "cann-([0-9]+\\.[0-9]+)" _ ${ASCEND_HOME_PATH})
    if(CMAKE_MATCH_1)
        set(CANN_VERSION ${CMAKE_MATCH_1})
        message(STATUS "Detected CANN version from path: ${CANN_VERSION}")
    else()
        message(WARNING "Could not detect CANN version, assuming 8.3")
        set(CANN_VERSION "8.3")
    endif()
endif()

# Set CANN version macro based on detected version
if(CANN_VERSION VERSION_EQUAL "8.2")
    set(CANN_VERSION_MACRO "USE_CANN82_PATH")
elseif(CANN_VERSION VERSION_EQUAL "8.3")
    set(CANN_VERSION_MACRO "USE_CANN83_PATH")
elseif(CANN_VERSION VERSION_EQUAL "8.5" OR CANN_VERSION VERSION_GREATER "8.3")
    # CANN 8.5 and later use 8.3 API
    set(CANN_VERSION_MACRO "USE_CANN83_PATH")
    message(STATUS "Using CANN 8.3 API for CANN ${CANN_VERSION}")
else()
    message(FATAL_ERROR "Unsupported CANN version: ${CANN_VERSION}")
endif()

message(STATUS "CANN version macro: ${CANN_VERSION_MACRO}")

if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

list(APPEND CMAKE_MODULE_PATH "${ASCENDC_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${ASCENDC_CMAKE_DIR}")

# Try to find ASC package
find_package(ASC QUIET)

if(NOT ASC_FOUND)
    message(WARNING "ASC package not found in ${ASCENDC_CMAKE_DIR}")
    message(STATUS "Attempting manual configuration for CANN ${CANN_VERSION}")
    
    # Manually set variables that ASCConfig.cmake would normally set
    # This is a fallback for CANN versions where ASCConfig.cmake might not be available
    if(NOT DEFINED vendor_name)
        set(vendor_name "hwcomputing")
    endif()
    
    # Include necessary cmake modules directly
    if(EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
        include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
    elseif(EXISTS "${ASCENDC_CMAKE_DIR}/CMakeLists.txt")
        # Some CANN versions use CMakeLists.txt instead
        message(STATUS "Found CMakeLists.txt in ASCENDC_CMAKE_DIR")
    else()
        message(FATAL_ERROR "Cannot find ascendc.cmake or CMakeLists.txt in ${ASCENDC_CMAKE_DIR}")
    endif()
else()
    message(STATUS "ASC package found successfully")
endif()

set(package_name ${vendor_name})

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}/
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
    add_subdirectory(framework)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    add_subdirectory(op_kernel)
endif()
