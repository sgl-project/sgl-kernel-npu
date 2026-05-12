
if(DEFINED ASCEND_HOME_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

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

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)


message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")
