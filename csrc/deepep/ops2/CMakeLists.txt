cmake_minimum_required(VERSION 3.20.0)
project(opp)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CANN_HOST_ARCH "x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(CANN_HOST_ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported host processor: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CANN82_PATH "${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform/platform/platform_infos_def.h")
set(CANN83_PATH "${ASCEND_CANN_PACKAGE_PATH}/include/platform/platform_infos_def.h")

if (EXISTS "${CANN83_PATH}")
    set(CANN_VERSION_MACRO "USE_CANN83_PATH")
    message(STATUS "Detected CANN 8.3+ include structure, using ${CANN83_PATH}")
elseif (EXISTS "${CANN82_PATH}")
    set(CANN_VERSION_MACRO "USE_CANN82_PATH")
    message(STATUS "Detected CANN 8.2 include structure, using ${CANN82_PATH}")
else()
    message(FATAL_ERROR
            "Cannot find 'platform_infos_def.h' in expected locations.\n"
            "Please verify that CANN is correctly installed and that either:\n"
            "  ${CANN83_PATH}\n"
            "  or\n"
            "  ${CANN82_PATH}\n"
            "exists and contains the header file."
    )
endif()

include(cmake/config.cmake)
include(func.cmake)
include(cmake/intf.cmake)

if (CANN_VERSION_MACRO)
    add_compile_definitions(${CANN_VERSION_MACRO})
endif()

set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
    add_subdirectory(framework)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    add_subdirectory(op_kernel)
endif()
if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/scripts
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/scripts/ ${CMAKE_BINARY_DIR}/scripts/
    COMMAND sed -i "s/vendor_name=customize/vendor_name=${vendor_name}/g" ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
    VERBATIM
)
add_custom_target(modify_vendor ALL DEPENDS ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh)

cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
if (DEFINED CANN_VERSION_MACRO AND NOT "${CANN_VERSION_MACRO}" STREQUAL "")
    set(CANN_VERSION_FLAG "-D${CANN_VERSION_MACRO}")
    message(STATUS "opbuild: Detected CANN_VERSION_MACRO = ${CANN_VERSION_MACRO}")
else()
    set(CANN_VERSION_FLAG "")
    message(WARNING "opbuild: No CANN_VERSION_MACRO defined! Possible #error in .cc files.")
endif()

execute_process(
        COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11
        ${OPBUILD_OPS_SRC}
        -D_GLIBCXX_USE_CXX11_ABI=0
        ${CANN_VERSION_FLAG}
        -I ${ASCEND_CANN_PACKAGE_PATH}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        -L ${ASCEND_CANN_PACKAGE_PATH}/lib64
        -lexe_graph -lregister -ltiling_api
        -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE EXEC_INFO
        ERROR_VARIABLE  EXEC_ERROR
)
if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
endif()

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(gen_version_info ALL
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_version_info.sh ${ASCEND_CANN_PACKAGE_PATH} ${CMAKE_CURRENT_BINARY_DIR}
)

if(NOT ASCEND_PACK_SHARED_LIBRARY)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/scripts/ DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ)

    install(FILES ${CMAKE_SOURCE_DIR}/custom.proto DESTINATION packages OPTIONAL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.info
            DESTINATION packages/vendors/${vendor_name}/)

    # CPack config
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack opp project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack opp project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})

    set(CPACK_PACKAGE_FILE_NAME "custom_opp_${SYSTEM_INFO}_${CMAKE_SYSTEM_PROCESSOR}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif()
