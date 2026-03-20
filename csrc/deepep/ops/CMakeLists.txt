cmake_minimum_required(VERSION 3.20.0)
project(opp)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CANN_HOST_ARCH "x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(CANN_HOST_ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported host processor: ${CMAKE_SYSTEM_PROCESSOR}. Please specify a valid architecture.")
endif()

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

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

if(NOT vendor_name)
    message(FATAL_ERROR "vendor_name is not defined. Please set vendor_name before this command.")
endif()
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/scripts
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/scripts/ ${CMAKE_BINARY_DIR}/scripts/
    COMMAND sed -i "s/vendor_name=customize/vendor_name=${vendor_name}/g" ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh
    VERBATIM
)
add_custom_target(modify_vendor ALL DEPENDS ${CMAKE_BINARY_DIR}/scripts/install.sh ${CMAKE_BINARY_DIR}/scripts/upgrade.sh)

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
