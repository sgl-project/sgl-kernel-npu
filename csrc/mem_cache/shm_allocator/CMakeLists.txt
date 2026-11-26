sgl_register_operator(
    ID shm_allocator
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES shm_allocator.cpp shm_allocator_op.cpp
    TORCH_ADAPTER torch_adapter.cpp
)

include("${CMAKE_CURRENT_LIST_DIR}/../shm_allocator_driver.cmake")
# Both build targets are created after operator discovery.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL sgl_configure_shm_allocator_driver
    shm_allocator "${PROJECT_SOURCE_DIR}/mem_cache/shm_allocator/shm_allocator.cpp")
