add_executable(test_kv_h2d_copy_plan test_kv_h2d_copy_plan.cpp)
target_include_directories(test_kv_h2d_copy_plan PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc/utils
    ${PROJECT_SOURCE_DIR}/csrc/transfer_kv_dim_exchange/op_host)
add_test(NAME kv_h2d_copy_plan COMMAND test_kv_h2d_copy_plan)
