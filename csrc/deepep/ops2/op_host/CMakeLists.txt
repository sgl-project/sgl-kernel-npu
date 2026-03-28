# =====================================================
# Detect CANN version include path globally
# =====================================================
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

#1、使用npu_op_code_gen生成aclnn单算子调用代码、入图所需的原型定义代码等。
npu_op_code_gen(
     SRC ${ops_srcs}                                                           # Host侧实现文件集合
     PACKAGE ${package_name}                                                   # 与npu_op_package的package_name对应
     COMPILE_OPTIONS -g                                                        # 编译选项
     OUT_DIR ${ASCEND_AUTOGEN_PATH}                                            # 指定生成文件的保存路径
)
#2、单算子调用场景，使用npu_op_library编译aclnn单算子调用库。
file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnnInner_*.cpp)                # 将通过npu_op_code_gen生成的aclnn相关源文件添加至源文件列表中

set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)    # 设置对应的文件属性为编译过程中生成的文件
npu_op_library(cust_opapi ACLNN                                                # 指定library名称与类型，以及参与编译的源文件
     ${autogen_aclnn_src}
)
target_compile_options(cust_opapi PRIVATE                                      # 为对应library添加编译选项，与通用的cmake语法一致
     -fvisibility=hidden
)

#3、算子入图场景，使用npu_op_library编译算子入图所需的算子原型库。
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)                        # 将通过npu_op_code_gen生成的原型定义相关源文件添加至源文件列表中
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)            # 设置对应的文件属性为编译过程中生成的文件
npu_op_library(cust_op_proto GRAPH                                             # 指定library名称与类型，以及参与编译的源文件
     ${ops_srcs}
     ${proto_src}
)
target_compile_options(cust_op_proto PRIVATE                                   # 为对应library添加编译选项，与通用的cmake语法一致
     -fvisibility=hidden
)

#4、使用npu_op_library编译Tiling相关库。
file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)                  # 将通过npu_op_code_gen生成的Tiling相关源文件添加至源文件列表中
                                                                               # 若算子配置了EnableFallBack会生成该文件
set_source_files_properties(${fallback_src} PROPERTIES GENERATED TRUE)         # 设置对应的文件属性为编译过程中生成的文件
npu_op_library(cust_optiling  TILING                                           # 指定library名称与类型，以及参与编译的源文件
     ${ops_srcs}
     ${fallback_src}
)
target_compile_options(cust_optiling PRIVATE                                   # 为对应library添加编译选项，与通用的cmake语法一致
     -fvisibility=hidden
)

#5、使用npu_op_package_add添加上述Host侧库到对应package中。
npu_op_package_add(${package_name}                                             # 与npu_op_package的package_name对应
     LIBRARY                                                                   # 添加Host侧相关library，library名称与前述步骤的library名称对应
          cust_optiling
          cust_opapi
          cust_op_proto
)