# attention

处理注意力页表等地址映射元数据，不读写 KV-cache 内容。

公共入口：`sgl_kernel_npu.operators.attention`。

- [build_pooled_page_table_64](_build_pooled_page_table_64.md)

`auto` 与 `torch_native` 指向同一 Torch 实现，无运行时回退。输入只读，输出重新分配；导入公共 registry 需要 NPU 架构检测，后端按首次调用延迟加载。
