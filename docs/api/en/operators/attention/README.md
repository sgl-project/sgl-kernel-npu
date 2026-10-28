# attention

Transform attention address-mapping metadata such as page tables, without reading or writing KV-cache contents.

Public import: `sgl_kernel_npu.operators.attention`.

- [build_pooled_page_table_64](_build_pooled_page_table_64.md)

`auto` and `torch_native` alias one Torch implementation, with no runtime fallback. Inputs are read only and output is allocated. Registry import requires NPU architecture detection; the backend loads lazily on first execution.
