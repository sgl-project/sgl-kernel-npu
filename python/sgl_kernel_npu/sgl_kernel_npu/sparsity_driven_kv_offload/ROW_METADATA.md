# Sparse KV row metadata

`row_metadata` provides tensor-only Triton operators. Framework-specific batch,
forward-mode, CPU layout validation and workspace ownership remain in SGLang.
The operators do not copy KV payloads or publish MemFabric fences.

- `mtp_rows` creates request/position/valid rows, or publication source/destination,
  positions and validity when `publication_rows` is set. Pass acceptance counts
  only for draft-extend. Padding rows are invalid; publication also checks physical
  host capacity. The caller supplies same-device int32/int64 input metadata.
- `prefill_plan` expands validated request lengths through the physical token map.
  The caller supplies CPU-derived total and maximum lengths, request-row count
  and context capacity without synchronizing device metadata.
- `prefill_current_mapping` accepts uploaded contiguous int64 metadata in four
  batch-sized segments: current-row ends, full-KV starts, prefixes and live ends.
  Three ordered kernels initialize counts, count current physical rows, and reject
  duplicate/mismatched rows. Source maps, locations and masks must be contiguous.
- `copy_rows_metadata` creates byte-address/length/count descriptors for two split
  KV components. Payload widths and dtypes must agree between each source and
  destination. Invalid rows receive zero lengths. Runtime base addresses and
  address arithmetic retain 64-bit precision. Aligned row counts use separate
  component grids; other counts use a flat contiguous descriptor layout.

All launches use the caller's current stream and can be captured. Tensor contents
are read again on graph replay; storage pointers remain the captured pointers.
The framework's complete prefill-mapping adapter also uploads CPU layout metadata,
which must be prepared before graph capture when GLOBAL capture disallows that upload.

Run `tests/python/sgl_kernel_npu/test_sparse_kv_row_metadata.py` on an initialized
Ascend environment. It covers invalid/padded MTP rows, acceptance masks, changed
graph inputs, aligned/unaligned MF descriptors and duplicate physical mappings.
