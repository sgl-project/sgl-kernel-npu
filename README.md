# sgl-kernel-npu

SGLang kernel library for NPU
Contribution guide refer to [Contribution Guide](docs/developer_guide/contribution_guide.md).

## Quick start

DeepEP-Ascend: Ascend Implementation of DeepEP. [README](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)

SGL-Kernel-NPU: Other SGLang Kernels for Ascend NPU. [README](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)

## DeepEP-Ascend Performance

### Normal kernels with pure HCCS

We test normal kernels on A3 384 SuperPOD. And we follow the DeepSeek-V3/R1 pretraining setting (4096 tokens per batch, 7168 hidden, top-8 experts, INT8 dispatching and BF16 combining).

| Type      | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
| --------- | ------------ | -------------------- | ----------- | -------------------- |
| Intranode | 8            | 146 GB/s (HCCS)      | 8           | 125 GB/s (HCCS)      |
| Intranode | 16           | 107 GB/s (HCCS)      | 16          | 103 GB/s (HCCS)      |
| Intranode | 32           | 102 GB/s (HCCS)      | 32          | 95 GB/s (HCCS)       |
| Intranode | 64           | 81 GB/s (HCCS)       | 64          | 91 GB/s (HCCS)       |
| Intranode | 128          | 57 GB/s (HCCS)       | 128         | 81 GB/s (HCCS)       |

### Low-latency kernels with pure HCCS

We test low-latency kernels on A3 384 SuperPOD. And we follow a typical DeepSeek-V3/R1 production setting (128 tokens per batch, 7168 hidden, top-8 experts, INT8 dispatching and BF16 combining).

| Dispatch #EP | Latency | Bandwidth      | Combine #EP | Latency | Bandwidth       |
| ------------ | ------- | -------------- | ----------- | ------- | --------------- |
| 8            | 132 us  | 58 GB/s (HCCS) | 8           | 126 us  | 116 GB/s (HCCS) |
| 16           | 139 us  | 55 GB/s (HCCS) | 16          | 135 us  | 109 GB/s (HCCS) |
| 32           | 153 us  | 49 GB/s (HCCS) | 32          | 151 us  | 97 GB/s (HCCS)  |
