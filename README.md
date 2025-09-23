# sgl-kernel-npu
SGLang kernel library for NPU
Contribution guide refer to [Contribution Guide](docs/developer_guide/contribution_guide.md).

## Performance
### Normal kernels with pure HCCS

We follow the DeepSeek-V3/R1 pretraining setting (4096 tokens per batch, 7168 hidden, top-8 experts, BF16 dispatching and BF16 combining).

| Type      | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
| --------- | ------------ | -------------------- | ----------- | -------------------- |
| Intranode | 8            | 146 GB/s (HCCS)      | 8           | 125 GB/s (HCCS)      |
| Intranode | 16           | 107 GB/s (HCCS)      | 16          | 103 GB/s (HCCS)      |
| Internode | 32           | 102 GB/s (HCCS)      | 32          | 95 GB/s (HCCS)       |
| Internode | 64           | 81 GB/s (HCCS)       | 64          | 91 GB/s (HCCS)       |
| Internode | 128          | 57 GB/s (HCCS)       | 128         | 81 GB/s (HCCS)       |

### Low-latency kernels with pure HCCS

We follow a typical DeepSeek-V3/R1 production setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatching and BF16 combining).

| Dispatch #EP | Latency | Bandwidth      | Combine #EP | Latency | Bandwidth       |
| ------------ | ------- | -------------- | ----------- | ------- | --------------- |
| 8            | 175 us  | 43 GB/s (HCCS) | 8           | 116 us  | 126 GB/s (HCCS) |
| 16           | 184 us  | 41 GB/s (HCCS) | 16          | 136 us  | 108 GB/s (HCCS) |
| 32           | 200 us  | 38 GB/s (HCCS) | 32          | 149 us  | 99 GB/s (HCCS)  |
| 64           | 201 us  | 38 GB/s (HCCS) | 64          | 149 us  | 99 GB/s (HCCS)  |
| 128          |         |                | 128         |         |  
