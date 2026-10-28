# SGL-KERNEL-NPU Documentation

We recommend new contributors start from writing documentation, which helps you quickly understand codebase.

## API Documentation

[API index (English)](api/en/API_list.md) | [API 索引（中文）](api/zh/API_list.md)

Operator documentation is organized by `api/<language>/operators/<category>/<operator>.md`.

Module guides:

- attention: [English](api/en/operators/attention/README.md) | [中文](api/zh/operators/attention/README.md)
- attentions: [English](api/en/operators/attentions/README.md) | [中文](api/zh/operators/attentions/README.md)
- indexer: [English](api/en/operators/indexer/README.md) | [中文](api/zh/operators/indexer/README.md)
- mamba: [English](api/en/operators/mamba/README.md) | [中文](api/zh/operators/mamba/README.md)
- mem_cache: [English](api/en/operators/mem_cache/README.md) | [中文](api/zh/operators/mem_cache/README.md)

## Docs Workflow

### Update Documentation

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```
---
