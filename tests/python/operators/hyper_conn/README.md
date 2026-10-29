# Hyper-connection operator tests

`test_attn_residual_mix.py` tests the public `attn_res_mix_fused` operator.
Shapes use `(N, C, B, H)`: token count, bank capacity, valid bank rows and
hidden size. The cases borrow model dimensions and calling patterns from
SGLang; they are synthetic scenarios, not captured runtime traces.

## Coverage

Default tests cover accuracy, selected performance cases, dtype combinations
and empty inputs. Required additions include:

- `H=7168`, `C=8`, `B=1..8`, including the softmax boundary at `B=7/8`.
- Representative batch/prefill sizes `N=128/512`.
- Scheduling boundaries `N=M-1`, `M`, `M+1` and `2M+1`, where `M` is the
  current device's actual vector-core count.
- SP-style contiguous token slices with a nonzero starting offset.

Optional tests use `@pytest.mark.extra_checks` and are excluded by default:

- Larger token counts `N=2048/2049`.
- Hidden sizes `H=4096/8191/8192/8193` and deeper banks with `B=9/15/16/17`.
- Large-hidden-size padded layouts and mixed-dtype combinations.
- Invalid-input and output-contract checks, including empty-output identity.

The larger stress cases require more NPU memory, especially for the FP32
reference's intermediate tensors.

## Run

Run from the repository root in the configured NPU test environment:

```bash
# Required/default cases.
python -m pytest tests/python/operators/hyper_conn -v

# Optional supplementary cases only.
python -m pytest tests/python/operators/hyper_conn -m extra_checks -v

# All cases.
python -m pytest tests/python/operators/hyper_conn -m "" -v
```
