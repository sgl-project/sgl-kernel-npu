# API Index

| API | Supported backends |
| --- | --- |
| [rmsnorm_bias](operators/norm/rmsnorm_bias.md) | `triton`, `torch_native` |
| [l1_norm](operators/norm/l1_norm.md) | `triton`, `torch_native` |
| [l2norm_fwd](operators/norm/l2norm_fwd.md) | `triton`, `torch_native` |
| [layer_norm_fwd](operators/norm/layer_norm_fwd.md) | `triton`, `torch_native` |

All listed APIs default to `impl="auto"`, currently an alias of the `"triton"` entry point. Execution failures do not trigger an automatic fallback. See each API page for input constraints and hardware validation status.
