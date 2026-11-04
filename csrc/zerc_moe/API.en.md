# ZeRCMoE source and build

Contributors: chengyue, zhaoyuxin, hanggongmao, chenyukai.

The source under `source/` originates from the `mega_moe` branch of
[ops-transformer](https://gitcode.com/hanggongmao1/ops-transformer) at commit
`d4c5f65d93591f26895c126fc590f0d622440819`. Copyright notices and the CANN
Open Software License Agreement 2.0 are retained; see [LICENSE](LICENSE).
The repository root's MIT license does not replace that license for these files.

The imported operator has distinct CANN and Torch names to coexist with the
existing MegaMoe implementation. The public API, build selection, input
constraints, and communication buffer lifetime are documented in
[zerc_moe](../../docs/api/en/operators/moe/zerc_moe.md).

Build, device execution, and runtime coexistence are not yet validated. The
optional standard build uses a pinned ops-transformer MC2 build framework; the
standalone `python3 csrc/zerc_moe/build.py --plan` shows its corresponding
recipe without compiling or installing anything.
The standalone wheel retains the upstream `npu_ops_transformer` package name.
Do not import it together with the standard wheel's ZeRCMoE binding in the same
process: both register the `zerc_moe` Torch namespace.
