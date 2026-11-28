# sgl-kernel-npu domain knowledge base

Verified domain knowledge for AI agents. The entry point is `AGENTS.md` at the repository root; this file is the index and the writing convention.

This is not a documentation site: it only answers "what must I know before making this kind of change". Interfaces and usage stay in `docs/api/` and the operator READMEs.

## Layout

```text
.agent/knowledge/
├── README.md            this file: index and conventions
└── utils/               shared facilities under csrc/utils
    ├── README.md        domain scope and open topics
    └── dtype.md         DTYPE enum, semantics, extension flow, dav-2201 support
```

## Index

| Domain | File | Summary | Read when |
| --- | --- | --- | --- |
| utils | [utils/dtype.md](utils/dtype.md) | `sgl::kernel_npu::utils::DTYPE` values, cross-layer passing, extension steps, measured dav-2201 support matrix | Adding/changing a dtype; passing a dtype across layers; checking hardware support for FP32/FP16/BF16 |
| utils | [utils/README.md](utils/README.md) | What the utils domain covers, what is recorded and what is open | Adding knowledge to the utils domain |

## Writing conventions

1. **One topic per file**: `domain/topic.md` (lower case, hyphen-separated). Every domain directory carries a `README.md` describing its scope.
2. **Four required parts**: scope (branch / device / CANN version), conclusion, evidence, change log. Evidence means `path:line`, a reproducible command or a measured number; anything inferred must be marked "unverified".
3. **Verified facts only**: when a file disagrees with the source, the source wins and the file is fixed in the same change. Expired conclusions are updated or removed, never kept as historical claims.
4. **Cite, do not paste**: reference the key lines instead of copying functions. No machine-specific paths, credentials, one-off logs or performance transcripts.
5. **Time-bounded**: every conclusion states the environment it was measured in; re-verify on another device or CANN version.
6. **English only**: knowledge files and `AGENTS.md` are written in English, with code identifiers kept verbatim.

## Adding knowledge

1. Create `domain/topic.md` under the matching domain (create the directory plus its `README.md` for a new domain).
2. Update the index above and the index table in section 4 of `AGENTS.md`.
3. Run `pre-commit run --files .agent/knowledge/**`.
