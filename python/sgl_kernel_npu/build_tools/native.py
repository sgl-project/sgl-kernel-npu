"""Select and register repository-owned kernels from the current checkout."""

import argparse
import re
from pathlib import Path

DISPATCH_KEYS = {
    "compressor": "PrivateUse1",
    "sparse_attn_sharedkv": "PrivateUse1",
    "sparse_attn_sharedkv_metadata_host": "CPU",
    "kv_quant_sparse_attn_sharedkv": "PrivateUse1",
    "kv_quant_sparse_attn_sharedkv_metadata_host": "CatchAll",
    "kv_compress_epilog": "PrivateUse1",
}
SUPPORTED = {
    "2201": (
        "compressor",
        "sparse_attn_sharedkv",
        "sparse_attn_sharedkv_metadata_host",
    ),
    "3510": (
        "compressor",
        "kv_quant_sparse_attn_sharedkv",
        "kv_quant_sparse_attn_sharedkv_metadata_host",
        "kv_compress_epilog",
    ),
}


def split_operators(operators, arch_version, *, include_native=True):
    """Keep ACLNN selection separate from existing direct-launch Torch kernels."""
    supported = SUPPORTED[str(arch_version)]
    if operators == ("ALL",):
        return operators, supported if include_native else ()
    native = tuple(name for name in operators if name in DISPATCH_KEYS)
    unsupported = set(native) - set(supported)
    if unsupported:
        raise ValueError(
            "Native operators unsupported on ARCH {}: {}".format(
                arch_version, ", ".join(sorted(unsupported))
            )
        )
    if native and not include_native:
        raise ValueError("Direct-launch native operators require --artifact wheel")
    return tuple(name for name in operators if name not in DISPATCH_KEYS), native


def native_soc(arch_version):
    from build_tools.third_party import detect_soc_version

    soc = detect_soc_version()
    targets = {
        ("2201", "ascend910b"): "Ascend910B1",
        ("2201", "ascend910_93"): "Ascend910_9382",
        ("3510", "ascend950"): "Ascend950PR_9599",
    }
    try:
        return targets[(str(arch_version), soc)]
    except KeyError as error:
        raise RuntimeError(
            "Detected native SoC does not match --arch-version"
        ) from error


def write_registration(source_root, output, arch_version, operators):
    """Reuse current schemas and C++ entry points in the existing npu namespace."""
    _, selected = split_operators(tuple(operators), arch_version)
    if (
        not selected
        or len(selected) != len(operators)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Registration requires supported native operators only")
    declarations = (Path(source_root) / "csrc/pytorch_extensions.cpp").read_text()
    schemas = []
    groups = {}
    for name in selected:
        matches = re.findall(
            r'm\.def\(\s*"' + re.escape(name) + r"\([\s\S]*?\);", declarations
        )
        if len(matches) != 1:
            raise RuntimeError("Expected one repository native schema for " + name)
        schemas.append(matches[0])
        groups.setdefault(DISPATCH_KEYS[name], []).append(name)
    Path(output).write_text(
        '#include <ATen/ATen.h>\n#include <torch/library.h>\n#include "sgl_kenel_npu_ops.h"\n'
        + "TORCH_LIBRARY_FRAGMENT(npu, m) {\n"
        + "\n".join(schemas)
        + "\n}\n"
        + "\n".join(
            "TORCH_LIBRARY_IMPL(npu, "
            + dispatch
            + ", m) {\n"
            + "\n".join(
                'm.impl("' + name + '", TORCH_FN(sglang::npu_kernel::' + name + "));"
                for name in names
            )
            + "\n}\n"
            for dispatch, names in groups.items()
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch-version", required=True, choices=SUPPORTED)
    parser.add_argument("--ops", nargs="+", required=True)
    args = parser.parse_args()
    write_registration(args.source_root, args.output, args.arch_version, args.ops)
