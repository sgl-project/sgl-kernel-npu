"""Check staged operator registrations before publishing a binary wheel."""

import argparse
import importlib
import json
import sys
from pathlib import Path

# These migrated wrappers call Torch packets directly, rather than Python APIs.
RELEASED_PACKETS = {
    "sparse_flash_mla",
    "sparse_flash_mla_metadata",
    "mixed_quant_sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "mhc_pre_sinkhorn",
    "mhc_post",
    "inplace_partial_rotary_mul",
    "kv_compress_epilog",
    "indexer_quant_cache",
}


def verify(package):
    package = Path(package).resolve()
    sys.path.insert(0, str(package.parent))
    runtime = importlib.import_module("sgl_kernel_npu")
    if Path(runtime.__file__).resolve() != package / "__init__.py":
        raise RuntimeError("Binding verification imported an installed package")
    import torch

    manifest = json.loads((package / "_build_manifest.json").read_text())
    dependencies = manifest.get("third_party", [])
    bindings = []

    def check(namespace, name, required=(), dispatch=None):
        op = getattr(getattr(torch.ops, namespace), name, None)
        if op is None:
            raise RuntimeError("Staged wheel is missing " + namespace + "." + name)
        qualified = namespace + "::" + name
        if not torch._C._dispatch_has_kernel(qualified):
            raise RuntimeError("Staged binding has no implementation: " + qualified)
        if (
            dispatch
            and dispatch != "CatchAll"
            and not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, dispatch)
        ):
            raise RuntimeError(
                "Staged binding lacks " + dispatch + " implementation: " + qualified
            )
        if isinstance(op, torch._ops.OpOverloadPacket):
            parameters = {arg.name for arg in op.default._schema.arguments}
            missing = set(required) - parameters
            if missing:
                raise RuntimeError(
                    namespace
                    + "."
                    + name
                    + " lacks parameters: "
                    + ", ".join(sorted(missing))
                )
        bindings.append(namespace + "." + name)

    native = set(manifest.get("native_operators", []))
    for name in native:
        check("npu", name, dispatch=manifest["native_dispatch_keys"][name])

    for item in dependencies:
        if not item.get("python_bindings", True):
            continue  # Its ACLNN/AICPU payload is checked by the vendor builder.
        name = item["python_package"]
        module = importlib.import_module("sgl_kernel_npu._third_party." + name)
        if item["builder"] == "ops_transformer":
            module = importlib.import_module(module.__name__ + ".ops")
        for build_name in item["ops"]:
            op_name = item.get("python_exports", {}).get(build_name, build_name)
            if not callable(getattr(module, op_name, None)):
                raise RuntimeError(
                    "Staged Python binding is missing " + name + "." + op_name
                )
            if item["builder"] == "ops_transformer":
                if op_name in RELEASED_PACKETS:
                    check("cann_ops_transformer", op_name)
            else:
                check(
                    name, op_name, dispatch=item.get("dispatch_keys", {}).get(op_name)
                )

    names = {item["name"] for item in dependencies}
    versions = {item["cann_version"] for item in dependencies}
    if len(versions) > 1:
        raise RuntimeError("Mixed SDK profiles in one wheel")
    version = next(iter(versions), None)
    arch = manifest["arch_version"]
    if version and version.startswith("9.1."):
        required = {"sgl_cann_recipes_infer", "sgl_ops_transformer_experimental"}
        required_native = (
            {"compressor", "sparse_attn_sharedkv", "sparse_attn_sharedkv_metadata_host"}
            if arch == "2201"
            else {
                "kv_quant_sparse_attn_sharedkv",
                "kv_quant_sparse_attn_sharedkv_metadata_host",
                "kv_compress_epilog",
            }
        )
    elif version and version.startswith("9.2."):
        required = {"sgl_ops_transformer"}
        required_native = {"compressor"}
    else:
        required = None
        required_native = set()
    ready = required is not None and required <= names and required_native <= native
    if ready and version.startswith("9.2."):
        # This is a base torch_npu API, not a separately installed op plugin.
        check("npu", "npu_moe_gating_top_k", required=("input_ids", "tid2eid"))
    report = {
        "bindings": sorted(set(bindings)),
        "migration_dependencies_complete": ready,
    }
    (package / "_binding_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("STAGED_BINDINGS: " + json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    verify(parser.parse_args().package)
