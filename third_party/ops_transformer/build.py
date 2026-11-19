"""Build and package the custom ops-transformer vendor and Python bindings."""

import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

from build_tools.cann import run
from build_tools.third_party import extract_python, read_json, resolve_soc_version

from third_party.opapi import isolate_staged_opapis


def configure(config, jobs):
    """Resolve ops-transformer's names and command placeholders."""
    config = dict(config)
    config["soc"] = resolve_soc_version(config["arch_version"])
    config["vendor_name"] = "vendor_" + config["name"]
    config["python_package"] = config["name"]
    config["python_bindings"] = config.get("python_bindings", True)
    if not isinstance(config["python_bindings"], bool):
        raise ValueError("python_bindings must be a boolean")
    a3_ops = config.pop("a3_add_ops", [])
    a3_exclude = config.pop("a3_exclude_ops", [])
    if config["soc"] == "ascend910_93":
        config["ops"] = [op for op in config["ops"] if op not in a3_exclude] + a3_ops
        if a3_exclude:
            config["has_aicpu"] = False
        if a3_ops:
            config["a3_indexer_bindings"] = True
            config["has_aicpu"] = True
    values = dict(config, jobs=jobs or 8)
    values["ops"] = ",".join(config["ops"])
    config["commands"] = [
        [arg.format(**values) for arg in command] for command in config["commands"]
    ]
    return config


def stage_python(wheel, item, package):
    name = item["name"]
    destination = package / "_third_party" / item["python_package"]
    # Upstream adds its own package prefix; only its temporary wheel uses it.
    upstream_package = "cann_ops_transformer_" + item["vendor_name"]
    with zipfile.ZipFile(wheel) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist()}
    # The pinned 9.1 setup.py predates vendor-suffixed packages and op folders.
    if upstream_package not in roots:
        upstream_package = "cann_ops_transformer"
    extract_python(wheel, upstream_package, destination)
    ops_root = destination / "ops"
    modules = sorted(
        ".".join(path.parent.relative_to(ops_root).parts)
        for path in ops_root.glob("*/*/__init__.py")
        if path.parent.name in item["ops"]
    )
    if not modules:
        modules = sorted(
            path.stem for path in ops_root.glob("*.py") if path.stem in item["ops"]
        )
    if not modules:
        raise RuntimeError("No selected Python bindings found in upstream wheel")
    qualified = "sgl_kernel_npu._third_party." + name
    # Upstream's builder still contains absolute imports of the unsuffixed package.
    # Rewrite Python module paths; Torch operator namespaces must remain intact.
    pattern = (
        r"(?m)^(\s*(?:from|import)\s+)(?:"
        + re.escape(upstream_package)
        + r"|cann_ops_transformer)(?=[.\s])"
    )
    for path in destination.rglob("*.py"):
        source = re.sub(pattern, lambda m: m[1] + qualified, path.read_text())
        # The flat 9.1 builder references the package by its original root name.
        source = source.replace(
            "import " + qualified + "\n",
            "import " + qualified + " as cann_ops_transformer\n",
        )
        # CompressorV2 creates ops.ds41 using a Python module-cache lookup.
        source = source.replace(
            'sys.modules["cann_ops_transformer.ops"]',
            'sys.modules["{}.ops"]'.format(qualified),
        )
        path.write_text(source)
    if item.get("a3_indexer_bindings"):
        binding = (
            destination
            / "ops/attention/quant_lightning_indexer_v2/quant_lightning_indexer.py"
        )
        source = binding.read_text()
        start = source.index("            if return_value:")
        end = source.index(
            "            return (sparse_indices_out, sparse_values_out)", start
        )
        source = (
            source[:start]
            + source[start:end].replace(
                "dtype=torch.bfloat16",
                "dtype=torch.float32 if return_value else torch.bfloat16",
            )
            + source[end:]
        )
        source = source.replace(
            'return torch.empty((QLI_METADATA_SIZE), dtype=torch.int32, device="npu")',
            "device = next((t.device for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, "
            'seqused_k, cmp_residual_k) if t is not None), "npu")\n'
            "            return torch.empty(QLI_METADATA_SIZE, dtype=torch.int32, device=device)",
        )
        binding.write_text(source)
    # Do not discover globally installed entry points or import unrelated operators.
    (destination / "__init__.py").write_text("from .ops import *  # noqa: F403\n")
    (destination / "ops/__init__.py").write_text(
        "".join(
            "from .{} import *  # noqa: F403\n".format(module) for module in modules
        )
    )


def inspect_vendor(vendor, item):
    """Validate the custom ops-transformer layout, including selected op coverage."""
    vendor = Path(vendor)

    def require(path):
        if (
            not path.is_file()
            or not path.stat().st_size
            or vendor.resolve() not in path.resolve().parents
        ):
            raise RuntimeError("Missing or external vendor resource: " + str(path))
        return path

    def normalize(name):
        return name.replace("_", "").lower()

    require(vendor / "op_api/lib/libcust_opapi.so")
    covered = set()
    cpu_configs = sorted((vendor / "op_impl/cpu/config").glob("*.json"))
    cpu_count = 0
    for path in cpu_configs:
        declarations = read_json(require(path))
        for name, declaration in declarations.items():
            info = declaration.get("opInfo", {})
            library = info.get("kernelSo")
            if not library:
                raise RuntimeError("Missing AICPU kernelSo in " + str(path))
            if Path(library).name != library:
                raise RuntimeError("Invalid AICPU library name: " + library)
            matches = list((vendor / "op_impl/cpu").rglob(library))
            if len(matches) != 1:
                raise RuntimeError("Expected one AICPU library: " + library)
            require(matches[0])
            covered.add(normalize(name))
            cpu_count += 1
    if item.get("has_aicpu", False) != bool(cpu_count):
        raise RuntimeError("has_aicpu does not match installed AICPU declarations")

    kernel = vendor / "op_impl/ai_core/tbe/kernel"
    for op in item["ops"]:
        config = kernel / "config" / item["soc"] / (op + ".json")
        # AscendC template kernels use an _apt suffix in installed paths.
        if not config.exists():
            config = config.with_name(op + "_apt.json")
        if config.exists():
            read_json(require(config))
            binaries = sorted((kernel / item["soc"] / config.stem).glob("*.o"))
            if not binaries:
                raise RuntimeError("No AICore binaries for " + op)
            for binary in binaries:
                require(binary)
                read_json(require(binary.with_suffix(".json")))
            covered.add(normalize(op))
        if normalize(op) not in covered:
            raise RuntimeError("No installed AICore/AICPU implementation for " + op)


def fix_a3_indexer_workspace(source):
    # The arch22 kernels reduce heads before writing mm1ResGm, so its rows
    # are s1BaseSize (4), not mBaseSize (256). Keep the separate FP16 weight
    # workspace, sized for the largest supported head group (64).
    old = "uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;"
    new = (
        "uint32_t mm1ResSize = S1_BASE_SIZE * S2_BASE_SIZE;\n"
        "    workspaceSize += 16ULL * M_BASE_SIZE * DOUBLE_BUFFER * "
        "sizeof(uint16_t) * aicNum;"
    )
    for op in ("quant_lightning_indexer_v2", "quant_sparse_lightning_indexer"):
        path = source / "experimental/attention" / op / "op_host" / (op + "_tiling.cpp")
        text = path.read_text()
        if text.count(old) != 1 and new not in text:
            raise RuntimeError("Unexpected A3 indexer workspace formula: " + str(path))
        path.write_text(text.replace(old, new))


def optimize_a3_consumer_mask(source):
    """Reuse the pinned BF16 consumer's row-local membership-table strategy."""
    path = source / (
        "experimental/attention/quant_sparse_lightning_indexer/op_kernel/arch22/"
        "quant_sparse_lightning_indexer_service_vector_arch22.h"
    )
    text = path.read_text()
    marker = "// SGL A3 consumer: row-local block membership table."
    if marker in text:
        return
    anchor = "    int32_t tileBlockBase = cuBaseS2Idx / blockSize;\n"
    if text.count(anchor) != 1:
        raise RuntimeError("Unexpected A3 consumer mask: " + str(path))
    # candBuf_ already holds two rows of 2048 (value,index) pairs. Reuse
    # each row's 4096 floats as a table, without increasing UB or workspace.
    fast = r"""    // SGL A3 consumer: row-local block membership table.
    // Same strategy as SparseLightningIndexer in the pinned upstream source.
    // Preserve the original sorted-membership path for larger key domains.
    constexpr int32_t tableCapacity = BASE_TOPK * 2;
    int32_t blockCount = static_cast<int32_t>(
        (info.actS2Size + s2BaseSize_ - 1) / s2BaseSize_) * tileBlkNum;
    if (blockSize == 8 && tileBlkNum % 8 == 0 && blockCount <= tableCapacity) {
        LocalTensor<float> table = candBuf_.Get<float>()[innerS1Idx * tableCapacity];
        LocalTensor<int32_t> tableBits = table.template ReinterpretCast<int32_t>();
        if (info.isFirstS2InnerLoop) {
            // Drain previous vector readers before scalar table reuse. Keep
            // initialization and marking on S to avoid cross-pipe write races.
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            for (int32_t b = 0; b < blockCount; ++b) {
                tableBits.SetValue(b, 0x3F800000); // non-candidate: 1.0f
            }
            LocalTensor<int32_t> candidate = tmpBuf_.Get<float>()[12288]
                .template ReinterpretCast<int32_t>();
            AscendC::DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(candBlocks * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(candidate,
                candidateTopkIndexInGm[info.candidateOutOffset +
                    cuS1Idx * constInfo_.candidateTopkBlocks], copyParams, padParams);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (int32_t j = 0; j < candBlocks; ++j) {
                int32_t b = candidate.GetValue(j);
                if (b >= 0 && b < blockCount) {
                    tableBits.SetValue(b, 0); // duplicates are idempotent
                }
            }
            SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        }
        // Use a fixed staging address, as in the BF16 implementation: Brcb
        // directly from a row-dependent sub-tensor is unsafe on arch22.
        LocalTensor<float> stage = tmpBuf_.Get<float>()[4096];
        Adds(stage, table[tileBlockBase], 0.0f, tileBlkNum);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> positionMask = tmpBuf_.Get<float>()[12288];
        Brcb(positionMask, stage, static_cast<uint8_t>(tileBlkNum / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        candIsOut_ = positionMask;
        return;
    }
"""
    path.write_text(text.replace(anchor, anchor + fast, 1))


def build(item, source, directory):
    if item.get("a3_indexer_bindings"):
        fix_a3_indexer_workspace(source)
        optimize_a3_consumer_mask(source)
    for index, command in enumerate(item["commands"]):
        run(command, directory / ("build-{}.log".format(index)), cwd=source)
    # The pinned 9.1 build.sh has no --torch_extension option.
    if item["python_bindings"] and not any(
        "--torch_extension" in command for command in item["commands"]
    ):
        run(
            [sys.executable, "setup.py", "bdist_wheel"],
            directory / "build-python.log",
            cwd=source / "torch_extension",
        )
    installers = list((source / "build_out").glob("*.run"))
    wheels = list((source / "torch_extension/dist").glob("*.whl"))
    if len(installers) != 1 or (item["python_bindings"] and len(wheels) != 1):
        raise RuntimeError("Expected one upstream .run and, if enabled, Python wheel")
    installed = directory / "installed"
    installed.mkdir()
    run(
        ["bash", installers[0], "--quiet", "--install-path=" + str(installed)],
        directory / "install.log",
        cwd=directory,
    )
    vendors = [path for path in (installed / "vendors").glob("*") if path.is_dir()]
    if len(vendors) != 1 or vendors[0].is_symlink():
        raise RuntimeError("Expected exactly one installed vendor directory")
    vendor = vendors[0]
    inspect_vendor(vendor, item)
    return dict(
        config=item,
        vendor=vendor,
        wheel=wheels[0] if item["python_bindings"] else None,
        license=source / item["license"],
    )


def stage(result, package):
    item = result["config"]
    vendor = result["vendor"]
    for path in vendor.rglob("*"):
        if path.is_symlink() and vendor.resolve() not in path.resolve().parents:
            raise RuntimeError("External vendor symlink: " + str(path))
    target = package / "_cann/vendors" / vendor.name
    if target.exists():
        raise RuntimeError("Duplicate vendor: " + str(target))
    shutil.copytree(vendor, target)
    # Installer-generated scripts embed the temporary absolute installation path.
    if (target / "bin").exists():
        shutil.rmtree(target / "bin")
    opapi_sonames = isolate_staged_opapis(package)
    if item["python_bindings"]:
        stage_python(result["wheel"], item, package)
    licenses = package / "_third_party/licenses" / item["name"]
    licenses.mkdir(parents=True)
    shutil.copy2(result["license"], licenses / "LICENSE")
    hashes = {
        p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    return dict(
        item,
        vendor_directory=vendor.name,
        files=hashes,
        binding="upstream_jit" if item["python_bindings"] else "vendor_only",
        opapi_soname=opapi_sonames[vendor.name],
    )
