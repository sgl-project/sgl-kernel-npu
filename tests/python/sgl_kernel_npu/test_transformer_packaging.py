"""Packaging regressions that do not require an initialized NPU SDK."""

import zipfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "python/sgl_kernel_npu"


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    from build_tools.third_party import get_adapter

    return get_adapter("ops_transformer")


@pytest.mark.parametrize("modern", [False, True], ids=["cann91", "cann92"])
def test_reject_partially_packaged_bindings(builder, tmp_path, modern):
    wheel = tmp_path / "bindings.whl"
    prefix = "cann_ops_transformer_vendor_test" if modern else "cann_ops_transformer"
    module = "ops/mhc/mhc_post/__init__.py" if modern else "ops/mhc_post.py"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(prefix + "/__init__.py", "")
        archive.writestr(prefix + "/" + module, "def mhc_post(): pass\n")
    item = dict(
        name="test",
        python_package="test",
        vendor_name="vendor_test",
        ops=["mhc_post", "mhc_pre_sinkhorn"],
    )
    with pytest.raises(RuntimeError, match="Python binding for mhc_pre_sinkhorn"):
        builder.stage_python(wheel, item, tmp_path / "package")


def test_metadata_binding_comes_from_compute_module(builder, tmp_path):
    wheel = tmp_path / "bindings.whl"
    prefix = "cann_ops_transformer_vendor_test"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            prefix + "/ops/attention/sparse_flash_mla/__init__.py",
            "from .sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata\n",
        )
    item = dict(
        name="test",
        python_package="test",
        vendor_name="vendor_test",
        ops=["sparse_flash_mla", "sparse_flash_mla_metadata"],
    )
    builder.stage_python(wheel, item, tmp_path / "package")
    init = tmp_path / "package/_third_party/test/ops/__init__.py"
    assert (
        init.read_text() == "from .attention.sparse_flash_mla import *  # noqa: F403\n"
    )


@pytest.mark.parametrize("experimental", [False, True])
def test_source_directory_checked_before_build(builder, tmp_path, experimental):
    command = ["bash", "build.sh", "--pkg"]
    if experimental:
        command.append("--experimental")
    item = dict(ops=["test_op"], commands=[command], source_patches={})
    root = tmp_path / "experimental" if experimental else tmp_path
    target = root / "attention/test_op/CMakeLists.txt"
    with pytest.raises(RuntimeError, match="upstream operator directory for test_op"):
        builder.prepare_source(item, tmp_path, tmp_path)
    target.parent.mkdir(parents=True)
    target.touch()
    builder.prepare_source(item, tmp_path, tmp_path)
