import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "python" / "sgl_kernel_npu"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from build_tools.target_provider import collect_provider_modules, stage_target_providers


@pytest.mark.parametrize(
    ("target", "expected_call"),
    [
        ("Ascend910", "torch_npu.npu_gemma_rms_norm(input, weight, eps)"),
        ("Ascend950", "torch_npu.npu_rms_norm(input, 1.0 + weight, eps)"),
    ],
)
def test_stage_target_providers_stages_the_real_providers(
    tmp_path, target, expected_call
):
    stage_target_providers(source_root=PACKAGE_ROOT, build_lib=tmp_path, target=target)

    staged = tmp_path / "sgl_kernel_npu" / "norm" / "gemma_rmsnorm.py"

    assert staged.exists()
    assert expected_call in staged.read_text()
    assert not (tmp_path / "target_providers").exists()


def test_unknown_target_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported provider target"):
        stage_target_providers(
            source_root=PACKAGE_ROOT, build_lib=tmp_path, target="Ascend999"
        )


def test_missing_provider_directory_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="Missing provider directory"):
        stage_target_providers(
            source_root=tmp_path, build_lib=tmp_path, target="Ascend910"
        )


def test_conflict_with_common_module_is_rejected(tmp_path):
    provider = (
        tmp_path / "source" / "target_providers" / "Ascend910" / "norm" / "foo.py"
    )
    provider.parent.mkdir(parents=True)
    provider.write_text('"""provider"""\n')

    original = '"""common"""\n'
    common = tmp_path / "build" / "sgl_kernel_npu" / "norm" / "foo.py"
    common.parent.mkdir(parents=True)
    common.write_text(original)

    with pytest.raises(RuntimeError, match="conflicts with an existing common module"):
        stage_target_providers(
            source_root=tmp_path / "source",
            build_lib=tmp_path / "build",
            target="Ascend910",
        )

    assert common.read_text() == original


def test_provider_trees_are_symmetric():
    modules_910 = collect_provider_modules(PACKAGE_ROOT, "Ascend910")
    modules_950 = collect_provider_modules(PACKAGE_ROOT, "Ascend950")

    assert modules_910 == modules_950
    assert modules_910, "provider trees must not be empty"
