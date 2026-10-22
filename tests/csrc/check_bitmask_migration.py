"""Check against the frozen Kernel from d974d3de5b7b0d6586a41f227cba93a861f07fe1."""

import difflib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    fixtures = Path(__file__).resolve().parent / "fixtures/apply_token_bitmask"
    migrated = root / "csrc/sampling/apply_token_bitmask"
    original = (fixtures / "legacy_kernel.cpp").read_text().split('extern "C"')[0]
    expected = original
    # Only the CANN output address is added; every computational statement is retained.
    for before, after in (
        ("GM_ADDR bitmaskGmAddr, uint32_t numRows", "GM_ADDR bitmaskGmAddr, GM_ADDR outputGmAddr, uint32_t numRows"),
        ("        bitmaskGm.SetGlobalBuffer", "        outputGm.SetGlobalBuffer((__gm__ T *)outputGmAddr);\n        bitmaskGm.SetGlobalBuffer"),
        ("AscendC::DataCopy(logitsGm[outputGmOffset]", "AscendC::DataCopy(outputGm[outputGmOffset]"),
        ("    AscendC::GlobalTensor<T> logitsGm;", "    AscendC::GlobalTensor<T> logitsGm;\n    AscendC::GlobalTensor<T> outputGm;"),
    ):
        if expected.count(before) != 1:
            raise RuntimeError(f"Legacy Kernel changed; review the migration comparison: {before}")
        expected = expected.replace(before, after)
    expected = expected.rstrip("\n") + "\n"
    source = (migrated / "op_kernel/apply_token_bitmask.cpp").read_text()
    if source.count('extern "C"') != 1:
        raise RuntimeError("Expected one CANN Kernel entry")
    actual = source.split('extern "C"', 1)[0].rstrip("\n") + "\n"
    if actual != expected:
        difference = "".join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True)))
        raise RuntimeError(f"Kernel changed beyond the reviewed output binding:\n{difference}")

    print("PASS: Kernel body preserved except for the reviewed CANN output binding")


if __name__ == "__main__":
    main()
