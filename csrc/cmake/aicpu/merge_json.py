# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE-CANN in the root of the software repository for the full text of the License.
# Adapted from ops-transformer scripts/util/insert_op_info.py at
# 39b52ad5e252d98e13df76ed45b6c68b21b98859: merge selected registrations in one invocation.

"""Merge AICPU registrations for the selected CANN OpTypes and device library."""

import argparse
import json
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def merge_registrations(operators, library):
    # One JSON file may be shared by several selected declarations.
    sources = {}
    selected = set()
    for operator in operators:
        op_type, separator, source = operator.partition("=")
        if not separator or not op_type:
            raise ValueError(f"Expected OPTYPE=JSON, got: {operator}")
        if not source:
            raise ValueError(
                f"{op_type}: AICPU_JSON is required for registration packaging"
            )
        selected.add(op_type)
        sources.setdefault(Path(source), set()).add(op_type)

    merged = {}
    for source, expected in sources.items():
        try:
            entries = json.loads(
                source.read_text(encoding="utf-8"), object_pairs_hook=unique_object
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"{source}: {error}") from error
        if not isinstance(entries, dict):
            raise ValueError(f"{source}: expected a JSON object keyed by CANN OpType")
        missing = expected - entries.keys()
        if missing:
            raise ValueError(
                f"{source}: missing declared OpType: {', '.join(sorted(missing))}"
            )
        for op_type, entry in entries.items():
            if op_type in merged:
                raise ValueError(f"{source}: duplicate AICPU OpType: {op_type}")
            if op_type not in selected:
                raise ValueError(f"{source}: unselected AICPU OpType: {op_type}")
            info = entry.get("opInfo") if isinstance(entry, dict) else None
            if not isinstance(info, dict) or info.get("kernelSo") != library:
                raise ValueError(
                    f"{source}: {op_type}.opInfo.kernelSo must be {library}"
                )
            merged[op_type] = entry
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--operator", action="append", required=True, metavar="OPTYPE=JSON"
    )
    args = parser.parse_args()
    try:
        merged = merge_registrations(args.operator, args.library)
        # Validate every input before replacing the previous registration file.
        args.output.write_text(
            json.dumps(merged, indent=4, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"AICPU registration error: {error}\n")


if __name__ == "__main__":
    main()
