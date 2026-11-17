"""Print migration timing deltas from existing results without running an NPU."""

import argparse
import json
import re
from pathlib import Path

from report import build_report

COMPONENTS = ("indexer_or_lim", "attention_or_copysfa", "chain")


def standalone_accepted(result, row):
    # Frozen-plan attention does not consume the indexer's live TopK IDs.
    if not row.get("fixture_equal") or not result.get("accuracy", {}).get(
        "standalone", {}
    ).get("passed"):
        return False
    if result.get("arm", {}).get("name") == "native":
        return True
    return row.get("parity_to_native", {}).get("standalone", {}).get("passed", False)


def delta(before, after):
    return "n/a" if before <= 0 else f"{(after / before - 1) * 100:+.2f}%"


def show_matrix(root, statistic):
    print(f"\n=== {root.name} | NPU Event {statistic} | us ===")
    results = {}
    for path in sorted(root.glob("q*-*/*/result.json")):
        result = json.loads(path.read_text())
        case = result["case"]
        key = (case["query_width"], case["mode"], result["arm"]["name"])
        if key in results:
            raise ValueError(f"duplicate result: {key}")
        results[key] = result
    validated = {}
    attention_validated = {}
    try:
        report = build_report(
            root, parity_atol=0.002, parity_rtol=0.005, max_relative_l2=0.08
        )
        validated = {
            (r["query_width"], r["mode"], r["arm"]): r["passed"] for r in report["rows"]
        }
        attention_validated = {
            (r["query_width"], r["mode"], r["arm"]): standalone_accepted(
                results[(r["query_width"], r["mode"], r["arm"])], r
            )
            for r in report["rows"]
        }
    except (ValueError, KeyError, OSError) as error:
        print(f"Full validation unavailable: {error}")
    for width in (1, 4):
        for mode in ("eager", "graph"):
            if not any(
                (width, mode, arm) in results
                for arm in ("native", "pre_migration", "post_migration")
            ):
                continue
            print(f"\nQ{width} {mode}")
            values = {}
            for arm in ("native", "pre_migration", "post_migration"):
                key = (width, mode, arm)
                result = results.get(key)
                if result is None:
                    print(f"  {arm:15} MISSING")
                    continue
                timings = result.get("timings", {})
                if not all(c in timings for c in COMPONENTS):
                    print(
                        f"  {arm:15} FAILED: {result.get('error', 'no timings').strip().splitlines()[-1]}"
                    )
                    continue
                values[arm] = [timings[c][statistic] for c in COMPONENTS]
                status = "PASS" if validated.get(key) else "UNVALIDATED"
                print(
                    f"  {arm:15} index/LIM={values[arm][0]:9.3f}  attention/CopySFA={values[arm][1]:9.3f}  chain={values[arm][2]:9.3f}  {status}"
                )
                print(
                    f"    frozen-plan attention: {'PASS' if attention_validated.get(key) else 'NOT ACCEPTED'}"
                )
                for name, check in result.get("accuracy", {}).items():
                    if not check.get("passed"):
                        print(f"    accuracy failed: {name}")
                live = result.get("diagnostics", {}).get("live_chain_lim")
                if live and not live.get("matches_fixture_union_misses"):
                    print(
                        f"    live misses differ from fixture: {live.get('miss_counts')}"
                    )
                if result.get("status") != "passed":
                    print(f"    case status: {result.get('status')}")
                if validated.get(key) is False and result.get("status") == "passed":
                    print("    cross-arm fixture/output parity failed")
            if "pre_migration" in values and "post_migration" in values:
                hashes = [
                    results.get((width, mode, arm), {})
                    .get("accuracy", {})
                    .get("canonical_ids", {})
                    .get("sha256")
                    for arm in ("native", "pre_migration", "post_migration")
                ]
                if all(hashes):
                    print(
                        f"  live TopK set hashes native/pre/post: {'MATCH' if len(set(hashes)) == 1 else 'DIFFER'} (fixture-ID check remains separate)"
                    )
                changes = [
                    delta(a, b)
                    for a, b in zip(values["pre_migration"], values["post_migration"])
                ]
                passed = all(
                    validated.get((width, mode, a))
                    for a in ("native", "pre_migration", "post_migration")
                )
                print(
                    f"  post vs pre: LIM {changes[0]}, CopySFA {changes[1]}, chain {changes[2]}  [{'VALIDATED' if passed else 'NOT ACCEPTED: check failures above'}]"
                )
                if all(
                    attention_validated.get((width, mode, a))
                    for a in ("native", "pre_migration", "post_migration")
                ):
                    print(
                        f"  CopySFA standalone: {changes[1]} VALIDATED (fixed plan, golden and native parity passed; independent of canonical_ids)"
                    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stat", choices=("mean_us", "median_us"), default="mean_us")
    args = parser.parse_args()
    roots = (
        [args.root]
        if list(args.root.glob("q*-*/*/result.json"))
        else sorted(
            {p.parent.parent.parent for p in args.root.glob("*/q*-*/*/result.json")}
        )
    )
    if not roots:
        raise SystemExit(f"No result.json files under {args.root}")
    print(
        "Positive delta = slower; negative delta = faster. UNVALIDATED timings are diagnostic only."
    )
    matrices = {root.name: show_matrix(root, args.stat) for root in roots}
    misses_available = sorted(
        {
            int(match.group(1))
            for name in matrices
            if (match := re.fullmatch(r"(?:unfixed|fixed)-m(\d+)", name))
        }
    )
    for misses in misses_available:
        before, after = (
            matrices.get(f"unfixed-m{misses}"),
            matrices.get(f"fixed-m{misses}"),
        )
        if before is None or after is None:
            continue
        print(f"\n=== Scheduling fix, misses={misses}: fixed post vs unfixed post ===")
        for width in (1, 4):
            for mode in ("eager", "graph"):
                key = (width, mode, "post_migration")
                left, right = before.get(key, {}), after.get(key, {})
                if not all(
                    c in r.get("timings", {}) for r in (left, right) for c in COMPONENTS
                ):
                    continue
                if left.get("fixture") != right.get("fixture") or left.get(
                    "case"
                ) != right.get("case"):
                    print(
                        f"Q{width} {mode}: fixtures or cases differ; comparison rejected"
                    )
                    continue
                changes = [
                    delta(left["timings"][c][args.stat], right["timings"][c][args.stat])
                    for c in COMPONENTS
                ]
                print(
                    f"Q{width} {mode}: LIM {changes[0]}, CopySFA {changes[1]}, chain {changes[2]} (accept only if both matrices validated above)"
                )


if __name__ == "__main__":
    main()
