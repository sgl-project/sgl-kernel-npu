"""Plan PR kernel tests without importing torch, Triton or operator code."""

import argparse
import ast
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

PACKAGE = "python/sgl_kernel_npu/sgl_kernel_npu/"
PUBLIC = "tests/python/operators/"
LEGACY = "tests/python/sgl_kernel_npu/"
WORKFLOW = ".github/workflows/pr-test-kernels.yml"
MANIFEST = "scripts/kernel_test_manifest.json"
GLOBAL_FILES = {
    "build.sh",
    ".gitmodules",
    "scripts/prepare_kernel_tests.sh",
    "scripts/run_kernel_tests.sh",
    "scripts/run_selected_kernel_tests.py",
    "scripts/select_kernel_tests.py",
    MANIFEST,
    "scripts/npu_ci_install_dependency.sh",
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def snapshot(root, revision=None):
    """Read both sides, including removed imports/files, without executing PR code."""
    if revision:
        result = {}
        with tarfile.open(
            fileobj=io.BytesIO(git(root, "archive", revision))
        ) as archive:
            for item in archive:
                if item.isfile() and (
                    item.name.endswith(".py") or item.name == WORKFLOW
                ):
                    result[item.name] = archive.extractfile(item).read().decode("utf-8")
        return result
    paths = list((root / "python/sgl_kernel_npu").rglob("*.py"))
    paths += list((root / "tests/python").rglob("*.py"))
    paths += [root / WORKFLOW]
    return {str(p.relative_to(root)): p.read_text() for p in paths if p.is_file()}


def changes_between(root, base, head):
    merge_base = git(root, "merge-base", base, head).decode().strip()
    fields = (
        git(root, "diff", "--name-status", "-z", "--find-renames", merge_base, head)
        .decode()
        .split("\0")
    )
    changes = []
    i = 0
    while i < len(fields) and fields[i]:
        status, path = fields[i : i + 2]
        i += 2
        change = {"status": status, "path": path}
        if status.startswith(("R", "C")):
            change["old_path"] = path
            change["path"] = fields[i]
            i += 1
        changes.append(change)
    return merge_base, changes


def module_name(path):
    if path.startswith(PACKAGE):
        path = "sgl_kernel_npu/" + path[len(PACKAGE) :]
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


class Dependencies:
    """Static import closure, including function-local and relative imports.

    Package initializers are dependencies but their sibling re-exports are not
    expanded: executing one public operator must not select every sibling. The
    manifest supplies explicit entry points for dynamically dispatched backends.
    Unknown/unmapped files use the conservative ownership fallback in select().
    """

    def __init__(self, files, entrypoints=()):
        self.files = files
        self.entrypoints = set(entrypoints)
        self.modules = {module_name(p): p for p in files if p.endswith(".py")}
        self.cache = {}
        self.errors = []

    def direct(self, path):
        if path in self.cache:
            return self.cache[path]
        dependencies = set()
        self.cache[path] = dependencies
        if path not in self.files or not path.endswith(".py"):
            return dependencies
        name = module_name(path)
        package = name if path.endswith("/__init__.py") else name.rpartition(".")[0]
        # An initializer change affects descendants. Shared imports inside it
        # are followed, except mapped sibling operator/backend re-exports.
        for i in range(1, len(name.split("."))):
            ancestor = self.modules.get(".".join(name.split(".")[:i]))
            if ancestor and ancestor.endswith("/__init__.py"):
                dependencies.add(ancestor)
        try:
            tree = ast.parse(self.files[path], filename=path)
        except SyntaxError as error:
            self.errors.append(f"Cannot parse {path}: {error}")
            return dependencies
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    parents = package.split(".")
                    prefix = ".".join(
                        parents[: len(parents) - node.level + 1]
                        + ([prefix] if prefix else [])
                    )
                names = [prefix] + [prefix + "." + alias.name for alias in node.names]
                if (
                    any(alias.name == "*" for alias in node.names)
                    and prefix in self.modules
                ):
                    self.errors.append(
                        f"Unresolved wildcard import in {path}: {prefix}"
                    )
            elif isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                or isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", "import_module"}
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and not node.args[0].value.startswith(".")
                ):
                    names = [node.args[0].value]
                else:
                    self.errors.append(f"Unresolved dynamic import in {path}")
            for imported in names:
                target = self.modules.get(imported)
                if target and not (
                    path.endswith("/__init__.py") and target in self.entrypoints
                ):
                    dependencies.add(target)
                # Legacy script tests also import sibling helpers as `utils`.
                sibling = str(Path(path).parent / (imported.replace(".", "/") + ".py"))
                if sibling in self.files:
                    dependencies.add(sibling)
        return dependencies

    def closure(self, seeds):
        seen = set()
        pending = list(seeds)
        while pending:
            path = pending.pop()
            if path not in seen:
                seen.add(path)
                pending.extend(self.direct(path) - seen)
        return seen


def owners(path, manifest):
    for prefix in (PACKAGE + "operators/", PUBLIC):
        if path.startswith(prefix):
            name = path[len(prefix) :].split("/")[0]
            if name in manifest["modules"]:
                return {name}
    if path.startswith(PACKAGE + "implementation/"):
        parts = path[len(PACKAGE + "implementation/") :].split("/")
        if len(parts) > 1 and parts[1] in manifest["modules"]:
            return {parts[1]}
    return {
        module
        for module, config in manifest["modules"].items()
        if any(path.startswith(prefix) for prefix in config["legacy_source_prefixes"])
    }


def workflow_changes(before, after, modules):
    import yaml

    # BaseLoader preserves GitHub's YAML 1.2 `on` key (SafeLoader treats it as True).
    try:
        old, new = (yaml.load(s, Loader=yaml.BaseLoader) for s in (before, after))
    except yaml.YAMLError as error:
        raise ValueError(str(error)) from error
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise ValueError("Workflow must be a mapping")
    old_jobs, new_jobs = old.pop("jobs", {}), new.pop("jobs", {})
    reasons = []
    affected = set()
    if old != new:
        affected.update(modules)
        reasons.append("workflow global configuration/trigger changed")
    for job in old_jobs.keys() | new_jobs.keys():
        if old_jobs.get(job) == new_jobs.get(job):
            continue
        module = next((m for m in modules if job == f"test-{m}-ops"), None)
        if module:
            affected.add(module)
            reasons.append(f"workflow local job changed: {job}")
        else:
            affected.update(modules)
            reasons.append(f"workflow shared/unknown job changed: {job}")
    return affected, reasons


def select(root, changes, before=None, after=None, full=False):
    manifest = json.loads((root / MANIFEST).read_text())
    after = snapshot(root) if after is None else after
    before = {} if before is None else before
    modules = manifest["modules"]
    inventory = {m: set() for m in modules}
    seeds = {}
    notes = []
    for module, config in modules.items():
        for filename in config["legacy_tests"]:
            path = LEGACY + filename
            inventory[module].add(path)
            seeds[path] = {path}
    for op, config in manifest["operators"].items():
        for test in config["tests"]:
            inventory[config["module"]].add(test)
            seeds.setdefault(test, {test}).update(
                PACKAGE + p for p in config["sources"]
            )
    # New public tests in known modules cannot be silently omitted when the
    # manifest has not yet been updated; their source changes fall back to module.
    for path in before.keys() | after.keys():
        if path.startswith(PUBLIC) and Path(path).name.startswith("test_"):
            for module in owners(path, manifest):
                inventory[module].add(path)
                seeds.setdefault(path, {path})
    # A renamed test retains its old CI owner even when its new filename has
    # not reached the manifest yet. Source renames still evaluate both paths.
    for change in changes:
        old, new = change.get("old_path"), change["path"]
        if (
            change["status"].startswith("R")
            and old in seeds
            and new.startswith((LEGACY, PUBLIC))
        ):
            for module in modules:
                if old in inventory[module]:
                    inventory[module].add(new)
                    seeds.setdefault(new, {new}).update(seeds[old] - {old})
    entrypoints = {
        PACKAGE + source
        for config in manifest["operators"].values()
        for source in config["sources"]
    }
    graphs = [Dependencies(before, entrypoints), Dependencies(after, entrypoints)]
    dependencies = {}
    for test, entrypoints in seeds.items():
        dependencies[test] = set().union(*(g.closure(entrypoints) for g in graphs))
        # Match runner semantics: only pytest loads ancestor conftest. Walk
        # both sides so a removed fixture/import still selects its former users.
        pytest_style = test.startswith(PUBLIC) or any(
            "def test_" in files.get(test, "") and "__main__" not in files.get(test, "")
            for files in (before, after)
        )
        for parent in Path(test).parents if pytest_style else ():
            conf = str(parent / "conftest.py")
            if conf in before or conf in after:
                dependencies[test].update(
                    set().union(*(g.closure([conf]) for g in graphs))
                )
    selected = {m: set() for m in modules}
    reasons = {m: [] for m in modules}

    def expand(targets, reason):
        for module in sorted(targets):
            selected[module].update(inventory[module])
            reasons[module].append(reason)

    if full:
        expand(modules, "manual dispatch: full existing kernel suite")
    for change in changes:
        for path in dict.fromkeys([change.get("old_path"), change["path"]]):
            if not path:
                continue
            if path == WORKFLOW:
                try:
                    affected, why = workflow_changes(
                        before.get(path, ""), after.get(path, ""), modules
                    )
                    for reason in why:
                        # Each job reason belongs only to that job; global ones to all.
                        local = next(
                            (m for m in modules if reason.endswith(f": test-{m}-ops")),
                            None,
                        )
                        expand({local} if local else affected, reason)
                    if not why:
                        notes.append(
                            f"{path}: formatting/comments only; no execution change"
                        )
                except (ValueError, TypeError, KeyError) as error:
                    expand(
                        modules,
                        f"{path}: cannot compare workflow ({error}); conservative full suite",
                    )
                continue
            if (
                path in GLOBAL_FILES
                or Path(path).name == "CMakeLists.txt"
                or path.startswith(
                    (
                        "cmake/",
                        "include/",
                        "csrc/utils/",
                        "third_party/",
                        "python/sgl_kernel_npu/setup",
                        "python/sgl_kernel_npu/build",
                    )
                )
            ):
                expand(
                    modules,
                    f"{path}: shared build/install/selection dependency; all jobs use the same wheel",
                )
                continue
            if (
                path.endswith((".md", ".rst"))
                or path.startswith(("docs/", ".github/ISSUE_TEMPLATE/"))
                or path in {"LICENSE", ".gitignore", ".pre-commit-config.yaml"}
                or path.startswith("tests/ci/")
            ):
                notes.append(
                    f"{path}: documentation or unrelated CI metadata; no NPU execution change"
                )
                continue
            if (
                path.startswith(LEGACY)
                and Path(path).name in manifest["disabled_legacy_tests"]
            ):
                notes.append(
                    f"{path}: {manifest['disabled_legacy_tests'][Path(path).name]}"
                )
                continue
            matched = False
            for module in modules:
                hits = {
                    test
                    for test in inventory[module]
                    if path in dependencies.get(test, {test})
                }
                if hits:
                    selected[module].update(hits)
                    reasons[module].append(
                        f"{path}: explicit mapping/import dependency -> {', '.join(sorted(hits))}"
                    )
                    matched = True
            # Initializers and unrecognized module helpers can have side effects
            # invisible to static imports. Broaden instead of assuming no users.
            if matched and not path.endswith("/__init__.py"):
                continue
            owned = owners(path, manifest)
            if owned:
                expand(
                    owned,
                    f"{path}: module initializer or unmapped source/helper; conservative module scope",
                )
                continue
            if path.startswith((LEGACY, PUBLIC)) and Path(path).name.startswith(
                "test_"
            ):
                # Unassigned tests have no module metadata. Run the changed test
                # and all known tests; maintainers should assign it in the manifest.
                for module in modules:
                    inventory[module].add(path)
                expand(
                    modules,
                    f"{path}: unmapped test; conservative full scope including changed test",
                )
            elif path.startswith(
                (PACKAGE + "operators/utils/", "tests/python/utils/")
            ) or path in {
                PACKAGE + "operators/__init__.py",
                PACKAGE + "implementation/__init__.py",
            }:
                public_modules = {
                    m
                    for m in modules
                    if any(t.startswith(PUBLIC) for t in inventory[m])
                }
                expand(
                    public_modules,
                    f"{path}: shared public framework with unknown consumers; all public modules",
                )
            else:
                expand(modules, f"{path}: unknown impact; conservative full suite")
    for graph in graphs:
        for error in graph.errors if changes or full else ():
            expand(
                modules,
                error + "; dependency graph incomplete, conservative full suite",
            )
    result = {"changes": changes, "notes": notes, "modules": {}}
    import yaml

    workflow = yaml.load(after[WORKFLOW], Loader=yaml.BaseLoader)
    for module in modules:
        missing = sorted(t for t in selected[module] if t not in after)
        if missing:
            reasons[module].append(
                "Removed/missing tests omitted from execution: " + ", ".join(missing)
            )
        tests = sorted(t for t in selected[module] if t in after)
        matrix = workflow["jobs"][f"test-{module}-ops"]["strategy"]["matrix"]
        result["modules"][module] = {
            "selected": bool(tests),
            "tests": tests,
            "matrix": matrix,
            "reasons": reasons[module],
            "skip_reason": (
                None
                if tests
                else (
                    "no surviving affected tests"
                    if reasons[module]
                    else "no changed file or dependency affects this module"
                )
            ),
        }
    result["cann_exclusions"] = manifest["cann_exclusions"]
    result["existing_disabled_tests"] = manifest["disabled_legacy_tests"]
    return result


def summary(plan):
    lines = ["## Kernel test selection", "", "### PR changes", ""]
    lines.extend(
        f"- `{c['status']}` `{c.get('old_path', c['path'])}`"
        + (f" → `{c['path']}`" if "old_path" in c else "")
        for c in plan["changes"]
    )
    for module, config in plan["modules"].items():
        lines += [
            "",
            f"### {module}: {'RUN' if config['selected'] else 'SKIP'}",
            "",
            f"Matrix (unchanged coverage): `{json.dumps(config['matrix'])}`",
        ]
        lines += [f"- `{t}`" for t in config["tests"]]
        lines += [f"- Reason: {reason}" for reason in config["reasons"]]
        if config["skip_reason"]:
            lines.append(f"- Skipped: {config['skip_reason']}")
    lines += ["", "### Notes and existing exclusions", ""]
    lines += [f"- {note}" for note in plan["notes"]]
    lines += [
        f"- {name}: {reason}"
        for name, reason in plan["existing_disabled_tests"].items()
    ]
    for cann, exclusions in plan["cann_exclusions"].items():
        lines += [
            f"- CANN {cann}, {name}: {reason}" for name, reason in exclusions.items()
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("kernel-test-selection.json")
    )
    args = parser.parse_args()
    if not args.full and not args.base:
        parser.error("--base is required except for --full")
    merge_base, changes = (
        changes_between(args.root, args.base, args.head) if args.base else (None, [])
    )
    before = snapshot(args.root, merge_base) if merge_base else {}
    plan = select(args.root, changes, before=before, full=args.full)
    plan["comparison"] = {
        "base": args.base,
        "merge_base": merge_base,
        "head": args.head,
    }
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    report = summary(plan)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(report)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for module, config in plan["modules"].items():
                output.write(f"{module}_changed={str(config['selected']).lower()}\n")


if __name__ == "__main__":
    main()
