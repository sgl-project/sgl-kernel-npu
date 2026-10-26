"""CPU-only regression tests; deliberately outside tests/python/conftest.py."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_selected_kernel_tests as runner
import select_kernel_tests as selector


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = selector.snapshot(ROOT)
        cls.manifest = json.loads((ROOT / selector.MANIFEST).read_text())

    def plan(self, *paths, before=None, after=None):
        changes = [
            p if isinstance(p, dict) else {"status": "M", "path": p} for p in paths
        ]
        return selector.select(
            ROOT,
            changes,
            before=self.files if before is None else before,
            after=self.files if after is None else after,
        )

    def selected(self, plan):
        return {m for m, config in plan["modules"].items() if config["selected"]}

    def selected_tests(self, plan):
        return {t for config in plan["modules"].values() for t in config["tests"]}

    def test_independent_public_operator_and_both_backends(self):
        for path in self.manifest["operators"]["norm.l1_norm"]["sources"]:
            with self.subTest(path=path):
                plan = self.plan(selector.PACKAGE + path)
                self.assertEqual(self.selected(plan), {"norm"})
                self.assertEqual(
                    self.selected_tests(plan),
                    {selector.PUBLIC + "norm/test_l1_norm.py"},
                )

    def test_changed_public_or_legacy_test_is_individual(self):
        for test in [
            selector.PUBLIC + "norm/test_l1_norm.py",
            selector.LEGACY + "test_l1_norm.py",
        ]:
            self.assertEqual(self.selected_tests(self.plan(test)), {test})

    def test_legacy_source_is_independent(self):
        self.assertEqual(
            self.selected_tests(self.plan(selector.PACKAGE + "norm/l1_norm.py")),
            {selector.LEGACY + "test_l1_norm.py"},
        )

    def test_shared_norm_helper_follows_transitive_imports(self):
        plan = self.plan(selector.PACKAGE + "operators/norm/_qkv_inputs.py")
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(
            self.selected_tests(plan),
            {
                selector.PUBLIC + "norm/test_" + name + ".py"
                for name in [
                    "split_qkv_rmsnorm_rope",
                    "split_qkvgate_gemma_rmsnorm_rope",
                    "split_qkv_rmsnorm_rope_pos_cache_half_npu",
                    "split_qkv_tp_rmsnorm_rope",
                ]
            },
        )

    def test_reference_helper_selects_three_users(self):
        plan = self.plan(
            selector.PACKAGE + "implementation/torch_native/norm/_qkv_compute.py"
        )
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(len(self.selected_tests(plan)), 3)

    def test_unmapped_norm_helper_is_conservative(self):
        plan = self.plan(selector.PACKAGE + "operators/norm/new_helper.py")
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertIn(
            "conservative module", " ".join(plan["modules"]["norm"]["reasons"])
        )
        self.assertGreater(len(self.selected_tests(plan)), 9)

    def test_multiple_modules(self):
        plan = self.plan(
            selector.PACKAGE + "implementation/triton/norm/l1_norm.py",
            selector.LEGACY + "test_cache_assign.py",
        )
        self.assertEqual(self.selected(plan), {"norm", "cache"})
        self.assertEqual(len(self.selected_tests(plan)), 2)

    def test_cross_domain_legacy_norm_operator(self):
        plan = self.plan(selector.PACKAGE + "norm/split_qkv_rmsnorm_rope.py")
        self.assertEqual(self.selected(plan), {"attention"})
        self.assertIn(
            selector.LEGACY + "test_split_qkv_rmsnorm_rope.py",
            self.selected_tests(plan),
        )

    def test_registry_and_test_helpers_include_consumers(self):
        for path in [
            selector.PACKAGE + "operators/utils/registry.py",
            "tests/python/utils/accuracy.py",
            "tests/python/conftest.py",
        ]:
            with self.subTest(path=path):
                tests = self.selected_tests(self.plan(path))
                self.assertTrue(
                    {
                        t
                        for t in self.files
                        if t.startswith(selector.PUBLIC)
                        and Path(t).name.startswith("test_")
                    }
                    <= tests
                )
                self.assertNotIn(selector.LEGACY + "test_l1_norm.py", tests)

    def test_shared_triton_helper_reaches_legacy_and_public(self):
        plan = self.plan(selector.PACKAGE + "utils/triton_utils.py")
        self.assertIn(selector.LEGACY + "test_l1_norm.py", self.selected_tests(plan))
        self.assertIn(
            selector.PUBLIC + "norm/test_l1_norm.py", self.selected_tests(plan)
        )
        self.assertTrue({"norm", "fla", "attention"} <= self.selected(plan))

    def test_global_build_dependencies(self):
        for path in [
            "build.sh",
            "CMakeLists.txt",
            "include/sgl_kenel_npu_ops.h",
            "csrc/utils/common.h",
            "third_party/pto-isa",
            "scripts/prepare_kernel_tests.sh",
            "scripts/npu_ci_install_dependency.sh",
            "python/sgl_kernel_npu/setup.py",
        ]:
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assertEqual(self.selected(plan), set(self.manifest["modules"]))
                self.assertIn("shared build", plan["modules"]["norm"]["reasons"][0])

    def test_workflow_local_job_only(self):
        after = dict(self.files)
        after[selector.WORKFLOW] = after[selector.WORKFLOW].replace(
            "timeout-minutes: 35", "timeout-minutes: 36"
        )
        plan = self.plan(selector.WORKFLOW, after=after)
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(len(plan["modules"]["norm"]["tests"]), 13)

    def test_workflow_multiple_local_jobs(self):
        after = dict(self.files)
        after[selector.WORKFLOW] = (
            after[selector.WORKFLOW]
            .replace(
                "--selection kernel-test-selection.json norm",
                "--selection plan.json norm",
            )
            .replace(
                "--selection kernel-test-selection.json cache",
                "--selection plan.json cache",
            )
        )
        self.assertEqual(
            self.selected(self.plan(selector.WORKFLOW, after=after)), {"norm", "cache"}
        )

    def test_workflow_global_trigger_and_selector(self):
        for old, new in [
            ("cancel-in-progress: true", "cancel-in-progress: false"),
            ("branches: [ main ]", "branches: [ main, release ]"),
            ('--base "$BASE_SHA"', "--base HEAD^"),
        ]:
            after = dict(self.files)
            after[selector.WORKFLOW] = after[selector.WORKFLOW].replace(old, new)
            self.assertEqual(
                self.selected(self.plan(selector.WORKFLOW, after=after)),
                set(self.manifest["modules"]),
            )

    def test_workflow_comments_and_documentation_do_not_run_npu(self):
        after = dict(self.files)
        after[selector.WORKFLOW] += "\n# Comment only\n"
        plan = self.plan(
            selector.WORKFLOW, "README.md", "docs/operators.md", after=after
        )
        self.assertEqual(self.selected(plan), set())
        self.assertTrue(plan["notes"])

    def test_unmapped_tests_are_included_even_with_other_changes(self):
        after = dict(self.files)
        new_tests = [
            selector.LEGACY + "test_new.py",
            selector.PUBLIC + "new_module/test_new.py",
        ]
        for test in new_tests:
            after[test] = "def test_new(): pass\n"
        plan = self.plan(
            *new_tests, selector.PACKAGE + "operators/norm/l1_norm.py", after=after
        )
        self.assertEqual(self.selected(plan), set(self.manifest["modules"]))
        self.assertTrue(set(new_tests) <= self.selected_tests(plan))

    def test_unknown_path_runs_all_and_explains(self):
        plan = self.plan("new_build_dependency.lock")
        self.assertEqual(self.selected(plan), set(self.manifest["modules"]))
        self.assertIn("unknown impact", plan["modules"]["norm"]["reasons"][0])

    def test_deleted_test_never_reaches_runner(self):
        path = selector.PUBLIC + "norm/test_l1_norm.py"
        after = dict(self.files)
        del after[path]
        plan = self.plan({"status": "D", "path": path}, after=after)
        self.assertNotIn(path, self.selected_tests(plan))
        self.assertTrue(
            any("Removed/missing" in r for r in plan["modules"]["norm"]["reasons"])
        )

    def test_rename_uses_both_paths_and_only_existing_test(self):
        old = selector.PUBLIC + "norm/test_l1_norm.py"
        new = selector.PUBLIC + "norm/test_l1_normalization.py"
        after = dict(self.files)
        after[new] = after.pop(old)
        plan = self.plan({"status": "R100", "old_path": old, "path": new}, after=after)
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(self.selected_tests(plan), {new})

    def test_legacy_test_rename_retains_module_without_stale_path(self):
        old = selector.LEGACY + "test_l1_norm.py"
        new = selector.LEGACY + "test_l1_renamed.py"
        after = dict(self.files)
        after[new] = after.pop(old)
        plan = self.plan({"status": "R100", "old_path": old, "path": new}, after=after)
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(self.selected_tests(plan), {new})

    def test_deleted_helper_keeps_base_dependency_edges(self):
        helper = selector.PACKAGE + "operators/norm/_qkv_inputs.py"
        after = dict(self.files)
        del after[helper]
        for path in list(after):
            if path.startswith(selector.PACKAGE + "operators/norm/"):
                after[path] = after[path].replace(
                    "from ._qkv_inputs import", "from .new_inputs import"
                )
        plan = self.plan({"status": "D", "path": helper}, after=after)
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(len(self.selected_tests(plan)), 4)

    def test_new_backend_or_module_file_falls_back(self):
        self.assertEqual(
            self.selected(
                self.plan(selector.PACKAGE + "implementation/ascendc/norm/new_op.py")
            ),
            {"norm"},
        )
        self.assertEqual(
            self.selected(self.plan(selector.PACKAGE + "operators/future/new_op.py")),
            set(self.manifest["modules"]),
        )

    def test_bad_python_graph_expands_instead_of_silently_missing(self):
        after = dict(self.files)
        after[selector.PACKAGE + "operators/norm/l1_norm.py"] = "bad syntax!"
        plan = self.plan(selector.PACKAGE + "operators/norm/l1_norm.py", after=after)
        self.assertEqual(self.selected(plan), set(self.manifest["modules"]))
        self.assertIn("Cannot parse", " ".join(plan["modules"]["norm"]["reasons"]))

    def test_initializer_shared_import_is_tracked(self):
        after = dict(self.files)
        helper = selector.PACKAGE + "operators/norm/shared_setup.py"
        after[helper] = "READY = True\n"
        after[
            selector.PACKAGE + "operators/norm/__init__.py"
        ] += "\nfrom .shared_setup import READY\n"
        plan = self.plan(helper, before=after, after=after)
        self.assertEqual(self.selected(plan), {"norm"})
        self.assertEqual(len(self.selected_tests(plan)), 9)

    def test_unresolved_dynamic_import_expands_with_reason(self):
        after = dict(self.files)
        after[
            selector.PACKAGE + "operators/norm/l1_norm.py"
        ] += "\nimport importlib\nbackend = importlib.import_module(backend_name)\n"
        plan = self.plan(selector.PACKAGE + "operators/norm/l1_norm.py", after=after)
        self.assertEqual(self.selected(plan), set(self.manifest["modules"]))
        self.assertIn(
            "Unresolved dynamic import", " ".join(plan["modules"]["norm"]["reasons"])
        )

    def test_manifest_covers_public_sources_and_tests(self):
        mapped = {
            selector.PACKAGE + source
            for config in self.manifest["operators"].values()
            for source in config["sources"]
        }
        graph = selector.Dependencies(self.files)
        covered = graph.closure(mapped)
        for path in self.files:
            if path.startswith(
                (
                    selector.PACKAGE + "operators/norm/",
                    selector.PACKAGE + "implementation/triton/norm/",
                    selector.PACKAGE + "implementation/torch_native/norm/",
                )
            ) and not path.endswith("__init__.py"):
                self.assertIn(path, covered)
        expected = {
            test
            for config in self.manifest["operators"].values()
            for test in config["tests"]
        }
        self.assertEqual(
            expected,
            {
                p
                for p in self.files
                if p.startswith(selector.PUBLIC) and Path(p).name.startswith("test_")
            },
        )

    def test_matrices_and_existing_legacy_groups_are_preserved(self):
        import re

        import yaml

        shell = (ROOT / "scripts/run_kernel_tests.sh").read_text()
        workflow = yaml.load(self.files[selector.WORKFLOW], Loader=yaml.BaseLoader)
        for module, config in self.manifest["modules"].items():
            block = re.search(rf"{module.upper()}_TESTS=\((.*?)\n\)", shell, re.S)[1]
            expected = set(re.findall(r"^    (test_\w+\.py)", block, re.M))
            if module == "fused":
                expected.add("test_hello_world.py")
            self.assertEqual(set(config["legacy_tests"]), expected)
            self.assertEqual(
                workflow["jobs"][f"test-{module}-ops"]["strategy"]["matrix"],
                {"cann_version": ["9.0.0", "9.1.0"], "hardware": ["a3"]},
            )
        self.assertIn("get-changed-files", workflow["jobs"]["finish"]["needs"])


class GitChangesTests(unittest.TestCase):
    def test_whole_fork_pr_merge_base_add_delete_rename_and_multiple_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return selector.git(root, *args).decode().strip()

            git("init", "-q")
            git("config", "user.email", "ci@example.invalid")
            git("config", "user.name", "CI tests")
            for name in ["old name.py", "deleted.py", "retained.py"]:
                (root / name).write_text(name + "\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            git("checkout", "-qb", "fork-feature")
            (root / "first.py").write_text("first commit\n")
            git("add", ".")
            git("commit", "-qm", "first PR commit")
            git("mv", "old name.py", "new name.py")
            (root / "deleted.py").unlink()
            (root / "retained.py").write_text("changed\n")
            git("add", "-A")
            git("commit", "-qm", "second PR commit")
            head = git("rev-parse", "HEAD")
            git("checkout", "-qb", "upstream", base)
            (root / "upstream-only.py").write_text("not part of the PR\n")
            git("add", ".")
            git("commit", "-qm", "upstream advanced")
            merge_base, changes = selector.changes_between(root, "HEAD", head)
            self.assertEqual(merge_base, base)
            self.assertEqual(
                {c["path"] for c in changes},
                {"first.py", "deleted.py", "new name.py", "retained.py"},
            )
            self.assertIn(
                {"status": "R100", "old_path": "old name.py", "path": "new name.py"},
                changes,
            )
            self.assertIn({"status": "D", "path": "deleted.py"}, changes)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def plan(self, tests):
        return {"modules": {"norm": {"tests": tests}}, "cann_exclusions": {}}

    def test_script_failure_propagates_and_later_tests_still_run_with_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / selector.LEGACY
            folder.mkdir(parents=True)
            (folder / "test_bad.py").write_text("raise SystemExit(7)\n")
            (folder / "test_good.py").write_text("print('ok')\n")
            tests = [selector.LEGACY + "test_bad.py", selector.LEGACY + "test_good.py"]
            self.assertEqual(runner.run(self.plan(tests), "norm", root, "9.0.0"), 1)
            self.assertIn(
                'failures="1"', (root / "test-results/legacy-test_bad.xml").read_text()
            )
            self.assertIn(
                'failures="0"', (root / "test-results/legacy-test_good.xml").read_text()
            )

    def test_public_failure_still_runs_legacy_and_preserves_accuracy_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                selector.PUBLIC + "norm/test_one.py",
                selector.LEGACY + "test_two.py",
            ]
            for path in paths:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text("def test_one(): pass\n")
            with patch.object(
                runner.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 1),
                    subprocess.CompletedProcess([], 0),
                ],
            ) as execute:
                self.assertEqual(runner.run(self.plan(paths), "norm", root, "9.0.0"), 1)
            command = execute.call_args_list[0].args[0]
            self.assertIn(paths[0], command)
            self.assertIn("--skip-acc-check=false", command)
            self.assertIn("--impl-ref=torch_native", command)
            self.assertEqual(len(execute.call_args_list), 2)

    def test_missing_or_escaping_path_is_error_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, "run") as execute:
                for path in [selector.LEGACY + "test_missing.py", "../test_escape.py"]:
                    with self.assertRaises(ValueError):
                        runner.run(self.plan([path]), "norm", Path(directory), "9.0.0")
                execute.assert_not_called()

    def test_existing_cann_exclusion_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = selector.LEGACY + "test_triangular_inverse.py"
            (root / path).parent.mkdir(parents=True)
            (root / path).write_text("raise SystemExit(1)\n")
            plan = self.plan([path])
            plan["cann_exclusions"] = {
                "9.1.0": {"test_triangular_inverse.py": "existing exclusion"}
            }
            with patch.object(runner.subprocess, "run") as execute:
                self.assertEqual(runner.run(plan, "norm", root, "9.1.0"), 0)
                execute.assert_not_called()
            self.assertIn(
                'skipped="1"',
                (root / "test-results/legacy-test_triangular_inverse.xml").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
