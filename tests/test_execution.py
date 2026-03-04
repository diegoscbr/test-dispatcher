import csv
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def load_execution_module():
    for name in ("execution", "prompts", "query", "follow_up_prompts"):
        sys.modules.pop(name, None)

    prompts_mod = types.ModuleType("prompts")
    query_mod = types.ModuleType("query")
    follow_mod = types.ModuleType("follow_up_prompts")

    prompts_mod.main = lambda: 0
    query_mod.main = lambda: 0
    follow_mod.main = lambda: 0
    follow_mod.detects_follow_up_question = lambda _text: False

    sys.modules["prompts"] = prompts_mod
    sys.modules["query"] = query_mod
    sys.modules["follow_up_prompts"] = follow_mod

    return importlib.import_module("execution")


class TestExecutionHelpers(unittest.TestCase):
    def setUp(self):
        self.execution = load_execution_module()

    def test_normalize_exit_code(self):
        self.assertEqual(self.execution._normalize_exit_code(5), 5)
        self.assertEqual(self.execution._normalize_exit_code(None), 0)
        self.assertEqual(self.execution._normalize_exit_code("bad"), 1)

    def test_run_entrypoint_normal_return(self):
        code = self.execution._run_entrypoint("demo", lambda: None)
        self.assertEqual(code, 0)

    def test_run_entrypoint_handles_system_exit(self):
        def _exit():
            raise SystemExit("fail")

        code = self.execution._run_entrypoint("demo", _exit)
        self.assertEqual(code, 1)

    def test_wait_for_required_files_polls_until_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.execution.REQUIRED_FILES = ("a.csv", "b.csv")
            sleep_calls = {"count": 0}

            def fake_sleep(_seconds):
                sleep_calls["count"] += 1
                if sleep_calls["count"] == 1:
                    (base / "a.csv").write_text("", encoding="utf-8")
                    (base / "b.csv").write_text("", encoding="utf-8")

            with mock.patch.object(self.execution.time, "sleep", side_effect=fake_sleep):
                self.execution._wait_for_required_files(base)

            self.assertEqual(sleep_calls["count"], 1)

    def test_has_follow_up_candidates_detects_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["response"])
                writer.writeheader()
                writer.writerow({"response": "plain answer"})
                writer.writerow({"response": "needs follow-up"})

            with mock.patch.object(
                self.execution.follow_up_prompts,
                "detects_follow_up_question",
                side_effect=lambda text: "follow-up" in text,
            ):
                self.assertTrue(self.execution._has_follow_up_candidates(path))

    def test_has_follow_up_candidates_returns_false_when_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["response"])
                writer.writeheader()
                writer.writerow({"response": "all complete"})

            with mock.patch.object(
                self.execution.follow_up_prompts,
                "detects_follow_up_question",
                return_value=False,
            ):
                self.assertFalse(self.execution._has_follow_up_candidates(path))

    def test_run_query_with_clean_argv_restores_argv(self):
        original = ["execution.py", "--extra"]
        self.execution.sys.argv = original[:]

        def fake_main():
            self.assertEqual(self.execution.sys.argv, ["execution.py"])
            return 0

        with mock.patch.object(self.execution.query, "main", side_effect=fake_main):
            code = self.execution._run_query_with_clean_argv()

        self.assertEqual(code, 0)
        self.assertEqual(self.execution.sys.argv, original)

    def test_run_followup_with_clean_argv_restores_argv(self):
        original = ["execution.py", "--extra"]
        self.execution.sys.argv = original[:]

        def fake_main():
            self.assertEqual(self.execution.sys.argv, ["execution.py"])
            return 0

        with mock.patch.object(self.execution.follow_up_prompts, "main", side_effect=fake_main):
            code = self.execution._run_followup_with_clean_argv()

        self.assertEqual(code, 0)
        self.assertEqual(self.execution.sys.argv, original)


class TestExecutionMain(unittest.TestCase):
    def setUp(self):
        self.execution = load_execution_module()

    def test_main_returns_prompts_failure_code(self):
        with mock.patch.object(self.execution, "_run_entrypoint", return_value=3), mock.patch.object(
            self.execution, "_wait_for_required_files"
        ) as wait_mock:
            code = self.execution.main()

        self.assertEqual(code, 3)
        wait_mock.assert_not_called()

    def test_main_returns_query_failure_code(self):
        with mock.patch.object(self.execution, "_run_entrypoint", return_value=0), mock.patch.object(
            self.execution, "_wait_for_required_files"
        ), mock.patch.object(self.execution, "_run_query_with_clean_argv", return_value=7):
            code = self.execution.main()

        self.assertEqual(code, 7)

    def test_main_skips_followup_when_responses_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            with mock.patch.object(self.execution.Path, "cwd", return_value=workdir), mock.patch.object(
                self.execution, "_run_entrypoint", return_value=0
            ), mock.patch.object(self.execution, "_wait_for_required_files"), mock.patch.object(
                self.execution, "_run_query_with_clean_argv", return_value=0
            ), mock.patch.object(
                self.execution, "_has_follow_up_candidates"
            ) as has_followups:
                code = self.execution.main()

        self.assertEqual(code, 0)
        has_followups.assert_not_called()

    def test_main_skips_followup_when_no_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "responses.csv").write_text("response\nok\n", encoding="utf-8")
            with mock.patch.object(self.execution.Path, "cwd", return_value=workdir), mock.patch.object(
                self.execution, "_run_entrypoint", return_value=0
            ), mock.patch.object(self.execution, "_wait_for_required_files"), mock.patch.object(
                self.execution, "_run_query_with_clean_argv", return_value=0
            ), mock.patch.object(
                self.execution, "_has_follow_up_candidates", return_value=False
            ), mock.patch.object(
                self.execution, "_run_followup_with_clean_argv"
            ) as run_followup:
                code = self.execution.main()

        self.assertEqual(code, 0)
        run_followup.assert_not_called()

    def test_main_runs_followup_when_candidates_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "responses.csv").write_text("response\nneeds follow-up\n", encoding="utf-8")
            with mock.patch.object(self.execution.Path, "cwd", return_value=workdir), mock.patch.object(
                self.execution, "_run_entrypoint", return_value=0
            ), mock.patch.object(self.execution, "_wait_for_required_files"), mock.patch.object(
                self.execution, "_run_query_with_clean_argv", return_value=0
            ), mock.patch.object(
                self.execution, "_has_follow_up_candidates", return_value=True
            ), mock.patch.object(
                self.execution, "_run_followup_with_clean_argv", return_value=9
            ):
                code = self.execution.main()

        self.assertEqual(code, 9)


if __name__ == "__main__":
    unittest.main()
