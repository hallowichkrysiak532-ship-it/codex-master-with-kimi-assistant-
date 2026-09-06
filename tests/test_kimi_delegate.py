from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import kimi_delegate


class KimiDelegateTests(unittest.TestCase):
	def setUp(self) -> None:
		base_config = kimi_delegate.load_config(ROOT / "config" / "runtime.yaml")
		self.workspace_temp = tempfile.TemporaryDirectory(dir=base_config.workspace_root)
		self.runtime_temp = tempfile.TemporaryDirectory(dir=ROOT / ".agent_runtime")
		self.workspace = Path(self.workspace_temp.name)
		runtime_root = Path(self.runtime_temp.name)
		self.config = replace(
			base_config,
			task_dir=runtime_root / "tasks",
			result_dir=runtime_root / "results",
			log_dir=runtime_root / "logs",
			ledger_path=runtime_root / "ledger.csv",
		)

	def tearDown(self) -> None:
		self.runtime_temp.cleanup()
		self.workspace_temp.cleanup()

	def test_coder_requires_workspace(self) -> None:
		with self.assertRaisesRegex(kimi_delegate.DelegateError, "--workspace is required"):
			kimi_delegate.run_kimi("Implement a feature", "coder", self.config, dry_run=True)

	def test_workspace_must_stay_inside_configured_root(self) -> None:
		with self.assertRaisesRegex(kimi_delegate.DelegateError, "must stay inside"):
			kimi_delegate.run_kimi(
				"Implement a feature",
				"coder",
				self.config,
				workspace_value=ROOT,
				dry_run=True,
			)

	def test_dry_run_command_uses_configured_model(self) -> None:
		payload = json.loads(
			kimi_delegate.run_kimi(
				"Implement a feature",
				"coder",
				self.config,
				workspace_value=self.workspace,
				dry_run=True,
			)
		)
		command = payload["command"]
		self.assertIn("-m", command)
		self.assertEqual(command[command.index("-m") + 1], "kimi-code/k3")

	def test_thinking_effort_passed_via_environment(self) -> None:
		captured: dict[str, object] = {}

		def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
			captured.update(kwargs)
			return subprocess.CompletedProcess(args=[], returncode=0, stdout="Done.", stderr="")

		with (
			patch("kimi_delegate._find_executable", return_value="kimi"),
			patch("kimi_delegate.subprocess.run", side_effect=fake_run),
		):
			kimi_delegate.run_kimi(
				"Implement a feature",
				"coder",
				self.config,
				workspace_value=self.workspace,
			)

		environment = captured["env"]
		self.assertIsNotNone(environment)
		self.assertEqual(environment["KIMI_MODEL_THINKING_EFFORT"], "max")

	def test_returns_added_modified_and_deleted_files(self) -> None:
		modified = self.workspace / "modified.txt"
		deleted = self.workspace / "deleted.txt"
		modified.write_text("before", encoding="utf-8")
		deleted.write_text("delete me", encoding="utf-8")

		def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
			modified.write_text("after", encoding="utf-8")
			deleted.unlink()
			(self.workspace / "added.txt").write_text("new", encoding="utf-8")
			return subprocess.CompletedProcess(args=[], returncode=0, stdout="Implemented.", stderr="")

		with (
			patch("kimi_delegate._find_executable", return_value="kimi"),
			patch("kimi_delegate.subprocess.run", side_effect=fake_run),
		):
			output = kimi_delegate.run_kimi(
				"Implement a feature",
				"coder",
				self.config,
				workspace_value=self.workspace,
			)

		self.assertIn("# Workspace Changes", output)
		self.assertIn("`added.txt`", output)
		self.assertIn("`modified.txt`", output)
		self.assertIn("`deleted.txt`", output)

		manifest_files = list(self.config.result_dir.glob("*.changes.json"))
		self.assertEqual(len(manifest_files), 1)
		manifest = manifest_files[0].read_text(encoding="utf-8")
		self.assertIn('"added.txt"', manifest)
		self.assertIn('"modified.txt"', manifest)
		self.assertIn('"deleted.txt"', manifest)

	def _fake_successful_run(self) -> str:
		completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="Done.", stderr="")
		with (
			patch("kimi_delegate._find_executable", return_value="kimi"),
			patch("kimi_delegate.subprocess.run", return_value=completed),
		):
			return kimi_delegate.run_kimi(
				"Implement a feature",
				"coder",
				self.config,
				workspace_value=self.workspace,
			)

	def _ledger_rows(self) -> list[dict[str, str]]:
		with self.config.ledger_path.open(encoding="utf-8", newline="") as stream:
			return list(csv.DictReader(stream))

	def test_successful_run_appends_ledger_row(self) -> None:
		self._fake_successful_run()

		rows = self._ledger_rows()
		self.assertEqual(len(rows), 1)
		row = rows[0]
		self.assertEqual(row["agent"], "coder")
		self.assertEqual(row["task_chars"], str(len("Implement a feature")))
		self.assertEqual(row["dry_run"], "False")
		self.assertEqual(row["return_code"], "0")
		self.assertEqual(row["result_chars"], str(len("Done.")))
		self.assertEqual(row["truncated"], "False")
		self.assertEqual(row["error"], "")
		self.assertEqual(row["outcome"], "")
		self.assertTrue(row["duration_seconds"])

	def test_record_outcome_updates_ledger(self) -> None:
		self._fake_successful_run()
		task_id = self._ledger_rows()[0]["task_id"]

		message = kimi_delegate.record_outcome(self.config, task_id, "accepted")

		self.assertIn(task_id, message)
		rows = self._ledger_rows()
		self.assertEqual(len(rows), 1)
		self.assertEqual(rows[0]["outcome"], "accepted")
		self.assertEqual(rows[0]["agent"], "coder")

		with self.assertRaisesRegex(kimi_delegate.DelegateError, "not found"):
			kimi_delegate.record_outcome(self.config, "20990101T000000Z-missing", "reworked")


if __name__ == "__main__":
	unittest.main()
