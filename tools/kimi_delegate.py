from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from utils.logger import write_execution_log


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "runtime.yaml"


class DelegateError(RuntimeError):
	"""Raised when a delegated task cannot be prepared or completed."""


@dataclass(frozen=True)
class RuntimeConfig:
	executable: str
	model: str | None
	thinking_effort: str | None
	timeout_seconds: int
	max_output_chars: int
	default_agent: str
	writable_agents: frozenset[str]
	agents: dict[str, Path]
	workspace_root: Path
	task_dir: Path
	result_dir: Path
	log_dir: Path
	ledger_path: Path
	conventions_path: Path | None


def _resolve_project_path(value: str, field_name: str) -> Path:
	path = (ROOT / value).resolve()
	try:
		path.relative_to(ROOT)
	except ValueError as error:
		raise DelegateError(f"{field_name} must stay inside the project root: {value}") from error
	return path


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
	if not isinstance(value, dict):
		raise DelegateError(f"{field_name} must be a YAML mapping")
	return value


def load_config(path: Path) -> RuntimeConfig:
	try:
		raw = yaml.safe_load(path.read_text(encoding="utf-8"))
	except FileNotFoundError as error:
		raise DelegateError(f"Runtime config not found: {path}") from error
	except yaml.YAMLError as error:
		raise DelegateError(f"Invalid YAML in {path}: {error}") from error

	root = _require_mapping(raw, "runtime config")
	kimi = _require_mapping(root.get("kimi"), "kimi")
	paths = _require_mapping(root.get("paths"), "paths")
	agent_values = _require_mapping(root.get("agents"), "agents")

	agents = {
		str(name): _resolve_project_path(str(agent_path), f"agents.{name}")
		for name, agent_path in agent_values.items()
	}
	default_agent = str(kimi.get("default_agent", "worker"))
	if default_agent not in agents:
		raise DelegateError(f"Unknown default agent: {default_agent}")
	writable_agents = frozenset(str(name) for name in kimi.get("writable_agents", []))
	unknown_writable_agents = writable_agents.difference(agents)
	if unknown_writable_agents:
		raise DelegateError(
			"Unknown writable agents: " + ", ".join(sorted(unknown_writable_agents))
		)

	try:
		timeout_seconds = int(kimi.get("timeout_seconds", 1800))
		max_output_chars = int(kimi.get("max_output_chars", 8000))
	except (TypeError, ValueError) as error:
		raise DelegateError("timeout_seconds and max_output_chars must be integers") from error
	if timeout_seconds <= 0 or max_output_chars <= 0:
		raise DelegateError("timeout_seconds and max_output_chars must be positive")

	conventions_value = paths.get("conventions")
	model = kimi.get("model")
	thinking_effort = kimi.get("thinking_effort")
	return RuntimeConfig(
		executable=str(kimi.get("executable", "kimi")),
		model=str(model) if model else None,
		thinking_effort=str(thinking_effort) if thinking_effort else None,
		timeout_seconds=timeout_seconds,
		max_output_chars=max_output_chars,
		default_agent=default_agent,
		writable_agents=writable_agents,
		agents=agents,
		workspace_root=_resolve_project_path(
			str(paths.get("workspace_root", "workspace")), "paths.workspace_root"
		),
		task_dir=_resolve_project_path(str(paths.get("tasks", ".agent_runtime/tasks")), "paths.tasks"),
		result_dir=_resolve_project_path(
			str(paths.get("results", ".agent_runtime/results")), "paths.results"
		),
		log_dir=_resolve_project_path(str(paths.get("logs", ".agent_runtime/logs")), "paths.logs"),
		ledger_path=_resolve_project_path(
			str(paths.get("ledger", ".agent_runtime/ledger.csv")), "paths.ledger"
		),
		conventions_path=(
			_resolve_project_path(str(conventions_value), "paths.conventions")
			if conventions_value
			else None
		),
	)


def _new_task_id() -> str:
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _read_task(task: str | None, task_file: Path | None) -> str:
	if task is not None:
		value = task.strip()
	else:
		try:
			value = task_file.read_text(encoding="utf-8").strip() if task_file else ""
		except OSError as error:
			raise DelegateError(f"Cannot read task file: {error}") from error
	if not value:
		raise DelegateError("Task must not be empty")
	return value


def _prepare_runtime(config: RuntimeConfig) -> None:
	for directory in (config.task_dir, config.result_dir, config.log_dir):
		directory.mkdir(parents=True, exist_ok=True)
	for name, agent_file in config.agents.items():
		if not agent_file.is_file():
			raise DelegateError(f"Agent file for '{name}' not found: {agent_file}")


def _resolve_workspace(value: Path | None, agent_name: str, config: RuntimeConfig) -> Path:
	if value is None:
		if agent_name in config.writable_agents:
			raise DelegateError(f"--workspace is required for writable agent '{agent_name}'")
		return ROOT

	workspace = value if value.is_absolute() else ROOT / value
	workspace = workspace.resolve()
	try:
		workspace.relative_to(config.workspace_root)
	except ValueError as error:
		raise DelegateError(
			f"Workspace must stay inside {config.workspace_root}: {value}"
		) from error
	if not workspace.is_dir():
		raise DelegateError(f"Workspace directory not found: {workspace}")
	return workspace


def _build_prompt(
	task_file: Path,
	workspace: Path,
	max_output_chars: int,
	conventions: Path | None = None,
) -> str:
	readable = [f"Read the delegated task from: {task_file}"]
	if conventions is not None and conventions.is_file():
		readable.append(f"Read the shared cross-task conventions from: {conventions}")
	readable_files = "\n".join(readable)
	return f"""You are working for a GPT main agent.

{readable_files}
Your assigned workspace is: {workspace}
Treat that directory as the complete work scope. Do not read or modify files
outside it, except for reading the files listed above.

Complete the expensive reading, searching, or analysis yourself. Return only a
self-contained handoff with conclusions, changed files when applicable, key
findings, evidence, tests, risks, and recommended verification. Do not include
your chain of thought or copy large source passages. Keep the response under
{max_output_chars} characters.
"""


def _file_digest(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _snapshot_workspace(workspace: Path) -> dict[str, str]:
	snapshot: dict[str, str] = {}
	for current_root, directory_names, file_names in os.walk(workspace):
		directory_names[:] = [name for name in directory_names if name != ".git"]
		current_path = Path(current_root)
		for file_name in file_names:
			path = current_path / file_name
			relative_path = path.relative_to(workspace).as_posix()
			try:
				if path.is_symlink():
					snapshot[relative_path] = f"symlink:{os.readlink(path)}"
				elif path.is_file():
					snapshot[relative_path] = _file_digest(path)
			except OSError as error:
				raise DelegateError(f"Cannot snapshot workspace file {path}: {error}") from error
	return snapshot


def _workspace_changes(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
	before_paths = set(before)
	after_paths = set(after)
	return {
		"added": sorted(after_paths - before_paths),
		"modified": sorted(path for path in before_paths & after_paths if before[path] != after[path]),
		"deleted": sorted(before_paths - after_paths),
	}


def _format_change_list(
	changes: dict[str, list[str]], manifest_file: Path, max_chars: int = 3000
) -> str:
	lines = ["# Workspace Changes"]
	labels = (("added", "Added"), ("modified", "Modified"), ("deleted", "Deleted"))
	if not any(changes.values()):
		lines.append("\nNo file changes detected.")
	else:
		omitted = 0
		for key, label in labels:
			lines.append(f"\n## {label}")
			paths = changes[key]
			if not paths:
				lines.append("- None")
				continue
			for index, path in enumerate(paths):
				candidate = f"- `{path}`"
				if len("\n".join([*lines, candidate])) > max_chars:
					omitted += len(paths) - index
					break
				lines.append(candidate)
		if omitted:
			lines.append(f"\n{omitted} additional path(s) omitted from this handoff.")
	lines.append(f"\nFull manifest: {manifest_file}")
	return "\n".join(lines)


def _find_executable(executable: str) -> str | None:
	resolved = shutil.which(executable)
	if resolved or executable != "kimi":
		return resolved

	filename = "kimi.exe" if sys.platform == "win32" else "kimi"
	default_install = Path.home() / ".kimi-code" / "bin" / filename
	return str(default_install) if default_install.is_file() else None


LEDGER_FIELDS = [
	"task_id",
	"timestamp_utc",
	"agent",
	"workspace",
	"task_chars",
	"dry_run",
	"duration_seconds",
	"return_code",
	"result_chars",
	"truncated",
	"added",
	"modified",
	"deleted",
	"error",
	"outcome",
]


def _append_ledger(config: RuntimeConfig, row: dict[str, Any]) -> None:
	config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
	write_header = not config.ledger_path.is_file()
	with config.ledger_path.open("a", encoding="utf-8", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
		if write_header:
			writer.writeheader()
		writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def record_outcome(config: RuntimeConfig, task_id: str, outcome: str) -> str:
	if not config.ledger_path.is_file():
		raise DelegateError(f"Ledger not found: {config.ledger_path}")
	with config.ledger_path.open(encoding="utf-8", newline="") as stream:
		rows = list(csv.DictReader(stream))
	matched = False
	for row in rows:
		if row.get("task_id") == task_id:
			row["outcome"] = outcome
			matched = True
	if not matched:
		raise DelegateError(f"Task id not found in ledger: {task_id}")
	with config.ledger_path.open("w", encoding="utf-8", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
		writer.writeheader()
		writer.writerows(rows)
	return f"Ledger updated: {task_id} -> {outcome}"


def run_kimi(
	task: str,
	agent_name: str,
	config: RuntimeConfig,
	workspace_value: Path | None = None,
	dry_run: bool = False,
) -> str:
	_prepare_runtime(config)
	if agent_name not in config.agents:
		available = ", ".join(sorted(config.agents))
		raise DelegateError(f"Unknown agent '{agent_name}'. Available agents: {available}")
	workspace = _resolve_workspace(workspace_value, agent_name, config)

	task_id = _new_task_id()
	task_file = config.task_dir / f"{task_id}.md"
	result_file = config.result_dir / f"{task_id}.md"
	manifest_file = config.result_dir / f"{task_id}.changes.json"
	log_file = config.log_dir / f"{task_id}.log"
	task_file.write_text(
		f"# Delegated Task\n\n- ID: `{task_id}`\n- Agent: `{agent_name}`\n"
		f"- Workspace: `{workspace}`\n\n{task}\n",
		encoding="utf-8",
	)
	ledger_row: dict[str, Any] = {
		"task_id": task_id,
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"agent": agent_name,
		"workspace": str(workspace),
		"task_chars": len(task),
		"dry_run": dry_run,
	}

	executable = _find_executable(config.executable)
	if executable is None and not dry_run:
		raise DelegateError(
			f"Cannot find '{config.executable}' on PATH. Install Kimi Code CLI, "
			"run 'kimi login', then retry."
		)
	command = [
		executable or config.executable,
		"--agent-file",
		str(config.agents[agent_name]),
	]
	if config.model:
		command += ["-m", config.model]
	command += [
		"-p",
		_build_prompt(task_file, workspace, config.max_output_chars, config.conventions_path),
	]
	environment = None
	if config.thinking_effort:
		environment = os.environ.copy()
		environment["KIMI_MODEL_THINKING_EFFORT"] = config.thinking_effort

	if dry_run:
		_append_ledger(config, ledger_row)
		payload = {
			"task_id": task_id,
			"agent": agent_name,
			"workspace": str(workspace),
			"task_file": str(task_file),
			"result_file": str(result_file),
			"change_manifest": str(manifest_file),
			"log_file": str(log_file),
			"command": command,
		}
		return json.dumps(payload, ensure_ascii=False, indent=2)

	track_changes = agent_name in config.writable_agents
	before = _snapshot_workspace(workspace) if track_changes else {}
	started = time.monotonic()
	try:
		process = subprocess.run(
			command,
			cwd=workspace,
			env=environment,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			encoding="utf-8",
			errors="replace",
			timeout=config.timeout_seconds,
			check=False,
		)
	except subprocess.TimeoutExpired as error:
		after = _snapshot_workspace(workspace) if track_changes else {}
		changes = _workspace_changes(before, after)
		manifest_file.write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
		stderr = error.stderr if isinstance(error.stderr, str) else ""
		write_execution_log(
			log_file,
			command=command,
			return_code=None,
			stderr=f"Timed out after {config.timeout_seconds} seconds.\n{stderr}",
		)
		_append_ledger(
			config,
			ledger_row
			| {
				"duration_seconds": round(time.monotonic() - started, 3),
				"added": len(changes["added"]),
				"modified": len(changes["modified"]),
				"deleted": len(changes["deleted"]),
				"error": f"timed out after {config.timeout_seconds} seconds",
			},
		)
		raise DelegateError(
			f"Kimi timed out. See log: {log_file}. Changes: {manifest_file}"
		) from error
	except OSError as error:
		_append_ledger(config, ledger_row | {"error": f"failed to start: {error}"})
		raise DelegateError(f"Failed to start Kimi: {error}") from error

	after = _snapshot_workspace(workspace) if track_changes else {}
	changes = _workspace_changes(before, after)
	manifest_file.write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	write_execution_log(
		log_file,
		command=command,
		return_code=process.returncode,
		stderr=process.stderr,
	)
	ledger_row.update(
		{
			"duration_seconds": round(time.monotonic() - started, 3),
			"return_code": process.returncode,
			"added": len(changes["added"]),
			"modified": len(changes["modified"]),
			"deleted": len(changes["deleted"]),
		}
	)
	if process.returncode != 0:
		_append_ledger(config, ledger_row | {"error": f"exit code {process.returncode}"})
		raise DelegateError(
			f"Kimi exited with code {process.returncode}. See log: {log_file}. "
			f"Changes: {manifest_file}"
		)

	result = process.stdout.strip()
	if not result:
		_append_ledger(config, ledger_row | {"error": "empty response"})
		raise DelegateError(
			f"Kimi returned an empty response. See log: {log_file}. Changes: {manifest_file}"
		)
	change_list = _format_change_list(changes, manifest_file)
	complete_result = f"{result}\n\n{change_list}"
	result_file.write_text(complete_result + "\n", encoding="utf-8")

	truncated = len(complete_result) > config.max_output_chars
	_append_ledger(
		config,
		ledger_row | {"result_chars": len(result), "truncated": truncated},
	)
	if not truncated:
		return complete_result
	reserved = len(change_list) + 80
	result_budget = max(0, config.max_output_chars - reserved)
	return (
		result[:result_budget].rstrip()
		+ "\n\n[AGENT OUTPUT TRUNCATED]\n"
		+ f"Full result: {result_file}\n\n"
		+ change_list
	)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Delegate a task to Kimi Code CLI.")
	source = parser.add_mutually_exclusive_group()
	source.add_argument("--task", help="Task text delegated by the main agent")
	source.add_argument("--task-file", type=Path, help="UTF-8 Markdown or text task file")
	parser.add_argument("--agent", help="Agent alias from config/runtime.yaml")
	parser.add_argument(
		"--workspace",
		type=Path,
		help="Assigned directory under the configured workspace root; required for writable agents",
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Runtime YAML path")
	parser.add_argument("--dry-run", action="store_true", help="Prepare the task without starting Kimi")
	parser.add_argument(
		"--record-outcome",
		metavar="TASK_ID",
		help="Set the outcome label of an existing ledger row instead of delegating",
	)
	parser.add_argument(
		"--outcome",
		choices=["accepted", "reworked", "not_used"],
		help="Outcome label recorded with --record-outcome",
	)
	return parser


def main() -> int:
	args = build_parser().parse_args()
	try:
		config = load_config(args.config.resolve())
		if args.record_outcome is not None:
			if args.task is not None or args.task_file is not None:
				raise DelegateError("Do not combine --record-outcome with --task/--task-file")
			if args.outcome is None:
				raise DelegateError("--record-outcome requires --outcome")
			print(record_outcome(config, args.record_outcome, args.outcome))
			return 0
		if args.outcome is not None:
			raise DelegateError("--outcome requires --record-outcome")
		if args.task is None and args.task_file is None:
			raise DelegateError("Provide --task/--task-file, or --record-outcome with --outcome")
		task = _read_task(args.task, args.task_file)
		agent_name = args.agent or config.default_agent
		print(
			run_kimi(
				task,
				agent_name,
				config,
				workspace_value=args.workspace,
				dry_run=args.dry_run,
			)
		)
	except DelegateError as error:
		print(f"KIMI_DELEGATE_ERROR: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
