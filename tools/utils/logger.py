from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def write_execution_log(
	path: Path,
	*,
	command: Sequence[str],
	return_code: int | None,
	stderr: str,
) -> None:
	"""Persist subprocess diagnostics without mixing them into agent output."""
	path.parent.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now(timezone.utc).isoformat()
	rendered_command = " ".join(command)
	content = (
		f"timestamp_utc: {timestamp}\n"
		f"return_code: {return_code}\n"
		f"command: {rendered_command}\n"
		"\n--- stderr ---\n"
		f"{stderr.rstrip()}\n"
	)
	path.write_text(content, encoding="utf-8")
