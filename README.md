# Codex + Kimi Multi-Agent

This project uses a GPT/Codex main agent for planning and final judgment, while
delegating context-heavy analysis and scoped first-pass implementation to Kimi
Code CLI. It uses subscription login through each official CLI and does not
require OpenAI or Moonshot API keys.

## Architecture

```text
User -> GPT/Codex main agent -> tools/kimi_delegate.py -> Kimi Code CLI
										  |                       |
										  |                       +-- read/search/analyze/edit
										  +-- tasks/results/change manifests/logs
```

Kimi is asked for a concise handoff; the wrapper truncates oversized output
before returning it. Full output and diagnostics remain under `.agent_runtime/`.

## Cost and waiting policy

The main policy is [AGENTS.md](AGENTS.md); the structured counterpart is
[delegation_policy.yaml](config/delegation_policy.yaml). Prefer Kimi for long
reading, extraction and comparisons. Code tasks should normally cover one
bounded module and focused tests. The main agent owns design and final review.

Kimi may take tens of minutes. **Do not interpret silence as failure or start a
duplicate implementation while it runs.** The runtime deadline is **60 minutes**,
increased from 30 minutes. For a known larger task, configure a longer deadline
before launch. Prefer background completion; maintain user updates as required
by the host without repeatedly opening logs. Do only already-needed independent
work while waiting.

Aim for **1500 characters** of handoff. `runtime.yaml` caps returned output at
**4000 characters**, reduced from 8000. Verify cited evidence; for code, inspect
the change manifest and changed code and run focused tests. Route a localized
failure back as one focused correction in the same workspace, within the normal
call budget (three calls per user request, or five for coder/tester loops).
Short handoffs do not replace code review; delegation alone does not guarantee
lower GPT usage.

The wrapper enforces its timeout, output truncation and workspace checks.
Routing, waiting, ownership and review rules guide the main agent; the policy
YAML is not an automatic scheduler. On failure or timeout, confirm the old worker
and relevant child processes have stopped and inspect saved results before
retrying or taking over.

Windows examples use `python -X utf8` to avoid GBK display failures. If printing
fails after the worker completed, recover the saved result and change manifest;
do not rerun the model just to print its answer again.

A compact task file can contain:

```text
Question/deliverable: one specific result
Inputs/workspace: exact allowed paths; stage only needed inputs if necessary
Allowed changes: named files/interfaces, or read-only
Acceptance: focused checks and invariants
Handoff: aim for 1500 characters, evidence locations and unresolved issues
Deadline: 60 minutes; longer when chosen before launch
```

## Prerequisites

- Python 3.10 or newer
- Kimi Code CLI with a Kimi Code membership login
- Codex CLI if Codex is used as the main agent

Install Kimi Code CLI on Windows PowerShell using the command from the official
Kimi documentation:

```powershell
irm https://code.kimi.com/kimi-code/install.ps1 | iex
$env:Path = "$HOME\.kimi-code\bin;$env:Path"
kimi --version
kimi login
```

The installer updates the user-level `PATH`, but an already-open VS Code or
PowerShell process may still have the old value. The `$env:Path` line refreshes
the current shell immediately. Restarting VS Code also makes the persisted PATH
change available to new terminals.

Install the Python dependency:

```powershell
python -m pip install -r requirements.txt
```

Optional Codex installation:

```powershell
npm install -g @openai/codex
codex
```

Use subscription/OAuth login in both CLIs. Do not add API keys to this project.

## Usage

Test configuration and task preparation without calling Kimi:

```powershell
python -X utf8 tools/kimi_delegate.py --agent worker --task "Inspect workspace/repos" --dry-run
```

Delegate repository analysis:

```powershell
python -X utf8 tools/kimi_delegate.py --agent worker --task "Scan workspace/repos/my-project and summarize its architecture with file evidence. Do not modify files."
```

Delegate a paper review:

```powershell
python -X utf8 tools/kimi_delegate.py --agent reader --task "Read workspace/papers/example.pdf and summarize the method, evidence, and limitations."
```

Delegate current web research:

```powershell
python -X utf8 tools/kimi_delegate.py --agent researcher --task "Compare current official documentation for the selected tools and cite sources."
```

Delegate a scoped first-pass implementation:

```powershell
python -X utf8 tools/kimi_delegate.py --agent coder --workspace workspace/repos/my-project --task "Implement the requested feature and add focused tests. Do not make unrelated changes."
```

Run the focused tests after a coder handoff, keeping the test-fix loop on the
Kimi side:

```powershell
python -X utf8 tools/kimi_delegate.py --agent tester --workspace workspace/repos/my-project --task "Run python -m pytest -q and report exact results. Do not modify files."
```

`--workspace` may only resolve beneath the `workspace_root` configured in
`config/runtime.yaml`, and it is mandatory for writable agents. Kimi runs with
that directory as its working directory. This is a routing boundary, not an OS
sandbox, so review the resulting diff before accepting changes.

The wrapper fingerprints files before and after a writable task. Its response
includes an added/modified/deleted list, while the complete machine-readable
manifest is saved as `.agent_runtime/results/<task-id>.changes.json`. Existing
unrelated dirty files are not reported unless their contents changed during the
Kimi invocation.

For very long task specifications, use `--task-file path/to/task.md` instead of
`--task`. Stable cross-task rules live in [CONVENTIONS.md](CONVENTIONS.md) and
are referenced from the generated prompt, so task files only need the
task-specific question, allowed files, invariants and acceptance checks.
Runtime defaults and agent aliases are defined in
`config/runtime.yaml`; routing guidance is in `config/delegation_policy.yaml`.

## Runtime Data

- `.agent_runtime/tasks/`: exact delegated task records
- `.agent_runtime/results/`: complete Kimi final responses
- `.agent_runtime/results/*.changes.json`: deterministic file change manifests
- `.agent_runtime/logs/`: Kimi stderr, exit code, and command metadata
- `.agent_runtime/ledger.csv`: one row per delegation (agent, workspace, task
  and handoff sizes, duration, return code, change counts, error). After
  review, record the outcome label with
  `python -X utf8 tools/kimi_delegate.py --record-outcome TASK_ID --outcome accepted|reworked|not_used`.
- `outputs/`: reports and summaries intended for people

Runtime records may contain private prompts, paths, or tool output. They are
ignored by Git and should be reviewed before sharing.

## Safety Model

The worker, reader, and researcher allow only read and search tools. The coder
also allows `Write` and `Edit`, but denies `Bash`, so it cannot run tests or
commands. The tester allows `Bash` but denies `Write` and `Edit`; it is
snapshot-tracked like the coder, so unexpected file changes during a test run
are attributed. Kimi results and edits must still be treated as untrusted work:
the main agent reviews the manifest and diff, then runs the relevant tests.
