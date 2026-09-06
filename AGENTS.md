# Multi-Agent Routing Policy

The main agent owns user intent, scientific/architectural decisions, critical
verification, acceptance of changes and the final response. Delegate only when
it reduces main-agent work after specification, review and repair are counted.

## Routing and task scope

- `reader`: long documents/papers, structured evidence and source locations.
- `worker`: repository/log inspection, comparisons and repetitive extraction.
- `researcher`: literature discovery and primary-source comparison.
- `coder`: one bounded function/module plus focused tests, or a small mechanical
  change. Requires `--workspace`; cannot execute commands.
- `tester`: runs the focused test suite or verification commands in an assigned
  workspace and reports exact results. Requires `--workspace`; cannot edit
  files. Use it to close the coder test loop before main-agent review.

Prefer the first three for context-heavy work. Keep short local tasks, user
intent, scientific validity, architecture, conflicting evidence and synthesis
with the main agent. Do not hand off a whole experiment that the main agent
would need to read and reconstruct in full.

Delegate plumbing and artifact code (I/O, schemas, plotting, scaffolding) whose
acceptance is mechanical. Keep statistical or algorithmic cores with the main
agent: if a task spec starts needing formulas, that layer is not delegable.

Write a short task contract: exact question/output, allowed inputs and paths,
acceptance checks, handoff size and deadline. For code, name allowed files and
interfaces/invariants. Broader work needs an explicit reviewable boundary.
Stable cross-task rules live in [CONVENTIONS.md](CONVENTIONS.md); task files
reference it instead of repeating those rules. Pass paths and precise
questions; do not read the whole corpus before delegating or paste it into the
prompt. Stage only selected inputs once if necessary.
Never send credentials, private keys or unrelated sensitive data.

Use UTF-8 on Windows and task files for nontrivial instructions:

```powershell
python -X utf8 tools/kimi_delegate.py --agent reader --task-file path/to/task.md
python -X utf8 tools/kimi_delegate.py --agent coder --workspace workspace/repos/task --task-file path/to/task.md
python -X utf8 tools/kimi_delegate.py --agent tester --workspace workspace/repos/task --task "Run python -m pytest -q and report exact results"
```

Workspace boundaries, role restrictions and filesystem permissions still apply.

## Waiting and ownership

Kimi may take tens of minutes. Silence, slow responses and absent output files
are not evidence of failure. The default deadline is **60 minutes**
(`runtime.yaml: timeout_seconds=3600`). For known larger tasks, configure a longer
deadline before launch. Never shorten it merely because the worker is quiet.

Once delegated, the deliverable belongs to the worker until completion or a
confirmed failure/cancellation. **Do not write a second implementation, repeat
the document reading, or launch a duplicate worker while it is running.** Do not
take over merely to appear busy or respond faster.

Prefer completion notifications or background waits. Follow the host's user
update and wait-duration requirements; a brief update does not require reading
logs. Without an error, diagnostic checks should be no more frequent than every
five minutes and only when they can change a decision. Do not repeatedly open
raw session logs or inspect processes/files. Meanwhile, do only already-needed
independent work; do not invent extra reports, tests or plots to fill waiting time.

Takeover requires an actual worker failure, unrecoverable tool/auth error,
configured deadline expiry, or explicit user cancellation/deadline change.
Before retrying or taking over, confirm the old worker and relevant child
processes have stopped; inspect saved results and the change manifest first.
A wrapper display/encoding error after successful completion does not justify
a new model call. Recover the saved result instead.

## Handoff and verification

Aim for **1500 characters** of handoff; the configured return cap is **4000**.
Include conclusions, exact evidence locations, changed paths, tests actually
run/not run and unresolved issues. Keep code and large tables in files; omit
working traces and full source dumps.

Read the handoff first, then verify important claims at their cited locations.
For code, send the coder's suggested test command to `tester` first and review
the diff after tests are green; inspect the full change manifest and changed
code. A short summary never replaces code review; scope tasks small enough
that full review of their changes is affordable. Avoid rereading untouched
files or repeating the whole delegated investigation.

For a localized failure, send one focused correction to Kimi in the same
workspace with the failure and acceptance condition. Repair the existing code
instead of starting a second version. If the approach is invalid, the main agent
decides the new approach and records why.

## Cost discipline

- Normally at most **3 Kimi calls per user request**, including corrections.
  Code tasks using the coder + tester loop may use up to **5**; the extra calls
  stay on the Kimi side and must not add main-agent turns.
- Reuse known configuration and task records; no repeated setup audit or test
  model invocation when nothing changed. Do not delegate trivial work.
- Do not pay both agents to process the same corpus unless independent review
  is needed; route large reading/extraction before reading it yourself.
- Record the outcome label with
  `python -X utf8 tools/kimi_delegate.py --record-outcome TASK_ID --outcome accepted|reworked|not_used`
  so `.agent_runtime/ledger.csv` stays complete, plus a brief reason in the
  existing handoff or task report. Do not create a long coordination report.
- Do not claim allowance savings without usage evidence. If coordination and
  review exceed work avoided, narrow or stop delegating that task type.

`config/delegation_policy.yaml` guides the main agent; it is not an automatic
scheduler. The wrapper enforces runtime timeout/return cap and workspace checks.
The main agent enforces routing, waiting, ownership and review.
