# Kimi Delegation Conventions

Stable rules that apply to every delegated task. Task files reference this
document and state only task-specific content instead of repeating these rules.

## Scope

- Work only inside the assigned workspace. Besides workspace files, read only
  the task file and this conventions file; do not access other paths.
- Do not create reports, notebooks, experiment pipelines, or output
  directories unless the task explicitly requires them.

## Code defaults (unless the task says otherwise)

- No file or network I/O, no plotting, no global mutable state, no randomness.
- Preserve specified interfaces, statistical definitions, units, axes,
  sampling rules, and invariants. Flag ambiguity instead of silently changing
  it.
- Do not mutate caller-provided input arrays or data structures.
- Match the existing style of the workspace and avoid unrelated refactors.
- For a correction, repair the existing code in the same workspace; do not
  start a second implementation.

## Testing

- Never claim that tests passed unless you actually ran them. If you cannot
  run commands, state the exact command the main agent should run.
- Default test command: `python -m pytest -q` from the workspace root.
- On Windows, use `python -X utf8` when output may contain non-ASCII text.

## Handoff

- Use the section layout defined by your agent role.
- Aim for 1500 characters; the hard cap is 4000. Cite file paths and line
  numbers instead of copying code, tables, or raw logs.
- Do not narrate your process and do not repeat the task specification.
- State unresolved issues and recommended verification explicitly.
