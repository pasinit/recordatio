# Claude Instructions

## Project

agentmem is an HTTP proxy that sits between an agent and the Anthropic API. It injects relevant knowledge into system prompts (briefing), observes responses, extracts typed knowledge units (semantic, episodic, procedural), and consolidates them over time. See README.md for the full architecture.

The package lives at `src/tpasini/recordatio/`.

## Project Tracking

Use GitHub issues on `pasinit/recordatio` as the single source of truth for work tracking and planning. Before starting any non-trivial task, check open issues for context. When implementing a feature or fix, reference the relevant issue in commit messages. New work that isn't captured in an issue should be logged as one before proceeding.

## Code Style

Write idiomatic Python 3.13. Concretely:

- **No `from __future__ import annotations`** — postponed evaluation is not needed; use native syntax (`X | Y`, `list[str]`, `str | None`).
- Use built-in generic types (`list`, `dict`, `tuple`, `set`) instead of `typing.List`, `typing.Dict`, etc.
- Use `X | Y` union syntax instead of `typing.Union[X, Y]` and `X | None` instead of `typing.Optional[X]`.
- Use `typing.TypeAlias` / `type` statement (PEP 695) for type aliases where appropriate.
- Prefer `pathlib.Path` over `os.path`.

## Development

```bash
# set up
python3.13 -m venv .venv
.venv/bin/pip install pip-tools
make compile
.venv/bin/pip install -r requirements-dev.txt -e .

# daily workflow
make lint
make fix-lint
make test
make compile   # after editing requirements.in / requirements-dev.in
```
