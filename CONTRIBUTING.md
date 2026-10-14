# Contributing

Thanks for helping improve Apex Driver Lab. Contributions should be focused, tested, and clear about their impact on simulation behavior.

## Before you start

- Search existing issues and pull requests before opening a duplicate.
- For a substantial change, open an issue first to discuss scope and compatibility.
- Do not commit trained model checkpoints, virtual environments, secrets, or generated output.

## Development setup

Use Python 3.12 and install the project in editable mode with test dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest -q
```

Linux/macOS users can activate `.venv` with their shell's standard `source .venv/bin/activate` command.

## Pull request expectations

- Keep changes within the stated problem and preserve the existing architecture.
- Add or update tests for changed behavior.
- Run `python -m pytest -q` before submitting.
- Update user-facing documentation when commands, configuration, or behavior changes.
- Explain important tradeoffs and note any known limitations in the pull request description.
- Keep commits free of local model files and unrelated generated artifacts.

## Review process

A maintainer will review correctness, tests, maintainability, and documentation. Changes may receive requests for revision before they are accepted. No response or merge timeline is guaranteed.
