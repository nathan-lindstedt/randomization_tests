# Contributing

Thank you for your interest in contributing to **randomization_tests**.

## Development setup

```bash
git clone https://github.com/nathanlindstedt/randomization_tests.git
cd randomization_tests
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Linting and formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting. Configuration lives in `pyproject.toml`.

```bash
ruff check src/ tests/       # lint
ruff format src/ tests/       # format
```

## Type checking

```bash
mypy src/randomization_tests/
```

## Pull request guidelines

1. Fork the repository and create your branch from `main`.
2. Add tests for any new functionality.
3. Ensure `pytest`, `ruff check`, and `mypy` all pass before opening a PR.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Keep commits focused — one logical change per commit.

## Reporting issues

Please open an issue on GitHub with a minimal reproducible example and the
output of `python -c "import randomization_tests; print(randomization_tests.__version__)"`.
