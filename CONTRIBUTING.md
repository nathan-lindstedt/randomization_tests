# Contributing

Thank you for your interest in contributing to **randomization_tests**.

## Branch strategy

| Branch                        | Purpose                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `main`                      | Instructional branch — not actively developed.                                                        |
| `experimental`              | Active development branch where all hardening, performance, and feature work lands.                    |
| `v0.X.0` (feature branches) | Branch off `experimental` for a major version scope, merge back into `experimental` when complete. |

All CI runs against `experimental`.  `main` remains a simple,
self-contained reference implementation.

## Development setup

```bash
git clone https://github.com/nathanlindstedt/randomization_tests.git
cd randomization_tests
git checkout experimental
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Pre-commit hooks

The repository includes a `.pre-commit-config.yaml` that runs ruff
lint, ruff format, mypy, and basic file hygiene checks on every commit.

```bash
pip install pre-commit
pre-commit install
```

After this, hooks run automatically on `git commit`.  To run them
manually against all files:

```bash
pre-commit run --all-files
```

## Running tests

```bash
pytest                  # default — excludes slow tests
pytest -m slow          # large-n smoke tests only
pytest -m ""            # all tests including slow
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

## Dependency compatibility

- Lower bounds in `pyproject.toml` are the **oldest tested versions**.
  The `test-deps-lower` CI job installs exact lower bounds on Python
  3.10 and runs the test suite to validate them.
- **No upper bounds** are pinned.  This is library best practice —
  upper-bound pins cause resolver conflicts for downstream users.
- Do not use APIs introduced in a version newer than the declared lower
  bound without bumping the floor in `pyproject.toml`.
- Consult the lower bounds before using any recently-added function:

| Package      | Lower bound |
| ------------ | ----------- |
| numpy        | 1.24        |
| pandas       | 2.0         |
| scipy        | 1.10        |
| statsmodels  | 0.14        |
| scikit-learn | 1.3         |

## Pull request guidelines

1. Fork the repository and create your branch from `experimental`.
2. Add tests for any new functionality.
3. Ensure `pytest`, `ruff check`, and `mypy` all pass before opening a PR.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Keep commits focused — one logical change per commit.

## Reporting issues

Please open an issue on GitHub with a minimal reproducible example and the
output of `python -c "import randomization_tests; print(randomization_tests.__version__)"`.
