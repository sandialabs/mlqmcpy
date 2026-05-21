[![Tests](https://github.com/PieterjanRobbe/mlqmcpy/actions/workflows/tests.yml/badge.svg)](https://github.com/PieterjanRobbe/mlqmcpy/actions/workflows/tests.yml)

# mlqmcpy

Multilevel (quasi-)Monte Carlo methods in Python.

`mlqmcpy` provides iterators and example problems for estimating expectations with
Monte Carlo, quasi-Monte Carlo, multilevel Monte Carlo, and multilevel
quasi-Monte Carlo methods.

## Installation

```bash
pip install mlqmcpy
```

## Quick start

```python
from mlqmcpy import GreedyMLQMCIterator
from mlqmcpy.problems.analytic import analytic

iterator = GreedyMLQMCIterator(2, error_tolerance=1e-2, seed=1234)

for new_samples in iterator:
    new_results = {
        level: analytic.ml(level, samples)
        for level, samples in new_samples.items()
    }
    iterator.update(new_results)

print(iterator.mean)
print(iterator.standard_error)
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
As a library, `mlqmcpy` does not commit a lockfile; development and CI resolve
from `pyproject.toml`.

```bash
uv sync --group dev --refresh
uv run pytest
uv run ruff check mlqmcpy tests
uv run black --check mlqmcpy tests
uv run isort --check-only mlqmcpy tests
uv build
```

Releases are published from GitHub Releases through PyPI Trusted Publishing.
Configure the PyPI trusted publisher for repository `PieterjanRobbe/mlqmcpy`,
workflow `.github/workflows/release.yml`, and environment `pypi`.
