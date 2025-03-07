import numpy as np
import pytest

from mlqmcpy import GreedyMLMCIterator
from mlqmcpy.problems.analytic import analytic


def test_analytic_mc():

    # Define greedy MC iterator
    mc_level = 4
    max_budget = 1024
    seed = 1234
    iterator = GreedyMLMCIterator(2, max_budget=max_budget, seed=seed)

    # Take samples
    results = {0: analytic(mc_level, np.random.rand(1024, 2))}
    iterator.update(results)

    # Test accuracy
    assert analytic.exact.Q.mean(mc_level) == pytest.approx(
        iterator.mean, abs=3 * iterator.standard_error
    )


def test_analytic_mlmc():

    # Define greedy MC iterator
    num_levels = 5
    cost_per_level = [2**level for level in range(num_levels)]
    error_tolerance = 1e-2
    seed = 12
    iterator = GreedyMLMCIterator(
        2,
        cost_per_level=cost_per_level,
        error_tolerance=error_tolerance,
        seed=seed,
    )

    # Take samples
    results = {
        level: analytic.ml(level, np.random.rand(1024, 2))
        for level in range(num_levels)
    }
    iterator.update(results)

    # Test accuracy
    for level in range(1, num_levels):
        mean = iterator._response_accumulators[level].mean
        variance = iterator._response_accumulators[level].variance
        standard_error = np.sqrt(
            iterator._response_accumulators[level].squared_standard_error
        )
        assert analytic.exact.Y.mean(level) == pytest.approx(
            mean, abs=3 * standard_error
        )
        assert analytic.exact.Y.variance(level) == pytest.approx(
            variance, abs=3 * standard_error
        )
