import pytest

from mlqmcpy import AnalyticMLMCIterator, GreedyMLMCIterator, GreedyMLQMCIterator
from mlqmcpy.problems.analytic import analytic


@pytest.mark.parametrize(
    "solver", [AnalyticMLMCIterator, GreedyMLMCIterator, GreedyMLQMCIterator]
)
def test_greedy_mc(solver):

    # Define greedy MC iterator
    mc_level = 4
    error_tolerance = 1e-2
    seed = 1234
    iterator = solver(2, error_tolerance=error_tolerance, seed=seed)

    # Take samples
    for new_samples in iterator:
        new_results = {
            level: analytic(level + mc_level, samples)
            for level, samples in new_samples.items()
        }
        iterator.update(new_results)

    # Test accuracy
    assert iterator.standard_error <= error_tolerance
    assert analytic.exact.Q.mean(mc_level) == pytest.approx(
        iterator.mean, abs=3 * error_tolerance
    )

    # Get budget
    max_budget = iterator.cost

    # Define greedy MC iterator with same budget
    iterator = solver(2, max_budget=max_budget, seed=seed)

    # Take samples
    for new_samples in iterator:
        new_results = {
            level: analytic(level + mc_level, samples)
            for level, samples in new_samples.items()
        }
        iterator.update(new_results)

    # Test accuracy
    assert iterator.standard_error <= error_tolerance
    assert analytic.exact.Q.mean(mc_level) == pytest.approx(
        iterator.mean, abs=error_tolerance
    )


@pytest.mark.parametrize(
    "solver", [AnalyticMLMCIterator, GreedyMLMCIterator, GreedyMLQMCIterator]
)
def test_greedy_mlmc(solver):

    # Define greedy MC iterator
    num_levels = 5
    cost_per_level = [2**level for level in range(num_levels)]
    error_tolerance = 1e-2
    seed = 12
    iterator = solver(
        2,
        cost_per_level=cost_per_level,
        error_tolerance=error_tolerance,
        seed=seed,
    )

    # Take samples
    for new_samples in iterator:
        new_results = {
            level: analytic.ml(level, samples) for level, samples in new_samples.items()
        }
        iterator.update(new_results)

    # Test accuracy
    assert iterator.standard_error <= error_tolerance
    assert analytic.exact.Q.mean(num_levels - 1) == pytest.approx(
        iterator.mean, abs=3 * error_tolerance
    )

    # Get budget
    max_budget = iterator.cost

    # Define greedy MC iterator with same budget
    iterator = solver(
        2, cost_per_level=cost_per_level, max_budget=max_budget, seed=seed
    )

    # Take samples
    for new_samples in iterator:
        new_results = {
            level: analytic.ml(level, samples) for level, samples in new_samples.items()
        }
        iterator.update(new_results)

    # Test accuracy
    assert iterator.standard_error <= error_tolerance
    assert analytic.exact.Q.mean(num_levels - 1) == pytest.approx(
        iterator.mean, abs=error_tolerance
    )
