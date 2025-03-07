from abc import ABC

import numpy as np
from prettytable import PrettyTable

from .factories import (
    AbstractMultilevelFactory,
    AnalyticMLMCFactory,
    GreedyFastGaussianProcessMLQMCFactory,
    GreedyMLMCFactory,
    GreedyMLQMCFactory,
)
from .stopping_criteria import AccuracyConstrained, BudgetConstrained


class AbstractMultilevelIterator(ABC):
    """
    Base class for multilevel iterators.

    Manages sample generation, response accumulation, and stopping criteria.
    """

    def __init__(
        self,
        dimension,
        cost_per_level=1,
        initial_sample_size=8,
        max_budget=None,
        error_tolerance=None,
        seed=None,
        replications=8,
        discrete_distribution_type=None,
        factory: AbstractMultilevelFactory = None,
    ):
        if factory is None:
            raise ValueError("A factory instance must be provided.")
        self._factory = factory

        # Determine number of levels.
        cost_per_level = np.atleast_1d(cost_per_level)
        self._num_levels = len(cost_per_level)

        # Dimension per level.
        if isinstance(dimension, int):
            dimension_list = np.full(len(self), dimension, dtype=int)
        else:
            dimension_list = np.array(dimension, dtype=int)

        # Seed per level.
        if seed is None or isinstance(seed,int):
            seed = np.random.SeedSequence(seed)
        else:
            assert isinstance(seed,np.random.SeedSequence), "require seed is None, an int, or a np.random.SeedSequence"
        seeds = seed.spawn(self._num_levels)

        # Replications per level.
        if isinstance(replications, int):
            replications_list = np.full(len(self), replications, dtype=int)
        else:
            replications_list = np.array(replications, dtype=int)

        # Initialize response accumulators.
        self._response_accumulators = [
            self._factory.create_response_accumulator(
                cost_per_level[level], replications_list[level]
            )
            for level in range(len(self))
        ]

        # Initialize point generators.
        self._point_generators = [
            self._factory.create_point_generator(
                discrete_distribution_type,
                dimension_list[level],
                seeds[level],
                replications_list[level],
            )
            for level in range(len(self))
        ]

        # Define stopping criterion.
        if max_budget:
            self._stopping_criterion = BudgetConstrained(max_budget)
        elif error_tolerance:
            self._stopping_criterion = AccuracyConstrained(error_tolerance)
        else:
            raise ValueError(
                "You must specify either a maximum budget 'max_budget' or an error tolerance 'error_tolerance'."
            )

        # Set initial sample size.
        if isinstance(initial_sample_size, int):
            initial_sample_size_list = np.full(
                len(self), initial_sample_size, dtype=int
            )
        else:
            initial_sample_size_list = np.array(initial_sample_size, dtype=int)

        # Set sample allocation solver.
        self._sample_allocation_solver = factory.create_sample_allocation_solver(
            initial_sample_size_list
        )

    def __len__(self):
        """Return the number of levels."""
        return self._num_levels

    def __iter__(self):
        return self

    def __next__(self):
        """Generate new samples at each level; stop if converged."""
        if self.is_converged:
            raise StopIteration

        new_sample_sizes = self._sample_allocation_solver.step(
            self._response_accumulators, self._stopping_criterion
        )
        return {
            level: AbstractMultilevelIterator._concatenate(
                self._point_generators[level].gen_samples(number_of_samples)
            )
            for level, number_of_samples in new_sample_sizes.items()
        }

    @staticmethod
    def _concatenate(points):
        """Concatenate point arrays along axis 0."""
        return np.concatenate(points, axis=0) if points.ndim == 3 else points

    def update(self, all_responses):
        """Update accumulators with new responses."""
        for level, responses in all_responses.items():
            self._response_accumulators[level].add_responses(responses)

    @property
    def is_converged(self):
        """Return True if the stopping criterion is met."""
        return self._stopping_criterion.is_achieved(self._response_accumulators)

    @property
    def mean(self):
        """Return the total mean across levels."""
        return sum(accumulator.mean for accumulator in self._response_accumulators)

    @property
    def standard_error(self):
        """Return the combined standard error across levels."""
        return np.sqrt(
            sum(
                accumulator.squared_standard_error
                for accumulator in self._response_accumulators
            )
        )

    @property
    def cost(self):
        """Return the total cost across levels."""
        return sum(
            len(accumulator) * accumulator.cost
            for accumulator in self._response_accumulators
        )

    def print_status(self):
        """Print status summary in a table format."""
        table = PrettyTable()
        table.field_names = ["number of samples", "mean", "variance", "standard error"]
        for accumulator in self._response_accumulators:
            table.add_row(
                [
                    accumulator.num_samples_str,
                    accumulator.mean,
                    accumulator.variance,
                    np.sqrt(accumulator.squared_standard_error),
                ]
            )
        table.float_format = ".10"
        table.align["number of samples"] = "r"
        print(table)


class GreedyMLMCIterator(AbstractMultilevelIterator):
    """Iterator for Greedy MLMC."""

    ALLOWED_KEYS = {
        "cost_per_level",
        "initial_sample_size",
        "max_budget",
        "error_tolerance",
        "seed",
    }

    def __init__(self, *args, **kwargs):
        for key in kwargs.keys():
            if key not in self.ALLOWED_KEYS:
                raise ValueError(f"Invalid keyword argument provided: '{key}'")
        factory = GreedyMLMCFactory()
        super().__init__(*args, factory=factory, **kwargs)


class AnalyticMLMCIterator(AbstractMultilevelIterator):
    """Iterator for Analytic MLMC."""

    ALLOWED_KEYS = {
        "cost_per_level",
        "initial_sample_size",
        "max_budget",
        "error_tolerance",
        "seed",
    }

    def __init__(self, *args, **kwargs):
        for key in kwargs.keys():
            if key not in self.ALLOWED_KEYS:
                raise ValueError(f"Invalid keyword argument provided: '{key}'")
        factory = AnalyticMLMCFactory()
        super().__init__(*args, factory=factory, **kwargs)


class GreedyMLQMCIterator(AbstractMultilevelIterator):
    """Iterator for Greedy MLQMC using replications."""

    ALLOWED_KEYS = {
        "cost_per_level",
        "initial_sample_size",
        "max_budget",
        "error_tolerance",
        "seed",
        "replications",
        "discrete_distribution_type",
    }

    def __init__(self, *args, **kwargs):
        for key in kwargs.keys():
            if key not in self.ALLOWED_KEYS:
                raise ValueError(f"Invalid keyword argument provided: '{key}'")
        factory = GreedyMLQMCFactory()
        super().__init__(*args, factory=factory, **kwargs)


class FastGaussianProcessMLQMCIterator(AbstractMultilevelIterator):
    """Iterator for Fast Gaussian Process MLQMC using replications."""

    ALLOWED_KEYS = {
        "cost_per_level",
        "initial_sample_size",
        "max_budget",
        "error_tolerance",
        "seed",
        "discrete_distribution_type",
    }

    def __init__(self, *args, **kwargs):
        for key in kwargs.keys():
            if key not in self.ALLOWED_KEYS:
                raise ValueError(f"Invalid keyword argument provided: '{key}'")
        factory = GreedyFastGaussianProcessMLQMCFactory()
        super().__init__(*args, factory=factory, **kwargs)
