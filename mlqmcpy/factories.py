from abc import ABC, abstractmethod

from qmcpy import DigitalNetB2, DiscreteDistribution, IIDStdUniform

from .accumulators import (
    FastGaussianProcessResponseAccumulator,
    ReplicatedResponseAccumulator,
    ResponseAccumulator,
)
from .solvers import (
    AnalyticSampleAllocationProblemSolver,
    GreedySampleAllocationProblemSolver,
)


class AbstractMultilevelFactory(ABC):
    """Abstract factory interface for creating multilevel MLMC components."""

    @abstractmethod
    def create_response_accumulator(self, cost: float, replications: int):
        """Return a response accumulator given cost and replications."""
        pass

    @abstractmethod
    def create_point_generator(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        """Return a point generator for the given parameters."""
        pass

    @abstractmethod
    def create_sample_allocation_solver(self, initial_sample_size_list):
        """Return a sample allocation solver based on the initial sample sizes."""
        pass


class GreedyMLMCFactory(AbstractMultilevelFactory):
    """Concrete factory for a Greedy MLMC iterator using IID samples."""

    def create_response_accumulator(self, cost: float, replications: int):
        return ResponseAccumulator(cost)

    def create_point_generator(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        return IIDStdUniform(dimension, seed=seed)

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)


class AnalyticMLMCFactory(AbstractMultilevelFactory):
    """Concrete factory for an Analytic MLMC iterator using IID samples."""

    def create_response_accumulator(self, cost: float, replications: int):
        return ResponseAccumulator(cost)

    def create_point_generator(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        return IIDStdUniform(dimension, seed=seed)

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return AnalyticSampleAllocationProblemSolver(initial_sample_size_list)


class GreedyMLQMCFactory(AbstractMultilevelFactory):
    """Concrete factory for a Greedy MLQMC iterator using QMC replications."""

    def create_response_accumulator(self, cost: float, replications: int):
        return ReplicatedResponseAccumulator(replications, cost)

    def create_point_generator(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        if discrete_distribution_type is None:
            discrete_distribution_type = DigitalNetB2
        return discrete_distribution_type(
            dimension, seed=seed, replications=replications
        )

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)


class GreedyFastGaussianProcessMLQMCFactory(AbstractMultilevelFactory):
    """Concrete factory for a Greedy MLQMC iterator using QMC replications."""

    def create_response_accumulator(self, cost: float, replications: int):
        return FastGaussianProcessResponseAccumulator(cost)

    def create_point_generator(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        if discrete_distribution_type is None:
            discrete_distribution_type = DigitalNetB2
        return discrete_distribution_type(dimension, seed=seed)

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)
