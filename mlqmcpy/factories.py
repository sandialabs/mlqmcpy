from abc import ABC, abstractmethod

import torch
from fastgp import FastGPDigitalNetB2, FastGPLattice
from qmcpy import DigitalNetB2, DiscreteDistribution, Lattice

from .accumulators import (
    FastGaussianProcessResponseAccumulator,
    ReplicatedResponseAccumulator,
    ResponseAccumulator,
)
from .point_generators import (
    FastGaussianProcessPointGenerator,
    IIDPointGenerator,
    LDPointGenerator,
)
from .solvers import (
    AnalyticSampleAllocationProblemSolver,
    GreedySampleAllocationProblemSolver,
)

torch.set_default_dtype(torch.float64)


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
        return IIDPointGenerator(dimension, seed)

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
        return IIDPointGenerator(dimension, seed)

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
        return LDPointGenerator(
            discrete_distribution_type, dimension, seed, replications
        )

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)


class GreedyFastGaussianProcessMLQMCFactory(AbstractMultilevelFactory):
    """Concrete factory for a Greedy MLQMC iterator using Fast GPs."""

    def __init__(self):
        self.fgp_list = []
        self.fgp_counter = 0

    def create_response_accumulator(self, cost: float, replications: int):
        accumulator = FastGaussianProcessResponseAccumulator(
            self.fgp_list[self.fgp_counter], cost
        )
        self.fgp_counter += 1
        return accumulator

    def create_point_generator(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        if discrete_distribution_type == Lattice:
            self.fgp_list.append(
                FastGPLattice(
                    seq=Lattice(dimension=dimension, seed=seed),
                )
            )
        else:
            self.fgp_list.append(
                FastGPDigitalNetB2(
                    seq=DigitalNetB2(dimension=dimension, seed=seed),
                )
            )
        return FastGaussianProcessPointGenerator(self.fgp_list[-1])

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)
