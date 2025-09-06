from abc import ABC, abstractmethod

import fastgps
import qmcpy as qp

from .accumulators import (
    GaussianProcessResponseAccumulator,
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

class AbstractMultilevelFactory(ABC):
    """Abstract factory interface for creating multilevel MLMC components."""

    @abstractmethod
    def create_response_accumulator(self, cost: float, replications: int):
        """Return a response accumulator given cost and replications."""
        pass

    @abstractmethod
    def create_point_generator(
        self,
        discrete_distribution_type: qp.DiscreteDistribution,
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
        discrete_distribution_type: qp.DiscreteDistribution,
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
        discrete_distribution_type: qp.DiscreteDistribution,
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
        discrete_distribution_type: qp.DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        return LDPointGenerator(
            discrete_distribution_type, dimension, seed, replications
        )

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)


class GreedyGaussianProcessMLQMCFactory(AbstractMultilevelFactory):
    """Concrete factory for a Greedy MLQMC iterator using Fast GPs."""

    def __init__(self, kwargs_fastgp_construct, kwargs_discrete_distrib_construct, kernel_class, kwargs_kernel_construct, kwargs_fastgp_fit, fast, refit_gps):
        self.fgp_list = []
        self.fgp_counter = 0
        self.kwargs_fastgp_construct = kwargs_fastgp_construct
        self.kwargs_discrete_distrib_construct = kwargs_discrete_distrib_construct
        self.kernel_class = kernel_class
        self.kwargs_kernel_construct = kwargs_kernel_construct
        self.kwargs_fastgp_fit = kwargs_fastgp_fit
        self.fast = fast
        self.refit_gps = refit_gps

    def create_response_accumulator(self, cost: float, replications: int):
        accumulator = GaussianProcessResponseAccumulator(
            self.fgp_list[self.fgp_counter], cost, self.kwargs_fastgp_fit, self.refit_gps
        )
        self.fgp_counter += 1
        return accumulator

    def create_point_generator(
        self,
        discrete_distribution_type: qp.DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        import torch 
        torch.set_default_dtype(torch.float64)

        if self.fast and discrete_distribution_type==qp.Lattice:
            GPClass = fastgps.FastGPLattice
            KernelClass = qp.KernelShiftInvar if self.kernel_class is None else self.kernel_class
        elif self.fast and discrete_distribution_type==qp.DigitalNetB2:
            GPClass = fastgps.FastGPDigitalNetB2
            KernelClass = qp.KernelDigShiftInvar if self.kernel_class is None else self.kernel_class
        else:
            assert not self.fast, "GreedyGaussianProcessMLQMCIterator does not support fast=True when discrete_distribution_type not in [qp.Lattice, qp.DigitalNetB2]"
            GPClass = fastgps.StandardGP
            KernelClass = qp.KernelSquaredExponential if self.kernel_class is None else self.kernel_class
        
        discrete_distrib = discrete_distribution_type(dimension=dimension,seed=seed,**self.kwargs_discrete_distrib_construct)
        kernel = KernelClass(d=dimension,torchify=True,**self.kwargs_kernel_construct)
        fgp = GPClass(kernel,discrete_distrib,**self.kwargs_fastgp_construct)
        self.fgp_list.append(fgp)
        
        return FastGaussianProcessPointGenerator(self.fgp_list[-1])

    def create_sample_allocation_solver(self, initial_sample_size_list):
        return GreedySampleAllocationProblemSolver(initial_sample_size_list)
