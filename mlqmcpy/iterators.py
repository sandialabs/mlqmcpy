from abc import ABC

import numpy as np
from prettytable import PrettyTable

from .factories import (
    AbstractMultilevelFactory,
    AnalyticMLMCFactory,
    GreedyGaussianProcessMLQMCFactory,
    GreedyMLMCFactory,
    GreedyMLQMCFactory,
)
from .stopping_criteria import AccuracyConstrained, BudgetConstrained
from copy import deepcopy

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

        # Determine number of levels
        cost_per_level = np.atleast_1d(cost_per_level)
        self._num_levels = len(cost_per_level)

        # Dimension per level
        if isinstance(dimension, int):
            dimension_list = np.full(len(self), dimension, dtype=int)
        else:
            dimension_list = np.array(dimension, dtype=int)

        # Seed per level
        if seed is None or isinstance(seed, int):
            seed = np.random.SeedSequence(seed)
        else:
            assert isinstance(
                seed, np.random.SeedSequence
            ), "require seed is None, an int, or a np.random.SeedSequence"
        seeds = seed.spawn(self._num_levels)

        # Replications per level
        if isinstance(replications, int):
            replications_list = np.full(len(self), replications, dtype=int)
        else:
            replications_list = np.array(replications, dtype=int)

        # Initialize point generators

        self.discrete_distribution_type = discrete_distribution_type

        self._point_generators = [
            self._factory.create_point_generator(
                self.discrete_distribution_type,
                dimension_list[level],
                seeds[level],
                replications_list[level],
            )
            for level in range(len(self))
        ]

        # Initialize response accumulators
        # NOTE: for FastGP, must initialize response accumulators AFTER point generators
        self._response_accumulators = [
            self._factory.create_response_accumulator(
                cost_per_level[level], replications_list[level]
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
                self._point_generators[level].generate(number_of_samples)
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

    @property
    def total_samples_per_level(self):
        return np.array(
            [len(accumulator) for accumulator in self._response_accumulators], dtype=int
        )

    def set_max_budget(self, max_budget):
        self._stopping_criterion = BudgetConstrained(max_budget)

    def set_error_tolerance(self, error_tolerance):
        self._stopping_criterion = AccuracyConstrained(error_tolerance)

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


class GreedyGaussianProcessMLQMCIterator(AbstractMultilevelIterator):
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
        if "kwargs_fastgp_construct" in kwargs:
            kwargs_fastgp_construct = kwargs["kwargs_fastgp_construct"]
            del kwargs["kwargs_fastgp_construct"]
        else:
            kwargs_fastgp_construct = {}

        if "kwargs_discrete_distrib_construct" in kwargs:
            kwargs_discrete_distrib_construct = kwargs["kwargs_discrete_distrib_construct"]
            del kwargs["kwargs_discrete_distrib_construct"]
        else:
            kwargs_discrete_distrib_construct = {}

        if "kernel_class" in kwargs:
            kernel_class = kwargs["kernel_class"]
            del kwargs["kernel_class"]
        else:
            kernel_class = None
        
        if "kwargs_kernel_construct" in kwargs:
            kwargs_kernel_construct = kwargs["kwargs_kernel_construct"]
            del kwargs["kwargs_kernel_construct"]
        else:
            kwargs_kernel_construct = {}

        if "kwargs_fastgp_fit" in kwargs:
            kwargs_fastgp_fit = kwargs["kwargs_fastgp_fit"]
            del kwargs["kwargs_fastgp_fit"]
        else:
            kwargs_fastgp_fit = {}

        if "fast" in kwargs:
            fast = kwargs["fast"]
            del kwargs["fast"]
        else:
            fast = True

        if "refit_gps" in kwargs:
            refit_gps = kwargs["refit_gps"]
            del kwargs["refit_gps"]
        else:
            refit_gps = True

        for key in kwargs.keys():
            if key not in self.ALLOWED_KEYS:
                raise ValueError(f"Invalid keyword argument provided: '{key}'")

        factory = GreedyGaussianProcessMLQMCFactory(
            kwargs_fastgp_construct=kwargs_fastgp_construct,
            kwargs_discrete_distrib_construct=kwargs_discrete_distrib_construct,
            kernel_class=kernel_class,
            kwargs_kernel_construct=kwargs_kernel_construct,
            kwargs_fastgp_fit=kwargs_fastgp_fit,
            fast=fast,
            refit_gps=refit_gps,
        )

        super().__init__(*args, factory=factory, **kwargs)


class AbstractMultiTaskGaussianProcessMLQMCIterator(AbstractMultilevelIterator):
    """Iterator for Fast MultiTask Gaussian Process MLQMC."""

    def __init__(
        self,
        dimension,
        cost_per_level=1,
        initial_sample_size=8,
        max_budget=None,
        # error_tolerance = None,
        seed=None,
        # replications = 8,
        discrete_distribution_type=None,
        # factory: AbstractMultilevelFactory = None,
        # budget_scheme = "full",
        budget_scheme="greedy",
        kwargs_fastgp_construct={},
        kwargs_discrete_distrib_construct = {},
        kernel_class = None,
        kwargs_kernel_construct = {},
        kwargs_kernel_mt_construct = {},
        kwargs_fastgp_fit={},
        fast=True,
        refit_gps=True,
    ):

        assert isinstance(
            dimension, int
        ), "FastMultiTaskGaussianProcessMLQMCIterator requires the dimension is the same for each level"

        cost_per_level = np.atleast_1d(cost_per_level)
        assert cost_per_level.ndim == 1, "np.atleast_1d(cost_per_level) should be 1d"
        self.cost_per_level = cost_per_level

        self._num_levels = len(cost_per_level)

        initial_sample_size = np.atleast_1d(initial_sample_size).astype(int)
        assert (
            initial_sample_size.ndim == 1 and (initial_sample_size > 0).all()
        ), "initial_sample_size must be positive ints"
        if initial_sample_size.size == 1:
            initial_sample_size = initial_sample_size * np.ones(
                self._num_levels, dtype=int
            )
        assert initial_sample_size.shape == (self._num_levels,), (
            "initial_sample_size should have shape (%d,)" % self._num_levels
        )
        self.initial_sample_size = initial_sample_size

        assert (
            np.isscalar(max_budget) and max_budget >= 0
        ), "max_budget must be a positive scalar"
        self.max_budget = max_budget

        assert seed is None or isinstance(seed, int), "seed must be None or an int"

        import fastgps
        import qmcpy as qp

        if discrete_distribution_type is None:
            discrete_distribution_type == qp.DigitalNetB2

        if fast and discrete_distribution_type == qp.Lattice:
            GPClass = fastgps.FastGPLattice
            KernelClass = qp.KernelShiftInvar if kernel_class is None else kernel_class
        elif fast and discrete_distribution_type == qp.DigitalNetB2:
            GPClass = fastgps.FastGPDigitalNetB2
            KernelClass = qp.KernelDigShiftInvar if kernel_class is None else kernel_class
            if "randomize" not in kwargs_discrete_distrib_construct:
                kwargs_discrete_distrib_construct = deepcopy(kwargs_discrete_distrib_construct)
                kwargs_discrete_distrib_construct["randomize"] = "DS"
        else:
            GPClass = fastgps.StandardGP
            KernelClass = qp.KernelSquaredExponential if kernel_class is None else kernel_class

        self.discrete_distribution_type = discrete_distribution_type

        import torch
        torch.set_default_dtype(torch.float64)

        discrete_distribs = np.array([discrete_distribution_type(dimension,seed=seed,**kwargs_discrete_distrib_construct) for seed in np.random.SeedSequence(seed).spawn(self._num_levels)],dtype=object)
        kernel = KernelClass(d=dimension,torchify=True,**kwargs_kernel_construct)
        kernel_mt = qp.KernelMultiTask(kernel,num_tasks=self._num_levels,**kwargs_kernel_mt_construct)
        self.fgp = GPClass(kernel_mt,discrete_distribs,**kwargs_fastgp_construct)

        self.refit_gps = refit_gps

        self.iteration = 0

        assert budget_scheme.lower() in ["full", "greedy"]
        self.budget_scheme = budget_scheme.lower()
        self.max_iterations = 1 if self.budget_scheme == "full" else np.inf

        self.kwargs_fastgp_fit = kwargs_fastgp_fit

        self.weights = self.get_weights()

    def _get_next_n(self, future_budget):
        import torch

        max_budget = self.cost + future_budget

        n0 = self.fgp.n.cpu().numpy()
        m0 = self.fgp.m.cpu().numpy()

        nnew_max = np.floor(future_budget / self.cost_per_level).astype(int)
        m_max = np.floor(np.log2(n0 + nnew_max)).astype(int)
        n_max = 2**m_max
        if (n_max <= n0).all():
            raise StopIteration

        m_feasible = [
            np.arange(m0[level], m_max[level] + 1) for level in range(self._num_levels)
        ]
        # num_m_feasible = torch.tensor([len(m_feasible[l]) for l in range(self._num_levels)])
        # num_feasible_combs = num_m_feasible.prod()

        mfmesh = np.meshgrid(*m_feasible)
        mf_comb = np.vstack([mfmesh_l.flatten() for mfmesh_l in mfmesh]).T
        nf_comb = 2**mf_comb
        nf_comb_cost = (nf_comb * self.cost_per_level).sum(1)
        cheap_comb = nf_comb_cost <= max_budget
        mf_comb_cheap = mf_comb[cheap_comb]
        # mf_comb_expensive = mf_comb[~cheap_comb]

        boundary = (
            (
                (2 ** (mf_comb_cheap[:, None, :] + np.eye(self._num_levels, dtype=int)))
                * self.cost_per_level
            ).sum(-1)
            > max_budget
        ).all(-1)
        # num_boundary = boundary.sum()
        mf_comb_cheap_boundary = mf_comb_cheap[boundary]
        nf_comb_cheap_boundary = 2**mf_comb_cheap_boundary

        pcvars = np.array([self.projected_post_cubature_var(nf_comb_cheap_boundary[i]) for i in range(len(nf_comb_cheap_boundary))])
        imin = pcvars.argmin()
        # pcvar_min = pcvars[imin]
        n = nf_comb_cheap_boundary[imin]
        return n
    
    def __next__(self):
        """Generate new samples at each level; stop if converged."""

        import torch

        if self.iteration > self.max_iterations or self.cost >= self.max_budget:
            raise StopIteration

        if self.iteration == 0:
            n = self.initial_sample_size

        elif self.budget_scheme == "full":
            remaining_budget = self.max_budget - self.cost
            n = self._get_next_n(remaining_budget)

        else:  # self.budget_scheme=="greedy"
            remaining_budget = self.max_budget - self.cost
            n_curr = self.fgp.n.cpu().numpy()
            future_budgets = n_curr * self.cost_per_level
            if (future_budgets < remaining_budget).all():
                future_budget = future_budgets.max()
            else:
                if (future_budgets <= remaining_budget).any():
                    future_budget = future_budgets[
                        future_budgets <= remaining_budget
                    ].max()
                else:  # (future_budgets>remaining_budget).any()
                    future_budget = future_budgets.min()
            n = self._get_next_n(future_budget)

        x_next = self.fgp.get_x_next(n=torch.from_numpy(n).to(self.fgp.device))
        samples_dict = {}
        for level in range(self._num_levels):
            x_next_level = x_next[level].cpu().numpy()
            if x_next_level.size > 0:
                samples_dict[level] = x_next_level

        self.iteration += 1

        return samples_dict

    def update(self, all_responses):
        """Update accumulators with new responses."""

        import torch

        tasks = list(all_responses.keys())
        y_next = [
            torch.tensor(all_responses[task]).to(self.fgp.device) for task in tasks
        ]
        self.fgp.add_y_next(y_next, torch.tensor(tasks).to(self.fgp.device))
        if self.iteration == 1 or self.refit_gps:
            # data = self.fgp.fit(**self.kwargs_fastgp_fit)
            self.fgp.fit(**self.kwargs_fastgp_fit)
        self.post_cubature_means = self.fgp.post_cubature_mean().numpy()
        self.post_cubature_cov = self.fgp.post_cubature_cov().numpy()
        self.optimal_weights = (self.weights*self.post_cubature_means).sum()*np.linalg.lstsq(self.post_cubature_cov+self.post_cubature_means[:,None]*self.post_cubature_means[None,:],self.post_cubature_means)[0]
        self.optimal_mean_pred = (self.optimal_weights*self.post_cubature_means).sum()
        self.optimal_standard_error = np.sqrt(max(self.optimal_weights@self.post_cubature_cov@self.optimal_weights,0))
    @property
    def mean(self):
        """Return the total mean across levels."""
        return self.optimal_mean_pred

    @property
    def standard_error(self):
        """Return the combined standard error across levels."""
        return self.optimal_standard_error

    def projected_post_cubature_var(self, n):
        assert n.shape==(self.fgp.num_tasks,)
        import torch 
        projected_cov = self.fgp.post_cubature_cov(n=torch.from_numpy(n).to(self.fgp.device)).numpy()
        optimal_projected_cov = self.optimal_weights@projected_cov@self.optimal_weights
        return optimal_projected_cov
    
    @property
    def cost(self):
        """Return the total cost across levels."""
        return (self.fgp.n.cpu().numpy() * self.cost_per_level).sum().item()
    
    @property
    def total_samples_per_level(self):
        return self.fgp.n.numpy().astype(int)
    
    def print_status(self):
        """Print status summary in a table format."""
        table = PrettyTable()
        table.field_names = ["number of samples", "mean", "variance", "standard error"]
        print("\t number of samples: %s"%self.total_samples_per_level)
        print("\t              mean: %s"%self.mean)
        print("\t    standard error: %s"%self.standard_error)

class MultiTaskGaussianProcessMLQMCIteratorFunction(AbstractMultiTaskGaussianProcessMLQMCIterator):
    
    def get_weights(self):
        return (np.arange(1,self._num_levels+1)==self._num_levels).astype(float)

class MultiTaskGaussianProcessMLQMCIteratorDifference(AbstractMultiTaskGaussianProcessMLQMCIterator):

    def get_weights(self):
        return np.ones(self._num_levels)