from abc import ABC, abstractmethod

import numpy as np

from .stopping_criteria import AccuracyConstrained, BudgetConstrained


class AbstractSampleAllocationProblemSolver(ABC):
    """
    Abstract base class for sample allocation solvers.

    Uses an initial sample size list if no samples have been collected.
    """

    def __init__(self, initial_sample_size_list):
        self._initial_sample_size_list = initial_sample_size_list

    def step(self, accumulators, stopping_criterion):
        """
        Determine new sample sizes based on current accumulators and stopping criterion.
        Returns the initial sample sizes if no samples have been collected.
        """
        if any(len(accumulator) for accumulator in accumulators):
            return self._step(accumulators, stopping_criterion)
        else:
            return {
                level: self._initial_sample_size_list[level]
                for level in range(len(accumulators))
            }

    @abstractmethod
    def _step(self, accumulators, stopping_criterion):
        """Compute new sample sizes given current accumulators and stopping criterion."""
        pass


class GreedySampleAllocationProblemSolver(AbstractSampleAllocationProblemSolver):
    """
    Greedy solver that allocates additional samples to the level with the highest cost-normalized squared standard error.
    """

    def _step(self, accumulators, stopping_criterion):
        # print("errors:", [
        #         accumulator.squared_standard_error
        #         for accumulator in accumulators
        #     ])
        # print("costs:", [
        #         accumulator.cost
        #         for accumulator in accumulators
        #     ])
        # print("ratios:", [
        #         accumulator.squared_standard_error / accumulator.cost
        #         for accumulator in accumulators
        #     ])

        levels,sses,costs,n,ntotal = np.array([
                [l, accumulator.squared_standard_error, accumulator.cost, accumulator.num_samples, len(accumulator)]
                for l,accumulator in enumerate(accumulators)
            ]
        ).T

        levels = levels.astype(int) 
        n = n.astype(int) 
        ntotal = ntotal.astype(int) 

        current_costs = costs * ntotal
        ratios = sses/current_costs

        if isinstance(stopping_criterion, BudgetConstrained):
            total_current_cost = np.sum(current_costs)
            remaining_budget = stopping_criterion.max_budget - total_current_cost
            feasible = current_costs <= remaining_budget # the current cost is also the next cost as we double the sample size on each level                                       
            if not np.any(feasible):
                raise StopIteration
            ratios = ratios[feasible]
            levels = levels[feasible] 
            n = n[feasible]
                                           
        idx = np.argmax(ratios)

        new_sample_sizes = {
            int(levels[idx]): int(n[idx])
        }
        return new_sample_sizes


class AnalyticSampleAllocationProblemSolver(AbstractSampleAllocationProblemSolver):
    """
    Analytic solver that computes new sample sizes based on an analytic formula
    using variance-cost ratios and either an error tolerance or a maximum budget.
    """

    def _step(self, accumulators, stopping_criterion):
        num_levels = len(accumulators)
        sum_var_cost = sum(
            accumulator.variance * accumulator.cost for accumulator in accumulators
        )
        var_cost_ratio = [
            np.sqrt(accumulator.variance / accumulator.cost)
            for accumulator in accumulators
        ]

        if isinstance(stopping_criterion, AccuracyConstrained):
            return {
                level: int(
                    np.ceil(
                        sum_var_cost
                        / stopping_criterion.error_tolerance**2
                        * var_cost_ratio[level]
                    )
                )
                for level in range(num_levels)
            }
        elif isinstance(stopping_criterion, BudgetConstrained):
            return {
                level: int(
                    np.ceil(
                        stopping_criterion.max_budget
                        / sum_var_cost
                        * var_cost_ratio[level]
                    )
                )
                for level in range(num_levels)
            }
        else:
            raise NotImplementedError(
                f"Analytic solver not implemented for stopping criterion of type '{type(stopping_criterion)}'"
            )
