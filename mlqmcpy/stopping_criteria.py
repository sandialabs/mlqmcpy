from abc import ABC, abstractmethod


class AbstractStoppingCriterion(ABC):
    """
    Abstract base class for stopping criteria.
    """

    def __init__(self):
        pass

    @abstractmethod
    def is_achieved(self, accumulators):
        """
        Return True if the stopping criterion is met given the accumulators.
        """
        pass


class BudgetConstrained(AbstractStoppingCriterion):
    """
    Stops when the total cost reaches or exceeds the maximum budget.
    """

    def __init__(self, max_budget):
        self.max_budget = max_budget

    def is_achieved(self, accumulators):
        return (
            sum(len(accumulator) * accumulator.cost for accumulator in accumulators)
            >= self.max_budget
        )


class AccuracyConstrained(AbstractStoppingCriterion):
    """
    Stops when the total squared standard error is within the specified error tolerance.
    """

    def __init__(self, error_tolerance):
        self.error_tolerance = error_tolerance

    def is_achieved(self, accumulators):
        return (
            sum(accumulator.squared_standard_error for accumulator in accumulators)
            <= self.error_tolerance**2
        )
