import numpy as np

# from boost_histogram.accumulators import Mean


class Accumulator:
    """
    Wraps a boost_histogram Mean accumulator.
    """

    def __init__(self):
        # self._accumulator = Mean()
        self._accumulator = np.empty(0)  # Mean()

    def add(self, sample):
        """Add a sample."""
        # self._accumulator(sample)
        self._accumulator = np.append(self._accumulator, sample)

    @property
    def mean(self):
        """Return the current mean or NaN if no samples."""
        # return self._accumulator.value if len(self) else np.nan
        return self._accumulator.mean() if len(self) else np.nan

    @property
    def variance(self):
        """Return the current variance or NaN if no samples."""
        # return self._accumulator.variance if len(self) else np.nan
        return self._accumulator.var() if len(self) else np.nan

    def __len__(self):
        """Return the number of samples."""
        # return int(self._accumulator.count)
        return int(len(self._accumulator))

    @property
    def squared_standard_error(self):
        """Return the squared standard error or NaN if no samples."""
        return self.variance / len(self) if len(self) else np.nan

    # def reset(self):
    #     """Reset the accumulator."""
    #     self._accumulator = Mean()


class ResponseAccumulator:
    """
    Accumulates responses with an associated cost.
    """

    def __init__(self, cost=1):
        self._cost = cost
        self._accumulator = Accumulator()

    def add_responses(self, responses):
        """Add multiple responses."""
        for response in responses:
            self._accumulator.add(response)

    @property
    def cost(self):
        """Return the cost."""
        return self._cost

    @property
    def mean(self):
        """Return the mean of responses."""
        return self._accumulator.mean

    @property
    def variance(self):
        """Return the variance of responses."""
        return self._accumulator.variance

    @property
    def squared_standard_error(self):
        """Return the squared standard error."""
        return self._accumulator.squared_standard_error

    def __len__(self):
        """Return the number of responses."""
        return len(self._accumulator)

    @property
    def num_samples(self):
        """Return the number of samples."""
        return len(self)

    @property
    def num_samples_str(self):
        """Return a string representation of sample count."""
        return str(len(self))


class ReplicatedResponseAccumulator:
    """
    Accumulates responses using multiple replications.
    """

    def __init__(self, num_replications, cost=1):
        self.num_replications = num_replications
        self._cost = cost
        self._accumulators = [Accumulator() for _ in range(self.num_replications)]

    def add_responses(self, responses):
        """Split responses among replications and add to each accumulator."""
        responses_split = np.array_split(responses, self.num_replications)
        for accumulator, responses in zip(self._accumulators, responses_split):
            for response in responses:
                accumulator.add(response)

    @property
    def cost(self):
        """Return the cost."""
        return self._cost

    @property
    def mean(self):
        """Return the mean over replications."""
        return np.mean([accumulator.mean for accumulator in self._accumulators])

    @property
    def variance(self):
        """Return the mean variance over replications."""
        return np.mean([accumulator.variance for accumulator in self._accumulators])

    @property
    def squared_standard_error(self):
        """Return the squared standard error across replications."""
        return (
            np.var([accumulator.mean for accumulator in self._accumulators])
            / self.num_replications
        )

    def __len__(self):
        """Return the total number of responses across all replications."""
        return np.sum([len(accumulator) for accumulator in self._accumulators])

    @property
    def num_samples(self):
        """Return the number of samples per replication."""
        return int(len(self) / self.num_replications)

    @property
    def num_samples_str(self):
        """Return a string representation of replications and samples per replication."""
        return str(self.num_replications) + " x " + str(self.num_samples)


class GaussianProcessResponseAccumulator(ResponseAccumulator):
    """
    Accumulates responses using a Fast Gaussian Process fit
    """

    def __init__(self, fgp, cost, kwargs_fastgp_fit, refit_gps):
        # self._cost = cost
        super().__init__(cost)
        self._fgp = fgp
        self.kwargs_fastgp_fit = kwargs_fastgp_fit
        self.refit_gps = refit_gps
        # self.n = 0

    def add_responses(self, responses):
        """Add multiple responses."""
        import torch

        for response in responses:
            self._accumulator.add(response)
        self._fgp.add_y_next(torch.tensor(responses).to(self._fgp.device))
        if (
            self._fgp.n.item() == len(responses) or self.refit_gps
        ):  # either the first iteration or we are forced to refit GPs
            self._fgp.fit(**self.kwargs_fastgp_fit)
        # self.n += len(responses)
        # print("self.n is now", self.n)

    # @property
    # def cost(self):
    #     """Return the cost."""
    #     return self._cost

    @property
    def mean(self):
        """Return the mean of responses."""
        return self._fgp.post_cubature_mean().cpu().numpy() if len(self) else np.nan

    @property
    def variance(self):
        """Return the variance of responses."""
        # raise NotImplementedError()
        return np.nan

    @property
    def squared_standard_error(self):
        """Return the squared standard error."""
        return self._fgp.post_cubature_var().cpu().numpy() if len(self) else np.nan
        # if not len(self):
        #     return np.nan
        # else:
        #     _, _, _, lb, ub = self._fgp.post_cubature_ci(confidence=0.68)
        # return (ub.numpy() - lb.numpy()) / 2

    # def __len__(self):
    #     """Return the number of responses."""
    #     return self.n

    # @property
    # def num_samples(self):
    #     """Return the number of samples."""
    #     return len(self)

    # @property
    # def num_samples_str(self):
    #     """Return a string representation of sample count."""
    #     return str(len(self))
