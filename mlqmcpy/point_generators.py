from abc import ABC, abstractmethod

from qmcpy import DigitalNetB2, DiscreteDistribution, IIDStdUniform


class AbstractPointGenerator(ABC):
    """
    Base class for point generators.

    Generates points in the unit cube.
    """

    def __init__(self):
        self.n = 0

    def generate(self, n: int):
        samples = self._generate(n)
        self.n += n
        return samples

    @abstractmethod
    def _generate(self, n: int):
        pass


class IIDPointGenerator(AbstractPointGenerator):
    """Concrete point set generator type for IID uniform samples."""

    def __init__(self, dimension: int, seed: int):
        super().__init__()
        self.discrete_distribution = IIDStdUniform(dimension, seed=seed)

    def _generate(self, n: int):
        return self.discrete_distribution.gen_samples(n)


class LDPointGenerator(AbstractPointGenerator):
    """Concrete point set generator type for quasi-random samples."""

    def __init__(
        self,
        discrete_distribution_type: DiscreteDistribution,
        dimension: int,
        seed: int,
        replications: int,
    ):
        super().__init__()
        if discrete_distribution_type is None:
            discrete_distribution_type = DigitalNetB2
        self.discrete_distribution = discrete_distribution_type(
            dimension, seed=seed, replications=replications
        )

    def _generate(self, n: int):
        return self.discrete_distribution.gen_samples(n_min=self.n, n_max=self.n + n)


class FastGaussianProcessPointGenerator(AbstractPointGenerator):
    """Concrete point set generator type for fast GP construction."""

    def __init__(self, fgp):
        super().__init__()
        self.fgp = fgp

    def _generate(self, n: int):
        return self.fgp.get_x_next(self.n + n)  # FastGP needs total number of points
