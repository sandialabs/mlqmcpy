import numpy as np

from .utils import multilevel
import scipy.stats 

class AbstractRidgeFunc(object):
    def v(self, x):
        return scipy.stats.norm.ppf(x).sum(-1)/np.sqrt(x.shape[-1])
    def __call__(self, level, samples=None):
        assert level==0, "ridge functions are single level"
        if samples is None:
            samples = np.random.rand(1)
        return self.g(self.v(samples))
    def ml(self, level, samples=None):
        assert level==0, "ridge functions are single level"
        return self.__call__(level=level,samples=samples)

class RidgeJump(AbstractRidgeFunc):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> rf = RidgeJump()
    >>> rf(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([1., 0., 0., 0., 0.])
    """
    def g(self, v):
        return 1.*(v>=1)

class RidgeKink(AbstractRidgeFunc):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> rf = RidgeKink()
    >>> rf(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([1.        , 0.66736881, 0.81266728, 0.28062307, 0.84361713])
    """
    def g(self, v):
        return (np.minimum(np.maximum(-2,v),1)+2)/3

class RidgeSmooth(AbstractRidgeFunc):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> rf = RidgeSmooth()
    >>> rf(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([0.983021  , 0.8418539 , 0.92478323, 0.43717687, 0.93709694])
    """
    def g(self, v):
        return scipy.stats.norm.cdf(v+1)

class RidgeFinance(AbstractRidgeFunc):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> rf = RidgeFinance()
    >>> rf(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([0.88325678, 0.70747905, 0.7807051 , 0.45876715, 0.79543249])
    """
    def g(self, v):
        return np.minimum(1,np.sqrt(np.maximum(v+2,0))/2)
         
