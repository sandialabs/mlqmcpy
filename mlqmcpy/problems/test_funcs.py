import numpy as np

from .utils import multilevel
import scipy.stats 
import numpy as np 
from scipy.stats import norm,johnsonsu
from scipy.special import erf,erfc
import types 

"""
https://inria.hal.science/hal-04088085/document
https://par.nsf.gov/servlets/purl/10513787
https://arxiv.org/pdf/2504.18677
"""

class Abstract0MeanSLTF(object):
    def __init__(self):
        self.exact = types.SimpleNamespace()
        self.exact.Q = types.SimpleNamespace()
        self.exact.Q.mean = lambda level: 0.
    def __call__(self, level, samples=None):
        assert level==0, "function is single level"
        if samples is None:
            samples = np.random.rand(1)
        d = samples.shape[-1]
        y = self.f(samples,d)
        return y
    def ml(self, level, samples=None):
        assert level==0, "function is single level"
        return self.__call__(level=level,samples=samples)

class SumUeU(Abstract0MeanSLTF):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = SumUeU()
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 1.36859343, -0.03305734,  0.49776613, -0.12781599,  0.51586839])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.0014598375114869312)
    """
    def f(self, u, d):
        return -d+(u*np.exp(u)).sum(-1)

class MC2(Abstract0MeanSLTF):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = MC2()
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([-0.32612189, -0.03426478, -0.1489903 ,  0.04503848, -0.18090316])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(0.00031402440713339746)
    """
    def f(self, u, d):
        return -1+(d-1/2)**(-d)*(d-u).prod(-1)

class AbstractRidgeFunc(Abstract0MeanSLTF):
    def __init__(self, weights):
        self.weights = weights.upper()
    def f(self, x, d):
        if self.weights=="EQUAL":
            theta = d**(-1/2)
        elif self.weights=="SPARSE":
            eta = 2.**(-np.arange(1,d+1))
            theta = eta/np.linalg.norm(eta)
        else:
            raise Exception("invalid weights %s"%self.weights)
        v = (theta*scipy.stats.norm.ppf(x)).sum(-1)
        y = self.g(v) 
        return y
    
class RidgeJump(AbstractRidgeFunc):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = RidgeJump(weights="EQUAL")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.84134475, -0.15865525, -0.15865525, -0.15865525, -0.15865525])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.0006018557991328786)
    >>> f = RidgeJump(weights="SPARSE")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([-0.15865525, -0.15865525, -0.15865525,  0.84134475, -0.15865525])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(0.000182064488953059)
    """
    def g(self, v):
        tau = 1
        return -norm.cdf(-tau)+1.*(v>=tau)
    
class RidgePL(AbstractRidgeFunc):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = RidgePL(weights="EQUAL")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.03725466, -0.08331547, -0.08331547, -0.08331547, -0.08331547])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.0004054826452037408)
    >>> f = RidgePL(weights="SPARSE")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([-0.08331547, -0.08331547, -0.08331547,  0.38834334, -0.08331547])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(7.69645859731687e-05)
    """
    def g(self, v):
        tau = 1
        return np.maximum(v-tau,0)-norm.pdf(tau)+tau*norm.cdf(-tau)

class RidgeKink(AbstractRidgeFunc):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = RidgeKink(weights="EQUAL")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.35827492,  0.02564373,  0.1709422 , -0.36110201,  0.20189206])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.00027797502290289233)
    >>> f = RidgeKink(weights="SPARSE")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.05548968, -0.44172236, -0.21192092,  0.35827492, -0.10299155])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(7.722452957113127e-05)
    """
    def g(self, v):
        C = (np.exp(2)*np.sqrt(np.pi)*(2*erf(1/np.sqrt(2))+2*erf(np.sqrt(2))+3*erfc(1/np.sqrt(2)))+np.sqrt(2)-np.sqrt(2)*np.exp(3/2))/(6*np.exp(2)*np.sqrt(np.pi))
        return (np.minimum(np.maximum(-2,v),1)+2)/3-C

class RidgeFinance(AbstractRidgeFunc):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = RidgeFinance(weights="EQUAL")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.20595727,  0.03017954,  0.1034056 , -0.21853235,  0.11813299])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.00023984816198385452)
    >>> f = RidgeFinance(weights="SPARSE")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.04582639, -0.28999854, -0.10953801,  0.25431989, -0.04165012])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(5.940901363686472e-05)
    """
    def g(self, v):
        C = 0.6772995069448933065809810480429866200664279525563499321239796826208436941383270035261275255944734882 
        return np.minimum(1,np.sqrt(np.maximum(v+2,0))/2)-C
         
class RidgeSmooth(AbstractRidgeFunc):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = RidgeSmooth(weights="EQUAL")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.22277106,  0.08160396,  0.1645333 , -0.32307307,  0.176847  ])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.00021692810159278314)
    >>> f = RidgeSmooth(weights="SPARSE")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.10225531, -0.41566868, -0.14638281,  0.23302567, -0.02913101])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(1.597599954016813e-05)
    """
    def g(self, v):
        return -norm.cdf(1/np.sqrt(2))+norm.cdf(1+v)

class RidgeJSU(AbstractRidgeFunc):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> f = RidgeJSU(weights="EQUAL")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 2.05844167,  0.76562579,  1.34552657, -2.3321227 ,  1.45103025])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.0031911209684395755)
    >>> f = RidgeJSU(weights="SPARSE")
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(5,2)))
    array([ 0.89905201, -3.52860473, -0.73814641,  2.42692126,  0.06787512])
    >>> f(level=0,samples=rng.uniform(low=0,high=1,size=(2**20,2))).mean()
    np.float64(-0.0007203754542004417)
    """
    def __init__(self, weights):
        self.jsu = johnsonsu(a=1,b=1)
        self.jsu_mean = self.jsu.mean()
        super().__init__(weights=weights)
    def g(self, v):
        return -self.jsu_mean+self.jsu.ppf(norm.cdf(v))

