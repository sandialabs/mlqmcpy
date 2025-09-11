import qmcpy as qp 
import numpy as np 

class Genz(object):
    """
    >>> d = 10 
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> x = rng.uniform(low=0,high=1,size=(5,d))
    >>> for kind_func in ['OSCILLATORY','CORNER PEAK']:
    ...     for kind_coeff in [1,2,3]:
    ...         genz = Genz(d,kind_func=kind_func,kind_coeff=kind_coeff)
    ...         y = genz(level=0,samples=x)
    ...         print(y)
    [-0.81058746 -0.99970132 -0.63981095  0.08633115 -0.90016393]
    [-0.980256    0.07648516  0.45406144 -0.24082697 -0.36262924]
    [-0.98758037  0.22187334  0.56469737 -0.44850209 -0.03397158]
    [0.23713158 0.16825311 0.27150224 0.41826406 0.21598055]
    [0.18909183 0.41593843 0.52090812 0.34777664 0.32401829]
    [0.18504038 0.45209795 0.56114246 0.30768315 0.39085638]
    """
    def __init__(self, d, kind_func, kind_coeff):
        self.d = d 
        self.qp_genz = qp.Genz(qp.IIDStdUniform(d),kind_func=kind_func,kind_coeff=kind_coeff)
    def __call__(self, level, samples=None):
        assert level==0, "Genz functions are single level"
        if samples is None:
            samples = np.random.rand(self.d)
        assert samples.shape[-1]==self.d 
        y = self.qp_genz.f(samples) 
        return y
    def ml(self, level, samples=None):
        assert level==0, "Genz functions are single level"
        return self.__call__(level=level,samples=samples)
