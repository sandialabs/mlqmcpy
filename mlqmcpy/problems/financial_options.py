import qmcpy as qp 
import numpy as np 
import types
import copy 

class MLFinancialOption(object):
    """"
    Multilevel Financial Option 
    
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> mlopts = MLFinancialOption()
    >>> mlopts.ds 
    array([   8,   16,   32,   64,  128,  256,  512, 1024])
    >>> l = 3
    >>> x = rng.uniform(size=(5,mlopts.ds[l])
    >>> mlopts(level=l,samples=x)
    >>> mlopts.ml(level=l,samples=x)
    >>> mlopts.exact_values
    array([4.40269838, 4.16711993, 4.05041723, 3.99232671, 3.96334524,
           3.94887027, 3.9416367 , 3.9380209 ])
    >>> mlopts.exact_diffs
    array([ 4.40269838e+00, -2.35578448e-01, -1.16702694e-01, -5.80905233e-02,
           -2.89814721e-02, -1.44749702e-02, -7.23356662e-03, -3.61580658e-03])
    >>> mlopts.exact.Q.mean(-1)
    np.float64(3.938020895052585)
    >>> mlopts.exact.Y.mean(-1)
    np.float64(-0.003615806583026515)
    >>> mlopts.exact.Q.mean(np.inf)
    np.float64(3.9344057385143216)
    """
    DEFAULT_KWARGS = {
        "option": "ASIAN",
        "call_put": "CALL",
        "asian_mean": "GEOMETRIC",
        "asian_mean_quadrature_rule": "RIGHT",
        "volatility": .5,
        "start_price": 39,
        "strike_price": 40,
        "interest_rate": 0.05,
        "t_final": 1,
        "barrier_in_out": "IN", 
        "barrier_price": 38,
        "digital_payout": 10,
    }
    def __init__(
            self,
            levels = 8,
            qmcpy_financial_option_args = DEFAULT_KWARGS,
            d_coarsest = 8,
            ):
        option_l0 = qp.FinancialOption(
            qp.IIDStdUniform(d_coarsest),
            level = 0, 
            d_coarsest = d_coarsest,
            **qmcpy_financial_option_args
        )
        self.levels = levels
        self.options = [option_l0]+option_l0.spawn([l for l in range(1,self.levels)])
        self.ds = np.array([option.d for option in self.options],dtype=int)
        self.exact = types.SimpleNamespace()
        self.exact.Q = types.SimpleNamespace()
        self.exact.Y = types.SimpleNamespace()
        try:
            self.exact_value_inf_dim = self.options[0].get_exact_value_inf_dim()
        except:
            pass 
        try:
            self.exact_values = np.array([self.options[l].get_exact_value() for l in range(self.levels)],dtype=float)
            self.exact_diffs = np.hstack([self.exact_values[[0]],self.exact_values[1:]-self.exact_values[:-1]])
            self.exact.Q.mean = lambda level: self.exact_value_inf_dim if level==np.inf else self.exact_values[level]
            self.exact.Y.mean = lambda level: self.exact_diffs[level]
        except:
            pass

    def evaluate(self, level, samples=None):
        if samples is None:
            samples = np.random.rand(self.ds[level])
        ogdim = samples.ndim 
        y = self.options[level].f(np.atleast_2d(samples))
        if ogdim==1:
            y = y[...,0]
        return y
    
    def __call__(self, level, samples=None):
        y = self.evaluate(level,samples)
        return y[1]

    def ml(self, level, samples):
        y = self.evaluate(level,samples) 
        return y[1]-y[0]
