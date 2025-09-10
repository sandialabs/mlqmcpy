import qmcpy as qp 
import numpy as np 
import types
import json 
import os 

class MLFinancialOption(object):
    """"
    Multilevel Financial Option 
    
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> mlopts = MLFinancialOption()
    >>> mlopts.ds 
    array([   8,   16,   32,   64,  128,  256,  512, 1024])
    >>> l = 3
    >>> x = rng.uniform(size=(5,mlopts.ds[l]))
    >>> mlopts(level=l,samples=x)
    array([6.84079169, 0.        , 0.        , 0.        , 0.        ])
    >>> mlopts.ml(level=l,samples=x)
    array([-0.13455718,  0.        ,  0.        ,  0.        ,  0.        ])
    >>> mlopts.exact_values
    array([6.13765156, 5.84167235, 5.69411372, 5.62043443, 5.58361875,
           5.56521677, 5.55601722, 5.55141781])
    >>> mlopts.exact_diffs
    array([ 6.13765156e+00, -2.95979207e-01, -1.47558630e-01, -7.36792928e-02,
           -3.68156810e-02, -1.84019843e-02, -9.19954533e-03, -4.59941314e-03])
    >>> mlopts.exact.Q.mean(-1)
    np.float64(5.5514178080755245)
    >>> mlopts.exact.Y.mean(-1)
    np.float64(-0.00459941313865464)
    >>> mlopts.exact.Q.mean(np.inf)
    np.float64(5.546818633789201)

    >>> for opt in ["ASIAN","LOOKBACK","BARRIER"]:
    ...     print(opt)
    ...     mlopts = MLFinancialOption(qmcpy_financial_option_args=opt)
    ...     qhat_prev = 0
    ...     for l in range(mlopts.levels):
    ...         x = qp.DigitalNetB2(mlopts.ds[l],seed=7)(2**11)
    ...         q = mlopts(level=l,samples=x)
    ...         qhat_l = q.mean()
    ...         yhat_l = qhat_l-qhat_prev
    ...         print("    Qhat[l] = %-10.3f Yhat[l] = %.2e"%(qhat_l,yhat_l))
    ...         qhat_prev = qhat_l
    ASIAN
        Qhat[l] = 6.137      Yhat[l] = 6.14e+00
        Qhat[l] = 5.842      Yhat[l] = -2.96e-01
        Qhat[l] = 5.694      Yhat[l] = -1.48e-01
        Qhat[l] = 5.624      Yhat[l] = -6.96e-02
        Qhat[l] = 5.584      Yhat[l] = -4.02e-02
        Qhat[l] = 5.568      Yhat[l] = -1.58e-02
        Qhat[l] = 5.556      Yhat[l] = -1.22e-02
        Qhat[l] = 5.551      Yhat[l] = -5.01e-03
    LOOKBACK
        Qhat[l] = 12.991     Yhat[l] = 1.30e+01
        Qhat[l] = 14.384     Yhat[l] = 1.39e+00
        Qhat[l] = 15.278     Yhat[l] = 8.94e-01
        Qhat[l] = 15.917     Yhat[l] = 6.39e-01
        Qhat[l] = 16.324     Yhat[l] = 4.07e-01
        Qhat[l] = 16.610     Yhat[l] = 2.86e-01
        Qhat[l] = 16.786     Yhat[l] = 1.76e-01
        Qhat[l] = 16.892     Yhat[l] = 1.05e-01
    BARRIER
        Qhat[l] = 10.105     Yhat[l] = 1.01e+01
        Qhat[l] = 10.136     Yhat[l] = 3.12e-02
        Qhat[l] = 10.187     Yhat[l] = 5.07e-02
        Qhat[l] = 10.241     Yhat[l] = 5.47e-02
        Qhat[l] = 10.291     Yhat[l] = 4.94e-02
        Qhat[l] = 10.285     Yhat[l] = -5.41e-03
        Qhat[l] = 10.273     Yhat[l] = -1.22e-02
        Qhat[l] = 10.297     Yhat[l] = 2.36e-02
    """
    
    def __init__(
            self,
            levels = 8,
            qmcpy_financial_option_args = "ASIAN",
            d_coarsest = 8,
            ):
        if isinstance(qmcpy_financial_option_args,str):
            with open(os.path.dirname(os.path.realpath(__file__))+"/financial_options_settings.json","r") as file:
                kwargs = json.load(file)
            qmcpy_financial_option_args = kwargs[qmcpy_financial_option_args.upper()]
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
