import qmcpy as qp 
import numpy as np 
import scipy.linalg
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
    ...     for weights in ["TELESCOPING","INDEPENDENT"]:
    ...         print("    %s"%weights)
    ...         mlopts = MLFinancialOption(qmcpy_financial_option_args=opt,weights=weights)
    ...         qhat = 0
    ...         ys = [None]*mlopts.levels
    ...         for l in range(mlopts.levels):
    ...             x = qp.DigitalNetB2(mlopts.ds[l],seed=7)(2**11)
    ...             y = mlopts.ml(level=l,samples=x)
    ...             yhat_l = y.mean()
    ...             qhat += yhat_l
    ...             print("        Qhat[l] = %-10.3f Yhat[l] = %.2e"%(qhat,yhat_l))
    ...             ys[l] = y
    ...         ys = np.stack(ys,axis=-1)
    ...         cmat = ys.T@ys
    ...         with np.printoptions(formatter={"float":lambda x: "%.1e"%x}):
    ...             print(cmat)
    ASIAN
        TELESCOPING
            Qhat[l] = 6.137      Yhat[l] = 6.14e+00
            Qhat[l] = 5.843      Yhat[l] = -2.94e-01
            Qhat[l] = 5.695      Yhat[l] = -1.48e-01
            Qhat[l] = 5.621      Yhat[l] = -7.41e-02
            Qhat[l] = 5.584      Yhat[l] = -3.68e-02
            Qhat[l] = 5.566      Yhat[l] = -1.85e-02
            Qhat[l] = 5.557      Yhat[l] = -9.20e-03
            Qhat[l] = 5.552      Yhat[l] = -4.87e-03
    [[2.3e+05 -1.1e+01 -9.4e+01 -1.1e+03 8.0e+00 -2.0e+00 2.6e+00 -1.1e+02]
     [-1.1e+01 1.2e+03 1.9e+02 1.7e+00 3.7e+01 1.3e+01 9.9e+00 -2.0e-01]
     [-9.4e+01 1.9e+02 2.8e+02 -7.1e-01 1.6e+01 8.2e+00 3.0e+00 1.1e-02]
     [-1.1e+03 1.7e+00 -7.1e-01 7.3e+01 -1.9e-01 -2.9e-01 -1.2e-02 1.6e+00]
     [8.0e+00 3.7e+01 1.6e+01 -1.9e-01 1.8e+01 2.9e+00 1.7e+00 5.9e-03]
     [-2.0e+00 1.3e+01 8.2e+00 -2.9e-01 2.9e+00 4.5e+00 9.5e-01 -1.1e-02]
     [2.6e+00 9.9e+00 3.0e+00 -1.2e-02 1.7e+00 9.5e-01 1.1e+00 -1.3e-02]
     [-1.1e+02 -2.0e-01 1.1e-02 1.6e+00 5.9e-03 -1.1e-02 -1.3e-02 2.9e-01]]
        INDEPENDENT
            Qhat[l] = 5.533      Yhat[l] = 5.53e+00
            Qhat[l] = 5.545      Yhat[l] = 1.20e-02
            Qhat[l] = 5.550      Yhat[l] = 4.21e-03
            Qhat[l] = 5.551      Yhat[l] = 1.79e-03
            Qhat[l] = 5.552      Yhat[l] = 4.96e-04
            Qhat[l] = 5.552      Yhat[l] = 1.39e-04
            Qhat[l] = 5.552      Yhat[l] = 2.46e-04
            Qhat[l] = 5.552      Yhat[l] = -7.65e-05
    [[1.8e+05 2.3e+02 6.8e+01 5.3e+01 1.9e+01 1.1e+00 2.5e+00 -5.0e+00]
     [2.3e+02 8.2e+02 8.7e+00 2.0e-01 3.1e+00 -3.0e+00 2.2e+00 -3.0e-02]
     [6.8e+01 8.7e+00 1.8e+02 6.8e-01 -4.8e-01 1.1e-01 1.1e-01 -1.5e-02]
     [5.3e+01 2.0e-01 6.8e-01 4.6e+01 3.9e-01 -2.5e-01 6.3e-02 -1.0e-01]
     [1.9e+01 3.1e+00 -4.8e-01 3.9e-01 1.1e+01 6.4e-02 1.5e-01 8.3e-02]
     [1.1e+00 -3.0e+00 1.1e-01 -2.5e-01 6.4e-02 2.5e+00 -2.7e-02 2.8e-03]
     [2.5e+00 2.2e+00 1.1e-01 6.3e-02 1.5e-01 -2.7e-02 5.9e-01 -2.3e-03]
     [-5.0e+00 -3.0e-02 -1.5e-02 -1.0e-01 8.3e-02 2.8e-03 -2.3e-03 1.3e-01]]
    LOOKBACK
        TELESCOPING
            Qhat[l] = 12.991     Yhat[l] = 1.30e+01
            Qhat[l] = 14.397     Yhat[l] = 1.41e+00
            Qhat[l] = 15.298     Yhat[l] = 9.01e-01
            Qhat[l] = 15.886     Yhat[l] = 5.88e-01
            Qhat[l] = 16.279     Yhat[l] = 3.93e-01
            Qhat[l] = 16.550     Yhat[l] = 2.71e-01
            Qhat[l] = 16.749     Yhat[l] = 2.00e-01
            Qhat[l] = 16.880     Yhat[l] = 1.30e-01
    [[7.2e+05 3.0e+04 2.4e+04 1.5e+04 9.3e+03 6.5e+03 4.5e+03 3.6e+03]
     [3.0e+04 1.4e+04 3.0e+03 1.5e+03 1.0e+03 7.5e+02 6.0e+02 3.6e+02]
     [2.4e+04 3.0e+03 5.7e+03 9.2e+02 8.0e+02 5.2e+02 3.8e+02 2.4e+02]
     [1.5e+04 1.5e+03 9.2e+02 2.4e+03 4.2e+02 3.4e+02 2.2e+02 1.6e+02]
     [9.3e+03 1.0e+03 8.0e+02 4.2e+02 1.1e+03 2.4e+02 1.8e+02 1.1e+02]
     [6.5e+03 7.5e+02 5.2e+02 3.4e+02 2.4e+02 5.1e+02 1.1e+02 7.7e+01]
     [4.5e+03 6.0e+02 3.8e+02 2.2e+02 1.8e+02 1.1e+02 2.7e+02 5.2e+01]
     [3.6e+03 3.6e+02 2.4e+02 1.6e+02 1.1e+02 7.7e+01 5.2e+01 1.1e+02]]
        INDEPENDENT
            Qhat[l] = 15.244     Yhat[l] = 1.52e+01
            Qhat[l] = 16.039     Yhat[l] = 7.96e-01
            Qhat[l] = 16.377     Yhat[l] = 3.38e-01
            Qhat[l] = 16.565     Yhat[l] = 1.88e-01
            Qhat[l] = 16.699     Yhat[l] = 1.34e-01
            Qhat[l] = 16.761     Yhat[l] = 6.12e-02
            Qhat[l] = 16.836     Yhat[l] = 7.50e-02
            Qhat[l] = 16.874     Yhat[l] = 3.83e-02
    [[9.9e+05 1.2e+04 1.7e+04 3.1e+03 3.9e+03 3.4e+03 2.0e+03 1.2e+03]
     [1.2e+04 1.8e+04 -1.1e+02 3.6e+02 2.3e+02 -8.6e+00 2.2e+02 6.9e+01]
     [1.7e+04 -1.1e+02 7.2e+03 -7.5e+01 1.2e+02 1.2e+02 4.9e+01 3.6e+01]
     [3.1e+03 3.6e+02 -7.5e+01 2.4e+03 1.2e+01 2.3e+01 1.7e+01 1.9e+01]
     [3.9e+03 2.3e+02 1.2e+02 1.2e+01 9.9e+02 3.6e+01 3.8e+01 1.1e+01]
     [3.4e+03 -8.6e+00 1.2e+02 2.3e+01 3.6e+01 4.1e+02 6.9e+00 7.7e+00]
     [2.0e+03 2.2e+02 4.9e+01 1.7e+01 3.8e+01 6.9e+00 2.1e+02 4.2e+00]
     [1.2e+03 6.9e+01 3.6e+01 1.9e+01 1.1e+01 7.7e+00 4.2e+00 8.9e+01]]
    BARRIER
        TELESCOPING
            Qhat[l] = 10.105     Yhat[l] = 1.01e+01
            Qhat[l] = 10.165     Yhat[l] = 6.02e-02
            Qhat[l] = 10.193     Yhat[l] = 2.76e-02
            Qhat[l] = 10.220     Yhat[l] = 2.78e-02
            Qhat[l] = 10.237     Yhat[l] = 1.63e-02
            Qhat[l] = 10.255     Yhat[l] = 1.82e-02
            Qhat[l] = 10.261     Yhat[l] = 6.29e-03
            Qhat[l] = 10.270     Yhat[l] = 8.64e-03
    [[6.6e+05 1.5e+03 6.3e+02 7.8e+02 2.1e+02 1.8e+02 1.3e+02 0.0e+00]
     [1.5e+03 7.5e+02 0.0e+00 3.1e+01 0.0e+00 0.0e+00 0.0e+00 0.0e+00]
     [6.3e+02 0.0e+00 3.5e+02 1.0e+01 0.0e+00 0.0e+00 0.0e+00 0.0e+00]
     [7.8e+02 3.1e+01 1.0e+01 3.3e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00]
     [2.1e+02 0.0e+00 0.0e+00 0.0e+00 1.6e+02 0.0e+00 0.0e+00 0.0e+00]
     [1.8e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00 2.4e+02 0.0e+00 0.0e+00]
     [1.3e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 1.0e+02 0.0e+00]
     [0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 1.6e+02]]
        INDEPENDENT
            Qhat[l] = 10.105     Yhat[l] = 1.01e+01
            Qhat[l] = 10.165     Yhat[l] = 6.02e-02
            Qhat[l] = 10.193     Yhat[l] = 2.76e-02
            Qhat[l] = 10.220     Yhat[l] = 2.78e-02
            Qhat[l] = 10.237     Yhat[l] = 1.63e-02
            Qhat[l] = 10.255     Yhat[l] = 1.82e-02
            Qhat[l] = 10.258     Yhat[l] = 3.14e-03
            Qhat[l] = 10.269     Yhat[l] = 1.07e-02
    [[6.6e+05 1.5e+03 6.3e+02 7.8e+02 2.1e+02 1.8e+02 6.5e+01 2.4e-09]
     [1.5e+03 7.5e+02 8.3e-12 3.1e+01 1.3e-11 -3.7e-14 -5.1e-15 5.1e+00]
     [6.3e+02 8.3e-12 3.5e+02 1.0e+01 8.9e-14 -2.0e-14 -3.6e-14 4.4e-14]
     [7.8e+02 3.1e+01 1.0e+01 3.3e+02 1.7e-13 8.5e-14 -3.7e-14 7.1e-15]
     [2.1e+02 1.3e-11 8.9e-14 1.7e-13 1.6e+02 -3.5e-14 -5.2e-15 1.0e-13]
     [1.8e+02 -3.7e-14 -2.0e-14 8.5e-14 -3.5e-14 2.4e+02 -1.3e-25 2.2e-14]
     [6.5e+01 -5.1e-15 -3.6e-14 -3.7e-14 -5.2e-15 -1.3e-25 2.5e+01 3.3e-15]
     [2.4e-09 5.1e+00 4.4e-14 7.1e-15 1.0e-13 2.2e-14 3.3e-15 1.7e+02]]
    """
    
    def __init__(
            self,
            levels = 8,
            qmcpy_financial_option_args = "ASIAN",
            weights = "TELESCOPING", 
            d_coarsest = 8,
            ):
        if weights=="TELESCOPING":
            self.mmat = np.eye(levels)
            self.mmat[np.arange(1,levels),np.arange(levels-1)] = -1
        elif weights=="INDEPENDENT":
            assert isinstance(qmcpy_financial_option_args,str)
        else:
            raise Exception("invalid weights = %s"%weights)
        if isinstance(qmcpy_financial_option_args,str):
            datadir = os.path.dirname(os.path.realpath(__file__))+"/fo_data"
            if weights=="INDEPENDENT":
                data = np.load(datadir+"/%s.mmat.npy"%qmcpy_financial_option_args,allow_pickle=True)[()]
                self.mmat = data["mmat"]
            with open(datadir+"/financial_options_settings.json","r") as file:
                kwargs = json.load(file)
            qmcpy_financial_option_args = kwargs[qmcpy_financial_option_args.upper()]
        self.weights = weights
        self.levels = levels
        self.ds = d_coarsest*2**np.arange(levels)
        self.options = [qp.FinancialOption(qp.IIDStdUniform(d),**qmcpy_financial_option_args) for d in self.ds]
        self.exact = types.SimpleNamespace()
        self.exact.Q = types.SimpleNamespace()
        self.exact.Y = types.SimpleNamespace()
        try:
            self.exact_value_inf_dim = self.options[-1].get_exact_value_inf_dim()
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
        samples = np.atleast_2d(samples)
        t = self.options[level].true_measure._transform(samples)
        ys = [None]*(level+1)
        for l in range(level+1):
            t_l = t[:,::-2**(level-l)][...,::-1]
            ys[l] = self.options[l].g(t_l)
        y = np.stack(ys,axis=-1)
        if ogdim==1:
            y = y[0]
        return y
    
    def __call__(self, level, samples=None):
        ys = self.evaluate(level,samples)
        y = ys[...,-1]
        return y

    def ml(self, level, samples):
        y = self.evaluate(level,samples)
        z = (self.mmat[level,:(level+1)]*y).sum(-1)
        return z

if __name__=="__main__":
    qmcpy_financial_option_args = "BARRIER"
    fo = MLFinancialOption(qmcpy_financial_option_args=qmcpy_financial_option_args)
    x = qp.DigitalNet(fo.ds[-1],seed=7)(2**10) 
    y = fo.evaluate(level=fo.levels-1,samples=x)
    cmat = y.T@y+1e-8*np.eye(y.shape[1])
    lchol = scipy.linalg.cholesky(cmat,lower=True) 
    lcholinv = scipy.linalg.solve_triangular(lchol,np.eye(lchol.shape[1]),lower=True)
    mmat = lchol[-1,:,None]*lcholinv
    np.save(os.path.dirname(os.path.abspath(__file__))+"/fo_data/%s.mmat.npy"%qmcpy_financial_option_args,{"mmat":mmat})

