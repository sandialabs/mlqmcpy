import json
import os
import types

import numpy as np
import qmcpy as qp
import scipy.linalg


class MLFinancialOption(object):
    """ "
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
    ...         qmean = 0
    ...         ys = [None]*mlopts.levels
    ...         for l in range(mlopts.levels):
    ...             x = rng.uniform(size=(2**12,mlopts.ds[l]))
    ...             y = mlopts.ml(level=l,samples=x)
    ...             ymean_l = y.mean()
    ...             ystd_l = y.std(ddof=1)
    ...             qmean += ymean_l
    ...             print("        Qmean[l] = %-10.3f Ymean[l] = %-15.3e Ystd[l] = %.3e"%(qmean,ymean_l,ystd_l))
    ...             ys[l] = y
    ...         ys = np.stack(ys,axis=-1)
    ...         cmat = ys.T@ys
    ...         with np.printoptions(formatter={"float":lambda x: "%.1e"%x}):
    ...             print(cmat)
    ASIAN
        TELESCOPING
            Qmean[l] = 6.340      Ymean[l] = 6.340e+00       Ystd[l] = 8.667e+00
            Qmean[l] = 6.043      Ymean[l] = -2.969e-01      Ystd[l] = 6.821e-01
            Qmean[l] = 5.896      Ymean[l] = -1.471e-01      Ystd[l] = 3.499e-01
            Qmean[l] = 5.824      Ymean[l] = -7.194e-02      Ystd[l] = 1.714e-01
            Qmean[l] = 5.787      Ymean[l] = -3.725e-02      Ystd[l] = 8.426e-02
            Qmean[l] = 5.769      Ymean[l] = -1.770e-02      Ystd[l] = 4.252e-02
            Qmean[l] = 5.760      Ymean[l] = -9.336e-03      Ystd[l] = 2.132e-02
            Qmean[l] = 5.755      Ymean[l] = -4.729e-03      Ystd[l] = 1.083e-02
    [[4.7e+05 -8.2e+03 -3.8e+03 -1.7e+03 -1.0e+03 -4.6e+02 -2.7e+02 -1.2e+02]
     [-8.2e+03 2.3e+03 1.7e+02 9.1e+01 4.6e+01 2.2e+01 1.2e+01 5.5e+00]
     [-3.8e+03 1.7e+02 5.9e+02 4.7e+01 2.4e+01 9.9e+00 5.7e+00 2.9e+00]
     [-1.7e+03 9.1e+01 4.7e+01 1.4e+02 9.9e+00 4.4e+00 2.9e+00 1.3e+00]
     [-1.0e+03 4.6e+01 2.4e+01 9.9e+00 3.5e+01 2.8e+00 1.6e+00 7.4e-01]
     [-4.6e+02 2.2e+01 9.9e+00 4.4e+00 2.8e+00 8.7e+00 7.3e-01 2.7e-01]
     [-2.7e+02 1.2e+01 5.7e+00 2.9e+00 1.6e+00 7.3e-01 2.2e+00 2.1e-01]
     [-1.2e+02 5.5e+00 2.9e+00 1.3e+00 7.4e-01 2.7e-01 2.1e-01 5.7e-01]]
        INDEPENDENT
            Qmean[l] = 5.627      Ymean[l] = 5.627e+00       Ystd[l] = 7.748e+00
            Qmean[l] = 5.626      Ymean[l] = -9.842e-04      Ystd[l] = 6.250e-01
            Qmean[l] = 5.634      Ymean[l] = 7.543e-03       Ystd[l] = 3.002e-01
            Qmean[l] = 5.638      Ymean[l] = 4.559e-03       Ystd[l] = 1.505e-01
            Qmean[l] = 5.638      Ymean[l] = -5.437e-04      Ystd[l] = 7.329e-02
            Qmean[l] = 5.639      Ymean[l] = 9.015e-04       Ystd[l] = 3.484e-02
            Qmean[l] = 5.640      Ymean[l] = 6.947e-04       Ystd[l] = 1.726e-02
            Qmean[l] = 5.640      Ymean[l] = 1.406e-04       Ystd[l] = 8.047e-03
    [[3.8e+05 2.4e+02 2.9e+02 1.2e+02 -6.9e+00 1.8e+01 2.4e+01 1.8e+00]
     [2.4e+02 1.6e+03 -1.9e+01 7.5e-01 -3.6e-01 4.6e-01 -1.4e-02 5.2e-02]
     [2.9e+02 -1.9e+01 3.7e+02 1.5e-02 -2.3e-01 -1.5e+00 4.9e-01 -2.0e-01]
     [1.2e+02 7.5e-01 1.5e-02 9.3e+01 -1.2e+00 -3.7e-01 4.7e-01 4.3e-02]
     [-6.9e+00 -3.6e-01 -2.3e-01 -1.2e+00 2.2e+01 4.3e-01 4.1e-03 7.1e-02]
     [1.8e+01 4.6e-01 -1.5e+00 -3.7e-01 4.3e-01 5.0e+00 4.3e-03 4.7e-03]
     [2.4e+01 -1.4e-02 4.9e-01 4.7e-01 4.1e-03 4.3e-03 1.2e+00 1.2e-02]
     [1.8e+00 5.2e-02 -2.0e-01 4.3e-02 7.1e-02 4.7e-03 1.2e-02 2.7e-01]]
    LOOKBACK
        TELESCOPING
            Qmean[l] = 12.988     Ymean[l] = 1.299e+01       Ystd[l] = 1.326e+01
            Qmean[l] = 14.367     Ymean[l] = 1.379e+00       Ystd[l] = 2.136e+00
            Qmean[l] = 15.257     Ymean[l] = 8.900e-01       Ystd[l] = 1.370e+00
            Qmean[l] = 15.838     Ymean[l] = 5.802e-01       Ystd[l] = 9.042e-01
            Qmean[l] = 16.244     Ymean[l] = 4.067e-01       Ystd[l] = 6.263e-01
            Qmean[l] = 16.522     Ymean[l] = 2.778e-01       Ystd[l] = 4.211e-01
            Qmean[l] = 16.704     Ymean[l] = 1.816e-01       Ystd[l] = 2.899e-01
            Qmean[l] = 16.832     Ymean[l] = 1.283e-01       Ystd[l] = 1.966e-01
    [[1.4e+06 7.2e+04 4.8e+04 3.2e+04 2.1e+04 1.5e+04 9.8e+03 7.1e+03]
     [7.2e+04 2.6e+04 4.7e+03 3.2e+03 2.2e+03 1.6e+03 1.0e+03 7.1e+02]
     [4.8e+04 4.7e+03 1.1e+04 2.1e+03 1.5e+03 1.0e+03 6.4e+02 4.7e+02]
     [3.2e+04 3.2e+03 2.1e+03 4.7e+03 9.4e+02 6.1e+02 4.3e+02 2.9e+02]
     [2.1e+04 2.2e+03 1.5e+03 9.4e+02 2.3e+03 4.6e+02 3.0e+02 2.1e+02]
     [1.5e+04 1.6e+03 1.0e+03 6.1e+02 4.6e+02 1.0e+03 2.2e+02 1.5e+02]
     [9.8e+03 1.0e+03 6.4e+02 4.3e+02 3.0e+02 2.2e+02 4.8e+02 9.9e+01]
     [7.1e+03 7.1e+02 4.7e+02 2.9e+02 2.1e+02 1.5e+02 9.9e+01 2.3e+02]]
        INDEPENDENT
            Qmean[l] = 15.236     Ymean[l] = 1.524e+01       Ystd[l] = 1.588e+01
            Qmean[l] = 16.026     Ymean[l] = 7.899e-01       Ystd[l] = 2.880e+00
            Qmean[l] = 16.319     Ymean[l] = 2.932e-01       Ystd[l] = 1.812e+00
            Qmean[l] = 16.532     Ymean[l] = 2.133e-01       Ystd[l] = 1.110e+00
            Qmean[l] = 16.667     Ymean[l] = 1.348e-01       Ystd[l] = 6.873e-01
            Qmean[l] = 16.736     Ymean[l] = 6.877e-02       Ystd[l] = 4.386e-01
            Qmean[l] = 16.798     Ymean[l] = 6.256e-02       Ystd[l] = 2.950e-01
            Qmean[l] = 16.836     Ymean[l] = 3.728e-02       Ystd[l] = 2.004e-01
    [[2.0e+06 4.4e+04 1.8e+04 1.4e+04 8.6e+03 4.2e+03 4.4e+03 2.2e+03]
     [4.4e+04 3.7e+04 1.6e+03 9.8e+02 3.6e+02 2.1e+02 1.5e+02 1.1e+02]
     [1.8e+04 1.6e+03 1.4e+04 1.0e+02 2.0e+02 7.8e+01 6.4e+01 2.3e+01]
     [1.4e+04 9.8e+02 1.0e+02 5.2e+03 1.1e+02 8.2e+01 2.2e+01 3.4e+01]
     [8.6e+03 3.6e+02 2.0e+02 1.1e+02 2.0e+03 2.7e+01 5.6e+01 2.5e+01]
     [4.2e+03 2.1e+02 7.8e+01 8.2e+01 2.7e+01 8.1e+02 4.3e+00 1.1e+01]
     [4.4e+03 1.5e+02 6.4e+01 2.2e+01 5.6e+01 4.3e+00 3.7e+02 1.2e+01]
     [2.2e+03 1.1e+02 2.3e+01 3.4e+01 2.5e+01 1.1e+01 1.2e+01 1.7e+02]]
    BARRIER
        TELESCOPING
            Qmean[l] = 9.933      Ymean[l] = 9.933e+00       Ystd[l] = 1.443e+01
            Qmean[l] = 10.001     Ymean[l] = 6.782e-02       Ystd[l] = 6.526e-01
            Qmean[l] = 10.035     Ymean[l] = 3.381e-02       Ystd[l] = 4.437e-01
            Qmean[l] = 10.068     Ymean[l] = 3.361e-02       Ystd[l] = 4.793e-01
            Qmean[l] = 10.090     Ymean[l] = 2.136e-02       Ystd[l] = 3.653e-01
            Qmean[l] = 10.109     Ymean[l] = 1.902e-02       Ystd[l] = 3.388e-01
            Qmean[l] = 10.124     Ymean[l] = 1.565e-02       Ystd[l] = 3.261e-01
            Qmean[l] = 10.134     Ymean[l] = 9.973e-03       Ystd[l] = 2.576e-01
    [[1.3e+06 2.5e+03 1.3e+03 1.5e+03 8.4e+02 8.1e+02 2.1e+02 5.7e+02]
     [2.5e+03 1.8e+03 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00]
     [1.3e+03 0.0e+00 8.1e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00]
     [1.5e+03 0.0e+00 0.0e+00 9.5e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00]
     [8.4e+02 0.0e+00 0.0e+00 0.0e+00 5.5e+02 0.0e+00 0.0e+00 0.0e+00]
     [8.1e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00 4.7e+02 0.0e+00 0.0e+00]
     [2.1e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 4.4e+02 0.0e+00]
     [5.7e+02 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 2.7e+02]]
        INDEPENDENT
            Qmean[l] = 9.827      Ymean[l] = 9.827e+00       Ystd[l] = 1.464e+01
            Qmean[l] = 9.902      Ymean[l] = 7.526e-02       Ystd[l] = 7.064e-01
            Qmean[l] = 9.951      Ymean[l] = 4.898e-02       Ystd[l] = 5.748e-01
            Qmean[l] = 9.991      Ymean[l] = 3.991e-02       Ystd[l] = 5.068e-01
            Qmean[l] = 10.011     Ymean[l] = 2.038e-02       Ystd[l] = 3.255e-01
            Qmean[l] = 10.022     Ymean[l] = 1.092e-02       Ystd[l] = 2.765e-01
            Qmean[l] = 10.025     Ymean[l] = 3.393e-03       Ystd[l] = 9.444e-02
            Qmean[l] = 10.037     Ymean[l] = 1.125e-02       Ystd[l] = 1.921e-01
    [[1.3e+06 2.1e+03 2.4e+03 1.4e+03 9.6e+02 3.3e+02 3.4e+02 2.8e+02]
     [2.1e+03 2.1e+03 1.4e-09 4.5e-10 3.9e-11 1.3e-11 8.5e-11 8.2e+00]
     [2.4e+03 1.4e-09 1.4e+03 2.9e+00 2.5e-09 -8.9e-14 2.5e-10 9.8e-14]
     [1.4e+03 4.5e-10 2.9e+00 1.1e+03 1.6e-09 -7.5e-14 1.7e+00 6.0e-10]
     [9.6e+02 3.9e-11 2.5e-09 1.6e-09 4.4e+02 -6.0e-14 -2.3e-14 9.3e-14]
     [3.3e+02 1.3e-11 -8.9e-14 -7.5e-14 -6.0e-14 3.1e+02 -2.3e-14 2.1e-14]
     [3.4e+02 8.5e-11 2.5e-10 1.7e+00 -2.3e-14 -2.3e-14 3.7e+01 9.5e-15]
     [2.8e+02 8.2e+00 9.8e-14 6.0e-10 9.3e-14 2.1e-14 9.5e-15 1.5e+02]]
    """

    def __init__(
        self,
        levels=8,
        qmcpy_financial_option_args="ASIAN",
        weights="TELESCOPING",
        d_coarsest=8,
    ):
        if weights == "TELESCOPING":
            self.mmat = np.eye(levels)
            self.mmat[np.arange(1, levels), np.arange(levels - 1)] = -1
        elif weights == "INDEPENDENT":
            assert isinstance(qmcpy_financial_option_args, str)
        else:
            raise Exception("invalid weights = %s" % weights)
        if isinstance(qmcpy_financial_option_args, str):
            datadir = os.path.dirname(os.path.realpath(__file__)) + "/fo_data"
            if weights == "INDEPENDENT":
                data = np.load(
                    datadir + "/%s.mmat.npy" % qmcpy_financial_option_args,
                    allow_pickle=True,
                )[()]
                self.mmat = data["mmat"]
            with open(datadir + "/financial_options_settings.json", "r") as file:
                kwargs = json.load(file)
            qmcpy_financial_option_args = kwargs[qmcpy_financial_option_args.upper()]
        self.weights = weights
        self.levels = levels
        self.ds = d_coarsest * 2 ** np.arange(levels)
        self.options = [
            qp.FinancialOption(qp.IIDStdUniform(d), **qmcpy_financial_option_args)
            for d in self.ds
        ]
        self.exact = types.SimpleNamespace()
        self.exact.Q = types.SimpleNamespace()
        self.exact.Y = types.SimpleNamespace()
        try:
            self.exact_value_inf_dim = self.options[-1].get_exact_value_inf_dim()
        except:
            pass
        try:
            self.exact_values = np.array(
                [self.options[l].get_exact_value() for l in range(self.levels)],
                dtype=float,
            )
            self.exact_diffs = np.hstack(
                [self.exact_values[[0]], self.exact_values[1:] - self.exact_values[:-1]]
            )
            self.exact.Q.mean = lambda level: (
                self.exact_value_inf_dim
                if level == np.inf
                else self.exact_values[level]
            )
            self.exact.Y.mean = lambda level: self.exact_diffs[level]
        except:
            pass

    def evaluate(self, level, samples=None):
        if samples is None:
            samples = np.random.rand(self.ds[level])
        ogdim = samples.ndim
        samples = np.atleast_2d(samples)
        t = self.options[level].true_measure._transform(samples)
        ys = [None] * (level + 1)
        for l in range(level + 1):
            t_l = t[:, :: -(2 ** (level - l))][..., ::-1]
            ys[l] = self.options[l].g(t_l)
        y = np.stack(ys, axis=-1)
        if ogdim == 1:
            y = y[0]
        return y

    def __call__(self, level, samples=None):
        ys = self.evaluate(level, samples)
        y = ys[..., -1]
        return y

    def ml(self, level, samples):
        y = self.evaluate(level, samples)
        z = (self.mmat[level, : (level + 1)] * y).sum(-1)
        return z


if __name__ == "__main__":
    qmcpy_financial_option_args = "BARRIER"
    fo = MLFinancialOption(qmcpy_financial_option_args=qmcpy_financial_option_args)
    x = qp.DigitalNet(fo.ds[-1], seed=7)(2**10)
    y = fo.evaluate(level=fo.levels - 1, samples=x)
    cmat = y.T @ y + 1e-8 * np.eye(y.shape[1])
    lchol = scipy.linalg.cholesky(cmat, lower=True)
    lcholinv = scipy.linalg.solve_triangular(lchol, np.eye(lchol.shape[1]), lower=True)
    mmat = lchol[-1, :, None] * lcholinv
    np.save(
        os.path.dirname(os.path.abspath(__file__))
        + "/fo_data/%s.mmat.npy" % qmcpy_financial_option_args,
        {"mmat": mmat},
    )
