import contextlib
import time

import numpy as np
import qmcpy as qp


class SlopeStability:
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> ss = SlopeStability()
    >>> ss(level=0,samples=rng.uniform(size=(ss.d,)))
    np.float64(1.144850101622033)
    >>> ss(level=1,samples=rng.uniform(size=(8,ss.d)))
    array([1.16301781, 1.16735533, 1.18786954, 1.18498913, 1.16385709,
           1.19648797, 1.16572205, 1.16658728])
    >>> ss(level=2,samples=rng.uniform(size=(2,3,ss.d)))
    array([[1.18293864, 1.1890043 , 1.18992091],
           [1.19080121, 1.18377155, 1.18160977]])
    >>> ss.ml(level=0,samples=rng.uniform(size=(ss.d,)))
    np.float64(1.147744513383981)
    >>> ss.ml(level=1,samples=rng.uniform(size=(8,ss.d)))
    array([0.02923768, 0.02855916, 0.02796304, 0.02921632, 0.02643973,
           0.02622207, 0.02792861, 0.02575347])
    >>> ss.ml(level=2,samples=rng.uniform(size=(2,3,ss.d)))
    array([[0.00350414, 0.00434304, 0.00429302],
           [0.00475261, 0.00446014, 0.00434224]])

    >>> x = qp.DigitalNetB2(ss.d,seed=7)(2**7)
    >>> for l in range(ss.levels):
    ...     q_l = ss(level=l,samples=x)
    ...     mu_q_l = q_l.mean()
    ...     sigma_q_l = q_l.std(ddof=1)
    ...     t0 = time.perf_counter()
    ...     y_l = ss.ml(level=l,samples=x)
    ...     time_y_l = time.perf_counter()-t0
    ...     mu_y_l = y_l.mean()
    ...     sigma_y_l = y_l.std(ddof=1)
    ...     print("level = %-5d mu_q_l = %-10.4f sigma_q_l = %-10.2e mu_y_l = %-10.4f sigma_y_l = %-10.2e time_y_l = %-10.2f"%(l,mu_q_l,sigma_q_l,mu_y_l,sigma_y_l,time_y_l))
    level = 0     mu_q_l = 1.1502     sigma_q_l = 1.19e-02   mu_y_l = 1.1502     sigma_y_l = 1.19e-02   time_y_l = ...
    level = 1     mu_q_l = 1.1776     sigma_q_l = 1.08e-02   mu_y_l = 0.0275     sigma_y_l = 1.07e-03   time_y_l = ...
    level = 2     mu_q_l = 1.1819     sigma_q_l = 1.04e-02   mu_y_l = 0.0042     sigma_y_l = 4.25e-04   time_y_l = ...
    """

    def __init__(self, levels=None, height=None, angle=None, num_strata=None):
        if levels is None and height is None and angle is None and num_strata is None:
            costs = np.array([2.72, 6.10, 7.68])
            self.adjusted_costs = costs / costs[-1]
        self.levels = 3 if levels is None else levels
        self.height = 2 if height is None else height
        self.angle = 50 if angle is None else angle
        self.num_strata = 4 if num_strata is None else num_strata
        self.tolerances = np.logspace(-1, -3, self.levels)
        self.d = 1

    def __call__(self, level, samples=None):
        import pyslope

        samples = np.atleast_1d(samples)
        assert samples.shape[-1] == 1
        input1d = samples.ndim == 1
        samples = np.atleast_2d(samples)
        y = np.ones(samples.shape[:-1])
        for i in np.ndindex(y.shape):
            s = pyslope.Slope(height=self.height, angle=self.angle, length=None)
            s.set_materials(
                *[
                    pyslope.Material(
                        unit_weight=20 + 2 * idx,
                        friction_angle=45 - 5 * idx,
                        cohesion=2 + 0.1 * idx**2,
                        depth_to_bottom=(idx + 1) * self.height / self.num_strata,
                        name=f"Material {idx + 1}",
                    )
                    for idx in range(self.num_strata)
                ]
            )
            s.set_udls(
                pyslope.Udl(magnitude=float(10 + 2 * (2 * samples[i].item() - 1)))
            )
            s.set_water_table(3)
            s.update_analysis_options(
                slices=10, iterations=100, tolerance=self.tolerances[level]
            )
            with contextlib.redirect_stderr(open("slopelog.log", "w")):
                s.analyse_slope()
            y[i] = s.get_min_FOS()
        if input1d:
            y = y[0]
        return y

    def ml(self, level, samples):
        y = self.__call__(level, samples)
        if level > 0:
            ylm1 = self.__call__(level - 1, samples)
            y = y - ylm1
        return y
