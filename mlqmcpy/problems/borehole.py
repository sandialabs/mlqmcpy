import numpy as np

from .utils import multilevel


@multilevel
def borehole(level, samples=None):
    """
    https://www.sfu.ca/~ssurjano/borehole.html

    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> samples = rng.uniform(low=0,high=1,size=(4,8))
    >>> borehole(level=0,samples=samples)
    array([81.90804636, 69.47379113, 35.3578304 , 18.44431368])
    >>> borehole(level=1,samples=samples)
    array([204.7694352 , 173.68403634,  88.3945061 ,  46.11076045])
    """
    import scipy.stats

    distribs = [
        scipy.stats.norm(loc=0.10, scale=0.0161812),
        scipy.stats.lognorm(scale=np.exp(7.71), s=1.0056),
        scipy.stats.uniform(loc=63070, scale=115600 - 63070),
        scipy.stats.uniform(loc=990, scale=1110 - 990),
        scipy.stats.uniform(loc=63.1, scale=116 - 63.1),
        scipy.stats.uniform(loc=700, scale=820 - 700),
        scipy.stats.uniform(loc=1120, scale=1680 - 1120),
        scipy.stats.uniform(
            loc=1500, scale=15000 - 1500
        ),  # scipy.stats.uniform(loc=9855,scale=12045-9855)
    ]
    assert samples.shape[-1] == 8
    assert level in [
        0,
        1,
    ], "borehole only supports level=0 (low fidelity) or level=1 (high fidelity)"
    samples = np.stack([distribs[j].ppf(samples[..., j]) for j in range(8)], axis=-1)
    rw = samples[:, 0]
    r = samples[:, 1]
    Tu = samples[:, 2]
    Hu = samples[:, 3]
    Tl = samples[:, 4]
    Hl = samples[:, 5]
    L = samples[:, 6]
    Kw = samples[:, 7]
    if level == 0:
        C1 = 2 * np.pi
        C2 = 1
    elif level == 1:
        C1 = 5
        C2 = 1.5
    frac1 = C1 * Tu * (Hu - Hl)
    frac2a = 2 * L * Tu / (np.log(r / rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (C2 + frac2a + frac2b)
    y = frac1 / frac2
    return y
