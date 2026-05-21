import numpy as np
from scipy.stats import norm

from .utils import multilevel


def standard_brownian_motion_kl(t, x, t_final=1):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> t = np.linspace(0,1,5)
    >>> x = rng.uniform(low=0,high=1,size=(2,8))
    >>> standard_brownian_motion_kl(t,x)
    array([[ 0.        ,  0.62586887,  0.55912079, -0.05659721, -0.23763712],
           [ 0.        ,  0.18122323,  0.57028506,  0.7157861 ,  0.69326928]])

    (6.8) in https://artowen.su.domains/mc/Ch-processes.pdf
    """
    assert np.isscalar(t_final) and t_final > 0
    assert (
        isinstance(t, np.ndarray)
        and t.ndim == 1
        and (0 <= t).all()
        and (t <= t_final).all()
    )
    assert isinstance(x, np.ndarray) and x.ndim >= 1
    t01 = t / t_final
    z = norm.ppf(x)
    j = np.arange(z.shape[-1])
    factor = 2 / (2 * j + 1)
    sbm01 = (
        np.sqrt(2)
        / np.pi
        * z[..., None]
        * factor[..., None]
        * np.sin(np.pi * t01 / factor[..., None])
    ).sum(-2)
    sbm = np.sqrt(t_final) * sbm01
    return sbm


@multilevel
def asian_option(
    level=5,
    coeffs=None,
    start_price=30,
    strike_price=35,
    drift=0,
    volatility=1.0,
    interest_rate=0.01,
    option_type="call",
    floating_option_weight=1,
    t_final=1,
):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> discounted_payoffs = asian_option(level=0,coeffs=rng.uniform(low=0,high=1,size=(1000,8)))
    >>> discounted_payoffs.shape
    (1000,)
    >>> discounted_payoffs.mean()
    np.float64(4.904107607942888)

    https://en.wikipedia.org/wiki/Asian_option
    """
    if coeffs is None:
        coeffs = np.random.rand(8)
    assert isinstance(coeffs, np.ndarray)
    assert np.isscalar(start_price)
    assert np.isscalar(strike_price)
    assert isinstance(option_type, str)
    option_type = option_type.lower()
    assert option_type in ["call", "put", "call", "floating call", "floating put"]
    assert np.isscalar(drift)
    assert np.isscalar(volatility)
    assert np.isscalar(floating_option_weight)
    assert np.isscalar(interest_rate)
    assert np.isscalar(t_final)
    N = 2 ** (level + 2)
    t = np.linspace(0, t_final, N)
    dt = t[1] - t[0]
    sbm = standard_brownian_motion_kl(t, coeffs, t_final=t_final)
    gbm = start_price * np.exp((drift - volatility**2 / 2) * t + volatility * sbm)
    avg = (dt * (gbm[..., :-1] + gbm[..., 1:]) / 2).sum(-1)  # trapezoidal rule
    final_price = gbm[..., -1]
    if option_type == "call":
        payoff = np.maximum(avg - strike_price, 0)
    elif option_type == "put":
        payoff = np.maximum(strike_price - avg, 0)
    elif option_type == "floating call":
        payoff = np.maximum(final_price - floating_option_weight * avg, 0)
    elif option_type == "floating put":
        payoff = np.maximum(floating_option_weight * avg - final_price, 0)
    else:
        assert False, "case parsing error"
    discounted_payoff = payoff * np.exp(-interest_rate * t_final)
    return discounted_payoff
