import numpy as np
from numpy.polynomial import Polynomial as P
from functools import reduce
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filters_path = os.path.join(project_root, 'arch', 'poc', 'parallel_iir_filters')
if filters_path not in sys.path:
    sys.path.append(filters_path)

from src.filters_wrapper import filters_wrapper
from src.config.filter_cfg import FilterCfg

def get_inv_exp_sum_as_rational_filter(A: np.ndarray, tau: np.ndarray, A_dc: float, Ts: float=0.5) -> \
        tuple[np.ndarray, np.ndarray]:
    """get_inv_exp_sum_as_rational_filter
    For analog linear distortion H, characterized by step response:
    s_H(t) = (A_dc + sum(A_i * exp(-t/tau_i), for i in 0...(N-1)))*u(t)
    , we want to calculate the coefficients of a digital rational pre-emphasis filter.
    The pre-emphasis filter is applied at sampling rate 1/Ts (by default, Ts=0.5 ns).
    The pre-emphasis filter can be applied by:
    y = scipy.signal.lfilter(b, a, x)
    where b and a are the coefficients of the filter, and x is the desired signal.
    Applying the filter H on y yields back an approximation of the desired signal x.
    :return b, a
    """

    ba_sum = [get_rational_filter_single_exp(A_i, tau_i, Ts) for A_i, tau_i in zip(A, tau)]
    if A_dc != 0:
        ba_sum += [([A_dc], [1])]

    b, a = add_rational_terms(ba_sum)
    scale = b[0]
    b, a = a/scale, b/scale

    return b, a


def get_rational_filter_single_exp(A: float, tau: float, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """get_rational_filter_single_exp - Get the feedforward and feedback taps of a single exponential filter
    with step response u(t)*(A*exp(-t/tau)), in discrete time with sampling period Ts.
    """

    a = np.array([1, (Ts - 2 * tau) / (Ts + 2 * tau)], dtype=float)
    b = np.array(A / (Ts / 2 / tau + 1) * np.array([1, -1]), dtype=float)
    return b, a


def add_rational_terms(terms: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    # Convert to Polynomial objects
    rational_terms = [(P(num), P(den)) for num, den in terms]

    # Compute common denominator
    common_den = reduce(lambda acc, t: acc * t[1], rational_terms, P([1]))

    # Adjust numerators to have the common denominator
    adjusted_numerators = []
    for num, den in rational_terms:
        multiplier = common_den // den
        adjusted_numerators.append(num * multiplier)

    # Sum all adjusted numerators
    final_numerator = sum(adjusted_numerators, P([0]))

    # Return as coefficient lists
    return final_numerator.coef, common_den.coef


if __name__ == '__main__':
    from scipy.signal import lfilter
    import matplotlib.pyplot as plt

    # # Example usage
    A = [np.float64(0.002574310165356726),
        np.float64(-0.002556892765239572),
 np.float64(0.001593735277595851),
 np.float64(0.0012087418959507998)] #np.array([1.0, 0.5])
    tau = [np.float64(18521.604589010905),
 np.float64(12081.52795202227),
 np.float64(3308.943257264784),
 np.float64(106.19481905288188)] #np.array([50, 2000])
    A_dc = 1.0
    Ts = 0.5

    b, a = get_inv_exp_sum_as_rational_filter(A, tau, A_dc, Ts)

    print(f"b: {b.tolist()}\na: {a.tolist()}")

    t = np.arange(1e6)*Ts + Ts/2
    x = A_dc + sum([A_i * np.exp(-t / tau_i) for A_i, tau_i in zip(A, tau)])
    y = lfilter(b, a, x)

    y_filtered = filters_wrapper(x=x * 0.4, A=A, tau=[tau_i*1e-9 for tau_i in tau], A_b=A_dc)

    plt.figure(figsize=(12, 8))
    plt.semilogx(t, x, label='Original signal', color='k')
    plt.semilogx(t, y, label='Filtered signal', color='r')
    plt.semilogx(t, y_filtered / 0.4, label='Filtered signal with filters_wrapper', color='b')
    plt.legend()
    plt.show()