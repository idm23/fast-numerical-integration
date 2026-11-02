"""Testing utilities: run SciPy with the Cython engine's floating-point summation order.

:func:`scipy_with_our_summation_order` temporarily patches SciPy's explicit
Runge-Kutta internals (``rk_step``, the embedded error norms and the RMS norm
used by the initial-step heuristic) so they add terms in exactly the order the
Cython loops do. Everything else, including step-size control, acceptance and
bookkeeping, remains SciPy's own code.

With the patch applied, SciPy and :func:`fast_numerical_integration.integrators.solve_ivp`
produce bit-identical trajectories (given the same RHS object). This separates
genuine algorithmic differences, of which there are none, from rounding
differences. Rounding differences are otherwise amplified on problems where the
step controller is driven by round-off: DOP853 near the noise floor, stiff
problems on the stability boundary, chaotic systems.

Assumes the extension is compiled without fused multiply-add contraction (the
default x86-64 build flags).
"""

import contextlib

import numpy as np
import scipy.integrate._ivp.common as scipy_common
import scipy.integrate._ivp.rk as scipy_rk


def _rk_step_our_order(fun, t, y, f, h, A, B, C, K):
    K[0] = f
    n_stages = len(B)
    for s in range(1, n_stages):
        acc = np.zeros_like(y)
        for j in range(s):
            if A[s, j] != 0.0:
                acc = acc + A[s, j] * K[j]
        K[s] = fun(t + C[s] * h, y + h * acc)
    acc = np.zeros_like(y)
    for j in range(n_stages):
        if B[j] != 0.0:
            acc = acc + B[j] * K[j]
    y_new = y + h * acc
    f_new = fun(t + h, y_new)
    K[-1] = f_new
    return y_new, f_new


def _sequential_sum_of_squares(x):
    acc = 0.0
    for v in np.ravel(x):
        acc += v * v
    return acc


def _rk_error_norm_our_order(self, K, h, scale):
    e = np.zeros(K.shape[1])
    for j in range(K.shape[0]):
        e = e + K[j] * self.E[j]
    return np.sqrt(_sequential_sum_of_squares(h * e / scale) / scale.size)


def _dop853_error_norm_our_order(self, K, h, scale):
    e5 = np.zeros(K.shape[1])
    e3 = np.zeros(K.shape[1])
    for j in range(K.shape[0]):
        e5 = e5 + K[j] * self.E5[j]
        e3 = e3 + K[j] * self.E3[j]
    inv = 1.0 / scale
    a = _sequential_sum_of_squares(e5 * inv)
    b = _sequential_sum_of_squares(e3 * inv)
    if a == 0 and b == 0:
        return 0.0
    return abs(h) * a / np.sqrt((a + 0.01 * b) * len(scale))


def _rms_norm_our_order(x):
    return np.sqrt(_sequential_sum_of_squares(x) / x.size)


@contextlib.contextmanager
def scipy_with_our_summation_order():
    saved = (scipy_rk.rk_step, scipy_rk.RungeKutta._estimate_error_norm,
             scipy_rk.DOP853._estimate_error_norm, scipy_common.norm)
    scipy_rk.rk_step = _rk_step_our_order
    scipy_rk.RungeKutta._estimate_error_norm = _rk_error_norm_our_order
    scipy_rk.DOP853._estimate_error_norm = _dop853_error_norm_our_order
    scipy_common.norm = _rms_norm_our_order
    try:
        yield
    finally:
        (scipy_rk.rk_step, scipy_rk.RungeKutta._estimate_error_norm,
         scipy_rk.DOP853._estimate_error_norm, scipy_common.norm) = saved


def matched_solve_ivp(*args, **kwargs):
    """:func:`scipy.integrate.solve_ivp` run under :func:`scipy_with_our_summation_order`.

    Use this, not stock SciPy, wherever a test requires *exact* agreement
    (RHS evaluation counts, step sequences). Stock SciPy's rounding depends on
    the platform (NumPy's SIMD kernels and the C runtime's ``pow``), and in
    rounding-sensitive regimes (error estimates at the noise floor, stiff
    problems on the stability boundary) its step counts then vary between
    machines. The matched run uses deterministic elementwise arithmetic in our
    order, so it is bit-identical to our engine on every platform.

    Dense output is not patched: event roots can still differ by an ulp.
    """
    from scipy.integrate import solve_ivp as scipy_solve_ivp

    with scipy_with_our_summation_order():
        return scipy_solve_ivp(*args, **kwargs)
