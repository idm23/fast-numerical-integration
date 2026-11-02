"""Chaotic dynamics: the Lorenz system.

Trajectories separate at the rate of the largest Lyapunov exponent
(lambda_max ~ 0.906 for sigma=10, rho=28, beta=8/3). An integrator can only be
accurate for a time of order ``log(1 / tol) / lambda_max``. After that the
meaningful checks are statistical: the solution must stay on the attractor.
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog
from fast_numerical_integration.problems.native import Lorenz

LYAPUNOV = 0.906


def test_divergence_rate_estimates_the_lyapunov_exponent():
    p = catalog.get("lorenz")
    tq = np.linspace(0.0, 20.0, 401)
    a = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=1e-12, atol=1e-14, t_eval=tq)
    b = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", rtol=1e-8, atol=1e-10, t_eval=tq)
    d = np.linalg.norm(a.y - b.y, axis=0)
    window = (tq > 3.0) & (d < 1.0)
    rate = np.polyfit(tq[window], np.log(d[window]), 1)[0]
    assert rate == pytest.approx(LYAPUNOV, abs=0.25)


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
@pytest.mark.parametrize("rtol", [1e-6, 1e-9])
def test_error_stays_inside_the_lyapunov_envelope(method, rtol):
    """Global error grows no faster than ~rtol * exp(lambda_max t)."""
    p = catalog.get("lorenz")
    tq = np.linspace(0.0, 10.0, 101)
    sol = solve_ivp(p.native(), (0.0, 10.0), p.y0, method=method, rtol=rtol, atol=rtol * 1e-3, t_eval=tq)
    ref = catalog.reference(p, tq)
    err = np.max(np.abs(sol.y - ref), axis=0)
    envelope = rtol * np.max(np.abs(ref)) * np.exp(LYAPUNOV * tq)
    assert np.max(err / envelope) < 1e3


@pytest.mark.parametrize("method", ["RK23", "RK45", "DOP853"])
def test_stays_on_the_attractor(method):
    sol = solve_ivp(Lorenz(), (0.0, 200.0), [-8.0, 8.0, 27.0], method=method, rtol=1e-6, atol=1e-9)
    assert sol.success
    x, y, z = sol.y[:, sol.t > 1.0]
    # Bounding box of the Lorenz attractor (rho=28).
    assert np.abs(x).max() < 25 and np.abs(y).max() < 30 and 0 < z.min() and z.max() < 55
    # Both lobes are visited.
    assert (x > 5).any() and (x < -5).any()
    # Long-time mean of z is ~ rho - 1 = 27 to within a few units.
    assert 21 < z.mean() < 27
