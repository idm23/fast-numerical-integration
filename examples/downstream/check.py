"""Verify the downstream extension against fast-numerical-integration."""

import numpy as np

from downstream_kernels.osc import Oscillator, endpoint
from fast_numerical_integration.integrators import integrate, solve_ivp

osc = Oscillator(4.0)
x, v, nfev = endpoint(4.0, 3.0)
ref = solve_ivp(osc, (0.0, 3.0), [1.0, 0.0], method="DOP853", rtol=1e-10, atol=1e-12)
assert (x, v) == tuple(ref.y[:, -1]), "C-API endpoint differs from solve_ivp"
assert nfev == ref.nfev
assert abs(x - np.cos(6.0)) < 1e-9
assert integrate(osc, (0.0, 3.0), [1.0, 0.0]).gil_released
osc.params = [9.0]
assert abs(solve_ivp(osc, (0.0, 1.0), [1.0, 0.0], rtol=1e-10, atol=1e-12).y[0, -1] - np.cos(3.0)) < 1e-8
print("downstream cimport of the C API: OK")
