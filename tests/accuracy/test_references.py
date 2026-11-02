"""Cross-check of the computed reference solutions against an independent method.

Problems without a closed form use a DOP853 solve at rtol=1e-13 as reference.
Comparing it against DOP853 itself would be circular, so it is checked against
SciPy's Radau (an implicit collocation method, a completely different
algorithm) and LSODA.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration.problems import catalog

# Problems that rely on the computed reference (no exact solution), with the
# interval over which an independent check is meaningful.
CHECKS = {
    "A5": None, "B1": None, "B3": None, "B4": None, "C5": None, "E2": None, "E3": None,
    "pleiades": None, "brusselator": None, "lorenz": 5.0, "robertson": None,
    "vdp_mu1": None, "vdp_mu10": None,
}


@pytest.mark.parametrize("name", list(CHECKS))
def test_reference_agrees_with_radau(name):
    p = catalog.get(name)
    t_end = CHECKS[name] or p.t_span[1]
    tq = np.linspace(p.t_span[0], t_end, 41)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        radau = scipy_solve_ivp(p.rhs, (p.t_span[0], t_end), p.y0, method="Radau",
                                rtol=1e-12, atol=1e-14, t_eval=tq)
    assert radau.success
    ref = catalog.reference(p, tq)
    scale = catalog.solution_scale(p)
    assert np.max(np.abs(ref - radau.y)) < 1e-8 * scale


@pytest.mark.parametrize("name", ["A5", "B1", "C5", "E3", "pleiades"])
def test_reference_agrees_with_lsoda(name):
    p = catalog.get(name)
    lsoda = scipy_solve_ivp(p.rhs, p.t_span, p.y0, method="LSODA", rtol=1e-12, atol=1e-14)
    assert lsoda.success
    scale = catalog.solution_scale(p)
    assert np.max(np.abs(catalog.reference_end(p) - lsoda.y[:, -1])) < 1e-7 * scale


def test_exact_solutions_satisfy_their_odes():
    """Each closed-form solution has dy/dt = f(t, y) (central differences)."""
    for p in catalog.all_problems():
        if p.exact is None:
            continue
        t1 = p.t_span[1] if not p.expect_failure else 0.9
        ts = np.linspace(p.t_span[0] + 0.1, min(t1, p.t_span[0] + 15.0) - 0.1, 7)
        h = 1e-5
        dy = (p.exact(ts + h) - p.exact(ts - h)) / (2 * h)
        f = np.column_stack([p.rhs(t, p.exact(np.array([t]))[:, 0]) for t in ts])
        scale = max(1.0, np.max(np.abs(f)))
        assert np.max(np.abs(dy - f)) < 1e-6 * scale, p.name
        np.testing.assert_allclose(p.exact(np.array([p.t_span[0]]))[:, 0], p.y0, atol=1e-12, err_msg=p.name)
