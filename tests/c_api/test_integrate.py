"""The GIL-free C API: ``fni_integrate`` (through :func:`integrate`) and
cross-module use (through :func:`batch.integrate_batch`).

``solve_ivp``, ``integrate`` and ``integrate_batch`` all run on the same C core,
so they must agree *bit for bit*: same steps, same RHS evaluation count, same
endpoint, same events.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fast_numerical_integration.batch import integrate_batch
from fast_numerical_integration.integrators import ComponentCrossing, integrate, solve_ivp
from fast_numerical_integration.problems import catalog, events, native

METHODS = ("RK23", "RK45", "DOP853")
PROBLEMS = ["A3", "B5", "C5", "D5", "E3", "arenstorf", "pleiades", "vdp_mu100", "robertson", "lorenz"]


# Problems that are well-posed in reverse time. Lorenz and Van der Pol are not:
# reversed, the attractor / limit cycle repels and the state escapes to
# infinity (see test_unstable_reverse_time_runs_match).
REVERSIBLE = {"A3", "B5", "C5", "D5", "E3", "arenstorf", "pleiades"}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", PROBLEMS)
@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_integrate_matches_solve_ivp_bit_for_bit(name, method, backward):
    if backward and name not in REVERSIBLE:
        pytest.skip("not well-posed backward in time")
    p = catalog.get(name)
    kw = dict(method=method, rtol=1e-7, atol=1e-10)
    if backward:
        y_end = solve_ivp(p.native(), p.t_span, p.y0, **kw).y[:, -1]
        span, y0 = (p.t_span[1], p.t_span[0]), y_end
    else:
        span, y0 = p.t_span, p.y0
    ref = solve_ivp(p.native(), span, y0, **kw)
    res = integrate(p.native(), span, y0, **kw)
    assert res.gil_released
    assert res.status == ref.status == 0
    assert res.nfev == ref.nfev
    assert res.n_accepted == ref.n_accepted and res.n_rejected == ref.n_rejected
    assert res.t == ref.t[-1]
    np.testing.assert_array_equal(res.y, ref.y[:, -1])


@pytest.mark.parametrize("method", METHODS)
def test_integrate_events_match_solve_ivp(method):
    sc = events.kepler_apsides(e=0.6, n_orbits=4)
    kw = dict(method=method, rtol=1e-9, atol=1e-12)
    ref = solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.native_events(), **kw)
    res = integrate(sc.native(), sc.t_span, sc.y0, events=sc.native_events(), **kw)
    assert res.gil_released
    assert res.n_event_records == sum(t.size for t in ref.t_events)
    for a, b, ya, yb in zip(res.t_events, ref.t_events, res.y_events, ref.y_events):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ya, yb)
    np.testing.assert_array_equal(res.y, ref.y[:, -1])


@pytest.mark.parametrize("method", METHODS)
def test_integrate_terminal_event_endpoint(method):
    p = catalog.get("D3")
    stop = native.DotProductEvent([0, 1], [2, 3], direction=-1, terminal=2)   # 2nd apoapsis
    kw = dict(method=method, rtol=1e-10, atol=1e-13)
    ref = solve_ivp(p.native(), p.t_span, p.y0, events=stop, **kw)
    res = integrate(p.native(), p.t_span, p.y0, events=stop, **kw)
    assert res.status == ref.status == 1
    assert res.t == ref.t[-1] == pytest.approx(3 * np.pi, abs=1e-7)
    np.testing.assert_array_equal(res.y, ref.y[:, -1])
    assert res.nfev == ref.nfev


def test_integrate_with_python_callables_keeps_the_gil():
    p = catalog.get("B1")

    def ev(t, y):
        return y[0] - 1.0

    res = integrate(p.rhs, p.t_span, p.y0, method="RK45", events=ev, rtol=1e-8, atol=1e-10)
    ref = solve_ivp(p.rhs, p.t_span, p.y0, method="RK45", events=ev, rtol=1e-8, atol=1e-10)
    assert not res.gil_released
    np.testing.assert_array_equal(res.y, ref.y[:, -1])
    np.testing.assert_array_equal(res.t_events[0], ref.t_events[0])
    with pytest.raises(TypeError, match="release_gil"):
        integrate(p.rhs, p.t_span, p.y0, release_gil=True)
    # A native system with a Python event also can't release the GIL.
    with pytest.raises(TypeError, match="release_gil"):
        integrate(p.native(), p.t_span, p.y0, events=ev, release_gil=True)


def test_integrate_python_exceptions_propagate():
    def bad(t, y):
        raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError, match="boom"):
        integrate(bad, (0.0, 1.0), [1.0])

    def bad_event(t, y):
        raise KeyError("event boom")

    with pytest.raises(KeyError, match="event boom"):
        integrate(native.Kepler2D(), (0.0, 1.0), [1.0, 0.0, 0.0, 1.0], events=bad_event)


def test_integrate_record_capacity():
    lin = native.LinearSystem([[0.0, 1.0], [-1.0, 0.0]])
    res = integrate(lin, (0.0, 20 * np.pi + 1.0), [0.0, 1.0], method="DOP853", rtol=1e-10,
                    atol=1e-12, events=ComponentCrossing(0), max_records=5)
    assert res.n_event_records == 21          # every zero of sin t, k*pi for k = 0..20, is counted...
    assert res.t_events[0].size == 5          # ...but only max_records are stored
    np.testing.assert_allclose(res.t_events[0], np.pi * np.arange(5), atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_unstable_reverse_time_runs_match(method):
    """Van der Pol (mu = 100) backward from its limit cycle: the cycle repels
    in reverse time. DOP853 ends with "step size too small" while RK23/RK45
    get through. Whatever the outcome, both entry points must agree exactly."""
    p = catalog.get("vdp_mu100")
    kw = dict(method=method, rtol=1e-7, atol=1e-10)
    y_end = solve_ivp(p.native(), p.t_span, p.y0, **kw).y[:, -1]
    span = (p.t_span[1], p.t_span[0])
    ref = solve_ivp(p.native(), span, y_end, **kw)
    res = integrate(p.native(), span, y_end, **kw)
    assert res.status == ref.status
    if method == "DOP853":
        assert res.status == -1
    assert res.message == ref.message
    assert res.nfev == ref.nfev
    assert res.t == ref.t[-1]
    np.testing.assert_array_equal(res.y, ref.y[:, -1])


def test_integrate_blowup_reports_failure():
    res = integrate(native.Quadratic(), (0.0, 2.0), [1.0], method="DOP853", rtol=1e-8, atol=1e-11)
    assert res.status == -1 and not res.success
    assert "step size" in res.message
    assert abs(res.t - 1.0) < 1e-6


def test_integrate_validation():
    k = native.Kepler2D()
    with pytest.raises(ValueError, match="expects 4"):
        integrate(k, (0, 1), [1.0, 0.0])
    with pytest.raises(ValueError, match="max_step"):
        integrate(k, (0, 1), [1, 0, 0, 1], max_step=0.0)
    with pytest.raises(ValueError, match="first_step"):
        integrate(k, (0, 1), [1, 0, 0, 1], first_step=5.0)
    with pytest.raises(IndexError):
        integrate(k, (0, 1), [1, 0, 0, 1], events=ComponentCrossing(9))


def test_threads_run_concurrently_and_agree():
    """Same system object from 8 threads, GIL released: identical results."""
    p = catalog.get("pleiades")
    sys = p.native()
    kw = dict(method="DOP853", rtol=1e-10, atol=1e-12)
    ref = integrate(sys, p.t_span, p.y0, **kw)
    seen = set()

    def work(_):
        seen.add(threading.get_ident())
        return integrate(sys, p.t_span, p.y0, **kw)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(32)))
    for r in results:
        assert r.gil_released
        np.testing.assert_array_equal(r.y, ref.y)
        assert r.nfev == ref.nfev
    assert len(seen) > 1


# --- batch: the C API used from another Cython module ------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_batch_matches_individual_solves(method):
    p = catalog.get("D4")
    rng = np.random.default_rng(3)
    Y0 = p.y0 + 0.01 * rng.standard_normal((12, 4))
    T1 = rng.uniform(1.0, 20.0, 12)
    out = integrate_batch(p.native(), 0.0, T1, Y0, method=method, rtol=1e-9, atol=1e-12)
    for k in range(12):
        ref = solve_ivp(p.native(), (0.0, T1[k]), Y0[k], method=method, rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(out["y"][k], ref.y[:, -1])
        assert out["nfev"][k] == ref.nfev
        assert out["t"][k] == T1[k]
    assert np.all(out["status"] == 0)


def test_batch_multiple_shooting_segments_chain():
    """Segments [t_k, t_k+1] propagated in parallel compose to the full solve."""
    p = catalog.get("arenstorf")
    nodes = np.linspace(*p.t_span, 9)
    kw = dict(rtol=1e-12, atol=1e-14)
    full = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", dense_output=True, **kw)
    Y0 = full.sol(nodes[:-1]).T
    out = integrate_batch(p.native(), nodes[:-1], nodes[1:], Y0, n_threads=4, **kw)
    defects = out["y"] - full.sol(nodes[1:]).T
    assert np.max(np.abs(defects)) < 1e-9


def test_batch_threads_agree_with_serial():
    p = catalog.get("C5")
    rng = np.random.default_rng(0)
    Y0 = p.y0 * (1 + 1e-4 * rng.standard_normal((40, 30)))
    a = integrate_batch(p.native(), 0.0, 20.0, Y0, rtol=1e-8, atol=1e-11, n_threads=1)
    b = integrate_batch(p.native(), 0.0, 20.0, Y0, rtol=1e-8, atol=1e-11, n_threads=6)
    np.testing.assert_array_equal(a["y"], b["y"])
    np.testing.assert_array_equal(a["nfev"], b["nfev"])


def test_batch_requires_native_system():
    from fast_numerical_integration.integrators import PythonOdeSystem

    with pytest.raises(TypeError, match="native"):
        integrate_batch(PythonOdeSystem(lambda t, y: -y), 0.0, 1.0, [[1.0]])
