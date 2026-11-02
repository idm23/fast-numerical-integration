"""Runtime parameters on native systems and events.

Changing ``system.params`` between solves must give exactly the result of a
freshly constructed system with those values. This is what continuation
relies on: one compiled kernel, many parameter values, no recompilation.
"""

import numpy as np
import pytest

from fast_numerical_integration.integrators import ComponentCrossing, solve_ivp
from fast_numerical_integration.nbody import NBodyGravity
from fast_numerical_integration.problems import native

KW = dict(method="DOP853", rtol=1e-10, atol=1e-12)


def _same(a, b):
    assert a.nfev == b.nfev
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.y, b.y)


@pytest.mark.parametrize("factory,values,y0,t_end", [
    (native.VanDerPol, [7.5], [2.0, 0.0], 10.0),
    (native.Kepler2D, [2.3], [1.0, 0.0, 0.0, 1.2], 6.0),
    (native.Lorenz, [9.0, 25.0, 2.5], [-8.0, 8.0, 27.0], 3.0),
    (native.Brusselator, [1.2, 2.5], [1.5, 3.0], 10.0),
    (native.Arenstorf, [0.02], [0.994, 0.0, 0.0, -2.0], 2.0),
    (native.Pendulum, [4.0], [1.0, 0.0], 5.0),
    (native.EulerRigidBody, [1.0, -0.5, -0.8], [0.0, 1.0, 1.0], 8.0),
    (native.ProtheroRobinson, [-50.0], [1.0], 2.0),
])
def test_changing_params_equals_constructing_with_them(factory, values, y0, t_end):
    fresh = factory(*values)
    reused = factory()  if factory is not native.ProtheroRobinson else factory(-1.0)
    reused.params = values
    _same(solve_ivp(reused, (0.0, t_end), y0, **KW), solve_ivp(fresh, (0.0, t_end), y0, **KW))
    np.testing.assert_array_equal(reused.params, values)


def test_params_metadata_and_validation():
    vdp = native.VanDerPol(3.0)
    assert vdp.n_params == 1
    assert vdp.param_names == ("mu",)
    np.testing.assert_array_equal(vdp.params, [3.0])
    with pytest.raises(ValueError, match="1 parameters"):
        vdp.params = [1.0, 2.0]
    lor = native.Lorenz()
    assert lor.param_names == ("sigma", "rho", "beta")
    assert native.Robertson().n_params == 0
    # params returns a copy: mutating it does not change the system
    p = vdp.params
    p[0] = 99.0
    assert vdp.params[0] == 3.0


def test_continuation_sweep_reuses_one_object():
    """Van der Pol mu: 1 -> 20, one compiled kernel, matching fresh objects."""
    vdp = native.VanDerPol()
    for mu in (1.0, 2.0, 5.0, 10.0, 20.0):
        vdp.params = [mu]
        a = solve_ivp(vdp, (0.0, 5.0), [2.0, 0.0], **KW)
        b = solve_ivp(native.VanDerPol(mu), (0.0, 5.0), [2.0, 0.0], **KW)
        _same(a, b)


def test_nbody_params_are_G_and_masses():
    sys = NBodyGravity([1.0, 2.0, 3.0], G=1.5, n_dims=2)
    assert sys.param_names == ("G", "m1", "m2", "m3")
    np.testing.assert_array_equal(sys.params, [1.5, 1.0, 2.0, 3.0])
    assert sys.G == 1.5
    np.testing.assert_array_equal(sys.masses_array, [1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0])
    f1 = sys(0.0, y)
    sys.params = [3.0, 2.0, 4.0, 6.0]      # doubles G and every mass
    np.testing.assert_allclose(sys(0.0, y)[6:], 4.0 * f1[6:], rtol=1e-14)


def test_event_params_move_the_level():
    ev = ComponentCrossing(0, 0.5)
    assert ev.param_names == ("value",)
    lin = native.LinearSystem([[0.0]], b=[1.0])     # y = t
    assert solve_ivp(lin, (0.0, 2.0), [0.0], events=ev).t_events[0][0] == pytest.approx(0.5)
    ev.params = [1.25]
    assert ev.value == 1.25
    assert solve_ivp(lin, (0.0, 2.0), [0.0], events=ev).t_events[0][0] == pytest.approx(1.25)
    te = native.TimeEvent(0.3)
    te.params = [0.7]
    assert solve_ivp(lin, (0.0, 2.0), [0.0], events=te).t_events[0][0] == pytest.approx(0.7)


def test_args_with_native_system_points_to_params():
    with pytest.raises(TypeError, match="params"):
        solve_ivp(native.VanDerPol(), (0, 1), [2.0, 0.0], args=(3.0,))


def test_every_native_system_is_gil_free():
    from fast_numerical_integration.problems import catalog

    for p in catalog.all_problems():
        assert p.native().has_native_fn, p.name
    assert ComponentCrossing(0).has_native_fn
    assert native.DotProductEvent([0], [1]).has_native_fn
    assert native.TimeEvent(1.0).has_native_fn
