"""The decoupled tangent propagator (``integrate_tangent``).

``y`` must be integrated exactly as without the tangent, and ``X`` must solve the
variational equation ``X' = J(t, y(t)) X`` to the tangent's own tolerance, across
resets through the tangent reset.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from fast_numerical_integration.integrators import integrate, integrate_hybrid, integrate_tangent


def damped(t, y):
    return np.array([y[1], -4.0 * y[0] - 0.3 * y[1]])


A = np.array([[0.0, 1.0], [-4.0, -0.3]])


def test_linear_system_gives_the_matrix_exponential():
    r = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A, rtol=1e-10,
                          atol=1e-12, tangent_rtol=1e-8, tangent_atol=1e-10)
    assert r.success
    assert np.abs(r.X - expm(3.0 * A)).max() < 1e-7
    # Two Jacobians per piece, one at the start.
    assert r.n_jac == 2 * r.n_tangent_pieces + 1


def test_the_state_is_integrated_exactly_as_without_the_tangent():
    for method in ("RK45", "DOP853"):
        plain = integrate(damped, (0.0, 3.0), [1.0, 0.0], method=method, rtol=1e-9, atol=1e-11)
        with_x = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A,
                                   method=method, rtol=1e-9, atol=1e-11)
        assert np.array_equal(plain.y, with_x.y)
        assert plain.n_accepted == with_x.n_accepted
        assert plain.n_rejected == with_x.n_rejected


def vdp(t, y, mu=2.0):
    return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])


def vdp_jac(t, y, mu=2.0):
    return np.array([[0.0, 1.0], [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)]])


def flow_jacobian_fd(fun, t1, y0, h=1e-6, **kw):
    """The flow's Jacobian by central differences of tight integrations."""
    y0 = np.asarray(y0, float)
    cols = []
    for j in range(y0.size):
        e = np.zeros_like(y0)
        e[j] = h
        yp = integrate(fun, (0.0, t1), y0 + e, method="DOP853", rtol=1e-13, atol=1e-13, **kw).y
        ym = integrate(fun, (0.0, t1), y0 - e, method="DOP853", rtol=1e-13, atol=1e-13, **kw).y
        cols.append((yp - ym) / (2 * h))
    return np.column_stack(cols)


def test_nonlinear_flow_matches_finite_differences():
    y0 = [2.0, 0.0]
    fd = flow_jacobian_fd(vdp, 4.0, y0)
    r = integrate_tangent(vdp, (0.0, 4.0), y0, vdp_jac, rtol=1e-10, atol=1e-12,
                          tangent_rtol=1e-9, tangent_atol=1e-11, max_depth=6)
    assert r.success
    assert np.abs(r.X - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())


def test_a_tangent_of_several_columns_and_a_starting_value():
    X0 = np.array([[1.0], [0.5]])
    r = integrate_tangent(damped, (0.0, 2.0), [1.0, 0.0], lambda t, y: A, X0=X0,
                          rtol=1e-10, atol=1e-12)
    assert r.X.shape == (2, 1)
    assert np.abs(r.X - expm(2.0 * A) @ X0).max() < 1e-7


def test_the_tolerance_controls_the_pieces():
    kw = dict(rtol=1e-10, atol=1e-12, tangent_atol=0.0)
    loose = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A,
                              tangent_rtol=1e-4, **kw)
    tight = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A,
                              tangent_rtol=1e-9, max_depth=8, **kw)
    exact = expm(3.0 * A)
    assert tight.n_tangent_pieces > loose.n_tangent_pieces
    assert np.abs(tight.X - exact).max() < np.abs(loose.X - exact).max()
    once = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A, max_depth=0, **kw)
    assert once.n_tangent_pieces == once.n_accepted


def test_a_sparsity_pattern_gives_the_same_tangent():
    dense = integrate_tangent(vdp, (0.0, 2.0), [2.0, 0.0], vdp_jac, rtol=1e-9, atol=1e-11)
    pattern = np.array([[False, True], [True, True]])
    sparse = integrate_tangent(vdp, (0.0, 2.0), [2.0, 0.0], vdp_jac, rtol=1e-9, atol=1e-11,
                               sparsity=pattern)
    assert np.array_equal(dense.y, sparse.y)
    assert np.abs(dense.X - sparse.X).max() < 1e-14 * np.abs(dense.X).max()


# A ball under gravity with a restitution e: the guard h = 0 (falling) resets
# v <- -e v. Across the reset the flow's Jacobian jumps by the saltation matrix
# S = R + (f+ - R f-) grad(g)^T / (grad(g) . f-), with R = diag(1, -e).
G, E = 9.81, 0.8


def ball(t, y):
    return np.array([y[1], -G])


def ball_jac(t, y):
    return np.array([[0.0, 1.0], [0.0, 0.0]])


def ground(t, y):
    return y[0]


ground.terminal = True
ground.direction = -1


def bounce(event, t, y):
    return np.array([y[0], -E * y[1]])


def saltation(event, t, y_before, y_after, X):
    R = np.diag([1.0, -E])
    f_minus, f_plus = ball(t, y_before), ball(t, y_after)
    grad = np.array([1.0, 0.0])
    S = R + np.outer(f_plus - R @ f_minus, grad) / (grad @ f_minus)
    return S @ X


def test_resets_carry_the_tangent_through_the_saltation_matrix():
    y0 = np.array([1.0, 0.5])
    t1 = 1.6
    kw = dict(rtol=1e-12, atol=1e-12, max_resets=10)
    r = integrate_tangent(ball, (0.0, t1), y0, ball_jac, events=[ground], reset=bounce,
                          tangent_reset=saltation, **kw)
    assert r.success and r.n_resets >= 1
    plain = integrate_hybrid(ball, (0.0, t1), y0, [ground], bounce, method="DOP853", **kw)
    assert np.array_equal(plain.y, r.y)
    # The flow's Jacobian across the bounce, by central differences.
    h = 1e-6
    fd = np.column_stack([
        (integrate_hybrid(ball, (0.0, t1), y0 + h * e, [ground], bounce, method="DOP853", **kw).y
         - integrate_hybrid(ball, (0.0, t1), y0 - h * e, [ground], bounce, method="DOP853",
                            **kw).y) / (2 * h)
        for e in np.eye(2)])
    assert np.abs(r.X - fd).max() < 1e-6
    # Without the tangent reset X is continuous across the bounce, which is wrong here.
    no_reset = integrate_tangent(ball, (0.0, t1), y0, ball_jac, events=[ground], reset=bounce,
                                 **kw)
    assert np.abs(no_reset.X - fd).max() > 1e-2


def test_a_terminal_event_without_a_reset_ends_the_tangent_at_the_event():
    r = integrate_tangent(ball, (0.0, 5.0), [1.0, 0.0], ball_jac, events=[ground],
                          rtol=1e-12, atol=1e-12)
    assert r.status == 1
    # Free fall: X(t) = [[1, t], [0, 1]] at the event time.
    assert np.abs(r.X - np.array([[1.0, r.t], [0.0, 1.0]])).max() < 1e-10


def test_bad_arguments():
    with pytest.raises(ValueError):
        integrate_tangent(damped, (0.0, 1.0), [1.0, 0.0], lambda t, y: A, X0=np.eye(3))
    with pytest.raises(ValueError):
        integrate_tangent(damped, (0.0, 1.0), [1.0, 0.0], lambda t, y: A, sparsity=np.eye(3))
    with pytest.raises(ValueError):
        integrate_tangent(damped, (0.0, 1.0), [1.0, 0.0], lambda t, y: np.eye(3))


def test_native_callbacks_run_without_the_gil():
    from fast_numerical_integration.problems import native

    system = native.LinearSystem(A)
    r = integrate_tangent(system, (0.0, 3.0), [1.0, 0.0], native.LinearJacobian(A),
                          rtol=1e-10, atol=1e-12)
    assert r.gil_released
    assert np.abs(r.X - expm(3.0 * A)).max() < 1e-7
    with pytest.raises(TypeError):
        integrate_tangent(system, (0.0, 3.0), [1.0, 0.0], lambda t, y: A, release_gil=True)


def test_sixth_order_is_more_accurate_per_piece():
    exact = expm(3.0 * A)
    kw = dict(rtol=1e-10, atol=1e-12, max_depth=0)
    four = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A, tangent_order=4, **kw)
    six = integrate_tangent(damped, (0.0, 3.0), [1.0, 0.0], lambda t, y: A, tangent_order=6, **kw)
    assert four.n_tangent_pieces == six.n_tangent_pieces == six.n_accepted
    # Four new Jacobians per piece against two.
    assert six.n_jac == 4 * six.n_tangent_pieces + 1
    assert np.abs(six.X - exact).max() < 0.05 * np.abs(four.X - exact).max()
    # On the nonlinear flow and across a reset too.
    fd = flow_jacobian_fd(vdp, 4.0, [2.0, 0.0])
    r = integrate_tangent(vdp, (0.0, 4.0), [2.0, 0.0], vdp_jac, rtol=1e-10, atol=1e-12,
                          tangent_order=6, tangent_rtol=1e-9, tangent_atol=1e-11)
    assert np.abs(r.X - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())
    with pytest.raises(ValueError):
        integrate_tangent(damped, (0.0, 1.0), [1.0, 0.0], lambda t, y: A, tangent_order=5)
