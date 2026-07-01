"""Smoke tests for the three-body baseline.

These guard the RHS and initial conditions, and double as a correctness oracle
for the hand-written integrators we'll add later: any custom solver should match
the SciPy reference on energy conservation and boundedness.
"""

import numpy as np
from scipy.integrate import solve_ivp

from fast_numerical_integration.three_body import (
    figure_eight_initial_conditions,
    figure_eight_system,
    three_body_rhs,
)


def _solve_one_period():
    system = figure_eight_system()
    sol = solve_ivp(
        system.rhs,
        (0.0, system.period),
        system.y0,
        method="DOP853",
        t_eval=np.linspace(0.0, system.period, 500),
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success, sol.message
    return system, sol


def test_center_of_mass_starts_at_origin():
    y0, masses, _, _ = figure_eight_initial_conditions()
    positions = y0[:6].reshape(3, 2)
    velocities = y0[6:].reshape(3, 2)
    com_pos = np.average(positions, axis=0, weights=masses)
    com_vel = np.average(velocities, axis=0, weights=masses)
    assert np.allclose(com_pos, 0.0, atol=1e-12)
    assert np.allclose(com_vel, 0.0, atol=1e-12)


def test_energy_conserved_over_one_period():
    system, sol = _solve_one_period()
    energies = np.array([system.energy(sol.y[:, i]) for i in range(sol.y.shape[1])])
    drift = np.max(np.abs(energies - energies[0]))
    assert drift < 1e-6, f"energy drift too large: {drift:.2e}"


def test_trajectories_stay_bounded():
    _, sol = _solve_one_period()
    positions = sol.y[:6]
    assert np.all(np.abs(positions) < 2.0), "figure-eight orbit escaped expected box"


def test_figure_eight_is_periodic():
    system, sol = _solve_one_period()
    # After exactly one period the choreography returns near its start.
    assert np.allclose(sol.y[:, -1], system.y0, atol=1e-2)


def test_rhs_shape_and_velocity_block():
    y0, masses, G, _ = figure_eight_initial_conditions()
    dydt = three_body_rhs(0.0, y0, masses, G)
    assert dydt.shape == y0.shape
    # First half of dy/dt is exactly the velocity block of the state.
    assert np.array_equal(dydt[:6], y0[6:])
