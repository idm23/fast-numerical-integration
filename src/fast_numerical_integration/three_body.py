"""Planar three-body gravitational dynamics.

State vector layout (length 12), chosen so the same ``y`` feeds both
:func:`scipy.integrate.solve_ivp` and our future hand-written integrators
unchanged::

    y = [ x1, y1,  x2, y2,  x3, y3,          # positions  (6)
          vx1, vy1, vx2, vy2, vx3, vy3 ]     # velocities (6)

The right-hand side signature mirrors ``f(t, y)`` expected by ``solve_ivp`` so it
is a drop-in RHS for any of our solvers. It is written in a form that a
SymPy -> C code printer can later replace with a compiled kernel.
"""

from dataclasses import dataclass

import numpy as np

N_BODIES = 3
N_DIMS = 2
STATE_SIZE = 2 * N_BODIES * N_DIMS  # positions + velocities = 12

# Softening length: guards the 1/r^2 term against blow-up when two bodies pass
# arbitrarily close. Tiny relative to the figure-eight length scale (~1), so it
# does not perturb the reference orbit.
_SOFTENING2 = 1e-12


def three_body_rhs(
    t: float,
    y: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
) -> np.ndarray:
    """Return dy/dt for the planar three-body problem.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous, but kept for the ``f(t, y)`` API).
    y : ndarray, shape (12,)
        Packed state; see module docstring.
    masses : ndarray, shape (3,)
        Body masses.
    G : float
        Gravitational constant (``1.0`` in the non-dimensional figure-eight units).
    """
    positions = y[: N_BODIES * N_DIMS].reshape(N_BODIES, N_DIMS)
    velocities = y[N_BODIES * N_DIMS :].reshape(N_BODIES, N_DIMS)

    # Pairwise separation vectors r_j - r_i, shape (N, N, 2).
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist2 = np.sum(diff * diff, axis=-1) + _SOFTENING2
    inv_dist3 = dist2 ** -1.5
    np.fill_diagonal(inv_dist3, 0.0)  # no self-interaction

    # a_i = G * sum_j m_j (r_j - r_i) / |r_j - r_i|^3
    accel = G * np.einsum("ij,ijk,j->ik", inv_dist3, diff, masses)

    dydt = np.empty_like(y)
    dydt[: N_BODIES * N_DIMS] = velocities.reshape(-1)
    dydt[N_BODIES * N_DIMS :] = accel.reshape(-1)
    return dydt


def total_energy(y: np.ndarray, masses: np.ndarray, G: float = 1.0) -> float:
    """Total mechanical energy (kinetic + gravitational potential).

    Conserved by the exact dynamics; used to gauge integrator accuracy.
    """
    positions = y[: N_BODIES * N_DIMS].reshape(N_BODIES, N_DIMS)
    velocities = y[N_BODIES * N_DIMS :].reshape(N_BODIES, N_DIMS)

    kinetic = 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=-1))

    potential = 0.0
    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            r = np.linalg.norm(positions[j] - positions[i])
            potential -= G * masses[i] * masses[j] / r
    return float(kinetic + potential)


def figure_eight_initial_conditions() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Classic Chenciner-Montgomery figure-eight choreography.

    Three equal unit masses chase each other around a single figure-eight curve.
    A stable, periodic, reproducible orbit -- ideal as a correctness reference
    and timing target.

    Returns
    -------
    y0 : ndarray, shape (12,)
        Initial packed state.
    masses : ndarray, shape (3,)
        Unit masses.
    G : float
        Gravitational constant (1.0).
    period : float
        Orbital period T.
    """
    # Positions: body 3 is the mirror image of body 1; body 2 sits at the origin.
    r1 = np.array([0.97000436, -0.24308753])
    r2 = np.array([0.0, 0.0])
    r3 = -r1

    # Velocities: v1 = v3 = -v2 / 2 keeps the centre of mass fixed at the origin.
    v2 = np.array([-0.93240737, -0.86473146])
    v1 = -v2 / 2.0
    v3 = v1

    y0 = np.concatenate([r1, r2, r3, v1, v2, v3])
    masses = np.ones(N_BODIES)
    G = 1.0
    period = 6.32591398  # T for these non-dimensional units
    return y0, masses, G, period


@dataclass
class ThreeBodySystem:
    """Bundle of everything needed to integrate a three-body problem.

    Shared by :mod:`main`, the future benchmark harness, and custom-engine tests
    so the physics is defined exactly once.
    """

    masses: np.ndarray
    G: float
    y0: np.ndarray
    period: float
    name: str = "three-body"

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """``f(t, y)`` closure ready for ``solve_ivp`` or our own integrators."""
        return three_body_rhs(t, y, self.masses, self.G)

    def energy(self, y: np.ndarray) -> float:
        """Total mechanical energy of state ``y``."""
        return total_energy(y, self.masses, self.G)


def figure_eight_system() -> ThreeBodySystem:
    """Return a :class:`ThreeBodySystem` for the figure-eight choreography."""
    y0, masses, G, period = figure_eight_initial_conditions()
    return ThreeBodySystem(
        masses=masses, G=G, y0=y0, period=period, name="figure-eight"
    )
