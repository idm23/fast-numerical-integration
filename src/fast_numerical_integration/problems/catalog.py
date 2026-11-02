"""Canonical initial-value test problems with references, invariants and plot specs.

Every problem has a NumPy right-hand side (used by SciPy and as an independent
check of the native code), a factory for the equivalent native
:class:`~fast_numerical_integration.integrators.OdeSystem`, and where possible
an exact solution or a published reference value.

Sources
-------
DETEST (A1-E5)
    T. E. Hull, W. H. Enright, B. M. Fellen and A. E. Sedgwick, "Comparing
    numerical methods for ordinary differential equations", SIAM J. Numer.
    Anal. 9 (1972). Definitions and the reference end values ``y_end_ref``
    follow the Enright & Pryce (1987) version, as implemented in J. Burkardt's
    TEST_ODE (``test_ode.f90``, MIT licence).
Hairer-Norsett-Wanner
    E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential
    Equations I", 2nd ed. (Arenstorf orbit, Pleiades, Brusselator, Lorenz,
    Van der Pol) and vol. II (Robertson).
Geometric / long-time
    E. Hairer, C. Lubich, G. Wanner, "Geometric Numerical Integration": Kepler
    problem over many periods and the outer solar system (here DETEST C5's
    data integrated for 2e5 days).
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Mapping

import numpy as np
from scipy.linalg import expm
from scipy.special import ellipj

from fast_numerical_integration.integrators import OdeSystem, solve_ivp
from fast_numerical_integration.nbody import NBodyGravity
from fast_numerical_integration.problems import native
from fast_numerical_integration.three_body import figure_eight_system

GROUPS = ("detest", "hairer", "long_time", "stability")


@dataclass(frozen=True, eq=False)
class Problem:
    """An initial-value problem with everything needed to test and benchmark it.

    Attributes
    ----------
    rhs : callable
        NumPy ``f(t, y)``.
    native : callable
        Zero-argument factory returning the equivalent native ``OdeSystem``.
    exact : callable, optional
        ``exact(t) -> (n, m)`` for 1-D ``t``; the solution in closed form.
    y_end_ref : ndarray, optional
        Published reference value at ``t_span[1]``.
    invariants : mapping
        ``name -> g(y)`` with ``y`` of shape (n, m); quantities that the exact
        flow conserves.
    plot, plot_pairs, labels, plot_opts
        How :mod:`fast_numerical_integration.suite` draws the solution.
    expect_failure : bool
        The solution blows up inside ``t_span``; integrators should stop with
        status -1.
    """

    name: str
    title: str
    group: str
    rhs: Callable
    native: Callable[[], OdeSystem]
    y0: np.ndarray
    t_span: tuple
    exact: Callable | None = None
    y_end_ref: np.ndarray | None = None
    ref_source: str = ""
    invariants: Mapping[str, Callable] = field(default_factory=dict)
    plot: str = "timeseries"
    plot_pairs: tuple = ()
    labels: tuple = ()
    plot_opts: Mapping = field(default_factory=dict)
    stiff: bool = False
    expect_failure: bool = False
    notes: str = ""

    @property
    def n(self) -> int:
        return self.y0.size

    def with_span(self, t_end: float, name: str | None = None) -> "Problem":
        """Copy of this problem integrated to ``t_end`` (published end value dropped)."""
        from dataclasses import replace

        return replace(self, name=name or self.name, t_span=(self.t_span[0], t_end), y_end_ref=None)


# =============================================================================
# Helpers for exact solutions and invariants
# =============================================================================

def _linear_exact(A, y0, t0=0.0):
    A = np.asarray(A, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    def exact(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.column_stack([expm(A * (ti - t0)) @ y0 for ti in t])

    return exact


def kepler_exact(e: float):
    """Exact solution of the planar Kepler problem (mu = 1, a = 1) starting at
    periapsis ``(1 - e, 0)`` with speed ``sqrt((1 + e) / (1 - e))``."""
    b = np.sqrt(1.0 - e * e)

    def exact(t):
        M = np.atleast_1d(np.asarray(t, dtype=float))
        E = M + e * np.sin(M)
        for _ in range(60):
            E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        Edot = 1.0 / (1.0 - e * np.cos(E))
        return np.vstack([np.cos(E) - e, b * np.sin(E), -np.sin(E) * Edot, b * np.cos(E) * Edot])

    return exact


def kepler_y0(e: float) -> np.ndarray:
    return np.array([1.0 - e, 0.0, 0.0, np.sqrt((1.0 + e) / (1.0 - e))])


def kepler_invariants():
    def energy(y):
        return 0.5 * (y[2] ** 2 + y[3] ** 2) - 1.0 / np.hypot(y[0], y[1])

    def angular_momentum(y):
        return y[0] * y[3] - y[1] * y[2]

    def runge_lenz(y):
        r = np.hypot(y[0], y[1])
        L = y[0] * y[3] - y[1] * y[2]
        ax = y[3] * L - y[0] / r
        ay = -y[2] * L - y[1] / r
        return np.hypot(ax, ay)

    return {"energy": energy, "angular momentum": angular_momentum, "Runge-Lenz |A|": runge_lenz}


def _kepler_rhs(t, y):
    r3 = np.hypot(y[0], y[1]) ** 3
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])


def nbody_rhs(masses, G, n_dims):
    masses = np.asarray(masses, dtype=float)
    N = masses.size

    def rhs(t, y):
        pos = y[: N * n_dims].reshape(N, n_dims)
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        d2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(d2, 1.0)
        inv3 = d2 ** -1.5
        np.fill_diagonal(inv3, 0.0)
        acc = G * np.einsum("ij,ijk,j->ik", inv3, diff, masses)
        return np.concatenate([y[N * n_dims:], acc.reshape(-1)])

    return rhs


def nbody_invariants(masses, G, n_dims):
    masses = np.asarray(masses, dtype=float)
    N = masses.size

    def split(y):
        pos = y[: N * n_dims].reshape(N, n_dims, -1)
        vel = y[N * n_dims:].reshape(N, n_dims, -1)
        return pos, vel

    def energy(y):
        pos, vel = split(y)
        kin = 0.5 * np.einsum("i,idm->m", masses, vel * vel)
        pot = np.zeros(y.shape[1])
        for i in range(N):
            for j in range(i + 1, N):
                pot -= G * masses[i] * masses[j] / np.linalg.norm(pos[j] - pos[i], axis=0)
        return kin + pot

    def angular_momentum(y):
        pos, vel = split(y)
        if n_dims == 2:
            return np.einsum("i,im->m", masses, pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
        L = np.einsum("i,idm->dm", masses, np.cross(pos, vel, axis=1))
        return np.linalg.norm(L, axis=0)

    return {"energy": energy, "angular momentum": angular_momentum}


# =============================================================================
# DETEST (Hull et al. 1972; Enright & Pryce 1987)
# =============================================================================

DETEST_SOURCE = "DETEST (Hull et al. 1972; Enright & Pryce 1987 via Burkardt TEST_ODE)"


def _detest_A():
    probs = []
    probs.append(Problem(
        "A1", "Exponential decay", "detest", lambda t, y: -y,
        lambda: native.LinearSystem([[-1.0]]), np.array([1.0]), (0.0, 20.0),
        exact=lambda t: np.exp(-np.atleast_1d(t))[None, :],
        y_end_ref=np.array([2.061153353012535e-09]), ref_source=DETEST_SOURCE,
        plot_opts={"logy": True}))
    probs.append(Problem(
        "A2", "y' = -y^3 / 2", "detest", lambda t, y: -0.5 * y ** 3,
        native.DetestA2, np.array([1.0]), (0.0, 20.0),
        exact=lambda t: (1.0 / np.sqrt(np.atleast_1d(t) + 1.0))[None, :],
        y_end_ref=np.array([0.2182178902359887]), ref_source=DETEST_SOURCE))
    probs.append(Problem(
        "A3", "y' = y cos t", "detest", lambda t, y: y * np.cos(t),
        native.DetestA3, np.array([1.0]), (0.0, 20.0),
        exact=lambda t: np.exp(np.sin(np.atleast_1d(t)))[None, :],
        y_end_ref=np.array([2.491650271850414]), ref_source=DETEST_SOURCE))
    probs.append(Problem(
        "A4", "Logistic growth", "detest", lambda t, y: y * (20.0 - y) / 80.0,
        native.DetestA4, np.array([1.0]), (0.0, 20.0),
        exact=lambda t: (20.0 / (1.0 + 19.0 * np.exp(-np.atleast_1d(t) / 4.0)))[None, :],
        y_end_ref=np.array([17.73016648131483]), ref_source=DETEST_SOURCE))
    probs.append(Problem(
        "A5", "y' = (y - t) / (y + t)", "detest", lambda t, y: (y - t) / (y + t),
        native.DetestA5, np.array([4.0]), (0.0, 20.0),
        y_end_ref=np.array([-0.788782668896419]), ref_source=DETEST_SOURCE))
    return probs


def _detest_B():
    probs = []

    def b1(t, y):
        return np.array([2.0 * y[0] * (1.0 - y[1]), -y[1] * (1.0 - y[0])])

    probs.append(Problem(
        "B1", "Lotka-Volterra", "detest", b1, native.DetestB1, np.array([1.0, 3.0]), (0.0, 20.0),
        y_end_ref=np.array([0.6761876008576667, 0.1860816099640036]), ref_source=DETEST_SOURCE,
        invariants={"H = y1 - ln y1 + 2(y2 - ln y2)":
                    lambda y: y[0] - np.log(y[0]) + 2.0 * (y[1] - np.log(y[1]))},
        plot="phase", plot_pairs=((0, 1),), labels=("prey", "predator")))

    A_b2 = np.array([[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]])
    probs.append(Problem(
        "B2", "Linear 3x3 chain", "detest", lambda t, y: A_b2 @ y,
        lambda: native.LinearSystem(A_b2), np.array([2.0, 0.0, 1.0]), (0.0, 20.0),
        exact=_linear_exact(A_b2, [2.0, 0.0, 1.0]),
        y_end_ref=np.array([1.000000001030576, 1.000000000000000, 0.9999999989694235]),
        ref_source=DETEST_SOURCE, invariants={"sum y": lambda y: y.sum(axis=0)}))

    def b3(t, y):
        return np.array([-y[0], y[0] - y[1] ** 2, y[1] ** 2])

    probs.append(Problem(
        "B3", "Nonlinear chain", "detest", b3, native.DetestB3, np.array([1.0, 0.0, 0.0]), (0.0, 20.0),
        y_end_ref=np.array([2.061153488557776e-09, 5.257228022048349e-02, 9.474277177183630e-01]),
        ref_source=DETEST_SOURCE, invariants={"sum y": lambda y: y.sum(axis=0)}))

    def b4(t, y):
        norm = np.hypot(y[0], y[1])
        return np.array([-y[1] - y[0] * y[2] / norm, y[0] - y[1] * y[2] / norm, y[0] / norm])

    probs.append(Problem(
        "B4", "Spiral to a limit cycle", "detest", b4, native.DetestB4, np.array([3.0, 0.0, 0.0]), (0.0, 20.0),
        y_end_ref=np.array([9.826950928005993e-01, 2.198447081694832e00, 9.129452507276399e-01]),
        ref_source=DETEST_SOURCE, plot="phase", plot_pairs=((0, 1),)))

    def b5(t, y):
        return np.array([y[1] * y[2], -y[0] * y[2], -0.51 * y[0] * y[1]])

    def b5_exact(t):
        sn, cn, dn, _ = ellipj(np.atleast_1d(t), 0.51)
        return np.vstack([sn, cn, dn])

    probs.append(Problem(
        "B5", "Euler rigid body (Jacobi elliptic)", "detest", b5, native.EulerRigidBody,
        np.array([0.0, 1.0, 1.0]), (0.0, 20.0), exact=b5_exact,
        y_end_ref=np.array([-9.396570798729192e-01, -3.421177754000779e-01, 7.414126596199957e-01]),
        ref_source=DETEST_SOURCE, labels=("sn", "cn", "dn"),
        invariants={"y1^2 + y2^2": lambda y: y[0] ** 2 + y[1] ** 2,
                    "0.51 y1^2 + y3^2": lambda y: 0.51 * y[0] ** 2 + y[2] ** 2}))
    return probs


def _tridiagonal(n):
    return -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)


# Burkardt's C4 values at t = 20 (51 components).
_C4_END = np.array([
    3.124111453721466e-03, 6.015416842150318e-03, 8.470021834842650e-03, 1.033682931733337e-02,
    1.153249572873923e-02, 1.204549525737964e-02, 1.192957068015293e-02, 1.128883207111195e-02,
    1.025804501391024e-02, 8.982017581934167e-03, 7.597500902492453e-03, 6.219920556824985e-03,
    4.935916341009131e-03, 3.801432544256119e-03, 2.844213677587894e-03, 2.069123394222672e-03,
    1.464687282843915e-03, 1.009545263941126e-03, 6.779354330227017e-04, 4.437815269118510e-04,
    2.833264542938954e-04, 1.765005798796805e-04, 1.073342592697238e-04, 6.374497601777217e-05,
    3.698645309704183e-05, 2.097466832643746e-05, 1.162956710412555e-05, 6.306710405783322e-06,
    3.346286430868515e-06, 1.737760074184334e-06, 8.835366904275847e-07, 4.399520411127637e-07,
    2.146181897152360e-07, 1.025981211654928e-07, 4.807864068784215e-08, 2.209175152474847e-08,
    9.956251263138180e-09, 4.402193653748924e-09, 1.910149382204028e-09, 8.135892921473050e-10,
    3.402477118549235e-10, 1.397485617545782e-10, 5.638575303049199e-11, 2.235459707956947e-11,
    8.710498036398032e-12, 3.336554275346643e-12, 1.256679567784939e-12, 4.654359053128788e-13,
    1.693559145599857e-13, 5.996593816663054e-14, 1.891330702629865e-14,
])

# DETEST C5: Jupiter, Saturn, Uranus, Neptune, Pluto around the Sun (+ inner
# planets folded into m0). Time unit 100 days, so k2 = G * 1e4.
C5_K2 = 2.95912208286
C5_M0 = 1.00000597682
C5_MASSES = np.array([0.954786104043e-03, 0.285583733151e-03, 0.437273164546e-04,
                      0.517759138449e-04, 0.277777777778e-05])
C5_Y0 = np.array([
    3.42947415189, 3.35386959711, 1.35494901715,
    6.64145542550, 5.97156957878, 2.18231499728,
    11.2630437207, 14.6952576794, 6.27960525067,
    -30.1552268759, 1.65699966404, 1.43785752721,
    -21.1238353380, 28.4465098142, 15.3882659679,
    -0.557160570446, 0.505696783289, 0.230578543901,
    -0.415570776342, 0.365682722812, 0.169143213293,
    -0.325325669158, 0.189706021964, 0.0877265322780,
    -0.0240476254170, -0.287659532608, -0.117219543175,
    -0.176860753121, -0.216393453025, -0.0148647893090,
])
C5_END = np.array([
    -4.792730224323733e00, -2.420550725448973e00, -9.212509306014886e-01,
    -4.217310404035213e00, 7.356202947498970e00, 3.223785985421212e00,
    4.035559443262270e00, 1.719865528670555e01, 7.478910794233703e00,
    -2.998759326324844e01, -4.107310937550929e00, -9.277008321754407e-01,
    -2.442125302518482e01, 2.381459045746554e01, 1.492096306951359e01,
    3.499208963063806e-01, -5.748487687912825e-01, -2.551694020879149e-01,
    -5.237040978903326e-01, -2.493000463579661e-01, -8.045341642044464e-02,
    -3.875289237334110e-01, 5.648603288767891e-02, 3.023606472143342e-02,
    4.133856546712445e-02, -2.862393029841379e-01, -1.183032405136207e-01,
    -1.511986457359206e-01, -2.460068894318766e-01, -3.189687411323877e-02,
])


def heliocentric_rhs(masses, m0, k2):
    masses = np.asarray(masses, dtype=float)
    N = masses.size

    def rhs(t, y):
        q = y[: 3 * N].reshape(N, 3)
        r3 = np.linalg.norm(q, axis=1) ** 3
        acc = -(m0 + masses)[:, None] * q / r3[:, None]
        for i in range(N):
            for j in range(N):
                if j != i:
                    d = q[j] - q[i]
                    acc[i] += masses[j] * (d / np.linalg.norm(d) ** 3 - q[j] / r3[j])
        return np.concatenate([y[3 * N:], k2 * acc.reshape(-1)])

    return rhs


def heliocentric_invariants(masses, m0, k2):
    """Barycentric energy and angular momentum of a heliocentric state."""
    masses = np.asarray(masses, dtype=float)
    N = masses.size
    all_m = np.concatenate([[m0], masses])
    M = all_m.sum()

    def barycentric(y):
        q = y[: 3 * N].reshape(N, 3, -1)
        v = y[3 * N:].reshape(N, 3, -1)
        q_sun = -np.einsum("i,idm->dm", masses, q) / M
        v_sun = -np.einsum("i,idm->dm", masses, v) / M
        pos = np.concatenate([q_sun[None], q + q_sun[None]], axis=0)
        vel = np.concatenate([v_sun[None], v + v_sun[None]], axis=0)
        return pos, vel

    def energy(y):
        pos, vel = barycentric(y)
        kin = 0.5 * np.einsum("i,idm->m", all_m, vel * vel)
        pot = np.zeros(y.shape[1])
        for i in range(N + 1):
            for j in range(i + 1, N + 1):
                pot -= k2 * all_m[i] * all_m[j] / np.linalg.norm(pos[j] - pos[i], axis=0)
        return kin + pot

    def angular_momentum(y):
        pos, vel = barycentric(y)
        return np.linalg.norm(np.einsum("i,idm->dm", all_m, np.cross(pos, vel, axis=1)), axis=0)

    return {"energy": energy, "angular momentum": angular_momentum}


def _detest_C():
    probs = []
    n = 10
    A_c1 = -np.eye(n) + np.eye(n, k=-1)
    A_c1[-1, -1] = 0.0
    y0 = np.zeros(n)
    y0[0] = 1.0
    probs.append(Problem(
        "C1", "Radioactive decay chain (10)", "detest", lambda t, y: A_c1 @ y,
        lambda: native.LinearSystem(A_c1), y0, (0.0, 20.0), exact=_linear_exact(A_c1, y0),
        y_end_ref=np.array([2.061153622240064e-09, 4.122307244619555e-08, 4.122307244716968e-07,
                            2.748204829855288e-06, 1.374102414941961e-05, 5.496409659803266e-05,
                            1.832136553274552e-04, 5.234675866508716e-04, 1.308668966628220e-03,
                            9.979127409508656e-01]),
        ref_source=DETEST_SOURCE, invariants={"sum y": lambda y: y.sum(axis=0)},
        plot_opts={"logy": True}))

    A_c2 = np.zeros((n, n))
    A_c2[0, 0] = -1.0
    for i in range(1, n - 1):
        A_c2[i, i - 1] = i
        A_c2[i, i] = -(i + 1)
    A_c2[n - 1, n - 2] = n - 1
    probs.append(Problem(
        "C2", "Decay chain, increasing rates (10)", "detest", lambda t, y: A_c2 @ y,
        lambda: native.LinearSystem(A_c2), y0, (0.0, 20.0), exact=_linear_exact(A_c2, y0),
        y_end_ref=np.array([2.061153577984930e-09, 2.061153573736588e-09, 2.061153569488245e-09,
                            2.061153565239902e-09, 2.061153560991560e-09, 2.061153556743217e-09,
                            2.061153552494874e-09, 2.061153548246532e-09, 2.061153543998189e-09,
                            9.999999814496180e-01]),
        ref_source=DETEST_SOURCE, invariants={"sum y": lambda y: y.sum(axis=0)},
        plot_opts={"logy": True}))

    A_c3 = _tridiagonal(10)
    probs.append(Problem(
        "C3", "Tridiagonal diffusion (10)", "detest", lambda t, y: A_c3 @ y,
        lambda: native.LinearSystem(A_c3), y0, (0.0, 20.0), exact=_linear_exact(A_c3, y0),
        y_end_ref=np.array([2.948119211022058e-03, 5.635380154844266e-03, 7.829072515926013e-03,
                            9.348257908594937e-03, 1.007943610301970e-02, 9.982674171429909e-03,
                            9.088693332766085e-03, 7.489115195185912e-03, 5.322964130953349e-03,
                            2.762434379029886e-03]),
        ref_source=DETEST_SOURCE, plot="heatmap"))

    A_c4 = _tridiagonal(51)
    y0_51 = np.zeros(51)
    y0_51[0] = 1.0
    probs.append(Problem(
        "C4", "Tridiagonal diffusion (51)", "detest", lambda t, y: A_c4 @ y,
        lambda: native.LinearSystem(A_c4), y0_51, (0.0, 20.0), exact=_linear_exact(A_c4, y0_51),
        y_end_ref=_C4_END, ref_source=DETEST_SOURCE, plot="heatmap"))

    probs.append(Problem(
        "C5", "Outer planets, heliocentric (30)", "detest",
        heliocentric_rhs(C5_MASSES, C5_M0, C5_K2),
        lambda: native.HeliocentricPlanets(C5_MASSES, C5_M0, C5_K2), C5_Y0.copy(), (0.0, 20.0),
        y_end_ref=C5_END, ref_source=DETEST_SOURCE,
        invariants=heliocentric_invariants(C5_MASSES, C5_M0, C5_K2),
        plot="orbit", plot_pairs=tuple((3 * i, 3 * i + 1) for i in range(5)),
        labels=("Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"), plot_opts={"sun": True}))
    return probs


_D_END = {
    0.1: [2.198835352008397e-01, 9.427076846341813e-01, -9.787659841058176e-01, 3.287977990962036e-01],
    0.3: [-1.777027357140412e-01, 9.467784719905892e-01, -1.030294163192969e00, 1.211074890053952e-01],
    0.5: [-5.780432953035361e-01, 8.633840009194193e-01, -9.595083730380727e-01, -6.504915126712089e-02],
    0.7: [-9.538990293416394e-01, 6.907409024219432e-01, -8.212674270877433e-01, -1.539574259125825e-01],
    0.9: [-1.295266250987574e00, 4.003938963792321e-01, -6.775390924707566e-01, -1.270838154278686e-01],
}


def _detest_D():
    probs = []
    for k, e in enumerate([0.1, 0.3, 0.5, 0.7, 0.9], start=1):
        probs.append(Problem(
            f"D{k}", f"Kepler orbit, e = {e}", "detest", _kepler_rhs, native.Kepler2D,
            kepler_y0(e), (0.0, 20.0), exact=kepler_exact(e), y_end_ref=np.array(_D_END[e]),
            ref_source=DETEST_SOURCE, invariants=kepler_invariants(),
            plot="orbit", plot_pairs=((0, 1),), plot_opts={"sun": True}))
    return probs


def _detest_E():
    probs = []

    def e1(t, y):
        x = t + 1.0
        return np.array([y[1], -(y[1] / x + (1.0 - 0.25 / x ** 2) * y[0])])

    def e1_exact(t):
        x = np.atleast_1d(t) + 1.0
        c = np.sqrt(2.0 / np.pi)
        return np.vstack([c * np.sin(x) / np.sqrt(x),
                          c * (np.cos(x) / np.sqrt(x) - 0.5 * np.sin(x) / x ** 1.5)])

    probs.append(Problem(
        "E1", "Bessel J_1/2", "detest", e1, native.DetestE1,
        np.array([0.6713967071418030, 0.09540051444747446]), (0.0, 20.0), exact=e1_exact,
        y_end_ref=np.array([1.456723600728308e-01, -9.883500195574063e-02]), ref_source=DETEST_SOURCE,
        plot="phase", plot_pairs=((0, 1),)))

    def e2(t, y):
        return np.array([y[1], (1.0 - y[0] ** 2) * y[1] - y[0]])

    probs.append(Problem(
        "E2", "Van der Pol, mu = 1", "detest", e2, lambda: native.VanDerPol(1.0),
        np.array([2.0, 0.0]), (0.0, 20.0),
        y_end_ref=np.array([2.008149762174948e00, -4.250887527320057e-02]), ref_source=DETEST_SOURCE,
        plot="phase", plot_pairs=((0, 1),)))

    def e3(t, y):
        return np.array([y[1], y[0] ** 3 / 6.0 - y[0] + 2.0 * np.sin(2.78535 * t)])

    probs.append(Problem(
        "E3", "Forced Duffing", "detest", e3, native.DetestE3, np.array([0.0, 0.0]), (0.0, 20.0),
        y_end_ref=np.array([-1.004178858647128e-01, 2.411400132095954e-01]), ref_source=DETEST_SOURCE,
        plot="phase", plot_pairs=((0, 1),)))

    def e4(t, y):
        return np.array([y[1], 0.032 - 0.4 * y[1] ** 2])

    def e4_exact(t):
        t = np.atleast_1d(t)
        a, b = 0.032, 0.4
        w = np.sqrt(a * b)
        return np.vstack([30.0 + np.log(np.cosh(w * t)) / b, np.sqrt(a / b) * np.tanh(w * t)])

    probs.append(Problem(
        "E4", "Falling body with drag", "detest", e4, native.DetestE4, np.array([30.0, 0.0]), (0.0, 20.0),
        exact=e4_exact, y_end_ref=np.array([3.395091444646555e01, 2.767822659672869e-01]),
        ref_source=DETEST_SOURCE))

    def e5(t, y):
        return np.array([y[1], np.sqrt(1.0 + y[1] ** 2) / (25.0 - t)])

    def e5_exact(t):
        t = np.atleast_1d(t)
        s = 25.0 - t
        v = 0.5 * (25.0 / s - s / 25.0)
        y = 0.5 * (-25.0 * np.log(s) + s ** 2 / 50.0) - 0.5 * (-25.0 * np.log(25.0) + 12.5)
        return np.vstack([y, v])

    probs.append(Problem(
        "E5", "Pursuit curve", "detest", e5, native.DetestE5, np.array([0.0, 0.0]), (0.0, 20.0),
        exact=e5_exact, y_end_ref=np.array([1.411797390542629e01, 2.400000000000002e00]),
        ref_source=DETEST_SOURCE))
    return probs


# =============================================================================
# Hairer-Norsett-Wanner
# =============================================================================

HNW_SOURCE = "Hairer, Norsett & Wanner, Solving ODEs I"
ARENSTORF_MU = 0.012277471
ARENSTORF_Y0 = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])
ARENSTORF_PERIOD = 17.0652165601579625588917206249


def arenstorf_rhs(t, y, mu=ARENSTORF_MU):
    mu1 = 1.0 - mu
    D1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    D2 = ((y[0] - mu1) ** 2 + y[1] ** 2) ** 1.5
    return np.array([
        y[2], y[3],
        y[0] + 2 * y[3] - mu1 * (y[0] + mu) / D1 - mu * (y[0] - mu1) / D2,
        y[1] - 2 * y[2] - mu1 * y[1] / D1 - mu * y[1] / D2,
    ])


def jacobi_constant(y, mu=ARENSTORF_MU):
    mu1 = 1.0 - mu
    r1 = np.hypot(y[0] + mu, y[1])
    r2 = np.hypot(y[0] - mu1, y[1])
    return y[0] ** 2 + y[1] ** 2 + 2 * mu1 / r1 + 2 * mu / r2 - (y[2] ** 2 + y[3] ** 2)


PLEIADES_MASSES = np.arange(1.0, 8.0)
PLEIADES_Y0 = np.concatenate([
    np.column_stack([[3, 3, -1, -3, 2, -2, 2], [3, -3, 2, 0, 0, -4, 4]]).reshape(-1),
    np.column_stack([[0, 0, 0, 0, 0, 1.75, -1.5], [0, 0, 0, -1.25, 1, 0, 0]]).reshape(-1),
]).astype(float)


def _hairer():
    probs = []
    probs.append(Problem(
        "arenstorf", "Arenstorf orbit (restricted 3-body)", "hairer", arenstorf_rhs, native.Arenstorf,
        ARENSTORF_Y0.copy(), (0.0, ARENSTORF_PERIOD), y_end_ref=ARENSTORF_Y0.copy(),
        ref_source=HNW_SOURCE + " (periodic: y(T) = y0)",
        invariants={"Jacobi constant": jacobi_constant},
        plot="orbit", plot_pairs=((0, 1),), plot_opts={"bodies": [(-ARENSTORF_MU, 0.0, "Earth"),
                                                                    (1 - ARENSTORF_MU, 0.0, "Moon")]}))
    probs.append(Problem(
        "pleiades", "Pleiades (7-body, near collisions)", "hairer",
        nbody_rhs(PLEIADES_MASSES, 1.0, 2),
        lambda: NBodyGravity(PLEIADES_MASSES, G=1.0, n_dims=2), PLEIADES_Y0.copy(), (0.0, 3.0),
        ref_source=HNW_SOURCE + " / Hairer-Wanner II (PLEI)",
        invariants=nbody_invariants(PLEIADES_MASSES, 1.0, 2),
        plot="orbit", plot_pairs=tuple((2 * i, 2 * i + 1) for i in range(7)),
        labels=tuple(f"m={i}" for i in range(1, 8))))

    def brusselator(t, y):
        return np.array([1.0 + y[0] ** 2 * y[1] - 4.0 * y[0], 3.0 * y[0] - y[0] ** 2 * y[1]])

    probs.append(Problem(
        "brusselator", "Brusselator (A=1, B=3)", "hairer", brusselator, native.Brusselator,
        np.array([1.5, 3.0]), (0.0, 20.0), ref_source=HNW_SOURCE,
        plot="phase", plot_pairs=((0, 1),)))
    return probs


# =============================================================================
# Long-time / geometric integration
# =============================================================================

def _long_time():
    probs = []
    e = 0.6
    probs.append(Problem(
        "kepler_long", "Kepler e = 0.6, 100 orbits", "long_time", _kepler_rhs, native.Kepler2D,
        kepler_y0(e), (0.0, 200 * np.pi), exact=kepler_exact(e),
        ref_source="Hairer, Lubich & Wanner, Geometric Numerical Integration (exact)",
        invariants=kepler_invariants(), plot="orbit", plot_pairs=((0, 1),), plot_opts={"sun": True}))
    probs.append(Problem(
        "outer_solar_system", "Outer solar system, 2e5 days", "long_time",
        heliocentric_rhs(C5_MASSES, C5_M0, C5_K2),
        lambda: native.HeliocentricPlanets(C5_MASSES, C5_M0, C5_K2), C5_Y0.copy(), (0.0, 2000.0),
        ref_source="DETEST C5 data over the Hairer-Lubich-Wanner interval",
        invariants=heliocentric_invariants(C5_MASSES, C5_M0, C5_K2),
        plot="orbit", plot_pairs=tuple((3 * i, 3 * i + 1) for i in range(5)),
        labels=("Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"), plot_opts={"sun": True}))

    fe = figure_eight_system()
    probs.append(Problem(
        "figure_eight_long", "Figure-eight choreography, 50 periods", "long_time",
        nbody_rhs(fe.masses, fe.G, 2), lambda: NBodyGravity(fe.masses, G=fe.G, n_dims=2),
        fe.y0.copy(), (0.0, 50 * fe.period),
        ref_source="Chenciner & Montgomery; periodic, but the published ICs (8 digits) "
                   "close the orbit only to ~2e-6, so the reference is a tight solve",
        invariants=nbody_invariants(fe.masses, fe.G, 2),
        plot="orbit", plot_pairs=((0, 1), (2, 3), (4, 5))))
    probs.append(Problem(
        "arenstorf_long", "Arenstorf orbit, 5 periods", "long_time", arenstorf_rhs, native.Arenstorf,
        ARENSTORF_Y0.copy(), (0.0, 5 * ARENSTORF_PERIOD),
        ref_source=HNW_SOURCE + "; the orbit is unstable, so after 5 periods even an "
                   "rtol=1e-13 solve is off by O(0.1): only invariants are meaningful",
        invariants={"Jacobi constant": jacobi_constant},
        plot="orbit", plot_pairs=((0, 1),), plot_opts={"bodies": [(-ARENSTORF_MU, 0.0, "Earth"),
                                                                    (1 - ARENSTORF_MU, 0.0, "Moon")]}))
    return probs


# =============================================================================
# Stability: stiff, unstable, chaotic, blow-up
# =============================================================================

LINEAR_STIFF_LAMBDA = -1000.0


def _stability():
    probs = []
    lam = LINEAR_STIFF_LAMBDA
    probs.append(Problem(
        "linear_stiff", f"y' = {lam:g} (y - cos t) - sin t", "stability",
        lambda t, y: lam * (y - np.cos(t)) - np.sin(t),
        lambda: native.ProtheroRobinson(lam), np.array([1.0]), (0.0, 10.0),
        exact=lambda t: np.cos(np.atleast_1d(t))[None, :],
        ref_source="Prothero-Robinson test equation (exact)", stiff=True,
        notes="Smooth solution cos t, but step size is capped by the stability "
              "region at |h lambda| ~ 2.5-6."))

    A_osc = np.array([[-1.0, -100.0], [100.0, -1.0]])
    probs.append(Problem(
        "damped_oscillator", "Lightly damped fast oscillator", "stability", lambda t, y: A_osc @ y,
        lambda: native.LinearSystem(A_osc), np.array([1.0, 0.0]), (0.0, 5.0),
        exact=_linear_exact(A_osc, [1.0, 0.0]), ref_source="linear (exact)",
        plot="phase", plot_pairs=((0, 1),)))

    for mu in (1.0, 10.0, 100.0, 1000.0):
        probs.append(Problem(
            f"vdp_mu{int(mu)}", f"Van der Pol, mu = {mu:g}", "stability",
            (lambda m: lambda t, y: np.array([y[1], m * (1.0 - y[0] ** 2) * y[1] - y[0]]))(mu),
            (lambda m: lambda: native.VanDerPol(m))(mu), np.array([2.0, 0.0]), (0.0, 20.0),
            ref_source=HNW_SOURCE, stiff=mu >= 100, plot="phase", plot_pairs=((0, 1),),
            plot_opts={"symlog_y": mu >= 100}))

    def robertson(t, y):
        r1 = 0.04 * y[0]
        r2 = 1e4 * y[1] * y[2]
        r3 = 3e7 * y[1] ** 2
        return np.array([-r1 + r2, r1 - r2 - r3, r3])

    probs.append(Problem(
        "robertson", "Robertson kinetics (stiff)", "stability", robertson, native.Robertson,
        np.array([1.0, 0.0, 0.0]), (0.0, 2.0), ref_source="Hairer & Wanner, Solving ODEs II",
        invariants={"sum y": lambda y: y.sum(axis=0)}, stiff=True,
        labels=("y1", "y2 x 1e4", "y3"), plot_opts={"scale": (1.0, 1e4, 1.0)}))

    probs.append(Problem(
        "lorenz", "Lorenz attractor (chaotic)", "stability",
        lambda t, y: np.array([10.0 * (y[1] - y[0]), y[0] * (28.0 - y[2]) - y[1],
                               y[0] * y[1] - 8.0 / 3.0 * y[2]]),
        native.Lorenz, np.array([-8.0, 8.0, 27.0]), (0.0, 20.0), ref_source=HNW_SOURCE,
        plot="phase", plot_pairs=((0, 2),), labels=("x", "y", "z"),
        notes="Max Lyapunov exponent ~0.9: errors grow ~e^{0.9 t}, so accuracy "
              "is only meaningful for t of order ten."))

    probs.append(Problem(
        "blowup", "y' = y^2 (singular at t = 1)", "stability", lambda t, y: y ** 2,
        native.Quadratic, np.array([1.0]), (0.0, 2.0),
        exact=lambda t: (1.0 / (1.0 - np.atleast_1d(t)))[None, :],
        ref_source="exact: y = 1 / (1 - t)", expect_failure=True, plot_opts={"logy": True}))
    return probs


# =============================================================================
# Registry
# =============================================================================

@lru_cache(maxsize=None)
def all_problems() -> tuple:
    return tuple(_detest_A() + _detest_B() + _detest_C() + _detest_D() + _detest_E()
                 + _hairer() + _long_time() + _stability())


def get(name: str) -> Problem:
    for p in all_problems():
        if p.name == name:
            return p
    raise KeyError(name)


def by_group(group: str) -> tuple:
    return tuple(p for p in all_problems() if p.group == group)


def detest() -> tuple:
    return by_group("detest")


# =============================================================================
# References
# =============================================================================

REFERENCE_RTOL = 1e-13
REFERENCE_ATOL = 1e-15


@lru_cache(maxsize=None)
def _tight_solution(name: str):
    p = get(name)
    sol = solve_ivp(p.native(), p.t_span, p.y0, method="DOP853", dense_output=True,
                    rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    if not sol.success:
        raise RuntimeError(f"reference integration of {name} failed: {sol.message}")
    return sol


def reference(p: Problem, t) -> np.ndarray:
    """Reference solution at times ``t`` (shape (n, m)).

    The exact solution if known, otherwise a DOP853 solution at rtol=1e-13.
    The tight solution is cross-checked against SciPy's (independent) Radau in
    ``tests/accuracy``.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if p.exact is not None:
        return p.exact(t)
    return _tight_solution(p.name).sol(t)


def reference_end(p: Problem) -> np.ndarray:
    """Best available value at ``t_span[1]``: exact, then published, then computed."""
    if p.exact is not None:
        return p.exact(np.array([p.t_span[1]]))[:, 0]
    if p.y_end_ref is not None:
        return np.asarray(p.y_end_ref, dtype=float)
    return _tight_solution(p.name).y[:, -1]


def solution_scale(p: Problem) -> float:
    """Magnitude used to make errors relative: ``max(|y0|, |y_end|)`` (max-norm).

    Using the end value alone would make decaying problems (A1 ends at 2e-9)
    report the absolute-tolerance floor as a huge relative error.
    """
    return float(max(np.max(np.abs(p.y0)), np.max(np.abs(reference_end(p))), 1e-300))


def end_error(p: Problem, y_end) -> float:
    """Max-norm error at ``t_span[1]`` divided by :func:`solution_scale`."""
    ref = reference_end(p)
    return float(np.max(np.abs(np.asarray(y_end) - ref)) / solution_scale(p))
