# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Native (Cython) right-hand sides and events for the test-problem catalog.

Each class mirrors a NumPy RHS in :mod:`fast_numerical_integration.problems.catalog`
and is checked against it in the tests. These are hand-written for now; the
SymPy-generated kernels planned for later can replace them one for one.
"""

from libc.math cimport cos, sin, sqrt

import numpy as np

from fast_numerical_integration.integrators cimport EventFunction, OdeSystem


# --- generic -----------------------------------------------------------------------

cdef class LinearSystem(OdeSystem):
    """Affine system ``y' = A y + b`` (dense ``A``)."""

    cdef double[:, ::1] A
    cdef double[::1] b

    def __init__(self, A, b=None):
        A = np.ascontiguousarray(A, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        self.A = A
        self.b = np.zeros(A.shape[0]) if b is None else np.ascontiguousarray(b, dtype=np.float64)
        self.n_states = A.shape[0]

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef Py_ssize_t i, j
        cdef double acc
        for i in range(n):
            acc = self.b[i]
            for j in range(n):
                acc += self.A[i, j] * y[j]
            dydt[i] = acc
        return 0


cdef class Quadratic(OdeSystem):
    """``y' = y**2``: finite-time blow-up at ``t = 1 / y0``."""

    def __init__(self):
        self.n_states = 1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[0] * y[0]
        return 0


cdef class ProtheroRobinson(OdeSystem):
    """``y' = lam (y - cos t) - sin t``: exact solution ``cos t`` for ``y0 = 1``.

    For ``lam << 0`` the solution is smooth but explicit methods are limited by
    their stability region, not by accuracy.
    """

    cdef double lam

    def __init__(self, double lam):
        self.lam = lam
        self.n_states = 1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = self.lam * (y[0] - cos(t)) - sin(t)
        return 0


# --- DETEST class A ----------------------------------------------------------------

cdef class DetestA2(OdeSystem):
    def __init__(self):
        self.n_states = 1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = -0.5 * y[0] * y[0] * y[0]
        return 0


cdef class DetestA3(OdeSystem):
    def __init__(self):
        self.n_states = 1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[0] * cos(t)
        return 0


cdef class DetestA4(OdeSystem):
    def __init__(self):
        self.n_states = 1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[0] * (20.0 - y[0]) / 80.0
        return 0


cdef class DetestA5(OdeSystem):
    def __init__(self):
        self.n_states = 1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = (y[0] - t) / (y[0] + t)
        return 0


# --- DETEST class B ----------------------------------------------------------------

cdef class DetestB1(OdeSystem):
    """Lotka-Volterra predator-prey."""

    def __init__(self):
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = 2.0 * y[0] * (1.0 - y[1])
        dydt[1] = -y[1] * (1.0 - y[0])
        return 0


cdef class DetestB3(OdeSystem):
    def __init__(self):
        self.n_states = 3

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = -y[0]
        dydt[1] = y[0] - y[1] * y[1]
        dydt[2] = y[1] * y[1]
        return 0


cdef class DetestB4(OdeSystem):
    def __init__(self):
        self.n_states = 3

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef double norm = sqrt(y[0] * y[0] + y[1] * y[1])
        if norm > 0.0:
            dydt[0] = -y[1] - y[0] * y[2] / norm
            dydt[1] = y[0] - y[1] * y[2] / norm
            dydt[2] = y[0] / norm
        else:
            dydt[0] = 0.0
            dydt[1] = 0.0
            dydt[2] = 0.0
        return 0


cdef class EulerRigidBody(OdeSystem):
    """``y1' = a y2 y3, y2' = b y1 y3, y3' = c y1 y2`` (DETEST B5: a=1, b=-1, c=-0.51)."""

    cdef double a, b, c

    def __init__(self, double a=1.0, double b=-1.0, double c=-0.51):
        self.a = a
        self.b = b
        self.c = c
        self.n_states = 3

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = self.a * y[1] * y[2]
        dydt[1] = self.b * y[0] * y[2]
        dydt[2] = self.c * y[0] * y[1]
        return 0


# --- DETEST class C ----------------------------------------------------------------

cdef class HeliocentricPlanets(OdeSystem):
    """DETEST C5: heliocentric outer-planet problem (3-D).

    ``q_i'' = k2 * (-(m0 + m_i) q_i / r_i^3
                    + sum_{j != i} m_j ((q_j - q_i) / d_ij^3 - q_j / r_j^3))``
    with state ``[q_1..q_N (3 each), q_1'..q_N' (3 each)]``.
    """

    cdef double k2, m0
    cdef double[::1] m
    cdef double[::1] inv_r3
    cdef Py_ssize_t N

    def __init__(self, masses, double m0, double k2):
        self.m = np.ascontiguousarray(masses, dtype=np.float64).copy()
        self.N = self.m.shape[0]
        self.m0 = m0
        self.k2 = k2
        self.inv_r3 = np.empty(self.N)
        self.n_states = 6 * self.N

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef Py_ssize_t i, j, k, N = self.N, half = 3 * self.N
        cdef double r2, d2, inv_d3, dx
        cdef double acc[3]
        for i in range(N):
            r2 = y[3 * i] ** 2 + y[3 * i + 1] ** 2 + y[3 * i + 2] ** 2
            self.inv_r3[i] = 1.0 / (r2 * sqrt(r2))
        for i in range(half):
            dydt[i] = y[half + i]
        for i in range(N):
            for k in range(3):
                acc[k] = -(self.m0 + self.m[i]) * y[3 * i + k] * self.inv_r3[i]
            for j in range(N):
                if j == i:
                    continue
                d2 = 0.0
                for k in range(3):
                    dx = y[3 * j + k] - y[3 * i + k]
                    d2 += dx * dx
                inv_d3 = 1.0 / (d2 * sqrt(d2))
                for k in range(3):
                    acc[k] += self.m[j] * ((y[3 * j + k] - y[3 * i + k]) * inv_d3
                                           - y[3 * j + k] * self.inv_r3[j])
            for k in range(3):
                dydt[half + 3 * i + k] = self.k2 * acc[k]
        return 0


# --- DETEST class D / Kepler -------------------------------------------------------

cdef class Kepler2D(OdeSystem):
    """Planar two-body problem ``q'' = -mu q / |q|^3``; state ``[x, y, vx, vy]``."""

    cdef double mu

    def __init__(self, double mu=1.0):
        self.mu = mu
        self.n_states = 4

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef double r2 = y[0] * y[0] + y[1] * y[1]
        cdef double f = -self.mu / (r2 * sqrt(r2))
        dydt[0] = y[2]
        dydt[1] = y[3]
        dydt[2] = f * y[0]
        dydt[3] = f * y[1]
        return 0


# --- DETEST class E ----------------------------------------------------------------

cdef class DetestE1(OdeSystem):
    """Bessel equation of order 1/2 in ``x = t + 1``."""

    def __init__(self):
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef double x = t + 1.0
        dydt[0] = y[1]
        dydt[1] = -(y[1] / x + (1.0 - 0.25 / (x * x)) * y[0])
        return 0


cdef class VanDerPol(OdeSystem):
    """``y'' = mu (1 - y^2) y' - y`` (DETEST E2: mu = 1)."""

    cdef double mu

    def __init__(self, double mu=1.0):
        self.mu = mu
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[1]
        dydt[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0]
        return 0


cdef class DetestE3(OdeSystem):
    """Forced Duffing oscillator."""

    def __init__(self):
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[1]
        dydt[1] = y[0] * y[0] * y[0] / 6.0 - y[0] + 2.0 * sin(2.78535 * t)
        return 0


cdef class DetestE4(OdeSystem):
    def __init__(self):
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[1]
        dydt[1] = 0.032 - 0.4 * y[1] * y[1]
        return 0


cdef class DetestE5(OdeSystem):
    def __init__(self):
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[1]
        dydt[1] = sqrt(1.0 + y[1] * y[1]) / (25.0 - t)
        return 0


# --- Hairer-Norsett-Wanner problems --------------------------------------------------

cdef class Arenstorf(OdeSystem):
    """Restricted three-body problem in the rotating frame; state ``[x, y, vx, vy]``."""

    cdef double mu

    def __init__(self, double mu=0.012277471):
        self.mu = mu
        self.n_states = 4

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef double mu = self.mu, mu1 = 1.0 - mu
        cdef double a = (y[0] + mu) ** 2 + y[1] ** 2
        cdef double b = (y[0] - mu1) ** 2 + y[1] ** 2
        cdef double D1 = a * sqrt(a)
        cdef double D2 = b * sqrt(b)
        dydt[0] = y[2]
        dydt[1] = y[3]
        dydt[2] = y[0] + 2.0 * y[3] - mu1 * (y[0] + mu) / D1 - mu * (y[0] - mu1) / D2
        dydt[3] = y[1] - 2.0 * y[2] - mu1 * y[1] / D1 - mu * y[1] / D2
        return 0


cdef class Brusselator(OdeSystem):
    """``y1' = A + y1^2 y2 - (B + 1) y1``, ``y2' = B y1 - y1^2 y2``."""

    cdef double A, B

    def __init__(self, double A=1.0, double B=3.0):
        self.A = A
        self.B = B
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef double y1y1y2 = y[0] * y[0] * y[1]
        dydt[0] = self.A + y1y1y2 - (self.B + 1.0) * y[0]
        dydt[1] = self.B * y[0] - y1y1y2
        return 0


cdef class Lorenz(OdeSystem):
    cdef double sigma, rho, beta

    def __init__(self, double sigma=10.0, double rho=28.0, double beta=8.0 / 3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.n_states = 3

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = self.sigma * (y[1] - y[0])
        dydt[1] = y[0] * (self.rho - y[2]) - y[1]
        dydt[2] = y[0] * y[1] - self.beta * y[2]
        return 0


cdef class Robertson(OdeSystem):
    """Robertson's stiff chemical kinetics."""

    def __init__(self):
        self.n_states = 3

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef double r1 = 0.04 * y[0]
        cdef double r2 = 1e4 * y[1] * y[2]
        cdef double r3 = 3e7 * y[1] * y[1]
        dydt[0] = -r1 + r2
        dydt[1] = r1 - r2 - r3
        dydt[2] = r3
        return 0


cdef class Pendulum(OdeSystem):
    """``theta'' = -(g / L) sin(theta)``; state ``[theta, omega]``."""

    cdef double w2

    def __init__(self, double w2=1.0):
        self.w2 = w2
        self.n_states = 2

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        dydt[0] = y[1]
        dydt[1] = -self.w2 * sin(y[0])
        return 0


# --- native events -------------------------------------------------------------------

cdef class DotProductEvent(EventFunction):
    """``g = sum_k y[left[k]] * y[right[k]] - value``.

    With ``left`` = positions and ``right`` = velocities this is ``r . v``, whose
    zeros are the apsides of an orbit (direction +1: periapsis, -1: apoapsis).
    """

    cdef Py_ssize_t[::1] left
    cdef Py_ssize_t[::1] right
    cdef readonly double value

    def __init__(self, left, right, double value=0.0, terminal=False, double direction=0.0):
        super().__init__(terminal, direction)
        left = np.ascontiguousarray(left, dtype=np.intp)
        right = np.ascontiguousarray(right, dtype=np.intp)
        if left.shape != right.shape or left.ndim != 1:
            raise ValueError("left and right must be 1-D and the same length")
        if left.size and (left.min() < 0 or right.min() < 0):
            raise ValueError("indices must be non-negative")
        self.left = left
        self.right = right
        self.value = value

    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1:
        cdef Py_ssize_t k
        cdef double acc = -self.value
        for k in range(self.left.shape[0]):
            if self.left[k] >= n or self.right[k] >= n:
                raise IndexError("event index out of range")
            acc += y[self.left[k]] * y[self.right[k]]
        return acc


cdef class TimeEvent(EventFunction):
    """``g = t - t_event``: fires at a prescribed time."""

    cdef readonly double t_event

    def __init__(self, double t_event, terminal=False, double direction=0.0):
        super().__init__(terminal, direction)
        self.t_event = t_event

    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1:
        return t - self.t_event
