# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Native (Cython) right-hand sides and events for the test-problem catalog.

Each class mirrors a NumPy RHS in :mod:`fast_numerical_integration.problems.catalog`
and is checked against it in the tests. Every one is a GIL-free C function
(``c_fn``) plus a context, the same shape the SymPy-generated kernels will
have. Physical constants are runtime parameters (``system.params``), so
continuation in them needs no recompilation. The kernels are reentrant (no
shared scratch), so one system object can be integrated from several threads.
"""

from libc.math cimport cos, sin, sqrt

import numpy as np

from fast_numerical_integration.integrators cimport EventFunction, OdeSystem

cdef enum:
    MAX_PLANETS = 32


cdef inline void _bind(OdeSystem system, int (*fn)(double, const double*, double*, Py_ssize_t, void*) noexcept nogil,
                       void* ctx) noexcept:
    system.c_fn = fn
    system.c_ctx = ctx


# --- generic -----------------------------------------------------------------------

cdef struct _linear_ctx:
    Py_ssize_t n
    const double* A
    const double* b


cdef int _linear_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef _linear_ctx* c = <_linear_ctx*> ctx
    cdef Py_ssize_t i, j
    cdef double acc
    for i in range(n):
        acc = c.b[i]
        for j in range(n):
            acc += c.A[i * n + j] * y[j]
        dydt[i] = acc
    return 0


cdef class LinearSystem(OdeSystem):
    """Affine system ``y' = A y + b`` (dense ``A``)."""

    cdef double[:, ::1] A
    cdef double[::1] b
    cdef _linear_ctx data

    def __init__(self, A, b=None):
        A = np.array(A, dtype=np.float64, copy=True)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("A must be square and non-empty")
        self.A = A
        self.b = np.zeros(A.shape[0]) if b is None else np.array(b, dtype=np.float64, copy=True)
        if self.b.shape[0] != A.shape[0]:
            raise ValueError("b must have one entry per row of A")
        self.n_states = A.shape[0]
        self.data.n = A.shape[0]
        self.data.A = &self.A[0, 0]
        self.data.b = &self.b[0]
        _bind(self, _linear_fn, &self.data)


cdef int _quadratic_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = y[0] * y[0]
    return 0


cdef class Quadratic(OdeSystem):
    """``y' = y**2``: finite-time blow-up at ``t = 1 / y0``."""

    def __init__(self):
        self.n_states = 1
        _bind(self, _quadratic_fn, NULL)


cdef int _prothero_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    dydt[0] = p[0] * (y[0] - cos(t)) - sin(t)
    return 0


cdef class ProtheroRobinson(OdeSystem):
    """``y' = lam (y - cos t) - sin t``: exact solution ``cos t`` for ``y0 = 1``.

    For ``lam << 0`` the solution is smooth but explicit methods are limited by
    their stability region, not by accuracy. Parameter: ``lam``.
    """

    def __init__(self, double lam):
        self.n_states = 1
        self._init_params([lam], ("lam",))
        _bind(self, _prothero_fn, self._params_ptr())


# --- DETEST class A ----------------------------------------------------------------

cdef int _a2_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = -0.5 * y[0] * y[0] * y[0]
    return 0


cdef int _a3_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = y[0] * cos(t)
    return 0


cdef int _a4_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = y[0] * (20.0 - y[0]) / 80.0
    return 0


cdef int _a5_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = (y[0] - t) / (y[0] + t)
    return 0


cdef class DetestA2(OdeSystem):
    def __init__(self):
        self.n_states = 1
        _bind(self, _a2_fn, NULL)


cdef class DetestA3(OdeSystem):
    def __init__(self):
        self.n_states = 1
        _bind(self, _a3_fn, NULL)


cdef class DetestA4(OdeSystem):
    def __init__(self):
        self.n_states = 1
        _bind(self, _a4_fn, NULL)


cdef class DetestA5(OdeSystem):
    def __init__(self):
        self.n_states = 1
        _bind(self, _a5_fn, NULL)


# --- DETEST class B ----------------------------------------------------------------

cdef int _b1_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = 2.0 * y[0] * (1.0 - y[1])
    dydt[1] = -y[1] * (1.0 - y[0])
    return 0


cdef int _b3_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = -y[0]
    dydt[1] = y[0] - y[1] * y[1]
    dydt[2] = y[1] * y[1]
    return 0


cdef int _b4_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
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


cdef int _euler_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    dydt[0] = p[0] * y[1] * y[2]
    dydt[1] = p[1] * y[0] * y[2]
    dydt[2] = p[2] * y[0] * y[1]
    return 0


cdef class DetestB1(OdeSystem):
    """Lotka-Volterra predator-prey."""

    def __init__(self):
        self.n_states = 2
        _bind(self, _b1_fn, NULL)


cdef class DetestB3(OdeSystem):
    def __init__(self):
        self.n_states = 3
        _bind(self, _b3_fn, NULL)


cdef class DetestB4(OdeSystem):
    def __init__(self):
        self.n_states = 3
        _bind(self, _b4_fn, NULL)


cdef class EulerRigidBody(OdeSystem):
    """``y1' = a y2 y3, y2' = b y1 y3, y3' = c y1 y2`` (DETEST B5: a=1, b=-1, c=-0.51).

    Parameters: ``a``, ``b``, ``c``.
    """

    def __init__(self, double a=1.0, double b=-1.0, double c=-0.51):
        self.n_states = 3
        self._init_params([a, b, c], ("a", "b", "c"))
        _bind(self, _euler_fn, self._params_ptr())


# --- DETEST class C ----------------------------------------------------------------

cdef struct _planets_ctx:
    Py_ssize_t N
    const double* p          # [k2, m0, m_1, ..., m_N]


cdef int _planets_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef _planets_ctx* c = <_planets_ctx*> ctx
    cdef Py_ssize_t i, j, k, N = c.N, half = 3 * c.N
    cdef double k2 = c.p[0], m0 = c.p[1]
    cdef const double* m = c.p + 2
    cdef double r2, d2, inv_d3, dx
    cdef double acc[3]
    cdef double inv_r3[MAX_PLANETS]      # on the stack: keeps the kernel reentrant
    for i in range(N):
        r2 = y[3 * i] ** 2 + y[3 * i + 1] ** 2 + y[3 * i + 2] ** 2
        inv_r3[i] = 1.0 / (r2 * sqrt(r2))
    for i in range(half):
        dydt[i] = y[half + i]
    for i in range(N):
        for k in range(3):
            acc[k] = -(m0 + m[i]) * y[3 * i + k] * inv_r3[i]
        for j in range(N):
            if j == i:
                continue
            d2 = 0.0
            for k in range(3):
                dx = y[3 * j + k] - y[3 * i + k]
                d2 += dx * dx
            inv_d3 = 1.0 / (d2 * sqrt(d2))
            for k in range(3):
                acc[k] += m[j] * ((y[3 * j + k] - y[3 * i + k]) * inv_d3
                                  - y[3 * j + k] * inv_r3[j])
        for k in range(3):
            dydt[half + 3 * i + k] = k2 * acc[k]
    return 0


cdef class HeliocentricPlanets(OdeSystem):
    """DETEST C5: heliocentric outer-planet problem (3-D).

    ``q_i'' = k2 * (-(m0 + m_i) q_i / r_i^3
                    + sum_{j != i} m_j ((q_j - q_i) / d_ij^3 - q_j / r_j^3))``
    with state ``[q_1..q_N (3 each), q_1'..q_N' (3 each)]``.
    Parameters: ``k2``, ``m0``, ``m1`` ... ``mN``.
    """

    cdef _planets_ctx data

    def __init__(self, masses, double m0, double k2):
        masses = np.ascontiguousarray(masses, dtype=np.float64).reshape(-1)
        N = masses.shape[0]
        if not 1 <= N <= MAX_PLANETS:
            raise ValueError(f"HeliocentricPlanets supports 1 to {MAX_PLANETS} bodies")
        self._init_params(np.concatenate([[k2, m0], masses]),
                          ("k2", "m0") + tuple(f"m{i + 1}" for i in range(N)))
        self.data.N = N
        self.data.p = self._params_ptr()
        self.n_states = 6 * N
        _bind(self, _planets_fn, &self.data)


# --- DETEST class D / Kepler -------------------------------------------------------

cdef int _kepler_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    cdef double r2 = y[0] * y[0] + y[1] * y[1]
    cdef double f = -p[0] / (r2 * sqrt(r2))
    dydt[0] = y[2]
    dydt[1] = y[3]
    dydt[2] = f * y[0]
    dydt[3] = f * y[1]
    return 0


cdef class Kepler2D(OdeSystem):
    """Planar two-body problem ``q'' = -mu q / |q|^3``; state ``[x, y, vx, vy]``.

    Parameter: ``mu``.
    """

    def __init__(self, double mu=1.0):
        self.n_states = 4
        self._init_params([mu], ("mu",))
        _bind(self, _kepler_fn, self._params_ptr())


# --- DETEST class E ----------------------------------------------------------------

cdef int _e1_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef double x = t + 1.0
    dydt[0] = y[1]
    dydt[1] = -(y[1] / x + (1.0 - 0.25 / (x * x)) * y[0])
    return 0


cdef int _vdp_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    dydt[0] = y[1]
    dydt[1] = p[0] * (1.0 - y[0] * y[0]) * y[1] - y[0]
    return 0


cdef int _e3_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = y[1]
    dydt[1] = y[0] * y[0] * y[0] / 6.0 - y[0] + 2.0 * sin(2.78535 * t)
    return 0


cdef int _e4_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = y[1]
    dydt[1] = 0.032 - 0.4 * y[1] * y[1]
    return 0


cdef int _e5_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    dydt[0] = y[1]
    dydt[1] = sqrt(1.0 + y[1] * y[1]) / (25.0 - t)
    return 0


cdef class DetestE1(OdeSystem):
    """Bessel equation of order 1/2 in ``x = t + 1``."""

    def __init__(self):
        self.n_states = 2
        _bind(self, _e1_fn, NULL)


cdef class VanDerPol(OdeSystem):
    """``y'' = mu (1 - y^2) y' - y`` (DETEST E2: mu = 1). Parameter: ``mu``."""

    def __init__(self, double mu=1.0):
        self.n_states = 2
        self._init_params([mu], ("mu",))
        _bind(self, _vdp_fn, self._params_ptr())


cdef class DetestE3(OdeSystem):
    """Forced Duffing oscillator."""

    def __init__(self):
        self.n_states = 2
        _bind(self, _e3_fn, NULL)


cdef class DetestE4(OdeSystem):
    def __init__(self):
        self.n_states = 2
        _bind(self, _e4_fn, NULL)


cdef class DetestE5(OdeSystem):
    def __init__(self):
        self.n_states = 2
        _bind(self, _e5_fn, NULL)


# --- Hairer-Norsett-Wanner problems --------------------------------------------------

cdef int _arenstorf_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    cdef double mu = p[0], mu1 = 1.0 - mu
    cdef double a = (y[0] + mu) ** 2 + y[1] ** 2
    cdef double b = (y[0] - mu1) ** 2 + y[1] ** 2
    cdef double D1 = a * sqrt(a)
    cdef double D2 = b * sqrt(b)
    dydt[0] = y[2]
    dydt[1] = y[3]
    dydt[2] = y[0] + 2.0 * y[3] - mu1 * (y[0] + mu) / D1 - mu * (y[0] - mu1) / D2
    dydt[3] = y[1] - 2.0 * y[2] - mu1 * y[1] / D1 - mu * y[1] / D2
    return 0


cdef int _brusselator_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    cdef double y1y1y2 = y[0] * y[0] * y[1]
    dydt[0] = p[0] + y1y1y2 - (p[1] + 1.0) * y[0]
    dydt[1] = p[1] * y[0] - y1y1y2
    return 0


cdef int _lorenz_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    dydt[0] = p[0] * (y[1] - y[0])
    dydt[1] = y[0] * (p[1] - y[2]) - y[1]
    dydt[2] = y[0] * y[1] - p[2] * y[2]
    return 0


cdef int _robertson_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef double r1 = 0.04 * y[0]
    cdef double r2 = 1e4 * y[1] * y[2]
    cdef double r3 = 3e7 * y[1] * y[1]
    dydt[0] = -r1 + r2
    dydt[1] = r1 - r2 - r3
    dydt[2] = r3
    return 0


cdef int _pendulum_fn(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    dydt[0] = y[1]
    dydt[1] = -p[0] * sin(y[0])
    return 0


cdef class Arenstorf(OdeSystem):
    """Restricted three-body problem in the rotating frame; state ``[x, y, vx, vy]``.

    Parameter: ``mu`` (mass ratio).
    """

    def __init__(self, double mu=0.012277471):
        self.n_states = 4
        self._init_params([mu], ("mu",))
        _bind(self, _arenstorf_fn, self._params_ptr())


cdef class Brusselator(OdeSystem):
    """``y1' = A + y1^2 y2 - (B + 1) y1``, ``y2' = B y1 - y1^2 y2``. Parameters: ``A``, ``B``."""

    def __init__(self, double A=1.0, double B=3.0):
        self.n_states = 2
        self._init_params([A, B], ("A", "B"))
        _bind(self, _brusselator_fn, self._params_ptr())


cdef class Lorenz(OdeSystem):
    """Lorenz system. Parameters: ``sigma``, ``rho``, ``beta``."""

    def __init__(self, double sigma=10.0, double rho=28.0, double beta=8.0 / 3.0):
        self.n_states = 3
        self._init_params([sigma, rho, beta], ("sigma", "rho", "beta"))
        _bind(self, _lorenz_fn, self._params_ptr())


cdef class Robertson(OdeSystem):
    """Robertson's stiff chemical kinetics."""

    def __init__(self):
        self.n_states = 3
        _bind(self, _robertson_fn, NULL)


cdef class Pendulum(OdeSystem):
    """``theta'' = -(g / L) sin(theta)``; state ``[theta, omega]``. Parameter: ``w2`` (= g / L)."""

    def __init__(self, double w2=1.0):
        self.n_states = 2
        self._init_params([w2], ("w2",))
        _bind(self, _pendulum_fn, self._params_ptr())


# --- native events -------------------------------------------------------------------

cdef struct _dot_ctx:
    Py_ssize_t m
    const Py_ssize_t* left
    const Py_ssize_t* right
    const double* value      # -> params[0]


cdef int _dot_fn(double t, const double* y, Py_ssize_t n, void* ctx, double* g) noexcept nogil:
    cdef _dot_ctx* c = <_dot_ctx*> ctx
    cdef Py_ssize_t k
    cdef double acc = -c.value[0]
    for k in range(c.m):
        acc += y[c.left[k]] * y[c.right[k]]
    g[0] = acc
    return 0


cdef class DotProductEvent(EventFunction):
    """``g = sum_k y[left[k]] * y[right[k]] - value``.

    With ``left`` = positions and ``right`` = velocities this is ``r . v``, whose
    zeros are the apsides of an orbit (direction +1: periapsis, -1: apoapsis).
    Parameter: ``value``.
    """

    cdef Py_ssize_t[::1] left
    cdef Py_ssize_t[::1] right
    cdef _dot_ctx data

    def __init__(self, left, right, double value=0.0, terminal=False, double direction=0.0):
        super().__init__(terminal, direction)
        left = np.array(left, dtype=np.intp, copy=True).reshape(-1)
        right = np.array(right, dtype=np.intp, copy=True).reshape(-1)
        if left.shape != right.shape or left.size == 0:
            raise ValueError("left and right must be non-empty and the same length")
        if left.min() < 0 or right.min() < 0:
            raise ValueError("indices must be non-negative")
        self.left = left
        self.right = right
        self._init_params([value], ("value",))
        self.data.m = left.shape[0]
        self.data.left = &self.left[0]
        self.data.right = &self.right[0]
        self.data.value = self._params_ptr()
        self.c_fn = _dot_fn
        self.c_ctx = &self.data

    @property
    def value(self):
        return self._params[0]

    cdef int check_size(self, Py_ssize_t n) except -1:
        cdef Py_ssize_t k
        for k in range(self.data.m):
            if self.left[k] >= n or self.right[k] >= n:
                raise IndexError("event index out of range")
        return 0


cdef int _time_fn(double t, const double* y, Py_ssize_t n, void* ctx, double* g) noexcept nogil:
    g[0] = t - (<const double*> ctx)[0]
    return 0


cdef class TimeEvent(EventFunction):
    """``g = t - t_event``: fires at a prescribed time. Parameter: ``t_event``."""

    def __init__(self, double t_event, terminal=False, double direction=0.0):
        super().__init__(terminal, direction)
        self._init_params([t_event], ("t_event",))
        self.c_fn = _time_fn
        self.c_ctx = self._params_ptr()

    @property
    def t_event(self):
        return self._params[0]
