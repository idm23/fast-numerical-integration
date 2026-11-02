# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Native N-body gravity right-hand side for the Cython integrators.

A hand-written stand-in for the SymPy-generated C kernel planned for later
milestones. It uses the same packed state layout as
:mod:`fast_numerical_integration.three_body`::

    y = [ r_1 (d), ..., r_N (d),  v_1 (d), ..., v_N (d) ]

The gravitational constant and the masses are runtime parameters
(``params = [G, m_1, ..., m_N]``), and the RHS is a GIL-free C function, so
the system also works with the C-level ``fni_integrate`` API.
"""

from libc.math cimport sqrt

import numpy as np

from fast_numerical_integration.integrators cimport OdeSystem

cdef enum:
    MAX_DIMS = 8


cdef struct _nbody_ctx:
    Py_ssize_t n_bodies
    Py_ssize_t n_dims
    double softening2
    const double* p          # [G, m_1, ..., m_N]


cdef int _nbody_fn(double t, const double* y, double* dydt, Py_ssize_t n,
                   void* ctx) noexcept nogil:
    cdef _nbody_ctx* c = <_nbody_ctx*> ctx
    cdef Py_ssize_t N = c.n_bodies, d = c.n_dims
    cdef Py_ssize_t half = N * d
    cdef Py_ssize_t i, j, k
    cdef const double* pos = y
    cdef const double* masses = c.p + 1
    cdef double G = c.p[0]
    cdef double* acc = dydt + half
    cdef double dist2, inv_d3, wi, wj, dx
    cdef double r[MAX_DIMS]

    # d(position)/dt = velocity
    for i in range(half):
        dydt[i] = y[half + i]
        acc[i] = 0.0

    # Pairwise forces, each pair visited once (Newton's third law).
    for i in range(N):
        for j in range(i + 1, N):
            dist2 = c.softening2
            for k in range(d):
                dx = pos[j * d + k] - pos[i * d + k]
                r[k] = dx
                dist2 += dx * dx
            inv_d3 = G / (dist2 * sqrt(dist2))
            wi = masses[j] * inv_d3
            wj = masses[i] * inv_d3
            for k in range(d):
                acc[i * d + k] += wi * r[k]
                acc[j * d + k] -= wj * r[k]
    return 0


cdef class NBodyGravity(OdeSystem):
    """Newtonian gravity for ``N`` point masses in ``n_dims`` dimensions.

    ``softening2`` is added to every squared separation, as in
    :func:`fast_numerical_integration.three_body.three_body_rhs`.
    Runtime parameters: ``params = [G, m_1, ..., m_N]``.
    """

    cdef _nbody_ctx data

    def __init__(self, masses, double G=1.0, Py_ssize_t n_dims=2, double softening2=0.0):
        masses = np.ascontiguousarray(masses, dtype=np.float64).reshape(-1)
        if not 1 <= n_dims <= MAX_DIMS:
            raise ValueError(f"NBodyGravity supports 1 to {MAX_DIMS} dimensions")
        N = masses.shape[0]
        self._init_params(np.concatenate([[G], masses]),
                          ("G",) + tuple(f"m{i + 1}" for i in range(N)))
        self.data.n_bodies = N
        self.data.n_dims = n_dims
        self.data.softening2 = softening2
        self.data.p = self._params_ptr()
        self.n_states = 2 * N * n_dims
        self.c_fn = _nbody_fn
        self.c_ctx = &self.data

    @property
    def n_bodies(self):
        return self.data.n_bodies

    @property
    def n_dims(self):
        return self.data.n_dims

    @property
    def softening2(self):
        return self.data.softening2

    @property
    def G(self):
        return self._params[0]

    @property
    def masses_array(self):
        return self.params[1:]
