# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Native N-body gravity right-hand side for the Cython integrators.

A hand-written stand-in for the SymPy-generated C kernel planned for later
milestones. It uses the same packed state layout as
:mod:`fast_numerical_integration.three_body`::

    y = [ r_1 (d), ..., r_N (d),  v_1 (d), ..., v_N (d) ]
"""

from libc.math cimport sqrt

import numpy as np

from fast_numerical_integration.integrators cimport OdeSystem


cdef class NBodyGravity(OdeSystem):
    """Newtonian gravity for ``N`` point masses in ``n_dims`` dimensions.

    ``softening2`` is added to every squared separation, as in
    :func:`fast_numerical_integration.three_body.three_body_rhs`.
    """

    cdef readonly Py_ssize_t n_bodies
    cdef readonly Py_ssize_t n_dims
    cdef readonly double G
    cdef readonly double softening2
    cdef double[::1] masses

    def __init__(self, masses, double G=1.0, Py_ssize_t n_dims=2,
                 double softening2=0.0):
        self.masses = np.ascontiguousarray(masses, dtype=np.float64).copy()
        self.n_bodies = self.masses.shape[0]
        self.n_dims = n_dims
        self.G = G
        self.softening2 = softening2
        self.n_states = 2 * self.n_bodies * n_dims

    @property
    def masses_array(self):
        return np.asarray(self.masses).copy()

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        cdef Py_ssize_t N = self.n_bodies, d = self.n_dims
        cdef Py_ssize_t half = N * d
        cdef Py_ssize_t i, j, k
        cdef const double* pos = y
        cdef double* acc = dydt + half
        cdef double dist2, inv_d3, wi, wj, dx
        cdef double r[8]   # separation vector; supports n_dims <= 8

        if d > 8:
            raise ValueError("NBodyGravity supports at most 8 dimensions")

        # d(position)/dt = velocity
        for i in range(half):
            dydt[i] = y[half + i]
            acc[i] = 0.0

        # Pairwise forces, each pair visited once (Newton's third law).
        for i in range(N):
            for j in range(i + 1, N):
                dist2 = self.softening2
                for k in range(d):
                    dx = pos[j * d + k] - pos[i * d + k]
                    r[k] = dx
                    dist2 += dx * dx
                inv_d3 = self.G / (dist2 * sqrt(dist2))
                wi = self.masses[j] * inv_d3
                wj = self.masses[i] * inv_d3
                for k in range(d):
                    acc[i * d + k] += wi * r[k]
                    acc[j * d + k] -= wj * r[k]
        return 0
