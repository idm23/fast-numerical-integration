# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Batch propagation on the GIL-free C API.

:func:`integrate_batch` propagates many initial states, each over its own time
span (as in multiple shooting), with one ``nogil`` loop over
``fni_integrate``. It is also the reference example of *using* the C API from
another Cython module::

    from fast_numerical_integration.integrators cimport (
        fni_integrate, fni_options, fni_stats, FNI_DOP853, FNI_OK)

    cdef int my_rhs(double t, const double* y, double* dydt, Py_ssize_t n,
                    void* ctx) noexcept nogil:
        ...

    with nogil:
        rc = fni_integrate(FNI_DOP853, my_rhs, ctx, n, t0, y0, t1, &opt,
                           NULL, 0, y_out, NULL, NULL, NULL, 0, &stats)
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fast_numerical_integration.integrators cimport (
    FNI_DOP853,
    FNI_OK,
    FNI_RK23,
    FNI_RK45,
    OdeSystem,
    fni_integrate,
    fni_options,
    fni_rhs_fn,
    fni_stats,
)

_METHODS = {"RK23": FNI_RK23, "RK45": FNI_RK45, "DOP853": FNI_DOP853}


cdef int _run_rows(int method, fni_rhs_fn fn, void* ctx, Py_ssize_t n,
                   const double[::1] t0, const double[::1] t1, const double[:, ::1] y0,
                   double[:, ::1] y1, double[::1] t_final, int[::1] status,
                   Py_ssize_t[::1] nfev, const fni_options* opt,
                   Py_ssize_t start, Py_ssize_t stop) noexcept nogil:
    cdef Py_ssize_t k
    cdef fni_stats stats
    cdef int rc
    for k in range(start, stop):
        rc = fni_integrate(method, fn, ctx, n, t0[k], &y0[k, 0], t1[k], opt,
                           NULL, 0, &y1[k, 0], NULL, NULL, NULL, 0, &stats)
        if rc != FNI_OK:
            return rc
        t_final[k] = stats.t_final
        status[k] = stats.status
        nfev[k] = stats.nfev
    return FNI_OK


def integrate_batch(OdeSystem system not None, t0, t1, y0, method="DOP853", *, double rtol=1e-3,
                    double atol=1e-6, double max_step=np.inf, n_threads=1):
    """Propagate each row of ``y0`` from ``t0[k]`` to ``t1[k]``, without the GIL.

    Parameters
    ----------
    system : OdeSystem
        Must provide a native ``c_fn`` (``system.has_native_fn``). It is shared
        by all rows (and threads), so its kernel must be reentrant.
    t0, t1 : float or array_like, shape (m,)
        Start and end time per row (scalars broadcast).
    y0 : array_like, shape (m, n)
        Initial states.
    n_threads : int
        Split the rows over this many threads. Each chunk runs ``nogil``, so
        threads execute truly in parallel.

    Returns
    -------
    dict
        ``y`` (m, n) final states, ``t`` (m,) final times, ``status`` (m,)
        (0 finished, -1 step size too small), ``nfev`` (m,).
    """
    cdef fni_options opt
    cdef Py_ssize_t m, n
    cdef int meth

    if not system.has_native_fn:
        raise TypeError("integrate_batch needs a system with a native c_fn")
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {list(_METHODS)}")
    meth = _METHODS[method]
    y0_arr = np.array(y0, dtype=np.float64, copy=True, ndmin=2)
    m, n = y0_arr.shape
    system.check_size(n)
    t0_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(t0, dtype=np.float64), (m,))).copy()
    t1_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(t1, dtype=np.float64), (m,))).copy()
    if not max_step > 0:
        raise ValueError("`max_step` must be positive.")

    y1 = np.zeros((m, max(n, 1)))
    t_final = np.zeros(m)
    status = np.zeros(m, dtype=np.intc)
    nfev = np.zeros(m, dtype=np.intp)
    if m == 0:
        return {"y": y1[:, :n], "t": t_final, "status": status, "nfev": nfev}
    if n == 0:
        y0_arr = np.zeros((m, 1))

    opt.rtol = rtol
    opt.rtol_vec = NULL
    opt.atol = atol
    opt.atol_vec = NULL
    opt.first_step = -1.0
    opt.max_step = max_step

    cdef const double[::1] t0_v = t0_arr
    cdef const double[::1] t1_v = t1_arr
    cdef const double[:, ::1] y0_v = y0_arr
    cdef double[:, ::1] y1_v = y1
    cdef double[::1] tf_v = t_final
    cdef int[::1] st_v = status
    cdef Py_ssize_t[::1] nf_v = nfev
    cdef fni_rhs_fn fn = system.c_fn
    cdef void* ctx = system.c_ctx

    def run(Py_ssize_t start, Py_ssize_t stop):
        cdef int rc
        with nogil:
            rc = _run_rows(meth, fn, ctx, n, t0_v, t1_v, y0_v, y1_v, tf_v, st_v, nf_v,
                           &opt, start, stop)
        if rc != FNI_OK:
            raise RuntimeError(f"integration failed with code {rc}")

    n_threads = max(1, min(int(n_threads), m))
    if n_threads == 1:
        run(0, m)
    else:
        bounds = np.linspace(0, m, n_threads + 1).astype(int)
        with ThreadPoolExecutor(n_threads) as pool:
            for fut in [pool.submit(run, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]:
                fut.result()
    return {"y": y1[:, :n], "t": t_final, "status": status, "nfev": nfev}
