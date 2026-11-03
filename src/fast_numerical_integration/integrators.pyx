# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython explicit Runge-Kutta integrators: RK23, RK45 and DOP853.

A step-for-step reimplementation of :func:`scipy.integrate.solve_ivp` with the
``RK23``, ``RK45`` and ``DOP853`` methods. It uses the same tableaus, the same
initial-step heuristic, the same error norms and the same step-size controller
(safety 0.9, factors in [0.2, 10]), so it takes the same steps and makes the
same number of RHS evaluations as SciPy. Differences are limited to
floating-point summation order.

Architecture
------------
All numerics live in a plain-C, GIL-free core (the ``fni_*`` functions
declared in ``integrators.pxd``): a solver struct, single-step and
dense-output routines, event detection with Brent root finding, and
``fni_integrate``, which runs a whole integration. Right-hand sides and events
are C function pointers plus a context pointer. Other Cython modules can
``cimport`` the core and call it inside ``with nogil`` (multiple shooting,
threads, tight loops).

The Python layer (:func:`solve_ivp`, the stepper classes, :func:`integrate`)
is a thin wrapper over that single core, so both paths take bit-identical
steps. :class:`OdeSystem` / :class:`EventFunction` subclasses that provide a
``nogil`` C function (``c_fn``) are called directly. Python callables and
Cython ``eval`` overrides are reached through a trampoline that holds the GIL.

Supported ``solve_ivp`` features: ``t_eval``, ``dense_output`` (returns an
:class:`OdeSolution`), ``events`` (terminal as a bool or count, plus direction;
roots found by Brent's method on the dense interpolant; see the event scan
below for detection within a step), ``vectorized``,
``args``, ``first_step``, ``max_step``, and scalar or per-component
``rtol``/``atol``. Native systems and events also carry a runtime ``params``
vector for continuation without recompiling.

Not supported: complex-valued ``y0``.
"""

cimport numpy as cnp
from cpython.exc cimport PyErr_Occurred
from libc.math cimport fabs, sqrt, pow, nextafter, INFINITY, NAN
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import operator
import warnings

import numpy as np

from fast_numerical_integration import tableaus

cnp.import_array()

# Step-size controller constants (identical to scipy.integrate._ivp.rk).
cdef double SAFETY = 0.9
cdef double MIN_FACTOR = 0.2
cdef double MAX_FACTOR = 10.0

cdef double EPS = 2.220446049250313e-16   # np.finfo(float).eps
cdef int BRENTQ_MAXITER = 100

_STATUS_NAMES = {FNI_RUNNING: "running", FNI_FINISHED: "finished", FNI_FAILED: "failed"}

MESSAGES = {
    0: "The solver successfully reached the end of the integration interval.",
    1: "A termination event occurred.",
    -2: "Reset limit reached: more than `max_resets` resets.",
    -3: "Zeno behaviour: two resets closer together than `min_dwell`.",
}
MSG_TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

METHOD_IDS = {"RK23": FNI_RK23, "RK45": FNI_RK45, "DOP853": FNI_DOP853}


cdef inline cnp.ndarray _empty1d(Py_ssize_t n):
    cdef cnp.npy_intp dims[1]
    dims[0] = n
    return cnp.PyArray_EMPTY(1, dims, cnp.NPY_FLOAT64, 0)


cdef inline cnp.ndarray _empty2d(Py_ssize_t rows, Py_ssize_t cols):
    cdef cnp.npy_intp dims[2]
    dims[0] = rows
    dims[1] = cols
    return cnp.PyArray_EMPTY(2, dims, cnp.NPY_FLOAT64, 0)


cdef inline cnp.ndarray _array_from(const double* src, Py_ssize_t n):
    """Fresh 1-D ndarray holding a copy of ``n`` doubles."""
    out = _empty1d(n)
    if n > 0:
        memcpy(cnp.PyArray_DATA(out), src, n * sizeof(double))
    return out


# =============================================================================
# Tableaus (module-level C structs pointing into kept-alive NumPy arrays)
# =============================================================================

cdef fni_tableau _TABLEAUS[3]
_TABLEAU_ARRAYS = []   # keeps the coefficient buffers alive


cdef const double* _keep(arr):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    _TABLEAU_ARRAYS.append(a)
    return <const double*> cnp.PyArray_DATA(a)


cdef void _init_tableaus():
    cdef fni_tableau* t = &_TABLEAUS[FNI_RK23]
    t.method = FNI_RK23
    t.order = tableaus.RK23_ORDER
    t.error_estimator_order = tableaus.RK23_ERROR_ESTIMATOR_ORDER
    t.n_stages = tableaus.RK23_N_STAGES
    t.n_k_rows = t.n_stages + 1
    t.dense_kind = FNI_DENSE_RK
    t.n_dense_coef = tableaus.RK23_P.shape[1]
    t.A = _keep(tableaus.RK23_A)
    t.a_stride = tableaus.RK23_A.shape[1]
    t.B = _keep(tableaus.RK23_B)
    t.C = _keep(tableaus.RK23_C)
    t.E = _keep(tableaus.RK23_E)
    t.E3 = NULL
    t.P = _keep(tableaus.RK23_P)
    t.D = NULL

    t = &_TABLEAUS[FNI_RK45]
    t.method = FNI_RK45
    t.order = tableaus.RK45_ORDER
    t.error_estimator_order = tableaus.RK45_ERROR_ESTIMATOR_ORDER
    t.n_stages = tableaus.RK45_N_STAGES
    t.n_k_rows = t.n_stages + 1
    t.dense_kind = FNI_DENSE_RK
    t.n_dense_coef = tableaus.RK45_P.shape[1]
    t.A = _keep(tableaus.RK45_A)
    t.a_stride = tableaus.RK45_A.shape[1]
    t.B = _keep(tableaus.RK45_B)
    t.C = _keep(tableaus.RK45_C)
    t.E = _keep(tableaus.RK45_E)
    t.E3 = NULL
    t.P = _keep(tableaus.RK45_P)
    t.D = NULL

    t = &_TABLEAUS[FNI_DOP853]
    t.method = FNI_DOP853
    t.order = tableaus.DOP853_ORDER
    t.error_estimator_order = tableaus.DOP853_ERROR_ESTIMATOR_ORDER
    t.n_stages = tableaus.DOP853_N_STAGES
    t.n_k_rows = tableaus.DOP853_N_STAGES_EXTENDED
    t.dense_kind = FNI_DENSE_DOP853
    t.n_dense_coef = tableaus.DOP853_INTERPOLATOR_POWER
    # The full 16x16 matrix: its leading 12x12 block is the main tableau and
    # rows 13-15 are the dense-output stages.
    t.A = _keep(tableaus.DOP853_A)
    t.a_stride = tableaus.DOP853_A.shape[1]
    t.B = _keep(tableaus.DOP853_B)
    t.C = _keep(tableaus.DOP853_C)
    t.E = _keep(tableaus.DOP853_E5)
    t.E3 = _keep(tableaus.DOP853_E3)
    t.P = NULL
    t.D = _keep(tableaus.DOP853_D)


_init_tableaus()


cdef const fni_tableau* fni_get_tableau(int method) noexcept nogil:
    if method < FNI_RK23 or method > FNI_DOP853:
        return NULL
    return &_TABLEAUS[method]


# =============================================================================
# Plain-C core
# =============================================================================

cdef inline double _rms_norm_scaled(const double* x, const double* scale, Py_ssize_t n) noexcept nogil:
    """RMS norm of ``x / scale`` (scipy's ``norm(x / scale)``)."""
    cdef Py_ssize_t i
    cdef double v, acc = 0.0
    for i in range(n):
        v = x[i] / scale[i]
        acc += v * v
    return sqrt(acc / n)


cdef void _interp_eval(int kind, Py_ssize_t ncoef, Py_ssize_t n, double t_old,
                       double h, const double* y_old, const double* coef,
                       double t, double* out) noexcept nogil:
    """Evaluate one step's dense-output polynomial at ``t``.

    ``coef`` is (ncoef, n) row-major. ``h == 0`` is the constant interpolant
    scipy uses for zero-length steps.
    """
    cdef Py_ssize_t i, k
    cdef double x, acc
    if h == 0.0:
        if n > 0:
            memcpy(out, y_old, n * sizeof(double))
        return
    x = (t - t_old) / h
    if kind == FNI_DENSE_RK:
        # Horner form of h * sum_k coef[k] x^(k+1).
        for i in range(n):
            acc = coef[(ncoef - 1) * n + i]
            for k in range(ncoef - 2, -1, -1):
                acc = acc * x + coef[k * n + i]
            out[i] = y_old[i] + h * (acc * x)
    else:
        # scipy's Dop853DenseOutput: alternate multiplication by x and (1 - x).
        for i in range(n):
            acc = 0.0
            for k in range(ncoef - 1, -1, -1):
                acc += coef[k * n + i]
                if (ncoef - 1 - k) % 2 == 0:
                    acc *= x
                else:
                    acc *= 1.0 - x
            out[i] = acc + y_old[i]


cdef inline int _fun(fni_solver* s, double t, const double* y, double* dydt) noexcept nogil:
    s.nfev += 1
    if s.fun(t, y, dydt, s.n, s.ctx) != 0:
        return FNI_ERR_CALLBACK
    return FNI_OK


cdef int _select_initial_step(fni_solver* s, double* h_out) noexcept nogil:
    """Hairer-Norsett-Wanner initial step heuristic (scipy's select_initial_step)."""
    cdef Py_ssize_t i, n = s.n
    cdef double interval_length = fabs(s.t_bound - s.t)
    cdef double d0, d1, d2, h0, h1
    cdef double* y0 = s.y
    cdef double* f0 = s.f
    cdef double* scale = s.scale
    cdef double* y1 = s.y_new
    cdef double* f1 = s.K          # free scratch before the first step
    cdef double* diff = s.y_stage
    cdef int rc

    if interval_length == 0.0:
        h_out[0] = 0.0
        return FNI_OK

    for i in range(n):
        scale[i] = s.atol[i] + fabs(y0[i]) * s.rtol[i]
    d0 = _rms_norm_scaled(y0, scale, n)
    d1 = _rms_norm_scaled(f0, scale, n)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, interval_length)

    for i in range(n):
        y1[i] = y0[i] + h0 * s.direction * f0[i]
    rc = _fun(s, s.t + h0 * s.direction, y1, f1)
    if rc != FNI_OK:
        return rc
    for i in range(n):
        diff[i] = f1[i] - f0[i]
    d2 = _rms_norm_scaled(diff, scale, n) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = pow(0.01 / max(d1, d2), 1.0 / (s.tab.error_estimator_order + 1))

    h_out[0] = min(100 * h0, h1, interval_length, s.max_step)
    return FNI_OK


cdef int fni_solver_init(fni_solver* s, int method, fni_rhs_fn fun, void* ctx, Py_ssize_t n,
                         double t0, const double* y0, double t_bound,
                         const fni_options* opt) noexcept nogil:
    """Set up a solver at (t0, y0); evaluates f(t0, y0) and the initial step.

    Always call :func:`fni_solver_free` afterwards, even if this fails.
    """
    cdef const fni_tableau* tab = fni_get_tableau(method)
    cdef Py_ssize_t i, total
    cdef double r, a
    s.mem = NULL
    if tab == NULL or n < 0 or fun == NULL or not (opt.max_step > 0):
        return FNI_ERR_ARG
    if not (opt.event_detection == FNI_EVENTS_ENDPOINTS or (
            opt.event_detection == FNI_EVENTS_INTERIOR and opt.n_interior >= 1)):
        return FNI_ERR_ARG
    total = (8 + tab.n_k_rows + tab.n_dense_coef) * n
    s.mem = <double*> malloc((total if total > 0 else 1) * sizeof(double))
    if s.mem == NULL:
        return FNI_ERR_ALLOC
    s.y = s.mem
    s.y_old = s.y + n
    s.f = s.y_old + n
    s.y_new = s.f + n
    s.y_stage = s.y_new + n
    s.scale = s.y_stage + n
    s.atol = s.scale + n
    s.rtol = s.atol + n
    s.K = s.rtol + n
    s.dense = s.K + tab.n_k_rows * n

    for i in range(n):
        s.y[i] = y0[i]
        r = opt.rtol_vec[i] if opt.rtol_vec != NULL else opt.rtol
        if r < 100 * EPS:
            r = 100 * EPS
        s.rtol[i] = r
        a = opt.atol_vec[i] if opt.atol_vec != NULL else opt.atol
        if a < 0:
            return FNI_ERR_ARG
        s.atol[i] = a

    s.tab = tab
    s.fun = fun
    s.ctx = ctx
    s.n = n
    s.t = t0
    s.t_bound = t_bound
    s.direction = 1.0 if t_bound >= t0 else -1.0
    s.max_step = opt.max_step
    s.event_detection = opt.event_detection
    s.n_interior = opt.n_interior
    s.t_old = NAN
    s.has_t_old = False
    s.dense_valid = False
    s.error_exponent = -1.0 / (tab.error_estimator_order + 1)
    s.h_previous = NAN
    s.nfev = 0
    s.n_accepted = 0
    s.n_rejected = 0
    s.status = FNI_RUNNING

    if n == 0:
        s.h_abs = INFINITY
        return FNI_OK

    cdef int rc = _fun(s, s.t, s.y, s.f)
    if rc != FNI_OK:
        return rc
    if opt.first_step > 0:
        if opt.first_step > fabs(t_bound - t0):
            return FNI_ERR_ARG
        s.h_abs = opt.first_step
        return FNI_OK
    return _select_initial_step(s, &s.h_abs)


cdef void fni_solver_free(fni_solver* s) noexcept nogil:
    if s.mem != NULL:
        free(s.mem)
    s.mem = NULL


cdef int fni_solver_restart(fni_solver* s, double t, const double* y,
                            bint reselect_step) noexcept nogil:
    """Continue a running solver from a new point (t, y), e.g. after a reset map.

    Discards the last step and its dense output, then re-evaluates f(t, y).
    The derivative is discontinuous across a reset, so the FSAL value cannot
    be reused. The next step starts from the controller's current step-size
    proposal, or from the initial-step heuristic if ``reselect_step``.
    Statistics (nfev, step counts) accumulate.
    """
    cdef int rc
    if s.direction * (t - s.t_bound) > 0:
        return FNI_ERR_ARG
    memcpy(s.y, y, s.n * sizeof(double))
    s.t = t
    s.t_old = NAN
    s.has_t_old = False
    s.dense_valid = False
    s.status = FNI_RUNNING
    if s.n == 0:
        return FNI_OK
    rc = _fun(s, s.t, s.y, s.f)
    if rc != FNI_OK:
        return rc
    if reselect_step and s.t != s.t_bound:
        return _select_initial_step(s, &s.h_abs)
    return FNI_OK


cdef int _rk_stages(fni_solver* s, double h) noexcept nogil:
    """Fill K with all stages for step ``h``; write the solution into y_new.

    Each stage argument is built with column sweeps ``acc += a_sj * K[j, :]``
    over contiguous rows, skipping zero tableau entries (DOP853's A is
    roughly half zeros).
    """
    cdef const fni_tableau* tab = s.tab
    cdef Py_ssize_t st, j, i, n = s.n, ns = tab.n_stages
    cdef double a
    cdef double* y = s.y
    cdef double* acc = s.y_stage
    cdef double* y_new = s.y_new
    cdef double* Kj
    cdef int rc

    memcpy(s.K, s.f, n * sizeof(double))

    for st in range(1, ns):
        for i in range(n):
            acc[i] = 0.0
        for j in range(st):
            a = tab.A[st * tab.a_stride + j]
            if a != 0.0:
                Kj = s.K + j * n
                for i in range(n):
                    acc[i] += a * Kj[i]
        for i in range(n):
            acc[i] = y[i] + h * acc[i]
        rc = _fun(s, s.t + tab.C[st] * h, acc, s.K + st * n)
        if rc != FNI_OK:
            return rc

    for i in range(n):
        acc[i] = 0.0
    for j in range(ns):
        a = tab.B[j]
        if a != 0.0:
            Kj = s.K + j * n
            for i in range(n):
                acc[i] += a * Kj[i]
    for i in range(n):
        y_new[i] = y[i] + h * acc[i]

    # FSAL: the derivative at the new point is row n_stages of K.
    return _fun(s, s.t + h, y_new, s.K + ns * n)


cdef double _error_norm(const fni_solver* s, double h) noexcept nogil:
    """Embedded error norm: RMS of ``h K^T E / scale`` (DOP853: blended 5th/3rd)."""
    cdef const fni_tableau* tab = s.tab
    cdef Py_ssize_t i, j, n = s.n, nk = tab.n_stages + 1
    cdef double err, acc = 0.0
    cdef double e5, e3, k, inv_scale
    cdef double err5_norm_2 = 0.0, err3_norm_2 = 0.0, denom
    if tab.E3 == NULL:
        for i in range(n):
            err = 0.0
            for j in range(nk):
                err += s.K[j * n + i] * tab.E[j]
            err = h * err / s.scale[i]
            acc += err * err
        return sqrt(acc / n)
    for i in range(n):
        e5 = 0.0
        e3 = 0.0
        for j in range(nk):
            k = s.K[j * n + i]
            e5 += k * tab.E[j]
            e3 += k * tab.E3[j]
        inv_scale = 1.0 / s.scale[i]
        e5 *= inv_scale
        e3 *= inv_scale
        err5_norm_2 += e5 * e5
        err3_norm_2 += e3 * e3
    if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
        return 0.0
    denom = err5_norm_2 + 0.01 * err3_norm_2
    return fabs(h) * err5_norm_2 / sqrt(denom * n)


cdef int _step_impl(fni_solver* s, bint* accepted) noexcept nogil:
    """One accepted step with rejection/retry (scipy's RungeKutta._step_impl)."""
    cdef Py_ssize_t i, n = s.n
    cdef double t = s.t
    cdef double min_step = 10 * fabs(nextafter(t, s.direction * INFINITY) - t)
    cdef double h_abs, h, t_new, error_norm, factor, yi, yn
    cdef bint step_rejected = False
    cdef double* tmp
    cdef int rc

    accepted[0] = False
    if s.h_abs > s.max_step:
        h_abs = s.max_step
    elif s.h_abs < min_step:
        h_abs = min_step
    else:
        h_abs = s.h_abs

    while True:
        if h_abs < min_step:
            return FNI_OK

        h = h_abs * s.direction
        t_new = t + h
        if s.direction * (t_new - s.t_bound) > 0:
            t_new = s.t_bound
        h = t_new - t
        h_abs = fabs(h)

        rc = _rk_stages(s, h)
        if rc != FNI_OK:
            return rc

        for i in range(n):
            yi = fabs(s.y[i])
            yn = fabs(s.y_new[i])
            s.scale[i] = s.atol[i] + (yi if yi > yn else yn) * s.rtol[i]
        error_norm = _error_norm(s, h)

        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * pow(error_norm, s.error_exponent))
            if step_rejected:
                factor = min(1.0, factor)
            h_abs *= factor
            break
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * pow(error_norm, s.error_exponent))
            step_rejected = True
            s.n_rejected += 1

    # Accept: keep y_old for dense output, swap y <-> y_new, take f from FSAL.
    s.h_previous = h
    memcpy(s.y_old, s.y, n * sizeof(double))
    tmp = s.y
    s.y = s.y_new
    s.y_new = tmp
    memcpy(s.f, s.K + s.tab.n_stages * n, n * sizeof(double))
    s.t = t_new
    s.h_abs = h_abs
    s.n_accepted += 1
    accepted[0] = True
    return FNI_OK


cdef int fni_solver_step(fni_solver* s) noexcept nogil:
    """Advance one accepted step (scipy's OdeSolver.step). Updates ``s.status``."""
    cdef double t = s.t
    cdef bint accepted
    cdef int rc
    if s.status != FNI_RUNNING:
        return FNI_ERR_ARG
    s.dense_valid = False
    if s.n == 0 or s.t == s.t_bound:
        s.t_old = t
        s.has_t_old = True
        if s.n > 0:
            memcpy(s.y_old, s.y, s.n * sizeof(double))
        s.t = s.t_bound
        s.status = FNI_FINISHED
        return FNI_OK
    rc = _step_impl(s, &accepted)
    if rc != FNI_OK:
        return rc
    if not accepted:
        s.status = FNI_FAILED
    else:
        s.t_old = t
        s.has_t_old = True
        if s.direction * (s.t - s.t_bound) >= 0:
            s.status = FNI_FINISHED
    return FNI_OK


cdef int _build_dense(fni_solver* s) noexcept nogil:
    cdef const fni_tableau* tab = s.tab
    cdef Py_ssize_t k, j, i, st, r, n = s.n, nk = tab.n_stages + 1
    cdef Py_ssize_t n_ext = tab.n_k_rows, ncoef = tab.n_dense_coef
    cdef double p, a, d, dy, h, t_old
    cdef double* row
    cdef double* Kj
    cdef double* acc
    cdef double* f_old
    cdef int rc

    if tab.dense_kind == FNI_DENSE_RK:
        # RK23/RK45: Q = K^T P, stored transposed as (n_coef, n).
        for k in range(ncoef):
            row = s.dense + k * n
            for i in range(n):
                row[i] = 0.0
            for j in range(nk):
                p = tab.P[j * ncoef + k]
                if p != 0.0:
                    Kj = s.K + j * n
                    for i in range(n):
                        row[i] += p * Kj[i]
        return FNI_OK

    # DOP853: extra stages 13-15, then F (7, n) exactly as scipy's _dense_output_impl.
    h = s.h_previous
    t_old = s.t_old
    acc = s.y_stage
    f_old = s.K
    for st in range(tab.n_stages + 1, n_ext):
        for i in range(n):
            acc[i] = 0.0
        for j in range(st):
            a = tab.A[st * tab.a_stride + j]
            if a != 0.0:
                Kj = s.K + j * n
                for i in range(n):
                    acc[i] += a * Kj[i]
        for i in range(n):
            acc[i] = s.y_old[i] + acc[i] * h
        rc = _fun(s, t_old + tab.C[st] * h, acc, s.K + st * n)
        if rc != FNI_OK:
            return rc

    for i in range(n):
        dy = s.y[i] - s.y_old[i]
        s.dense[i] = dy
        s.dense[n + i] = h * f_old[i] - dy
        s.dense[2 * n + i] = 2 * dy - h * (s.f[i] + f_old[i])

    for r in range(ncoef - 3):
        row = s.dense + (3 + r) * n
        for i in range(n):
            row[i] = 0.0
        for j in range(n_ext):
            d = tab.D[r * n_ext + j]
            if d != 0.0:
                Kj = s.K + j * n
                for i in range(n):
                    row[i] += d * Kj[i]
        for i in range(n):
            row[i] *= h
    return FNI_OK


cdef int fni_solver_ensure_dense(fni_solver* s) noexcept nogil:
    """Build the last step's interpolant once (DOP853 costs 3 RHS calls)."""
    cdef int rc
    if not s.dense_valid:
        if s.n > 0 and s.t != s.t_old:
            rc = _build_dense(s)
            if rc != FNI_OK:
                return rc
        s.dense_valid = True
    return FNI_OK


cdef void fni_solver_dense_eval(const fni_solver* s, double t, double* out) noexcept nogil:
    """Interpolate over the last step. Requires :func:`fni_solver_ensure_dense`."""
    _interp_eval(s.tab.dense_kind, s.tab.n_dense_coef, s.n, s.t_old, s.t - s.t_old,
                 s.y_old, s.dense, t, out)


# --- events --------------------------------------------------------------------------

cdef int fni_events_eval(const fni_event* events, Py_ssize_t n_events, double t,
                         const double* y, Py_ssize_t n, double* g) noexcept nogil:
    cdef Py_ssize_t e
    for e in range(n_events):
        if events[e].fn(t, y, n, events[e].ctx, &g[e]) != 0:
            return FNI_ERR_CALLBACK
    return FNI_OK


cdef inline int _event_on_dense(const fni_solver* s, const fni_event* ev, double t,
                                double* buf, double* g) noexcept nogil:
    fni_solver_dense_eval(s, t, buf)
    if ev.fn(t, buf, s.n, ev.ctx, g) != 0:
        return FNI_ERR_CALLBACK
    return FNI_OK


cdef int _brentq_event(const fni_solver* s, const fni_event* ev, double xa, double xb,
                       double* buf, double* root) noexcept nogil:
    """Root of ``ev(t, sol(t))`` on [xa, xb]: scipy's C ``brentq`` with
    ``xtol = rtol = 4 * EPS``, ``maxiter = 100``."""
    cdef double xtol = 4 * EPS, rtol = 4 * EPS
    cdef double xpre = xa, xcur = xb, xblk = 0.0
    cdef double fpre, fcur, fblk = 0.0, spre = 0.0, scur = 0.0, sbis
    cdef double delta, stry, dpre, dblk
    cdef int i, rc

    rc = _event_on_dense(s, ev, xpre, buf, &fpre)
    if rc != FNI_OK:
        return rc
    rc = _event_on_dense(s, ev, xcur, buf, &fcur)
    if rc != FNI_OK:
        return rc
    if fpre * fcur > 0:
        return FNI_ERR_ROOT_SIGN
    if fpre == 0:
        root[0] = xpre
        return FNI_OK
    if fcur == 0:
        root[0] = xcur
        return FNI_OK

    for i in range(BRENTQ_MAXITER):
        if fpre != 0 and fcur != 0 and ((fpre < 0) != (fcur < 0)):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        if fabs(fblk) < fabs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * fabs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or fabs(sbis) < delta:
            root[0] = xcur
            return FNI_OK

        if fabs(spre) > delta and fabs(fcur) < fabs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * fabs(stry) < min(fabs(spre), 3 * fabs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if fabs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta
        rc = _event_on_dense(s, ev, xcur, buf, &fcur)
        if rc != FNI_OK:
            return rc

    root[0] = xcur
    return FNI_ERR_ROOT_CONV


# --- event scan ----------------------------------------------------------------------
#
# Events are located by a scan over each accepted step, in the direction of
# integration. The step is cut into pieces at test points: its end (and with
# FNI_EVENTS_INTERIOR, n_interior evenly spaced points inside it, evaluated on
# the interpolant), and at every root located in it. An event is found on a
# piece when its sign differs between the piece's ends (scipy's test, so a
# value of exactly zero counts as either sign). The scan reports roots one at
# a time, earliest first. Before reporting a root it checks every other event
# at that time, so a crossing hidden between two test points but before the
# root (one that a later pair of crossings would cancel at the test points) is
# found and reported first. A terminal event ends the scan, as it ends the step.
#
# With FNI_EVENTS_ENDPOINTS and no such hidden crossings, the roots are the
# ones scipy finds: the same brackets and the same Brent iterations.

cdef int REFINE_MAXITER = 64


cdef inline bint _crosses(double ga, double gb, double direction) noexcept nogil:
    cdef bint up = ga <= 0 and gb >= 0
    cdef bint down = ga >= 0 and gb <= 0
    return (up and direction > 0) or (down and direction < 0) or ((up or down)
                                                                   and direction == 0)


cdef inline bint _before(double x, double y, double direction) noexcept nogil:
    """``x`` comes strictly before ``y`` in the direction of integration."""
    return (y - x) * direction > 0


cdef void fni_options_init(fni_options* opt) noexcept nogil:
    """Fill ``opt`` with the defaults of :func:`solve_ivp` (rtol 1e-3, atol 1e-6,
    automatic first step, no step limit, endpoint event detection). Call it
    before setting fields, so fields added in later versions get defaults."""
    opt.rtol = 1e-3
    opt.rtol_vec = NULL
    opt.atol = 1e-6
    opt.atol_vec = NULL
    opt.first_step = -1.0
    opt.max_step = INFINITY
    opt.event_detection = FNI_EVENTS_ENDPOINTS
    opt.n_interior = 4


cdef int fni_event_scan_init(fni_event_scan* sc, Py_ssize_t n_events,
                             Py_ssize_t n) noexcept nogil:
    """Allocate the scan scratch; call :func:`fni_event_scan_free` afterwards."""
    cdef Py_ssize_t m = n_events if n_events > 0 else 1
    sc.n_events = n_events
    sc.mem = <double*> malloc((7 * m + (n if n > 0 else 1)) * sizeof(double))
    if sc.mem == NULL:
        return FNI_ERR_ALLOC
    sc.ga = sc.mem
    sc.ta = sc.ga + m
    sc.gb = sc.ta + m
    sc.gr = sc.gb + m
    sc.root = sc.gr + m
    sc.rt = sc.root + m
    sc.rv = sc.rt + m
    sc.buf = sc.rv + m
    sc.g_end = NULL
    sc.i_point = 0
    sc.n_points = 0
    return FNI_OK


cdef void fni_event_scan_free(fni_event_scan* sc) noexcept nogil:
    if sc.mem != NULL:
        free(sc.mem)
    sc.mem = NULL


cdef int _scan_bracket(fni_solver* s, const fni_event* events, Py_ssize_t e,
                       fni_event_scan* sc) noexcept nogil:
    """Locate event ``e`` on [ta, b] if its sign changes there."""
    cdef int rc
    if not (_before(sc.ta[e], sc.b, s.direction)
            and _crosses(sc.ga[e], sc.gb[e], events[e].direction)):
        return FNI_OK
    rc = fni_solver_ensure_dense(s)
    if rc != FNI_OK:
        return rc
    rc = _brentq_event(s, &events[e], sc.ta[e], sc.b, sc.buf, &sc.root[e])
    if rc != FNI_OK:
        return rc
    sc.rt[e] = sc.b
    sc.rv[e] = sc.gb[e]
    return FNI_OK


cdef int _scan_step_off_zero(fni_solver* s, const fni_event* ev, Py_ssize_t e,
                             fni_event_scan* sc) noexcept nogil:
    """Move event ``e``'s search start from a zero at ta (inside the step) to
    just after it: a thousand ulps of the time scale, as the post-reset probe.
    If it is still zero there, the event is not searched again in this step."""
    cdef double delta = 1000 * EPS * (fabs(sc.ta[e]) + fabs(s.t - s.t_old)) * s.direction
    cdef double tp = sc.ta[e] + delta
    if _before(tp, s.t, s.direction):
        fni_solver_dense_eval(s, tp, sc.buf)
        if ev.fn(tp, sc.buf, s.n, ev.ctx, &sc.ga[e]) != 0:
            return FNI_ERR_CALLBACK
        sc.ta[e] = tp
        if sc.ga[e] != 0:
            return FNI_OK
    sc.ga[e] = sc.g_end[e]
    sc.ta[e] = s.t
    return FNI_OK


cdef int _scan_load(fni_solver* s, const fni_event* events, Py_ssize_t n_events,
                    const fni_hybrid_state* h, fni_event_scan* sc) noexcept nogil:
    """Move to the next test point and locate the events that change sign
    between the scan position and it."""
    cdef Py_ssize_t e
    cdef int rc
    sc.i_point += 1
    if sc.i_point == sc.n_points:
        sc.b = s.t
        memcpy(sc.gb, sc.g_end, n_events * sizeof(double))
    else:
        rc = fni_solver_ensure_dense(s)
        if rc != FNI_OK:
            return rc
        sc.b = s.t_old + (s.t - s.t_old) * (<double> sc.i_point / <double> sc.n_points)
        fni_solver_dense_eval(s, sc.b, sc.buf)
        rc = fni_events_eval(events, n_events, sc.b, sc.buf, s.n, sc.gb)
        if rc != FNI_OK:
            return rc
    for e in range(n_events):
        if (h != NULL and h.masked[e]) or sc.root[e] == sc.root[e]:
            continue
        rc = _scan_bracket(s, events, e, sc)
        if rc != FNI_OK:
            return rc
    return FNI_OK


cdef int fni_events_begin(fni_solver* s, const fni_event* events, Py_ssize_t n_events,
                          const double* g, double* g_new, const fni_hybrid_state* h,
                          fni_event_scan* sc) noexcept nogil:
    """Start the event scan of the step just taken; report its events with
    :func:`fni_events_next`.

    ``g`` holds the event values at the start of the step and ``g_new``
    receives the values at its end (the caller then copies them into ``g``).
    ``h`` is a hybrid integration's guard state (may be NULL): masked events
    are ignored, and the search for an event probed after its reset starts at
    the probe time. ``sc`` comes from :func:`fni_event_scan_init`.
    """
    cdef Py_ssize_t e
    cdef double tp, t_old = s.t_old, t = s.t
    cdef int rc
    rc = fni_events_eval(events, n_events, t, s.y, s.n, g_new)
    if rc != FNI_OK:
        return rc
    sc.g_end = g_new
    sc.a = t_old
    for e in range(n_events):
        sc.ga[e] = g[e]
        sc.ta[e] = t_old
        sc.root[e] = NAN
        if h != NULL:
            tp = h.t_probe[e]
            if (tp > t_old and tp < t) or (tp < t_old and tp > t):
                sc.ta[e] = tp
    sc.n_points = (s.n_interior + 1 if s.event_detection == FNI_EVENTS_INTERIOR
                   and n_events > 0 else 1)
    sc.i_point = 0
    return _scan_load(s, events, n_events, h, sc)


cdef int fni_events_next(fni_solver* s, const fni_event* events, Py_ssize_t n_events,
                         double* counts, const fni_hybrid_state* h, fni_event_scan* sc,
                         Py_ssize_t* event, double* root, bint* terminal) noexcept nogil:
    """Report the next event of the step: its index in ``event`` (-1 when the
    step has no more) and its time in ``root``.

    ``counts`` holds the occurrences per event and is updated here.
    ``terminal`` is set when this occurrence reaches the event's
    ``max_events``; the step then ends at ``root`` and the scan must not be
    continued. Events at the same time are reported in index order.
    """
    cdef Py_ssize_t e, best, it = 0
    cdef double r, c
    cdef double dirn = s.direction
    cdef bint improved
    cdef int rc

    event[0] = -1
    terminal[0] = False
    while True:
        best = -1
        for e in range(n_events):
            if sc.root[e] == sc.root[e] and (best < 0
                                             or _before(sc.root[e], sc.root[best], dirn)):
                best = e
        if best < 0:
            if sc.i_point >= sc.n_points:
                return FNI_OK
            # Nothing left before b: move the scan position there.
            for e in range(n_events):
                if not _before(sc.b, sc.ta[e], dirn):
                    sc.ga[e] = sc.gb[e]
                    sc.ta[e] = sc.b
            sc.a = sc.b
            rc = _scan_load(s, events, n_events, h, sc)
            if rc != FNI_OK:
                return rc
            continue

        r = sc.root[best]
        if not _before(sc.a, r, dirn):
            memcpy(sc.gr, sc.ga, n_events * sizeof(double))
            break
        # Any other event whose sign differs between its search start and r
        # crosses before r (or at it): locate it and take the earliest.
        fni_solver_dense_eval(s, r, sc.buf)
        rc = fni_events_eval(events, n_events, r, sc.buf, s.n, sc.gr)
        if rc != FNI_OK:
            return rc
        improved = False
        for e in range(n_events):
            if e == best or (h != NULL and h.masked[e]):
                continue
            if not (_before(sc.ta[e], r, dirn) and _crosses(sc.ga[e], sc.gr[e],
                                                             events[e].direction)):
                continue
            if sc.root[e] == sc.root[e] and not _before(r, sc.root[e], dirn):
                continue      # already located at or before r
            rc = _brentq_event(s, &events[e], sc.ta[e], r, sc.buf, &c)
            if rc != FNI_OK:
                return rc
            sc.root[e] = c
            sc.rt[e] = r
            sc.rv[e] = sc.gr[e]
            improved = True
        it += 1
        if improved and it < REFINE_MAXITER:
            continue          # an earlier root was found: check before it
        break

    # Report ``best`` at r. Every other event's search now starts at r (unless
    # it starts later, at a probe), and ``best`` resumes from the right end of
    # its bracket, where its sign is known: a further crossing beyond it in
    # this piece is located now.
    for e in range(n_events):
        if e != best and not _before(r, sc.ta[e], dirn):
            sc.ga[e] = sc.gr[e]
            sc.ta[e] = r
    sc.ga[best] = sc.rv[best]
    sc.ta[best] = sc.rt[best]
    sc.root[best] = NAN
    sc.a = r
    if sc.rv[best] == 0 and sc.rt[best] != s.t:
        # The root sits exactly on its bracket's right end, inside the step
        # (a test point, or another event's root): the zero there must not
        # count as a second crossing, so search on from just after it.
        rc = _scan_step_off_zero(s, &events[best], best, sc)
        if rc != FNI_OK:
            return rc
    # Look again for crossings on [ta, b]: ``best`` from the end of its
    # bracket, and events whose sign changed at r in the direction they ignore.
    for e in range(n_events):
        if (h != NULL and h.masked[e]) or sc.root[e] == sc.root[e]:
            continue
        rc = _scan_bracket(s, events, e, sc)
        if rc != FNI_OK:
            return rc

    counts[best] += 1
    event[0] = best
    root[0] = r
    terminal[0] = counts[best] >= events[best].max_events
    return FNI_OK


# --- hybrid integration --------------------------------------------------------------

cdef int fni_hybrid_init(fni_hybrid_state* h, Py_ssize_t n_events) noexcept nogil:
    """Set up guard bookkeeping; call :func:`fni_hybrid_free` afterwards."""
    cdef Py_ssize_t e
    h.n_resets = 0
    h.last_reset_t = NAN
    h.masked = <bint*> malloc((n_events if n_events > 0 else 1) * sizeof(bint))
    h.t_probe = <double*> malloc((n_events if n_events > 0 else 1) * sizeof(double))
    if h.masked == NULL or h.t_probe == NULL:
        return FNI_ERR_ALLOC
    for e in range(n_events):
        h.masked[e] = False
        h.t_probe[e] = NAN
    return FNI_OK


cdef void fni_hybrid_free(fni_hybrid_state* h) noexcept nogil:
    if h.masked != NULL:
        free(h.masked)
    if h.t_probe != NULL:
        free(h.t_probe)
    h.masked = NULL
    h.t_probe = NULL


cdef void fni_hybrid_unmask(fni_hybrid_state* h, Py_ssize_t n_events,
                            const double* g_new) noexcept nogil:
    """After an accepted step without a reset: re-arm each masked guard whose
    value at the end of the step is nonzero, and forget the probe times."""
    cdef Py_ssize_t e
    for e in range(n_events):
        if h.masked[e] and g_new[e] != 0:
            h.masked[e] = False
        h.t_probe[e] = NAN


cdef int fni_hybrid_reset(fni_solver* s, fni_hybrid_state* h, const fni_hybrid_options* hopt,
                          const fni_event* events, Py_ssize_t n_events, double* counts,
                          Py_ssize_t event, double t_event, double* y,
                          fni_reset_fn reset, void* reset_ctx, double* g,
                          int* status) noexcept nogil:
    """Handle terminal event ``event`` located at ``t_event`` with state ``y``.

    Checks the reset limits, applies the reset map to ``y`` in place, and on
    FNI_RESET_CONTINUE restarts the solver from (t_event, y) and refreshes
    the event values ``g``. ``status`` is set to FNI_STATUS_EVENT,
    FNI_STATUS_RESET_LIMIT or FNI_STATUS_ZENO when the integration ends here
    (always FNI_STATUS_EVENT if ``reset`` is NULL), and to 2 when it continues.

    Right after the reset the guard that fired sits on zero, where round-off
    decides its sign. So its value is probed a little later, at
    ``t_event + delta`` along the new flow (``y + delta * f``, with ``delta``
    a thousand ulps of the time scale), and taken as its value at the start
    of the next step. The next root search for it starts at the probe time.
    If the probe is still exactly zero (a tangential flow, or a guard that
    stays at zero), the guard is masked until a step ends with it away from
    zero. Its occurrence count restarts, so it fires on every later crossing.
    """
    cdef int rc
    cdef Py_ssize_t i
    cdef double delta, gp
    status[0] = 2          # still running
    if reset == NULL:
        status[0] = FNI_STATUS_EVENT
        return FNI_OK
    if hopt.max_resets >= 0 and h.n_resets >= hopt.max_resets:
        status[0] = FNI_STATUS_RESET_LIMIT
        return FNI_OK
    if (hopt.min_dwell > 0 and h.n_resets > 0
            and fabs(t_event - h.last_reset_t) < hopt.min_dwell):
        status[0] = FNI_STATUS_ZENO
        return FNI_OK
    rc = reset(event, t_event, y, s.n, reset_ctx)
    if rc < 0:
        return FNI_ERR_CALLBACK
    if rc == FNI_RESET_STOP:
        status[0] = FNI_STATUS_EVENT
        return FNI_OK
    if rc != FNI_RESET_CONTINUE:
        return FNI_ERR_CALLBACK
    h.n_resets += 1
    h.last_reset_t = t_event
    counts[event] = 0.0
    rc = fni_solver_restart(s, t_event, y, hopt.reselect_step)
    if rc != FNI_OK:
        return rc
    rc = fni_events_eval(events, n_events, t_event, s.y, s.n, g)
    if rc != FNI_OK:
        return rc
    for i in range(n_events):
        h.t_probe[i] = NAN
    delta = 1000 * EPS * (fabs(t_event) + (s.h_abs if s.h_abs < INFINITY else 0.0))
    if delta == 0:
        delta = 1000 * EPS
    delta *= s.direction
    for i in range(s.n):
        s.y_new[i] = s.y[i] + delta * s.f[i]      # y_new is scratch until the next step
    if events[event].fn(t_event + delta, s.y_new, s.n, events[event].ctx, &gp) != 0:
        return FNI_ERR_CALLBACK
    if gp != 0:
        g[event] = gp
        h.t_probe[event] = t_event + delta
        h.masked[event] = False
    else:
        h.masked[event] = True
    return FNI_OK


cdef int fni_integrate_hybrid(int method, fni_rhs_fn fun, void* ctx, Py_ssize_t n,
                              double t0, const double* y0, double t1, const fni_options* opt,
                              const fni_event* events, Py_ssize_t n_events,
                              fni_reset_fn reset, void* reset_ctx,
                              const fni_hybrid_options* hopt,
                              double* y_out, Py_ssize_t* record_event, double* record_t,
                              double* record_y, double* record_y_after, bint* record_reset,
                              Py_ssize_t max_records,
                              fni_stats* stats, Py_ssize_t* n_resets) noexcept nogil:
    """Integrate a hybrid system from (t0, y0) to t1 without the GIL.

    As :func:`fni_integrate`, except that a terminal event is a *guard*: when
    it fires, ``reset`` maps the state in place and integration continues from
    the event (see ``fni_reset_fn``), in one call and with the step size
    carried across. Non-terminal events are only recorded. ``hopt`` bounds the
    number of resets (FNI_STATUS_RESET_LIMIT) and their spacing
    (FNI_STATUS_ZENO). ``reset`` may be NULL, which makes every terminal event
    stop the integration as in :func:`fni_integrate`.

    Every located event is recorded, up to ``max_records``: ``record_y`` holds
    the state just before it and ``record_y_after`` (if not NULL) the state
    just after, which differs only for resets; ``record_reset`` (if not NULL)
    flags the records where the reset map was applied. ``n_resets`` (if not NULL)
    receives the number of resets applied. The final state at an ended
    integration is the state after the last reset map call.
    """
    cdef fni_solver s
    cdef fni_hybrid_state h
    cdef fni_event_scan scan
    cdef Py_ssize_t e, e_found = -1, last = -1
    cdef bint terminate
    cdef double* work = NULL
    cdef double* g
    cdef double* g_new
    cdef double* counts
    cdef double* y_event
    cdef double* tmp
    cdef int rc, hstatus
    cdef double t_event = NAN

    stats.status = FNI_STATUS_FAILED
    stats.t_final = t0
    stats.nfev = 0
    stats.n_accepted = 0
    stats.n_rejected = 0
    stats.n_records = 0
    if n_resets != NULL:
        n_resets[0] = 0
    h.masked = NULL
    h.t_probe = NULL
    scan.mem = NULL

    rc = fni_solver_init(&s, method, fun, ctx, n, t0, y0, t1, opt)
    if rc == FNI_OK:
        rc = fni_hybrid_init(&h, n_events)
    if rc == FNI_OK:
        rc = fni_event_scan_init(&scan, n_events, n)
    if rc != FNI_OK:
        fni_solver_free(&s)
        fni_hybrid_free(&h)
        fni_event_scan_free(&scan)
        return rc

    work = <double*> malloc((3 * n_events + (n if n > 0 else 1) + 1) * sizeof(double))
    if work == NULL:
        rc = FNI_ERR_ALLOC
    else:
        g = work
        g_new = g + n_events
        counts = g_new + n_events
        y_event = counts + n_events
        for e in range(n_events):
            counts[e] = 0.0
        rc = fni_events_eval(events, n_events, t0, s.y, n, g)

    while rc == FNI_OK:
        rc = fni_solver_step(&s)
        if rc != FNI_OK:
            break
        if s.status == FNI_FAILED:
            stats.status = FNI_STATUS_FAILED
            stats.t_final = s.t
            memcpy(y_out, s.y, n * sizeof(double))
            break

        terminate = False
        if n_events > 0:
            rc = fni_events_begin(&s, events, n_events, g, g_new, &h, &scan)
            while rc == FNI_OK:
                rc = fni_events_next(&s, events, n_events, counts, &h, &scan, &e_found,
                                     &t_event, &terminate)
                if rc != FNI_OK or e_found < 0:
                    break
                if stats.n_records < max_records:
                    record_event[stats.n_records] = e_found
                    record_t[stats.n_records] = t_event
                    if record_y != NULL:
                        fni_solver_dense_eval(&s, t_event, record_y + stats.n_records * n)
                    if record_y_after != NULL:
                        fni_solver_dense_eval(&s, t_event,
                                              record_y_after + stats.n_records * n)
                    if record_reset != NULL:
                        record_reset[stats.n_records] = False
                    last = stats.n_records
                stats.n_records += 1
                if terminate:
                    break
            if rc != FNI_OK:
                break

        if terminate:
            fni_solver_dense_eval(&s, t_event, y_event)
            rc = fni_hybrid_reset(&s, &h, hopt, events, n_events, counts, e_found,
                                  t_event, y_event, reset, reset_ctx, g, &hstatus)
            if rc != FNI_OK:
                break
            if last == stats.n_records - 1 and reset != NULL and (
                    hstatus == 2 or hstatus == FNI_STATUS_EVENT):
                if record_y_after != NULL:
                    memcpy(record_y_after + last * n, y_event, n * sizeof(double))
                if record_reset != NULL:
                    record_reset[last] = True
            if hstatus != 2:
                stats.status = hstatus
                stats.t_final = t_event
                memcpy(y_out, y_event, n * sizeof(double))
                break
            continue

        if n_events > 0:
            fni_hybrid_unmask(&h, n_events, g_new)
            tmp = g
            g = g_new
            g_new = tmp
        if s.status == FNI_FINISHED:
            stats.status = FNI_STATUS_FINISHED
            stats.t_final = s.t
            memcpy(y_out, s.y, n * sizeof(double))
            break

    stats.nfev = s.nfev
    stats.n_accepted = s.n_accepted
    stats.n_rejected = s.n_rejected
    if n_resets != NULL:
        n_resets[0] = h.n_resets
    fni_solver_free(&s)
    fni_hybrid_free(&h)
    fni_event_scan_free(&scan)
    if work != NULL:
        free(work)
    return rc


cdef int fni_integrate(int method, fni_rhs_fn fun, void* ctx, Py_ssize_t n,
                       double t0, const double* y0, double t1, const fni_options* opt,
                       const fni_event* events, Py_ssize_t n_events,
                       double* y_out, Py_ssize_t* record_event, double* record_t,
                       double* record_y, Py_ssize_t max_records,
                       fni_stats* stats) noexcept nogil:
    """Integrate from (t0, y0) to t1 (or a terminal event) without the GIL.

    Writes the final state to ``y_out`` (the state at the terminal event, if one
    fired) and fills ``stats``. Located events are recorded into
    ``record_event`` / ``record_t`` (and ``record_y``, ``max_records x n``
    row-major, if not NULL) up to ``max_records``; ``stats.n_records`` counts
    all of them. Same steps, events and results as :func:`solve_ivp`.
    """
    cdef fni_hybrid_options hopt
    hopt.max_resets = -1
    hopt.min_dwell = 0.0
    hopt.reselect_step = False
    return fni_integrate_hybrid(method, fun, ctx, n, t0, y0, t1, opt, events, n_events,
                                NULL, NULL, &hopt, y_out, record_event, record_t, record_y,
                                NULL, NULL, max_records, stats, NULL)


# =============================================================================
# Python <-> core glue
# =============================================================================

cdef int _check(int rc) except -1:
    """Turn a core return code into a Python exception (or propagate one)."""
    if rc == FNI_OK:
        return 0
    if PyErr_Occurred():
        return -1          # an eval()/callable raised: propagate that exception
    if rc == FNI_ERR_CALLBACK:
        raise RuntimeError("a native right-hand side or event function returned an error code")
    if rc == FNI_ERR_ALLOC:
        raise MemoryError()
    if rc == FNI_ERR_ROOT_SIGN:
        raise ValueError("f(a) and f(b) must have different signs")
    if rc == FNI_ERR_ROOT_CONV:
        raise RuntimeError(f"Failed to converge after {BRENTQ_MAXITER} iterations.")
    raise ValueError("invalid argument passed to the integration core")


# Trampolines: let the GIL-free core call eval() overrides / Python callables.
# They need the GIL, so they are only used when the caller holds it (the Python
# path never releases it while one is installed). An exception raised inside
# eval() stays set, the core unwinds with FNI_ERR_CALLBACK, and _check
# re-raises it.

cdef int _system_trampoline(double t, const double* y, double* dydt, Py_ssize_t n,
                            void* ctx) except -1:
    return (<OdeSystem> ctx).eval(t, y, dydt, n)


cdef int _event_trampoline(double t, const double* y, Py_ssize_t n, void* ctx,
                           double* g) except -1:
    g[0] = (<EventFunction> ctx).eval(t, y, n)
    return 0


cdef inline fni_rhs_fn _rhs_of(OdeSystem system, void** ctx) noexcept:
    if system.c_fn != NULL:
        ctx[0] = system.c_ctx
        return system.c_fn
    ctx[0] = <void*> system
    return <fni_rhs_fn> _system_trampoline


cdef inline void _event_of(EventFunction ev, double max_events, fni_event* out) noexcept:
    if ev.c_fn != NULL:
        out.fn = ev.c_fn
        out.ctx = ev.c_ctx
    else:
        out.fn = <fni_event_fn> _event_trampoline
        out.ctx = <void*> ev
    out.direction = ev.direction
    out.max_events = max_events


cdef int _reset_trampoline(Py_ssize_t event, double t, double* y, Py_ssize_t n,
                           void* ctx) except -1:
    return (<ResetMap> ctx).eval(event, t, y, n)


cdef inline fni_reset_fn _reset_of(ResetMap reset, void** ctx) noexcept:
    if reset.c_fn != NULL:
        ctx[0] = reset.c_ctx
        return reset.c_fn
    ctx[0] = <void*> reset
    return <fni_reset_fn> _reset_trampoline


cdef inline Py_ssize_t _searchsorted(const double[::1] a, double v, bint left) noexcept:
    """``np.searchsorted(a, v, side='left'|'right')`` for ascending ``a``."""
    cdef Py_ssize_t lo = 0, hi = a.shape[0], mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if (a[mid] < v) if left else (a[mid] <= v):
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef _init_param_buffer(obj_name, values, names):
    arr = np.array([] if values is None else values, dtype=np.float64).reshape(-1)
    k = arr.size
    if names is None:
        names = tuple(f"p{i}" for i in range(k))
    else:
        names = tuple(str(nm) for nm in names)
        if len(names) != k:
            raise ValueError(f"{obj_name}: {len(names)} parameter names for {k} parameters")
    buf = np.zeros(max(k, 1))
    buf[:k] = arr
    return buf, k, names


cdef _set_params(double[::1] buf, Py_ssize_t k, value, obj_name):
    cdef Py_ssize_t i
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != k:
        raise ValueError(f"{obj_name} has {k} parameters, got {arr.size} values")
    for i in range(k):
        buf[i] = arr[i]      # in place: c_ctx pointers into the buffer stay valid


# =============================================================================
# Right-hand sides
# =============================================================================

cdef class OdeSystem:
    """Base class for native right-hand sides ``dy/dt = f(t, y; params)``.

    Subclass in Cython. Either:

    - set ``c_fn`` (a ``nogil`` C function, see ``fni_rhs_fn``) and ``c_ctx``
      in ``__init__``. The engine then calls it directly, and the system also
      works with the GIL-free C API; or
    - override the ``cdef`` method ``eval`` (works everywhere except GIL-free
      integration).

    Runtime parameters: call ``self._init_params(values, names)`` in
    ``__init__`` and read ``self._params[i]``, or point ``c_ctx`` at
    ``self._params_ptr()``. Users change them between solves with
    ``system.params = [...]`` without recompiling. Instances are also callable
    from Python as ``system(t, y) -> ndarray``, so the same object works with
    :func:`scipy.integrate.solve_ivp`.
    """

    def __cinit__(self, *args, **kwargs):
        self.n_states = -1
        self.n_params = 0
        self.param_names = ()
        self._params = np.zeros(1)
        self.c_fn = NULL
        self.c_ctx = NULL

    cdef int _init_params(self, values, names=None) except -1:
        buf, k, nm = _init_param_buffer(type(self).__name__, values, names)
        self._params = buf
        self.n_params = k
        self.param_names = nm
        return 0

    cdef double* _params_ptr(self) noexcept:
        return &self._params[0]

    @property
    def params(self):
        """Runtime parameters (copy). Assign to change them in place."""
        return np.asarray(self._params)[:self.n_params].copy()

    @params.setter
    def params(self, value):
        _set_params(self._params, self.n_params, value, type(self).__name__)

    @property
    def has_native_fn(self):
        """True if a GIL-free C function is available (usable by :func:`integrate` without the GIL)."""
        return self.c_fn != NULL

    cdef int check_size(self, Py_ssize_t n) except -1:
        if self.n_states >= 0 and n != self.n_states:
            raise ValueError(f"system expects {self.n_states} states, y0 has {n}")
        return 0

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        if self.c_fn == NULL:
            raise NotImplementedError(f"{type(self).__name__} must implement eval() or set c_fn")
        if self.c_fn(t, y, dydt, n, self.c_ctx) != 0:
            raise RuntimeError(f"{type(self).__name__}: native RHS returned an error code")
        return 0

    def __call__(self, double t, y):
        cdef const double[::1] y_view = np.ascontiguousarray(y, dtype=np.float64)
        cdef Py_ssize_t n = y_view.shape[0]
        if self.n_states >= 0 and n != self.n_states:
            raise ValueError(f"expected state of size {self.n_states}, got {n}")
        out = _empty1d(n)
        cdef double[::1] out_view = out
        if n > 0:
            self.eval(t, &y_view[0], &out_view[0], n)
        return out


cdef class PythonOdeSystem(OdeSystem):
    """Adapter that wraps a Python callable ``fun(t, y, *args) -> array_like``.

    This path calls back into Python on every RHS evaluation. It is useful for
    correctness checks and for measuring integrator overhead separately from
    RHS cost. For speed, write a native :class:`OdeSystem` instead.

    With ``vectorized=True`` the callable receives ``y`` of shape (n, 1) and may
    return shape (n, 1), as with scipy's ``vectorized`` option.
    """

    cdef object fun
    cdef tuple args
    cdef bint vectorized

    def __init__(self, fun, args=(), vectorized=False):
        if not callable(fun):
            raise TypeError("fun must be callable")
        self.fun = fun
        self.args = tuple(args)
        self.vectorized = vectorized

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        # Give the callable a fresh array each call (as scipy does) so it may
        # safely keep a reference to its input.
        y_arr = _array_from(y, n)
        if self.vectorized:
            result = self.fun(t, y_arr[:, None], *self.args)
        else:
            result = self.fun(t, y_arr, *self.args)
        cdef const double[::1] r_view = np.ascontiguousarray(result, dtype=np.float64).reshape(-1)
        if r_view.shape[0] != n:
            raise ValueError(
                f"fun returned {r_view.shape[0]} values, expected {n}")
        if n > 0:
            memcpy(dydt, &r_view[0], n * sizeof(double))
        return 0



# =============================================================================
# Event functions
# =============================================================================

cdef class EventFunction:
    """Base class for native event functions ``g(t, y; params)``.

    Subclass in Cython and set ``c_fn``/``c_ctx`` (GIL-free) or override
    ``eval``. Runtime ``params`` work as for :class:`OdeSystem`.
    ``terminal`` and ``direction`` behave like the attributes of scipy event
    callables.
    """

    def __cinit__(self, *args, **kwargs):
        self.n_params = 0
        self.param_names = ()
        self._params = np.zeros(1)
        self.c_fn = NULL
        self.c_ctx = NULL

    def __init__(self, terminal=False, double direction=0.0):
        self.terminal = terminal
        self.direction = direction

    cdef int _init_params(self, values, names=None) except -1:
        buf, k, nm = _init_param_buffer(type(self).__name__, values, names)
        self._params = buf
        self.n_params = k
        self.param_names = nm
        return 0

    cdef double* _params_ptr(self) noexcept:
        return &self._params[0]

    @property
    def params(self):
        return np.asarray(self._params)[:self.n_params].copy()

    @params.setter
    def params(self, value):
        _set_params(self._params, self.n_params, value, type(self).__name__)

    @property
    def has_native_fn(self):
        return self.c_fn != NULL

    cdef int check_size(self, Py_ssize_t n) except -1:
        return 0

    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1:
        cdef double g
        if self.c_fn == NULL:
            raise NotImplementedError(f"{type(self).__name__} must implement eval() or set c_fn")
        if self.c_fn(t, y, n, self.c_ctx, &g) != 0:
            raise RuntimeError(f"{type(self).__name__}: native event function returned an error code")
        return g

    def __call__(self, double t, y):
        cdef const double[::1] y_view = np.ascontiguousarray(y, dtype=np.float64)
        cdef Py_ssize_t n = y_view.shape[0]
        self.check_size(n)
        return self.eval(t, &y_view[0] if n > 0 else NULL, n)


cdef class PythonEvent(EventFunction):
    """Adapter for a scipy-style event callable ``event(t, y, *args) -> float``.

    Reads ``terminal`` and ``direction`` from the callable's attributes.
    """

    cdef object fun
    cdef tuple args

    def __init__(self, fun, args=()):
        if not callable(fun):
            raise TypeError("event must be callable")
        self.fun = fun
        self.args = tuple(args)
        self.terminal = getattr(fun, "terminal", None)
        self.direction = getattr(fun, "direction", 0)

    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1:
        return float(self.fun(t, _array_from(y, n), *self.args))



cdef struct _crossing_ctx:
    Py_ssize_t index
    const double* value     # -> params[0]


cdef int _component_crossing_fn(double t, const double* y, Py_ssize_t n, void* ctx,
                                double* g) noexcept nogil:
    cdef _crossing_ctx* c = <_crossing_ctx*> ctx
    g[0] = y[c.index] - c.value[0]
    return 0


cdef class ComponentCrossing(EventFunction):
    """Native event ``g = y[index] - value``: a state component crosses a level.

    ``value`` is runtime parameter 0 (``ev.params = [new_level]``). Example:
    ``ComponentCrossing(2, 0.0, direction=-1)`` fires when ``z`` passes
    downward through zero.
    """

    cdef _crossing_ctx data

    def __init__(self, Py_ssize_t index, double value=0.0, terminal=False,
                 double direction=0.0):
        super().__init__(terminal, direction)
        if index < 0:
            raise ValueError("index must be non-negative")
        self._init_params([value], ("value",))
        self.data.index = index
        self.data.value = self._params_ptr()
        self.c_fn = _component_crossing_fn
        self.c_ctx = &self.data

    @property
    def index(self):
        return self.data.index

    @property
    def value(self):
        return self._params[0]

    cdef int check_size(self, Py_ssize_t n) except -1:
        if self.data.index >= n:
            raise IndexError(f"event index {self.data.index} out of range for state of size {n}")
        return 0


# =============================================================================
# Reset maps (hybrid integration)
# =============================================================================

cdef class ResetMap:
    """Base class for native reset maps of :func:`solve_hybrid` / :func:`integrate_hybrid`.

    When a terminal event (a *guard*) fires, the reset map receives the event
    index, the time and the state just before it, and overwrites the state
    with the one just after. Subclass in Cython and either set ``c_fn`` (a
    ``nogil`` :c:type:`fni_reset_fn`) and ``c_ctx``, which also works with
    the GIL-free C API, or override the ``cdef`` method ``eval``. A discrete
    mode, such as which arc or contact state is active, belongs in a context
    that the reset map shares with the system's right-hand side.
    """

    def __cinit__(self, *args, **kwargs):
        self.c_fn = NULL
        self.c_ctx = NULL

    @property
    def has_native_fn(self):
        return self.c_fn != NULL

    cdef int check_size(self, Py_ssize_t n) except -1:
        return 0

    cdef int eval(self, Py_ssize_t event, double t, double* y, Py_ssize_t n) except -1:
        cdef int rc
        if self.c_fn == NULL:
            raise NotImplementedError(f"{type(self).__name__} must implement eval() or set c_fn")
        rc = self.c_fn(event, t, y, n, self.c_ctx)
        if rc < 0:
            raise RuntimeError(f"{type(self).__name__}: native reset map returned an error code")
        return rc

    def __call__(self, Py_ssize_t event, double t, y):
        """``reset(event, t, y)``: the post-event state, or None to stop."""
        out = np.array(y, dtype=np.float64, copy=True).reshape(-1)
        cdef double[::1] view = out if out.shape[0] > 0 else np.zeros(1)
        rc = self.eval(event, t, &view[0], out.shape[0])
        return None if rc == FNI_RESET_STOP else out


cdef class PythonReset(ResetMap):
    """Adapter for a callable ``reset(event, t, y, *args)``.

    It returns the post-event state (array-like of length n) to continue, or
    ``None`` to stop the integration at the event.
    """

    cdef object fun
    cdef tuple args

    def __init__(self, fun, args=()):
        if not callable(fun):
            raise TypeError("reset must be callable")
        self.fun = fun
        self.args = tuple(args)

    cdef int eval(self, Py_ssize_t event, double t, double* y, Py_ssize_t n) except -1:
        cdef Py_ssize_t i
        out = self.fun(event, t, _array_from(y, n), *self.args)
        if out is None:
            return FNI_RESET_STOP
        arr = np.asarray(out, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"reset returned {arr.shape[0]} values for a state of size {n}")
        cdef const double[::1] v = np.ascontiguousarray(arr)
        for i in range(n):
            y[i] = v[i]
        return FNI_RESET_CONTINUE


cdef struct _scale_ctx:
    Py_ssize_t index
    const double* factor     # -> params[0]


cdef int _scale_component_fn(Py_ssize_t event, double t, double* y, Py_ssize_t n,
                             void* ctx) noexcept nogil:
    cdef _scale_ctx* c = <_scale_ctx*> ctx
    y[c.index] *= c.factor[0]
    return FNI_RESET_CONTINUE


cdef class ScaleComponent(ResetMap):
    """Native reset ``y[index] *= factor``, for every guard.

    ``factor`` is runtime parameter 0. With ``factor = -e`` on a velocity it
    is an impact with coefficient of restitution ``e`` (a bouncing ball).
    """

    cdef _scale_ctx data
    cdef readonly Py_ssize_t n_params
    cdef double[::1] _params

    def __init__(self, Py_ssize_t index, double factor):
        if index < 0:
            raise ValueError("index must be non-negative")
        self._params = np.array([factor])
        self.n_params = 1
        self.data.index = index
        self.data.factor = &self._params[0]
        self.c_fn = _scale_component_fn
        self.c_ctx = &self.data

    @property
    def index(self):
        return self.data.index

    cdef int check_size(self, Py_ssize_t n) except -1:
        if self.data.index >= n:
            raise IndexError(f"reset index {self.data.index} out of range for state of size {n}")
        return 0

    @property
    def params(self):
        return np.asarray(self._params).copy()

    @params.setter
    def params(self, value):
        _set_params(self._params, 1, value, "ScaleComponent")


# =============================================================================
# Dense output
# =============================================================================

cdef class OdeSolution:
    """Piecewise dense-output interpolant (the ``sol`` of scipy's result).

    ``sol(t)`` returns shape (n,) for scalar ``t`` and (n, m) for a 1-D array.
    Segment selection at breakpoints and extrapolation outside
    ``[t_min, t_max]`` follow :class:`scipy.integrate.OdeSolution`.
    """

    cdef readonly object ts
    cdef readonly double t_min
    cdef readonly double t_max
    cdef readonly Py_ssize_t n_segments
    cdef bint ascending
    cdef bint side_left
    cdef double[::1] ts_sorted
    cdef double[:, ::1] segs   # rows: [t_old, h, y_old (n), coef (ncoef * n)]
    cdef int kind
    cdef Py_ssize_t n
    cdef Py_ssize_t ncoef

    def __init__(self, ts, segments, int kind, Py_ssize_t n, Py_ssize_t ncoef):
        ts = np.ascontiguousarray(ts, dtype=np.float64)
        if ts.ndim != 1 or ts.shape[0] == 0:
            raise ValueError("`ts` must be a non-empty 1-D array.")
        # (No negative indexing: this module compiles with wraparound=False.)
        cdef double t_first = ts[0]
        cdef double t_last = ts[ts.shape[0] - 1]
        d = np.diff(ts)
        if not ((ts.size == 2 and t_first == t_last)
                or np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")
        segments = np.ascontiguousarray(segments, dtype=np.float64)
        self.n_segments = segments.shape[0]
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants don't match.")
        if segments.shape[1] != 2 + n + ncoef * n:
            raise ValueError("segment rows have the wrong width")
        self.ts = ts
        self.segs = segments
        self.kind = kind
        self.n = n
        self.ncoef = ncoef
        if t_last >= t_first:
            self.t_min, self.t_max = t_first, t_last
            self.ascending = True
            self.side_left = True
            self.ts_sorted = ts
        else:
            self.t_min, self.t_max = t_last, t_first
            self.ascending = False
            self.side_left = False
            self.ts_sorted = ts[::-1].copy()

    cdef inline Py_ssize_t _segment(self, double t) noexcept:
        cdef Py_ssize_t seg = _searchsorted(self.ts_sorted, t, self.side_left) - 1
        if seg < 0:
            seg = 0
        if seg > self.n_segments - 1:
            seg = self.n_segments - 1
        if not self.ascending:
            seg = self.n_segments - 1 - seg
        return seg

    cdef inline void _eval(self, double t, double* out) noexcept:
        cdef const double* row = &self.segs[self._segment(t), 0]
        _interp_eval(self.kind, self.ncoef, self.n, row[0], row[1], row + 2,
                     row + 2 + self.n, t, out)

    def __call__(self, t):
        cdef Py_ssize_t j, m
        cdef const double[::1] tv
        cdef double[:, ::1] out_t
        if self.n_segments == 0:
            raise ValueError("OdeSolution has no segments to evaluate.")
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.ndim == 0:
            out = _empty1d(self.n)
            self._eval(float(t_arr), <double*> cnp.PyArray_DATA(out))
            return out
        if t_arr.ndim != 1:
            raise ValueError("`t` must be a float or a 1-D array.")
        tv = np.ascontiguousarray(t_arr)
        m = tv.shape[0]
        out = _empty2d(m, self.n)
        out_t = out
        if self.n > 0:
            for j in range(m):
                self._eval(tv[j], &out_t[j, 0])
        return out.T


cdef class _RowBuffer:
    """Growable (time, row) storage with amortised doubling."""

    cdef Py_ssize_t count
    cdef Py_ssize_t cap
    cdef Py_ssize_t width
    cdef object t_arr
    cdef object r_arr
    cdef double[::1] t
    cdef double[:, ::1] r

    def __cinit__(self, Py_ssize_t width, Py_ssize_t cap=64):
        self.width = width
        self.cap = cap
        self.count = 0
        self.t_arr = _empty1d(cap)
        self.r_arr = _empty2d(cap, width)
        self.t = self.t_arr
        self.r = self.r_arr

    cdef Py_ssize_t push(self, double t) except -1:
        """Append a time; return the row index for the caller to fill."""
        if self.count == self.cap:
            self.cap *= 2
            new_t = _empty1d(self.cap)
            new_r = _empty2d(self.cap, self.width)
            new_t[:self.count] = self.t_arr[:self.count]
            new_r[:self.count] = self.r_arr[:self.count]
            self.t_arr, self.r_arr = new_t, new_r
            self.t = self.t_arr
            self.r = self.r_arr
        self.t[self.count] = t
        self.count += 1
        return self.count - 1

    cdef int push_row(self, double t, const double* row) except -1:
        cdef Py_ssize_t idx = self.push(t)
        if self.width > 0:
            memcpy(&self.r[idx, 0], row, self.width * sizeof(double))
        return 0

    def times(self):
        return self.t_arr[:self.count].copy()

    def rows(self):
        return self.r_arr[:self.count].copy()



# =============================================================================
# Integrators
# =============================================================================

def _warn_extraneous(extraneous):
    if extraneous:
        warnings.warn(
            "The following arguments have no effect for a chosen solver: "
            + ", ".join(f"`{x}`" for x in extraneous) + ".",
            stacklevel=3,
        )



cdef class RungeKutta:
    """Base class for explicit embedded Runge-Kutta methods with adaptive steps.

    Mirrors :class:`scipy.integrate.OdeSolver`: construct it with a problem,
    then call :meth:`step` until ``status != 'running'``. ``fun`` may be a
    native :class:`OdeSystem` or a Python callable ``fun(t, y)``. All stepping
    happens in the GIL-free core (``fni_solver``).
    """

    def __cinit__(self, *args, **kwargs):
        self.s.mem = NULL
        self.method = -1

    def __dealloc__(self):
        fni_solver_free(&self.s)

    def __init__(self, fun, double t0, y0, double t_bound, double max_step=INFINITY,
                 rtol=1e-3, atol=1e-6, vectorized=False, first_step=None,
                 event_detection="endpoints", n_interior=4, **extraneous):
        cdef fni_options opt
        cdef double[::1] y0_view, rtol_view, atol_view
        cdef void* ctx
        cdef fni_rhs_fn fn
        cdef Py_ssize_t n

        _warn_extraneous(extraneous)
        self._init_tableau()
        detection, n_int = _event_detection(event_detection, n_interior)

        if isinstance(fun, OdeSystem):
            system = <OdeSystem> fun
        else:
            system = PythonOdeSystem(fun, (), vectorized)

        y0_arr = np.asarray(y0)
        if np.issubdtype(y0_arr.dtype, np.complexfloating):
            raise ValueError("`y0` is complex, but the Cython integrators do not "
                             "support integration in a complex domain.")
        y0_arr = np.array(y0_arr, dtype=np.float64, copy=True)
        if y0_arr.ndim != 1:
            raise ValueError("`y0` must be 1-dimensional.")
        if not np.isfinite(y0_arr).all():
            raise ValueError("All components of the initial state `y0` must be finite.")
        n = y0_arr.shape[0]
        system.check_size(n)

        if max_step <= 0:
            raise ValueError("`max_step` must be positive.")
        rtol_arr = np.asarray(rtol, dtype=np.float64)
        if rtol_arr.ndim > 0 and rtol_arr.shape != (n,):
            raise ValueError("`rtol` has wrong shape.")
        if np.any(rtol_arr < 100 * EPS):
            warnings.warn(
                "At least one element of `rtol` is too small. "
                f"Setting `rtol = np.maximum(rtol, {100 * EPS})`.",
                stacklevel=2,
            )
            rtol_arr = np.maximum(rtol_arr, 100 * EPS)
        atol_arr = np.asarray(atol, dtype=np.float64)
        if atol_arr.ndim > 0 and atol_arr.shape != (n,):
            raise ValueError("`atol` has wrong shape.")
        if np.any(atol_arr < 0):
            raise ValueError("`atol` must be positive.")
        if first_step is not None:
            first = float(first_step)
            if first <= 0:
                raise ValueError("`first_step` must be positive.")
            if first > fabs(t_bound - t0):
                raise ValueError("`first_step` exceeds bounds.")
        else:
            first = -1.0

        rtol_b = np.array(np.broadcast_to(rtol_arr, (max(n, 1),)), dtype=np.float64)
        atol_b = np.array(np.broadcast_to(atol_arr, (max(n, 1),)), dtype=np.float64)
        y0_b = y0_arr if n > 0 else np.zeros(1)
        y0_view = y0_b
        rtol_view = rtol_b
        atol_view = atol_b
        fni_options_init(&opt)
        opt.rtol_vec = &rtol_view[0]
        opt.atol_vec = &atol_view[0]
        opt.first_step = first
        opt.max_step = max_step
        opt.event_detection = detection
        opt.n_interior = n_int

        self.system = system
        self.message = None
        fn = _rhs_of(system, &ctx)
        fni_solver_free(&self.s)
        _check(fni_solver_init(&self.s, self.method, fn, ctx, n, t0, &y0_view[0], t_bound, &opt))

    def _init_tableau(self):
        raise NotImplementedError("use RK23, RK45 or DOP853")

    # --- public Python API (mirrors scipy.integrate.OdeSolver) ----------------

    @property
    def n(self):
        return self.s.n

    @property
    def t(self):
        return self.s.t

    @property
    def t_bound(self):
        return self.s.t_bound

    @property
    def direction(self):
        return self.s.direction

    @property
    def h_abs(self):
        return self.s.h_abs

    @property
    def nfev(self):
        return self.s.nfev

    @property
    def n_accepted(self):
        return self.s.n_accepted

    @property
    def n_rejected(self):
        return self.s.n_rejected

    @property
    def n_stages(self):
        return self.s.tab.n_stages

    @property
    def order(self):
        return self.s.tab.order

    @property
    def error_estimator_order(self):
        return self.s.tab.error_estimator_order

    @property
    def status(self):
        return _STATUS_NAMES[self.s.status]

    @property
    def y(self):
        """Copy of the current state."""
        return _array_from(self.s.y, self.s.n)

    @property
    def t_old(self):
        """Time at the start of the last step, or None before the first step."""
        return self.s.t_old if self.s.has_t_old else None

    @property
    def step_size(self):
        """Size of the last successful step, or None before the first step."""
        return fabs(self.s.t - self.s.t_old) if self.s.has_t_old else None

    @property
    def njev(self):
        return 0

    @property
    def nlu(self):
        return 0

    def step(self):
        """Advance one accepted step. Returns the failure message, or None."""
        if self.s.status != FNI_RUNNING:
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        self._step()
        return self.message

    def dense_output(self):
        """Interpolant over the last successful step, as a one-segment :class:`OdeSolution`."""
        if not self.s.has_t_old:
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")
        self._ensure_dense()
        cdef Py_ssize_t n = self.s.n, ncoef = self.s.tab.n_dense_coef
        row = np.concatenate((
            [self.s.t_old, self.s.t - self.s.t_old],
            _array_from(self.s.y_old, n),
            _array_from(self.s.dense, ncoef * n),
        ))
        return OdeSolution([self.s.t_old, self.s.t], row[None, :], self.s.tab.dense_kind,
                           n, ncoef)

    cdef int _step(self) except -1:
        _check(fni_solver_step(&self.s))
        if self.s.status == FNI_FAILED:
            self.message = MSG_TOO_SMALL_STEP
        return 0

    cdef int _ensure_dense(self) except -1:
        return _check(fni_solver_ensure_dense(&self.s))


cdef class RK23(RungeKutta):
    """Bogacki-Shampine 3(2) pair (``scipy.integrate.RK23``)."""

    def _init_tableau(self):
        self.method = FNI_RK23


cdef class RK45(RungeKutta):
    """Dormand-Prince 5(4) pair (``scipy.integrate.RK45``)."""

    def _init_tableau(self):
        self.method = FNI_RK45


cdef class DOP853(RungeKutta):
    """Hairer-Wanner 8(5,3) method (``scipy.integrate.DOP853``).

    Uses the blended 5th/3rd-order error estimate. Dense output evaluates the
    three extra stages of the 7th-order continuous extension on demand.
    """

    def _init_tableau(self):
        self.method = FNI_DOP853


METHODS = {"RK23": RK23, "RK45": RK45, "DOP853": DOP853}


# =============================================================================
# Event preparation
# =============================================================================

def _prepare_events(events, args):
    """Normalise ``events`` to EventFunction objects (scipy's prepare_events)."""
    if events is None:
        return [], np.empty(0), np.empty(0)
    if callable(events) or isinstance(events, EventFunction):
        events = (events,)
    out = []
    max_events = np.empty(len(events))
    direction = np.empty(len(events))
    message = ("The `terminal` attribute of each event "
               "must be a boolean or positive integer.")
    for i, event in enumerate(events):
        if isinstance(event, EventFunction):
            ev = event
        else:
            ev = PythonEvent(event, () if args is None else args)
        terminal = ev.terminal
        direction[i] = ev.direction
        if terminal is None or terminal == 0:
            max_events[i] = np.inf
        elif int(terminal) == terminal and terminal > 0:
            max_events[i] = terminal
        else:
            raise ValueError(message)
        out.append(ev)
    return out, max_events, direction



cdef class _EventArray:
    """Owns a C array of fni_event built from EventFunction objects (kept alive)."""

    cdef fni_event* ev
    cdef Py_ssize_t n
    cdef list objs
    cdef bint all_native

    def __cinit__(self, list events, max_events, Py_ssize_t n_states):
        cdef Py_ssize_t e
        cdef EventFunction obj
        self.n = len(events)
        self.objs = events
        self.all_native = True
        self.ev = <fni_event*> malloc((self.n if self.n > 0 else 1) * sizeof(fni_event))
        if self.ev == NULL:
            raise MemoryError()
        for e in range(self.n):
            obj = <EventFunction> events[e]
            obj.check_size(n_states)
            _event_of(obj, max_events[e], &self.ev[e])
            if obj.c_fn == NULL:
                self.all_native = False

    def __dealloc__(self):
        if self.ev != NULL:
            free(self.ev)


# =============================================================================
# Driver
# =============================================================================

class OdeResult(dict):
    """Integration result: scipy's ``OdeResult`` fields, with attribute access.

    Adds ``n_accepted`` / ``n_rejected`` step statistics.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())

    def __repr__(self):
        return (f"OdeResult(success={self.success}, status={self.status}, "
                f"nfev={self.nfev}, n_points={np.size(self.t)}, "
                f"message={self.message!r})")



cdef object _drive(RungeKutta solver, double t0, object t_eval_obj,
                   Py_ssize_t t_eval_i, bint dense_output, _EventArray events,
                   ResetMap reset=None, object hybrid=None):
    """The solve_ivp main loop (same control flow as scipy's), on the C core.

    With ``hybrid`` (a dict of fni_hybrid_options fields), terminal events are
    guards handled by ``reset`` through ``fni_hybrid_reset``, exactly as in
    ``fni_integrate_hybrid``.
    """
    cdef fni_solver* s = &solver.s
    cdef Py_ssize_t n = s.n
    cdef Py_ssize_t ncoef = s.tab.n_dense_coef
    cdef Py_ssize_t n_events = events.n
    cdef bint has_t_eval = t_eval_obj is not None
    cdef bint is_hybrid = hybrid is not None
    cdef const double[::1] t_eval
    cdef Py_ssize_t t_eval_i_new, j, k, e, idx, e_found
    cdef int status = 2   # 2 = still running
    cdef int hstatus
    cdef double t_old, t, root
    cdef const double* y_ptr
    cdef double* seg_row
    cdef bint terminate, reset_done, reset_called
    cdef fni_hybrid_options hopt
    cdef fni_hybrid_state hst
    cdef fni_event_scan scan
    cdef fni_reset_fn reset_fn = NULL
    cdef void* reset_ctx = NULL

    if has_t_eval:
        t_eval = t_eval_obj

    traj = _RowBuffer(n)
    cdef _RowBuffer segs = _RowBuffer(2 + n + ncoef * n) if dense_output else None
    # Hybrid dense output always keeps its own breakpoints: the trajectory
    # repeats each reset time (before and after), which ts may not.
    cdef _RowBuffer bounds = (_RowBuffer(0) if (dense_output and (has_t_eval or is_hybrid))
                              else None)
    cdef double[::1] buf = _empty1d(max(n, 1))       # interpolation scratch
    cdef double[::1] y_term = _empty1d(max(n, 1))    # state at a terminal event
    cdef double[::1] y_minus = _empty1d(max(n, 1))   # hybrid: state before a reset

    if not has_t_eval:
        traj.push_row(t0, s.y)
    if bounds is not None:
        bounds.push(t0)

    # Event bookkeeping.
    cdef double[::1] g = _empty1d(max(n_events, 1))
    cdef double[::1] g_new = _empty1d(max(n_events, 1))
    cdef double[::1] counts = np.zeros(max(n_events, 1))
    t_events = [[] for _ in range(n_events)]
    y_events = [[] for _ in range(n_events)]
    reset_t, reset_e, reset_before, reset_after = [], [], [], []
    if n_events > 0:
        _check(fni_events_eval(events.ev, n_events, t0, s.y, n, &g[0]))

    hst.masked = NULL
    hst.t_probe = NULL
    scan.mem = NULL
    _check(fni_event_scan_init(&scan, n_events, n))
    if is_hybrid:
        hopt.max_resets = hybrid["max_resets"]
        hopt.min_dwell = hybrid["min_dwell"]
        hopt.reselect_step = hybrid["reselect_step"]
        if reset is not None:
            reset_fn = _reset_of(reset, &reset_ctx)
        _check(fni_hybrid_init(&hst, n_events))

    try:
        while status == 2:
            solver._step()
            if s.status == FNI_FINISHED:
                status = 0
            elif s.status == FNI_FAILED:
                status = -1
                break

            t_old = s.t_old
            t = s.t
            y_ptr = s.y
            reset_done = False
            reset_called = False

            if dense_output:
                solver._ensure_dense()
                idx = segs.push(t_old)
                seg_row = &segs.r[idx, 0]
                seg_row[0] = t_old
                seg_row[1] = t - t_old
                if n > 0:
                    memcpy(seg_row + 2, s.y_old, n * sizeof(double))
                    memcpy(seg_row + 2 + n, s.dense, ncoef * n * sizeof(double))

            if n_events > 0:
                _check(fni_events_begin(s, events.ev, n_events, &g[0], &g_new[0],
                                        &hst if is_hybrid else NULL, &scan))
                terminate = False
                while not terminate:
                    _check(fni_events_next(s, events.ev, n_events, &counts[0],
                                           &hst if is_hybrid else NULL, &scan, &e_found,
                                           &root, &terminate))
                    if e_found < 0:
                        break
                    t_events[e_found].append(root)
                    fni_solver_dense_eval(s, root, &buf[0])
                    y_events[e_found].append(_array_from(&buf[0], n))
                if terminate:
                    status = 1
                    t = root
                    fni_solver_dense_eval(s, t, &y_term[0])
                    y_ptr = &y_term[0]
                    if is_hybrid:
                        if n > 0:
                            memcpy(&y_minus[0], &y_term[0], n * sizeof(double))
                        # Take the t_eval points up to the event from this step's
                        # interpolant before a restart discards it.
                        if has_t_eval:
                            _take_t_eval(s, t_eval, &t_eval_i, t, traj, n)
                        _check(fni_hybrid_reset(s, &hst, &hopt, events.ev, n_events,
                                                &counts[0], e_found, t,
                                                &y_term[0], reset_fn, reset_ctx, &g[0],
                                                &hstatus))
                        reset_called = reset_fn != NULL and (hstatus == 2
                                                             or hstatus == FNI_STATUS_EVENT)
                        if reset_called:
                            reset_t.append(t)
                            reset_e.append(e_found)
                            reset_before.append(_array_from(&y_minus[0], n))
                            reset_after.append(_array_from(&y_term[0], n))
                        status = 2 if hstatus == 2 else hstatus
                        reset_done = True
                if not reset_done:
                    if is_hybrid:
                        fni_hybrid_unmask(&hst, n_events, &g_new[0])
                    g, g_new = g_new, g

            if is_hybrid:
                if not has_t_eval:
                    if reset_done:
                        # Both sides of the event: the state before, then after.
                        traj.push_row(t, &y_minus[0])
                        if reset_called:
                            traj.push_row(t, &y_term[0])
                    elif not (t == t_old and traj.t[traj.count - 1] == t):
                        traj.push_row(t, y_ptr)
                elif not reset_done:
                    _take_t_eval(s, t_eval, &t_eval_i, t, traj, n)
                if bounds is not None:
                    if bounds.t[bounds.count - 1] == t:
                        segs.count -= 1     # zero-length piece (event at the step start)
                    else:
                        bounds.push(t)
                continue

            if not has_t_eval:
                if dense_output and traj.count > 1 and traj.t[traj.count - 1] == t:
                    segs.count -= 1    # scipy's donot_append: drop the duplicate segment
                else:
                    traj.push_row(t, y_ptr)
            else:
                _take_t_eval(s, t_eval, &t_eval_i, t, traj, n)
                if bounds is not None:
                    bounds.push(t)
    finally:
        fni_hybrid_free(&hst)
        fni_event_scan_free(&scan)

    message = MESSAGES.get(status, solver.message)

    if n_events > 0:
        t_events_out = [np.asarray(te) for te in t_events]
        y_events_out = [np.asarray(ye) for ye in y_events]
    else:
        t_events_out = None
        y_events_out = None

    ts = traj.times()
    ys = traj.rows().T

    if dense_output:
        sol = OdeSolution(ts if bounds is None else bounds.times(),
                          segs.rows(), s.tab.dense_kind, n, ncoef)
    else:
        sol = None

    out = OdeResult(
        t=ts, y=ys, sol=sol, t_events=t_events_out, y_events=y_events_out,
        nfev=s.nfev, njev=0, nlu=0, status=status, message=message,
        success=status >= 0, n_accepted=s.n_accepted, n_rejected=s.n_rejected,
    )
    if is_hybrid:
        out.t_resets = np.asarray(reset_t, dtype=np.float64)
        out.reset_events = np.asarray(reset_e, dtype=np.intp)
        out.y_before_resets = np.asarray(reset_before).reshape(-1, n)
        out.y_after_resets = np.asarray(reset_after).reshape(-1, n)
    return out


cdef int _take_t_eval(fni_solver* s, const double[::1] t_eval, Py_ssize_t* t_eval_i,
                      double t, _RowBuffer traj, Py_ssize_t n) except -1:
    """Interpolate the t_eval points reached by time ``t`` in the last step."""
    cdef Py_ssize_t j, idx, t_eval_i_new
    # The value in t_eval equal to t is included.
    if s.direction > 0:
        t_eval_i_new = _searchsorted(t_eval, t, False)
        if t_eval_i_new > t_eval_i[0]:
            _check(fni_solver_ensure_dense(s))
            for j in range(t_eval_i[0], t_eval_i_new):
                idx = traj.push(t_eval[j])
                if n > 0:
                    fni_solver_dense_eval(s, t_eval[j], &traj.r[idx, 0])
            t_eval_i[0] = t_eval_i_new
    else:
        t_eval_i_new = _searchsorted(t_eval, t, True)
        if t_eval_i[0] > t_eval_i_new:
            _check(fni_solver_ensure_dense(s))
            for j in range(t_eval_i[0] - 1, t_eval_i_new - 1, -1):
                idx = traj.push(t_eval[j])
                if n > 0:
                    fni_solver_dense_eval(s, t_eval[j], &traj.r[idx, 0])
            t_eval_i[0] = t_eval_i_new
    return 0


_EVENT_DETECTION = {"endpoints": FNI_EVENTS_ENDPOINTS, "interior": FNI_EVENTS_INTERIOR}


def _event_detection(event_detection, n_interior):
    """Validate the event detection options; return (mode id, n_interior)."""
    if event_detection not in _EVENT_DETECTION:
        raise ValueError(f"`event_detection` must be one of {list(_EVENT_DETECTION)}.")
    n_interior = operator.index(n_interior)
    if n_interior < 1:
        raise ValueError("`n_interior` must be at least 1.")
    return _EVENT_DETECTION[event_detection], n_interior


def _resolve_method(method):
    if isinstance(method, str):
        if method not in METHODS:
            raise ValueError(f"`method` must be one of {list(METHODS)} or a "
                             "RungeKutta subclass.")
        return METHODS[method]
    if isinstance(method, type) and issubclass(method, RungeKutta):
        return method
    raise ValueError(f"`method` must be one of {list(METHODS)} or a RungeKutta subclass.")


def _normalize_args(args):
    if args is None:
        return None
    try:
        return tuple(args)
    except TypeError as exp:
        raise TypeError(
            "Supplied 'args' cannot be unpacked. Please supply `args`"
            f" as a tuple (e.g. `args=({args},)`)"
        ) from exp


def _as_system(fun, args, vectorized):
    if isinstance(fun, OdeSystem):
        if args is not None:
            raise TypeError("`args` cannot be used with a native OdeSystem; set "
                            "runtime parameters with `system.params = ...` instead.")
        return fun
    return PythonOdeSystem(fun, () if args is None else args, vectorized)


def solve_ivp(fun, t_span, y0, method="RK45", t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):
    """Solve an initial value problem; drop-in for :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    fun : OdeSystem or callable
        Right-hand side. Pass a native :class:`OdeSystem` for full speed. A
        Python callable ``fun(t, y, *args)`` also works, but every evaluation
        goes back through Python.
    t_span : (float, float)
        Interval of integration ``(t0, tf)``; ``tf < t0`` integrates backwards.
    y0 : array_like, shape (n,)
        Initial state (real-valued).
    method : {'RK23', 'RK45', 'DOP853'} or RungeKutta subclass
        Default 'RK45', as in scipy.
    t_eval : array_like, optional
        Times at which to store the solution (interpolated with dense output).
    dense_output : bool
        If True, ``result.sol`` is an :class:`OdeSolution`.
    events : callable, EventFunction, or list of them
        Python events follow scipy's conventions (``terminal`` / ``direction``
        attributes). Native :class:`EventFunction` objects avoid the Python
        call overhead.
    vectorized : bool
        For Python callables: call ``fun`` with ``y`` of shape (n, 1).
    args : tuple, optional
        Extra arguments for Python callables ``fun`` and ``events``. Native
        systems take runtime parameters through ``system.params`` instead.
    **options
        ``first_step``, ``max_step``, ``rtol``, ``atol`` (scalar or per
        component). Unknown options trigger a warning, as in scipy. Two
        options go beyond scipy:

        ``event_detection`` : {'endpoints', 'interior'}
            How a crossing is detected within a step. ``'endpoints'`` (the
            default) does as scipy does: it compares each event's sign at the
            two ends of the step, so a pair of crossings inside one step
            (in and back out) goes unseen. ``'interior'`` also compares the
            signs at ``n_interior`` evenly spaced points inside every step,
            evaluated on the step's interpolant. It catches pairs of
            crossings at least ``h / (n_interior + 1)`` apart, at the cost of
            building the interpolant on every step (3 extra RHS evaluations
            per step for DOP853, none for RK23 and RK45) and ``n_interior``
            event evaluations.
        ``n_interior`` : int
            Interior test points per step for ``'interior'`` (default 4).

        In both modes, every root located in a step splits it for the other
        events: an event that crosses and crosses back around another
        event's root is found too, and roots are reported in time order.

    Returns
    -------
    OdeResult
        ``t``, ``y``, ``sol``, ``t_events``, ``y_events``, ``nfev``, ``njev``,
        ``nlu``, ``status`` (-1 failed, 0 finished, 1 terminal event),
        ``message``, ``success``, plus ``n_accepted`` and ``n_rejected``.
    """
    return _solve(fun, t_span, y0, method, t_eval, dense_output, events, vectorized, args,
                  options, None, None)


cdef object _solve(fun, t_span, y0, method, t_eval, dense_output, events, vectorized, args,
                   dict options, object reset, object hybrid):
    cdef Py_ssize_t t_eval_i = 0
    cdef ResetMap reset_map = None

    solver_cls = _resolve_method(method)
    t0, tf = map(float, t_span)
    args = _normalize_args(args)
    system = _as_system(fun, args, vectorized)

    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=np.float64)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")
        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")
        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")
        if tf > t0:
            t_eval_i = 0
        else:
            # Make t_eval increasing so it can be binary-searched.
            t_eval = t_eval[::-1]
            t_eval_i = t_eval.shape[0]
        t_eval = np.ascontiguousarray(t_eval)

    solver = solver_cls(system, t0, y0, tf, **options)

    event_list, max_events, _ = _prepare_events(events, args)
    ev_array = _EventArray(event_list, max_events, solver.n)
    if hybrid is not None:
        reset_map = _as_reset(reset, args, solver.n)

    return _drive(solver, t0, t_eval, t_eval_i, dense_output, ev_array, reset_map, hybrid)


def _as_reset(reset, args, Py_ssize_t n):
    if reset is None:
        return None
    if not isinstance(reset, ResetMap):
        reset = PythonReset(reset, () if args is None else args)
    (<ResetMap> reset).check_size(n)
    return reset


def _hybrid_options(max_resets, min_dwell, reselect_step):
    max_resets = -1 if max_resets is None else int(max_resets)
    min_dwell = float(min_dwell)
    if not min_dwell >= 0:
        raise ValueError("`min_dwell` must be non-negative.")
    return {"max_resets": max_resets, "min_dwell": min_dwell,
            "reselect_step": bool(reselect_step)}


def solve_hybrid(fun, t_span, y0, events, reset, method="RK45", t_eval=None,
                 dense_output=False, vectorized=False, args=None, max_resets=10000,
                 min_dwell=0.0, reselect_step=False, **options):
    """Solve a hybrid system: a flow with guards and reset maps.

    Like :func:`solve_ivp`, except that every **terminal** event is a guard.
    When one fires, ``reset`` maps the state across it and integration
    continues from the event, in the same solver and with the step size
    carried over. Non-terminal events are only recorded, as in ``solve_ivp``.
    The flow may depend on a discrete mode (a bang-bang arc, a contact state)
    that ``reset`` updates. Keep the mode in an object the system and the
    reset map share.

    Parameters
    ----------
    fun, t_span, y0, method, t_eval, dense_output, vectorized, args, **options
        As in :func:`solve_ivp`.
    events : callable, EventFunction, or list of them
        Guards are the terminal ones (``terminal=True``, or a count *k*: the
        guard fires on every *k*-th crossing).
    reset : ResetMap or callable
        ``reset(event, t, y, *args)``: ``event`` is the index of the guard in
        ``events`` and ``y`` the state just before it. Return the state just
        after (length n) to continue, or ``None`` to stop with status 1. A
        native :class:`ResetMap` avoids the Python call overhead.
    max_resets : int or None
        Stop with status -2 when one more reset would exceed this many
        (``None``: no limit).
    min_dwell : float
        Stop with status -3 (Zeno behaviour) if a reset follows the previous
        one by less than this. ``0`` disables the check.
    reselect_step : bool
        Rerun the initial-step heuristic after each reset instead of keeping
        the controller's current step size.

    Returns
    -------
    OdeResult
        The fields of :func:`solve_ivp`, plus ``t_resets``, ``reset_events``,
        ``y_before_resets`` and ``y_after_resets``. Without ``t_eval``, ``t``
        and ``y`` contain each reset time twice: the state before it, then
        after. ``sol`` is piecewise across resets and returns the state before
        a reset at its exact time. ``t_events`` / ``y_events`` record every
        event with the state before it.

    Notes
    -----
    Right after a reset, the guard that fired sits on zero, and round-off
    decides its sign. FNI therefore probes it slightly later along the new
    flow (as SUNDIALS' root finder does). That probe sets which side it
    starts on, so the zero it sits on cannot fire it again, while a real
    crossing later in the same step still does. A guard that stays exactly
    zero is masked until it moves away. Events are re-evaluated after the
    reset, and a jump of the state is never counted as a crossing.
    """
    return _solve(fun, t_span, y0, method, t_eval, dense_output, events, vectorized, args,
                  options, reset, _hybrid_options(max_resets, min_dwell, reselect_step))


# =============================================================================
# Endpoint integration on the C API
# =============================================================================

def integrate(fun, t_span, y0, method="RK45", *, rtol=1e-3, atol=1e-6, first_step=None,
              max_step=np.inf, events=None, args=None, max_records=1024, release_gil=None,
              event_detection="endpoints", n_interior=4):
    """Integrate to the end of ``t_span`` (or a terminal event) via ``fni_integrate``.

    The lean counterpart to :func:`solve_ivp` for when only the endpoint (and
    the event locations) matter, as in shooting methods. No trajectory is
    stored. It takes exactly the same steps as :func:`solve_ivp`.

    If ``fun`` and every event provide a native C function (``has_native_fn``),
    the whole integration runs without the GIL, so Python threads calling
    ``integrate`` run in parallel. ``release_gil=True`` demands that (raising if
    impossible), ``False`` keeps the GIL, and ``None`` decides automatically.
    ``event_detection`` and ``n_interior`` are as in :func:`solve_ivp`.

    Returns
    -------
    OdeResult
        ``t`` (final time), ``y`` (final state, shape (n,)), ``status``,
        ``message``, ``success``, ``nfev``, ``n_accepted``, ``n_rejected``,
        ``t_events`` / ``y_events`` (as in solve_ivp; ``None`` without events),
        ``n_event_records`` (all located events, which may exceed
        ``max_records``) and ``gil_released``.
    """
    return _integrate(fun, t_span, y0, method, rtol, atol, first_step, max_step, events, args,
                      max_records, release_gil, None, None, event_detection, n_interior)


def integrate_hybrid(fun, t_span, y0, events, reset, method="RK45", *, rtol=1e-3, atol=1e-6,
                     first_step=None, max_step=np.inf, args=None, max_resets=10000,
                     min_dwell=0.0, reselect_step=False, max_records=1024,
                     release_gil=None, event_detection="endpoints", n_interior=4):
    """Integrate a hybrid system to the end of ``t_span`` via ``fni_integrate_hybrid``.

    The endpoint counterpart of :func:`solve_hybrid` (guards, reset maps and
    limits as documented there), as :func:`integrate` is of
    :func:`solve_ivp`: same steps, same events and resets, no trajectory. It
    runs without the GIL when the system, every event and the reset map are
    native.

    Returns
    -------
    OdeResult
        The fields of :func:`integrate`, plus ``n_resets`` and, over the
        recorded events, ``t_resets``, ``reset_events``, ``y_before_resets``
        and ``y_after_resets``.
    """
    return _integrate(fun, t_span, y0, method, rtol, atol, first_step, max_step, events, args,
                      max_records, release_gil, reset,
                      _hybrid_options(max_resets, min_dwell, reselect_step), event_detection,
                      n_interior)


cdef object _integrate(fun, t_span, y0, method, rtol, atol, first_step, max_step, events,
                       args, max_records, release_gil, reset, hybrid, event_detection,
                       n_interior):
    cdef fni_options opt
    cdef fni_hybrid_options hopt
    cdef fni_reset_fn reset_fn = NULL
    cdef void* reset_ctx = NULL
    cdef ResetMap reset_map = None
    cdef double[::1] rec_y_after
    cdef int[::1] rec_reset
    cdef Py_ssize_t n_resets = 0
    cdef fni_stats stats
    cdef void* ctx
    cdef fni_rhs_fn fn
    cdef double[::1] y0_view, y_out, rtol_view, atol_view, rec_t, rec_y_flat
    cdef Py_ssize_t[::1] rec_e
    cdef Py_ssize_t n, cap
    cdef int m, rc
    cdef double t0, t1
    cdef bint nogil_ok
    cdef _EventArray ev_array

    solver_cls = _resolve_method(method)
    m = METHOD_IDS[solver_cls.__name__] if solver_cls.__name__ in METHOD_IDS else -1
    if m < 0:
        raise ValueError("integrate() supports RK23, RK45 and DOP853")
    t0, t1 = map(float, t_span)
    args = _normalize_args(args)
    system = _as_system(fun, args, False)

    y0_arr = np.array(y0, dtype=np.float64, copy=True).reshape(-1)
    if not np.isfinite(y0_arr).all():
        raise ValueError("All components of the initial state `y0` must be finite.")
    n = y0_arr.shape[0]
    (<OdeSystem> system).check_size(n)
    if not max_step > 0:
        raise ValueError("`max_step` must be positive.")
    if first_step is not None and not (0 < first_step <= abs(t1 - t0)):
        raise ValueError("`first_step` must be positive and within bounds.")
    rtol_arr = np.maximum(np.asarray(rtol, dtype=np.float64), 100 * EPS)
    atol_arr = np.asarray(atol, dtype=np.float64)
    for name, arr in (("rtol", rtol_arr), ("atol", atol_arr)):
        if arr.ndim > 0 and arr.shape != (n,):
            raise ValueError(f"`{name}` has wrong shape.")
    if np.any(atol_arr < 0):
        raise ValueError("`atol` must be positive.")

    rtol_view = np.array(np.broadcast_to(rtol_arr, (max(n, 1),)), dtype=np.float64)
    atol_view = np.array(np.broadcast_to(atol_arr, (max(n, 1),)), dtype=np.float64)
    y0_view = y0_arr if n > 0 else np.zeros(1)
    y_out = np.zeros(max(n, 1))
    fni_options_init(&opt)
    opt.rtol_vec = &rtol_view[0]
    opt.atol_vec = &atol_view[0]
    opt.first_step = -1.0 if first_step is None else float(first_step)
    opt.max_step = max_step
    opt.event_detection, opt.n_interior = _event_detection(event_detection, n_interior)

    event_list, max_events, _ = _prepare_events(events, args)
    ev_array = _EventArray(event_list, max_events, n)
    cap = max(int(max_records), 0) if ev_array.n > 0 else 0
    rec_e = np.zeros(max(cap, 1), dtype=np.intp)
    rec_t = np.zeros(max(cap, 1))
    rec_y_flat = np.zeros(max(cap * n, 1))
    rec_y_after = np.zeros(max(cap * n, 1))
    rec_reset = np.zeros(max(cap, 1), dtype=np.intc)

    fn = _rhs_of(<OdeSystem> system, &ctx)
    nogil_ok = (<OdeSystem> system).c_fn != NULL and ev_array.all_native
    if hybrid is not None:
        hopt.max_resets = hybrid["max_resets"]
        hopt.min_dwell = hybrid["min_dwell"]
        hopt.reselect_step = hybrid["reselect_step"]
        reset_map = _as_reset(reset, args, n)
        if reset_map is not None:
            reset_fn = _reset_of(reset_map, &reset_ctx)
            nogil_ok = nogil_ok and reset_map.c_fn != NULL
    else:
        hopt.max_resets = -1
        hopt.min_dwell = 0.0
        hopt.reselect_step = False
    if release_gil is None:
        release_gil = nogil_ok
    elif release_gil and not nogil_ok:
        raise TypeError("release_gil=True needs a native c_fn on the system, every event "
                        "and the reset map")

    if release_gil:
        with nogil:
            rc = fni_integrate_hybrid(m, fn, ctx, n, t0, &y0_view[0], t1, &opt, ev_array.ev,
                                      ev_array.n, reset_fn, reset_ctx, &hopt, &y_out[0],
                                      &rec_e[0], &rec_t[0], &rec_y_flat[0], &rec_y_after[0],
                                      <bint*> &rec_reset[0], cap, &stats, &n_resets)
    else:
        rc = fni_integrate_hybrid(m, fn, ctx, n, t0, &y0_view[0], t1, &opt, ev_array.ev,
                                  ev_array.n, reset_fn, reset_ctx, &hopt, &y_out[0],
                                  &rec_e[0], &rec_t[0], &rec_y_flat[0], &rec_y_after[0],
                                  <bint*> &rec_reset[0], cap, &stats, &n_resets)
    _check(rc)

    if ev_array.n > 0:
        k_rec = min(stats.n_records, cap)
        e_arr = np.asarray(rec_e)[:k_rec]
        t_arr = np.asarray(rec_t)[:k_rec]
        y_arr = np.asarray(rec_y_flat)[:k_rec * n].reshape(k_rec, n)
        t_events = [t_arr[e_arr == e].copy() for e in range(ev_array.n)]
        y_events = [y_arr[e_arr == e].copy() if np.any(e_arr == e) else np.asarray([])
                    for e in range(ev_array.n)]
    else:
        t_events = y_events = None

    status = stats.status
    out = OdeResult(
        t=stats.t_final, y=np.asarray(y_out)[:n].copy(), status=status,
        message=MESSAGES.get(status, MSG_TOO_SMALL_STEP), success=status >= 0,
        nfev=stats.nfev, n_accepted=stats.n_accepted, n_rejected=stats.n_rejected,
        t_events=t_events, y_events=y_events, n_event_records=stats.n_records,
        gil_released=bool(release_gil),
    )
    if hybrid is not None:
        k_rec = min(stats.n_records, cap)
        is_reset = np.asarray(rec_reset)[:k_rec].astype(bool)
        out.n_resets = n_resets
        out.t_resets = np.asarray(rec_t)[:k_rec][is_reset].copy()
        out.reset_events = np.asarray(rec_e)[:k_rec][is_reset].copy()
        out.y_before_resets = np.asarray(rec_y_flat)[:k_rec * n].reshape(k_rec, n)[is_reset].copy()
        out.y_after_resets = np.asarray(rec_y_after)[:k_rec * n].reshape(k_rec, n)[is_reset].copy()
    return out
