# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython explicit Runge-Kutta integrators: RK23, RK45 and DOP853.

A step-for-step reimplementation of :class:`scipy.integrate.RK23`, ``RK45`` and
``DOP853``. It uses the same tableaus, the same initial-step heuristic, the same
error norms and the same step-size controller (safety 0.9, factors in
[0.2, 10]), so it takes the same steps and makes the same number of RHS
evaluations as ``solve_ivp``. Differences are limited to floating-point
summation order.

Every state-sized operation is a typed-memoryview loop over arrays that cnumpy
allocates once per solve. The step loop makes no NumPy calls and creates no
temporary arrays. When the right-hand side is a native :class:`OdeSystem`
subclass, the whole integration runs without entering the Python interpreter
except to record accepted steps.

Not yet implemented (planned): dense output, ``t_eval``, and events.
"""

cimport numpy as cnp
from libc.math cimport fabs, sqrt, pow, nextafter, INFINITY
from libc.string cimport memcpy

import numpy as np

from fast_numerical_integration import tableaus

cnp.import_array()

# Step-size controller constants (identical to scipy.integrate._ivp.rk).
cdef double SAFETY = 0.9
cdef double MIN_FACTOR = 0.2
cdef double MAX_FACTOR = 10.0

cdef enum:
    STATUS_RUNNING = 0
    STATUS_FINISHED = 1
    STATUS_FAILED = -1

_STATUS_NAMES = {STATUS_RUNNING: "running", STATUS_FINISHED: "finished",
                 STATUS_FAILED: "failed"}

MSG_SUCCESS = "The solver successfully reached the end of the integration interval."
MSG_TOO_SMALL_STEP = "Required step size is less than spacing between numbers."


cdef inline cnp.ndarray _empty1d(Py_ssize_t n):
    cdef cnp.npy_intp dims[1]
    dims[0] = n
    return cnp.PyArray_EMPTY(1, dims, cnp.NPY_FLOAT64, 0)


cdef inline cnp.ndarray _empty2d(Py_ssize_t rows, Py_ssize_t cols):
    cdef cnp.npy_intp dims[2]
    dims[0] = rows
    dims[1] = cols
    return cnp.PyArray_EMPTY(2, dims, cnp.NPY_FLOAT64, 0)


cdef inline double _rms_norm_scaled(const double* x, const double* scale, Py_ssize_t n) noexcept:
    """RMS norm of ``x / scale`` (scipy's ``norm(x / scale)``)."""
    cdef Py_ssize_t i
    cdef double v, acc = 0.0
    for i in range(n):
        v = x[i] / scale[i]
        acc += v * v
    return sqrt(acc / n)


# =============================================================================
# Right-hand sides
# =============================================================================

cdef class OdeSystem:
    """Base class for native right-hand sides ``dy/dt = f(t, y)``.

    Subclass in Cython and override the ``cdef`` method ``eval``. Instances are
    also callable from Python as ``system(t, y) -> ndarray``, so the same object
    works with :func:`scipy.integrate.solve_ivp` for comparisons.
    """

    def __cinit__(self, *args, **kwargs):
        self.n_states = -1

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        raise NotImplementedError(f"{type(self).__name__} must implement eval()")

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
    """

    cdef object fun
    cdef tuple args

    def __init__(self, fun, args=()):
        if not callable(fun):
            raise TypeError("fun must be callable")
        self.fun = fun
        self.args = tuple(args)

    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1:
        # Give the callable a fresh array each call (as scipy does) so it may
        # safely keep a reference to its input.
        y_arr = _empty1d(n)
        cdef double[::1] y_view = y_arr
        memcpy(&y_view[0], y, n * sizeof(double))

        result = self.fun(t, y_arr, *self.args)
        cdef const double[::1] r_view = np.ascontiguousarray(result, dtype=np.float64).reshape(-1)
        if r_view.shape[0] != n:
            raise ValueError(
                f"fun returned {r_view.shape[0]} values, expected {n}")
        memcpy(dydt, &r_view[0], n * sizeof(double))
        return 0


# =============================================================================
# Integrators
# =============================================================================

cdef class RungeKutta:
    """Base class for explicit embedded Runge-Kutta methods with adaptive steps.

    Mirrors :class:`scipy.integrate.OdeSolver`: construct it with a problem,
    then call :meth:`step` until ``status != 'running'``. Most users want
    :func:`solve_ivp` instead.
    """

    def __init__(self, OdeSystem system not None, double t0, y0, double t_bound,
                 *, double rtol=1e-3, atol=1e-6, first_step=None,
                 double max_step=INFINITY):
        cdef Py_ssize_t i
        self._init_tableau()

        y0_arr = np.array(y0, dtype=np.float64, copy=True).reshape(-1)
        self.n = y0_arr.shape[0]
        if system.n_states >= 0 and system.n_states != self.n:
            raise ValueError(
                f"system expects {system.n_states} states, y0 has {self.n}")

        if max_step <= 0:
            raise ValueError("`max_step` must be positive.")
        if not rtol > 0:
            raise ValueError("`rtol` must be positive.")
        atol_arr = np.asarray(atol, dtype=np.float64)
        if atol_arr.ndim > 0 and atol_arr.shape != (self.n,):
            raise ValueError("`atol` has wrong shape.")
        if np.any(atol_arr < 0):
            raise ValueError("`atol` must be positive.")
        if rtol < 100 * np.finfo(np.float64).eps:
            # scipy warns and clamps; we clamp silently to the same floor.
            rtol = 100 * np.finfo(np.float64).eps

        self.system = system
        self.t = t0
        self.t_bound = t_bound
        self.direction = 1.0 if t_bound >= t0 else -1.0
        if t_bound == t0:
            self.direction = 1.0
        self.rtol = rtol
        self.max_step = max_step

        # cnumpy-allocated working storage, viewed through typed memoryviews.
        self.y = y0_arr
        self.f = _empty1d(self.n)
        self.y_new = _empty1d(self.n)
        self.y_stage = _empty1d(self.n)
        self.scale = _empty1d(self.n)
        self.atol = np.broadcast_to(atol_arr, (self.n,)).copy()
        self.K = _empty2d(self.n_stages + 1, self.n)

        self.error_exponent = -1.0 / (self.error_estimator_order + 1)
        self.h_previous = np.nan
        self.nfev = 0
        self.n_accepted = 0
        self.n_rejected = 0
        self.message = None
        self._status = STATUS_RUNNING

        if self.n == 0:
            self.h_abs = INFINITY
            return

        self._fun(self.t, &self.y[0], &self.f[0])
        if first_step is None:
            self.h_abs = self._select_initial_step()
        else:
            first = float(first_step)
            if first <= 0:
                raise ValueError("`first_step` must be positive.")
            if first > fabs(t_bound - t0):
                raise ValueError("`first_step` exceeds bounds.")
            self.h_abs = first

    def _init_tableau(self):
        raise NotImplementedError("use RK23, RK45 or DOP853")

    # --- public Python API --------------------------------------------------

    @property
    def status(self):
        return _STATUS_NAMES[self._status]

    @property
    def y_current(self):
        """Copy of the current state."""
        return np.asarray(self.y).copy()

    def step(self):
        """Advance one accepted step. Returns the failure message, or None."""
        if self._status != STATUS_RUNNING:
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        self._step()
        return self.message

    # --- C-level core --------------------------------------------------------

    cdef int _fun(self, double t, const double* y, double* dydt) except -1:
        self.nfev += 1
        return self.system.eval(t, y, dydt, self.n)

    cdef double _select_initial_step(self) except -1.0:
        """Hairer-Norsett-Wanner initial step heuristic (scipy's select_initial_step)."""
        cdef Py_ssize_t i, n = self.n
        cdef double interval_length = fabs(self.t_bound - self.t)
        cdef double d0, d1, d2, h0, h1, h
        cdef double* y0 = &self.y[0]
        cdef double* f0 = &self.f[0]
        cdef double* scale = &self.scale[0]
        cdef double* y1 = &self.y_new[0]
        cdef double* f1 = &self.K[0, 0]    # free scratch before the first step
        cdef double* diff = &self.y_stage[0]

        if interval_length == 0.0:
            return 0.0

        for i in range(n):
            scale[i] = self.atol[i] + fabs(y0[i]) * self.rtol
        d0 = _rms_norm_scaled(y0, scale, n)
        d1 = _rms_norm_scaled(f0, scale, n)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        h0 = min(h0, interval_length)

        for i in range(n):
            y1[i] = y0[i] + h0 * self.direction * f0[i]
        self._fun(self.t + h0 * self.direction, y1, f1)
        for i in range(n):
            diff[i] = f1[i] - f0[i]
        d2 = _rms_norm_scaled(diff, scale, n) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = pow(0.01 / max(d1, d2), 1.0 / (self.error_estimator_order + 1))

        return min(100 * h0, h1, interval_length, self.max_step)

    cdef int _rk_stages(self, double h) except -1:
        """Fill K with all stages for step ``h``; write the solution into y_new.

        Each stage argument is built with column sweeps ``acc += a_sj * K[j, :]``
        over contiguous rows, skipping zero tableau entries (DOP853's A is
        roughly half zeros).
        """
        cdef Py_ssize_t s, j, i, n = self.n, ns = self.n_stages
        cdef double a
        cdef double* y = &self.y[0]
        cdef double* acc = &self.y_stage[0]
        cdef double* y_new = &self.y_new[0]
        cdef double* Kj

        memcpy(&self.K[0, 0], &self.f[0], n * sizeof(double))

        for s in range(1, ns):
            for i in range(n):
                acc[i] = 0.0
            for j in range(s):
                a = self.A[s, j]
                if a != 0.0:
                    Kj = &self.K[j, 0]
                    for i in range(n):
                        acc[i] += a * Kj[i]
            for i in range(n):
                acc[i] = y[i] + h * acc[i]
            self._fun(self.t + self.C[s] * h, acc, &self.K[s, 0])

        for i in range(n):
            acc[i] = 0.0
        for j in range(ns):
            a = self.B[j]
            if a != 0.0:
                Kj = &self.K[j, 0]
                for i in range(n):
                    acc[i] += a * Kj[i]
        for i in range(n):
            y_new[i] = y[i] + h * acc[i]

        # FSAL: the derivative at the new point is the last row of K.
        self._fun(self.t + h, y_new, &self.K[ns, 0])
        return 0

    cdef double _error_norm(self, double h):
        """RMS norm of the embedded error estimate ``h * K^T E`` scaled by ``scale``."""
        cdef Py_ssize_t i, j, n = self.n, nk = self.n_stages + 1
        cdef double err, acc = 0.0
        for i in range(n):
            err = 0.0
            for j in range(nk):
                err += self.K[j, i] * self.E[j]
            err = h * err / self.scale[i]
            acc += err * err
        return sqrt(acc / n)

    cdef int _step_impl(self) except -1:
        """One accepted step with rejection/retry. 1 = ok, 0 = step too small."""
        cdef Py_ssize_t i, n = self.n
        cdef double t = self.t
        cdef double min_step = 10 * fabs(nextafter(t, self.direction * INFINITY) - t)
        cdef double h_abs, h, t_new, error_norm, factor, yi, yn
        cdef bint step_rejected = False

        if self.h_abs > self.max_step:
            h_abs = self.max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        while True:
            if h_abs < min_step:
                return 0

            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
            h = t_new - t
            h_abs = fabs(h)

            self._rk_stages(h)

            for i in range(n):
                yi = fabs(self.y[i])
                yn = fabs(self.y_new[i])
                self.scale[i] = self.atol[i] + (yi if yi > yn else yn) * self.rtol
            error_norm = self._error_norm(h)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * pow(error_norm, self.error_exponent))
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                break
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * pow(error_norm, self.error_exponent))
                step_rejected = True
                self.n_rejected += 1

        # Accept: swap y <-> y_new buffers, take f from the FSAL row.
        self.h_previous = h
        self.y, self.y_new = self.y_new, self.y
        memcpy(&self.f[0], &self.K[self.n_stages, 0], n * sizeof(double))
        self.t = t_new
        self.h_abs = h_abs
        self.n_accepted += 1
        return 1

    cdef int _step(self) except -1:
        """scipy's OdeSolver.step(): handles the trivial cases and status updates."""
        if self.n == 0 or self.t == self.t_bound:
            self.t = self.t_bound
            self._status = STATUS_FINISHED
            return 0
        if not self._step_impl():
            self._status = STATUS_FAILED
            self.message = MSG_TOO_SMALL_STEP
        elif self.direction * (self.t - self.t_bound) >= 0:
            self._status = STATUS_FINISHED
        return 0


cdef class RK23(RungeKutta):
    """Bogacki-Shampine 3(2) pair (``scipy.integrate.RK23``)."""

    def _init_tableau(self):
        self.order = tableaus.RK23_ORDER
        self.error_estimator_order = tableaus.RK23_ERROR_ESTIMATOR_ORDER
        self.n_stages = tableaus.RK23_N_STAGES
        self.A = np.ascontiguousarray(tableaus.RK23_A, dtype=np.float64)
        self.B = np.ascontiguousarray(tableaus.RK23_B, dtype=np.float64)
        self.C = np.ascontiguousarray(tableaus.RK23_C, dtype=np.float64)
        self.E = np.ascontiguousarray(tableaus.RK23_E, dtype=np.float64)


cdef class RK45(RungeKutta):
    """Dormand-Prince 5(4) pair (``scipy.integrate.RK45``)."""

    def _init_tableau(self):
        self.order = tableaus.RK45_ORDER
        self.error_estimator_order = tableaus.RK45_ERROR_ESTIMATOR_ORDER
        self.n_stages = tableaus.RK45_N_STAGES
        self.A = np.ascontiguousarray(tableaus.RK45_A, dtype=np.float64)
        self.B = np.ascontiguousarray(tableaus.RK45_B, dtype=np.float64)
        self.C = np.ascontiguousarray(tableaus.RK45_C, dtype=np.float64)
        self.E = np.ascontiguousarray(tableaus.RK45_E, dtype=np.float64)


cdef class DOP853(RungeKutta):
    """Hairer-Wanner 8(5,3) method (``scipy.integrate.DOP853``).

    Uses the blended 5th/3rd-order error estimate. The four extra stages that
    exist only for dense output are not evaluated yet.
    """

    cdef const double[::1] E3
    cdef const double[::1] E5

    def _init_tableau(self):
        cdef Py_ssize_t ns = tableaus.DOP853_N_STAGES
        self.order = tableaus.DOP853_ORDER
        self.error_estimator_order = tableaus.DOP853_ERROR_ESTIMATOR_ORDER
        self.n_stages = ns
        self.A = np.ascontiguousarray(tableaus.DOP853_A[:ns, :ns], dtype=np.float64)
        self.B = np.ascontiguousarray(tableaus.DOP853_B, dtype=np.float64)
        self.C = np.ascontiguousarray(tableaus.DOP853_C[:ns], dtype=np.float64)
        self.E5 = np.ascontiguousarray(tableaus.DOP853_E5, dtype=np.float64)
        self.E = self.E5
        self.E3 = np.ascontiguousarray(tableaus.DOP853_E3, dtype=np.float64)

    cdef double _error_norm(self, double h):
        cdef Py_ssize_t i, j, n = self.n, nk = self.n_stages + 1
        cdef double e5, e3, k, inv_scale
        cdef double err5_norm_2 = 0.0, err3_norm_2 = 0.0, denom
        for i in range(n):
            e5 = 0.0
            e3 = 0.0
            for j in range(nk):
                k = self.K[j, i]
                e5 += k * self.E5[j]
                e3 += k * self.E3[j]
            inv_scale = 1.0 / self.scale[i]
            e5 *= inv_scale
            e3 *= inv_scale
            err5_norm_2 += e5 * e5
            err3_norm_2 += e3 * e3
        if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return fabs(h) * err5_norm_2 / sqrt(denom * n)


METHODS = {"RK23": RK23, "RK45": RK45, "DOP853": DOP853}


# =============================================================================
# Driver
# =============================================================================

class OdeResult:
    """Integration result. Its attributes match scipy's ``OdeResult``."""

    def __init__(self, t, y, nfev, njev, nlu, status, message, success,
                 n_accepted, n_rejected):
        self.t = t
        self.y = y
        self.nfev = nfev
        self.njev = njev
        self.nlu = nlu
        self.status = status
        self.message = message
        self.success = success
        self.n_accepted = n_accepted
        self.n_rejected = n_rejected

    def __repr__(self):
        return (f"OdeResult(success={self.success}, status={self.status}, "
                f"nfev={self.nfev}, n_steps={self.t.shape[0] - 1}, "
                f"message={self.message!r})")


def solve_ivp(fun, t_span, y0, method="DOP853", *, double rtol=1e-3, atol=1e-6,
              first_step=None, double max_step=INFINITY, args=None):
    """Solve an initial value problem. A subset of :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    fun : OdeSystem or callable
        Right-hand side. Pass a native :class:`OdeSystem` for full speed. A
        Python callable ``fun(t, y, *args)`` also works, but every evaluation
        goes back through Python.
    t_span : (float, float)
        Interval of integration ``(t0, tf)``; ``tf < t0`` integrates backwards.
    y0 : array_like, shape (n,)
        Initial state.
    method : {'RK23', 'RK45', 'DOP853'}
    rtol, atol, first_step, max_step
        As in :func:`scipy.integrate.solve_ivp`.
    args : tuple, optional
        Extra arguments for a Python callable ``fun``.

    Returns
    -------
    OdeResult
        ``t`` (n_points,) holds every accepted step including ``t0``, and ``y``
        has shape (n, n_points). This matches ``solve_ivp`` called without
        ``t_eval``.
    """
    cdef double t0, tf
    cdef RungeKutta solver
    cdef Py_ssize_t count, cap, n
    cdef double[::1] ts_view
    cdef double[:, ::1] ys_view

    try:
        solver_cls = METHODS[method]
    except KeyError:
        raise ValueError(f"`method` must be one of {sorted(METHODS)}.") from None

    t0, tf = map(float, t_span)

    if isinstance(fun, OdeSystem):
        if args is not None:
            raise TypeError("`args` is only supported for Python callables")
        system = fun
    else:
        system = PythonOdeSystem(fun, () if args is None else args)

    solver = solver_cls(system, t0, y0, tf, rtol=rtol, atol=atol,
                        first_step=first_step, max_step=max_step)
    n = solver.n

    # Amortised-doubling output buffers (rows = accepted steps).
    cap = 256
    ts = _empty1d(cap)
    ys = _empty2d(cap, n)
    ts_view = ts
    ys_view = ys
    ts_view[0] = solver.t
    ys_view[0, :] = solver.y
    count = 1

    while solver._status == STATUS_RUNNING:
        solver._step()
        if solver._status == STATUS_FAILED:
            break
        if count == cap:
            cap *= 2
            new_ts = _empty1d(cap)
            new_ys = _empty2d(cap, n)
            new_ts[:count] = ts[:count]
            new_ys[:count] = ys[:count]
            ts, ys = new_ts, new_ys
            ts_view = ts
            ys_view = ys
        ts_view[count] = solver.t
        ys_view[count, :] = solver.y
        count += 1

    status = 0 if solver._status == STATUS_FINISHED else -1
    message = MSG_SUCCESS if status == 0 else solver.message
    return OdeResult(
        t=ts[:count].copy(),
        y=ys[:count].T.copy(),
        nfev=solver.nfev,
        njev=0,
        nlu=0,
        status=status,
        message=message,
        success=status >= 0,
        n_accepted=solver.n_accepted,
        n_rejected=solver.n_rejected,
    )
