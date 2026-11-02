# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython explicit Runge-Kutta integrators: RK23, RK45 and DOP853.

A step-for-step reimplementation of :func:`scipy.integrate.solve_ivp` with the
``RK23``, ``RK45`` and ``DOP853`` methods. It uses the same tableaus, the same
initial-step heuristic, the same error norms and the same step-size controller
(safety 0.9, factors in [0.2, 10]), so it takes the same steps and makes the
same number of RHS evaluations as SciPy. Differences are limited to
floating-point summation order.

Supported ``solve_ivp`` features: ``t_eval``, ``dense_output`` (returns an
:class:`OdeSolution`), ``events`` (terminal as a bool or count, plus direction;
roots found by Brent's method on the dense interpolant), ``vectorized``,
``args``, ``first_step``, ``max_step``, and scalar or per-component
``rtol``/``atol``. The stepper classes mirror ``scipy.integrate.OdeSolver``
(``step()``, ``dense_output()``, ``t_old``, ``step_size``, ...).

Not supported: complex-valued ``y0``.

Every state-sized operation is a typed-memoryview loop over arrays that cnumpy
allocates once per solve. The step loop makes no NumPy calls and creates no
temporary arrays. Native right-hand sides (:class:`OdeSystem`) and native events
(:class:`EventFunction`) are called through the vtable, so a fully native
problem runs without entering the Python interpreter except to record results.
"""

cimport numpy as cnp
from libc.math cimport fabs, sqrt, pow, nextafter, INFINITY
from libc.string cimport memcpy

import warnings

import numpy as np

from fast_numerical_integration import tableaus

cnp.import_array()

# Step-size controller constants (identical to scipy.integrate._ivp.rk).
cdef double SAFETY = 0.9
cdef double MIN_FACTOR = 0.2
cdef double MAX_FACTOR = 10.0

cdef double EPS = np.finfo(np.float64).eps
cdef int BRENTQ_MAXITER = 100

cdef enum:
    STATUS_RUNNING = 0
    STATUS_FINISHED = 1
    STATUS_FAILED = -1

cdef enum:
    DENSE_RK = 0       # y_old + h * sum_k Q[k] x^(k+1)
    DENSE_DOP853 = 1   # nested Hairer-Wanner form over F[0..6]

_STATUS_NAMES = {STATUS_RUNNING: "running", STATUS_FINISHED: "finished",
                 STATUS_FAILED: "failed"}

MESSAGES = {
    0: "The solver successfully reached the end of the integration interval.",
    1: "A termination event occurred.",
}
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


cdef inline cnp.ndarray _array_from(const double* src, Py_ssize_t n):
    """Fresh 1-D ndarray holding a copy of ``n`` doubles."""
    out = _empty1d(n)
    if n > 0:
        memcpy(cnp.PyArray_DATA(out), src, n * sizeof(double))
    return out


cdef inline double _rms_norm_scaled(const double* x, const double* scale, Py_ssize_t n) noexcept:
    """RMS norm of ``x / scale`` (scipy's ``norm(x / scale)``)."""
    cdef Py_ssize_t i
    cdef double v, acc = 0.0
    for i in range(n):
        v = x[i] / scale[i]
        acc += v * v
    return sqrt(acc / n)


cdef void _interp_eval(int kind, Py_ssize_t ncoef, Py_ssize_t n, double t_old,
                       double h, const double* y_old, const double* coef,
                       double t, double* out) noexcept:
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
    if kind == DENSE_RK:
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
    """Base class for native event functions ``g(t, y)``.

    Subclass in Cython and override ``eval``. ``terminal`` and ``direction``
    behave like the attributes of scipy event callables.
    """

    def __init__(self, terminal=False, double direction=0.0):
        self.terminal = terminal
        self.direction = direction

    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1:
        raise NotImplementedError(f"{type(self).__name__} must implement eval()")

    def __call__(self, double t, y):
        cdef const double[::1] y_view = np.ascontiguousarray(y, dtype=np.float64)
        cdef Py_ssize_t n = y_view.shape[0]
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


cdef class ComponentCrossing(EventFunction):
    """Native event ``g = y[index] - value``: a state component crosses a level.

    Example: ``ComponentCrossing(2, 0.0, direction=-1)`` fires when ``z``
    passes downward through zero.
    """

    cdef readonly Py_ssize_t index
    cdef readonly double value

    def __init__(self, Py_ssize_t index, double value=0.0, terminal=False,
                 double direction=0.0):
        super().__init__(terminal, direction)
        if index < 0:
            raise ValueError("index must be non-negative")
        self.index = index
        self.value = value

    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1:
        if self.index >= n:
            raise IndexError(f"event index {self.index} out of range for state of size {n}")
        return y[self.index] - self.value


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
    native :class:`OdeSystem` or a Python callable ``fun(t, y)``.
    """

    def __init__(self, fun, double t0, y0, double t_bound, double max_step=INFINITY,
                 rtol=1e-3, atol=1e-6, vectorized=False, first_step=None,
                 **extraneous):
        _warn_extraneous(extraneous)
        self._init_tableau()

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
        self.n = y0_arr.shape[0]
        if system.n_states >= 0 and system.n_states != self.n:
            raise ValueError(
                f"system expects {system.n_states} states, y0 has {self.n}")

        if max_step <= 0:
            raise ValueError("`max_step` must be positive.")
        rtol_arr = np.asarray(rtol, dtype=np.float64)
        if rtol_arr.ndim > 0 and rtol_arr.shape != (self.n,):
            raise ValueError("`rtol` has wrong shape.")
        if np.any(rtol_arr < 100 * EPS):
            warnings.warn(
                "At least one element of `rtol` is too small. "
                f"Setting `rtol = np.maximum(rtol, {100 * EPS})`.",
                stacklevel=2,
            )
            rtol_arr = np.maximum(rtol_arr, 100 * EPS)
        atol_arr = np.asarray(atol, dtype=np.float64)
        if atol_arr.ndim > 0 and atol_arr.shape != (self.n,):
            raise ValueError("`atol` has wrong shape.")
        if np.any(atol_arr < 0):
            raise ValueError("`atol` must be positive.")

        self.system = system
        self.t = t0
        self.t_bound = t_bound
        self.direction = 1.0 if t_bound >= t0 else -1.0
        self.max_step = max_step
        self._has_t_old = False
        self._t_old = np.nan

        # cnumpy-allocated working storage, viewed through typed memoryviews.
        self._y = y0_arr
        self._y_old = _empty1d(self.n)
        self._f = _empty1d(self.n)
        self._y_new = _empty1d(self.n)
        self._y_stage = _empty1d(self.n)
        self._scale = _empty1d(self.n)
        self._atol = np.broadcast_to(atol_arr, (self.n,)).copy()
        self._rtol = np.broadcast_to(rtol_arr, (self.n,)).copy()
        self.K = _empty2d(self.n_k_rows, self.n)
        self.dense_coef = _empty2d(self.n_dense_coef, self.n)
        self.dense_valid = False

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

        self._fun(self.t, &self._y[0], &self._f[0])
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

    # --- public Python API (mirrors scipy.integrate.OdeSolver) ----------------

    @property
    def status(self):
        return _STATUS_NAMES[self._status]

    @property
    def y(self):
        """Copy of the current state."""
        return np.asarray(self._y).copy()

    @property
    def t_old(self):
        """Time at the start of the last step, or None before the first step."""
        return self._t_old if self._has_t_old else None

    @property
    def step_size(self):
        """Size of the last successful step, or None before the first step."""
        return fabs(self.t - self._t_old) if self._has_t_old else None

    @property
    def njev(self):
        return 0

    @property
    def nlu(self):
        return 0

    def step(self):
        """Advance one accepted step. Returns the failure message, or None."""
        if self._status != STATUS_RUNNING:
            raise RuntimeError("Attempt to step on a failed or finished solver.")
        self._step()
        return self.message

    def dense_output(self):
        """Interpolant over the last successful step, as a one-segment :class:`OdeSolution`."""
        if not self._has_t_old:
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")
        self._ensure_dense()
        row = np.concatenate((
            [self._t_old, self.t - self._t_old],
            np.asarray(self._y_old),
            np.asarray(self.dense_coef).reshape(-1),
        ))
        return OdeSolution([self._t_old, self.t], row[None, :], self.dense_kind,
                           self.n, self.n_dense_coef)

    # --- C-level core --------------------------------------------------------

    cdef int _fun(self, double t, const double* y, double* dydt) except -1:
        self.nfev += 1
        return self.system.eval(t, y, dydt, self.n)

    cdef double _select_initial_step(self) except -1.0:
        """Hairer-Norsett-Wanner initial step heuristic (scipy's select_initial_step)."""
        cdef Py_ssize_t i, n = self.n
        cdef double interval_length = fabs(self.t_bound - self.t)
        cdef double d0, d1, d2, h0, h1
        cdef double* y0 = &self._y[0]
        cdef double* f0 = &self._f[0]
        cdef double* scale = &self._scale[0]
        cdef double* y1 = &self._y_new[0]
        cdef double* f1 = &self.K[0, 0]    # free scratch before the first step
        cdef double* diff = &self._y_stage[0]

        if interval_length == 0.0:
            return 0.0

        for i in range(n):
            scale[i] = self._atol[i] + fabs(y0[i]) * self._rtol[i]
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
        cdef double* y = &self._y[0]
        cdef double* acc = &self._y_stage[0]
        cdef double* y_new = &self._y_new[0]
        cdef double* Kj

        memcpy(&self.K[0, 0], &self._f[0], n * sizeof(double))

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

        # FSAL: the derivative at the new point is row n_stages of K.
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
            err = h * err / self._scale[i]
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
                yi = fabs(self._y[i])
                yn = fabs(self._y_new[i])
                self._scale[i] = self._atol[i] + (yi if yi > yn else yn) * self._rtol[i]
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

        # Accept: keep y_old for dense output, swap y <-> y_new, take f from FSAL.
        self.h_previous = h
        memcpy(&self._y_old[0], &self._y[0], n * sizeof(double))
        self._y, self._y_new = self._y_new, self._y
        memcpy(&self._f[0], &self.K[self.n_stages, 0], n * sizeof(double))
        self.t = t_new
        self.h_abs = h_abs
        self.n_accepted += 1
        return 1

    cdef int _step(self) except -1:
        """scipy's OdeSolver.step(): handles the trivial cases and status updates."""
        cdef double t = self.t
        self.dense_valid = False
        if self.n == 0 or self.t == self.t_bound:
            self._t_old = t
            self._has_t_old = True
            if self.n > 0:
                memcpy(&self._y_old[0], &self._y[0], self.n * sizeof(double))
            self.t = self.t_bound
            self._status = STATUS_FINISHED
            return 0
        if not self._step_impl():
            self._status = STATUS_FAILED
            self.message = MSG_TOO_SMALL_STEP
        else:
            self._t_old = t
            self._has_t_old = True
            if self.direction * (self.t - self.t_bound) >= 0:
                self._status = STATUS_FINISHED
        return 0

    cdef int _build_dense(self) except -1:
        """RK23/RK45: Q = K^T P, stored transposed as (n_coef, n)."""
        cdef Py_ssize_t k, j, i, n = self.n, nk = self.n_stages + 1
        cdef double p
        cdef double* row
        cdef double* Kj
        for k in range(self.n_dense_coef):
            row = &self.dense_coef[k, 0]
            for i in range(n):
                row[i] = 0.0
            for j in range(nk):
                p = self.P[j, k]
                if p != 0.0:
                    Kj = &self.K[j, 0]
                    for i in range(n):
                        row[i] += p * Kj[i]
        return 0

    cdef int _ensure_dense(self) except -1:
        """Build the last step's interpolant once (DOP853 costs 3 RHS calls)."""
        if not self.dense_valid:
            if self.n > 0 and self.t != self._t_old:
                self._build_dense()
            self.dense_valid = True
        return 0

    cdef void _dense_eval(self, double t, double* out) noexcept:
        _interp_eval(self.dense_kind, self.n_dense_coef, self.n, self._t_old,
                     self.t - self._t_old, &self._y_old[0], &self.dense_coef[0, 0],
                     t, out)


cdef class RK23(RungeKutta):
    """Bogacki-Shampine 3(2) pair (``scipy.integrate.RK23``)."""

    def _init_tableau(self):
        self.order = tableaus.RK23_ORDER
        self.error_estimator_order = tableaus.RK23_ERROR_ESTIMATOR_ORDER
        self.n_stages = tableaus.RK23_N_STAGES
        self.n_k_rows = self.n_stages + 1
        self.A = np.ascontiguousarray(tableaus.RK23_A, dtype=np.float64)
        self.B = np.ascontiguousarray(tableaus.RK23_B, dtype=np.float64)
        self.C = np.ascontiguousarray(tableaus.RK23_C, dtype=np.float64)
        self.E = np.ascontiguousarray(tableaus.RK23_E, dtype=np.float64)
        self.P = np.ascontiguousarray(tableaus.RK23_P, dtype=np.float64)
        self.dense_kind = DENSE_RK
        self.n_dense_coef = self.P.shape[1]


cdef class RK45(RungeKutta):
    """Dormand-Prince 5(4) pair (``scipy.integrate.RK45``)."""

    def _init_tableau(self):
        self.order = tableaus.RK45_ORDER
        self.error_estimator_order = tableaus.RK45_ERROR_ESTIMATOR_ORDER
        self.n_stages = tableaus.RK45_N_STAGES
        self.n_k_rows = self.n_stages + 1
        self.A = np.ascontiguousarray(tableaus.RK45_A, dtype=np.float64)
        self.B = np.ascontiguousarray(tableaus.RK45_B, dtype=np.float64)
        self.C = np.ascontiguousarray(tableaus.RK45_C, dtype=np.float64)
        self.E = np.ascontiguousarray(tableaus.RK45_E, dtype=np.float64)
        self.P = np.ascontiguousarray(tableaus.RK45_P, dtype=np.float64)
        self.dense_kind = DENSE_RK
        self.n_dense_coef = self.P.shape[1]


cdef class DOP853(RungeKutta):
    """Hairer-Wanner 8(5,3) method (``scipy.integrate.DOP853``).

    Uses the blended 5th/3rd-order error estimate. Dense output evaluates the
    three extra stages of the 7th-order continuous extension on demand.
    """

    cdef const double[::1] E3
    cdef const double[::1] E5
    cdef const double[:, ::1] A_ext   # full 16x16 tableau (extra stages included)
    cdef const double[::1] C_ext
    cdef const double[:, ::1] D

    def _init_tableau(self):
        cdef Py_ssize_t ns = tableaus.DOP853_N_STAGES
        self.order = tableaus.DOP853_ORDER
        self.error_estimator_order = tableaus.DOP853_ERROR_ESTIMATOR_ORDER
        self.n_stages = ns
        self.n_k_rows = tableaus.DOP853_N_STAGES_EXTENDED
        self.A = np.ascontiguousarray(tableaus.DOP853_A[:ns, :ns], dtype=np.float64)
        self.B = np.ascontiguousarray(tableaus.DOP853_B, dtype=np.float64)
        self.C = np.ascontiguousarray(tableaus.DOP853_C[:ns], dtype=np.float64)
        self.E5 = np.ascontiguousarray(tableaus.DOP853_E5, dtype=np.float64)
        self.E = self.E5
        self.E3 = np.ascontiguousarray(tableaus.DOP853_E3, dtype=np.float64)
        self.A_ext = np.ascontiguousarray(tableaus.DOP853_A, dtype=np.float64)
        self.C_ext = np.ascontiguousarray(tableaus.DOP853_C, dtype=np.float64)
        self.D = np.ascontiguousarray(tableaus.DOP853_D, dtype=np.float64)
        self.dense_kind = DENSE_DOP853
        self.n_dense_coef = tableaus.DOP853_INTERPOLATOR_POWER

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
            inv_scale = 1.0 / self._scale[i]
            e5 *= inv_scale
            e3 *= inv_scale
            err5_norm_2 += e5 * e5
            err3_norm_2 += e3 * e3
        if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return fabs(h) * err5_norm_2 / sqrt(denom * n)

    cdef int _build_dense(self) except -1:
        """Extra stages 13-15, then F (7, n) exactly as scipy's _dense_output_impl."""
        cdef Py_ssize_t s, j, i, r, n = self.n
        cdef Py_ssize_t n_ext = self.n_k_rows
        cdef double a, d, dy
        cdef double h = self.h_previous
        cdef double t_old = self._t_old
        cdef double* y_old = &self._y_old[0]
        cdef double* y = &self._y[0]
        cdef double* f = &self._f[0]
        cdef double* f_old = &self.K[0, 0]
        cdef double* acc = &self._y_stage[0]
        cdef double* row
        cdef double* Kj

        for s in range(self.n_stages + 1, n_ext):
            for i in range(n):
                acc[i] = 0.0
            for j in range(s):
                a = self.A_ext[s, j]
                if a != 0.0:
                    Kj = &self.K[j, 0]
                    for i in range(n):
                        acc[i] += a * Kj[i]
            for i in range(n):
                acc[i] = y_old[i] + acc[i] * h
            self._fun(t_old + self.C_ext[s] * h, acc, &self.K[s, 0])

        for i in range(n):
            dy = y[i] - y_old[i]
            self.dense_coef[0, i] = dy
            self.dense_coef[1, i] = h * f_old[i] - dy
            self.dense_coef[2, i] = 2 * dy - h * (f[i] + f_old[i])

        for r in range(self.n_dense_coef - 3):
            row = &self.dense_coef[3 + r, 0]
            for i in range(n):
                row[i] = 0.0
            for j in range(n_ext):
                d = self.D[r, j]
                if d != 0.0:
                    Kj = &self.K[j, 0]
                    for i in range(n):
                        row[i] += d * Kj[i]
            for i in range(n):
                row[i] *= h
        return 0


METHODS = {"RK23": RK23, "RK45": RK45, "DOP853": DOP853}


# =============================================================================
# Event handling
# =============================================================================

cdef double _event_on_dense(RungeKutta solver, EventFunction ev, double t,
                            double* buf) except? -1:
    solver._dense_eval(t, buf)
    return ev.eval(t, buf, solver.n)


cdef double _brentq_event(RungeKutta solver, EventFunction ev, double xa,
                          double xb, double* buf) except? -1:
    """Root of ``ev(t, sol(t))`` on [xa, xb]: scipy's C ``brentq`` with
    ``xtol = rtol = 4 * EPS``, ``maxiter = 100``."""
    cdef double xtol = 4 * EPS, rtol = 4 * EPS
    cdef double xpre = xa, xcur = xb, xblk = 0.0
    cdef double fpre, fcur, fblk = 0.0, spre = 0.0, scur = 0.0, sbis
    cdef double delta, stry, dpre, dblk
    cdef int i

    fpre = _event_on_dense(solver, ev, xpre, buf)
    fcur = _event_on_dense(solver, ev, xcur, buf)
    if fpre * fcur > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

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
            return xcur

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
        fcur = _event_on_dense(solver, ev, xcur, buf)

    raise RuntimeError(
        f"Failed to converge after {BRENTQ_MAXITER} iterations, value is {xcur}.")


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
                f"nfev={self.nfev}, n_points={self.t.shape[0]}, "
                f"message={self.message!r})")


cdef object _drive(RungeKutta solver, double t0, object t_eval_obj,
                   Py_ssize_t t_eval_i, bint dense_output, list events,
                   double[::1] max_events, double[::1] event_dir):
    """The solve_ivp main loop (same control flow as scipy's)."""
    cdef Py_ssize_t n = solver.n
    cdef Py_ssize_t ncoef = solver.n_dense_coef
    cdef Py_ssize_t n_events = len(events)
    cdef bint has_t_eval = t_eval_obj is not None
    cdef const double[::1] t_eval
    cdef Py_ssize_t t_eval_i_new, j, k, e, n_active, idx, cut
    cdef int status = 2   # 2 = still running
    cdef double t_old, t, root, tmp_r
    cdef double* y_ptr
    cdef double* seg_row
    cdef bint terminate
    cdef EventFunction ev

    if has_t_eval:
        t_eval = t_eval_obj

    traj = _RowBuffer(n)
    cdef _RowBuffer segs = _RowBuffer(2 + n + ncoef * n) if dense_output else None
    cdef _RowBuffer bounds = _RowBuffer(0) if (dense_output and has_t_eval) else None
    cdef double[::1] buf = _empty1d(max(n, 1))       # interpolation scratch
    cdef double[::1] y_term = _empty1d(max(n, 1))    # state at a terminal event
    cdef double[::1] y0_copy = _array_from(&solver._y[0], n) if n > 0 else _empty1d(1)

    if not has_t_eval:
        traj.push_row(t0, &solver._y[0])
    if bounds is not None:
        bounds.push(t0)

    # Event bookkeeping.
    cdef double[::1] g = _empty1d(n_events)
    cdef double[::1] g_new = _empty1d(n_events)
    cdef double[::1] event_count = np.zeros(n_events)
    cdef double[::1] roots = _empty1d(n_events)
    cdef Py_ssize_t[::1] active = np.empty(n_events, dtype=np.intp)
    t_events = [[] for _ in range(n_events)]
    y_events = [[] for _ in range(n_events)]
    for e in range(n_events):
        ev = <EventFunction> events[e]
        g[e] = ev.eval(t0, &y0_copy[0], n)

    while status == 2:
        solver._step()
        if solver._status == STATUS_FINISHED:
            status = 0
        elif solver._status == STATUS_FAILED:
            status = -1
            break

        t_old = solver._t_old
        t = solver.t
        y_ptr = &solver._y[0]

        if dense_output:
            solver._ensure_dense()
            idx = segs.push(t_old)
            seg_row = &segs.r[idx, 0]
            seg_row[0] = t_old
            seg_row[1] = t - t_old
            if n > 0:
                memcpy(seg_row + 2, &solver._y_old[0], n * sizeof(double))
                memcpy(seg_row + 2 + n, &solver.dense_coef[0, 0], ncoef * n * sizeof(double))

        if n_events > 0:
            n_active = 0
            for e in range(n_events):
                ev = <EventFunction> events[e]
                g_new[e] = ev.eval(t, y_ptr, n)
                # find_active_events
                if ((g[e] <= 0 and g_new[e] >= 0 and event_dir[e] > 0) or
                        (g[e] >= 0 and g_new[e] <= 0 and event_dir[e] < 0) or
                        (((g[e] <= 0 and g_new[e] >= 0) or (g[e] >= 0 and g_new[e] <= 0))
                         and event_dir[e] == 0)):
                    active[n_active] = e
                    n_active += 1

            if n_active > 0:
                solver._ensure_dense()
                terminate = False
                for k in range(n_active):
                    e = active[k]
                    event_count[e] += 1
                    roots[k] = _brentq_event(solver, <EventFunction> events[e],
                                             t_old, t, &buf[0])
                    if event_count[e] >= max_events[e]:
                        terminate = True

                if terminate:
                    # Stable sort (active, roots) in the direction of integration,
                    # then keep events up to and including the first terminal one.
                    for k in range(1, n_active):
                        tmp_r = roots[k]
                        idx = active[k]
                        j = k - 1
                        while j >= 0 and ((roots[j] > tmp_r) if t > t_old else (roots[j] < tmp_r)):
                            roots[j + 1] = roots[j]
                            active[j + 1] = active[j]
                            j -= 1
                        roots[j + 1] = tmp_r
                        active[j + 1] = idx
                    cut = n_active
                    for k in range(n_active):
                        if event_count[active[k]] >= max_events[active[k]]:
                            cut = k + 1
                            break
                    n_active = cut

                for k in range(n_active):
                    e = active[k]
                    root = roots[k]
                    t_events[e].append(root)
                    solver._dense_eval(root, &buf[0])
                    y_events[e].append(_array_from(&buf[0], n))

                if terminate:
                    status = 1
                    t = roots[n_active - 1]
                    solver._dense_eval(t, &y_term[0])
                    y_ptr = &y_term[0]

            g, g_new = g_new, g

        if not has_t_eval:
            if dense_output and traj.count > 1 and traj.t[traj.count - 1] == t:
                segs.count -= 1    # scipy's donot_append: drop the duplicate segment
            else:
                traj.push_row(t, y_ptr)
        else:
            # The value in t_eval equal to t is included.
            if solver.direction > 0:
                t_eval_i_new = _searchsorted(t_eval, t, False)
                if t_eval_i_new > t_eval_i:
                    solver._ensure_dense()
                    for j in range(t_eval_i, t_eval_i_new):
                        idx = traj.push(t_eval[j])
                        if n > 0:
                            solver._dense_eval(t_eval[j], &traj.r[idx, 0])
                    t_eval_i = t_eval_i_new
            else:
                t_eval_i_new = _searchsorted(t_eval, t, True)
                if t_eval_i > t_eval_i_new:
                    solver._ensure_dense()
                    for j in range(t_eval_i - 1, t_eval_i_new - 1, -1):
                        idx = traj.push(t_eval[j])
                        if n > 0:
                            solver._dense_eval(t_eval[j], &traj.r[idx, 0])
                    t_eval_i = t_eval_i_new
            if bounds is not None:
                bounds.push(t)

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
        sol = OdeSolution(ts if not has_t_eval else bounds.times(),
                          segs.rows(), solver.dense_kind, n, ncoef)
    else:
        sol = None

    return OdeResult(
        t=ts, y=ys, sol=sol, t_events=t_events_out, y_events=y_events_out,
        nfev=solver.nfev, njev=0, nlu=0, status=status, message=message,
        success=status >= 0, n_accepted=solver.n_accepted,
        n_rejected=solver.n_rejected,
    )


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
        Extra arguments for Python callables ``fun`` and ``events``.
    **options
        ``first_step``, ``max_step``, ``rtol``, ``atol`` (scalar or per
        component). Unknown options trigger a warning, as in scipy.

    Returns
    -------
    OdeResult
        ``t``, ``y``, ``sol``, ``t_events``, ``y_events``, ``nfev``, ``njev``,
        ``nlu``, ``status`` (-1 failed, 0 finished, 1 terminal event),
        ``message``, ``success``, plus ``n_accepted`` and ``n_rejected``.
    """
    cdef Py_ssize_t t_eval_i = 0

    if isinstance(method, str):
        if method not in METHODS:
            raise ValueError(f"`method` must be one of {list(METHODS)} or a "
                             "RungeKutta subclass.")
        solver_cls = METHODS[method]
    elif isinstance(method, type) and issubclass(method, RungeKutta):
        solver_cls = method
    else:
        raise ValueError(f"`method` must be one of {list(METHODS)} or a "
                         "RungeKutta subclass.")

    t0, tf = map(float, t_span)

    if args is not None:
        try:
            args = tuple(args)
        except TypeError as exp:
            raise TypeError(
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            ) from exp

    if isinstance(fun, OdeSystem):
        if args is not None:
            raise TypeError("`args` cannot be used with a native OdeSystem; "
                            "store parameters on the system instead.")
        system = fun
    else:
        system = PythonOdeSystem(fun, () if args is None else args, vectorized)

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

    event_list, max_events, event_dir = _prepare_events(events, args)

    return _drive(solver, t0, t_eval, t_eval_i, dense_output, event_list,
                  max_events, event_dir)
