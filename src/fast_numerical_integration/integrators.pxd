# cython: language_level=3
#
# Public C-level interface of the Cython integration engine.
#
# Other extension modules (hand-written systems today, SymPy-generated C kernels
# later) ``cimport`` this file and subclass :class:`OdeSystem` (right-hand sides)
# or :class:`EventFunction` (event functions), overriding ``eval``. The engine
# calls ``eval`` through the vtable, so native systems and events never enter the
# Python interpreter inside the step loop.


cdef class OdeSystem:
    # Number of state variables this system expects, or -1 if any size is valid
    # (e.g. a wrapped Python callable that adapts to ``y0``).
    cdef readonly Py_ssize_t n_states

    # Write dy/dt = f(t, y) into ``dydt``. Both buffers hold ``n`` contiguous
    # doubles. Return 0 on success; return -1 with a Python exception set on
    # failure (the integrator propagates it).
    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1


cdef class EventFunction:
    # Same meaning as the ``terminal`` / ``direction`` attributes scipy reads
    # from event callables: ``terminal`` is False/True or a positive int (number
    # of occurrences that terminate the integration); ``direction`` selects
    # zero crossings by sign of slope (0 = both).
    cdef public object terminal
    cdef public double direction

    # Return g(t, y); events are located at sign changes of g.
    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1


cdef class RungeKutta:
    # --- problem ---
    cdef OdeSystem system
    cdef readonly Py_ssize_t n
    cdef readonly double t
    cdef readonly double t_bound
    cdef readonly double direction
    cdef double max_step
    cdef double _t_old
    cdef bint _has_t_old

    # --- method (Butcher tableau, read-only views onto module constants) ---
    cdef readonly Py_ssize_t n_stages
    cdef readonly int order
    cdef readonly int error_estimator_order
    cdef double error_exponent
    cdef const double[:, ::1] A
    cdef const double[::1] B
    cdef const double[::1] C
    cdef const double[::1] E
    cdef const double[:, ::1] P   # RK23/RK45 dense-output matrix
    cdef Py_ssize_t n_k_rows      # rows of K (DOP853 keeps 3 extra dense stages)

    # --- dense output for the last accepted step ---
    cdef int dense_kind
    cdef Py_ssize_t n_dense_coef
    cdef double[:, ::1] dense_coef  # (n_dense_coef, n)
    cdef bint dense_valid

    # --- working storage (allocated once, reused every step) ---
    cdef double[:, ::1] K       # (n_k_rows, n): stages + FSAL derivative (+ extras)
    cdef double[::1] _y         # current state
    cdef double[::1] _y_old     # state at the start of the last step
    cdef double[::1] _f         # f(t, y)
    cdef double[::1] _y_new     # candidate state at t + h
    cdef double[::1] _y_stage   # scratch for stage arguments
    cdef double[::1] _scale     # atol + rtol * max(|y|, |y_new|)
    cdef double[::1] _atol      # per-component absolute tolerance
    cdef double[::1] _rtol      # per-component relative tolerance

    # --- adaptive step state / statistics ---
    cdef readonly double h_abs
    cdef double h_previous
    cdef readonly Py_ssize_t nfev
    cdef readonly Py_ssize_t n_accepted
    cdef readonly Py_ssize_t n_rejected
    cdef int _status            # STATUS_RUNNING | STATUS_FINISHED | STATUS_FAILED
    cdef readonly str message   # failure reason, or None

    cdef int _fun(self, double t, const double* y, double* dydt) except -1
    cdef double _select_initial_step(self) except -1.0
    cdef int _rk_stages(self, double h) except -1
    cdef double _error_norm(self, double h)
    cdef int _step_impl(self) except -1
    cdef int _step(self) except -1
    cdef int _build_dense(self) except -1
    cdef int _ensure_dense(self) except -1
    cdef void _dense_eval(self, double t, double* out) noexcept
