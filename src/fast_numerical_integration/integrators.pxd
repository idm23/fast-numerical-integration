# cython: language_level=3
#
# Public C-level interface of the Cython integration engine.
#
# Other extension modules (hand-written systems today, SymPy-generated C kernels
# later) ``cimport`` this file and subclass :class:`OdeSystem`, overriding
# ``eval``. The integrators call ``eval`` directly through the vtable, so a native
# right-hand side never touches the Python interpreter inside the step loop.


cdef class OdeSystem:
    # Number of state variables this system expects, or -1 if any size is valid
    # (e.g. a wrapped Python callable that adapts to ``y0``).
    cdef readonly Py_ssize_t n_states

    # Write dy/dt = f(t, y) into ``dydt``. Both buffers hold ``n`` contiguous
    # doubles. Return 0 on success; return -1 with a Python exception set on
    # failure (the integrator propagates it).
    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1


cdef class RungeKutta:
    # --- problem ---
    cdef OdeSystem system
    cdef readonly Py_ssize_t n
    cdef readonly double t
    cdef readonly double t_bound
    cdef readonly double direction
    cdef double rtol
    cdef double max_step

    # --- method (Butcher tableau, read-only views onto module constants) ---
    cdef readonly Py_ssize_t n_stages
    cdef readonly int order
    cdef readonly int error_estimator_order
    cdef double error_exponent
    cdef const double[:, ::1] A
    cdef const double[::1] B
    cdef const double[::1] C
    cdef const double[::1] E

    # --- working storage (allocated once, reused every step) ---
    cdef double[:, ::1] K       # (n_stages + 1, n): stages + FSAL derivative
    cdef double[::1] y          # current state
    cdef double[::1] f          # f(t, y)
    cdef double[::1] y_new      # candidate state at t + h
    cdef double[::1] y_stage    # scratch for stage arguments
    cdef double[::1] scale      # atol + rtol * max(|y|, |y_new|)
    cdef double[::1] atol       # per-component absolute tolerance

    # --- adaptive step state / statistics ---
    cdef readonly double h_abs
    cdef readonly double h_previous
    cdef readonly Py_ssize_t nfev
    cdef readonly Py_ssize_t n_accepted
    cdef readonly Py_ssize_t n_rejected
    cdef int _status            # STATUS_RUNNING | STATUS_FINISHED | STATUS_FAILED
    cdef readonly str message   # failure reason, or None

    cdef int _step(self) except -1

    cdef int _fun(self, double t, const double* y, double* dydt) except -1
    cdef double _select_initial_step(self) except -1.0
    cdef int _rk_stages(self, double h) except -1
    cdef double _error_norm(self, double h)
    cdef int _step_impl(self) except -1
