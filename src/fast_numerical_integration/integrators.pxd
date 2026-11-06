# cython: language_level=3
#
# Public C-level interface of the Cython integration engine.
#
# Two layers:
#
# 1. A plain-C, GIL-free core (``fni_*``): right-hand sides and event functions
#    are C function pointers plus an opaque context pointer, and the solver is a
#    C struct. ``fni_integrate`` runs a whole integration (with events) without
#    touching Python, so other Cython modules can call it inside ``with nogil``,
#    from threads, or in tight loops such as multiple shooting.
#
# 2. Python-facing extension types built on that core. :class:`OdeSystem` and
#    :class:`EventFunction` are subclassed in Cython (hand-written systems, or
#    generated kernels) and carry a runtime ``params`` vector. A subclass that
#    sets ``c_fn``/``c_ctx`` to a ``nogil`` C function is usable from both layers.
#
# Thread safety: the core keeps all solver state in the ``fni_solver`` struct,
# so concurrent integrations are safe as long as the user's ``c_fn``
# functions are reentrant (no shared scratch buffers) and ``params`` is not
# modified while they run.


# =============================================================================
# Plain-C core
# =============================================================================

# dy/dt = f(t, y; ctx). Write n values into dydt; return 0 on success, nonzero
# on failure (the integration stops with FNI_ERR_CALLBACK).
ctypedef int (*fni_rhs_fn)(double t, const double* y, double* dydt, Py_ssize_t n,
                           void* ctx) noexcept nogil

# Event function g(t, y; ctx): write the value into *g; return 0 on success.
ctypedef int (*fni_event_fn)(double t, const double* y, Py_ssize_t n, void* ctx,
                             double* g) noexcept nogil

# Reset map, called when terminal event ``event`` fires at ``t`` during a
# hybrid integration (fni_integrate_hybrid). ``y`` holds the state just before
# the event; overwrite it in place with the state just after. Return
# FNI_RESET_CONTINUE to keep integrating from there, FNI_RESET_STOP to end the
# integration (status FNI_STATUS_EVENT, final state ``y``), or a negative
# value on error (FNI_ERR_CALLBACK). A discrete mode (e.g. which arc or contact
# state is active) lives in the caller's context, shared with the RHS.
ctypedef int (*fni_reset_fn)(Py_ssize_t event, double t, double* y, Py_ssize_t n,
                             void* ctx) noexcept nogil

cdef enum:
    FNI_RESET_CONTINUE = 0
    FNI_RESET_STOP = 1

cdef enum:
    # methods
    FNI_RK23 = 0
    FNI_RK45 = 1
    FNI_DOP853 = 2

cdef enum:
    # solver status
    FNI_RUNNING = 0
    FNI_FINISHED = 1
    FNI_FAILED = -1          # required step size below floating-point spacing

cdef enum:
    # integration outcome (fni_stats.status, mirrors solve_ivp's status)
    FNI_STATUS_FAILED = -1
    FNI_STATUS_FINISHED = 0
    FNI_STATUS_EVENT = 1     # terminated by a terminal event
    FNI_STATUS_RESET_LIMIT = -2   # hybrid: more than max_resets resets
    FNI_STATUS_ZENO = -3          # hybrid: two resets closer than min_dwell

cdef enum:
    # return codes (0 = success)
    FNI_OK = 0
    FNI_ERR_CALLBACK = -1    # a RHS or event function returned nonzero
    FNI_ERR_ALLOC = -2
    FNI_ERR_ARG = -3         # invalid argument (e.g. max_step <= 0)
    FNI_ERR_ROOT_SIGN = -4   # event root bracket without a sign change
    FNI_ERR_ROOT_CONV = -5   # event root finding did not converge

cdef enum:
    # event detection (fni_options.event_detection)
    FNI_EVENTS_ENDPOINTS = 0  # scipy: sign test at the ends of each step; the
                              # interpolant is built only when a sign changes
    FNI_EVENTS_INTERIOR = 1   # also test at n_interior points inside every step,
                              # on the interpolant (built on every step)

cdef enum:
    FNI_DENSE_RK = 0         # y_old + h * sum_k Q[k] x^(k+1)
    FNI_DENSE_DOP853 = 1     # nested Hairer-Wanner form

cdef struct fni_tableau:
    int method
    int order
    int error_estimator_order
    Py_ssize_t n_stages
    Py_ssize_t n_k_rows      # stages + FSAL row (+ DOP853's 3 dense-only stages)
    int dense_kind
    Py_ssize_t n_dense_coef
    const double* A          # row-major, row stride a_stride (covers all n_k_rows stages)
    Py_ssize_t a_stride
    const double* B          # n_stages
    const double* C          # n_k_rows
    const double* E          # n_stages + 1 (DOP853: E5)
    const double* E3         # DOP853 only, else NULL
    const double* P          # RK23/RK45 dense matrix, (n_stages + 1) x n_dense_coef
    const double* D          # DOP853 dense matrix, (n_dense_coef - 3) x n_k_rows

cdef struct fni_options:
    double rtol              # used when rtol_vec is NULL (clamped to >= 100 eps)
    const double* rtol_vec   # optional, length n
    double atol              # used when atol_vec is NULL
    const double* atol_vec   # optional, length n
    double first_step        # <= 0: choose automatically (scipy's heuristic)
    double max_step          # > 0 (INFINITY for no limit)
    int event_detection      # FNI_EVENTS_*
    Py_ssize_t n_interior    # FNI_EVENTS_INTERIOR: interior test points per step (>= 1)

cdef struct fni_solver:
    const fni_tableau* tab
    fni_rhs_fn fun
    void* ctx
    Py_ssize_t n
    double t
    double t_bound
    double direction
    double max_step
    int event_detection
    Py_ssize_t n_interior
    double t_old
    bint has_t_old
    double h_abs
    double h_previous
    double error_exponent
    int status               # FNI_RUNNING / FNI_FINISHED / FNI_FAILED
    bint dense_valid
    Py_ssize_t nfev
    Py_ssize_t n_accepted
    Py_ssize_t n_rejected
    double* mem              # single allocation backing every buffer below
    double* y
    double* y_old
    double* f
    double* y_new
    double* y_stage
    double* scale
    double* atol
    double* rtol
    double* K                # n_k_rows x n
    double* dense            # n_dense_coef x n

cdef struct fni_event:
    fni_event_fn fn
    void* ctx
    double direction         # >0: increasing only, <0: decreasing only, 0: both
    double max_events        # terminate after this many occurrences (INFINITY: never)

# Scratch and position of the event scan over one step (see fni_events_begin).
cdef struct fni_event_scan:
    Py_ssize_t n_events
    double* mem              # single allocation backing every buffer below
    double* ga               # per event: value at the left end ta of its search
    double* ta
    double* gb               # values at the current test point b
    double* gr               # values at a candidate root
    double* root             # per event: located root not yet reported (NaN: none)
    double* rt               # right end of that root's bracket
    double* rv               # value there
    double* buf              # state scratch, length n
    const double* g_end      # values at the end of the step
    double a                 # scan position: every event before it is reported
    double b                 # current test point
    Py_ssize_t i_point       # index of b among the step's test points (1..n_points)
    Py_ssize_t n_points

cdef struct fni_hybrid_options:
    Py_ssize_t max_resets    # stop with FNI_STATUS_RESET_LIMIT beyond this many (< 0: no limit)
    double min_dwell         # stop with FNI_STATUS_ZENO if a reset follows the previous
                             # one by less than this (0: off)
    bint reselect_step       # after a reset, rerun the initial-step heuristic instead
                             # of keeping the current step size

# Guard bookkeeping of a hybrid integration (see fni_hybrid_init).
cdef struct fni_hybrid_state:
    Py_ssize_t n_resets
    double last_reset_t
    bint* masked             # per event: ignored until it leaves zero after its reset
    double* t_probe          # per event: where its value was probed after its reset
                             # (NaN otherwise); the next root search starts there
    double* t_fired          # per event: when it last reset (NaN before that)
    double* g_pending        # per event: its value before a root it shares with the
                             # event that just fired (NaN otherwise)

cdef struct fni_stats:
    int status               # FNI_STATUS_*
    double t_final
    Py_ssize_t nfev
    Py_ssize_t n_accepted
    Py_ssize_t n_rejected
    Py_ssize_t n_records     # events found (may exceed the record capacity)

cdef const fni_tableau* fni_get_tableau(int method) noexcept nogil
cdef void fni_options_init(fni_options* opt) noexcept nogil
cdef int fni_solver_init(fni_solver* s, int method, fni_rhs_fn fun, void* ctx, Py_ssize_t n,
                         double t0, const double* y0, double t_bound,
                         const fni_options* opt) noexcept nogil
cdef void fni_solver_free(fni_solver* s) noexcept nogil
cdef int fni_solver_step(fni_solver* s) noexcept nogil
cdef int fni_solver_ensure_dense(fni_solver* s) noexcept nogil
cdef void fni_solver_dense_eval(const fni_solver* s, double t, double* out) noexcept nogil
cdef int fni_events_eval(const fni_event* events, Py_ssize_t n_events, double t,
                         const double* y, Py_ssize_t n, double* g) noexcept nogil
cdef int fni_solver_restart(fni_solver* s, double t, const double* y,
                            bint reselect_step) noexcept nogil
cdef int fni_event_scan_init(fni_event_scan* sc, Py_ssize_t n_events,
                             Py_ssize_t n) noexcept nogil
cdef void fni_event_scan_free(fni_event_scan* sc) noexcept nogil
cdef int fni_events_begin(fni_solver* s, const fni_event* events, Py_ssize_t n_events,
                          const double* g, double* g_new, const fni_hybrid_state* h,
                          fni_event_scan* sc) noexcept nogil
cdef int fni_events_next(fni_solver* s, const fni_event* events, Py_ssize_t n_events,
                         double* counts, const fni_hybrid_state* h, fni_event_scan* sc,
                         Py_ssize_t* event, double* root, bint* terminal) noexcept nogil
cdef int fni_hybrid_init(fni_hybrid_state* h, Py_ssize_t n_events) noexcept nogil
cdef void fni_hybrid_free(fni_hybrid_state* h) noexcept nogil
cdef void fni_hybrid_unmask(fni_hybrid_state* h, Py_ssize_t n_events,
                            const double* g_new) noexcept nogil
cdef int fni_hybrid_reset(fni_solver* s, fni_hybrid_state* h, const fni_hybrid_options* hopt,
                          const fni_event* events, Py_ssize_t n_events, double* counts,
                          Py_ssize_t event, double t_event, double* y,
                          fni_reset_fn reset, void* reset_ctx, double* g,
                          int* status) noexcept nogil
cdef int fni_integrate_hybrid(int method, fni_rhs_fn fun, void* ctx, Py_ssize_t n,
                              double t0, const double* y0, double t1, const fni_options* opt,
                              const fni_event* events, Py_ssize_t n_events,
                              fni_reset_fn reset, void* reset_ctx,
                              const fni_hybrid_options* hopt,
                              double* y_out, Py_ssize_t* record_event, double* record_t,
                              double* record_y, double* record_y_after, bint* record_reset,
                              Py_ssize_t max_records,
                              fni_stats* stats, Py_ssize_t* n_resets) noexcept nogil
cdef int fni_integrate(int method, fni_rhs_fn fun, void* ctx, Py_ssize_t n,
                       double t0, const double* y0, double t1, const fni_options* opt,
                       const fni_event* events, Py_ssize_t n_events,
                       double* y_out, Py_ssize_t* record_event, double* record_t,
                       double* record_y, Py_ssize_t max_records,
                       fni_stats* stats) noexcept nogil


# =============================================================================
# Python-facing extension types
# =============================================================================

cdef class OdeSystem:
    # Number of state variables this system expects, or -1 if any size is valid.
    cdef readonly Py_ssize_t n_states
    # Runtime parameters: change them between solves (continuation) without
    # recompiling. Subclasses read self._params[i] (or receive a pointer via
    # c_ctx); the buffer is never reallocated after _init_params.
    cdef readonly Py_ssize_t n_params
    cdef readonly tuple param_names
    cdef double[::1] _params
    # Optional GIL-free implementation. When set, the engine calls it directly
    # (and fni_integrate can run without the GIL); eval() then defaults to it.
    cdef fni_rhs_fn c_fn
    cdef void* c_ctx

    cdef int _init_params(self, values, names=*) except -1
    cdef double* _params_ptr(self) noexcept
    # Validate a state size before integrating; raises on mismatch.
    cdef int check_size(self, Py_ssize_t n) except -1
    # Write dy/dt = f(t, y) into ``dydt``. Return 0, or -1 with an exception set.
    cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1


cdef class EventFunction:
    # ``terminal`` is False/True or a positive int (occurrences that terminate
    # the integration); ``direction`` selects crossings by sign of slope.
    cdef public object terminal
    cdef public double direction
    cdef readonly Py_ssize_t n_params
    cdef readonly tuple param_names
    cdef double[::1] _params
    cdef fni_event_fn c_fn
    cdef void* c_ctx

    cdef int _init_params(self, values, names=*) except -1
    cdef double* _params_ptr(self) noexcept
    cdef int check_size(self, Py_ssize_t n) except -1
    # Return g(t, y); events are located at sign changes of g.
    cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1


cdef class ResetMap:
    # Optional GIL-free implementation (see fni_reset_fn); otherwise override
    # eval. A reset map usually shares its context (and mode) with the system.
    cdef fni_reset_fn c_fn
    cdef void* c_ctx

    cdef int check_size(self, Py_ssize_t n) except -1

    # Overwrite y with the post-event state; return FNI_RESET_CONTINUE or
    # FNI_RESET_STOP, or -1 with an exception set.
    cdef int eval(self, Py_ssize_t event, double t, double* y, Py_ssize_t n) except -1


cdef class RungeKutta:
    cdef fni_solver s
    cdef int method
    cdef OdeSystem system
    cdef readonly str message   # failure reason, or None

    cdef int _step(self) except -1
    cdef int _ensure_dense(self) except -1
