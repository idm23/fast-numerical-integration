# cython: language_level=3
"""Example kernel: a native OdeSystem, a direct nogil call to fni_integrate, and the
decoupled tangent propagator through fni_integrate_hybrid_tangent."""
from fast_numerical_integration.integrators cimport (
    OdeSystem, fni_hybrid_options, fni_integrate, fni_integrate_hybrid_tangent, fni_options,
    fni_options_init, fni_stats, fni_tangent, fni_tangent_init, FNI_DOP853, FNI_OK)

cdef int _osc(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx
    dydt[0] = y[1]
    dydt[1] = -p[0] * y[0]
    return 0

cdef class Oscillator(OdeSystem):
    def __init__(self, double w2):
        self.n_states = 2
        self._init_params([w2], ("w2",))
        self.c_fn = _osc
        self.c_ctx = self._params_ptr()

def endpoint(double w2, double t1):
    cdef double y0[2]
    cdef double y1[2]
    cdef double p = w2
    cdef fni_options opt
    cdef fni_stats st
    cdef int rc
    y0[0] = 1.0; y0[1] = 0.0
    fni_options_init(&opt)     # defaults for every field, then override
    opt.rtol = 1e-10
    opt.atol = 1e-12
    with nogil:
        rc = fni_integrate(FNI_DOP853, _osc, &p, 2, 0.0, y0, t1, &opt, NULL, 0, y1, NULL, NULL, NULL, 0, &st)
    assert rc == FNI_OK
    return y1[0], y1[1], st.nfev


cdef int _osc_jac(double t, const double* y, Py_ssize_t n, void* ctx, double* J) noexcept nogil:
    cdef const double* p = <const double*> ctx
    J[0] = 0.0
    J[1] = 1.0
    J[2] = -p[0]
    J[3] = 0.0
    return 0

def endpoint_tangent(double w2, double t1):
    """The endpoint and its state transition matrix, X advanced alongside y."""
    cdef double y0[2]
    cdef double y1[2]
    cdef double X[4]
    cdef double p = w2
    cdef fni_options opt
    cdef fni_hybrid_options hopt
    cdef fni_tangent tg
    cdef fni_stats st
    cdef int rc
    y0[0] = 1.0; y0[1] = 0.0
    X[0] = 1.0; X[1] = 0.0; X[2] = 0.0; X[3] = 1.0
    fni_options_init(&opt)
    opt.rtol = 1e-10
    opt.atol = 1e-12
    hopt.max_resets = -1
    hopt.min_dwell = 0.0
    hopt.reselect_step = False
    fni_tangent_init(&tg)
    tg.jac = _osc_jac
    tg.ctx = &p
    tg.m = 2
    tg.X = X
    with nogil:
        rc = fni_integrate_hybrid_tangent(FNI_DOP853, _osc, &p, 2, 0.0, y0, t1, &opt, NULL, 0,
                                          NULL, NULL, &hopt, y1, NULL, NULL, NULL, NULL, NULL,
                                          0, &st, NULL, &tg)
    assert rc == FNI_OK
    return (y1[0], y1[1]), ((X[0], X[1]), (X[2], X[3])), tg.n_jac
