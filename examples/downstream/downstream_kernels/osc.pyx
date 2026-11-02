# cython: language_level=3
"""Example kernel: a native OdeSystem plus a direct nogil call to fni_integrate."""
from fast_numerical_integration.integrators cimport (
    OdeSystem, fni_integrate, fni_options, fni_stats, FNI_DOP853, FNI_OK)

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
    opt.rtol = 1e-10; opt.rtol_vec = NULL; opt.atol = 1e-12; opt.atol_vec = NULL
    opt.first_step = -1.0; opt.max_step = 1e300
    with nogil:
        rc = fni_integrate(FNI_DOP853, _osc, &p, 2, 0.0, y0, t1, &opt, NULL, 0, y1, NULL, NULL, NULL, 0, &st)
    assert rc == FNI_OK
    return y1[0], y1[1], st.nfev
