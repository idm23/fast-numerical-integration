# fast-numerical-integration

A bridge for **fast numerical integration** of differential equations in Python.

The core idea: author a system of ODEs in a computer algebra system (SymPy or
similar), **code-print that system to C**, compile it and **bridge it into Python
with Cython**, then integrate it with hand-written numerical integrators
(DOP853, Runge-Kutta families, ...). The goal is to be **considerably faster than
`scipy.integrate.solve_ivp`** while producing reusable, composable tooling that
other projects can drop in.

## Why

`solve_ivp` is flexible and battle-tested, but every right-hand-side evaluation
crosses back into Python, which dominates the cost for stiff or long-horizon
problems. By generating a compiled C RHS and driving it from a Cython integration
loop, we keep the hot path entirely in native code — no per-step Python overhead —
while still authoring the physics symbolically.

## Architecture (target)

```
  SymPy / CAS            Code printing            Cython bridge            Integrators
 ┌────────────┐   ->    ┌──────────────┐   ->   ┌───────────────┐   ->   ┌──────────────┐
 │  dy/dt =   │         │  generated   │        │  compiled .so │        │ DOP853 / RK  │
 │  f(t, y)   │         │  C kernel    │        │  + Python API │        │  step loop   │
 └────────────┘         └──────────────┘        └───────────────┘        └──────────────┘
```

Each stage is meant to be reusable on its own: the code printer, the Cython
build glue, and the integrators are independent tools.

## Current status

**Milestone 1 — SciPy baseline (done).** A `main` entry point solves the classic
[figure-eight three-body choreography](https://en.wikipedia.org/wiki/Three-body_problem#Special-case_solutions)
with `scipy.integrate.solve_ivp` (DOP853) and saves a static trajectory plot.
This gives us a **correct reference solution** and a **timing target** for the
custom engine that follows.

**Milestone 3 — Cython integrators (done).** `RK23`, `RK45`, and `DOP853` are
reimplemented in Cython as a step-for-step copy of SciPy's `RungeKutta`
solvers. They use the same tableaus, initial-step heuristic, error norms, and
step-size controller, so they accept the same steps and make the same number of
RHS evaluations as `solve_ivp`. State-sized work is done in typed-memoryview loops
over buffers that cnumpy allocates once per solve. The step loop makes no NumPy
calls and creates no temporary arrays. Right-hand sides are `cdef` subclasses of
`OdeSystem`, and the integrator calls them through the vtable. A Python callable
also works through an adapter, which is slower.

**Milestone 3b — the rest of `solve_ivp` (done).** `t_eval`, `dense_output`
(an `OdeSolution`), `events` (`terminal` as a bool or a count, plus
`direction`), `vectorized`, `args`, per-component `rtol`/`atol`, `first_step`,
`max_step`, and SciPy's validation and warnings are all supported. Output
points, interpolants, and event times and states match SciPy to round-off,
with identical RHS evaluation counts. That includes DOP853's 3 extra
dense-output stages, which are evaluated only when an interpolant is needed.
Event roots are found with a C port of SciPy's `brentq`. Events can be native
`EventFunction` subclasses, such as the built-in `ComponentCrossing`, so a fully
compiled problem never calls into Python. The stepper classes mirror
`scipy.integrate.OdeSolver` (`step()`, `dense_output()`, `t_old`, `step_size`).
Not supported: complex-valued `y0`.

Package layout (`src/` layout):

```
src/fast_numerical_integration/
  integrators.pyx/.pxd  # OdeSystem, RK23/RK45/DOP853 steppers, solve_ivp (Cython)
  nbody.pyx             # NBodyGravity: native N-body RHS (stand-in for generated C)
  tableaus.py           # Butcher tableaus, bit-identical to SciPy's
  three_body.py         # figure-eight ICs, three_body_rhs, ThreeBodySystem(.native_rhs())
  visualize.py          # plot_trajectories(solution, out_path) -> PNG
  main.py               # main(): solve with solve_ivp, print stats, write outputs/figure_eight.png
  benchmark.py          # timing harness: scipy vs cython (Python RHS) vs cython (native RHS)
setup.py                # Cython extension build (metadata stays in pyproject.toml)
```

### Using the Cython engine

`integrators.solve_ivp` takes the same arguments as `scipy.integrate.solve_ivp`
(default method `RK45`):

```python
import numpy as np
from fast_numerical_integration.integrators import ComponentCrossing, solve_ivp
from fast_numerical_integration.three_body import figure_eight_system

system = figure_eight_system()
sol = solve_ivp(
    system.native_rhs(), (0.0, system.period), system.y0, method="DOP853",
    t_eval=np.linspace(0.0, system.period, 2000), dense_output=True,
    events=[ComponentCrossing(1, direction=1)],    # y1 crosses 0 upward
    rtol=1e-10, atol=1e-12,
)
sol.t, sol.y              # solution at t_eval
sol.sol(1.234)            # dense interpolant anywhere in the interval
sol.t_events, sol.y_events
```

Plain Python RHS and event callables also work, with SciPy's `terminal` and
`direction` attributes. They are slower because every call goes back through
Python.

To write a native RHS or event, subclass `OdeSystem` or `EventFunction` in a
`.pyx` file and implement its `cdef eval`:

```cython
cdef int eval(self, double t, const double* y, double* dydt, Py_ssize_t n) except -1   # OdeSystem
cdef double eval(self, double t, const double* y, Py_ssize_t n) except? -1              # EventFunction
```

`nbody.pyx` is an example. The SymPy-generated C kernels in later milestones will
plug in through the same hook.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                        # create the venv, install deps, compile Cython
uv run fni                                     # or: uv run python -m fast_numerical_integration.main
```

This writes `outputs/figure_eight.png` and prints solver stats, e.g.:

```
System:        figure-eight
Method:        DOP853 (scipy solve_ivp)
Steps taken:   2000 sampled, 1454 RHS evals
Wall time:     22.30 ms
Energy:        E0=-1.287142e+00  ET=-1.287142e+00  |drift|=2.61e-10
Saved plot ->  outputs/figure_eight.png
```

The tiny energy drift over one period confirms the dynamics and initial
conditions are correct.

The Cython extensions build when the package installs. `uv` tracks the `.pyx` and
`.pxd` files as cache keys, so `uv sync` or `uv run` rebuilds them after an
edit. Building needs a C compiler.

### Benchmark

```bash
uv run fni-bench                               # or: uv run python -m fast_numerical_integration.benchmark
```

Times `RK23`, `RK45`, and `DOP853` on the figure-eight system (one period,
rtol=1e-10, atol=1e-12) with three engines:

- `scipy`: `scipy.integrate.solve_ivp`.
- `cython+pyrhs`: the Cython integrators calling the NumPy RHS through Python.
  The RHS cost matches SciPy's, so this measures integrator overhead alone.
- `cython`: the Cython integrators with the native `NBodyGravity` RHS.

Representative run:

```
solver                 best (ms)      mean±sd (ms)    nfev   eval/ms     E drift   period err
---------------------------------------------------------------------------------------------
scipy RK23               626.758    636.353±10.984   43064      68.7    7.66e-10     6.93e-08
cython+pyrhs RK23        432.419     435.549±2.256   43064      99.6    7.67e-10     6.93e-08
cython RK23                3.747       3.966±0.299   43064   11494.1    7.67e-10     6.93e-08
scipy RK45                39.150      39.270±0.093    2936      75.0    2.98e-10     7.78e-08
cython+pyrhs RK45         29.834      31.064±1.231    2936      98.4    2.98e-10     7.78e-08
cython RK45                0.201       0.207±0.005    2936   14633.5    2.98e-10     7.78e-08
scipy DOP853              15.282      15.353±0.060    1202      78.7    2.61e-10     7.29e-08
cython+pyrhs DOP853       12.178      12.234±0.037    1202      98.7    2.61e-10     7.29e-08
cython DOP853              0.086       0.091±0.003    1202   13906.7    2.61e-10     7.29e-08
```

The Cython engine makes exactly the same number of RHS evaluations as SciPy and
reaches the same accuracy. With a native RHS, DOP853 runs about **175x faster**
than `solve_ivp`. With the Python RHS the gain is only ~1.3x, so most of SciPy's
time is spent calling the Python RHS. That is why the remaining milestones
generate compiled kernels. Numbers vary by machine.

Run the tests:

```bash
uv run pytest
```

## Roadmap

- [x] **Milestone 1** — SciPy `solve_ivp` figure-eight baseline + visualization.
- [x] **Milestone 2** — Benchmark harness timing `solve_ivp` across `RK23`,
      `RK45`, and `DOP853` on the same system (baseline numbers to beat).
- [x] **Milestone 3** — Cython RK23 / RK45 / DOP853 with adaptive steps, matching
      SciPy step for step; native `OdeSystem` RHS interface.
- [x] **Milestone 3b** — Dense output, `t_eval`, events, and the rest of
      SciPy's `solve_ivp` options for these methods.
- [ ] **Milestone 4** — SymPy → C code printing for the RHS.
- [ ] **Milestone 5** — Build glue that compiles the generated C into an
      `OdeSystem` subclass; benchmark against the baseline.

## License

TBD.
