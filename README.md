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
  integrators.pyx/.pxd  # GIL-free C core (fni_*), OdeSystem/EventFunction, steppers, solve_ivp, integrate
  batch.pyx             # integrate_batch: many segments in one nogil loop (C API example)
  nbody.pyx             # NBodyGravity: native N-body RHS (stand-in for generated C)
  tableaus.py           # Butcher tableaus, bit-identical to SciPy's
  three_body.py         # figure-eight ICs, three_body_rhs, ThreeBodySystem(.native_rhs())
  visualize.py          # plot_trajectories(solution, out_path) -> PNG
  main.py               # main(): solve with solve_ivp, print stats, write outputs/figure_eight.png
  benchmark.py          # timing harness: scipy vs cython (Python RHS) vs cython (native RHS)
  suite.py              # fni-suite: benchmark + figures for every catalog problem and event scenario
  testing.py            # run SciPy with our summation order (bit-identical comparisons)
  problems/
    catalog.py          # 41 canonical test problems: RHS, exact/published references, invariants
    native.pyx          # native (Cython) RHS and event classes for every catalog problem
    events.py           # event scenarios with exact event times (apsides, pendulum, bouncing ball)
setup.py                # Cython extension build (metadata stays in pyproject.toml)
```

**Milestone 3c — canonical test suite (done).** A catalog of 41 standard
problems, each with a NumPy RHS and an equivalent native one:

- **DETEST** A1–E5, with the published reference end values (Hull et al. 1972;
  Enright & Pryce 1987).
- **Hairer–Nørsett–Wanner:** Arenstorf, Pleiades, Brusselator.
- **Long-time:** Kepler over 100–1000 orbits, the outer solar system over
  2×10⁵ days, the figure-eight choreography over 50 periods.
- **Stability:** Prothero–Robinson, Van der Pol for μ = 1…1000, Robertson,
  Lorenz, finite-time blow-up.
- **Event scenarios:** Kepler apsides, a large-amplitude pendulum, a bouncing
  ball with restarts.

The tests are grouped by the oracle they check against:

| directory | checks |
|---|---|
| `tests/parity/` | SciPy's own `solve_ivp` tests, ported; RHS-evaluation parity on every catalog problem; **bit-identical** trajectories against SciPy patched to our summation order |
| `tests/accuracy/` | DETEST global error vs tolerance; observed order 3/5/8 with fixed steps; dense-output order; references cross-checked against Radau and LSODA |
| `tests/long_time/` | Kepler: linear invariant drift and quadratic phase-error growth; two independent solar-system formulations agree; linear invariants exact to round-off |
| `tests/stability/` | step size rides the stability boundary computed from each tableau (2.51 / 3.31 / 6.39); stiffness scaling; blow-up; Lorenz error growth at the Lyapunov rate |
| `tests/events/` | exact event times on physical problems; edge cases (t0, t_end, simultaneous/terminal ordering, root pairs and grazing touches missed exactly as in SciPy); crossings found by splitting a step at roots and by interior test points |
| `tests/baseline/` | SciPy baseline, benchmark harness, suite smoke test |

Findings worth knowing:

- RK23 and RK45 match SciPy's RHS-evaluation count on every problem. DOP853
  can differ by a few percent: its error estimate is a heavily cancelling
  sum, so near round-off level summation order flips accept/reject decisions.
  SciPy shows the same sensitivity to a 1-ulp change in `y0`. With SciPy
  patched to our summation order (`fast_numerical_integration.testing`),
  every trajectory is bit-identical.
- RK45 at rtol=1e-3 goes unstable on Robertson (|y| ~ 1e7, then fails). This
  is a known Dormand–Prince stiffness pathology, and SciPy does the same.

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

### Native systems, runtime parameters and the C API

A native RHS is a GIL-free C function plus a context pointer. To write one,
subclass `OdeSystem` (or `EventFunction` for events) in a `.pyx` file. In
`__init__`, register the C function with `c_fn` / `c_ctx` and declare the
runtime parameters:

```cython
from fast_numerical_integration.integrators cimport OdeSystem

cdef int _vdp(double t, const double* y, double* dydt, Py_ssize_t n, void* ctx) noexcept nogil:
    cdef const double* p = <const double*> ctx          # p[0] = mu
    dydt[0] = y[1]
    dydt[1] = p[0] * (1.0 - y[0] * y[0]) * y[1] - y[0]
    return 0

cdef class VanDerPol(OdeSystem):
    def __init__(self, double mu=1.0):
        self.n_states = 2
        self._init_params([mu], ("mu",))
        self.c_fn = _vdp
        self.c_ctx = self._params_ptr()
```

Parameters change between solves without recompiling, which is what
continuation needs:

```python
vdp = VanDerPol()
for mu in (1.0, 10.0, 100.0):
    vdp.params = [mu]
    sol = solve_ivp(vdp, (0, 20), [2.0, 0.0], method="DOP853")
```

(Overriding `cdef eval` still works, but such systems need the GIL.)

All three entry points run on one C core, so they take bit-identical steps:

| entry point | what it does |
|---|---|
| `solve_ivp(...)` | full SciPy-compatible API: trajectory, `t_eval`, dense output, events |
| `integrate(...)` | endpoint + event locations only (shooting). Releases the GIL when the system and events are native, so Python threads run in parallel |
| `fni_integrate(...)` (C) | `cimport`able from other Cython modules and callable inside `with nogil`. `batch.integrate_batch` is the reference example: it propagates many segments at ~1 µs per segment |

```cython
from fast_numerical_integration.integrators cimport (
    fni_integrate, fni_options, fni_options_init, fni_stats, FNI_DOP853, FNI_OK)

fni_options_init(&opt)          # defaults first, so fields added later get defaults
opt.rtol = 1e-10
with nogil:
    rc = fni_integrate(FNI_DOP853, rhs_fn, ctx, n, t0, y0, t1, &opt,
                       events, n_events, y_out, rec_event, rec_t, rec_y, max_records, &stats)
```

The lower-level `fni_solver_init` / `fni_solver_step` / `fni_solver_dense_eval` /
`fni_events_begin` / `fni_events_next` / `fni_solver_restart` functions expose single steps for custom drivers. All
solver state lives in a caller-owned struct, so concurrent integrations are
safe if the RHS kernels are reentrant. Every catalog kernel in
`problems/native.pyx` is written that way.

### Hybrid systems: guards and reset maps

A hybrid system is a flow whose state jumps, or whose equations change, at
events: impacts, contact and lift-off, stage separation, impulsive burns,
bang-bang control switches. `solve_hybrid` and `integrate_hybrid` treat every
**terminal** event as a *guard*. When it fires, a **reset map** takes the
event index, time and state just before it, and returns the state just after.
Integration then continues from the event in the same solver, with the step
size carried over.

```python
from fast_numerical_integration.integrators import ComponentCrossing, ScaleComponent, solve_hybrid
from fast_numerical_integration.problems.native import LinearSystem

ball = LinearSystem([[0, 1], [0, 0]], [0, -9.81])        # y = [height, velocity]
ground = ComponentCrossing(0, 0.0, terminal=True, direction=-1)
bounce = ScaleComponent(1, -0.8)                          # v -> -0.8 v
res = solve_hybrid(ball, (0, 3), [1.0, 0.0], ground, bounce, method="DOP853",
                   rtol=1e-10, atol=1e-12, dense_output=True, min_dwell=1e-6)
res.t_resets, res.y_before_resets, res.y_after_resets
```

- **Reset maps:** a Python callable `reset(event, t, y)` returns the new
  state, or `None` to stop at the event. Subclass `ResetMap` with a `nogil`
  `c_fn` (an `fni_reset_fn`) for speed and for GIL-free runs.
- **Modes:** a discrete mode (which arc, which contact state) lives in a
  context that the right-hand side and the reset map share; the reset map
  updates it.
- **Re-triggering:** right after a reset, the guard that fired sits on zero.
  FNI probes it slightly later along the new flow, as SUNDIALS' root finder
  does, so it cannot fire again on that zero, while a real crossing later in
  the same step still fires.
- **Guards with a root at the same instant** each fire once, in index order,
  whether they are exactly tied or apart by round-off.
- **Zeno behaviour:** `max_resets` (status -2) and `min_dwell` (status -3,
  two resets closer than this) end the integration cleanly instead of
  chattering forever.
- **Bit-identical drivers:** `solve_hybrid` (trajectory, `t_eval`, dense
  output) and `integrate_hybrid` (endpoint) take the same steps. In C,
  `fni_integrate_hybrid` runs a whole hybrid integration inside `with nogil`;
  `fni_integrate` is the same loop with no reset map.

### Detecting events inside a step

SciPy detects an event by comparing its sign at the two ends of each step.
If the event crosses zero twice within one step, in and back out, the signs
match and neither crossing is seen. That is easy to hit: away from the event
the solution is often smooth, the solver takes long steps, and a short dip
fits inside one.

```
 g    +          ╭──╮
    0 ───────────┼──┼──────────
    −   ●   ○   ○│ ○│  ○   ○   ●     ● step ends: same sign, nothing seen
        t_a      ╰──╯          t_b   ○ interior test points: the sign change shows
```

FNI keeps SciPy's behaviour by default and adds two things:

- **Every root splits the step for the other events** (always on). When one
  event is found at `t_e`, every other event is checked at `t_e` too, so a
  crossing on `[t_a, t_e]` that is cancelled at `t_b` is still found, and
  roots are reported in time order. This also fixes a crossing that was
  silently lost when a terminal event (or a guard's reset) cut the step
  short. Without hidden crossings, the roots are SciPy's, bit for bit.
- **`event_detection="interior"`** (opt-in) also compares signs at
  `n_interior` evenly spaced points inside every step, on the step's
  interpolant. It finds pairs of crossings at least `h / (n_interior + 1)`
  apart. The cost: the interpolant is built on every step (3 extra RHS
  evaluations per step for DOP853, free for RK23 and RK45), plus
  `n_interior` evaluations of each event.

```python
solve_ivp(f, (0, 10), y0, events=g)                                         # SciPy's detection
solve_ivp(f, (0, 10), y0, events=g, event_detection="interior", n_interior=8)
```

Both options work in `solve_ivp`, `solve_hybrid`, `integrate` and
`integrate_hybrid`; in C they are fields of `fni_options`
(`event_detection = FNI_EVENTS_INTERIOR`, `n_interior`).

The splitting makes a strong guarantee possible from outside: add an event on
the derivative of `g`. Its roots are the peaks and troughs of `g`, and `g` is
monotone between them, so it can cross zero at most once on each piece, and
the split finds that crossing.

## Installation

```bash
pip install fast-numerical-integration            # or: uv add fast-numerical-integration
pip install "fast-numerical-integration[plot]"    # + matplotlib, for fni / fni-suite / visualize
```

Pre-built wheels cover Linux x86_64 and macOS on Apple Silicon for CPython
3.11–3.14, and every wheel passes the full test suite before it's
published. Other platforms install from the source distribution, which needs
a C compiler. All extensions are built with floating-point contraction off,
so results are identical across platforms.

To `cimport` the C API from your own Cython package, add
`fast-numerical-integration` and `cython` to your `[build-system] requires`.
`examples/downstream/` is a minimal working package; the release workflow
builds it against every release.

## Quickstart (development)

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

### Benchmark and visualize every problem

```bash
uv run fni-suite                    # all problems + event scenarios, ~2 minutes
uv run fni-suite --problems D5 pleiades --quick
```

This writes one figure per problem to `outputs/suite/<group>/<name>.png`. Each
figure shows the solution, error along the trajectory (or conserved-quantity
drift, or step size), a DOP853 work-precision curve and wall-time bars for
SciPy and Cython. It also writes one figure per event scenario and
`summary_speedup.png`, plus `summary.md` / `summary.csv` tables. Cython's
end-to-end speedup with DOP853 ranges from about 6× (C4, a dense 51×51 linear
system, where the work is the same matrix-vector product as NumPy's) to about
500× (outer solar system), with identical step counts and accuracy.

Run the tests:

```bash
uv run pytest               # ~1.5 minutes
uv run pytest --runslow     # include long integrations
uv run pytest tests/events  # one oracle group
```

CI (`.github/workflows/tests.yml`) runs the full suite, including
`--runslow`, on Python 3.11, 3.12 and 3.13 for every pull request and push
to `main`. The `all-tests` check aggregates the matrix and is the status to
require before merging.

## Roadmap

- [x] **Milestone 1** — SciPy `solve_ivp` figure-eight baseline + visualization.
- [x] **Milestone 2** — Benchmark harness timing `solve_ivp` across `RK23`,
      `RK45`, and `DOP853` on the same system (baseline numbers to beat).
- [x] **Milestone 3** — Cython RK23 / RK45 / DOP853 with adaptive steps, matching
      SciPy step for step; native `OdeSystem` RHS interface.
- [x] **Milestone 3b** — Dense output, `t_eval`, events, and the rest of
      SciPy's `solve_ivp` options for these methods.
- [x] **Milestone 3c** — Canonical test problems (DETEST, Hairer, long-time,
      stability, events), split by oracle, with a benchmark/figure suite.
- [x] **Milestone 3d** — GIL-free C core with a `cimport`able C API
      (`fni_integrate`, single-step functions), runtime `params` on native
      systems and events, `integrate()` endpoint API, `integrate_batch`.
- [x] **Milestone 3e** — Hybrid integration: guards, reset maps, modes and
      Zeno limits in the C core (`fni_integrate_hybrid`), `solve_hybrid`,
      `integrate_hybrid` (0.3.0).
- [x] **Milestone 3f** — Event detection inside a step: roots split the step
      for the other events, opt-in interior test points
      (`event_detection="interior"`), `fni_options_init` (0.4.0).
- SymPy code generation lives in the companion package
  [auto-optimal-control](https://github.com/idm23/auto-optimal-control),
  which builds its compiled kernels on this C API.

## Releasing

Releases are published from GitHub Actions (`.github/workflows/release.yml`)
with PyPI trusted publishing, so no tokens are stored.

1. Bump `version` in `pyproject.toml` and merge the change to `main`.
2. Optional rehearsal: run the *Release* workflow manually with
   `target: testpypi`, then `pip install -i https://test.pypi.org/simple/
   --extra-index-url https://pypi.org/simple/ fast-numerical-integration`.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`. The workflow
   checks that the tag matches the version, builds and tests every wheel, and
   uploads to PyPI.

## License

MIT. See [LICENSE](LICENSE).
