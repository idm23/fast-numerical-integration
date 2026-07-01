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

Package layout (`src/` layout, ready for future C/Cython sources):

```
src/fast_numerical_integration/
  three_body.py   # figure-eight ICs, three_body_rhs(t, y, masses, G), ThreeBodySystem
  visualize.py    # plot_trajectories(solution, out_path) -> PNG (headless Agg backend)
  main.py         # main(): solve with solve_ivp, print stats, write outputs/figure_eight.png
```

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                        # create the venv and install deps
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

Run the tests:

```bash
uv run pytest
```

## Roadmap

- [x] **Milestone 1** — SciPy `solve_ivp` figure-eight baseline + visualization.
- [ ] **Milestone 2** — Benchmark harness timing `solve_ivp` across `RK23`,
      `RK45`, and `DOP853` on the same system (baseline numbers to beat).
- [ ] **Milestone 3** — Hand-written DOP853 / Runge-Kutta integrators in Python,
      validated against the SciPy reference.
- [ ] **Milestone 4** — SymPy → C code printing for the RHS.
- [ ] **Milestone 5** — Cython build bridge compiling the generated C and exposing
      a fast integration loop; benchmark against the baseline.

## License

TBD.
