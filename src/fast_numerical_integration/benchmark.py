"""Benchmark harness for three-body integrators.

Times :func:`scipy.integrate.solve_ivp` and our Cython engine across several
methods on a shared
:class:`~fast_numerical_integration.three_body.ThreeBodySystem`, reporting both
speed (wall time, RHS evaluations) and accuracy (energy drift and how closely a
periodic orbit returns to its start).

The Cython engine runs in two configurations. With the native compiled RHS it
shows the end-to-end speed-up. With the Python RHS it isolates the integrator
overhead, since the RHS cost is then the same as SciPy's.

:func:`benchmark_solver` accepts any callable that integrates the system, so
new engines can be added to the same table.
"""

import time
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from fast_numerical_integration import integrators
from fast_numerical_integration.three_body import (
    ThreeBodySystem,
    figure_eight_system,
)

# SciPy solve_ivp methods to compare by default.
DEFAULT_METHODS = ("RK23", "RK45", "DOP853")

# Integration engines to compare by default (see :func:`run_benchmarks`).
DEFAULT_ENGINES = ("scipy", "cython+pyrhs", "cython")


@dataclass
class BenchmarkResult:
    """Timing and accuracy summary for one solver on one system."""

    label: str
    n_repeats: int
    best_ms: float
    mean_ms: float
    stdev_ms: float
    nfev: int
    energy_drift: float
    periodicity_error: float
    success: bool

    @property
    def evals_per_ms(self) -> float:
        """RHS evaluations per millisecond of best wall time (throughput)."""
        return self.nfev / self.best_ms if self.best_ms > 0 else float("nan")


# A solver takes a system and returns a scipy-style result object exposing
# ``.success``, ``.nfev`` and ``.y`` (state columns, positions above velocities).
Solver = Callable[[ThreeBodySystem], object]


def make_scipy_solver(method: str, *, rtol: float, atol: float) -> Solver:
    """Build a solver that integrates one period of ``system`` with ``solve_ivp``.

    No ``t_eval`` is passed so we measure the raw integration cost rather than the
    overhead of densely sampling the solution.
    """

    def solver(system: ThreeBodySystem):
        return solve_ivp(
            system.rhs,
            (0.0, system.period),
            system.y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    return solver


def make_cython_solver(
    method: str, *, rtol: float, atol: float, native_rhs: bool = True
) -> Solver:
    """Build a solver that integrates one period with the Cython engine.

    With ``native_rhs`` the compiled :class:`~fast_numerical_integration.nbody.NBodyGravity`
    kernel is used; otherwise the NumPy ``system.rhs`` is called back through
    Python on every evaluation.
    """

    def solver(system: ThreeBodySystem):
        fun = system.native_rhs() if native_rhs else system.rhs
        return integrators.solve_ivp(
            fun,
            (0.0, system.period),
            system.y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    return solver


def benchmark_solver(
    system: ThreeBodySystem,
    solver: Solver,
    label: str,
    *,
    n_repeats: int = 7,
    warmup: bool = True,
) -> BenchmarkResult:
    """Time ``solver`` on ``system`` over ``n_repeats`` runs and gauge accuracy.

    Timing reports the best run (least noisy estimate of achievable speed) plus
    the mean and population standard deviation across runs. Accuracy is taken
    from the final run's solution: energy drift over the interval and, for a
    periodic orbit, the distance between the final state and the start.
    """
    if warmup:
        solver(system)  # prime caches / imports; result discarded

    timings_ms = []
    solution = None
    for _ in range(n_repeats):
        start = time.perf_counter()
        solution = solver(system)
        timings_ms.append((time.perf_counter() - start) * 1e3)

    success = bool(getattr(solution, "success", False))
    nfev = int(getattr(solution, "nfev", 0))

    if success:
        y = np.asarray(solution.y)
        energy_drift = abs(system.energy(y[:, -1]) - system.energy(y[:, 0]))
        periodicity_error = float(np.linalg.norm(y[:, -1] - system.y0))
    else:
        energy_drift = float("nan")
        periodicity_error = float("nan")

    return BenchmarkResult(
        label=label,
        n_repeats=n_repeats,
        best_ms=min(timings_ms),
        mean_ms=fmean(timings_ms),
        stdev_ms=pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        nfev=nfev,
        energy_drift=energy_drift,
        periodicity_error=periodicity_error,
        success=success,
    )


def run_benchmarks(
    system: Optional[ThreeBodySystem] = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    n_repeats: int = 7,
    engines: Sequence[str] = DEFAULT_ENGINES,
) -> list[BenchmarkResult]:
    """Benchmark every ``method`` with each engine on ``system`` (default: figure-eight).

    ``engines`` is any subset of :data:`DEFAULT_ENGINES`: ``"scipy"`` (SciPy
    ``solve_ivp``), ``"cython"`` (Cython engine with the native RHS) and
    ``"cython+pyrhs"`` (Cython engine calling the Python RHS).
    """
    if system is None:
        system = figure_eight_system()

    results = []
    for method in methods:
        for engine in engines:
            if engine == "scipy":
                solver = make_scipy_solver(method, rtol=rtol, atol=atol)
            elif engine == "cython":
                solver = make_cython_solver(method, rtol=rtol, atol=atol)
            elif engine == "cython+pyrhs":
                solver = make_cython_solver(
                    method, rtol=rtol, atol=atol, native_rhs=False
                )
            else:
                raise ValueError(f"unknown engine {engine!r}")
            label = f"{engine} {method}"
            results.append(
                benchmark_solver(system, solver, label, n_repeats=n_repeats)
            )
    return results


def format_table(results: Sequence[BenchmarkResult]) -> str:
    """Render benchmark results as a fixed-width comparison table."""
    width = max([len("solver")] + [len(r.label) for r in results]) + 2
    header = (
        f"{'solver':<{width}}{'best (ms)':>11}{'mean±sd (ms)':>18}"
        f"{'nfev':>8}{'eval/ms':>10}{'E drift':>12}{'period err':>13}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        mean_sd = f"{r.mean_ms:.3f}±{r.stdev_ms:.3f}"
        status = "" if r.success else "  (FAILED)"
        lines.append(
            f"{r.label:<{width}}{r.best_ms:>11.3f}{mean_sd:>18}"
            f"{r.nfev:>8}{r.evals_per_ms:>10.1f}"
            f"{r.energy_drift:>12.2e}{r.periodicity_error:>13.2e}{status}"
        )
    return "\n".join(lines)


def main() -> int:
    """Run the default benchmark suite and print a comparison table."""
    system = figure_eight_system()
    print(f"Benchmarking '{system.name}' over one period (T={system.period:.4f})")
    print("rtol=1e-10, atol=1e-12, 7 repeats each\n")
    results = run_benchmarks(system)
    print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
