"""Tests for the benchmark harness.

Kept fast (few repeats, loose tolerance) -- these check the harness wiring and
metrics, not integrator performance.
"""

import math

from fast_numerical_integration.benchmark import (
    DEFAULT_ENGINES,
    DEFAULT_METHODS,
    benchmark_solver,
    format_table,
    make_scipy_solver,
    run_benchmarks,
)
from fast_numerical_integration.three_body import figure_eight_system


def test_run_benchmarks_covers_all_methods():
    results = run_benchmarks(n_repeats=1, rtol=1e-8, atol=1e-10)
    assert [r.label for r in results] == [
        f"{e} {m}" for m in DEFAULT_METHODS for e in DEFAULT_ENGINES
    ]
    for r in results:
        assert r.success
        assert r.best_ms > 0.0
        assert r.nfev > 0
        assert math.isfinite(r.energy_drift)
        assert math.isfinite(r.periodicity_error)


def test_high_accuracy_solver_returns_to_start():
    system = figure_eight_system()
    solver = make_scipy_solver("DOP853", rtol=1e-10, atol=1e-12)
    result = benchmark_solver(system, solver, "scipy DOP853", n_repeats=2)
    # One period of the figure-eight should close on itself.
    assert result.periodicity_error < 1e-2
    assert result.energy_drift < 1e-6


def test_cython_engine_matches_scipy_nfev():
    results = run_benchmarks(n_repeats=1, rtol=1e-8, atol=1e-10)
    by_label = {r.label: r for r in results}
    for m in DEFAULT_METHODS:
        nfevs = {by_label[f"{e} {m}"].nfev for e in DEFAULT_ENGINES}
        assert len(nfevs) == 1, f"{m}: engines disagree on nfev {nfevs}"


def test_format_table_lists_every_solver():
    results = run_benchmarks(n_repeats=1, rtol=1e-8, atol=1e-10)
    table = format_table(results)
    for r in results:
        assert r.label in table
