"""Entry point: solve the figure-eight three-body problem with SciPy and plot it.

This is the SciPy-based baseline. It establishes a correct reference solution and
a timing target that our future hand-written engine must match and then beat.
"""

import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from fast_numerical_integration.three_body import figure_eight_system
from fast_numerical_integration.visualize import plot_trajectories

# Default output location, resolved relative to the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "figure_eight.png"


def main(output_path: str | Path = DEFAULT_OUTPUT) -> int:
    """Solve one period of the figure-eight orbit and save a trajectory PNG."""
    system = figure_eight_system()

    # Sample densely over one period for a smooth plot.
    t_span = (0.0, system.period)
    t_eval = np.linspace(*t_span, 2000)

    start = time.perf_counter()
    solution = solve_ivp(
        system.rhs,
        t_span,
        system.y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    elapsed = time.perf_counter() - start

    if not solution.success:
        print(f"Solver failed: {solution.message}")
        return 1

    energy_0 = system.energy(solution.y[:, 0])
    energy_T = system.energy(solution.y[:, -1])
    drift = abs(energy_T - energy_0)

    print(f"System:        {system.name}")
    print(f"Method:        DOP853 (scipy solve_ivp)")
    print(f"Period T:      {system.period:.6f}")
    print(f"Steps taken:   {solution.t.size} sampled, {solution.nfev} RHS evals")
    print(f"Wall time:     {elapsed * 1e3:.2f} ms")
    print(f"Energy:        E0={energy_0:.6e}  ET={energy_T:.6e}  |drift|={drift:.2e}")

    out = plot_trajectories(solution, output_path)
    print(f"Saved plot ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
