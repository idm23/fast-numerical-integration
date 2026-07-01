"""Static trajectory visualization for the three-body baseline.

Renders the orbital paths and writes a PNG. Kept independent of any particular
solver so it can plot SciPy solutions today and our own engine's output later.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fast_numerical_integration.three_body import N_BODIES, N_DIMS

_BODY_COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def plot_trajectories(
    solution,
    out_path: str | Path,
    title: str = "Three-body figure-eight (scipy solve_ivp)",
) -> Path:
    """Plot the orbital paths of all bodies and save to ``out_path``.

    Parameters
    ----------
    solution
        A :class:`scipy.integrate.OdeSolution`-style result with ``.y`` of shape
        ``(12, n_steps)`` (positions stacked above velocities). Any object with a
        matching ``.y`` array works.
    out_path
        Destination PNG path; parent directories are created as needed.
    title
        Figure title.

    Returns
    -------
    Path
        The path the figure was written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # positions: rows 0..5 are (x1,y1, x2,y2, x3,y3) over time.
    positions = np.asarray(solution.y[: N_BODIES * N_DIMS])

    fig, ax = plt.subplots(figsize=(7, 7))
    for body in range(N_BODIES):
        xs = positions[body * N_DIMS]
        ys = positions[body * N_DIMS + 1]
        color = _BODY_COLORS[body % len(_BODY_COLORS)]
        ax.plot(xs, ys, color=color, lw=1.2, label=f"body {body + 1}")
        ax.plot(xs[0], ys[0], "o", color=color, ms=8)  # start marker

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
