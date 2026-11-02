"""Benchmark and visualize every catalog problem and event scenario: ``fni-suite``.

For each problem in :mod:`fast_numerical_integration.problems.catalog` this:

1. times RK23 / RK45 / DOP853 with three engines:

   - ``cython``: our engine with the native RHS;
   - ``scipy``: :func:`scipy.integrate.solve_ivp` with the NumPy RHS (typical usage);
   - ``scipy + compiled RHS``: SciPy calling the same native RHS object, which
     isolates the integrator's own overhead;

2. sweeps DOP853 over tolerances (work-precision: accuracy vs wall time);
3. records error vs time (or conserved-quantity drift, or step size, by group);
4. writes one figure per problem, one per event scenario, summary figures, and
   Markdown / CSV tables to ``outputs/suite/``.

Run ``uv run fni-suite`` (all problems; a few minutes, mostly SciPy's time)
or ``uv run fni-suite --problems D5 arenstorf --quick``.
"""

import argparse
import csv
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp

from fast_numerical_integration.integrators import solve_ivp
from fast_numerical_integration.problems import catalog, events

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "outputs" / "suite"

METHODS = ("RK23", "RK45", "DOP853")
BENCH_RTOL = 1e-6
SWEEP_RTOLS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
DETAIL_RTOL = 1e-8


def _atol(rtol):
    return rtol * 1e-3


# =============================================================================
# Visual style (reference palette from the dataviz skill; validated light mode)
# =============================================================================

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order. The first three (blue, orange, aqua) pass
# the all-pairs CVD and normal-vision checks, so they encode the three engines.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = "Blues"

ENGINES = ("cython", "scipy", "scipy + compiled RHS")
ENGINE_STYLE = {
    "cython": dict(color=CATEGORICAL[0], linestyle="-", marker="o"),
    "scipy": dict(color=CATEGORICAL[1], linestyle="-", marker="s"),
    # Aqua is < 3:1 on the light surface: dashed line + legend label as relief.
    "scipy + compiled RHS": dict(color=CATEGORICAL[2], linestyle="--", marker="^"),
}
ENGINE_LABEL = {
    "cython": "Cython (native RHS)",
    "scipy": "SciPy (NumPy RHS)",
    "scipy + compiled RHS": "SciPy (compiled RHS)",
}


def _overlay(ax, engine, x, y, **kw):
    """Draw engines so coincident curves stay visible: SciPy as a wide halo
    underneath, Cython as a thin core on top (legend explains the encoding)."""
    st = ENGINE_STYLE[engine]
    if engine == "cython":
        ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], linewidth=1.1, zorder=3,
                label=ENGINE_LABEL[engine], **kw)
    else:
        ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], linewidth=3.4, alpha=0.55, zorder=2,
                solid_capstyle="round", label=ENGINE_LABEL[engine], **kw)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 5, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 8,
    })


# =============================================================================
# Running and timing
# =============================================================================

def time_call(fn, budget=0.4, max_reps=7):
    """Best-of-N wall time (seconds) within a time budget, plus the last result."""
    t0 = time.perf_counter()
    result = fn()
    best = time.perf_counter() - t0
    spent = best
    reps = 1
    while spent < budget and reps < max_reps:
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        spent += dt
        reps += 1
    return best, result


def _engine_call(engine, p, method, rtol, **kw):
    kw = dict(method=method, rtol=rtol, atol=_atol(rtol), **kw)
    if engine == "cython":
        nat = p.native()
        return lambda: solve_ivp(nat, p.t_span, p.y0, **kw)
    if engine == "scipy":
        return lambda: scipy_solve_ivp(p.rhs, p.t_span, p.y0, **kw)
    nat = p.native()
    return lambda: scipy_solve_ivp(nat, p.t_span, p.y0, **kw)


def _accuracy(p, sol):
    """End-point error (relative), or for blow-up problems the error at t = 0.9."""
    if p.expect_failure:
        idx = np.argmin(np.abs(sol.t - 0.9))
        exact = p.exact(sol.t[idx:idx + 1])[:, 0]
        return float(np.max(np.abs(sol.y[:, idx] - exact)) / np.max(np.abs(exact)))
    return catalog.end_error(p, sol.y[:, -1])


def _kw_for(p):
    # Blow-up runs fail by design; include t = 0.9 in the output for accuracy.
    return {"t_eval": np.array([0.0, 0.9])} if p.expect_failure else {}


@dataclass
class ProblemResult:
    problem: catalog.Problem
    bench: dict = field(default_factory=dict)   # (engine, method) -> dict(time, nfev, err, status)
    sweep: dict = field(default_factory=dict)   # engine -> list of (rtol, time, err, nfev)
    detail: dict = field(default_factory=dict)  # engine -> dict(t, err, h_t, h, inv_t, inv)
    reference: tuple | None = None              # (tq, yq) for the solution panel


def run_problem(p, quick=False):
    res = ProblemResult(p)
    engines = ENGINES if not quick else ("cython", "scipy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # overflow in rejected stiff trial steps
        for method in METHODS:
            for engine in engines:
                t, sol = time_call(_engine_call(engine, p, method, BENCH_RTOL, **_kw_for(p)))
                res.bench[(engine, method)] = dict(time=t, nfev=sol.nfev, status=sol.status,
                                                   err=_accuracy(p, sol))

        rtols = SWEEP_RTOLS if not quick else SWEEP_RTOLS[1::2]
        for engine in ("cython", "scipy"):
            pts = []
            for rtol in rtols:
                t, sol = time_call(_engine_call(engine, p, "DOP853", rtol, **_kw_for(p)))
                pts.append((rtol, t, _accuracy(p, sol), sol.nfev))
            res.sweep[engine] = pts

        # Detailed DOP853 runs: error along the trajectory, step sizes, invariants.
        t_end = 0.99 if p.expect_failure else p.t_span[1]
        tq = np.linspace(p.t_span[0], t_end, 600)
        ref = catalog.reference(p, tq)
        res.reference = (tq, ref)
        scale = max(np.max(np.abs(ref)), 1e-300)
        for engine in ("cython", "scipy"):
            fn = _engine_call(engine, p, "DOP853", DETAIL_RTOL, dense_output=True)
            sol = fn()
            y = sol.sol(tq)
            d = dict(t=tq, err=np.max(np.abs(y - ref), axis=0) / scale,
                     h_t=sol.t[1:], h=np.abs(np.diff(sol.t)))
            if p.invariants:
                name, inv = next(iter(p.invariants.items()))
                v = inv(y)
                d["inv_name"] = name
                d["inv"] = np.abs(v - inv(p.y0[:, None])) / max(abs(inv(p.y0[:, None])[0]), 1.0)
            res.detail[engine] = d
    return res


# =============================================================================
# Figures
# =============================================================================

def _plot_solution(ax, p, tq, ref):
    labels = p.labels or tuple(f"y{i + 1}" for i in range(p.n))
    opts = dict(p.plot_opts)
    if p.plot == "orbit":
        for k, (i, j) in enumerate(p.plot_pairs):
            c = CATEGORICAL[k % len(CATEGORICAL)]
            lab = labels[k] if p.labels and k < len(p.labels) else None
            ax.plot(ref[i], ref[j], color=c, linewidth=1.0, label=lab)
            ax.plot(ref[i, 0], ref[j, 0], "o", color=c, markersize=4)
        if opts.get("sun"):
            ax.plot(0, 0, marker="*", color=INK_2, markersize=9, linestyle="none")
        for bx, by, name in opts.get("bodies", []):
            ax.plot(bx, by, "o", color=INK_2, markersize=5)
            ax.annotate(name, (bx, by), textcoords="offset points", xytext=(4, 4), color=INK_2, fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        i, j = p.plot_pairs[0]
        ax.set_xlabel(f"y[{i}]")
        ax.set_ylabel(f"y[{j}]")
        if p.labels:
            ax.legend(loc="best", ncol=2 if len(p.plot_pairs) > 4 else 1)
    elif p.plot == "phase":
        i, j = p.plot_pairs[0]
        ax.plot(ref[i], ref[j], color=CATEGORICAL[0], linewidth=1.0)
        ax.plot(ref[i, 0], ref[j, 0], "o", color=CATEGORICAL[0], markersize=5)
        ax.annotate("start", (ref[i, 0], ref[j, 0]), textcoords="offset points", xytext=(5, 5),
                    color=INK_2, fontsize=8)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        if opts.get("symlog_y"):
            ax.set_yscale("symlog", linthresh=1.0)
    elif p.plot == "heatmap" or p.n > len(CATEGORICAL):
        from matplotlib.colors import LogNorm

        # |y_i| spans many decades (C4 ends at 1e-14): log colour scale, one hue.
        mag = np.maximum(np.abs(ref), 1e-16)
        im = ax.imshow(mag, aspect="auto", origin="lower", cmap=SEQUENTIAL,
                       norm=LogNorm(vmin=max(mag.min(), 1e-12), vmax=mag.max()),
                       extent=(tq[0], tq[-1], 0.5, p.n + 0.5), interpolation="nearest")
        plt.colorbar(im, ax=ax, label="|y_i| (log)", fraction=0.046, pad=0.02)
        ax.set_xlabel("t")
        ax.set_ylabel("component i")
        ax.grid(False)
    else:
        scale = opts.get("scale", (1.0,) * p.n)
        for k in range(p.n):
            ax.plot(tq, ref[k] * scale[k], color=CATEGORICAL[k % len(CATEGORICAL)], label=labels[k])
        if opts.get("logy"):
            ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        if p.n > 1:
            ax.legend(loc="best")
    ax.set_title("Solution (reference)")


def _plot_detail(ax, p, res):
    if p.group == "stability":
        for engine in ("scipy", "cython"):
            d = res.detail[engine]
            _overlay(ax, engine, d["h_t"], d["h"])
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("accepted step size h")
        ax.set_title(f"Step size (DOP853, rtol={DETAIL_RTOL:g})")
        if p.name == "linear_stiff":
            from fast_numerical_integration.problems.native import ProtheroRobinson  # noqa: F401
            x_star = 6.394  # DOP853 real stability boundary (tests/stability)
            ax.axhline(x_star / abs(catalog.LINEAR_STIFF_LAMBDA), color=INK_2, linewidth=0.8, linestyle=":")
            ax.annotate("stability limit 6.39/|λ|", (0.02, x_star / abs(catalog.LINEAR_STIFF_LAMBDA)),
                        xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, 4),
                        color=INK_2, fontsize=8)
    elif p.group == "long_time" and p.invariants:
        for engine in ("scipy", "cython"):
            d = res.detail[engine]
            _overlay(ax, engine, d["t"][1:], np.maximum(d["inv"][1:], 1e-17))
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("relative drift")
        ax.set_title(f"Conserved quantity: {res.detail['cython']['inv_name']} (DOP853, rtol={DETAIL_RTOL:g})")
    else:
        for engine in ("scipy", "cython"):
            d = res.detail[engine]
            _overlay(ax, engine, d["t"][1:], np.maximum(d["err"][1:], 1e-17))
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("max |y - y_ref| / max |y_ref|")
        ax.set_title(f"Error along the trajectory (DOP853, rtol={DETAIL_RTOL:g})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="best")


def _plot_work_precision(ax, res):
    for engine in ("cython", "scipy"):
        pts = res.sweep[engine]
        st = ENGINE_STYLE[engine]
        t = np.array([q[1] for q in pts]) * 1e3
        e = np.maximum(np.array([q[2] for q in pts]), 1e-17)
        ax.plot(t, e, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                label=ENGINE_LABEL[engine])
    # Selective labels: the tolerance at each end of the cython curve.
    pts = res.sweep["cython"]
    for q in (pts[0], pts[-1]):
        ax.annotate(f"rtol {q[0]:.0e}", (q[1] * 1e3, max(q[2], 1e-17)), textcoords="offset points",
                    xytext=(6, -2), color=INK_2, fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (ms)")
    ax.set_ylabel("error")
    ax.set_title("Work-precision (DOP853)")
    ax.legend(loc="best")


def _plot_timing_bars(ax, res):
    engines = [e for e in ENGINES if (e, "DOP853") in res.bench]
    x = np.arange(len(METHODS))
    width = 0.8 / len(engines)
    for k, engine in enumerate(engines):
        times = [res.bench[(engine, m)]["time"] * 1e3 for m in METHODS]
        ax.bar(x + (k - (len(engines) - 1) / 2) * width, times, width * 0.92,
               color=ENGINE_STYLE[engine]["color"], label=ENGINE_LABEL[engine])
    for i, m in enumerate(METHODS):
        speedup = res.bench[("scipy", m)]["time"] / res.bench[("cython", m)]["time"]
        top = max(res.bench[(e, m)]["time"] for e in engines) * 1e3
        ax.annotate(f"{speedup:,.0f}× faster", (x[i], top), textcoords="offset points", xytext=(0, 3),
                    ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, METHODS)
    ax.set_yscale("log")
    ax.set_ylabel("wall time (ms, log)")
    ax.set_title(f"Wall time at rtol={BENCH_RTOL:g}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(engines), fontsize=7)
    ax.margins(y=0.25)


def plot_problem(res, path):
    p = res.problem
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    tq, ref = res.reference
    _plot_solution(axes[0, 0], p, tq, ref)
    _plot_detail(axes[0, 1], p, res)
    _plot_work_precision(axes[1, 0], res)
    _plot_timing_bars(axes[1, 1], res)
    fig.suptitle(f"{p.name}: {p.title}", fontsize=12, fontweight="bold", x=0.01, ha="left")
    note = f"{p.n} state variables · t ∈ [{p.t_span[0]:g}, {p.t_span[1]:.4g}] · reference: {p.ref_source or 'DOP853 at rtol 1e-13'}"
    fig.text(0.01, 0.945, note, color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(results, path):
    names = [r.problem.name for r in results]
    vs_numpy = np.array([r.bench[("scipy", "DOP853")]["time"] / r.bench[("cython", "DOP853")]["time"]
                         for r in results])
    has_compiled = all(("scipy + compiled RHS", "DOP853") in r.bench for r in results)
    order = np.argsort(vs_numpy)
    n_panels = 2 if has_compiled else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels + 1, 0.24 * len(results) + 1.6),
                             sharey=True, squeeze=False)
    panels = [("vs SciPy with NumPy RHS (end-to-end)", vs_numpy)]
    if has_compiled:
        vs_comp = np.array([r.bench[("scipy + compiled RHS", "DOP853")]["time"]
                            / r.bench[("cython", "DOP853")]["time"] for r in results])
        panels.append(("vs SciPy calling the same compiled RHS (integrator only)", vs_comp))
    for ax, (title, vals) in zip(axes[0], panels):
        y = np.arange(len(results))
        ax.barh(y, vals[order], height=0.72, color=CATEGORICAL[0])
        for yi, v in zip(y, vals[order]):
            ax.annotate(f"{v:,.0f}×", (v, yi), textcoords="offset points", xytext=(3, -3), fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("speedup of Cython DOP853 (log)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
    axes[0, 0].set_yticks(np.arange(len(results)), [names[i] for i in order])
    fig.suptitle(f"Speedup per problem (DOP853, rtol={BENCH_RTOL:g}, best of several runs)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# Event scenarios
# =============================================================================

def run_event_scenario(sc):
    kw = dict(method="DOP853", rtol=DETAIL_RTOL, atol=_atol(DETAIL_RTOL))
    out = {"scenario": sc}
    if sc.name == "bouncing_ball":
        h0, e, n = sc.extra["h0"], sc.extra["e"], sc.extra["n_bounces"]
        bounce = events.bounce
        runs = {
            "scipy": lambda: bounce(scipy_solve_ivp, h0, e, n, native_rhs=False, native_event=False, **kw),
            "cython + Python events": lambda: bounce(solve_ivp, h0, e, n, native_rhs=True, native_event=False, **kw),
            "cython + native events": lambda: bounce(solve_ivp, h0, e, n, native_rhs=True, native_event=True, **kw),
        }
        for name, fn in runs.items():
            t, r = time_call(fn)
            out[name] = dict(time=t, t_events=[r["times"]], nfev=r["nfev"])
        r = bounce(solve_ivp, h0, e, n, **kw)
        # Smooth curve from each flight's dense output (steps alone are too sparse).
        segs = [(d, np.linspace(d.t_min, d.t_max, 200)) for d in r["dense"]]
        out["trajectory"] = (np.concatenate([tq for _, tq in segs]),
                             np.concatenate([d(tq) for d, tq in segs], axis=1))
        return out
    runs = {
        "scipy": lambda: scipy_solve_ivp(sc.rhs, sc.t_span, sc.y0, events=sc.python_events(), **kw),
        "cython + Python events": lambda: solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.python_events(), **kw),
        "cython + native events": lambda: solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.native_events(), **kw),
    }
    for name, fn in runs.items():
        t, r = time_call(fn)
        out[name] = dict(time=t, t_events=r.t_events, nfev=r.nfev)
    sol = solve_ivp(sc.native(), sc.t_span, sc.y0, dense_output=True, **kw)
    tq = np.linspace(*sc.t_span, 2000)
    out["trajectory"] = (tq, sol.sol(tq))
    out["y_events"] = solve_ivp(sc.native(), sc.t_span, sc.y0, events=sc.native_events(), **kw).y_events
    return out


EVENT_RUN_STYLE = {
    "scipy": ENGINE_STYLE["scipy"],
    "cython + Python events": dict(color=CATEGORICAL[2], linestyle="--", marker="^"),
    "cython + native events": ENGINE_STYLE["cython"],
}


def plot_event_scenario(out, path):
    sc = out["scenario"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    t, y = out["trajectory"]
    if sc.name == "kepler_apsides":
        ax.plot(y[0], y[1], color=INK_2, linewidth=0.8)
        for k, (ye, lab) in enumerate(zip(out["y_events"], sc.event_labels)):
            ax.plot(ye[:, 0], ye[:, 1], "o", color=CATEGORICAL[k], markersize=7, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.plot(0, 0, marker="*", color=INK_2, markersize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif sc.name == "pendulum":
        ax.plot(y[0], y[1], color=INK_2, linewidth=0.8)
        for k, (ye, lab) in enumerate(zip(out["y_events"], sc.event_labels)):
            ax.plot(ye[:, 0], ye[:, 1], "o", color=CATEGORICAL[k], markersize=7, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_xlabel("θ")
        ax.set_ylabel("ω")
    else:
        ax.plot(t, y[0], color=INK_2, linewidth=1.0)
        impacts = out["cython + native events"]["t_events"][0]
        ax.plot(impacts, np.zeros_like(impacts), "o", color=CATEGORICAL[0], markersize=7, label="impact",
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("height (m)")
    ax.legend(loc="best")
    ax.set_title("Trajectory and located events")

    ax = axes[1]
    widths = {"scipy": 4.2, "cython + Python events": 2.4, "cython + native events": 1.0}
    alphas = {"scipy": 0.5, "cython + Python events": 0.75, "cython + native events": 1.0}
    n_exact = 0
    for z, (name, st) in enumerate(EVENT_RUN_STYLE.items()):
        errs = np.concatenate([np.abs(np.asarray(f) - e)
                               for f, e in zip(out[name]["t_events"], sc.expected_times) if len(f) == len(e)])
        times = np.concatenate([np.asarray(e) for f, e in zip(out[name]["t_events"], sc.expected_times)
                                if len(f) == len(e)])
        order = np.argsort(times)
        keep = errs[order] > 0            # exact hits (e.g. the event at t0) can't go on a log axis
        n_exact = max(n_exact, int(np.sum(~keep)))
        ax.plot(times[order][keep], errs[order][keep], color=st["color"], linestyle=st["linestyle"],
                linewidth=widths[name], alpha=alphas[name], zorder=2 + z, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("exact event time")
    ax.set_ylabel("|t_found − t_exact|")
    title = f"Event-time error (DOP853, rtol={DETAIL_RTOL:g})"
    ax.set_title(title)
    if n_exact:
        ax.annotate(f"{n_exact} event(s) located exactly (not shown on log axis)", (0.02, 0.97),
                    xycoords="axes fraction", va="top", color=INK_2, fontsize=7)
    ax.legend(loc="lower right")

    ax = axes[2]
    names = list(EVENT_RUN_STYLE)
    vals = [out[n]["time"] * 1e3 for n in names]
    ax.bar(np.arange(3), vals, 0.6, color=[EVENT_RUN_STYLE[n]["color"] for n in names])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3g} ms", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(np.arange(3), ["SciPy\n(Python events)", "Cython\n(Python events)", "Cython\n(native events)"])
    ax.set_yscale("log")
    ax.set_ylabel("wall time (ms, log)")
    ax.set_title("Wall time")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.2)
    fig.suptitle(sc.title, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# Tables
# =============================================================================

def write_tables(results, event_outs, out_dir):
    rows = []
    for r in results:
        p = r.problem
        for m in METHODS:
            row = dict(problem=p.name, group=p.group, n=p.n, method=m)
            for e in ENGINES:
                b = r.bench.get((e, m))
                if b is None:
                    continue
                key = e.replace(" + compiled RHS", "_compiled").replace(" ", "_")
                row[f"{key}_ms"] = b["time"] * 1e3
                row[f"{key}_nfev"] = b["nfev"]
                row[f"{key}_error"] = b["err"]
            row["speedup"] = r.bench[("scipy", m)]["time"] / r.bench[("cython", m)]["time"]
            rows.append(row)
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Benchmark suite results", "",
             f"Wall times are best-of-several, at rtol={BENCH_RTOL:g} and atol={_atol(BENCH_RTOL):g}. "
             "Error is the max-norm end-point error relative to max(|y0|, |y_end|) "
             "(blow-up: error at t = 0.9).", "",
             "| problem | n | method | SciPy ms | SciPy (compiled RHS) ms | Cython ms | speedup | nfev SciPy / Cython | error SciPy / Cython |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        comp = f"{row['scipy_compiled_ms']:.3f}" if "scipy_compiled_ms" in row else "–"
        lines.append(
            f"| {row['problem']} | {row['n']} | {row['method']} | {row['scipy_ms']:.3f} | {comp} | "
            f"{row['cython_ms']:.4f} | {row['speedup']:,.0f}× | {row['scipy_nfev']} / {row['cython_nfev']} | "
            f"{row['scipy_error']:.1e} / {row['cython_error']:.1e} |")
    lines += ["", "## Event scenarios (DOP853, rtol=1e-8)", "",
              "| scenario | SciPy ms | Cython + Python events ms | Cython + native events ms | events found |",
              "|---|---:|---:|---:|---|"]
    for out in event_outs:
        sc = out["scenario"]
        counts = ", ".join(f"{lab}: {len(t)}" for lab, t in zip(sc.event_labels, out["cython + native events"]["t_events"]))
        lines.append(f"| {sc.title} | {out['scipy']['time'] * 1e3:.3f} | {out['cython + Python events']['time'] * 1e3:.3f} | "
                     f"{out['cython + native events']['time'] * 1e3:.3f} | {counts} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


# =============================================================================
# CLI
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--problems", nargs="*", help="problem names (default: all)")
    ap.add_argument("--quick", action="store_true", help="fewer tolerances, skip the compiled-RHS SciPy engine")
    ap.add_argument("--no-events", action="store_true", help="skip event scenarios")
    args = ap.parse_args(argv)

    _style()
    problems = catalog.all_problems() if not args.problems else [catalog.get(n) for n in args.problems]
    for group in catalog.GROUPS:
        (args.out / group).mkdir(parents=True, exist_ok=True)
    (args.out / "events").mkdir(parents=True, exist_ok=True)

    results = []
    for p in problems:
        t0 = time.perf_counter()
        res = run_problem(p, quick=args.quick)
        plot_problem(res, args.out / p.group / f"{p.name}.png")
        results.append(res)
        b = res.bench
        print(f"{p.name:20s} DOP853 scipy {b[('scipy', 'DOP853')]['time'] * 1e3:9.2f} ms  "
              f"cython {b[('cython', 'DOP853')]['time'] * 1e3:8.3f} ms  "
              f"speedup {b[('scipy', 'DOP853')]['time'] / b[('cython', 'DOP853')]['time']:7.0f}x  "
              f"[{time.perf_counter() - t0:.1f}s]", flush=True)

    event_outs = []
    if not args.no_events:
        for sc in events.all_scenarios():
            out = run_event_scenario(sc)
            plot_event_scenario(out, args.out / "events" / f"{sc.name}.png")
            event_outs.append(out)
            print(f"{sc.name:20s} scipy {out['scipy']['time'] * 1e3:8.2f} ms  native events "
                  f"{out['cython + native events']['time'] * 1e3:8.3f} ms", flush=True)

    plot_summary(results, args.out / "summary_speedup.png")
    write_tables(results, event_outs, args.out)
    print(f"\nWrote figures and tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
