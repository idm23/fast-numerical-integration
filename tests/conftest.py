"""Shared pytest configuration.

Tests are grouped by the oracle they check against:

- ``baseline/``  -- the SciPy reference setup and benchmark harness
- ``parity/``    -- step-for-step agreement with :func:`scipy.integrate.solve_ivp`
- ``accuracy/``  -- error against exact solutions and published references
- ``long_time/`` -- conserved quantities and periodicity over long integrations
- ``stability/`` -- stiff, unstable, chaotic and blow-up problems
- ``events/``    -- event location on physical problems and edge cases

Tests marked ``slow`` are skipped unless ``--runslow`` is given.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
