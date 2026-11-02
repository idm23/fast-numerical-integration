"""Smoke test for the benchmark / visualization suite (``fni-suite``)."""

from fast_numerical_integration import suite


def test_suite_runs_on_a_few_problems(tmp_path):
    assert suite.main(["--problems", "A1", "D2", "vdp_mu100", "kepler_long", "--quick", "--out", str(tmp_path)]) == 0
    for rel in ["detest/A1.png", "detest/D2.png", "stability/vdp_mu100.png", "long_time/kepler_long.png",
                "events/kepler_apsides.png", "events/pendulum.png", "events/bouncing_ball.png",
                "summary_speedup.png", "summary.md", "summary.csv"]:
        assert (tmp_path / rel).stat().st_size > 0, rel
    table = (tmp_path / "summary.md").read_text()
    assert "| D2 | 4 | DOP853 |" in table
