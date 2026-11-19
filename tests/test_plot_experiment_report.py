"""Convergence plots must not confuse generation samples with NFE."""

from experiments.plot_experiment_report import _normalized_curve


def test_normalized_curve_spans_full_progress_for_short_and_long_histories():
    short = _normalized_curve([[10.0, 8.0, 6.0]], points=5)
    long = _normalized_curve([[10.0, 9.0, 8.0, 7.0, 6.0]], points=5)

    assert len(short) == len(long) == 5
    assert short[0] == long[0] == 10.0
    assert short[-1] == long[-1] == 6.0


def test_normalized_curve_averages_runs_at_same_progress():
    curve = _normalized_curve([[10.0, 8.0], [12.0, 9.0, 6.0]], points=3)

    assert curve == [11.0, 9.5, 7.0]
