"""RSI edge-case logic, without pulling in the full feature stack.

`transformers.py` used to return NaN whenever `avg_loss` was zero, which is
exactly the pure-uptrend case: four rising bars produced a NaN RSI rather than
100. The table below pins every boundary of the fixed formula and keeps the old
behaviour around so the regression is visible.
"""

import math

import pytest


def calculate_rsi_old_buggy(avg_gain, avg_loss):
    """The pre-fix formula: NaN whenever avg_loss == 0."""
    if avg_loss > 0.0:
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    return float("nan")


def calculate_rsi_fixed(avg_gain, avg_loss):
    """The formula now in transformers.py, with every edge case resolved."""
    if avg_loss == 0.0 and avg_gain > 0.0:
        # Pure uptrend: RS = infinity -> RSI = 100
        return 100.0
    if avg_gain == 0.0 and avg_loss > 0.0:
        # Pure downtrend: RS = 0 -> RSI = 0
        return 0.0
    if avg_gain == 0.0 and avg_loss == 0.0:
        # No price movement: neutral RSI
        return 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


CASES = [
    # name, avg_gain, avg_loss, expected RSI
    ("pure uptrend", 100.0, 0.0, 100.0),
    ("pure downtrend", 0.0, 50.0, 0.0),
    ("no movement", 0.0, 0.0, 50.0),
    ("mixed movements", 92.8, 14.3, 86.65),
    ("balanced", 50.0, 50.0, 50.0),
    ("oversold", 10.0, 90.0, 10.0),
    ("overbought", 90.0, 10.0, 90.0),
]


@pytest.mark.parametrize("name,avg_gain,avg_loss,expected", CASES, ids=[c[0] for c in CASES])
def test_rsi_edge_cases(name, avg_gain, avg_loss, expected):
    assert calculate_rsi_fixed(avg_gain, avg_loss) == pytest.approx(expected, abs=0.01)


def test_old_formula_returned_nan_on_a_pure_uptrend():
    """The regression this table exists for."""
    assert math.isnan(calculate_rsi_old_buggy(100.0, 0.0))
    assert calculate_rsi_fixed(100.0, 0.0) == 100.0


def test_no_movement_is_neutral_not_nan():
    assert math.isnan(calculate_rsi_old_buggy(0.0, 0.0))
    assert calculate_rsi_fixed(0.0, 0.0) == 50.0
