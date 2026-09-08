import os
import sys
import types
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.getcwd())
# stub minimal optional modules required for import
try:  # prefer the compiled extension so later tests still see the real module
    import lob_state_cython  # noqa: F401
except ImportError:
    lob_state_stub = types.ModuleType("lob_state_cython")
    lob_state_stub.N_FEATURES = 1
    sys.modules["lob_state_cython"] = lob_state_stub


class _Mediator:
    def __init__(self, env):
        self.env = env

    def step(self, proto):
        return np.zeros(1), 0.0, False, False, {}

    def reset(self):
        return np.zeros(1, dtype=np.float32), {}

    def _build_observation(self, *, row=None, state=None, mark_price=None):
        """TradingEnv.step() calls this on every step; shape follows the env."""
        return np.zeros(self.env.observation_space.shape, dtype=np.float32)


import trading_patchnew
from trading_patchnew import TradingEnv


@pytest.fixture(autouse=True)
def _use_stub_mediator(monkeypatch):
    """Give TradingEnv the lightweight mediator, without touching sys.modules.

    Replacing sys.modules["mediator"] leaked the stub into every later test in
    the same worker process; patching the name TradingEnv bound keeps it here.
    """
    monkeypatch.setattr(trading_patchnew, "Mediator", _Mediator)


def test_close_not_in_observation():
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 1],
            "high": [1, 1, 1, 1],
            "low": [1, 1, 1, 1],
            "close": [100.0, 101.0, 102.0, 103.0],
            "price": [1, 1, 1, 1],
            "quote_asset_volume": [1, 1, 1, 1],
        }
    )
    env = TradingEnv(df)
    obs, _ = env.reset()
    assert not np.isclose(obs, 100.0).any()
    expected = pd.Series([np.nan, 100.0, 101.0, 102.0], name="close")
    pd.testing.assert_series_equal(env.df["close"], expected)
