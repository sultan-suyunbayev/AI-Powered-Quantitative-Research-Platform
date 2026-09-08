import os
import sys
import types
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.getcwd())

try:  # prefer the compiled extension so later tests still see the real module
    import lob_state_cython  # noqa: F401
except ImportError:
    lob_state_stub = types.ModuleType("lob_state_cython")
    lob_state_stub.N_FEATURES = 1
    sys.modules["lob_state_cython"] = lob_state_stub


class _Mediator:
    def __init__(self, env):
        self.env = env
        self.exec = None

    def step(self, proto):
        return np.zeros(1), 0.0, False, False, {}

    def reset(self):
        return np.zeros(1, dtype=np.float32), {}

    def _build_observation(self, *, row=None, state=None, mark_price=None):
        """TradingEnv.step() calls this on every step; shape follows the env."""
        return np.zeros(self.env.observation_space.shape, dtype=np.float32)


import trading_patchnew
from trading_patchnew import TradingEnv
from action_proto import ActionProto, ActionType


@pytest.fixture(autouse=True)
def _use_stub_mediator(monkeypatch):
    """Give TradingEnv the lightweight mediator, without touching sys.modules.

    Replacing sys.modules["mediator"] leaked the stub into every later test in
    the same worker process; patching the name TradingEnv bound keeps it here.
    """
    monkeypatch.setattr(trading_patchnew, "Mediator", _Mediator)


def test_feature_timestamp_validation():
    df_bad = pd.DataFrame(
        {
            "ts_ms": [0],
            "open": [1.0],
            "price": [1.0],
            "atr_pct": [0.0],
            "liq_roll": [0.0],
            "feat_ts": [2000],
            "quote_asset_volume": [1.0],
        }
    )
    env = TradingEnv(df_bad, decision_delay_ms=1000)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(ActionProto(ActionType.HOLD, 0.0))

    df_ok = df_bad.copy()
    df_ok["feat_ts"] = [500]
    env_ok = TradingEnv(df_ok, decision_delay_ms=1000)
    env_ok.reset()
    env_ok.step(ActionProto(ActionType.HOLD, 0.0))
