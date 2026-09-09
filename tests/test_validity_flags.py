"""
Simple test to verify all 8 validity flags work correctly.

This test ensures that:
1. All validity flags are at correct indices
2. Flags are 1.0 when indicators are valid (not NaN)
3. Flags are 0.0 when indicators are NaN
4. Fallback values are used correctly when NaN
"""

import pytest
import numpy as np

import feature_config as _fc

# Sizes come from the layout rather than being spelled out: obs_builder writes
# through typed memoryviews with bounds checking off, so a buffer that is too
# short corrupts memory instead of raising.
_N_FEATURES = _fc.N_FEATURES
_EXT_DIM = next(b["size"] for b in _fc.FEATURES_LAYOUT if b["name"] == "external")
_MAX_TOKENS = next(b["size"] for b in _fc.FEATURES_LAYOUT if b["name"] == "token")


def _block_start(name):
    """First index of a named block in the current feature layout."""
    offset = 0
    for block in _fc.FEATURES_LAYOUT:
        if block["name"] == name:
            return offset
        offset += block["size"]
    raise KeyError(name)


@pytest.fixture
def get_obs_builder():
    """Import obs_builder or skip test."""
    try:
        from obs_builder import build_observation_vector

        return build_observation_vector
    except ImportError:
        pytest.skip("obs_builder not compiled")


def create_base_params():
    """Create base parameters for observation building."""
    return {
        "price": 50000.0,
        "prev_price": 49900.0,
        "log_volume_norm": 0.5,
        "rel_volume": 0.5,
        "atr": 100.0,
        "bb_lower": 49000.0,
        "bb_upper": 51000.0,
        "is_high_importance": 0.0,
        "time_since_event": 0.0,
        "fear_greed_value": 50.0,
        "has_fear_greed": True,
        "risk_off_flag": False,
        "cash": 10000.0,
        "units": 0.5,
        "signal_pos": 0.0,
        "last_vol_imbalance": 0.1,
        "last_trade_intensity": 5.0,
        "last_realized_spread": 0.001,
        "last_agent_fill_ratio": 0.95,
        "token_id": 0,
        "max_num_tokens": _MAX_TOKENS,
        "num_tokens": _MAX_TOKENS,
        "norm_cols_values": np.zeros(_EXT_DIM, dtype=np.float32),
        "norm_cols_validity": np.ones(_EXT_DIM, dtype=np.uint8),
        "enable_validity_flags": True,
        "out_features": np.zeros(_N_FEATURES, dtype=np.float32),
    }


class TestValidityFlags:
    """Test all 8 validity flags (ma5, ma20, rsi, macd, macd_signal, momentum, cci, obv)."""

    def test_ma5_validity_flag(self, get_obs_builder):
        """Test ma5 and ma5_valid flags (indices 3-4)."""
        build_observation_vector = get_obs_builder

        # Test 1: Valid ma5
        params = create_base_params()
        params.update(
            {
                "ma5": 50100.0,  # Valid value
                "ma20": np.nan,
                "rsi14": np.nan,
                "macd": np.nan,
                "macd_signal": np.nan,
                "momentum": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        assert params["out_features"][3] == 50100.0, "ma5 value should be preserved"
        assert params["out_features"][4] == 1.0, "ma5_valid should be 1.0 when valid"

        # Test 2: NaN ma5
        params = create_base_params()
        params.update(
            {
                "ma5": np.nan,  # NaN
                "ma20": np.nan,
                "rsi14": np.nan,
                "macd": np.nan,
                "macd_signal": np.nan,
                "momentum": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        assert params["out_features"][3] == 0.0, "ma5 fallback should be 0.0"
        assert params["out_features"][4] == 0.0, "ma5_valid should be 0.0 when NaN"

    def test_rsi_validity_flag(self, get_obs_builder):
        """Test rsi14 and rsi_valid flags (indices 7-8)."""
        build_observation_vector = get_obs_builder

        # Test 1: Valid RSI
        params = create_base_params()
        params.update(
            {
                "ma5": np.nan,
                "ma20": np.nan,
                "rsi14": 55.0,  # Valid neutral RSI
                "macd": np.nan,
                "macd_signal": np.nan,
                "momentum": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        assert params["out_features"][7] == 55.0, "rsi14 value should be preserved"
        assert params["out_features"][8] == 1.0, "rsi_valid should be 1.0 when valid"

        # Test 2: NaN RSI (warmup period)
        params = create_base_params()
        params.update(
            {
                "ma5": np.nan,
                "ma20": np.nan,
                "rsi14": np.nan,  # NaN
                "macd": np.nan,
                "macd_signal": np.nan,
                "momentum": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        assert params["out_features"][7] == 50.0, "rsi14 fallback should be 50.0"
        assert params["out_features"][8] == 0.0, "rsi_valid should be 0.0 when NaN"

    def test_macd_validity_flags(self, get_obs_builder):
        """Test macd, macd_signal and their validity flags (indices 9-12)."""
        build_observation_vector = get_obs_builder

        # Test: Both valid
        params = create_base_params()
        params.update(
            {
                "ma5": np.nan,
                "ma20": np.nan,
                "rsi14": np.nan,
                "macd": 10.5,
                "macd_signal": 8.2,
                "momentum": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        assert params["out_features"][9] == 10.5, "macd value preserved"
        assert params["out_features"][10] == 1.0, "macd_valid = 1.0"
        # float32 buffer: 8.2 is not exactly representable
        assert params["out_features"][11] == pytest.approx(8.2), "macd_signal value preserved"
        assert params["out_features"][12] == 1.0, "macd_signal_valid = 1.0"

        # Test: Both NaN
        params = create_base_params()
        params.update(
            {
                "ma5": np.nan,
                "ma20": np.nan,
                "rsi14": np.nan,
                "macd": np.nan,
                "macd_signal": np.nan,
                "momentum": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        assert params["out_features"][9] == 0.0, "macd fallback = 0.0"
        assert params["out_features"][10] == 0.0, "macd_valid = 0.0"
        assert params["out_features"][11] == 0.0, "macd_signal fallback = 0.0"
        assert params["out_features"][12] == 0.0, "macd_signal_valid = 0.0"

    def test_all_validity_flags_indices(self, get_obs_builder):
        """Verify all 8 validity flag indices are correct."""
        build_observation_vector = get_obs_builder

        # Set all indicators to valid values
        params = create_base_params()
        params.update(
            {
                "ma5": 50100.0,
                "ma20": 50200.0,
                "rsi14": 55.0,
                "macd": 10.0,
                "macd_signal": 8.0,
                "momentum": 5.0,
                "cci": 20.0,
                "obv": 1000.0,
            }
        )
        build_observation_vector(**params)

        # Check all validity flags are 1.0
        # Derived from the layout: ma5 and ma20 are (value, valid) pairs, and the
        # indicators block is seven more such pairs in the order named in its
        # description. Spelling the numbers out is what let this map drift --
        # atr's pair was missing entirely, which pushed cci and obv down by one.
        _ind = _block_start("indicators")
        validity_indices = {
            _block_start("ma5") + 1: "ma5_valid",
            _block_start("ma20") + 1: "ma20_valid",
            _ind + 1: "rsi_valid",
            _ind + 3: "macd_valid",
            _ind + 5: "macd_signal_valid",
            _ind + 7: "momentum_valid",
            _ind + 9: "atr_valid",
            _ind + 11: "cci_valid",
            _ind + 13: "obv_valid",
        }

        for idx, name in validity_indices.items():
            assert (
                params["out_features"][idx] == 1.0
            ), f"{name} at index {idx} should be 1.0, got {params['out_features'][idx]}"

    def test_warmup_period_simulation(self, get_obs_builder):
        """Simulate early bars where all indicators are NaN."""
        build_observation_vector = get_obs_builder

        # Bar 1: Nothing is ready. atr belongs in this list -- the assertion
        # loop below covers atr_valid, and leaving atr at its real value left
        # that one flag set while every other one was cleared.
        params = create_base_params()
        params.update(
            {
                "ma5": np.nan,
                "ma20": np.nan,
                "rsi14": np.nan,
                "macd": np.nan,
                "macd_signal": np.nan,
                "momentum": np.nan,
                "atr": np.nan,
                "cci": np.nan,
                "obv": np.nan,
            }
        )
        build_observation_vector(**params)

        # All validity flags should be 0.0
        _ind = _block_start("indicators")
        validity_indices = [
            _block_start("ma5") + 1,
            _block_start("ma20") + 1,
        ] + [_ind + k for k in (1, 3, 5, 7, 9, 11, 13)]
        for idx in validity_indices:
            assert (
                params["out_features"][idx] == 0.0
            ), f"Validity flag at index {idx} should be 0.0 during warmup"

        # Check fallback values
        assert params["out_features"][_block_start("ma5")] == 0.0, "ma5 fallback"
        assert params["out_features"][_block_start("ma20")] == 0.0, "ma20 fallback"
        assert params["out_features"][_ind] == 50.0, "rsi14 fallback (neutral)"
        assert params["out_features"][_ind + 2] == 0.0, "macd fallback"
        assert params["out_features"][_ind + 4] == 0.0, "macd_signal fallback"
        assert params["out_features"][_ind + 6] == 0.0, "momentum fallback"
        assert params["out_features"][_ind + 10] == 0.0, "cci fallback"
        assert params["out_features"][_ind + 12] == 0.0, "obv fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
