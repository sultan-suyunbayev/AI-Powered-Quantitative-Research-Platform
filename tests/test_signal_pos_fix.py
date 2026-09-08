"""
Test suite for signal_pos fix in observation vector.

FIX (2025-11-24): signal_pos was missing from obs_builder observation vector.
This test verifies that signal_pos is correctly included in the observation.

Key changes tested:
1. obs_builder.pyx now accepts signal_pos parameter
2. mediator.py passes signal_pos to build_observation_vector()
3. feature_config.py has agent block size=7 (was 6)
4. N_FEATURES counts signal_pos among the agent features
"""

import pytest
import numpy as np
import sys
import os

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


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSignalPosInObservation:
    """Test signal_pos inclusion in observation vector."""

    def test_feature_config_agent_block_size(self):
        """Agent block should have size=7 (including signal_pos)."""
        from feature_config import FEATURES_LAYOUT, make_layout

        # Rebuild layout to ensure fresh state
        make_layout({})

        # Find agent block
        agent_block = None
        for block in FEATURES_LAYOUT:
            if block["name"] == "agent":
                agent_block = block
                break

        assert agent_block is not None, "Agent block not found in FEATURES_LAYOUT"
        assert (
            agent_block["size"] == 7
        ), f"Agent block size should be 7 (with signal_pos), got {agent_block['size']}"
        assert (
            "signal_pos" in agent_block["description"]
        ), "signal_pos should be mentioned in agent block description"

    def test_n_features_counts_signal_pos(self):
        """signal_pos is one of the agent features, and N_FEATURES is their sum."""
        import feature_config

        layout = feature_config.make_layout({})

        agent_block = next(b for b in layout if b["name"] == "agent")
        assert "signal_pos" in agent_block["description"]
        assert sum(b["size"] for b in layout) == feature_config.N_FEATURES

    def test_obs_builder_accepts_signal_pos(self):
        """obs_builder.build_observation_vector should accept signal_pos parameter."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not compiled (Cython module)")

        from feature_config import N_FEATURES, make_layout

        make_layout({})

        # Create output array
        obs = np.zeros(N_FEATURES, dtype=np.float32)
        norm_cols = np.zeros(_EXT_DIM, dtype=np.float32)
        norm_validity = np.ones(_EXT_DIM, dtype=np.uint8)

        # Call with signal_pos parameter (should not raise)
        signal_pos_value = 0.75
        try:
            build_observation_vector(
                100.0,  # price
                99.0,  # prev_price
                0.1,  # log_volume_norm
                0.5,  # rel_volume
                100.0,  # ma5
                100.0,  # ma20
                50.0,  # rsi14
                0.0,  # macd
                0.0,  # macd_signal
                0.0,  # momentum
                1.0,  # atr
                0.0,  # cci
                0.0,  # obv
                95.0,  # bb_lower
                105.0,  # bb_upper
                0.0,  # is_high_importance
                0.0,  # time_since_event
                50.0,  # fear_greed_value
                True,  # has_fear_greed
                False,  # risk_off_flag
                1000.0,  # cash
                10.0,  # units
                signal_pos_value,  # signal_pos (NEW)
                0.0,  # last_vol_imbalance
                0.0,  # last_trade_intensity
                0.0,  # last_realized_spread
                0.0,  # last_agent_fill_ratio
                0,  # token_id
                1,  # max_num_tokens
                1,  # num_tokens
                norm_cols,
                norm_validity,
                True,  # enable_validity_flags
                obs,
            )
        except TypeError as e:
            pytest.fail(f"build_observation_vector should accept signal_pos: {e}")

        # Verify signal_pos is in observation (index 29 in agent block)
        # Agent block starts at index 23, signal_pos is the 7th element (index 29)
        signal_pos_index = 29
        assert abs(obs[signal_pos_index] - signal_pos_value) < 0.01, (
            f"signal_pos at index {signal_pos_index} should be ~{signal_pos_value}, "
            f"got {obs[signal_pos_index]}"
        )

    def test_signal_pos_clipping(self):
        """signal_pos should be clipped to [-1.0, 1.0] range."""
        try:
            from obs_builder import build_observation_vector
        except ImportError:
            pytest.skip("obs_builder not compiled (Cython module)")

        from feature_config import N_FEATURES, make_layout

        make_layout({})

        obs = np.zeros(N_FEATURES, dtype=np.float32)
        norm_cols = np.zeros(_EXT_DIM, dtype=np.float32)
        norm_validity = np.ones(_EXT_DIM, dtype=np.uint8)

        # Test with out-of-range value
        build_observation_vector(
            100.0,
            99.0,
            0.1,
            0.5,
            100.0,
            100.0,
            50.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            95.0,
            105.0,
            0.0,
            0.0,
            50.0,
            True,
            False,
            1000.0,
            10.0,
            2.5,  # signal_pos > 1.0 (should be clipped)
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            1,
            1,
            norm_cols,
            norm_validity,
            True,
            obs,
        )

        signal_pos_index = 29
        assert (
            obs[signal_pos_index] <= 1.0
        ), f"signal_pos should be clipped to <= 1.0, got {obs[signal_pos_index]}"

    def test_mediator_passes_signal_pos(self):
        """Mediator should pass signal_pos to observation builder."""
        # Read the source file directly to avoid import dependencies
        import os

        mediator_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mediator.py"
        )

        with open(mediator_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Check that _build_observation method uses signal_pos
        assert "signal_pos" in source, "_build_observation should reference signal_pos"
        assert (
            "_last_signal_position" in source
        ), "_build_observation should get signal position from _last_signal_position"
        # Verify signal_pos is passed to build_observation_vector
        assert (
            "float(signal_pos)" in source
        ), "signal_pos should be passed to build_observation_vector as float"


class TestFeatureLayoutConsistency:
    """Test that feature layout is consistent after signal_pos addition."""

    def test_layout_block_order(self):
        """Verify block order is correct after signal_pos addition."""
        from feature_config import FEATURES_LAYOUT, make_layout

        make_layout({})

        # Order follows obs_builder.build_observation_vector_c, which writes the
        # externals, then the token metadata and one-hot, and only then the
        # validity flags.  The declaration used to put external_validity right
        # after external, which put every later block at the wrong offset.
        expected_order = [
            "bar",
            "ma5",
            "ma20",
            "indicators",
            "derived",
            "agent",  # 7 elements, signal_pos included
            "microstructure",
            "bb_context",
            "metadata",
            "external",
            "token_meta",
            "token",
            "external_validity",
        ]

        actual_order = [block["name"] for block in FEATURES_LAYOUT]
        assert (
            actual_order == expected_order
        ), f"Block order mismatch.\nExpected: {expected_order}\nActual: {actual_order}"

    def test_total_features_count(self):
        """Verify total feature count is correct."""
        from feature_config import FEATURES_LAYOUT, N_FEATURES, make_layout

        make_layout({})

        total = sum(block["size"] for block in FEATURES_LAYOUT)
        assert (
            total == N_FEATURES
        ), f"Sum of block sizes ({total}) should equal N_FEATURES ({N_FEATURES})"
        # The external block carries one validity flag per feature.
        sizes = {block["name"]: block["size"] for block in FEATURES_LAYOUT}
        assert sizes["external_validity"] == sizes["external"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
