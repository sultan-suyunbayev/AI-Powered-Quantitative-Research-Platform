"""
Test NaN handling for external features.

ISSUE #2 FIX: Verify that NaN values in external features are converted to default
values (typically 0.0) with proper logging when enabled.

This test validates the current behavior (NaN → 0.0) and documents the design
decision. Future enhancement would add validity flags to distinguish missing data
from zero values.
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import logging


def test_get_safe_float_nan_handling():
    """Test that _get_safe_float converts NaN to default value."""
    from mediator import Mediator

    # Mock row with NaN value
    row = {"cvd_24h": float("nan"), "garch_14d": 0.5, "ret_12h": None}

    # Test NaN conversion
    result_nan = Mediator._get_safe_float(row, "cvd_24h", default=0.0)
    assert result_nan == 0.0, "NaN should be converted to default (0.0)"
    assert math.isfinite(result_nan), "Result should be finite"

    # Test valid value
    result_valid = Mediator._get_safe_float(row, "garch_14d", default=0.0)
    assert abs(result_valid - 0.5) < 1e-6, "Valid value should be returned as-is"

    # Test None
    result_none = Mediator._get_safe_float(row, "ret_12h", default=0.0)
    assert result_none == 0.0, "None should be converted to default (0.0)"


def test_get_safe_float_inf_handling():
    """Test that _get_safe_float converts Inf/-Inf to default value."""
    from mediator import Mediator

    row = {"pos_inf": float("inf"), "neg_inf": float("-inf"), "valid": 42.0}

    # Positive infinity
    result_pos_inf = Mediator._get_safe_float(row, "pos_inf", default=0.0)
    assert result_pos_inf == 0.0, "Positive infinity should be converted to default"

    # Negative infinity
    result_neg_inf = Mediator._get_safe_float(row, "neg_inf", default=0.0)
    assert result_neg_inf == 0.0, "Negative infinity should be converted to default"

    # Valid value (sanity check)
    result_valid = Mediator._get_safe_float(row, "valid", default=0.0)
    assert result_valid == 42.0, "Valid value should pass through"


def test_get_safe_float_logging_enabled():
    """Test that NaN logging works when log_nan=True."""
    from mediator import Mediator

    row = {"nan_feature": float("nan"), "inf_feature": float("inf")}

    # Capture logs
    with patch("mediator.logger") as mock_logger:
        # NaN with logging enabled
        result = Mediator._get_safe_float(row, "nan_feature", default=0.0, log_nan=True)
        assert result == 0.0
        # Check that warning was logged
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "non-finite value" in warning_message.lower()
        assert "ambiguity" in warning_message.lower()


def test_get_safe_float_logging_disabled():
    """Test that NaN conversion is silent when log_nan=False (default)."""
    from mediator import Mediator

    row = {"nan_feature": float("nan")}

    with patch("mediator.logger") as mock_logger:
        # NaN with logging disabled (default)
        result = Mediator._get_safe_float(row, "nan_feature", default=0.0, log_nan=False)
        assert result == 0.0
        # Should NOT log
        mock_logger.warning.assert_not_called()


def test_get_safe_float_range_validation():
    """Test that range validation works correctly."""
    from mediator import Mediator

    row = {"value": 150.0}

    # Within range
    result_ok = Mediator._get_safe_float(row, "value", default=0.0, min_value=0.0, max_value=200.0)
    assert result_ok == 150.0, "Value within range should pass"

    # Below min
    result_low = Mediator._get_safe_float(row, "value", default=0.0, min_value=200.0)
    assert result_low == 0.0, "Value below min should return default"

    # Above max
    result_high = Mediator._get_safe_float(row, "value", default=0.0, max_value=100.0)
    assert result_high == 0.0, "Value above max should return default"


def test_get_safe_float_range_validation_with_logging():
    """Test that range violations are logged when log_nan=True."""
    from mediator import Mediator

    row = {"value": 150.0}

    with patch("mediator.logger") as mock_logger:
        # Value above max with logging
        result = Mediator._get_safe_float(row, "value", default=0.0, max_value=100.0, log_nan=True)
        assert result == 0.0
        mock_logger.debug.assert_called()
        debug_message = mock_logger.debug.call_args[0][0]
        assert "max_value" in debug_message


def test_nan_and_inf_never_reach_the_observation():
    """A NaN or Inf external column lands as a finite, clipped value.

    obs_builder._clipf is a cdef function; the module exports only
    build_observation_vector and compute_n_features, so the old test importing
    _clipf could never run and always reported "not compiled". Exercise the
    behaviour through the public entry point instead.
    """
    obs_builder = pytest.importorskip("obs_builder")

    import feature_config as fc

    ext_dim = fc.EXT_NORM_DIM
    norm_cols = np.zeros(ext_dim, dtype=np.float32)
    norm_cols[0] = np.nan
    norm_cols[1] = np.inf
    norm_cols[2] = -np.inf
    validity = np.ones(ext_dim, dtype=np.uint8)
    out = np.zeros(fc.N_FEATURES, dtype=np.float32)

    obs_builder.build_observation_vector(
        price=100.0,
        prev_price=100.0,
        log_volume_norm=0.0,
        rel_volume=0.0,
        ma5=100.0,
        ma20=100.0,
        rsi14=50.0,
        macd=0.0,
        macd_signal=0.0,
        momentum=0.0,
        atr=1.0,
        cci=0.0,
        obv=0.0,
        bb_lower=99.0,
        bb_upper=101.0,
        is_high_importance=0.0,
        time_since_event=0.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=1000.0,
        units=0.0,
        signal_pos=0.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols,
        norm_cols_validity=validity,
        enable_validity_flags=True,
        out_features=out,
    )

    offset = 0
    for block in fc.FEATURES_LAYOUT:
        if block["name"] == "external":
            break
        offset += block["size"]

    assert out[offset] == 0.0, "a NaN external column must land as 0.0"
    assert np.all(np.isfinite(out)), "no NaN or Inf may reach the observation"
    assert -3.0 <= out[offset + 1] <= 3.0, "+inf must be clipped into range"
    assert -3.0 <= out[offset + 2] <= 3.0, "-inf must be clipped into range"


def test_semantic_ambiguity_documented():
    """
    Document the semantic ambiguity issue: model cannot distinguish
    missing data (NaN) from zero values.
    """
    from mediator import Mediator

    # Scenario 1: Feature is genuinely zero
    row_zero = {"cvd_24h": 0.0}
    result_zero = Mediator._get_safe_float(row_zero, "cvd_24h", default=0.0)

    # Scenario 2: Feature is missing (NaN)
    row_nan = {"cvd_24h": float("nan")}
    result_nan = Mediator._get_safe_float(row_nan, "cvd_24h", default=0.0)

    # ISSUE #2: Both scenarios produce the same result!
    assert (
        result_zero == result_nan == 0.0
    ), "Genuine zero and missing data are indistinguishable (documented issue)"

    # This is the core problem: model cannot learn special handling for missing data
    # Future fix: Add validity flags like (value, is_valid) tuple


def test_extract_norm_cols_nan_handling():
    """Test that _extract_norm_cols converts NaN features to 0.0."""
    from mediator import Mediator

    # Create mediator instance (minimal setup)
    mediator = Mediator.__new__(Mediator)

    # Mock row with mix of valid and NaN values
    row = {
        "cvd_24h": 1.5,
        "cvd_7d": float("nan"),  # Missing
        "yang_zhang_48h": 0.8,
        "yang_zhang_7d": None,  # Missing
        "garch_200h": float("inf"),  # Invalid (inf)
        "garch_14d": 0.5,
        # ... other features would be default 0.0
    }

    # Since the validity-flag work, _extract_norm_cols returns (values, validity).
    norm_cols, validity = mediator._extract_norm_cols(row)

    # Verify shape: the external block width comes from the layout.
    import feature_config

    assert len(norm_cols) == feature_config.EXT_NORM_DIM
    assert len(validity) == feature_config.EXT_NORM_DIM

    # Verify valid values pass through
    assert abs(norm_cols[0] - 1.5) < 1e-6, "cvd_24h should be 1.5"
    assert abs(norm_cols[2] - 0.8) < 1e-6, "yang_zhang_48h should be 0.8"
    assert abs(norm_cols[5] - 0.5) < 1e-6, "garch_14d should be 0.5"

    # Verify NaN/None/Inf converted to 0.0 and reported as invalid
    assert norm_cols[1] == 0.0, "cvd_7d (NaN) should be 0.0"
    assert norm_cols[3] == 0.0, "yang_zhang_7d (None) should be 0.0"
    assert norm_cols[4] == 0.0, "garch_200h (Inf) should be 0.0"
    assert not validity[1] and not validity[3] and not validity[4]
    assert validity[0] and validity[2] and validity[5]

    # All results should be finite
    assert np.all(np.isfinite(norm_cols)), "All results should be finite"


def test_future_enhancement_roadmap():
    """
    Document the future enhancement roadmap for proper NaN handling.

    This test serves as documentation for future developers.
    """
    # Current behavior (as of Issue #2 fix):
    # - NaN → 0.0 (silent conversion with optional logging)
    # - No validity flags for external features
    # - Semantic ambiguity: missing data looks like zero

    # Future enhancement (requires breaking change):
    # Step 1: Modify _get_safe_float to return tuple
    #   def _get_safe_float(...) -> Tuple[float, bool]:
    #       return (value, is_valid)

    # Step 2: Update _extract_norm_cols to return values + validity
    #   def _extract_norm_cols(...) -> Tuple[np.ndarray, np.ndarray]:
    #       values = np.zeros(21)
    #       validity = np.ones(21, dtype=bool)
    #       return (values, validity)

    # Step 3: Expand observation space
    #   OLD: obs_dim = 62 (current)
    #   NEW: obs_dim = 62 + 21 = 83 (values + validity flags)

    # Step 4: Update obs_builder.pyx to include validity flags
    #   for i in range(21):
    #       out_features[idx] = values[i]
    #       out_features[idx+21] = validity[i]

    # Step 5: Retrain all models
    #   - Models trained before this change will be incompatible
    #   - Need to version models (pre-validity / post-validity)

    # Impact:
    # - Better handling of missing data
    # - Model can learn to ignore/interpolate missing values
    # - More robust to data quality issues
    # - Breaking change requiring retraining

    # TECH DEBT: Validity flags for NaN handling
    # Tracking: Model compatibility matrix needed for pre/post-validity model versions
    # Impact: Models trained before validity flags will be incompatible
    # Control artifact: Model versioning with compatibility metadata in model manifest
    # See: docs/SIMULATION_LIMITATIONS.md for simulation accuracy tracking

    # Verify tech debt is documented in registry
    import os

    registry_path = "docs/reports/TECH_DEBT_REGISTRY.md"
    assert os.path.exists(registry_path), f"Tech debt registry should exist at {registry_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
