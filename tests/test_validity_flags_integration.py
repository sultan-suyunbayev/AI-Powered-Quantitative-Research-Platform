"""Integration tests for validity flags in observation space.

Tests that validity flags are correctly integrated into the observation vector
and that the observation dimension is correctly computed.
"""

import numpy as np
import pytest

import feature_config as _fc

# Sizes come from the layout rather than being spelled out: obs_builder writes
# through typed memoryviews with bounds checking off, so a buffer that is too
# short corrupts memory instead of raising.
_N_FEATURES = _fc.N_FEATURES
_EXT_DIM = next(b["size"] for b in _fc.FEATURES_LAYOUT if b["name"] == "external")


def _block_start(name):
    """First index of a named block in the current feature layout."""
    offset = 0
    for block in _fc.FEATURES_LAYOUT:
        if block["name"] == name:
            return offset
        offset += block["size"]
    raise KeyError(name)


try:
    import obs_builder as _obs_builder

    _HAVE_OBS_BUILDER = hasattr(_obs_builder, "build_observation_vector")
except Exception:  # pragma: no cover - exercised only without the compiled module
    _obs_builder = None
    _HAVE_OBS_BUILDER = False

requires_obs_builder = pytest.mark.skipif(
    not _HAVE_OBS_BUILDER,
    reason="obs_builder not compiled (run `python setup.py build_ext --inplace`)",
)


def _build_obs(norm_cols_values, norm_cols_validity, out=None):
    """Call the builder with keywords and a correctly sized output buffer.

    Every argument is named: the signature has grown (``signal_pos``,
    ``norm_cols_validity``, ``enable_validity_flags``) and positional calls
    silently shift values into the wrong parameters.
    """
    if out is None:
        out = np.zeros(_N_FEATURES, dtype=np.float32)
    _obs_builder.build_observation_vector(
        price=1000.0,
        prev_price=990.0,
        log_volume_norm=0.5,
        rel_volume=1.2,
        ma5=1000.0,
        ma20=995.0,
        rsi14=50.0,
        macd=0.1,
        macd_signal=0.05,
        momentum=0.02,
        atr=10.0,
        cci=0.0,
        obv=1000.0,
        bb_lower=980.0,
        bb_upper=1020.0,
        is_high_importance=0.0,
        time_since_event=1.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=0.0,
        signal_pos=0.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=1.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols_values,
        norm_cols_validity=norm_cols_validity,
        enable_validity_flags=True,
        out_features=out,
    )
    return out


def _validity_slice(obs):
    start = _block_start("external_validity")
    return obs[start : start + _EXT_DIM]


def test_feature_layout_includes_validity_flags():
    """The layout carries exactly one validity flag per external feature."""
    from feature_config import make_layout

    layout = make_layout()

    validity_blocks = [b for b in layout if b["name"] == "external_validity"]
    assert len(validity_blocks) == 1, "Should have exactly one external_validity block"

    validity_block = validity_blocks[0]
    assert validity_block["size"] == _EXT_DIM, (
        f"Validity block has {validity_block['size']} entries but the external block "
        f"has {_EXT_DIM}: one flag per external feature is required"
    )
    assert validity_block["source"] == "external"

    assert sum(b["size"] for b in layout) == _N_FEATURES


def test_observation_dim_with_validity_flags():
    """compute_n_features and the layout sum must agree."""
    from feature_config import make_layout

    layout = make_layout()
    expected = sum(b["size"] for b in layout)

    if _HAVE_OBS_BUILDER:
        assert _obs_builder.compute_n_features(layout) == expected

    assert expected == _N_FEATURES


def test_observation_dim_backward_compatibility():
    """The base blocks plus the validity block account for every feature."""
    from feature_config import make_layout

    layout = make_layout()
    sizes = {b["name"]: b["size"] for b in layout}

    base = sum(size for name, size in sizes.items() if name != "external_validity")
    assert base + sizes["external_validity"] == _N_FEATURES
    assert sizes["external_validity"] == sizes["external"]


@requires_obs_builder
def test_validity_flags_in_observation_vector():
    """Validity flags land at the positions the layout declares for them."""
    norm_cols_values = np.arange(1.0, _EXT_DIM + 1.0, dtype=np.float32)
    norm_cols_values[2] = np.nan  # yang_zhang_48h
    norm_cols_values[5] = np.nan  # garch_14d

    norm_cols_validity = np.ones(_EXT_DIM, dtype=np.uint8)
    norm_cols_validity[2] = 0
    norm_cols_validity[5] = 0

    obs = _build_obs(norm_cols_values, norm_cols_validity)
    validity_flags = _validity_slice(obs)

    assert validity_flags[2] == 0.0, "yang_zhang_48h is invalid"
    assert validity_flags[5] == 0.0, "garch_14d is invalid"

    for i in range(_EXT_DIM):
        if i in (2, 5):
            continue
        assert (
            validity_flags[i] == 1.0
        ), f"Expected validity_flags[{i}]=1.0, got {validity_flags[i]}"

    assert np.all(
        (validity_flags == 0.0) | (validity_flags == 1.0)
    ), f"Validity flags should be binary, got {validity_flags}"

    assert np.all(np.isfinite(obs)), "A NaN input must not leak into the observation"


@requires_obs_builder
def test_nan_feature_sets_validity_false():
    """A NaN feature is reported as invalid and does not poison the vector."""
    norm_cols_values = np.zeros(_EXT_DIM, dtype=np.float32)
    norm_cols_values[0] = np.nan  # cvd_24h

    norm_cols_validity = np.ones(_EXT_DIM, dtype=np.uint8)
    norm_cols_validity[0] = 0

    obs = _build_obs(norm_cols_values, norm_cols_validity)
    validity_flags = _validity_slice(obs)

    assert validity_flags[0] == 0.0, "cvd_24h validity flag should be 0.0"
    assert np.all(validity_flags[1:] == 1.0), f"Expected the rest valid, got {validity_flags}"
    assert np.all(np.isfinite(obs))


@requires_obs_builder
def test_valid_feature_sets_validity_true():
    """With every external feature present, every flag is set."""
    norm_cols_values = np.arange(1.0, _EXT_DIM + 1.0, dtype=np.float32)
    norm_cols_validity = np.ones(_EXT_DIM, dtype=np.uint8)

    obs = _build_obs(norm_cols_values, norm_cols_validity)
    validity_flags = _validity_slice(obs)

    assert np.all(validity_flags == 1.0), f"Expected all flags 1.0, got {validity_flags}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
