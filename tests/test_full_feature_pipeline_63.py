"""
Комплексный end-to-end тест системы признаков AI-Powered Quantitative Research Platform.

Проверяет полный цикл прохождения данных:
1. Создание 24 технических признаков в transformers.py
2. Загрузка 21 признака в mediator.py через _extract_norm_cols
3. Построение 62 признаков в obs_builder.pyx
4. Правильная нормализация и отсутствие NaN/Inf
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from feature_config import make_layout, EXT_NORM_DIM, N_FEATURES

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


def _configure_risk_attrs(mock_env):
    """Mediator coerces these env attributes to float/int at construction.

    A bare Mock() auto-creates them as Mock objects, which float() rejects, so
    spell out the numeric defaults the real environment carries.
    """
    mock_env.max_abs_position = 1e12
    mock_env.max_notional = 2e12
    mock_env.max_drawdown_pct = 1.0
    mock_env.intrabar_dd_pct = 0.30
    mock_env.dd_window = 500
    mock_env.bankruptcy_cash_th = -1e12
    # A Mock run_config would make rate_limit a Mock and blow up the comparison.
    mock_env.run_config = None
    return mock_env


def test_ext_norm_dim_matches_layout():
    """EXT_NORM_DIM и блок external раскладки обязаны совпадать."""
    assert EXT_NORM_DIM == _EXT_DIM, (
        f"EXT_NORM_DIM={EXT_NORM_DIM} расходится с блоком external={_EXT_DIM}: "
        "obs_builder пишет по раскладке, mediator — по константе"
    )


def test_n_features_matches_layout_sum():
    """N_FEATURES должен быть суммой блоков, а не отдельно поддерживаемым числом."""
    from feature_config import FEATURES_LAYOUT
    from feature_config import N_FEATURES as computed_features

    assert computed_features == sum(b["size"] for b in FEATURES_LAYOUT)
    assert computed_features == _N_FEATURES


def test_feature_layout_structure():
    """Структурные инварианты раскладки, которые обязаны держаться при любом росте."""
    from feature_config import FEATURES_LAYOUT

    names = [block["name"] for block in FEATURES_LAYOUT]
    assert len(names) == len(set(names)), f"дубли блоков в раскладке: {names}"

    required = {
        "bar",
        "ma5",
        "ma20",
        "indicators",
        "derived",
        "agent",
        "microstructure",
        "bb_context",
        "metadata",
        "external",
        "external_validity",
        "token_meta",
        "token",
    }
    assert required.issubset(set(names)), f"в раскладке нет блоков {required - set(names)}"

    sizes = {block["name"]: block["size"] for block in FEATURES_LAYOUT}
    # На каждый внешний признак приходится ровно один флаг валидности.
    assert sizes["external_validity"] == sizes["external"]
    assert all(size > 0 for size in sizes.values()), f"пустой блок в раскладке: {sizes}"
    assert sum(sizes.values()) == _N_FEATURES

    # Блоки идут подряд, без дыр: смещение последнего + его размер = N_FEATURES.
    assert _block_start(names[-1]) + sizes[names[-1]] == _N_FEATURES


def test_mediator_extract_norm_cols_size():
    """Проверка что _extract_norm_cols возвращает массив размера 21"""
    from mediator import Mediator

    # Создаем mock environment
    mock_env = Mock()
    mock_env.state = Mock(units=0.0, cash=10000.0, max_position=1.0)
    mock_env.lob = None

    mediator = Mediator(_configure_risk_attrs(mock_env), event_level=0)

    # Создаем mock row с всеми 24 техническими признаками (обновлено для 4h таймфрейма)
    mock_row = pd.Series(
        {
            # Original 8
            "cvd_24h": 1000.0,
            "cvd_7d": 5000.0,  # было cvd_168h
            "yang_zhang_48h": 0.05,  # было yang_zhang_24h
            "yang_zhang_7d": 0.08,  # было yang_zhang_168h
            "garch_200h": 0.03,  # было garch_12h (42 бара = 10080 мин = 7d, минимум для GARCH на 4h)
            "garch_14d": 0.04,  # было garch_24h
            "ret_12h": 0.001,  # было ret_15m
            "ret_24h": 0.005,  # было ret_60m
            # Additional 8 (43->51)
            "ret_4h": 0.0005,  # было ret_5m
            "sma_12000": 50000.0,  # было sma_60 (50 баров = 12000 минут = 200h для 4h)
            "yang_zhang_30d": 0.12,  # было yang_zhang_720h
            "parkinson_48h": 0.06,  # было parkinson_24h
            "parkinson_7d": 0.09,  # было parkinson_168h
            "garch_30d": 0.025,  # было garch_500m
            "taker_buy_ratio": 0.52,
            "taker_buy_ratio_sma_24h": 0.51,
            # Additional 5 (51->63) - НОВЫЕ (обновлено для 4h)
            "taker_buy_ratio_sma_8h": 0.53,  # было 6h
            "taker_buy_ratio_sma_16h": 0.52,  # было 12h
            "taker_buy_ratio_momentum_4h": 0.01,  # было 1h
            "taker_buy_ratio_momentum_8h": 0.02,  # было 6h
            "taker_buy_ratio_momentum_12h": 0.015,  # без изменений
        }
    )

    norm_cols, norm_cols_validity = mediator._extract_norm_cols(mock_row)

    # Проверка размера
    assert (
        norm_cols.shape[0] == _EXT_DIM
    ), f"Expected {_EXT_DIM} norm_cols, got {norm_cols.shape[0]}"
    assert norm_cols_validity.shape[0] == _EXT_DIM

    # Проверка типа
    assert norm_cols.dtype == np.float32, f"Expected float32, got {norm_cols.dtype}"

    # Проверка что все значения не NaN и не Inf (до нормализации)
    assert np.all(np.isfinite(norm_cols)), "norm_cols contains NaN or Inf"

    # Проверка что значения НЕ нормализованы через tanh (должны быть исходные значения)
    # Например, cvd_24h=1000.0 не должен стать ~1.0 после tanh
    assert (
        norm_cols[0] > 10.0
    ), "norm_cols[0] (cvd_24h) seems to be already normalized, but should be raw"

    return norm_cols


def test_mediator_norm_cols_no_double_tanh():
    """Проверка что norm_cols НЕ проходят двойную нормализацию через tanh"""
    from mediator import Mediator

    mock_env = Mock()
    mock_env.state = Mock(units=0.0, cash=10000.0, max_position=1.0)
    mock_env.lob = None

    mediator = Mediator(_configure_risk_attrs(mock_env), event_level=0)

    # Большое значение для проверки
    mock_row = pd.Series(
        {
            "cvd_24h": 1000.0,  # Должно остаться 1000.0, а не tanh(1000.0)≈1.0
            **{f"feature_{i}": 0.0 for i in range(20)},  # Заполнение остальных
        }
    )

    norm_cols, _ = mediator._extract_norm_cols(mock_row)

    # Если бы применялся tanh, значение было бы близко к 1.0
    # Без tanh значение должно остаться 1000.0
    assert (
        norm_cols[0] > 100.0
    ), f"Expected raw value ~1000, got {norm_cols[0]} (seems like tanh was applied)"


def test_obs_builder_applies_tanh():
    """Проверка что obs_builder применяет tanh к norm_cols"""
    try:
        from obs_builder import build_observation_vector
    except ImportError:
        pytest.skip("obs_builder not compiled, skipping")

    # Создаем norm_cols с большими значениями
    norm_cols = np.array([1000.0] * _EXT_DIM, dtype=np.float32)

    # Создаем output array
    out = np.zeros(_N_FEATURES, dtype=np.float32)

    # Вызываем build_observation_vector с минимальными параметрами
    build_observation_vector(
        price=50000.0,
        prev_price=49900.0,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=50100.0,
        ma20=50200.0,
        rsi14=50.0,
        macd=0.0,
        macd_signal=0.0,
        momentum=0.0,
        atr=100.0,
        cci=0.0,
        obv=0.0,
        bb_lower=49000.0,
        bb_upper=51000.0,
        is_high_importance=0.0,
        time_since_event=0.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=0.0,
        signal_pos=0.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=_MAX_TOKENS,
        num_tokens=_MAX_TOKENS,
        norm_cols_values=norm_cols,
        norm_cols_validity=np.ones(_EXT_DIM, dtype=np.uint8),
        enable_validity_flags=True,
        out_features=out,
    )

    # Смещение блока берём из раскладки, а не фиксируем числом.
    external_start = _block_start("external")
    external_end = external_start + _EXT_DIM

    external_features = out[external_start:external_end]

    # После tanh все значения должны быть в диапазоне (-1, 1)
    assert np.all(
        external_features >= -1.0
    ), f"Some external features < -1: {external_features[external_features < -1.0]}"
    assert np.all(
        external_features <= 1.0
    ), f"Some external features > 1: {external_features[external_features > 1.0]}"

    # tanh(1000) ≈ 1.0, так что все должны быть близки к 1.0
    assert np.all(
        external_features > 0.99
    ), f"Expected ~1.0 after tanh(1000), got min={external_features.min()}"


def test_full_pipeline_integration():
    """Интеграционный тест полного цикла с реальным DataFrame"""
    try:
        from obs_builder import build_observation_vector
    except ImportError:
        pytest.skip("obs_builder not compiled, skipping")

    from mediator import Mediator

    # Создаем mock DataFrame с полным набором признаков (обновлено для 4h таймфрейма)
    df_data = {
        "timestamp": [1000000, 1000060, 1000120],
        "open": [50000.0, 50100.0, 50200.0],
        "high": [50200.0, 50300.0, 50400.0],
        "low": [49800.0, 49900.0, 50000.0],
        "close": [50100.0, 50200.0, 50300.0],
        "volume": [10.0, 12.0, 11.0],
        "quote_asset_volume": [500000.0, 600000.0, 550000.0],
        # Технические признаки
        "sma_5": [50000.0, 50050.0, 50100.0],
        "sma_15": [49900.0, 49950.0, 50000.0],
        "sma_12000": [49800.0, 49850.0, 49900.0],  # было sma_60 (50 баров = 12000 минут = 200h)
        "ret_4h": [0.001, 0.002, 0.001],  # было ret_5m
        "ret_12h": [0.003, 0.004, 0.003],  # было ret_15m
        "ret_24h": [0.010, 0.012, 0.011],  # было ret_60m
        "rsi": [55.0, 60.0, 58.0],
        "yang_zhang_48h": [0.05, 0.052, 0.051],  # было yang_zhang_24h
        "yang_zhang_7d": [0.08, 0.082, 0.081],  # было yang_zhang_168h
        "yang_zhang_30d": [0.12, 0.122, 0.121],  # было yang_zhang_720h
        "parkinson_48h": [0.06, 0.062, 0.061],  # было parkinson_24h
        "parkinson_7d": [0.09, 0.092, 0.091],  # было parkinson_168h
        "garch_30d": [0.025, 0.026, 0.0255],  # было garch_500m
        "garch_200h": [
            0.03,
            0.031,
            0.0305,
        ],  # было garch_12h (42 бара = 10080 мин = 7d, минимум для GARCH на 4h)
        "garch_14d": [0.04, 0.041, 0.0405],  # было garch_24h
        "taker_buy_ratio": [0.52, 0.53, 0.525],
        "taker_buy_ratio_sma_8h": [0.51, 0.52, 0.515],  # было 6h
        "taker_buy_ratio_sma_16h": [0.50, 0.51, 0.505],  # было 12h
        "taker_buy_ratio_sma_24h": [0.49, 0.50, 0.495],
        "taker_buy_ratio_momentum_4h": [0.01, 0.015, 0.012],  # было 1h
        "taker_buy_ratio_momentum_8h": [0.02, 0.025, 0.022],  # было 6h
        "taker_buy_ratio_momentum_12h": [0.015, 0.020, 0.017],
        "cvd_24h": [1000.0, 1200.0, 1100.0],
        "cvd_7d": [5000.0, 5500.0, 5250.0],  # было cvd_168h
        "fear_greed_value": [50.0, 55.0, 52.0],
        "is_high_importance": [0, 0, 0],
    }

    df = pd.DataFrame(df_data)

    # Создаем mock environment
    mock_env = Mock()
    mock_env.df = df
    mock_env.state = Mock(
        units=0.1,
        signal_pos=0.0,
        cash=10000.0,
        max_position=1.0,
        step_idx=1,
        last_vol_imbalance=0.01,
        last_trade_intensity=0.5,
        last_realized_spread=0.001,
        last_agent_fill_ratio=0.5,
        token_index=0,
    )
    mock_env.lob = None
    mock_env.last_mtm_price = 50200.0
    mock_env.last_mid = 50200.0
    mock_env._last_reward_price = 50100.0
    mock_env.observation_space = Mock(shape=(_N_FEATURES,))

    # Создаем mock sim
    mock_sim = Mock()
    mock_sim.get_macd = Mock(return_value=10.0)
    mock_sim.get_macd_signal = Mock(return_value=8.0)
    mock_sim.get_momentum = Mock(return_value=5.0)
    mock_sim.get_atr = Mock(return_value=100.0)
    mock_sim.get_cci = Mock(return_value=50.0)
    mock_sim.get_obv = Mock(return_value=1000.0)
    mock_sim.get_bb_lower = Mock(return_value=49000.0)
    mock_sim.get_bb_upper = Mock(return_value=51000.0)

    mock_env.sim = mock_sim
    mock_env._resolve_reward_price = Mock(return_value=50200.0)

    mediator = Mediator(_configure_risk_attrs(mock_env), event_level=0)
    mediator._context_row_idx = 1

    # Строим observation
    row = df.iloc[1]
    obs = mediator._build_observation(row=row, state=mock_env.state, mark_price=50200.0)

    # Проверки
    assert obs.shape == (_N_FEATURES,), f"Expected shape ({_N_FEATURES},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert np.all(np.isfinite(obs)), f"Observation contains NaN or Inf: {obs[~np.isfinite(obs)]}"

    # Проверка диапазонов
    assert np.all(obs >= -1e6), f"Some features too negative: {obs.min()}"
    assert np.all(obs <= 1e6), f"Some features too positive: {obs.max()}"

    # Проверка что блок external находится в разумном диапазоне
    _ext_start = _block_start("external")
    external = obs[_ext_start : _ext_start + _EXT_DIM]
    assert np.all(external >= -3.0), f"External features below -3: {external[external < -3.0]}"
    assert np.all(external <= 3.0), f"External features above 3: {external[external > 3.0]}"

    print(f"✓ Full pipeline test passed!")
    print(f"  Observation shape: {obs.shape}")
    print(f"  External features range: [{external.min():.3f}, {external.max():.3f}]")
    print(f"  All features range: [{obs.min():.3f}, {obs.max():.3f}]")


def test_crypto_norm_cols_present_in_dataframe():
    """Имена крипто-подблока external (индексы 0..20) — уникальны и не поехали."""
    expected_cols = [
        "cvd_24h",
        "cvd_7d",  # было cvd_168h
        "yang_zhang_48h",  # было yang_zhang_24h
        "yang_zhang_7d",  # было yang_zhang_168h
        "garch_200h",  # было garch_12h (42 бара = 10080 мин = 7d, минимум для GARCH на 4h)
        "garch_14d",  # было garch_24h
        "ret_12h",  # было ret_15m
        "ret_24h",  # было ret_60m
        "ret_4h",  # было ret_5m
        "sma_12000",  # было sma_60 (50 баров = 12000 минут = 200h для 4h)
        "yang_zhang_30d",  # было yang_zhang_720h
        "parkinson_48h",  # было parkinson_24h
        "parkinson_7d",  # было parkinson_168h
        "garch_30d",  # было garch_500m
        "taker_buy_ratio",
        "taker_buy_ratio_sma_24h",
        # НОВЫЕ 5 признаков (обновлено для 4h)
        "taker_buy_ratio_sma_8h",  # было 6h
        "taker_buy_ratio_sma_16h",  # было 12h
        "taker_buy_ratio_momentum_4h",  # было 1h
        "taker_buy_ratio_momentum_8h",  # было 6h
        "taker_buy_ratio_momentum_12h",  # без изменений
    ]

    assert len(set(expected_cols)) == len(expected_cols), "Duplicate column names found"
    # Крипто-подблок — начало external; за ним идут stock- и macro-признаки.
    assert len(expected_cols) <= _EXT_DIM

    print(f"✓ {len(expected_cols)} crypto norm_cols have unique names")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
