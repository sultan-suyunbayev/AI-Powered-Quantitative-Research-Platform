#!/usr/bin/env python3
"""
Интеграционный тест Yang-Zhang волатильности.
Проверяет работу признака в онлайн и оффлайн режимах.

Имена признаков строит _format_window_name: окна короче часа выводятся в
минутах (24 -> "24m", 100 -> "100m"). Раньше они усекались до целых часов,
и окно в 24 минуты называлось "0h".
"""
import pandas as pd
import numpy as np
from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features


def test_online_mode():
    """Тест онлайн-режима с OHLC данными."""
    print("Тест 1: Онлайн-режим (Online Transformer)")

    spec = FeatureSpec(
        lookbacks_prices=[5, 15],
        rsi_period=14,
        yang_zhang_windows=[24, 100],  # 24 минуты, 100 минут
        # The synthetic frame is one-minute bars; the default 240 would
        # round both windows down to a single bar and Yang-Zhang needs two.
        bar_duration_minutes=1,
    )

    transformer = OnlineFeatureTransformer(spec)

    # Симулируем 150 баров
    base_price = 100.0
    results = []

    for i in range(150):
        noise = 0.01 * np.sin(i * 0.1)
        open_p = base_price * (1 + noise)
        high = open_p * 1.005
        low = open_p * 0.995
        close = base_price * (1 + 0.01 * np.sin((i + 0.5) * 0.1))

        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=1000000000 + i * 60000,  # каждую минуту
            close=close,
            open_price=open_p,
            high=high,
            low=low,
        )

        results.append(feats)
        base_price *= 1.0001

    # Проверяем последние признаки
    last_feats = results[-1]

    # Должны быть Yang-Zhang признаки
    assert "yang_zhang_24m" in last_feats, "Отсутствует yang_zhang_24m"
    assert "yang_zhang_100m" in last_feats, "Отсутствует yang_zhang_100m"

    # Проверяем что значения не NaN (у нас достаточно данных)
    yz_24m = last_feats.get("yang_zhang_24m")
    yz_100m = last_feats.get("yang_zhang_100m")

    if pd.notna(yz_24m):
        assert yz_24m > 0, f"Yang-Zhang 24min должна быть положительной: {yz_24m}"
        print(f"  ✓ Yang-Zhang 24min: {yz_24m:.6f}")
    else:
        print(f"  ! Yang-Zhang 24min: NaN (недостаточно данных)")

    if pd.notna(yz_100m):
        assert yz_100m > 0, f"Yang-Zhang 100min должна быть положительной: {yz_100m}"
        print(f"  ✓ Yang-Zhang 100min: {yz_100m:.6f}")
    else:
        print(f"  ! Yang-Zhang 100min: NaN (недостаточно данных)")

    return True


def test_offline_mode():
    """Тест оффлайн-режима с OHLC данными."""
    print("\nТест 2: Оффлайн-режим (apply_offline_features)")

    # Создаем тестовый датафрейм с OHLC
    n_rows = 150
    base_price = 100.0
    data = []

    for i in range(n_rows):
        noise = 0.01 * np.sin(i * 0.1)
        open_p = base_price * (1 + noise)
        high = open_p * 1.005
        low = open_p * 0.995
        close = base_price * (1 + 0.01 * np.sin((i + 0.5) * 0.1))

        data.append(
            {
                "ts_ms": 1000000000 + i * 60000,
                "symbol": "BTCUSDT",
                "price": close,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
            }
        )
        base_price *= 1.0001

    df = pd.DataFrame(data)

    spec = FeatureSpec(
        lookbacks_prices=[5, 15],
        rsi_period=14,
        yang_zhang_windows=[24, 100],  # 24 минуты, 100 минут
        # The synthetic frame is one-minute bars; the default 240 would
        # round both windows down to a single bar and Yang-Zhang needs two.
        bar_duration_minutes=1,
    )

    # Применяем трансформацию
    feats_df = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        open_col="open",
        high_col="high",
        low_col="low",
    )

    # Проверяем что колонки созданы
    assert "yang_zhang_24m" in feats_df.columns, "Отсутствует колонка yang_zhang_24m"
    assert "yang_zhang_100m" in feats_df.columns, "Отсутствует колонка yang_zhang_100m"

    # Проверяем последнюю строку
    last_row = feats_df.iloc[-1]
    yz_24m = last_row["yang_zhang_24m"]
    yz_100m = last_row["yang_zhang_100m"]

    if pd.notna(yz_24m):
        assert yz_24m > 0, f"Yang-Zhang 24min должна быть положительной: {yz_24m}"
        print(f"  ✓ Yang-Zhang 24min: {yz_24m:.6f}")
    else:
        print(f"  ! Yang-Zhang 24min: NaN (недостаточно данных)")

    if pd.notna(yz_100m):
        assert yz_100m > 0, f"Yang-Zhang 100min должна быть положительной: {yz_100m}"
        print(f"  ✓ Yang-Zhang 100min: {yz_100m:.6f}")
    else:
        print(f"  ! Yang-Zhang 100min: NaN (недостаточно данных)")

    # Проверяем что не все значения NaN
    non_nan_24m = feats_df["yang_zhang_24m"].notna().sum()
    non_nan_100m = feats_df["yang_zhang_100m"].notna().sum()

    print(f"  ✓ Непустых значений yang_zhang_24m: {non_nan_24m}/{len(feats_df)}")
    print(f"  ✓ Непустых значений yang_zhang_100m: {non_nan_100m}/{len(feats_df)}")

    assert non_nan_24m > 0, "Все значения yang_zhang_24m - NaN"
    assert non_nan_100m > 0, "Все значения yang_zhang_100m - NaN"

    return True


def test_without_ohlc():
    """Тест без OHLC данных - волатильность должна быть NaN."""
    print("\nТест 3: Без OHLC данных (должны быть NaN)")

    # Создаем датафрейм только с price (без OHLC)
    df = pd.DataFrame(
        {
            "ts_ms": [1000000000 + i * 60000 for i in range(50)],
            "symbol": ["BTCUSDT"] * 50,
            "price": [100 + i * 0.1 for i in range(50)],
        }
    )

    spec = FeatureSpec(
        lookbacks_prices=[5], rsi_period=14, yang_zhang_windows=[24], bar_duration_minutes=1
    )

    feats_df = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        # Не передаем OHLC колонки
    )

    # Должна быть колонка, но все значения NaN
    assert "yang_zhang_24m" in feats_df.columns, "Отсутствует колонка yang_zhang_24m"

    # calculate_yang_zhang_volatility takes close_prices and falls back to a
    # close-to-close estimate when there are no OHLC bars, so the column is
    # populated after the warm-up rather than staying NaN throughout.
    values = feats_df["yang_zhang_24m"].dropna()
    assert len(values) > 0, "Fallback должен давать значения без OHLC"
    assert (values >= 0).all(), "Волатильность не может быть отрицательной"
    assert feats_df["yang_zhang_24m"].isna().any(), "Первые бары — прогрев, ещё NaN"

    print(f"  ✓ Fallback к close-to-close: {len(values)}/{len(feats_df)} значений")

    return True


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ YANG-ZHANG ВОЛАТИЛЬНОСТИ")
    print("=" * 60 + "\n")

    tests = [
        test_online_mode,
        test_offline_mode,
        test_without_ohlc,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"  ❌ ОШИБКА: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ ИСКЛЮЧЕНИЕ: {e}\n")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТЫ: {passed} успешно, {failed} неудачно")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
