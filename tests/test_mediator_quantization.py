import json
import pathlib
import sys
import types

import pytest

base = pathlib.Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.append(str(base))

# Plain imports, not spec_from_file_location. Loading these from their paths and
# installing them under their own names replaced the copies every other module
# already held, so a later test comparing an ActionProto against
# trading_patchnew's ActionProto found two different classes.
import impl_quantizer as impl_mod
import quantizer as quant_mod  # noqa: F401  -- imported for its side effects on impl_quantizer
from action_proto import ActionProto, ActionType
from core_constants import PRICE_SCALE
from mediator import Mediator

QuantizerImpl = impl_mod.QuantizerImpl


class DummyLOB:
    def __init__(self):
        self.next_id = 1

    def add_limit_order(
        self,
        is_buy_side,
        price_ticks,
        volume,
        timestamp,
        taker_is_agent=True,
    ):
        oid = self.next_id
        self.next_id += 1
        return oid, 0

    def remove_order(self, is_buy_side, price_ticks, order_id):
        return True

    def match_market_order(
        self,
        is_buy_side,
        volume,
        timestamp,
        taker_is_agent,
        out_prices=None,
        out_volumes=None,
        out_is_buy=None,
        out_is_self=None,
        out_ids=None,
        max_len: int = 0,
    ):
        return 0, 0.0


class DummyState:
    def __init__(self):
        self.units = 0.0
        self.cash = 0.0
        self.max_position = 10.0


class DummyEnv:
    def __init__(
        self,
        lob,
        symbol: str = "BTCUSDT",
        quantizer_path: pathlib.Path | None = None,
    ):
        self.state = DummyState()
        self.lob = lob
        self.symbol = symbol
        self.last_mid = 100.0
        self.last_mtm_price = 100.0
        quantizer_cfg: dict[str, object] = {}
        if quantizer_path is not None:
            path_str = str(quantizer_path)
            quantizer_cfg = {
                "path": path_str,
                "filters_path": path_str,
                "strict_filters": True,
                "enforce_percent_price_by_side": True,
            }
        self.run_config = types.SimpleNamespace(quantizer=quantizer_cfg)


filters = {
    "BTCUSDT": {
        "PRICE_FILTER": {"minPrice": "0", "maxPrice": "1000000", "tickSize": "0.5"},
        "LOT_SIZE": {"minQty": "0.1", "maxQty": "1000", "stepSize": "0.1"},
        "MIN_NOTIONAL": {"minNotional": "5"},
        "PERCENT_PRICE_BY_SIDE": {"multiplierUp": "1000", "multiplierDown": "0"},
    }
}


@pytest.fixture
def filters_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "filters.json"
    path.write_text(json.dumps({"filters": filters}), encoding="utf-8")
    return path


def make_mediator(filters_path: pathlib.Path, *, use_exec: bool = False):
    env = DummyEnv(DummyLOB(), quantizer_path=filters_path)
    med = Mediator(env, use_exec_sim=use_exec)
    assert med.quantizer is not None
    assert med.quantizer_impl is not None
    return med


def test_unquantized_order_rejected(filters_file: pathlib.Path):
    med = make_mediator(filters_file)
    price = int(100.3 * PRICE_SCALE)
    oid, qpos = med.add_limit_order(is_buy_side=True, price_ticks=price, volume=0.25, timestamp=0)
    assert (oid, qpos) == (0, 0)


def test_quantized_order_accepted(filters_file: pathlib.Path):
    med = make_mediator(filters_file)
    price = int(100.5 * PRICE_SCALE)
    oid, qpos = med.add_limit_order(is_buy_side=True, price_ticks=price, volume=0.2, timestamp=0)
    assert oid != 0


def test_exec_simulator_receives_quantizer(filters_file: pathlib.Path):
    med = make_mediator(filters_file, use_exec=True)
    assert med.exec is not None
    assert getattr(med.exec, "quantizer", None) is med.quantizer
    assert getattr(med.exec, "quantizer_impl", None) is med.quantizer_impl
