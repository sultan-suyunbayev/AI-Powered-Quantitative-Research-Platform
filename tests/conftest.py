"""
tests/conftest.py

Central pytest configuration with optional dependency detection.

OPTIONAL DEPENDENCY PATTERN
============================
This module implements graceful test skipping for tests that require optional
dependencies (torch, gymnasium, stable-baselines3, etc.). This is an **intentional
design pattern** to support:

1. **Minimal CI Runs** - Core tests run without ML dependencies
2. **Full CI Runs** - All tests run with complete dependency set
3. **Local Development** - Developers can run subset of tests without full stack

The pattern uses pytest hooks to automatically skip tests based on:
- Import availability checks (cached at module load)
- Test file name patterns (e.g., "test_ppo" requires torch)
- Explicit pytest markers

This is NOT tech debt - it's a feature enabling flexible test execution.

Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#testing-optional-deps-pattern (Closed)
Related: docs/testing/TESTING_POLICY.md
"""

from __future__ import annotations

import ipaddress
import socket as _socket
import sys
import types
import urllib.parse
from pathlib import Path

import pytest

# Project root is the parent of the tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS = PROJECT_ROOT / "tests"

# Load stdlib logging before project paths are added
sys.path = [p for p in sys.path if p not in {str(TESTS)}]
import logging  # noqa: F401

# =============================================================================
# Optional dependency detection
# =============================================================================


def _check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Cache dependency availability checks
TORCH_AVAILABLE = _check_import("torch")
GYMNASIUM_AVAILABLE = _check_import("gymnasium")
SB3_AVAILABLE = _check_import("stable_baselines3")
PYARROW_AVAILABLE = _check_import("pyarrow")
HYPOTHESIS_AVAILABLE = _check_import("hypothesis")
SORTEDCONTAINERS_AVAILABLE = _check_import("sortedcontainers")
CLOUDPICKLE_AVAILABLE = _check_import("cloudpickle")
OPTUNA_AVAILABLE = _check_import("optuna")


# =============================================================================
# Pytest hooks for automatic test skipping
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests that require unavailable optional dependencies.

    This hook examines test file paths and module contents to determine
    which tests should be skipped based on missing dependencies.
    """
    skip_torch = pytest.mark.skip(reason="PyTorch not installed (install with: pip install torch)")
    skip_gymnasium = pytest.mark.skip(
        reason="gymnasium not installed (install with: pip install gymnasium)"
    )
    skip_sb3 = pytest.mark.skip(
        reason="stable-baselines3 not installed (install with: pip install stable-baselines3)"
    )
    skip_pyarrow = pytest.mark.skip(
        reason="pyarrow not installed (install with: pip install pyarrow)"
    )
    skip_sortedcontainers = pytest.mark.skip(
        reason="sortedcontainers not installed (install with: pip install sortedcontainers)"
    )
    skip_cloudpickle = pytest.mark.skip(
        reason="cloudpickle not installed (install with: pip install cloudpickle)"
    )
    skip_optuna = pytest.mark.skip(reason="optuna not installed (install with: pip install optuna)")
    skip_hypothesis = pytest.mark.skip(
        reason="hypothesis not installed (install with: pip install hypothesis)"
    )

    # Patterns indicating torch dependency
    torch_patterns = [
        "test_ppo",
        "test_twin_critics",
        "test_categorical",
        "test_vgs",
        "test_upgd",
        "test_gradient",
        "test_quantile",
        "test_popart",
        "test_lstm",
        "test_pbt",
        "test_distributional",
        "test_numerical",
        "test_shared_memory",
        "test_vf_clip",
        "test_vf_variance",
        "test_gae",
        "test_kl_direction",
        "test_return_scale",
        "test_state_perturbation",
        "test_torch",
        "test_ev_",
        "test_bug_fixes_2025",
        "test_bug8",
        "test_bug10",
        "test_advantage_normalization",
        "test_adaptive_upgd",
        "test_actual_ppo",
        "test_four_problems",
        "test_potential_issues",
        "test_unit_custom_policy",
        "test_unit_train_model",
    ]

    # Patterns indicating gymnasium dependency
    gymnasium_patterns = [
        "test_bug7_grouped_ev",
        "test_bug_fixes_final_audit",
        "test_correct_api_usage",
        "test_forex_improvements",
        "test_forex_training",
        "test_futures_training",
        "test_timing_profiles",
    ]

    # Patterns indicating LOB/sortedcontainers dependency
    lob_patterns = [
        "test_lob",
        "test_l3",
        "test_matching_engine",
        "test_hidden_liquidity",
        "test_queue_tracker",
        "test_cme_l3",
        "test_cme_risk",
        "test_cme_settlement",
        "test_execution_providers_l3",
        "test_market_impact",
        "test_fill_probability",
    ]

    # Patterns indicating stable-baselines3 dependency
    sb3_patterns = [
        "test_shared_memory_vec_env",
    ]

    for item in items:
        test_path = str(item.fspath)
        test_name = item.name

        # Check markers first
        if "requires_torch" in [m.name for m in item.iter_markers()]:
            if not TORCH_AVAILABLE:
                item.add_marker(skip_torch)
                continue

        if "requires_gymnasium" in [m.name for m in item.iter_markers()]:
            if not GYMNASIUM_AVAILABLE:
                item.add_marker(skip_gymnasium)
                continue

        if "requires_sb3" in [m.name for m in item.iter_markers()]:
            if not SB3_AVAILABLE:
                item.add_marker(skip_sb3)
                continue

        if "requires_pyarrow" in [m.name for m in item.iter_markers()]:
            if not PYARROW_AVAILABLE:
                item.add_marker(skip_pyarrow)
                continue

        # Pattern-based detection
        if not TORCH_AVAILABLE:
            for pattern in torch_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_torch)
                    break

        if not GYMNASIUM_AVAILABLE:
            for pattern in gymnasium_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_gymnasium)
                    break

        if not SORTEDCONTAINERS_AVAILABLE:
            for pattern in lob_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_sortedcontainers)
                    break

        if not SB3_AVAILABLE:
            for pattern in sb3_patterns:
                if pattern in test_path.lower():
                    item.add_marker(skip_sb3)
                    break


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "requires_gymnasium: mark test as requiring gymnasium")
    config.addinivalue_line("markers", "requires_sb3: mark test as requiring stable-baselines3")
    config.addinivalue_line("markers", "requires_pyarrow: mark test as requiring pyarrow")
    config.addinivalue_line(
        "markers",
        "allow_network: let this test open sockets to the outside world",
    )
    config.addinivalue_line(
        "markers",
        "uses_filters_refresh: this test exercises QuantizerImpl._refresh_filters "
        "and stubs the subprocess itself",
    )


# =============================================================================
# Fixtures for optional dependencies
# =============================================================================


@pytest.fixture
def requires_torch():
    """Skip test if torch is not available."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def requires_gymnasium():
    """Skip test if gymnasium is not available."""
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("gymnasium not installed")


@pytest.fixture
def requires_sb3():
    """Skip test if stable-baselines3 is not available."""
    if not SB3_AVAILABLE:
        pytest.skip("stable-baselines3 not installed")


@pytest.fixture
def requires_pyarrow():
    """Skip test if pyarrow is not available."""
    if not PYARROW_AVAILABLE:
        pytest.skip("pyarrow not installed")


_requests_stub = types.ModuleType("requests")


def _unavailable(*args, **kwargs):  # pragma: no cover - network calls disabled in tests
    raise RuntimeError("requests module is not available in the test environment")


_requests_stub.get = _unavailable
_requests_stub.post = _unavailable
_requests_stub.put = _unavailable
_requests_stub.delete = _unavailable
_requests_stub.request = _unavailable


class _MockSession:
    """Mock Session class for testing."""

    def __init__(self):
        self.headers = {}

    def get(self, *args, **kwargs):
        raise RuntimeError("requests.Session.get is not available in the test environment")

    def post(self, *args, **kwargs):
        raise RuntimeError("requests.Session.post is not available in the test environment")

    def close(self):
        pass


_requests_stub.Session = _MockSession


class _PreparedRequest:
    """The part of requests.PreparedRequest that assembles a URL.

    ``rest_budget._make_cache_key`` uses it to canonicalise a request into a
    cache key; nothing here performs I/O, so the network guard has no reason to
    take it away.
    """

    def __init__(self, method: str, url: str, params=None):
        self.method = (method or "GET").upper()
        parts = urllib.parse.urlsplit(url or "")
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if params:
            items = params.items() if hasattr(params, "items") else params
            for key, value in items:
                if value is None:
                    continue
                if isinstance(value, (list, tuple, set)):
                    query.extend((str(key), str(item)) for item in value)
                else:
                    query.append((str(key), str(value)))
        self.url = urllib.parse.urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urllib.parse.urlencode(query, doseq=True),
                parts.fragment,
            )
        )
        self.headers: dict = {}
        self.body = None


class _Request:
    """Enough of requests.Request to prepare a URL."""

    def __init__(self, method: str = "GET", url: str = "", params=None, **kwargs):
        self.method = method
        self.url = url
        self.params = params
        self.kwargs = kwargs

    def prepare(self) -> _PreparedRequest:
        return _PreparedRequest(self.method, self.url, self.params)


class _Response:
    """Placeholder for annotations; the guard never produces one."""

    status_code = 0
    headers: dict = {}
    url = ""

    def json(self):  # pragma: no cover - nothing in the tests calls it
        raise RuntimeError("requests.Response is not available in the test environment")


_requests_stub.Request = _Request
_requests_stub.PreparedRequest = _PreparedRequest
_requests_stub.Response = _Response

# Create stub exceptions module for testing
_requests_exceptions_stub = types.ModuleType("requests.exceptions")


class RequestException(Exception):
    """Base exception for requests."""

    pass


class HTTPError(RequestException):
    """HTTP error occurred."""

    pass


class ConnectionError(RequestException):
    """Connection error occurred."""

    pass


class Timeout(RequestException):
    """Request timed out."""

    pass


class TooManyRedirects(RequestException):
    """Too many redirects."""

    pass


# Add exception classes to both modules
_requests_exceptions_stub.RequestException = RequestException
_requests_exceptions_stub.HTTPError = HTTPError
_requests_exceptions_stub.ConnectionError = ConnectionError
_requests_exceptions_stub.Timeout = Timeout
_requests_exceptions_stub.TooManyRedirects = TooManyRedirects

_requests_stub.exceptions = _requests_exceptions_stub
_requests_stub.RequestException = RequestException
_requests_stub.HTTPError = HTTPError
_requests_stub.ConnectionError = ConnectionError
_requests_stub.Timeout = Timeout

sys.modules.setdefault("requests", _requests_stub)
sys.modules.setdefault("requests.exceptions", _requests_exceptions_stub)

if sys.platform == "win32":
    # Mock unix resource module on Windows to allow test collection
    _resource_stub = types.ModuleType("resource")
    _resource_stub.RLIMIT_AS = 6
    _resource_stub.RLIMIT_CPU = 0
    _resource_stub.RLIMIT_NOFILE = 7
    _resource_stub.RLIMIT_NPROC = 8
    _resource_stub.RLIMIT_CORE = 4

    class ResourceError(Exception):
        pass

    _resource_stub.error = ResourceError

    def _setrlimit(limit, limits):
        pass

    _resource_stub.setrlimit = _setrlimit

    sys.modules["resource"] = _resource_stub

# Re-add tests directory to sys.path to resolve sibling imports in test modules
if str(TESTS) not in sys.path:
    sys.path.append(str(TESTS))


# ---------------------------------------------------------------------------
# Feature-layout guard
# ---------------------------------------------------------------------------
# feature_config.make_layout() rewrites the module-level FEATURES_LAYOUT and
# N_FEATURES in place.  Several tests call it with a narrower layout
# (ext_norm_dim=21, ext_norm_dim=28, max_num_tokens=16) and never put the
# default back, so every later test in the same worker sizes its buffers from
# the wrong total.
#
# That is not a cosmetic mismatch: obs_builder.build_observation_vector writes
# through typed memoryviews with bounds checking off, so an out_features array
# shorter than the vector it writes runs past the end of the allocation and
# corrupts the heap.  Under pytest-xdist the worker dies with
# "double free or corruption" / "Fatal Python error: Aborted", which can in
# turn take the whole run down with an xdist INTERNALERROR.
#
# Snapshot the layout once and restore it after every test.
try:
    import feature_config as _feature_config

    _DEFAULT_FEATURES_LAYOUT = [dict(block) for block in _feature_config.FEATURES_LAYOUT]
    _DEFAULT_N_FEATURES = _feature_config.N_FEATURES
    _DEFAULT_EXT_NORM_DIM = _feature_config.EXT_NORM_DIM
except Exception:  # pragma: no cover - feature_config is always importable in-tree
    _feature_config = None


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------
# Unseeded draws made the suite fail differently on every run: three CI runs in
# a row failed on three different tests, each asserting something about a random
# value that the distribution crosses some fraction of the time. Seeding before
# every test makes a pass mean the code passed, not that the draw was kind.
#
# Only generators that are already imported are seeded -- a test that never
# touches torch should not pay to import it. A test that wants its own seed sets
# it inside the test body and wins, because this runs first.

_TEST_SEED = 20260908


@pytest.fixture(autouse=True)
def _deterministic_randomness():
    """Put every imported random generator in a known state before each test."""
    import random as _random

    _random.seed(_TEST_SEED)

    _numpy = sys.modules.get("numpy")
    if _numpy is not None:
        try:
            _numpy.random.seed(_TEST_SEED)
        except Exception:  # pragma: no cover - a stubbed numpy, seen in this suite
            pass

    _torch = sys.modules.get("torch")
    if _torch is not None:
        try:
            _torch.manual_seed(_TEST_SEED)
        except Exception:  # pragma: no cover - a stubbed torch, seen in this suite
            pass

    yield


def _reset_feature_layout() -> None:
    if _feature_config is None:
        return
    if (
        _feature_config.N_FEATURES != _DEFAULT_N_FEATURES
        or _feature_config.EXT_NORM_DIM != _DEFAULT_EXT_NORM_DIM
        or _feature_config.FEATURES_LAYOUT != _DEFAULT_FEATURES_LAYOUT
    ):
        _feature_config.FEATURES_LAYOUT = [dict(block) for block in _DEFAULT_FEATURES_LAYOUT]
        _feature_config.N_FEATURES = _DEFAULT_N_FEATURES
        _feature_config.EXT_NORM_DIM = _DEFAULT_EXT_NORM_DIM


@pytest.fixture(autouse=True)
def _restore_feature_layout():
    """Put feature_config's global layout back around every test.

    Before as well as after: a module-level ``make_layout()`` call runs during
    *collection*, which is over before the first test starts, so restoring only
    on teardown would leave the whole session working from whatever the last
    collected module left behind.
    """
    _reset_feature_layout()
    yield
    _reset_feature_layout()


# =============================================================================
# The suite does not talk to the internet
# =============================================================================
#
# The ``requests`` stub above covers the HTTP libraries, but it is not the only
# way out: the daemon's clock sync sends an NTP packet over a raw UDP socket,
# and any client is free to use a library the stub has never heard of. So the
# block sits at the socket layer instead. Anything aimed outside loopback --
# connect, sendto, or even the name lookup -- raises and says which call it was.
#
# Loopback stays open. Local servers, ``socketpair()`` and in-process ASGI
# transports are how a good part of this suite is written, and none of them
# leave the machine. A test that genuinely has to go out can say so with
# ``@pytest.mark.allow_network``; nothing in the suite does today.

_NETWORK_ALLOWED = False

_LOCAL_HOSTNAMES = frozenset({"", "localhost", "localhost.localdomain", "ip6-localhost"})

_real_socket_connect = _socket.socket.connect
_real_socket_connect_ex = _socket.socket.connect_ex
_real_socket_sendto = _socket.socket.sendto
_real_create_connection = _socket.create_connection
_real_getaddrinfo = _socket.getaddrinfo


def _is_local_address(address) -> bool:
    """Whether an address stays on this machine.

    AF_UNIX paths and anything that is not a (host, port) pair are local by
    construction. A hostname that is not one of the loopback spellings is not:
    resolving it is itself a trip to a DNS server.
    """
    if isinstance(address, (str, bytes, bytearray)):
        return True  # AF_UNIX socket path
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if host is None:
        return True
    host = str(host)
    if host.lower() in _LOCAL_HOSTNAMES:
        return True
    try:
        parsed = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return parsed.is_loopback or parsed.is_unspecified


def _check_address(address, call: str) -> None:
    if _NETWORK_ALLOWED or _is_local_address(address):
        return
    raise RuntimeError(
        f"This test suite does not use the network: blocked {call}({address!r}). "
        "Stub the call out, or mark the test with @pytest.mark.allow_network if "
        "it genuinely has to leave the machine."
    )


def _guarded_connect(self, address):
    _check_address(address, "socket.connect")
    return _real_socket_connect(self, address)


def _guarded_connect_ex(self, address):
    _check_address(address, "socket.connect_ex")
    return _real_socket_connect_ex(self, address)


def _guarded_sendto(self, data, *args):
    # sendto(data, address) and sendto(data, flags, address) -- the address is
    # the last argument either way. UDP needs no connect, which is how the NTP
    # client leaves the machine without touching any of the calls above.
    _check_address(args[-1] if args else None, "socket.sendto")
    return _real_socket_sendto(self, data, *args)


def _guarded_create_connection(address, *args, **kwargs):
    _check_address(address, "socket.create_connection")
    return _real_create_connection(address, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    _check_address((host, port), "socket.getaddrinfo")
    return _real_getaddrinfo(host, port, *args, **kwargs)


_socket.socket.connect = _guarded_connect
_socket.socket.connect_ex = _guarded_connect_ex
_socket.socket.sendto = _guarded_sendto
_socket.create_connection = _guarded_create_connection
_socket.getaddrinfo = _guarded_getaddrinfo


@pytest.fixture(autouse=True)
def _network_is_off(request):
    """Open the guard only for a test that asked for it by name."""
    global _NETWORK_ALLOWED
    previous = _NETWORK_ALLOWED
    _NETWORK_ALLOWED = request.node.get_closest_marker("allow_network") is not None
    try:
        yield
    finally:
        _NETWORK_ALLOWED = previous


@pytest.fixture(autouse=True)
def _no_binance_filters_refresh(request, monkeypatch):
    """Keep the filters refresh from spawning a fetcher and writing to the tree.

    ``QuantizerImpl.__init__`` with ``refresh_on_start`` runs
    ``python -m scripts.fetch_binance_filters`` in a child interpreter, which
    talks to the live Binance REST API and lands ``data/binance_filters.json``
    wherever it succeeds -- inside the checkout, during a test run. A child
    process is outside any in-process socket guard, so the refresh is stubbed
    here. Its return shape is (executed, succeeded, returncode, message), and
    "not executed" is a state the caller already handles.
    """
    if request.node.get_closest_marker("uses_filters_refresh") is not None:
        # A test that is about the refresh itself. It has to stub subprocess.run
        # on its own, which the one that does already did.
        yield
        return

    try:
        from impl_quantizer import QuantizerImpl
    except Exception:  # pragma: no cover - the module has optional dependencies
        yield
        return

    monkeypatch.setattr(
        QuantizerImpl,
        "_refresh_filters",
        classmethod(
            lambda cls, out_path: (False, False, None, "Refresh disabled in the test suite")
        ),
        raising=True,
    )
    yield
