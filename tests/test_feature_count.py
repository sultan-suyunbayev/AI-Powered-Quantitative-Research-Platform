#!/usr/bin/env python3
"""How the feature layout's width follows ``max_num_tokens``.

This file used to be a script: three module-level ``make_layout()`` calls whose
results it printed, and no test functions at all. ``make_layout`` rewrites
``feature_config``'s module-level ``FEATURES_LAYOUT`` and ``N_FEATURES`` in
place, and a call at module level runs during *collection* -- in every xdist
worker, before any fixture can put the default back. Every later test that sized
a buffer from ``N_FEATURES`` was then working from ``max_num_tokens=15``, and
``obs_builder`` writes with bounds checking off. See ``tests/conftest.py``.

The mutation now happens inside tests, where the autouse guard in conftest
restores the layout around each one.
"""

import pytest

import feature_config
from feature_config import make_layout


@pytest.mark.parametrize("max_num_tokens", [1, 15, 16])
def test_token_block_is_sized_by_max_num_tokens(max_num_tokens: int) -> None:
    """The ``token`` block is exactly ``max_num_tokens`` wide."""
    layout = make_layout({"max_num_tokens": max_num_tokens})
    sizes = {block["name"]: block["size"] for block in layout}
    assert sizes["token"] == max_num_tokens


def test_total_width_moves_only_with_the_token_block() -> None:
    """Nothing else in the layout depends on ``max_num_tokens``."""
    totals = {}
    for max_num_tokens in (1, 15, 16):
        layout = make_layout({"max_num_tokens": max_num_tokens})
        totals[max_num_tokens] = sum(block["size"] for block in layout)

    assert totals[15] - totals[1] == 14
    assert totals[16] - totals[15] == 1


def test_default_layout_is_the_one_the_builder_writes() -> None:
    """The shipped default is a single token, 113 features wide.

    ``obs_builder.build_observation_vector`` writes exactly ``N_FEATURES``
    floats, so this number is the contract between the layout and the extension
    -- see ``test_feature_layout_correctness``.
    """
    assert feature_config.N_FEATURES == sum(b["size"] for b in feature_config.FEATURES_LAYOUT)
    assert feature_config.N_FEATURES == 113
