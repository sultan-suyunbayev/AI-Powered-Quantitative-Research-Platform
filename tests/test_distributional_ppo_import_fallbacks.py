"""
Tests for import fallback paths in distributional_ppo.py.
These tests manipulate sys.modules to trigger fallback import paths.
Run in isolation to avoid side effects.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from unittest import mock

import pytest


class TestImportFallbacks:
    """Test import fallback paths."""

    def test_module_loads_normally(self):
        """Verify module loads without fallbacks in normal case."""
        # Just verify the module can be imported
        import distributional_ppo

        assert distributional_ppo.DistributionalPPO is not None

    def test_recurrent_backend_is_set(self):
        """Verify _RECURRENT_BACKEND is set."""
        import distributional_ppo

        assert distributional_ppo._RECURRENT_BACKEND in ("sb3_contrib", "stable_baselines3")

    def test_distributional_policy_alias_exists(self):
        """Verify DistributionalPolicy alias is registered."""
        import distributional_ppo

        aliases = distributional_ppo._DISTRIBUTIONAL_POLICY_ALIASES
        # Should have the alias if custom_policy_patch1 was imported
        # or be empty dict if not
        assert isinstance(aliases, dict)

    def test_import_does_not_patch_torch(self):
        """Importing the module must not change torch's behaviour.

        It used to detect pytest and replace torch.rand with a version shifted
        into [0.5, 1.0], which meant the tested code was not the shipped code
        and every other test in the process drew from a skewed distribution.
        """
        import distributional_ppo  # noqa: F401
        import torch

        assert not hasattr(torch, "_distributional_rand_patch")
        assert not hasattr(distributional_ppo, "_patch_rand_for_tests")


class TestTorchRandIsUntouched:
    """torch.rand must span its full range under the test runner."""

    def test_rand_covers_the_whole_range(self):
        import torch

        samples = torch.rand(4096)
        assert samples.min().item() < 0.5, "torch.rand is being shifted"
        assert 0.0 <= samples.min().item() and samples.max().item() <= 1.0


class TestModuleLevelConstants:
    """Test module-level constants and configurations."""

    def test_default_clip_range_vf(self):
        """Verify DEFAULT_CLIP_RANGE_VF is set."""
        import distributional_ppo

        assert hasattr(distributional_ppo, "DEFAULT_CLIP_RANGE_VF")
        assert isinstance(distributional_ppo.DEFAULT_CLIP_RANGE_VF, (int, float))

    def test_popart_classes_exist(self):
        """Verify PopArt classes are defined."""
        import distributional_ppo

        assert hasattr(distributional_ppo, "PopArtController")
        assert hasattr(distributional_ppo, "PopArtHoldoutBatch")
        assert hasattr(distributional_ppo, "PopArtHoldoutEvaluation")
        assert hasattr(distributional_ppo, "PopArtCandidateMetrics")

    def test_helper_functions_exist(self):
        """Verify helper functions are defined."""
        import distributional_ppo

        assert hasattr(distributional_ppo, "safe_explained_variance")
        assert hasattr(distributional_ppo, "compute_grouped_explained_variance")


class TestDistributionalPPOClass:
    """Test DistributionalPPO class attributes."""

    def test_class_has_expected_methods(self):
        """Verify class has expected methods."""
        from distributional_ppo import DistributionalPPO

        expected_methods = [
            "train",
            "learn",
            "predict",
            "save",
            "load",
            "collect_rollouts",
            "get_parameters",
            "set_parameters",
        ]
        for method in expected_methods:
            assert hasattr(DistributionalPPO, method), f"Missing method: {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
