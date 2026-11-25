"""Shared fixtures and Hypothesis profiles for kernel tests.

Profiles (select with HYPOTHESIS_PROFILE=<name>):
  default   – quick local runs (~1 min for the whole suite)
  thorough  – 20x more examples; run before merging kernel changes

Per-test example counts are multiples of the profile's ``max_examples``
(see ``prefix.kernels.strategies.examples``), so one knob scales everything.
"""

import os

import pytest
import torch
from hypothesis import HealthCheck, settings

_COMMON = dict(deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("default", max_examples=200, **_COMMON)
settings.register_profile("thorough", max_examples=4000, **_COMMON)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture
def gpu() -> str:
    """Device string for GPU-only tests (skips when there is no GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return "cuda"
