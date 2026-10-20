from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sigs():
    from audio_signals import signals

    return signals()


@pytest.fixture(autouse=True)
def batch_threads_for_small_inputs(monkeypatch):
    """The batch functions keep little work on one thread (asrfront._audio.plan_chunks); the
    tests' inputs are small, so let them use threads and several chunks anyway."""
    from asrfront import _audio

    monkeypatch.setattr(_audio, "_MIN_COST_PER_THREAD", 1)
