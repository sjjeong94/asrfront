from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sigs():
    from audio_signals import signals

    return signals()
