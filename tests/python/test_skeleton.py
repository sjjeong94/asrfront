import numpy as np
import pytest

import asrfront
from asrfront import _ext


def test_version():
    assert asrfront.__version__ == "0.1.0"


def test_whisper_out_frames():
    assert _ext.WhisperFrontend().out_frames(16000) == 3000
    assert _ext.WhisperFrontend(pad_or_trim=False).out_frames(16000) == 100
    with pytest.raises(ValueError):
        _ext.WhisperFrontend(n_mels=64)


def test_sensevoice_out_frames():
    fe = _ext.SenseVoiceFrontend()
    assert fe.feat_dim == 560
    assert fe.raw_frames(16000) == 98
    assert fe.out_frames(16000) == 17


def test_sensevoice_cmvn_length_checked():
    with pytest.raises(ValueError):
        _ext.SenseVoiceFrontend(
            cmvn_shift=np.zeros(10, np.float32), cmvn_scale=np.ones(10, np.float32)
        )


def test_input_validation():
    with pytest.raises(ValueError):
        asrfront.whisper.log_mel(np.zeros((2, 16000), np.float32))


@pytest.mark.parametrize(
    "fn",
    [asrfront.whisper.log_mel, asrfront.sensevoice.fbank, asrfront.sensevoice.fbank_raw],
)
def test_compute_not_implemented_yet(fn):
    # M1 skeleton: compute paths are stubs. Remove once M4/M5 land.
    with pytest.raises(NotImplementedError):
        fn(np.zeros(16000, np.float32))
