import numpy as np
import numpy.typing as npt

__version__: str

class WhisperFrontend:
    n_mels: int
    def __init__(self, n_mels: int = 80, pad_or_trim: bool = True, n_threads: int = 1) -> None: ...
    def out_frames(self, n_samples: int) -> int: ...
    def log_mel(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

class SenseVoiceFrontend:
    n_mels: int
    feat_dim: int
    def __init__(
        self,
        n_mels: int = 80,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0.0,
        seed: int = 0,
        cmvn_shift: npt.NDArray[np.float32] | None = None,
        cmvn_scale: npt.NDArray[np.float32] | None = None,
        n_threads: int = 1,
    ) -> None: ...
    def raw_frames(self, n_samples: int) -> int: ...
    def out_frames(self, n_samples: int) -> int: ...
    def fbank(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
    def fbank_raw(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
