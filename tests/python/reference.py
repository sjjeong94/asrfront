"""Float64 numpy reference implementations ("ground truth") for asrfront.

- Whisper: port of ``whisper/audio.py`` from openai/whisper (MIT) with the mel filters of
  ``librosa.filters.mel`` (ISC).
- SenseVoice: port of ``torchaudio.compliance.kaldi.fbank`` (BSD-2-Clause) plus
  ``apply_lfr`` / ``apply_cmvn`` / ``load_cmvn`` from FunASR's ``WavFrontend`` (MIT).
- Gemma 4: port of transformers' ``Gemma4AudioFeatureExtractor`` and ``mel_filter_bank``
  (Apache-2.0) with the E2B / E4B checkpoints' settings.
- Audio loading: ffmpeg's decode -> resample -> downmix -> s16 pipeline as used by whisper's
  ``load_audio``, written from the behavior spec in ``docs/load.md`` (not from ffmpeg's code).

Every stage is exposed separately so intermediate values (window, filters, power spectrum,
...) can be compared when debugging the C implementation. Everything is computed in float64;
the original libraries compute in float32, and ``test_reference.py`` checks the two agree.

The upstream copyright notices and license texts are in ``LICENSES/`` at the repository root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------------------------

WHISPER_N_FFT = 400
WHISPER_HOP = 160
WHISPER_N_SAMPLES = 30 * SAMPLE_RATE  # 480000


def whisper_pad_or_trim(audio: np.ndarray, length: int = WHISPER_N_SAMPLES) -> np.ndarray:
    if audio.shape[0] > length:
        return audio[:length]
    return np.pad(audio, (0, length - audio.shape[0]))


def hann_window_periodic(n: int) -> np.ndarray:
    """``torch.hann_window(n)`` (periodic=True)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    f = np.asanyarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep
    return np.where(f >= min_log_hz, log_part, mels)


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asanyarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def slaney_mel_filters(
    sr: int = SAMPLE_RATE,
    n_fft: int = WHISPER_N_FFT,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """``librosa.filters.mel(sr, n_fft, n_mels)`` (htk=False, norm='slaney') in float64.

    Returns ``(n_mels, n_fft // 2 + 1)``.
    """
    if fmax is None:
        fmax = sr / 2.0
    fftfreqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    mel_f = _mel_to_hz_slaney(
        np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2)
    )
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


def whisper_stft_power(audio: np.ndarray) -> np.ndarray:
    """``|torch.stft(audio, 400, 160, hann, center=True, reflect)|**2`` minus the last frame.

    Returns ``(n_fft // 2 + 1, n_samples // hop)``.
    """
    pad = WHISPER_N_FFT // 2
    if audio.shape[0] <= pad:
        # torch's reflect padding requires pad < input length.
        raise ValueError(f"whisper STFT needs more than {pad} samples, got {audio.shape[0]}")
    x = np.pad(audio.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (x.shape[0] - WHISPER_N_FFT) // WHISPER_HOP
    idx = np.arange(WHISPER_N_FFT)[None, :] + WHISPER_HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * hann_window_periodic(WHISPER_N_FFT)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return power[:-1].T


def whisper_log_mel_stages(
    audio: np.ndarray,
    n_mels: int = 80,
    pad_or_trim: bool = True,
    filters: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float64)
    if pad_or_trim:
        audio = whisper_pad_or_trim(audio)
    if filters is None:
        filters = slaney_mel_filters(n_mels=n_mels)
    power = whisper_stft_power(audio)
    mel = filters @ power
    log = np.log10(np.maximum(mel, 1e-10))
    clamped = np.maximum(log, log.max() - 8.0)
    out = (clamped + 4.0) / 4.0
    return {"audio": audio, "power": power, "filters": filters, "mel": mel, "log": log, "out": out}


def whisper_log_mel(audio: np.ndarray, n_mels: int = 80, pad_or_trim: bool = True) -> np.ndarray:
    """Whisper log-mel spectrogram, ``(n_mels, n_frames)``."""
    return whisper_log_mel_stages(audio, n_mels, pad_or_trim)["out"]


def whisper_log_mel_padded(audio: np.ndarray, n_mels: int, n_padded: int) -> np.ndarray:
    """Whisper log-mel without pad_or_trim of ``audio`` zero-padded to ``n_padded`` samples
    (transformers' ``WhisperFeatureExtractor`` batches with ``padding=True``)."""
    padded = np.zeros(n_padded, dtype=np.float64)
    padded[: len(audio)] = audio
    return whisper_log_mel(padded, n_mels, pad_or_trim=False)


def qwen3_asr_transformers(audios: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3-ASR's feature extraction: transformers' ``WhisperFeatureExtractor`` configured by
    the checkpoints' ``preprocessor_config.json`` and called as ``Qwen3ASRProcessor`` does.
    Returns ``(features (B, 128, T), attention mask sums (B,))``. Needs the ``qwen`` and
    ``ref`` dependency groups."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=True,
    )
    out = fe(
        [np.asarray(a, dtype=np.float32) for a in audios],
        sampling_rate=16000,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="np",
    )
    return out["input_features"], out["attention_mask"].sum(-1)


# ---------------------------------------------------------------------------------------------
# Gemma 4 (E2B / E4B)
# ---------------------------------------------------------------------------------------------

GEMMA4_FRAME = 320
GEMMA4_HOP = 160
GEMMA4_FFT = 512
GEMMA4_MEL_FLOOR = 1e-3


def htk_mel_filters(
    n_bins: int = GEMMA4_FFT // 2 + 1, n_mels: int = 128, fmin: float = 0.0, fmax: float = 8000.0
) -> np.ndarray:
    """transformers' ``mel_filter_bank(..., norm=None, mel_scale="htk")``, ``(n_mels, n_bins)``."""
    mel = np.linspace(
        2595.0 * np.log10(1.0 + fmin / 700.0), 2595.0 * np.log10(1.0 + fmax / 700.0), n_mels + 2
    )
    hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    fft_f = np.linspace(0, SAMPLE_RATE // 2, n_bins)
    slopes = hz[None, :] - fft_f[:, None]
    down = -slopes[:, :-2] / np.diff(hz)[:-1]
    up = slopes[:, 2:] / np.diff(hz)[1:]
    return np.maximum(0.0, np.minimum(down, up)).T


def gemma4_padded_length(n_longest: int) -> int:
    """A batch's length: the longest item rounded up to a multiple of 128 samples."""
    return -(-n_longest // 128) * 128


def gemma4_log_mel_stages(audio: np.ndarray, n_padded: int | None = None) -> dict:
    """Gemma 4 audio features of ``audio`` (already truncated to 30 s if wanted) zero-padded to
    ``n_padded`` samples (default: to a multiple of 128): ``frames``, ``magnitude``, ``mel``,
    ``mask`` (valid frames) and ``out`` ``(T, 128)``, the padding's frames zeroed."""
    audio = np.asarray(audio, dtype=np.float64)
    n = audio.shape[0]
    if n_padded is None:
        n_padded = gemma4_padded_length(n)
    x = np.zeros(GEMMA4_FRAME // 2 + n_padded)  # semicausal padding, then the batch padding
    x[GEMMA4_FRAME // 2 : GEMMA4_FRAME // 2 + n] = audio
    n_frames = max(0, (x.shape[0] - (GEMMA4_FRAME + 1)) // GEMMA4_HOP + 1)
    idx = np.arange(n_frames)[:, None] * GEMMA4_HOP + np.arange(GEMMA4_FRAME)[None, :]
    frames = x[idx] * hann_window_periodic(GEMMA4_FRAME)
    magnitude = np.abs(np.fft.rfft(frames, n=GEMMA4_FFT, axis=-1))
    mel = magnitude @ htk_mel_filters().T
    out = np.log(mel + GEMMA4_MEL_FLOOR)
    # valid if the frame's (unused) last sample t * hop + 320 of the padded signal is input
    mask = np.arange(n_frames) * GEMMA4_HOP + GEMMA4_FRAME < GEMMA4_FRAME // 2 + n
    out = out * mask[:, None]
    return {"frames": frames, "magnitude": magnitude, "mel": mel, "mask": mask, "out": out}


def gemma4_log_mel(audio: np.ndarray, n_padded: int | None = None) -> np.ndarray:
    return gemma4_log_mel_stages(audio, n_padded)["out"]


def gemma4_transformers(audios: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """transformers' ``Gemma4AudioFeatureExtractor`` with the E2B / E4B checkpoints' settings
    (``processor_config.json``), called as ``Gemma4Processor`` does. Returns ``(features
    (B, T, 128), mask sums (B,))``. Needs transformers >= 5.5 (not a dependency group: the
    ``qwen`` group pins an older version)."""
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    fe = Gemma4AudioFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        frame_length_ms=20.0,
        hop_length_ms=10.0,
        min_frequency=0.0,
        max_frequency=8000.0,
        preemphasis=0.0,
        fft_overdrive=False,
        dither=0.0,
        input_scale_factor=1.0,
        mel_floor=0.001,
        padding_value=0.0,
        return_attention_mask=True,
    )
    out = fe([np.asarray(a, dtype=np.float32) for a in audios], return_tensors="np")
    return out["input_features"], out["input_features_mask"].sum(-1)


def load_whisper_mel_filters(path: str | Path, n_mels: int) -> np.ndarray:
    """Load ``mel_{n_mels}`` from whisper's ``assets/mel_filters.npz`` (float32)."""
    with np.load(path, allow_pickle=False) as f:
        return f[f"mel_{n_mels}"]


def whisper_log_mel_torch(
    audio: np.ndarray, n_mels: int, pad_or_trim: bool, mel_npz: str | Path
) -> np.ndarray:
    """Verbatim port of openai/whisper ``log_mel_spectrogram`` (MIT), float32 torch.

    Needs the ``ref`` dependency group.
    """
    import torch

    if pad_or_trim:
        audio = whisper_pad_or_trim(audio)
    x = torch.from_numpy(np.array(audio, dtype=np.float32))  # copy: inputs may be read-only
    filters = torch.from_numpy(load_whisper_mel_filters(mel_npz, n_mels))
    window = torch.hann_window(WHISPER_N_FFT)
    stft = torch.stft(x, WHISPER_N_FFT, WHISPER_HOP, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy()


# ---------------------------------------------------------------------------------------------
# Kaldi fbank (torchaudio.compliance.kaldi.fbank) as configured by FunASR WavFrontend
# ---------------------------------------------------------------------------------------------

KALDI_FRAME_LENGTH = 400  # 25 ms
KALDI_FRAME_SHIFT = 160  # 10 ms
KALDI_PADDED = 512  # round_to_power_of_two
KALDI_PREEMPH = 0.97
FLT_EPSILON = float(np.finfo(np.float32).eps)  # torch.finfo(torch.float).eps


def hamming_window_symmetric(n: int) -> np.ndarray:
    """``torch.hamming_window(n, periodic=False, alpha=0.54, beta=0.46)``."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _kaldi_mel(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def kaldi_mel_banks(
    n_mels: int = 80,
    padded: int = KALDI_PADDED,
    sr: int = SAMPLE_RATE,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> np.ndarray:
    """``torchaudio.compliance.kaldi.get_mel_banks`` (no VTLN) plus the zero Nyquist column.

    Returns ``(n_mels, padded // 2 + 1)``.
    """
    nyquist = 0.5 * sr
    if high_freq <= 0.0:
        high_freq += nyquist
    num_fft_bins = padded // 2
    fft_bin_width = sr / padded
    mel_low = _kaldi_mel(low_freq)
    mel_high = _kaldi_mel(high_freq)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)

    b = np.arange(n_mels, dtype=np.float64)[:, None]
    left = mel_low + b * mel_delta
    center = mel_low + (b + 1.0) * mel_delta
    right = mel_low + (b + 2.0) * mel_delta
    mel = _kaldi_mel(fft_bin_width * np.arange(num_fft_bins))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))


def kaldi_num_frames(n_samples: int) -> int:
    if n_samples < KALDI_FRAME_LENGTH:
        return 0
    return 1 + (n_samples - KALDI_FRAME_LENGTH) // KALDI_FRAME_SHIFT


def kaldi_fbank_stages(
    audio: np.ndarray,
    n_mels: int = 80,
    scale: float = 32768.0,
    banks: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Kaldi fbank with FunASR's settings: hamming, dither=0, snip_edges, no energy.

    ``audio`` is a float waveform in [-1, 1]; it is multiplied by ``scale`` first
    (FunASR's ``waveform * (1 << 15)``). ``banks`` overrides the mel filter bank.
    """
    x = np.asarray(audio, dtype=np.float64) * scale
    n_frames = kaldi_num_frames(x.shape[0])
    idx = np.arange(KALDI_FRAME_LENGTH)[None, :] + KALDI_FRAME_SHIFT * np.arange(n_frames)[:, None]
    frames = x[idx].reshape(n_frames, KALDI_FRAME_LENGTH)

    frames = frames - frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)  # replicate-pad by one
    frames = frames - KALDI_PREEMPH * prev
    frames = frames * hamming_window_symmetric(KALDI_FRAME_LENGTH)
    padded = np.pad(frames, ((0, 0), (0, KALDI_PADDED - KALDI_FRAME_LENGTH)))

    power = np.abs(np.fft.rfft(padded, axis=1)) ** 2
    if banks is None:
        banks = kaldi_mel_banks(n_mels)
    mel = power @ banks.T
    out = np.log(np.maximum(mel, FLT_EPSILON))
    return {"frames": frames, "power": power, "banks": banks, "mel": mel, "out": out}


def kaldi_fbank(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Raw kaldi log-mel fbank, ``(T, n_mels)``."""
    return kaldi_fbank_stages(audio, n_mels)["out"]


# ---------------------------------------------------------------------------------------------
# FunASR LFR / CMVN
# ---------------------------------------------------------------------------------------------


def apply_lfr(inputs: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """FunASR ``apply_lfr``: stack ``lfr_m`` frames every ``lfr_n`` frames."""
    t, dim = inputs.shape
    if t == 0:
        return np.zeros((0, dim * lfr_m), dtype=inputs.dtype)
    t_lfr = int(np.ceil(t / lfr_n))
    left = (lfr_m - 1) // 2
    inputs = np.vstack([np.repeat(inputs[:1], left, axis=0), inputs])
    t += left
    out = []
    for i in range(t_lfr):
        if lfr_m <= t - i * lfr_n:
            out.append(inputs[i * lfr_n : i * lfr_n + lfr_m].reshape(-1))
        else:  # last LFR frame: pad with the final input frame
            num_padding = lfr_m - (t - i * lfr_n)
            frame = inputs[i * lfr_n :].reshape(-1)
            out.append(np.concatenate([frame] + [inputs[-1]] * num_padding))
    return np.vstack(out)


def apply_cmvn(inputs: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """FunASR ``apply_cmvn``: ``(x + shift) * scale``."""
    dim = inputs.shape[1]
    return (inputs + shift[:dim]) * scale[:dim]


def load_cmvn(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """FunASR ``load_cmvn``: parse ``<AddShift>`` / ``<Rescale>`` from a Kaldi nnet ``am.mvn``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    shift: list[str] = []
    scale: list[str] = []
    for i, line in enumerate(lines):
        item = line.split()
        if not item:
            continue
        if item[0] in ("<AddShift>", "<Rescale>"):
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                values = nxt[3 : len(nxt) - 1]
                if item[0] == "<AddShift>":
                    shift = values
                else:
                    scale = values
    return np.array(shift, dtype=np.float32), np.array(scale, dtype=np.float32)


def sensevoice_fbank(
    audio: np.ndarray,
    cmvn: tuple[np.ndarray, np.ndarray] | None = None,
    n_mels: int = 80,
    lfr_m: int = 7,
    lfr_n: int = 6,
) -> np.ndarray:
    """FunASR ``WavFrontend`` features, ``(ceil(T / lfr_n), n_mels * lfr_m)``."""
    feats = apply_lfr(kaldi_fbank(audio, n_mels), lfr_m, lfr_n)
    if cmvn is not None:
        feats = apply_cmvn(feats, np.asarray(cmvn[0], np.float64), np.asarray(cmvn[1], np.float64))
    return feats


# ---------------------------------------------------------------------------------------------
# ffmpeg load pipeline (docs/load.md)
# ---------------------------------------------------------------------------------------------

FFMPEG_MAX_EXACT_PHASES = 1024
FFMPEG_FILTER_SIZE = 32
FFMPEG_CUTOFF = 0.97
FFMPEG_KAISER_BETA = 9.0

_SQRT_HALF = float(np.sqrt(0.5))
FFMPEG_DOWNMIX_WEIGHT = {
    "FC": 1.0,
    **dict.fromkeys(["FL", "FR", "FLC", "FRC", "TFL", "TFR"], _SQRT_HALF),
    **dict.fromkeys(["BL", "BR", "SL", "SR", "BC"], 0.5),
}  # anything else (LFE, top, wide, ...) -> 0

FFMPEG_DEFAULT_LAYOUTS = {
    1: ["FC"],
    2: ["FL", "FR"],
    3: ["FL", "FR", "LFE"],
    4: ["FL", "FR", "FC", "BC"],
    5: ["FL", "FR", "FC", "BL", "BR"],
    6: ["FL", "FR", "FC", "LFE", "BL", "BR"],
    7: ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
}


def ffmpeg_downmix_weights(layout: list[str]) -> np.ndarray:
    """Normalized mono downmix weights for a channel layout (list of position names)."""
    if len(layout) == 1:
        return np.ones(1)
    w = np.array([FFMPEG_DOWNMIX_WEIGHT.get(c, 0.0) for c in layout])
    total = np.abs(w).sum()
    return w / total if total > 1 else w


def ffmpeg_resample_params(sr_in: int, sr_out: int = SAMPLE_RATE) -> dict:
    """Filter parameters: taps N, cutoff fc, phase count, exact mode, ratio p/q."""
    from fractions import Fraction
    from math import ceil

    fc = min(1.0, FFMPEG_CUTOFF * sr_out / sr_in)
    n_taps = ceil(FFMPEG_FILTER_SIZE / fc)
    n_taps += n_taps % 2
    ratio = Fraction(sr_in, sr_out)
    exact = ratio.denominator <= FFMPEG_MAX_EXACT_PHASES
    phases = ratio.denominator if exact else FFMPEG_MAX_EXACT_PHASES
    return {
        "n_taps": n_taps,
        "fc": fc,
        "phases": phases,
        "exact": exact,
        "p": ratio.numerator,
        "q": ratio.denominator,
    }


def _ffmpeg_taps(frac: float, n_taps: int, fc: float) -> np.ndarray:
    """Unnormalized taps for x[b + r], r = -N/2+1 .. N/2, output at b + frac."""
    r = np.arange(-(n_taps // 2) + 1, n_taps // 2 + 1)
    d = frac - r
    sinc = np.sinc(fc * d)  # np.sinc(x) = sin(pi x) / (pi x)
    w = d / (n_taps / 2)
    win = np.i0(FFMPEG_KAISER_BETA * np.sqrt(np.clip(1.0 - w * w, 0.0, None)))
    return sinc * np.where(np.abs(w) <= 1, win, 0.0)


def ffmpeg_filter_bank(sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """(phases + 1, N) float64 filter bank; row m is the filter for frac = m / phases.

    Every row is divided by the tap sum of phase 0. The extra last row is phase 0 shifted by one
    input sample (used by interpolated mode; in exact mode it equals the frac = 1 position).
    """
    prm = ffmpeg_resample_params(sr_in, sr_out)
    n_taps, fc, phases = prm["n_taps"], prm["fc"], prm["phases"]
    bank = np.stack([_ffmpeg_taps(m / phases, n_taps, fc) for m in range(phases)])
    bank /= bank[0].sum()
    shifted = np.concatenate([[0.0], bank[0, :-1]])
    return np.vstack([bank, shifted])


def ffmpeg_out_len(n: int, sr_in: int, sr_out: int = SAMPLE_RATE) -> int:
    """Output length used by asrfront (ffmpeg's own rule is not reproducible exactly)."""
    if sr_in == sr_out:
        return n
    if n == 0:
        return 0
    m = n - 1 if sr_in > 2 * sr_out else n
    return -(-m * sr_out // sr_in)


def _fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Correctly rounded float32 fused multiply-add (vectorized).

    a*b is exact in float64; the float64 sum can only mis-round when it lands exactly on a
    float32 midpoint, which is fixed using the exact sum error (TwoSum).
    """
    p = a.astype(np.float64) * b.astype(np.float64)
    c64 = c.astype(np.float64)
    s = p + c64
    bb = s - p
    err = (p - (s - bb)) + (c64 - bb)
    r = s.astype(np.float32)
    d = s - r.astype(np.float64)
    toward = np.where(d > 0, np.float32(np.inf), np.float32(-np.inf))
    nb = np.nextafter(r, toward)
    tie = (d != 0) & (s == (r.astype(np.float64) + nb.astype(np.float64)) / 2) & (err != 0)
    fix = tie & (np.sign(err) == np.sign(d))
    return np.where(fix, nb, r)


def _dot8_fma(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise float32 dot product: 8 FMA accumulators + tree reduction (docs/load.md)."""
    n, taps = h.shape
    width = -(-taps // 8) * 8
    hp = np.zeros((n, width), np.float32)
    xp = np.zeros((n, width), np.float32)
    hp[:, :taps] = h
    xp[:, :taps] = x
    acc = np.zeros((n, 8), np.float32)
    for i in range(0, width, 8):
        acc = _fma32(hp[:, i : i + 8], xp[:, i : i + 8], acc)
    s = acc[:, :4] + acc[:, 4:]
    s = s[:, :2] + s[:, 2:]
    return s[:, 0] + s[:, 1]


def _mirror_index(idx: np.ndarray, n: int) -> np.ndarray:
    """Start: x[-j] = x[j]; end: x[n-1+j] = x[n-j]."""
    for _ in range(4):  # repeated for inputs shorter than the filter
        idx = np.where(idx < 0, -idx, idx)
        idx = np.where(idx > n - 1, 2 * n - 1 - idx, idx)
    return np.clip(idx, 0, n - 1)


def ffmpeg_resample(
    x: np.ndarray, sr_in: int, path: str, sr_out: int = SAMPLE_RATE, block: int = 32768
) -> np.ndarray:
    """Resample one channel like ffmpeg.

    path="float": x float32 in [-1, 1] -> float32;  "double": float64 -> float64;
    "int16": integer s16 values -> int64 s16 values (Q15 taps, saturated).
    """
    prm = ffmpeg_resample_params(sr_in, sr_out)
    bank = ffmpeg_filter_bank(sr_in, sr_out)
    n = len(x)
    n_out = ffmpeg_out_len(n, sr_in, sr_out)
    p, q, phases, exact = prm["p"], prm["q"], prm["phases"], prm["exact"]
    r = np.arange(-(prm["n_taps"] // 2) + 1, prm["n_taps"] // 2 + 1)
    if path == "float":
        bank_c = bank.astype(np.float32)
        out = np.empty(n_out, np.float32)
    elif path == "double":
        bank_c = bank
        out = np.empty(n_out, np.float64)
    elif path == "int16":
        # Q15 taps are stored as int16: a tap of exactly 1.0 (upsampling phase 0) becomes 32767.
        bank_c = np.clip(np.rint(bank * 32768.0), -32768, 32767).astype(np.int64)
        out = np.empty(n_out, np.int64)
    else:
        raise ValueError(path)
    x = np.asarray(x)
    for k0 in range(0, n_out, block):
        k = np.arange(k0, min(n_out, k0 + block), dtype=np.int64)
        num = k * p
        b, rem = num // q, num % q
        xs = x[_mirror_index(b[:, None] + r[None, :], n)]
        if exact:
            rows = (rem * phases) // q
            h = bank_c[rows]
            if path == "float":
                out[k] = _dot8_fma(h, xs.astype(np.float32))
            elif path == "double":
                out[k] = np.einsum("ij,ij->i", h, xs)
            else:
                acc = (h * xs.astype(np.int64)).sum(axis=1)
                out[k] = np.clip((acc + 16384) >> 15, -32768, 32767)
        else:
            fp = rem * phases
            m = fp // q
            a = (fp - m * q) / q
            if path == "float":
                xs32 = xs.astype(np.float32)
                v1 = _dot8_fma(bank_c[m], xs32)
                v2 = _dot8_fma(bank_c[m + 1], xs32)
                out[k] = v1 + (v2 - v1) * a.astype(np.float32)
            elif path == "double":
                h = bank_c[m] + (bank_c[m + 1] - bank_c[m]) * a[:, None]
                out[k] = np.einsum("ij,ij->i", h, xs)
            else:  # interpolate the accumulators (<= 1 LSB vs. ffmpeg, docs/load.md)
                xi = xs.astype(np.int64)
                a1 = (bank_c[m] * xi).sum(axis=1)
                a2 = (bank_c[m + 1] * xi).sum(axis=1)
                acc = a1 + ((a2 - a1) * (fp - m * q)) // q
                out[k] = np.clip((acc + 16384) >> 15, -32768, 32767)
    return out


FFMPEG_SAMPLE_FORMATS = ("u8", "s16", "s32", "f32", "f64")


def ffmpeg_load_ref(
    samples: np.ndarray,
    sample_fmt: str,
    sr: int,
    layout: list[str] | None = None,
) -> np.ndarray:
    """whisper ``load_audio`` result as int16, from decoded ``(n, channels)`` samples.

    samples dtype: u8 -> uint8, s16 -> int16, s32 -> int32 (24-bit audio in the top bits),
    f32 -> float32, f64 -> float64. ``layout`` defaults to ffmpeg's layout for the channel count.
    """
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    weights = ffmpeg_downmix_weights(layout or FFMPEG_DEFAULT_LAYOUTS[ch])
    resample = sr != SAMPLE_RATE

    def sat(v: np.ndarray) -> np.ndarray:
        return np.clip(v, -32768, 32767).astype(np.int16)

    def to_s16(v: np.ndarray) -> np.ndarray:  # float/double -> s16, round half to even
        with np.errstate(invalid="ignore", over="ignore"):
            r = np.rint(v.astype(np.float64) * 32768.0)
            if v.dtype == np.float32:  # SIMD conversion: NaN / outside int32 -> -32768
                ok = (r >= -(2.0**31)) & (r < 2.0**31)
                return np.where(ok, np.clip(np.nan_to_num(r), -32768, 32767), -32768).astype(
                    np.int16
                )
            # scalar conversion: 64-bit lrint (NaN / out of range -> INT64_MIN), low 32 bits
            ok = (r >= -(2.0**63)) & (r < 2.0**63)
            l64 = np.where(ok, np.nan_to_num(r), 0).astype(np.int64)
            l64 = np.where(ok, l64, np.iinfo(np.int64).min)
            return sat(l64.astype(np.int32).astype(np.int64))

    int_path = sample_fmt == "u8" or (sample_fmt == "s16" and not resample)
    if int_path:
        xi = (x.astype(np.int64) - 128) << 8 if sample_fmt == "u8" else x.astype(np.int64)
        if resample:
            xi = np.stack([ffmpeg_resample(xi[:, c], sr, "int16") for c in range(ch)], axis=1)
        if ch == 1:
            return sat(xi[:, 0])
        qw = np.rint(weights * 32768.0).astype(np.int64)
        # Not saturated: wraps to int16 when the Q15 weights sum above 32768 (7/8 channels).
        return ((xi @ qw + 16384) >> 15).astype(np.int16)

    if sample_fmt == "s32" and ch == 1 and not resample:
        return (x[:, 0].astype(np.int64) >> 16).astype(np.int16)

    if sample_fmt == "f64":
        xf, dt, path = x.astype(np.float64), np.float64, "double"
    elif sample_fmt == "s32":
        xf, dt, path = x.astype(np.float32) * np.float32(2.0**-31), np.float32, "float"
    elif sample_fmt == "s16":
        xf, dt, path = x.astype(np.float32) / np.float32(32768.0), np.float32, "float"
    elif sample_fmt == "f32":
        xf, dt, path = x.astype(np.float32), np.float32, "float"
    else:
        raise ValueError(f"unknown sample_fmt {sample_fmt!r}")
    if resample:
        xf = np.stack([ffmpeg_resample(xf[:, c], sr, path) for c in range(ch)], axis=1)
    if ch == 1:
        return to_s16(xf[:, 0])
    acc = np.zeros(xf.shape[0], dt)
    for c in range(ch):
        acc = (acc + xf[:, c] * dt(weights[c])).astype(dt)
    return to_s16(acc)


def ffmpeg_load_float(samples: np.ndarray, sample_fmt: str, sr: int, layout=None) -> np.ndarray:
    """``ffmpeg_load_ref(...) / 32768`` as float32, i.e. whisper's ``load_audio`` output."""
    return ffmpeg_load_ref(samples, sample_fmt, sr, layout).astype(np.float32) / np.float32(32768)
