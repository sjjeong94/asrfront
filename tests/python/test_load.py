"""asrfront.load: WAV / raw PCM decoding (M9.2).

The first part only needs Python's ``wave`` module and the reference pipeline in
``reference.py``. The second part (``load`` dependency group, imageio-ffmpeg) writes files with
ffmpeg in every supported codec and compares with whisper's exact ffmpeg command line.
"""

from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import asrfront

import reference as ref
from audio_signals import DATA_DIR


def write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> Path:
    """16-bit PCM WAV via the stdlib (x: int16, (n,) or (n, ch))."""
    x = np.asarray(x, np.int16)
    ch = 1 if x.ndim == 1 else x.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return path


def noise_s16(n: int, ch: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.clip(rng.standard_normal((n, ch)) * 8000, -32768, 32767).astype(np.int16)
    x[:4] = [[32767] * ch, [-32768] * ch, [0] * ch, [1] * ch]
    return x[:, 0] if ch == 1 else x


# ---------------------------------------------------------------------------------------------
# API behavior (no ffmpeg needed)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ch", [1, 2, 6])
def test_wav_s16_matches_reference(tmp_path, ch):
    x = noise_s16(20000, ch, seed=ch)
    path = write_wav(tmp_path / "a.wav", x)
    y = asrfront.load(path)
    assert y.dtype == np.float32 and y.ndim == 1 and y.flags.c_contiguous
    np.testing.assert_array_equal(y, ref.ffmpeg_load_float(x, "s16", 16000))


def test_jfk(tmp_path):
    with wave.open(str(DATA_DIR / "jfk.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    y = asrfront.load(DATA_DIR / "jfk.wav")
    np.testing.assert_array_equal(y, x.astype(np.float32) / 32768)
    np.testing.assert_array_equal(asrfront.load(str(DATA_DIR / "jfk.wav")), y)


def test_bytes_equal_path(tmp_path):
    path = write_wav(tmp_path / "a.wav", noise_s16(5000, 2))
    data = path.read_bytes()
    expected = asrfront.load(path)
    for src in (data, bytearray(data), memoryview(data), np.frombuffer(data, np.uint8)):
        np.testing.assert_array_equal(asrfront.load(src), expected)


def test_raw_pcm(tmp_path):
    x = noise_s16(3001)
    expected = x.astype(np.float32) / 32768
    for ext in (".pcm", ".raw", ".PCM"):
        p = tmp_path / f"a{ext}"
        p.write_bytes(x.tobytes())
        np.testing.assert_array_equal(asrfront.load(p), expected)
    # bytes without a container header are raw PCM
    np.testing.assert_array_equal(asrfront.load(x.tobytes()), expected)
    # odd trailing byte is dropped
    np.testing.assert_array_equal(asrfront.load(x.tobytes() + b"\x01"), expected)
    # a .pcm path is raw even if it starts like a WAV file
    wav = write_wav(tmp_path / "w.wav", x).read_bytes()
    (tmp_path / "w.pcm").write_bytes(wav)
    assert len(asrfront.load(tmp_path / "w.pcm")) == len(wav) // 2


@pytest.mark.parametrize("pattern", [b"\xff\xff", b"\xff\xfb", b"\xff\xfb\x90\x64", b"\x00\x00"])
def test_raw_bytes_not_mistaken_for_mp3(pattern):
    data = pattern * (8192 // len(pattern))
    assert len(asrfront.load(data)) == len(data) // 2


def test_empty():
    assert asrfront.load(b"").shape == (0,)
    assert asrfront.load(b"").dtype == np.float32


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        asrfront.load(tmp_path / "missing.wav")
    garbage = tmp_path / "x.wav"
    garbage.write_bytes(b"not audio at all")
    with pytest.raises(ValueError, match="x.wav.*unsupported"):
        asrfront.load(garbage)
    (tmp_path / "empty.wav").write_bytes(b"")
    with pytest.raises(ValueError):
        asrfront.load(tmp_path / "empty.wav")
    truncated = write_wav(tmp_path / "t.wav", noise_s16(100)).read_bytes()[:30]
    with pytest.raises(ValueError, match="malformed"):
        asrfront.load(truncated)
    with pytest.raises(ValueError, match="unsupported"):  # outside 1 kHz .. 384 kHz
        asrfront.load(write_wav(tmp_path / "r.wav", noise_s16(100), sr=500))
    with pytest.raises(TypeError):
        asrfront.load(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        asrfront.load(np.zeros(10, np.int16))
    with pytest.raises(ValueError, match="contiguous"):
        asrfront.load(memoryview(b"abcdef")[::2])
    with pytest.raises(FileNotFoundError):  # a str is always a path, never data
        asrfront.load("RIFF....WAVE")


def test_concurrent_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    paths = [write_wav(tmp_path / f"{i}.wav", noise_s16(16000, 2, seed=i)) for i in range(8)]
    expected = [asrfront.load(p) for p in paths]
    with ThreadPoolExecutor(8) as ex:
        for got, want in zip(ex.map(asrfront.load, paths * 4), expected * 4):
            np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------------------------
# Against ffmpeg (whisper's load_audio command line)
# ---------------------------------------------------------------------------------------------


def _ffmpeg():
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


def whisper_load_audio(path: Path) -> np.ndarray:
    cmd = [
        _ffmpeg(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768


def ffmpeg_write(
    path: Path,
    x: np.ndarray,
    codec: str,
    layout: str | None = None,
    extra: tuple[str, ...] = (),
    sr: int = 16000,
) -> Path:
    """Encode float64 samples (n, ch) at rate sr to a file with ffmpeg."""
    ch = x.shape[1]
    lay = ["-ch_layout", layout] if layout else ["-ac", str(ch)]
    cmd = [
        _ffmpeg(),
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f64le",
        "-ar",
        str(sr),
        *lay,
        "-i",
        "-",
        "-acodec",
        codec,
        *extra,
        str(path),
    ]
    subprocess.run(cmd, input=np.ascontiguousarray(x).tobytes(), check=True)
    return path


def signal(n: int, ch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None] / 16000
    x = 0.3 * rng.standard_normal((n, ch)) + 0.4 * np.sin(2 * np.pi * 300 * t * (1 + np.arange(ch)))
    x[rng.integers(0, n, 30)] = 1.2  # clipping peaks
    return x


CODECS = [
    "pcm_u8",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
    "pcm_alaw",
    "pcm_mulaw",
]


@pytest.mark.parametrize("ch", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("codec", CODECS)
def test_ffmpeg_codecs(tmp_path, codec, ch):
    if codec in ("pcm_alaw", "pcm_mulaw") and ch > 2:
        ch = 2
    path = ffmpeg_write(tmp_path / "a.wav", signal(12000, ch, ch), codec)
    np.testing.assert_array_equal(asrfront.load(path), whisper_load_audio(path))


@pytest.mark.parametrize(
    "layout",
    [
        "mono",
        "stereo",
        "2.1",
        "3.0",
        "3.0(back)",
        "quad",
        "quad(side)",
        "4.0",
        "5.0",
        "5.0(side)",
        "5.1",
        "5.1(side)",
        "6.1",
        "7.1",
        "7.1(wide)",
        "FL+FR+FLC+FRC",
        "FL+FR+TFL+TFR",
        "FL+FR+FC+LFE+BL+BR+FLC+FRC",
    ],
)
@pytest.mark.parametrize("codec", ["pcm_s16le", "pcm_f32le", "pcm_s24le"])
def test_ffmpeg_layouts(tmp_path, layout, codec):
    """WAVE_FORMAT_EXTENSIBLE channel masks written by ffmpeg."""
    n_ch = {
        "mono": 1,
        "stereo": 2,
        "2.1": 3,
        "3.0": 3,
        "3.0(back)": 3,
        "quad": 4,
        "quad(side)": 4,
        "4.0": 4,
        "5.0": 5,
        "5.0(side)": 5,
        "5.1": 6,
        "5.1(side)": 6,
        "6.1": 7,
        "7.1": 8,
        "7.1(wide)": 8,
    }.get(layout, layout.count("+") + 1)
    path = ffmpeg_write(tmp_path / "a.wav", signal(8000, n_ch, n_ch), codec, layout=layout)
    try:
        expected = whisper_load_audio(path)
    except subprocess.CalledProcessError:
        pytest.skip(f"ffmpeg cannot downmix {layout}")
    np.testing.assert_array_equal(asrfront.load(path), expected)


def test_ffmpeg_rf64(tmp_path):
    path = ffmpeg_write(
        tmp_path / "a.wav", signal(9000, 2, 1), "pcm_s16le", extra=("-rf64", "always")
    )
    assert path.read_bytes()[:4] == b"RF64"
    np.testing.assert_array_equal(asrfront.load(path), whisper_load_audio(path))


def _patch_data_size(data: bytes, value: int) -> bytes:
    i = data.index(b"data")
    return data[: i + 4] + struct.pack("<I", value) + data[i + 8 :]


@pytest.mark.parametrize("case", ["size0", "sizeFFFF", "truncated", "partial_frame", "trailing"])
def test_ffmpeg_damaged_files(tmp_path, case):
    good = ffmpeg_write(tmp_path / "g.wav", signal(4000, 2, 3), "pcm_s16le").read_bytes()
    if case == "size0":
        data = _patch_data_size(good, 0)
    elif case == "sizeFFFF":
        data = _patch_data_size(good, 0xFFFFFFFF)
    elif case == "truncated":
        data = good[: len(good) - 1000]
    elif case == "partial_frame":
        data = good[: len(good) - 3]
    else:  # a trailing chunk after the data
        data = good + b"LIST\x04\x00\x00\x00abcd"
    path = tmp_path / "d.wav"
    path.write_bytes(data)
    np.testing.assert_array_equal(asrfront.load(path), whisper_load_audio(path))


def _float_wav(path: Path, x: np.ndarray) -> Path:
    body = np.ascontiguousarray(x).tobytes()
    bits = x.dtype.itemsize * 8
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, 16000, 16000 * bits // 8, bits // 8, bits))
        f.write(b"data" + struct.pack("<I", len(body)) + body)
    return path


SPECIALS = [np.nan, np.inf, -np.inf, 1e30, -1e30, 1e5, -1e5, 65535.0, -65535.0, 1.5, -1.9, 0.5]


def test_ffmpeg_float_specials_f64(tmp_path):
    """f64 out-of-range / non-finite samples: ffmpeg's scalar rule, reproduced exactly."""
    x = np.repeat(np.array(SPECIALS, np.float64), 37)
    path = _float_wav(tmp_path / "s.wav", x)
    np.testing.assert_array_equal(asrfront.load(path), whisper_load_audio(path))


def test_ffmpeg_float_specials_f32(tmp_path):
    """f32: exact while |x| < 65536; beyond that ffmpeg is position-dependent (docs/load.md)."""
    x = np.repeat(np.array(SPECIALS, np.float32), 37)
    path = _float_wav(tmp_path / "s.wav", x)
    ours, theirs = asrfront.load(path), whisper_load_audio(path)
    with np.errstate(invalid="ignore", over="ignore"):
        wild = ~(np.abs(x.astype(np.float64)) < 65536)
    np.testing.assert_array_equal(ours[~wild], theirs[~wild])
    assert np.all(ours[wild] == -1.0)
    scalar_rule = ref.ffmpeg_load_float(x.astype(np.float64), "f64", 16000)
    assert np.all((theirs[wild] == -1.0) | (theirs[wild] == scalar_rule[wild]))


@pytest.mark.parametrize(
    "ch,mask",
    [
        (2, 0b111),
        (2, 0b1),
        (3, 0b11),
        (2, 0),
        (3, 0),
        (6, 0x3F),
        (6, 0x60F),
        (4, 0x33),
        (2, 1 << 19),
    ],
)
def test_ffmpeg_mismatched_channel_mask(tmp_path, ch, mask):
    """Masks naming too few / too many / unknown positions fall back to the default layout."""
    x = (np.random.default_rng(ch).standard_normal((5000, ch)) * 5000).astype(np.int16)
    body = x.tobytes()
    guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, ch, 16000, 32000 * ch, 2 * ch, 16, 22, 16, mask) + guid
    data = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    data += b"data" + struct.pack("<I", len(body)) + body
    path = tmp_path / "m.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(data)) + data)
    np.testing.assert_array_equal(asrfront.load(path), whisper_load_audio(path))


# ---------------------------------------------------------------------------------------------
# Resampling (M9.3), against ffmpeg
# ---------------------------------------------------------------------------------------------

# Rates whose output length matches ffmpeg exactly (docs/load.md, "Output length").
EXACT_LENGTH_RATES = {8000, 11025, 12000, 24000, 32000, 48000, 96000}


def _host_has_fma() -> bool:
    import platform

    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return " avx2" in flags and " fma" in flags


def assert_matches_ffmpeg(ours: np.ndarray, theirs: np.ndarray, sr: int, tol_lsb: int = 0) -> None:
    """Same samples (to tol_lsb); length exact or (for some rates) the last sample may differ."""
    if sr in EXACT_LENGTH_RATES:
        assert len(ours) == len(theirs)
    else:
        assert abs(len(ours) - len(theirs)) <= 1, (len(ours), len(theirs))
    n = min(len(ours), len(theirs))
    if not _host_has_fma():
        tol_lsb = max(tol_lsb, 1)  # ffmpeg's own float kernel differs without FMA3
    diff = np.abs(ours[:n].astype(np.float64) - theirs[:n]) * 32768
    assert diff.max(initial=0) <= tol_lsb, f"max diff {diff.max()} LSB, {np.sum(diff > 0)} differ"


RATES = [8000, 11025, 22050, 24000, 32000, 44100, 48000, 96000]


@pytest.mark.parametrize("sr", RATES)
@pytest.mark.parametrize("ch", [1, 2, 6])
@pytest.mark.parametrize("codec", ["pcm_u8", "pcm_s16le", "pcm_s24le", "pcm_f32le", "pcm_f64le"])
def test_ffmpeg_resampled(tmp_path, codec, ch, sr):
    path = ffmpeg_write(tmp_path / "a.wav", signal(sr // 2 + 13, ch, sr), codec, sr=sr)
    # f64 with >= 5 channels: <= 1 LSB in rare near-tie samples (docs/load.md, Downmix)
    tol = 1 if codec == "pcm_f64le" and ch >= 5 else 0
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), sr, tol_lsb=tol)


@pytest.mark.parametrize("sr", [16494, 44056])
@pytest.mark.parametrize("codec", ["pcm_u8", "pcm_s16le", "pcm_f64le"])
def test_ffmpeg_resampled_interpolated_mode(tmp_path, codec, sr):
    """Non-standard rates: <= 1 LSB (docs/load.md)."""
    path = ffmpeg_write(tmp_path / "a.wav", signal(sr // 2, 1, sr), codec, sr=sr)
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), sr, tol_lsb=1)


@pytest.mark.parametrize(
    "sr,ch,codec",
    [
        (44100, 2, "pcm_s16le"),
        (48000, 1, "pcm_s24le"),
        (22050, 2, "pcm_f32le"),
        (8000, 1, "pcm_mulaw"),
    ],
)
def test_ffmpeg_speech_resampled(tmp_path, sr, ch, codec):
    """11 s of real speech converted to another rate / layout / codec."""
    path = tmp_path / "speech.wav"
    subprocess.run(
        [
            _ffmpeg(),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(DATA_DIR / "jfk.wav"),
            "-ar",
            str(sr),
            "-ac",
            str(ch),
            "-acodec",
            codec,
            str(path),
        ],
        check=True,
    )
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), sr)
