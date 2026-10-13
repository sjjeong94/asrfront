"""asrfront.load: AIFF / AIFF-C, CAF, Wave64, NIST SPHERE and ADPCM codecs (M9.6).

Hand-built files test the container rules (docs/load.md, "Containers"); the second part compares
with ffmpeg (``load`` dependency group), both on files ffmpeg writes and on crafted ones that
exercise header quirks, damaged sizes and ADPCM corner cases.
"""

from __future__ import annotations

import math
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

import asrfront

import reference as ref
from audio_signals import DATA_DIR
from test_load import (
    _ffmpeg,
    assert_ffmpeg_equal,
    assert_matches_ffmpeg,
    ffmpeg_write,
    noise_s16,
    signal,
    whisper_load_audio,
)

# ---------------------------------------------------------------------------------------------
# File builders
# ---------------------------------------------------------------------------------------------


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def ext80(rate: float) -> bytes:
    """80-bit IEEE extended (AIFF COMM sample rate)."""
    e = math.frexp(rate)[1] - 1
    return struct.pack(">HQ", e + 16383, int(rate * 2 ** (63 - e)))


def aiff(
    body: bytes,
    ch: int = 1,
    bits: int = 16,
    rate: float = 16000,
    comp: bytes | None = None,
    offset: int = 0,
    ssnd_size: int | None = None,
    before: bytes = b"",
    after: bytes = b"",
    comm_last: bool = False,
) -> bytes:
    """AIFF (comp=None) or AIFF-C with compression type `comp`; `before` / `after` are extra
    chunks placed around SSND."""

    def chunk(cid: bytes, data: bytes, size: int | None = None) -> bytes:
        n = len(data) if size is None else size
        return cid + struct.pack(">I", n) + data + (b"\0" if len(data) & 1 else b"")

    comm = struct.pack(">hIh", ch, 0, bits) + ext80(rate)
    if comp is not None:
        comm += comp + b"\0\0"
    ssnd = chunk(b"SSND", struct.pack(">II", offset, 0) + bytes(offset) + body, ssnd_size)
    parts = [before, ssnd, after]
    parts.insert(len(parts) if comm_last else 0, chunk(b"COMM", comm))
    if comp is not None:
        parts.insert(0, chunk(b"FVER", struct.pack(">I", 0xA2805140)))
    data = b"".join(parts)
    return b"FORM" + struct.pack(">I", len(data) + 4) + (b"AIFC" if comp else b"AIFF") + data


def caf(
    body: bytes,
    fmt: bytes = b"lpcm",
    flags: int = 0,
    bytes_per_packet: int = 2,
    frames_per_packet: int = 1,
    ch: int = 1,
    bits: int = 16,
    rate: float = 16000.0,
    data_size: int | None = None,
    before: bytes = b"",
    after: bytes = b"",
) -> bytes:
    desc = struct.pack(">d4sIIIII", rate, fmt, flags, bytes_per_packet, frames_per_packet, ch, bits)
    data = struct.pack(">I", 0) + body  # edit count
    n = len(data) if data_size is None else data_size
    return (
        b"caff\x00\x01\x00\x00desc"
        + struct.pack(">q", len(desc))
        + desc
        + before
        + b"data"
        + struct.pack(">q", n)
        + data
        + after
    )


def caf_chunk(cid: bytes, data: bytes) -> bytes:
    return cid + struct.pack(">q", len(data)) + data


def channel_layout(tag: int = 0, bitmap: int = 0, labels: tuple[int, ...] = ()) -> bytes:
    """CoreAudio AudioChannelLayout (CAF 'chan' / AIFF 'CHAN' body)."""
    desc = b"".join(struct.pack(">IIfff", lab, 0, 0, 0, 0) for lab in labels)
    return struct.pack(">III", tag, bitmap, len(labels)) + desc


def aiff_chan(layout: bytes) -> bytes:
    return b"CHAN" + struct.pack(">I", len(layout)) + layout


SPHERE_FIELDS = {
    "sample_count": "-i 1000",
    "sample_n_bytes": "-i 2",
    "channel_count": "-i 1",
    "sample_byte_format": "-s2 01",
    "sample_rate": "-i 16000",
    "sample_coding": "-s3 pcm",
}


def sphere(body: bytes, header_size: int = 1024, size_line: bytes | None = None, **fields) -> bytes:
    """NIST SPHERE; keyword arguments override SPHERE_FIELDS (None drops a field)."""
    f = {**SPHERE_FIELDS, **fields}
    h = b"NIST_1A\n" + (b"%7d\n" % header_size if size_line is None else size_line)
    for k, v in f.items():
        if v is not None:
            h += f"{k} {v}\n".encode()
    h += b"end_head\n"
    assert len(h) <= header_size
    return h.ljust(header_size, b" ") + body


def wav(
    tag: int,
    ch: int,
    bits: int,
    block_align: int,
    body: bytes,
    ext: bytes = b"",
    mask: int | None = None,
) -> bytes:
    """WAV with an arbitrary format tag (e.g. ADPCM) and cbSize extension `ext`, or with
    WAVE_FORMAT_EXTENSIBLE and channel mask `mask` (then `ext` is not written)."""
    if mask is not None:
        guid = struct.pack("<H", tag) + bytes.fromhex("000000001000800000aa00389b71")
        ext = struct.pack("<HI", bits, mask) + guid
        tag = 0xFFFE
    fmt = struct.pack("<HHIIHHH", tag, ch, 16000, 8000 * block_align, block_align, bits, len(ext))
    fmt += ext
    data = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    data += b"data" + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(data)) + data


def w64(fmt_body: bytes, body: bytes, data_size: int | None = None) -> bytes:
    tail = bytes.fromhex("f3acd3118cd100c04f8edb8a")

    def chunk(fourcc: bytes, data: bytes, size: int | None = None) -> bytes:
        n = len(data) + 24 if size is None else size
        return fourcc + tail + struct.pack("<Q", n) + data + bytes(-len(data) % 8)

    riff = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
    chunks = chunk(b"fmt ", fmt_body) + chunk(b"data", body, data_size)
    return riff + struct.pack("<Q", 40 + len(chunks)) + b"wave" + tail + chunks


def pcm_fmt(ch: int, bits: int, rate: int = 16000) -> bytes:
    ba = ch * bits // 8
    return struct.pack("<HHIIHH", 1, ch, rate, rate * ba, ba, bits)


# ADPCM blocks
MS_COEFS = struct.pack("<HH", 500, 7) + b"".join(
    struct.pack("<hh", a, b)
    for a, b in [(256, 0), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232)]
)


def ms_block(pred: int, delta: int, s1: int, s2: int, nibbles: list[int]) -> bytes:
    data = bytes((nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2))
    return bytes([pred]) + struct.pack("<hhh", delta, s1, s2) + data


def ima_block(predictor: int, index: int, data: bytes, reserved: int = 0) -> bytes:
    return struct.pack("<hBB", predictor, index, reserved) + data


def qt_packet(predictor: int, index: int, data: bytes) -> bytes:
    return struct.pack(">H", (predictor & 0xFF80) | index) + data


def random_bytes(n: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


# ---------------------------------------------------------------------------------------------
# API behavior (no ffmpeg needed)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ch", [1, 2])
@pytest.mark.parametrize("container", ["aiff", "aifc_sowt", "caf", "caf_le", "w64", "sphere"])
def test_s16_containers_match_reference(tmp_path, container, ch):
    x = noise_s16(3000, ch, seed=ch)
    be = x.astype(">i2").tobytes()
    le = x.astype("<i2").tobytes()
    data = {
        "aiff": lambda: aiff(be, ch=ch),
        "aifc_sowt": lambda: aiff(le, ch=ch, comp=b"sowt"),
        "caf": lambda: caf(be, ch=ch, bytes_per_packet=2 * ch),
        "caf_le": lambda: caf(le, ch=ch, flags=2, bytes_per_packet=2 * ch),
        "w64": lambda: w64(pcm_fmt(ch, 16), le),
        "sphere": lambda: sphere(le, channel_count=f"-i {ch}"),
    }[container]()
    expected = ref.ffmpeg_load_float(x, "s16", 16000)
    np.testing.assert_array_equal(asrfront.load(data), expected)
    path = tmp_path / "a.wav"  # detection is by content, not by extension
    path.write_bytes(data)
    np.testing.assert_array_equal(asrfront.load(path), expected)


def test_8bit_signed_containers():
    """AIFF, CAF and SPHERE 8-bit PCM is signed (WAV's is unsigned)."""
    x = np.array([0, 1, 127, -128, -1], np.int8)
    expected = x.astype(np.float32) * 256 / 32768
    for data in (
        aiff(x.tobytes(), bits=8),
        caf(x.tobytes(), bits=8, bytes_per_packet=1),
        sphere(x.tobytes(), sample_n_bytes="-i 1"),
    ):
        np.testing.assert_array_equal(asrfront.load(data), expected)


@pytest.mark.parametrize(
    "data",
    [
        aiff(b"\0" * 64, comp=b"QDM2"),
        caf(b"\0" * 64, fmt=b"aac "),
        caf(b"\0" * 64, flags=1, bits=16),  # half float
        sphere(b"\0" * 64, sample_coding="-s26 pcm,embedded-shorten-v2.00"),
        sphere(b"\0" * 64, sample_n_bytes="-i 3"),
        wav(0x11, 1, 3, 8, ima_block(0, 0, b"\0" * 4)),  # 3-bit IMA
        wav(0x02, 3, 4, 21 + 8, b"\0" * 29, MS_COEFS),  # MS ADPCM, 3 channels
        b"RIFX" + bytes(4) + b"WAVEfmt " + bytes(64),  # big-endian WAV
    ],
    ids=[
        "aifc-qdm2",
        "caf-aac",
        "caf-f16",
        "sphere-shorten",
        "sphere-24bit",
        "ima-3bit",
        "ms-3ch",
        "rifx",
    ],
)
def test_unsupported(tmp_path, data):
    # a path, since bytes without a known header are raw PCM
    with pytest.raises(ValueError, match="unsupported"):
        asrfront.load(write(tmp_path, "x.wav", data))


@pytest.mark.parametrize(
    "data",
    [
        aiff(b"\0" * 64, ssnd_size=0),
        aiff(b"\0" * 64, ssnd_size=5),
        aiff(b"\0" * 64, ch=0),
        caf(b"\0" * 64, bytes_per_packet=0),
        caf(b"\0" * 64, ch=0),
        b"caff\x00\x01\x00\x00free" + struct.pack(">q", 0) + caf(b"\0" * 4)[8:],  # desc not 1st
        sphere(b"\0" * 64, sample_rate=None),
        sphere(b"\0" * 64, channel_count="-i 0"),
        sphere(b"\0" * 64, size_line=b"   abcd\n"),
        sphere(b"\0" * 64)[:500],  # file shorter than its header
        sphere(b"\0" * 64)[:100],  # no end_head
        w64(pcm_fmt(1, 16), b"\0" * 64, data_size=24),
        wav(0x02, 1, 4, 15, ms_block(7, 100, 0, 0, [1] * 16), MS_COEFS),  # predictor > 6
        wav(0x11, 1, 4, 8, ima_block(0, 89, b"\0" * 4)),  # step index > 88
        wav(0x11, 1, 4, 8, ima_block(0, 10, b"\0" * 4, reserved=1)),
        wav(0x11, 1, 4, 6, ima_block(0, 10, b"\0" * 2)),  # block not whole 4-byte groups
        aiff(qt_packet(0, 89, b"\0" * 32), comp=b"ima4"),
    ],
    ids=[
        "ssnd0",
        "ssnd5",
        "aiff-ch0",
        "caf-bpp0",
        "caf-ch0",
        "caf-desc-not-first",
        "sph-no-rate",
        "sph-ch0",
        "sph-size-line",
        "sph-short",
        "sph-no-end",
        "w64-data24",
        "ms-pred7",
        "ima-index89",
        "ima-reserved",
        "ima-align",
        "qt-index89",
    ],
)
def test_malformed(data):
    with pytest.raises(ValueError, match="malformed"):
        asrfront.load(data)


TRUNCATION_FILES = {
    "aiff": lambda: aiff(noise_s16(300, 2).astype(">i2").tobytes(), ch=2),
    "aifc-ima4": lambda: aiff(
        b"".join(qt_packet(0, 20, random_bytes(32, i)) for i in range(4)), ch=2, comp=b"ima4"
    ),
    "caf": lambda: caf(
        random_bytes(600),
        ch=2,
        bytes_per_packet=4,
        before=caf_chunk(b"chan", channel_layout(101 << 16 | 2)),
    ),
    "w64": lambda: w64(pcm_fmt(2, 16), random_bytes(600)),
    "sphere": lambda: sphere(random_bytes(600), channel_count="-i 2"),
    "ima-wav": lambda: wav(
        0x11,
        2,
        4,
        40,
        b"".join(
            ima_block(i * 100, 20, b"") + ima_block(-i * 100, 30, b"") + random_bytes(32, i)
            for i in range(4)
        ),
    ),
    "ms-wav": lambda: wav(
        0x02,
        2,
        4,
        40,
        b"".join(bytes([1, 2]) + random_bytes(12, i) + random_bytes(26, i) for i in range(4)),
        MS_COEFS,
    ),
}


@pytest.mark.parametrize("name", list(TRUNCATION_FILES))
def test_truncated_files(name):
    """Every prefix fails cleanly or decodes no more than the full file."""
    data = TRUNCATION_FILES[name]()
    full = len(asrfront.load(data))
    for cut in range(len(data)):
        try:
            y = asrfront.load(data[:cut])
        except ValueError:
            continue
        assert len(y) <= full and np.all(np.isfinite(y))


def test_coreaudio_layout_fallbacks():
    """Layouts ffmpeg cannot downmix (non-native channel order, wrong channel count, short
    description list) use the default layout for the channel count."""
    x = noise_s16(2000, 8, seed=3)
    body = x.astype(">i2").tobytes()
    expected = asrfront.load(aiff(body, ch=8))
    for layout in (
        channel_layout(128 << 16 | 8),  # MPEG_7_1_C: FL FR FC LFE SL SR BL BR
        channel_layout(101 << 16 | 2),  # stereo tag on 8 channels
        channel_layout(0, labels=(1, 2)),  # 2 descriptions for 8 channels
        channel_layout(0xFFFF << 16 | 8),  # unknown tag
    ):
        np.testing.assert_array_equal(
            asrfront.load(aiff(body, ch=8, before=aiff_chan(layout))), expected
        )
    with pytest.raises(ValueError, match="malformed"):  # chunk shorter than its header
        asrfront.load(aiff(body, ch=8, before=aiff_chan(b"\0" * 8)))


# ---------------------------------------------------------------------------------------------
# Against ffmpeg
# ---------------------------------------------------------------------------------------------


def check_crafted(tmp_path: Path, name: str, data: bytes) -> None:
    """Crafted file at 16 kHz: bit-exact (or within 1 LSB on non-reference ffmpeg builds)."""
    path = write(tmp_path, name, data)
    assert_ffmpeg_equal(asrfront.load(path), whisper_load_audio(path))


CONTAINER_CODECS = {
    "aiff": [
        "pcm_s8",
        "pcm_s16be",
        "pcm_s24be",
        "pcm_s32be",
        "pcm_f32be",
        "pcm_f64be",
        "pcm_alaw",
        "pcm_mulaw",
        "pcm_s16le",
        "adpcm_ima_qt",
    ],
    "caf": [
        "pcm_s8",
        "pcm_s16be",
        "pcm_s16le",
        "pcm_s24be",
        "pcm_s24le",
        "pcm_s32le",
        "pcm_f32be",
        "pcm_f32le",
        "pcm_f64le",
        "pcm_alaw",
        "pcm_mulaw",
        "adpcm_ima_qt",
    ],
    "w64": [
        "pcm_u8",
        "pcm_s16le",
        "pcm_s24le",
        "pcm_s32le",
        "pcm_f32le",
        "pcm_f64le",
        "pcm_alaw",
        "pcm_mulaw",
        "adpcm_ima_wav",
        "adpcm_ms",
    ],
    "wav": ["adpcm_ima_wav", "adpcm_ms"],
}


@pytest.mark.parametrize("sr", [16000, 44100])
@pytest.mark.parametrize("ch", [1, 2, 6])
@pytest.mark.parametrize(
    "ext,codec", [(e, c) for e, codecs in CONTAINER_CODECS.items() for c in codecs]
)
def test_ffmpeg_containers(tmp_path, ext, codec, ch, sr):
    """Files written by ffmpeg (incl. its CHAN / chan layout chunks)."""
    if codec == "adpcm_ms" and ch > 2:
        ch = 2
    path = ffmpeg_write(tmp_path / f"a.{ext}", signal(sr // 3 + 7, ch, ch + sr), codec, sr=sr)
    # f64 with >= 5 channels: <= 1 LSB in rare near-tie samples (docs/load.md, Downmix)
    tol = 1 if codec.startswith("pcm_f64") and ch >= 5 else 0
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), sr, tol_lsb=tol)


@pytest.mark.parametrize(
    "layout",
    [
        "stereo",
        "2.1",
        "3.0",
        "3.0(back)",
        "quad",
        "4.0",
        "5.0",
        "5.1",
        "6.1",
        "7.1(wide)",
        "FL+FR+FLC+FRC",
        "FL+FR+TFL+TFR",
        "FC+LFE",
    ],
)
@pytest.mark.parametrize("ext", ["aiff", "caf"])
def test_ffmpeg_coreaudio_layouts_written(tmp_path, ext, layout):
    n_ch = {
        "stereo": 2,
        "2.1": 3,
        "3.0": 3,
        "3.0(back)": 3,
        "quad": 4,
        "4.0": 4,
        "5.0": 5,
        "5.1": 6,
        "6.1": 7,
        "7.1(wide)": 8,
    }.get(layout, layout.count("+") + 1)
    codec = "pcm_s16be" if ext == "aiff" else "pcm_s16le"
    path = ffmpeg_write(tmp_path / f"a.{ext}", signal(6000, n_ch, n_ch), codec, layout=layout)
    assert_ffmpeg_equal(asrfront.load(path), whisper_load_audio(path))


LAYOUT_CASES = {
    # predefined tags whose downmix differs from the default layout
    "quadraphonic": (channel_layout(108 << 16 | 4), 4),
    "cube": (channel_layout(112 << 16 | 8), 8),
    "mpeg_3_0_a": (channel_layout(113 << 16 | 3), 3),
    "smpte_dtv": (channel_layout(130 << 16 | 8), 8),
    "itu_2_1": (channel_layout(131 << 16 | 3), 3),
    "itu_2_2": (channel_layout(132 << 16 | 4), 4),
    "dvd_5": (channel_layout(134 << 16 | 4), 4),
    "dvd_6": (channel_layout(135 << 16 | 5), 5),
    "dvd_10": (channel_layout(136 << 16 | 4), 4),
    "dvd_11": (channel_layout(137 << 16 | 5), 5),
    "ac3_1_0_1": (channel_layout(149 << 16 | 2), 2),
    # same as the default layout
    "mono": (channel_layout(100 << 16 | 1), 1),
    "matrix_stereo": (channel_layout(103 << 16 | 2), 2),
    "mpeg_5_1_a": (channel_layout(121 << 16 | 6), 6),
    "count_mismatch": (channel_layout(121 << 16 | 6), 4),
    "bitmap": (channel_layout(1 << 16, 0x33), 4),
    "bitmap_mismatch": (channel_layout(1 << 16, 0x3), 3),
    # channel descriptions (labels)
    "labels_5_1": (channel_layout(0, labels=(1, 2, 3, 4, 5, 6)), 6),
    "labels_extra": (channel_layout(0, labels=(1, 2, 3)), 2),
    **{
        f"label_{lab}": (channel_layout(0, labels=(1, 2, lab)), 3)
        for lab in (3, 4, 9, 12, 14, 16, 17, 18, 35, 36, 37, 38, 39)
    },
}


@pytest.mark.parametrize("case", list(LAYOUT_CASES))
@pytest.mark.parametrize("container", ["aiff", "caf"])
def test_ffmpeg_coreaudio_layouts(tmp_path, container, case):
    layout, ch = LAYOUT_CASES[case]
    body = noise_s16(3000, ch, seed=ch).reshape(3000, ch).astype(">i2").tobytes()
    if container == "aiff":
        data = aiff(body, ch=ch, before=aiff_chan(layout))
    else:
        data = caf(body, ch=ch, bytes_per_packet=2 * ch, before=caf_chunk(b"chan", layout))
    check_crafted(tmp_path, f"l.{container}", data)


AIFC_BODY = random_bytes(480, seed=7)


@pytest.mark.parametrize("bits", [1, 8, 12, 16, 20, 24, 32])
@pytest.mark.parametrize(
    "comp",
    [
        None,
        b"NONE",
        b"none",
        b"twos",
        b"sowt",
        b"raw ",
        b"in24",
        b"in32",
        b"fl32",
        b"FL32",
        b"fl64",
        b"FL64",
        b"alaw",
        b"ALAW",
        b"ulaw",
        b"ULAW",
    ],
)
def test_ffmpeg_aiff_codec_tags(tmp_path, comp, bits):
    """The compression type decides the codec; for NONE / twos the bit depth rounds up to
    whole bytes, other types ignore it."""
    body = AIFC_BODY
    if comp in (b"fl32", b"FL32", b"fl64", b"FL64"):  # finite floats
        dt = ">f4" if comp.lower() == b"fl32" else ">f8"
        body = (np.random.default_rng(bits).standard_normal(60) * 0.4).astype(dt).tobytes()
    check_crafted(tmp_path, "c.aiff", aiff(body, bits=bits, comp=comp))


@pytest.mark.parametrize(
    "case",
    [
        "offset",
        "odd_offset",
        "size_ffff",
        "size_large",
        "size_small",
        "size_odd",
        "truncated",
        "comm_after_ssnd",
        "chunk_before",
        "chunk_after",
        "stereo_partial_frame",
    ],
)
def test_ffmpeg_aiff_chunks(tmp_path, case):
    body = noise_s16(400, seed=4).astype(">i2").tobytes()
    data = {
        "offset": lambda: aiff(body, offset=6),
        "odd_offset": lambda: aiff(body, offset=7),
        "size_ffff": lambda: aiff(body, ssnd_size=0xFFFFFFFF),
        "size_large": lambda: aiff(body, ssnd_size=len(body) + 1000),
        "size_small": lambda: aiff(body, ssnd_size=108),
        "size_odd": lambda: aiff(body, ssnd_size=107),
        "truncated": lambda: aiff(body)[:-51],
        "comm_after_ssnd": lambda: aiff(body, comm_last=True),
        "chunk_before": lambda: aiff(body, before=b"NAME\0\0\0\3abc\0"),
        "chunk_after": lambda: aiff(body, after=b"NAME\0\0\0\4abcd"),
        "stereo_partial_frame": lambda: aiff(body + b"\1\2", ch=2),
    }[case]()
    check_crafted(tmp_path, "c.aiff", data)


@pytest.mark.parametrize("rate", [8000, 22050, 22050.5, 22050.4999, 44100, 8000.25])
def test_ffmpeg_aiff_rates(tmp_path, rate):
    """The 80-bit rate is rounded to the nearest integer, halves up (22050.5 -> 22051, a
    non-standard rate: <= 1 LSB, docs/load.md)."""
    body = noise_s16(int(rate) // 4, seed=5).astype(">i2").tobytes()
    path = write(tmp_path, "r.aiff", aiff(body, rate=rate))
    sr = int(rate + 0.5)
    tol = 0 if sr in (8000, 22050, 44100) else 1
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), sr, tol_lsb=tol)


CAF_BODY = noise_s16(500, seed=6).astype(">i2").tobytes()


@pytest.mark.parametrize(
    "case",
    [
        "s8_flags0",
        "s8_flags2",
        "s12",
        "s24_bpp4",
        "s16_bpp4",
        "f32le",
        "f64be",
        "alaw_bits16",
        "fpp2",
        "size_m1",
        "size_0",
        "size_4",
        "size_small",
        "size_odd",
        "size_large",
        "edit_count",
        "chunk_before",
        "chunk_after",
        "two_data",
        "truncated",
        "rate_fraction",
        "stereo",
    ],
)
def test_ffmpeg_caf_chunks(tmp_path, case):
    b = CAF_BODY
    floats = (np.random.default_rng(1).standard_normal(250) * 0.4).astype(np.float32)
    free = caf_chunk(b"free", b"abcd")
    data = {
        "s8_flags0": lambda: caf(b, bits=8, bytes_per_packet=1),
        "s8_flags2": lambda: caf(b, bits=8, bytes_per_packet=1, flags=2),
        "s12": lambda: caf(b, bits=12),
        "s24_bpp4": lambda: caf(b, bits=24, bytes_per_packet=4),  # bit depth wins
        "s16_bpp4": lambda: caf(b, bits=16, bytes_per_packet=4),
        "f32le": lambda: caf(floats.astype("<f4").tobytes(), flags=3, bits=32, bytes_per_packet=4),
        "f64be": lambda: caf(floats.astype(">f8").tobytes(), flags=1, bits=64, bytes_per_packet=8),
        "alaw_bits16": lambda: caf(b, fmt=b"alaw", bits=16, bytes_per_packet=1),
        "fpp2": lambda: caf(b, frames_per_packet=2, bytes_per_packet=4),
        "size_m1": lambda: caf(b, data_size=-1),
        "size_0": lambda: caf(b, data_size=0, after=free),  # to EOF, trailing chunk included
        "size_4": lambda: caf(b, data_size=4),
        "size_small": lambda: caf(b, data_size=104, after=free),
        "size_odd": lambda: caf(b, data_size=51),
        "size_large": lambda: caf(b, data_size=10**6),
        "edit_count": lambda: caf(b)[: -len(b) - 4] + struct.pack(">I", 7) + b,
        "chunk_before": lambda: caf(b, before=free),
        "chunk_after": lambda: caf(b, after=free),
        "two_data": lambda: caf(b, after=caf_chunk(b"data", bytes(4) + b[:100])),  # last wins
        "truncated": lambda: caf(b)[:-51],
        "rate_fraction": lambda: caf(b, rate=16000.75),
        "stereo": lambda: caf(b, ch=2, bytes_per_packet=4),
    }[case]()
    check_crafted(tmp_path, "c.caf", data)


@pytest.mark.parametrize("rate", [8000.0, 22050.5, 44100.0])
def test_ffmpeg_caf_rates(tmp_path, rate):
    """The float64 rate is truncated to an integer."""
    body = noise_s16(int(rate) // 4, seed=8).astype(">i2").tobytes()
    path = write(tmp_path, "r.caf", caf(body, rate=rate))
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), int(rate))


@pytest.mark.parametrize("case", ["size_small", "size_large", "truncated", "trailing_chunk"])
def test_ffmpeg_w64_chunks(tmp_path, case):
    body = noise_s16(1000, 2, seed=9).tobytes()
    data = {
        "size_small": lambda: w64(pcm_fmt(2, 16), body, data_size=1000),
        "size_large": lambda: w64(pcm_fmt(2, 16), body, data_size=1 << 62),
        "truncated": lambda: w64(pcm_fmt(2, 16), body)[:-101],
        "trailing_chunk": lambda: (
            w64(pcm_fmt(2, 16), body) + b"junk" + bytes(12) + struct.pack("<Q", 32) + bytes(8)
        ),
    }[case]()
    check_crafted(tmp_path, "c.w64", data)


SPHERE_BODY = random_bytes(800, seed=10)

SPHERE_CASES = {
    "s16le": {},
    "s16be": {"sample_byte_format": "-s2 10"},
    "format_1": {"sample_byte_format": "-s1 1"},
    "no_format": {"sample_byte_format": None},
    "s8": {"sample_n_bytes": "-i 1", "sample_byte_format": "-s1 1"},
    "ulaw": {"sample_coding": "-s4 ulaw", "sample_n_bytes": "-i 1"},
    "mu-law_2bytes": {"sample_coding": "-s6 mu-law"},  # sample_n_bytes is ignored
    "ULAW": {"sample_coding": "-s4 ULAW", "sample_n_bytes": "-i 1"},
    "alaw_stereo": {"sample_coding": "-s4 alaw", "sample_n_bytes": "-i 1", "channel_count": "-i 2"},
    "no_coding": {"sample_coding": None},
    "PCM": {"sample_coding": "-s3 PCM"},
    "stereo": {"channel_count": "-i 2"},
    "4ch": {"channel_count": "-i 4"},
    "rate_8000": {"sample_rate": "-i 8000"},
    "rate_real": {"sample_rate": "-r 8000.5"},
    "rate_string_type": {"sample_rate": "-s5 16000"},
    "sample_count_ignored": {"sample_count": "-i 10"},
    "prefix_key": {"sample_rate": None, "sample_rate_x": "-i 8000"},
    "extra_fields": {"sample_sig_bits": "-i 12", "database_id": "-s5 TIMIT"},
}


@pytest.mark.parametrize("case", list(SPHERE_CASES))
def test_ffmpeg_sphere(tmp_path, case):
    fields = SPHERE_CASES[case]
    path = write(tmp_path, "s.sph", sphere(SPHERE_BODY, **fields))
    sr = 8000 if any("8000" in str(v) for v in fields.values()) else 16000
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), sr)


def _insert_line(data: bytes, line: bytes) -> bytes:
    """Insert a header line after the size line, keeping the header size (padding shrinks)."""
    return data[:16] + line + data[16 : 1024 - len(line)] + data[1024:]


@pytest.mark.parametrize(
    "case",
    [
        "header_512",
        "header_2048",
        "size_line_1000",
        "last_wins",
        "trailing_odd_byte",
        "end_head_suffix",
    ],
)
def test_ffmpeg_sphere_header(tmp_path, case):
    b = SPHERE_BODY
    data = {
        "header_512": lambda: sphere(b, header_size=512),
        "header_2048": lambda: sphere(b, header_size=2048),
        # the data starts at the header size, whatever end_head's position
        "size_line_1000": lambda: sphere(b, size_line=b"   1000\n"),
        "last_wins": lambda: _insert_line(sphere(b), b"sample_rate -i 8000\n"),
        "trailing_odd_byte": lambda: sphere(b + b"\5"),
        "end_head_suffix": lambda: sphere(b).replace(b"end_head\n", b"end_head junk\n"),
    }[case]()
    check_crafted(tmp_path, "s.sph", data)


@pytest.mark.parametrize("ch", [1, 2, 6])
@pytest.mark.parametrize("codec", ["adpcm_ima_wav", "adpcm_ms"])
@pytest.mark.parametrize("cut", [1, 7, 100, 500, 1000])
def test_ffmpeg_adpcm_partial_block(tmp_path, codec, ch, cut):
    """A truncated last block decodes the whole sample groups it holds."""
    if codec == "adpcm_ms" and ch > 2:
        ch = 2
    data = ffmpeg_write(tmp_path / "a.wav", signal(5000, ch, ch), codec).read_bytes()
    check_crafted(tmp_path, "t.wav", data[: len(data) - cut])


def _ms_wav(*blocks: bytes, ch: int = 1, ext: bytes = MS_COEFS) -> bytes:
    assert len({len(b) for b in blocks}) == 1
    return wav(0x02, ch, 4, len(blocks[0]), b"".join(blocks), ext)


MS_CASES = {
    "negative_delta": lambda: _ms_wav(ms_block(1, -100, 1000, 500, [1, 9, 3, 12] * 4)),
    "delta_growth": lambda: _ms_wav(ms_block(2, 30000, 0, 0, [7] * 400)),
    "delta_growth_alternating": lambda: _ms_wav(ms_block(2, 30000, 0, 0, [7, 8] * 200)),
    "delta_zero": lambda: _ms_wav(ms_block(1, 0, 1000, 500, [1] * 16)),
    # delta is capped at INT_MAX / 768: grow it past that, let it decay, then observe it
    "delta_cap": lambda: _ms_wav(ms_block(2, 30000, 0, 0, [7] * 30 + [0] * 120 + [1, 15] * 25)),
    "custom_coefs_ignored": lambda: _ms_wav(
        ms_block(0, 100, 1000, 500, [1] * 16),
        ext=struct.pack("<HH", 2036 // 2, 7) + struct.pack("<hh", 100, 50) * 7,
    ),
    "no_extension": lambda: _ms_wav(ms_block(4, 77, -3000, 2000, [5, 2] * 16), ext=b""),
    "all_predictors": lambda: _ms_wav(
        *(ms_block(p, 300, 2000 - 500 * p, -1000, list(range(16)) * 2) for p in range(7))
    ),
    "stereo_extremes": lambda: _ms_wav(
        bytes([6, 5])
        + struct.pack("<hhhhhh", 32767, 16, 32767, -32768, -32768, 32767)
        + bytes(range(0, 256, 8)),
        ch=2,
    ),
}


@pytest.mark.parametrize("case", list(MS_CASES))
def test_ffmpeg_ms_adpcm_crafted(tmp_path, case):
    check_crafted(tmp_path, "ms.wav", MS_CASES[case]())


@pytest.mark.parametrize("index", [0, 1, 40, 87, 88])
@pytest.mark.parametrize("ch", [1, 2])
def test_ffmpeg_ima_wav_crafted(tmp_path, ch, index):
    blocks = b"".join(
        b"".join(ima_block(-32768 + 20000 * c + b, index, b"") for c in range(ch))
        + random_bytes(4 * ch * 6, seed=b)
        for b in range(3)
    )
    check_crafted(tmp_path, "ima.wav", wav(0x11, ch, 4, 4 * ch * 7, blocks, b"\x31\x00"))


@pytest.mark.parametrize("ch", [3, 6])
def test_ffmpeg_ima_wav_multichannel(tmp_path, ch):
    """ffmpeg cannot decode IMA ADPCM WAV with more than two channels (it leaves the channel
    layout unset). asrfront decodes it; each channel must equal ffmpeg's decoding of the same
    data as a mono file, downmixed with the default layout."""
    n_blocks, groups = 4, 6
    heads = [
        [ima_block(3000 * c - 9000 + b, 20 + c, b"") for c in range(ch)] for b in range(n_blocks)
    ]
    data = [
        [random_bytes(4 * groups, seed=100 * b + c) for c in range(ch)] for b in range(n_blocks)
    ]

    def block(b: int, chans: range) -> bytes:  # headers, then 4-byte groups interleaved
        out = b"".join(heads[b][c] for c in chans)
        for g in range(groups):
            out += b"".join(data[b][c][4 * g : 4 * g + 4] for c in chans)
        return out

    ba = 4 * ch * (1 + groups)
    multi = wav(0x11, ch, 4, ba, b"".join(block(b, range(ch)) for b in range(n_blocks)))
    per_channel = []
    for c in range(ch):
        mono = wav(
            0x11, 1, 4, ba // ch, b"".join(block(b, range(c, c + 1)) for b in range(n_blocks))
        )
        y = whisper_load_audio(write(tmp_path, f"m{c}.wav", mono))
        per_channel.append(np.round(y * 32768).astype(np.int16))
    expected = ref.ffmpeg_load_float(np.stack(per_channel, axis=1), "s16", 16000)
    np.testing.assert_array_equal(asrfront.load(multi), expected)


@pytest.mark.parametrize("offset", [-0x80, -0x7F, 0, 0x7F, 0x80, 0x100])
@pytest.mark.parametrize("same_index", [True, False])
def test_ffmpeg_ima_qt_resync(tmp_path, offset, same_index):
    """QuickTime IMA keeps its state across packets unless the next header's step index differs
    or its predictor is more than 0x7F away from the current one."""
    first = qt_packet(1024, 30, random_bytes(32, seed=1))
    st = asrfront.load(aiff(first, comp=b"ima4"))
    last = int(round(float(st[-1]) * 32768))
    # header predictors are multiples of 128: pick the one closest to last + offset
    predictor = int(np.clip((last + offset) // 128 * 128, -32768, 32640))
    second = qt_packet(predictor, 30 if same_index else 31, random_bytes(32, seed=2))
    check_crafted(tmp_path, "q.aiff", aiff(first + second, comp=b"ima4"))
    check_crafted(tmp_path, "q2.aiff", aiff(first + second + first[:20], comp=b"ima4"))


@pytest.mark.parametrize(
    "ext,codec,sr,ch",
    [
        ("wav", "adpcm_ima_wav", 44100, 2),
        ("wav", "adpcm_ms", 22050, 1),
        ("aiff", "adpcm_ima_qt", 48000, 2),
        ("caf", "adpcm_ima_qt", 16000, 1),
        ("aiff", "pcm_s24be", 44100, 2),
        ("caf", "pcm_f32le", 48000, 2),
        ("w64", "pcm_s16le", 44100, 2),
    ],
)
def test_ffmpeg_speech_formats(tmp_path, ext, codec, sr, ch):
    """11 s of real speech converted to another container / codec / rate."""
    path = tmp_path / f"speech.{ext}"
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


def test_ffmpeg_sphere_speech(tmp_path):
    """jfk as 8 kHz mu-law SPHERE (the classic telephone-corpus format)."""
    pcm = subprocess.run(
        [
            _ffmpeg(),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(DATA_DIR / "jfk.wav"),
            "-ar",
            "8000",
            "-f",
            "mulaw",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    data = sphere(pcm, sample_coding="-s4 ulaw", sample_n_bytes="-i 1", sample_rate="-i 8000")
    path = write(tmp_path, "speech.sph", data)
    assert_matches_ffmpeg(asrfront.load(path), whisper_load_audio(path), 8000)
