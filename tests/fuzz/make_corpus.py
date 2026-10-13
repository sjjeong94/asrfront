"""Regenerate the fuzzing seed corpus (tests/fuzz/corpus): one small file per container / codec
variant. Needs the `load` dependency group (ffmpeg):

    uv run --group load python tests/fuzz/make_corpus.py
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from test_load_formats import (  # noqa: E402
    MS_COEFS,
    aiff,
    aiff_chan,
    caf,
    caf_chunk,
    channel_layout,
    ima_block,
    ms_block,
    qt_packet,
    random_bytes,
    sphere,
    w64,
    wav,
)

OUT = HERE / "corpus"


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def encode(name: str, codec: str, sr: int, ch: int, n: int, *extra: str) -> None:
    rng = np.random.default_rng(len(name))
    t = np.arange(n)[:, None] / sr
    x = 0.3 * rng.standard_normal((n, ch)) + 0.4 * np.sin(2 * np.pi * 440 * t)
    cmd = [
        ffmpeg_exe(),
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f64le",
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        "-i",
        "-",
        "-acodec",
        codec,
        "-map_metadata",
        "-1",
        *extra,
        str(OUT / name),
    ]
    subprocess.run(cmd, input=x.tobytes(), check=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    # ffmpeg-written files
    for name, codec, sr, ch, n, extra in [
        ("u8.wav", "pcm_u8", 8000, 1, 200, ()),
        ("s16_stereo.wav", "pcm_s16le", 16000, 2, 200, ()),
        ("s24_44k.wav", "pcm_s24le", 44100, 1, 150, ()),
        ("s32.wav", "pcm_s32le", 16000, 1, 100, ()),
        ("f32_48k.wav", "pcm_f32le", 48000, 2, 100, ()),
        ("f64.wav", "pcm_f64le", 16000, 1, 60, ()),
        ("alaw.wav", "pcm_alaw", 8000, 1, 300, ()),
        ("mulaw.wav", "pcm_mulaw", 8000, 2, 200, ()),
        ("s16_6ch.wav", "pcm_s16le", 16000, 6, 60, ()),
        ("rf64.wav", "pcm_s16le", 16000, 1, 200, ("-rf64", "always")),
        ("ima.wav", "adpcm_ima_wav", 16000, 2, 1100, ("-block_size", "256")),
        ("ms.wav", "adpcm_ms", 22050, 1, 1100, ("-block_size", "256")),
        ("s16.w64", "pcm_s16le", 16000, 2, 200, ()),
        ("ima.w64", "adpcm_ima_wav", 8000, 1, 1100, ("-block_size", "256")),
        ("s16be.aiff", "pcm_s16be", 44100, 2, 200, ()),
        ("s8.aiff", "pcm_s8", 16000, 1, 300, ()),
        ("fl32.aiff", "pcm_f32be", 16000, 1, 100, ()),
        ("sowt.aiff", "pcm_s16le", 16000, 1, 200, ()),
        ("ima4.aiff", "adpcm_ima_qt", 16000, 2, 256, ()),
        ("s16le.caf", "pcm_s16le", 16000, 1, 200, ()),
        ("s24be_6ch.caf", "pcm_s24be", 48000, 6, 50, ()),
        ("f64le.caf", "pcm_f64le", 16000, 1, 60, ()),
        ("ulaw.caf", "pcm_mulaw", 8000, 1, 300, ()),
        ("ima4.caf", "adpcm_ima_qt", 16000, 1, 256, ()),
        ("s16.flac", "flac", 16000, 1, 400, ("-sample_fmt", "s16")),
        (
            "s24_stereo.flac",
            "flac",
            44100,
            2,
            300,
            ("-sample_fmt", "s32", "-bits_per_raw_sample", "24"),
        ),
        ("5ch.flac", "flac", 16000, 5, 200, ("-sample_fmt", "s16")),
        ("cbr.mp3", "libmp3lame", 16000, 1, 1500, ("-b:a", "32k")),
        ("vbr_stereo.mp3", "libmp3lame", 44100, 2, 2400, ("-q:a", "5")),
        ("no_xing.mp3", "libmp3lame", 8000, 1, 1500, ("-b:a", "16k", "-write_xing", "0")),
    ]:
        encode(name, codec, sr, ch, n, *extra)

    # hand-built files: header variants ffmpeg does not write
    pcm = np.random.default_rng(0).integers(-20000, 20000, 200).astype(">i2").tobytes()
    seeds = {
        "aiff_chan_labels.aiff": aiff(
            pcm[:300], ch=3, before=aiff_chan(channel_layout(0, labels=(1, 2, 3)))
        ),
        "aifc_in24_offset.aiff": aiff(pcm[:300], bits=24, comp=b"in24", offset=4, rate=22050.5),
        "caf_chan_bitmap.caf": caf(
            pcm, ch=2, bytes_per_packet=4, before=caf_chunk(b"chan", channel_layout(1 << 16, 3))
        ),
        "caf_streamed.caf": caf(pcm, data_size=-1),
        "caf_ima4_stereo.caf": caf(
            b"".join(qt_packet(i * 256, 20, random_bytes(32, i)) for i in range(4)),
            fmt=b"ima4",
            bytes_per_packet=68,
            frames_per_packet=64,
            ch=2,
        ),
        "sphere_s16le.sph": sphere(pcm[::-1][:256], header_size=512),
        "sphere_ulaw_stereo.sph": sphere(
            random_bytes(256),
            header_size=512,
            sample_coding="-s4 ulaw",
            sample_n_bytes="-i 1",
            channel_count="-i 2",
            sample_rate="-i 8000",
        ),
        "sphere_s8_be.sph": sphere(
            random_bytes(256),
            header_size=512,
            sample_n_bytes="-i 1",
            sample_byte_format="-s2 10",
        ),
        "ms_stereo.wav": wav(
            0x02,
            2,
            4,
            46,
            b"".join(
                bytes([1, 4])
                + struct.pack("<hhhhhh", 300, 40, 1000, -1000, 0, 5)
                + random_bytes(32, i)
                for i in range(3)
            ),
            MS_COEFS,
        ),
        "ima_3ch.wav": wav(
            0x11,
            3,
            4,
            48,
            b"".join(
                b"".join(ima_block(c * 1000, 30, b"") for c in range(3)) + random_bytes(36, b)
                for b in range(2)
            ),
        ),
        "ms_fmt_only.wav": wav(0x02, 1, 4, 15, ms_block(3, 20, 100, -100, [7, 8] * 8), MS_COEFS),
        "w64_ext.w64": w64(
            struct.pack("<HHIIHHHHI", 0xFFFE, 2, 16000, 64000, 4, 16, 22, 16, 3)
            + bytes.fromhex("0100000000001000800000aa00389b71"),
            pcm,
        ),
        "raw.pcm": pcm[:128],
    }
    # regressions
    flac = bytearray((OUT / "5ch.flac").read_bytes())
    total = (1 << 28) - 1  # STREAMINFO total samples (36 bits at byte 21): just below the cap
    flac[21] = (flac[21] & 0xF0) | (total >> 32)  # that once sized the first buffer (5.4 GB)
    flac[22:26] = (total & 0xFFFFFFFF).to_bytes(4, "big")
    seeds["regress_flac_huge_total.flac"] = bytes(flac)
    for name, data in seeds.items():
        (OUT / name).write_bytes(data)
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"{len(list(OUT.iterdir()))} files, {total} bytes in {OUT}")


if __name__ == "__main__":
    main()
