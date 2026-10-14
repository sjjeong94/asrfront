# asrfront

Super-fast frontend (feature extraction) for ASR models such as **Whisper** and
**SenseVoiceSmall**.

- native C core with no dependencies (own FFT, SIMD kernels with runtime CPU dispatch)
- audio loading: WAV / AIFF / CAF / FLAC / MP3 / raw PCM (and more) → 16 kHz mono, matching
  whisper's ffmpeg-based `load_audio` without spawning ffmpeg
- Python binding based on [nanobind](https://github.com/wjakob/nanobind), zero-copy NumPy output
- numerically matches the reference implementations (see [Accuracy](#accuracy))

```python
import asrfront

audio = asrfront.load("speech.mp3")  # 16 kHz mono float32
mel = asrfront.whisper.log_mel(audio)  # (80, 3000)
feat = asrfront.sensevoice.fbank(audio, cmvn="am.mvn")  # (T', 560)
```

## Install

```bash
pip install asrfront
```

Wheels: Linux (x86_64, aarch64), macOS (arm64, x86_64), Windows (x86_64); CPython 3.9+
(one `abi3` wheel covers 3.12 and later). Building from source needs a C11/C++17 compiler.

## Usage

`audio` is a 16 kHz mono waveform: a float array in [-1, 1], int16 PCM, a list, or a CPU
torch tensor — e.g. from `asrfront.load`.

### Loading audio

```python
audio = asrfront.load("speech.wav")  # path (str / os.PathLike)
audio = asrfront.load(request_body)  # bytes / bytearray / memoryview / uint8 array
audio = asrfront.load("stream.pcm")  # raw s16le 16 kHz mono (.pcm / .raw, or headerless bytes)
```

Returns a 1-D float32 array at 16 kHz mono (`asrfront.SAMPLE_RATE`), the input both Whisper and
SenseVoiceSmall expect — the same as `whisper.load_audio(path)`, i.e.
`ffmpeg -i FILE -f s16le -ac 1 -ar 16000` / 32768, but decoded in-process:

- formats (detected from the content):
  - WAV / RF64 / BW64 and Wave64: PCM 8–32 bit, float, A-law/µ-law, IMA and Microsoft ADPCM
  - AIFF / AIFF-C and CAF (Apple): PCM 8–32 bit (either byte order), float, A-law/µ-law, IMA4
  - NIST SPHERE (TIMIT-style corpora): 8/16-bit PCM, A-law/µ-law (not shorten-compressed)
  - FLAC, MP3
- not supported (→ `ValueError`; convert with ffmpeg first): AAC/M4A, Opus, Vorbis, WMA, video
  containers, big-endian RIFX WAV
- input rates 1 kHz – 384 kHz, any channel layout up to 8 channels (18 with a WAV channel mask)
- all formats but MP3: bit-identical to ffmpeg 7.0.2 at all common rates; MP3 (different
  decoder): within 1/32768, same length. Details and the few documented deviations:
  [`docs/load.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/load.md)
- for SenseVoice: identical to FunASR's own loading at 16 kHz; for other rates FunASR resamples
  with torchaudio, and the features differ less than between two standard resamplers
- untrusted input: malformed files fail with `ValueError` (the loader is fuzzed in CI), but
  memory use grows with the *decoded* length, which a small file can make huge — a 23 kB FLAC
  of 8-channel 384 kHz silence decodes to ~360 MB (ffmpeg behaves the same). Services that load
  user uploads should limit the upload size accordingly, or check the duration first
  ([`docs/load.md`, "Resource use"](https://github.com/sjjeong94/asrfront/blob/main/docs/load.md#resource-use))

### Whisper

```python
from asrfront import whisper

mel = whisper.log_mel(audio)  # (80, 3000): pad/trim to 30 s like whisper
mel = whisper.log_mel(audio, n_mels=128)  # large-v3
mel = whisper.log_mel(audio, pad_or_trim=False)  # (80, len(audio) // 160)

batch = whisper.log_mel_batch([a1, a2, a3])  # (3, 80, 3000), multi-threaded
```

Equivalent to `whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels)`.

### SenseVoice

```python
from asrfront import sensevoice

feat = sensevoice.fbank(audio, cmvn="path/to/am.mvn")  # (ceil(T / 6), 560)
feats, lengths = sensevoice.fbank_batch([a1, a2], cmvn="path/to/am.mvn")  # zero-padded
raw = sensevoice.fbank_raw(audio)  # (T, 80) kaldi fbank only
shift, scale = sensevoice.load_cmvn("path/to/am.mvn")
```

Equivalent to FunASR's `WavFrontend` (`torchaudio.compliance.kaldi.fbank` on the int16-scaled
waveform with a Hamming window, `dither=0`, then LFR `m=7, n=6` and CMVN from `am.mvn`).
`dither=` and `seed=` are available; the RNG is reseeded on every call so identical inputs
give identical outputs.

### Threads

All native calls release the GIL. The `*_batch` functions spread items over a thread pool
(`n_threads=None` uses every available core); each Python thread uses its own native handle.

## Performance

30 s of audio, single thread, Intel i5-14400F (AVX2), best of N runs
(`uv run --group ref --group bench --group load python bench/compare.py`):

| | asrfront | reference | speed-up |
|---|---|---|---|
| Whisper log-mel (80 × 3000) | 0.85–0.95 ms | openai-whisper `log_mel_spectrogram` (torch) 8.6–9.6 ms | ~10x |
| | | librosa STFT + mel (numpy) 6.6–7.0 ms | ~7.7x |
| SenseVoice fbank + LFR + CMVN | 1.3–1.4 ms | FunASR `WavFrontend` (torchaudio) 12–14 ms | ~9–10x |
| kaldi fbank only | 1.3 ms | `torchaudio.compliance.kaldi.fbank` 11–12 ms | ~9x |
| `load`: WAV 16 kHz | 0.3 ms | whisper `load_audio` (ffmpeg subprocess) 10 ms / `librosa.load` 2.5 ms | 33x / 8x |
| `load`: WAV 44.1 kHz stereo | 14 ms | whisper `load_audio` 36 ms / `librosa.load` 31 ms | 2.5x / 2.2x |
| `load`: MP3 44.1 kHz stereo | 25 ms | whisper `load_audio` 38 ms / `librosa.load` 34 ms | 1.5x / 1.4x |

Full results (thread scaling, input lengths, kernel variants, methodology):
[`docs/benchmark.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/benchmark.md).

Kernels process 8 frames at a time in SIMD lanes. On x86-64 an AVX2 (+FMA3) build is selected at
runtime when available (`asrfront._ext.kernel_isa()`); `ASRFRONT_KERNELS=generic` forces the
baseline build. All variants give bit-identical results. Batch of 32 × 30 s: ~7.5 ms with
16 threads.

## Accuracy

Maximum absolute error against float64 reference implementations
([`tests/python/reference.py`](https://github.com/sjjeong94/asrfront/blob/main/tests/python/reference.py)) over 16 test signals (silence,
tones, chirp, noise, clipping, DC offset, real speech, short and > 30 s inputs):

| | worst case | speech | target |
|---|---|---|---|
| Whisper log-mel (80 and 128 mels) | 4.0e-5 | 1.3e-5 | 1e-4 |
| kaldi fbank (log domain) | 7.7e-4 | 2.0e-4 | 1e-3 |
| SenseVoice features with SenseVoiceSmall's `am.mvn` | 1.1e-4 | | |

The whisper mel filters are bit-identical to whisper's `mel_filters.npz`. Against the
original float32 libraries: ≤ 1e-4 vs. whisper (torch) and ≤ 2e-3 vs. torchaudio's kaldi
fbank (torchaudio builds its mel banks in float32).

## Development

```bash
uv sync                       # create .venv and build the extension (needs a C compiler)
uv run pytest                 # tests; add --group ref for torch/torchaudio comparisons
uv run --group ref --group load pytest   # + comparisons against ffmpeg 7.0.2 (imageio-ffmpeg)
# on aarch64 / Apple silicon ffmpeg's float path differs by 1 LSB from x86, so those
# comparisons allow 1 LSB there (docs/load.md, "Other platforms")
uv run cmake -S . -B build/ctest -G Ninja -DASRFRONT_BUILD_TESTS=ON -DASRFRONT_BUILD_BENCH=ON
uv run cmake --build build/ctest && uv run ctest --test-dir build/ctest
```

The C API is in [`csrc/include/asrfront.h`](https://github.com/sjjeong94/asrfront/blob/main/csrc/include/asrfront.h). The preprocessing of each
model is documented step by step in [`docs/whisper.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/whisper.md) and
[`docs/sensevoice.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/sensevoice.md); see [`docs/design.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/design.md) for design
decisions.

## License

MIT. Test data in `tests/data` comes from [openai/whisper](https://github.com/openai/whisper)
(MIT); see [`tests/data/README.md`](https://github.com/sjjeong94/asrfront/blob/main/tests/data/README.md). FLAC and MP3 decoding use the vendored
[dr_flac / dr_mp3](https://github.com/mackron/dr_libs) (public domain or MIT-0); see
[`csrc/third_party/README.md`](https://github.com/sjjeong94/asrfront/blob/main/csrc/third_party/README.md).
