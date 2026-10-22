# asrfront

Super-fast frontend (feature extraction) for ASR models such as **Whisper**,
**SenseVoiceSmall**, **Qwen3-ASR** and the audio input of **Gemma 4** (E2B / E4B).

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
mel = asrfront.qwen3_asr.log_mel(audio)  # (128, T)
mel = asrfront.gemma4.log_mel(audio)  # (T, 128)
```

## Install

```bash
pip install asrfront
```

Wheels: Linux (x86_64, aarch64), macOS (arm64, x86_64), Windows (x86_64); CPython 3.9+
(one `abi3` wheel covers 3.12 and later) and free-threaded CPython 3.14t, which keeps running
without the GIL. Building from source needs a C11/C++17 compiler.

## Usage

`audio` is a 16 kHz mono waveform: a float array in [-1, 1], int16 PCM, a list, or a CPU
torch tensor — e.g. from `asrfront.load`.

### Loading audio

```python
audio = asrfront.load("speech.wav")  # path (str / os.PathLike)
audio = asrfront.load(request_body)  # bytes / bytearray / memoryview / uint8 array
audio = asrfront.load("stream.pcm")  # raw s16le 16 kHz mono (.pcm / .raw, or headerless bytes)
audio = asrfront.load(upload, max_duration=600)  # ValueError if longer than 10 min
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
  containers, big-endian RIFX WAV. The error names the reason, e.g. `unsupported audio format
  (WAV: format tag 0x0055 (MP3))` or `(MP4 / M4A container)`
- input rates 1 kHz – 384 kHz, any channel layout up to 8 channels (18 with a WAV channel mask)
- all formats but MP3: bit-identical to ffmpeg 7.0.2 at all common rates; MP3 (different
  decoder): within 1/32768, same length. Details and the few documented deviations:
  [`docs/load.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/load.md)
- for SenseVoice: identical to FunASR's own loading at 16 kHz; for other rates FunASR resamples
  with torchaudio, and the features differ less than between two standard resamplers
- untrusted input: malformed files fail with `ValueError` (the loader is fuzzed in CI), but
  memory use grows with the *decoded* length, which a small file can make huge — a 23 kB FLAC
  of 8-channel 384 kHz silence decodes to ~360 MB (ffmpeg behaves the same). Services that load
  user uploads should pass `max_duration` (seconds): longer audio raises `ValueError`, before
  it is decoded for uncompressed and ADPCM formats, at the limit for FLAC / MP3
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

### Qwen3-ASR

```python
from asrfront import qwen3_asr

mel = qwen3_asr.log_mel(audio)  # (128, len(audio) // 160), no 30 s pad / trim
feats, lengths = qwen3_asr.log_mel_batch([a1, a2])  # (2, 128, T) zero-padded, valid frames
feats, lengths = qwen3_asr.log_mel_batch([a1, a2], hf_padding=True)  # transformers batches
n_tokens = qwen3_asr.num_audio_tokens(lengths)  # audio placeholders in the prompt
```

Equivalent to the `WhisperFeatureExtractor` (128 mels) of Qwen3-ASR-0.6B / 1.7B as
`Qwen3ASRProcessor` calls it (`padding=True`, `truncation=False`). By default every item is
computed on its own (as vLLM does); `hf_padding=True` reproduces the transformers backend,
which zero-pads a batch to its longest item before the STFT, so an item's last frames depend
on the batch. Details: [`docs/qwen3_asr.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/qwen3_asr.md).

### Gemma 4 (E2B / E4B)

```python
from asrfront import gemma4

mel = gemma4.log_mel(audio)  # (T, 128), time first; at most 30 s, like transformers
feats, lengths = gemma4.log_mel_batch([a1, a2])  # (2, T, 128) zero frames after `lengths`
mask = np.arange(feats.shape[1]) < lengths[:, None]  # input_features_mask
n_tokens = gemma4.num_audio_tokens(lengths)  # audio soft tokens in the prompt (<= 750)
```

Equivalent to transformers' `Gemma4AudioFeatureExtractor` with the settings both
[gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) and
[gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) ship, as `Gemma4Processor`
calls it: 128 HTK mel bins of the magnitude spectrum, `log(mel + 0.001)`, 20 ms frames every
10 ms; inputs truncated to 30 s (`truncation=False` keeps all), a batch padded to a multiple of
128 samples, and the padding's frames zeroed. An item's features do not depend on the batch.
Details: [`docs/gemma4.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/gemma4.md).

### Threads

All native calls release the GIL. The `*_batch` functions spread items over a thread pool
(`n_threads=None` uses every available core); each Python thread uses its own native handle.
`n_threads` is an upper bound: a batch with little audio in total (about 2 s per thread) uses
fewer threads, since handing short items between threads costs more than computing them. Items
are processed in chunks with one GIL release each.

## Performance

30 s of audio, single thread, Intel i5-14400F (AVX2), best of N runs
(`uv run --group ref --group bench --group load python bench/compare.py`):

| | asrfront | reference | speed-up |
|---|---|---|---|
| Whisper log-mel (80 × 3000) | 0.85–0.95 ms | openai-whisper `log_mel_spectrogram` (torch) 8.6–9.6 ms | ~10x |
| | | librosa STFT + mel (numpy) 6.6–7.0 ms | ~7.7x |
| SenseVoice fbank + LFR + CMVN | 1.3–1.4 ms | FunASR `WavFrontend` (torchaudio) 12–14 ms | ~9–10x |
| kaldi fbank only | 1.3 ms | `torchaudio.compliance.kaldi.fbank` 11–12 ms | ~9x |
| Gemma 4 log-mel (2999 × 128) | 1.9 ms | transformers 5.20.2 `Gemma4AudioFeatureExtractor` (numpy) 21 ms | ~11x |
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
| Qwen3-ASR (vs. transformers 4.57.6, float32) | 6.7e-5 | 2.0e-5 | 1e-4 |
| kaldi fbank (log domain) | 7.7e-4 | 2.0e-4 | 1e-3 |
| SenseVoice features with SenseVoiceSmall's `am.mvn` | 1.1e-4 | | |
| Gemma 4 log-mel, bins above the mel floor (mel ≥ 0.01) | 1.6e-4 | 1.6e-4 | 5e-4 |
| Gemma 4 log-mel, all bins | 6.4e-3 | 1.1e-3 | 1e-2 |

The whisper mel filters are bit-identical to whisper's `mel_filters.npz`. Against the
original float32 libraries: ≤ 1e-4 vs. whisper (torch) and ≤ 2e-3 vs. torchaudio's kaldi
fbank (torchaudio builds its mel banks in float32). Gemma 4's `log(mel + 0.001)` magnifies
float32 FFT round-off (~1e-6 of a frame's peak) in bins far below the 0.001 floor, where the
features are ≈ log(0.001) = −6.9; there asrfront differs from transformers (whose numpy FFT
is more precise) by up to 6.3e-3 on pure tones ([`docs/gemma4.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/gemma4.md)).

## Development

```bash
uv sync                       # create .venv and build the extension (needs a C compiler)
uv run pytest                 # tests; add --group ref for torch/torchaudio comparisons
uv run --group ref --group load pytest   # + comparisons against ffmpeg 7.0.2 (imageio-ffmpeg)
uv run --group ref --group qwen pytest   # + Qwen3-ASR vs. transformers 4.57.6
# Gemma 4 vs. transformers needs transformers >= 5.5 (conflicts with the qwen pin): run
# tests/python/test_gemma4.py in an environment with it and asrfront installed
# on aarch64 / Apple silicon ffmpeg's float path differs by 1 LSB from x86, so those
# comparisons allow 1 LSB there (docs/load.md, "Other platforms")
uv run cmake -S . -B build/ctest -G Ninja -DASRFRONT_BUILD_TESTS=ON -DASRFRONT_BUILD_BENCH=ON
uv run cmake --build build/ctest && uv run ctest --test-dir build/ctest
```

The C API is in [`csrc/include/asrfront.h`](https://github.com/sjjeong94/asrfront/blob/main/csrc/include/asrfront.h). The preprocessing of each
model is documented step by step in [`docs/whisper.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/whisper.md),
[`docs/sensevoice.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/sensevoice.md),
[`docs/qwen3_asr.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/qwen3_asr.md) and
[`docs/gemma4.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/gemma4.md); see [`docs/design.md`](https://github.com/sjjeong94/asrfront/blob/main/docs/design.md) for design
decisions.

## License

MIT. Test data in `tests/data` comes from [openai/whisper](https://github.com/openai/whisper)
(MIT); see [`tests/data/README.md`](https://github.com/sjjeong94/asrfront/blob/main/tests/data/README.md). FLAC and MP3 decoding use the vendored
[dr_flac / dr_mp3](https://github.com/mackron/dr_libs) (public domain or MIT-0); see
[`csrc/third_party/README.md`](https://github.com/sjjeong94/asrfront/blob/main/csrc/third_party/README.md).
The binary extension statically links [nanobind](https://github.com/wjakob/nanobind)
(BSD-3-Clause, with tsl::robin_map, MIT). The license texts of these and of the projects the
test reference code is ported from are in
[`LICENSES/`](https://github.com/sjjeong94/asrfront/blob/main/LICENSES/README.md) and are
included in the wheels.
