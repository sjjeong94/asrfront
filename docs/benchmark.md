# Benchmark

Qwen3-ASR features (`asrfront.qwen3_asr`): see [`qwen3_asr.md`](qwen3_asr.md#performance).

## Apple M1 dither and blocked int16 / double resampling (2026-10-29)

Baseline: commit `23f1748`. Same machine as below (Apple M1, macOS 26.5.2 arm64, Apple Clang,
Python 3.12). Best of 20–30 calls through the Python API.

- **SenseVoice dither**: Box–Muller uses in-house, branch-free `ln` / `sin` / `cos` instead of
  four libm calls per pair, so the per-frame loop vectorizes (a `switch` on the random quadrant
  cost about half the gain in a first version: an unpredictable branch). The
  features for every tested seed are bit for bit unchanged ([`sensevoice.md`](sensevoice.md#dither)).
  The remaining cost is mostly the serial xorshift64* chain (~4 ms per 30 s), which cannot
  change without changing every seed's output.
- **Resampled u8 (int16 path) and f64 (double path) audio** is resampled and downmixed block
  by block, like the float path, instead of through full-length per-channel buffers. Output
  is bit for bit unchanged (u8 and f64, 1 / 2 / 6 channels, 7 input rates, 5 lengths).

| Measurement | Before | After |
|---|---:|---:|
| `sensevoice.fbank`, 30 s, `dither=0` | 3.17 ms | 3.16 ms |
| `sensevoice.fbank`, 30 s, `dither=1.0` | 21.18 ms | 14.87 ms |
| `load`, 600 s 48 kHz 6-channel u8 WAV: peak memory beyond input | 220 MiB | 37 MiB (= the output) |
| `load`, 60 s 48 kHz 6-channel u8 WAV: time | 93.5 ms | 94.7 ms |

## Apple M1 resampler and FFT updates (2026-10-20)

Baseline: commit `6ec1b7f`. Environment: Apple M1 (4 performance + 4 efficiency cores),
8 GB RAM, macOS 26.5.2 arm64, Apple Clang 21.0.0, CMake Release, Python 3.12.14,
NumPy 2.5.3. The `generic` kernel uses NEON on this platform.

Two changes preserve the arithmetic of the scalar implementation:

- The float resampler uses explicit 128-bit NEON FMA instructions. Clang previously
  generated four 64-bit FMA instructions per eight taps; it now uses two 128-bit
  instructions, keeping the same eight accumulator chains and reduction tree.
- Real FFT split post-processing computes bins `k` and `m-k` together, reusing their
  input loads. Both bins retain their original twiddles and subtraction order. This
  reduces memory access without changing the butterfly stages.

The new paths are limited to aarch64 builds with vector extensions; x86 and the
plain-C `scalar` implementation keep their previous code. Other ARM CPUs have not
been benchmarked locally.

Validation: Python suite **1883 passed, 19 skipped**, including the optional
torch / torchaudio, transformers and ffmpeg comparisons. C tests **4/4 passed**
in both Release and ASan/UBSan builds. New tests compare float resampling against
the generic and scalar kernels exactly across 18 input rates, including
interpolated-phase and extreme-rate cases.

### Audio loading

30 s JFK speech, re-encoded with ffmpeg 7.1. Timings are the median of 30 calls per
build, alternating baseline and updated calls after warming their resampler caches.
These measurements call `af_load_memory` through ctypes and include decoding,
resampling, downmix, output allocation and freeing. File reading and the Python
`asrfront.load` wrapper are excluded. Every output sample of each fixture matches
the baseline bit for bit (480,000 samples per file).

| Input | Before (ms) | After (ms) | Time reduction |
|---|---:|---:|---:|
| WAV 16 kHz mono | 0.093 | 0.093 | unchanged |
| WAV 44.1 kHz stereo, s16 | 15.526 | 12.422 | 20.0% |
| WAV 48 kHz mono, s16 | 7.580 | 5.771 | 23.9% |
| FLAC 44.1 kHz stereo, 24 bit | 29.047 | 25.818 | 11.1% |
| MP3 44.1 kHz stereo | 34.587 | 31.710 | 8.3% |

### Feature extraction

Synthetic 30 s input and reused outputs from `bench/bench_frontend.c`, median of
three best-of-50 measurements per build, alternating baseline and updated runs.
These C API timings exclude Python input validation and output allocation.

| Stage | Before (ms) | After (ms) | Time reduction |
|---|---:|---:|---:|
| Whisper log-mel | 2.512 | 2.465 | 1.9% |
| SenseVoice fbank, no CMVN | 3.127 | 3.075 | 1.7% |
| SenseVoice fbank + CMVN | 3.148 | 3.089 | 1.9% |
| 400-point FFT + power, 3000 frames | 1.244 | 1.210 | 2.7% |
| 512-point FFT + power, 3000 frames | 1.384 | 1.344 | 2.9% |

### Batch threads and output reuse

The batch default remains `n_threads=None`, which selects all eight available cores
on this M1. Qwen3-ASR already supports `out=` for reusing a writable output array.
The comparison runner now measures explicit thread counts, automatic selection,
and Qwen batches with and without output reuse. Run only these comparisons with:

```bash
uv run --no-sync python bench/current_environment.py --batch-only --json batch-results.json
```

Each of the runner's three passes warms the configurations, then measures 20 calls
per configuration in a shuffled order. It reports the best time plus median and
p95 latency; JSON retains the best-time sections and adds `*_latency` sections.
Interleaving avoids attributing scheduling changes between sequential runs to the
thread-count argument: automatic selection and explicit `n_threads=8` use the same
worker count here. Timings include input validation and output allocation where
needed. Each batch contains 32 references to one 30 s waveform, rather than 32
independently allocated recordings. These are warm measurements with a small
sample count; measure again for a deployment's actual workload.

Median batch times below are the median of the three per-pass medians, in ms:

| Batch of 32 × 30 s | 4 threads | 8 threads | Auto (8 here) |
|---|---:|---:|---:|
| Whisper | 22.334 | 16.815 | 16.930 |
| SenseVoice, no CMVN | 27.528 | 19.940 | 20.100 |
| Qwen3-ASR, new output | 27.714 | 21.371 | 21.288 |
| Qwen3-ASR, reused output | 26.930 | 20.632 | 20.591 |

Automatic selection and explicit eight-thread batches agree within about 1% in
this run. Qwen output reuse saves about 3.5% at eight threads, so extending the
output-buffer API is a lower priority than the native kernel changes on this M1.

## Apple M1 optimization results (2026-10-12)

The three changes were evaluated in order: a NEON 4×4 transpose for Whisper's
frame gather, fixed 400/512-point FFT stage schedules on aarch64, and fused
SenseVoice LFR copy + CMVN. Arithmetic order is preserved; FMA contraction remains
disabled. The scalar and x86 FFT/gather paths retain their previous implementation.
A NEON transpose for SenseVoice's scaled gather was also tried, but increased
latency and was discarded.

### Python API: before and after

Same environment, speech input and three-pass methodology as the original M1 run
below. The baseline is commit `e4c545e`; after measurements use the working changes.
All values are medians of three best-time measurements, in milliseconds.

| 30 s function | Before | After | After range | Time reduction |
|---|---:|---:|---:|---:|
| `whisper.log_mel` | 2.866 | 2.525 | 2.525–2.526 | 11.9% |
| `sensevoice.fbank + CMVN` | 3.462 | 3.312 | 3.312–3.313 | 4.3% |
| `sensevoice.fbank_raw` | 3.372 | 3.260 | 3.260–3.265 | 3.3% |

| Audio length | Whisper after (ms) | SenseVoice + CMVN after (ms) |
|---|---:|---:|
| 1 s | 0.091 | 0.122 |
| 5 s | 0.427 | 0.563 |
| 10 s | 0.844 | 1.110 |
| 30 s | 2.523 | 3.312 |
| 120 s | 10.175 | 13.265 |

| Batch threads (32 × 30 s) | Whisper after (ms) | SenseVoice after, no CMVN (ms) |
|---|---:|---:|
| 1 | 87.973 | 106.364 |
| 2 | 43.915 | 54.052 |
| 4 | 24.706 | 28.081 |
| 8 | 22.121 | 28.559 |

Reproduce after rebuilding the extension:

```bash
uv sync --offline
.venv/bin/python bench/current_environment.py --json results.json
```

### Isolated C measurements by stage

Synthetic 30 s waveform from `bench/bench_frontend.c`, output buffers reused,
best of 50 calls per measurement, median of three interleaved passes. These timings
are distinct from the Python/JFK timings above. The CMVN comparison uses the same
final kernels with only the LFR/CMVN implementation changed.

| Stage | Whisper (ms) | SenseVoice without CMVN (ms) | SenseVoice with CMVN (ms) |
|---|---:|---:|---:|
| Baseline | 2.861 | 3.399 | — |
| Whisper NEON gather | 2.605 | 3.396 | — |
| Fixed FFT schedules | 2.521 | 3.285 | 3.342 |
| Fused LFR + CMVN | 2.522 | 3.285 | 3.303 |

FFT + power alone decreased from 1.374 to 1.233 ms for 400 points, and from
1.584 to 1.382 ms for 512 points. CMVN fusion reduces about 0.039 ms (1.2%)
relative to the FFT-only version with CMVN. The negligible Whisper difference
between the last two rows is measurement variation.

The final C benchmark also reports CMVN timing:

```bash
.venv/bin/cmake -S . -B /private/tmp/asrfront-perf-review -DASRFRONT_BUILD_BENCH=ON -DASRFRONT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build /private/tmp/asrfront-perf-review -j 4
/private/tmp/asrfront-perf-review/bench_frontend
.venv/bin/ctest --test-dir /private/tmp/asrfront-perf-review --output-on-failure
```

Validation: C tests **3/3 passed**; Python Whisper, SenseVoice and kernel tests
**181 passed, 29 skipped**. Kernel tests compare default NEON results with scalar
results exactly. Twenty new CMVN cases check exact float32 results across short
inputs, boundary padding, 80/128 mel bins and different LFR configurations.
Optional reference-library cases remain skipped. Performance was measured on this
M1 only; improvements on other aarch64 CPUs and the CMVN change on x86 are unmeasured.

## Original Apple M1 baseline (2026-10-12)

The measurements in this section precede optimization. Reproducing this baseline
requires the source at commit `e4c545e`; running the commands with current sources
measures the optimized implementation instead.

Measured locally on a MacBook Air (MacBookAir10,1), Apple M1 with 4 performance
and 4 efficiency cores, 8 GB RAM, macOS 26.5.2 arm64. The installed extension
uses the `generic` kernel (arm64/NEON), CMake Release, Apple Clang 21.0.0.
Python 3.12.14, asrfront 0.1.0, NumPy 2.5.3.

Torch, torchaudio, librosa and kaldi-native-fbank were not installed, so this run
measures asrfront only; no reference speed-up is claimed for this machine.
The Intel/WSL measurements below are historical results from a different environment.

### Method and reproduction

```bash
.venv/bin/python bench/current_environment.py --json results.json
```

The runner reuses `bench/compare.py`: one warm-up, then the best wall-clock time
of at least 20 calls (5 for batches), targeting at least 0.5 s, with a maximum
of 1000 calls. Thus very fast cases can hit the call cap before 0.5 s.
Each table reports the median of three independent passes and their min–max range;
these are medians of best times, not typical per-call latency or percentiles.

Input is the repository's JFK speech WAV, tiled and sliced to exactly the requested
length, contiguous float32 mono at 16 kHz. Timings include the Python API, validation,
conversion and output allocation, after frontend caches are warmed. Audio decoding,
resampling, initial frontend construction and model inference are excluded.
Whisper uses 80 mel bins; the length sweep disables 30 s padding/trimming.
Single calls use one thread. SenseVoice uses 80 bins, LFR m=7/n=6, dither=0,
and synthetic CMVN shift=-10/scale=0.3 (560 dimensions), except `fbank_raw`
and batch calls, which omit CMVN in the existing comparison runner.
Batch inputs are 32 references to the same 30 s waveform, rather than 32 separately
allocated recordings; batch output allocation and thread scheduling are included.

### Single input: 30 s

| Function | Output shape | Median (ms) | Range (ms) |
|---|---|---:|---:|
| `whisper.log_mel` | `(80, 3000)` | 2.866 | 2.865–2.866 |
| `sensevoice.fbank + CMVN` | `(500, 560)` | 3.462 | 3.460–3.463 |
| `sensevoice.fbank_raw` | `(2998, 80)` | 3.372 | 3.372–3.375 |

### Input length: single thread

| Audio | Whisper median (ms) | Range (ms) | SenseVoice + CMVN median (ms) | Range (ms) |
|---|---:|---:|---:|---:|
| 1 s | 0.103 | 0.103–0.103 | 0.128 | 0.128–0.128 |
| 5 s | 0.485 | 0.485–0.485 | 0.588 | 0.588–0.588 |
| 10 s | 0.957 | 0.957–0.957 | 1.159 | 1.159–1.159 |
| 30 s | 2.865 | 2.864–2.865 | 3.460 | 3.460–3.460 |
| 120 s | 11.544 | 11.528–11.547 | 13.891 | 13.881–13.894 |

### Batch: 32 × 30 s

Times are for the entire batch (960 s of audio). Scaling is relative to the
one-thread median for the same function. SenseVoice batches omit CMVN.

| Threads | Whisper median (ms) | Range (ms) | Scaling | SenseVoice median (ms) | Range (ms) | Scaling |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 98.527 | 96.592–98.617 | 1.00x | 110.398 | 110.390–110.737 | 1.00x |
| 2 | 49.289 | 49.049–49.366 | 2.00x | 55.932 | 55.902–56.698 | 1.97x |
| 4 | 28.284 | 25.216–28.593 | 3.48x | 29.034 | 28.952–56.292 | 3.80x |
| 8 | 26.008 | 25.574–26.200 | 3.79x | 34.650 | 34.021–38.466 | 3.19x |

Single-input cost is approximately linear with audio length. In this run, the
lowest batch median is at 8 threads for Whisper and 4 for SenseVoice. The
SenseVoice 4-thread range is wide; these three best-time measurements do not
establish a consistently optimal thread count. Core assignment, background load
and thermal conditions were not controlled or profiled.

Validation: `pytest tests/python/test_whisper.py tests/python/test_sensevoice.py -q`
completed with **144 passed, 13 skipped**. Skipped cases are not accuracy evidence
for the unavailable reference libraries on this machine.

## Historical environment: Intel Core i5-14400F / WSL2

The following results were already recorded in this document; they were not rerun
on the current Mac. All environment-specific claims below refer to that Intel system.

Speed of `asrfront` against the reference implementations of each model's frontend.
Accuracy is covered in [whisper.md](whisper.md#accuracy) and
[sensevoice.md](sensevoice.md#accuracy); every implementation compared here produces the
same features up to float32 rounding.

## Summary

30 s of 16 kHz speech, single thread, median of 3 runs:

| Task | asrfront | Reference | Speed-up |
|---|---|---|---|
| Whisper log-mel `(80, 3000)` | **0.90 ms** | openai-whisper `log_mel_spectrogram` (torch) 8.28 ms | **9.2x** |
| | | librosa STFT + whisper mel filters 6.79 ms | 7.5x |
| SenseVoice features `(500, 560)` | **1.37 ms** | FunASR `WavFrontend` (torchaudio + LFR + CMVN) 11.95 ms | **8.7x** |
| kaldi fbank only `(2998, 80)` | **1.31 ms** | `torchaudio.compliance.kaldi.fbank` 10.89 ms | 8.3x |
| | | kaldi-native-fbank (Python API) 23.25 ms | 17.7x |

Even when torch uses all 16 threads and asrfront only one, asrfront is ~4.5x faster.
At 0.9 ms per 30 s, a single core computes Whisper features ~33 000x faster than real time.

## Environment

| | |
|---|---|
| CPU | Intel Core i5-14400F (6 P-cores with HT + 4 E-cores, 16 threads; 20 MB L3) |
| OS | Ubuntu 24.04 on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2) |
| Compiler | GCC 13.3.0, `-O3` (CMake Release), no `-march=native` |
| asrfront | 0.1.0, AVX2 kernels (runtime-selected) |
| Python | 3.12.15 |
| Reference libraries | torch 2.14.1+cpu, torchaudio 2.11.0+cpu, librosa 1.0.0, numpy 2.5.3, kaldi-native-fbank 1.22.3 |

## Method

```bash
uv run --group ref --group bench python bench/compare.py --json results.json
```

- Each measurement is the **best** of ≥ 20 calls (and ≥ 0.5 s of calls) after one warm-up
  call; the tables report the **median of 3 such runs**, with the range of the 3 runs where
  useful. Run-to-run variation is about ±5–10 %.
- Input: the JFK speech sample from the test data, tiled to the required length, float32.
- All timings are wall-clock from Python and include the Python call, input conversion and
  allocation of the output array — i.e. what a user of each library pays per call.
- "Single thread": `torch.set_num_threads(1)`; numpy/librosa run on one core for these
  operations. asrfront's single-input functions are always single-threaded.
- Reference pipelines (in `bench/compare.py`, built on `tests/python/reference.py`):
  - **openai-whisper**: a verbatim port of `whisper.audio.log_mel_spectrogram` with
    `pad_or_trim` (`torch.stft` + mel matmul + log10/clamp/scale), using whisper's
    `mel_filters.npz`.
  - **librosa**: `librosa.stft(n_fft=400, hop_length=160, window="hann")`, power, the same
    mel filters, and whisper's log/clamp/scale in numpy.
  - **FunASR WavFrontend**: `kaldi.fbank` on the ×32768 waveform (hamming, dither 0,
    energy_floor 0), FunASR's `apply_lfr` loop and `apply_cmvn` — the per-utterance body of
    `WavFrontend.forward`.
  - **kaldi-native-fbank**: `OnlineFbank` with the same options through its Python API,
    including the list conversion it needs and the per-frame `get_frame` calls.

## 1. 30 s, single thread

| | median | range (3 runs) | vs. asrfront |
|---|---|---|---|
| **Whisper log-mel** | | | |
| asrfront (AVX2) | 0.90 ms | 0.87 – 0.94 | 1.0x |
| openai-whisper (torch) | 8.28 ms | 8.07 – 8.40 | 9.2x |
| librosa + numpy | 6.79 ms | 6.49 – 6.80 | 7.5x |
| **SenseVoice features (fbank + LFR + CMVN)** | | | |
| asrfront (AVX2) | 1.37 ms | 1.31 – 1.40 | 1.0x |
| FunASR WavFrontend | 11.95 ms | 11.79 – 12.04 | 8.7x |
| **kaldi fbank only** | | | |
| asrfront `fbank_raw` | 1.31 ms | 1.27 – 1.35 | 1.0x |
| torchaudio `kaldi.fbank` | 10.89 ms | 10.62 – 11.53 | 8.3x |
| kaldi-native-fbank (Python API) | 23.25 ms | 22.76 – 23.87 | 17.7x |

LFR + CMVN add only ~0.06 ms on top of the raw fbank in asrfront; in FunASR the Python LFR
loop costs ~1 ms. The kaldi-native-fbank number is dominated by its Python interface
(`tolist()` of 480 000 samples and 2998 `get_frame` calls), not by its C++ core.

## 2. 30 s, reference libraries with 16 threads

Same inputs, `torch.set_num_threads(16)`; asrfront still single-threaded.

| | median | vs. asrfront |
|---|---|---|
| asrfront `whisper.log_mel` (1 thread) | 0.90 ms | |
| openai-whisper (torch, 16 threads) | 4.05 ms | 4.5x |
| asrfront `sensevoice.fbank` (1 thread) | 1.32 ms | |
| FunASR WavFrontend (torch, 16 threads) | 5.87 ms | 4.4x |

Torch gains ~2x from 16 threads on a single 30 s input; asrfront does not parallelize a
single input (not worth it at < 1.5 ms) and parallelizes batches instead (section 4).

## 3. Input length

Single thread; Whisper with `pad_or_trim=False` so that the work scales with the input.

| Length | asrfront whisper | openai-whisper (torch) | speed-up | asrfront sensevoice | FunASR WavFrontend | speed-up |
|---|---|---|---|---|---|---|
| 1 s | 0.04 ms | 0.37 ms | ~9x | 0.05 ms | 0.35 ms | ~7x |
| 5 s | 0.15 ms | 0.90 ms | 6.0x | 0.23 ms | 1.28 ms | 5.6x |
| 10 s | 0.30 ms | 1.58 ms | 5.3x | 0.45 ms | 2.49 ms | 5.5x |
| 30 s | 0.90 ms | 7.86 ms | 8.7x | 1.36 ms | 11.82 ms | 8.7x |
| 120 s | 3.76 ms | 28.81 ms | 7.7x | 5.54 ms | 51.55 ms | 9.3x |

- asrfront scales linearly: ~0.03 ms (Whisper) / ~0.045 ms (SenseVoice) per second of audio,
  with only a few µs of fixed per-call overhead.
- The references have a fixed per-call overhead (~0.3 ms) that dominates short inputs, and
  their cost grows faster than linearly between 10 s and 30 s (cache effects in the batched
  STFT / matmul), which is why the speed-up is lowest around 5–10 s.

With `pad_or_trim=True` (the default) the output always has 3000 frames, but blocks of
frames that read only the zero padding are filled with their known output instead of being
computed (`docs/whisper.md`). Measured 2026-10-14 on the same machine, C API, best of 200:

| Length | before | after |
|---|---|---|
| 1 s | 1.38 ms | 0.13 ms |
| 5 s | 1.31 ms | 0.23 ms |
| 10 s | 1.23 ms | 0.36 ms |
| 30 s | 0.89 ms | 0.89 ms |

Before this, short inputs were slower than 30 s ones: every padding frame was built by the
scalar edge path and then went through FFT, mel and log.

## 4. Batches and threads

`whisper.log_mel_batch` / `sensevoice.fbank_batch` on 32 × 30 s (= 16 minutes of audio),
time for the whole batch:

| `n_threads` | whisper | scaling | sensevoice | scaling |
|---|---|---|---|---|
| 1 | 29.39 ms | 1.0x | 46.47 ms | 1.0x |
| 2 | 15.90 ms | 1.8x | 26.54 ms | 1.8x |
| 4 | 9.31 ms | 3.2x | 15.05 ms | 3.1x |
| 8 | 6.76 ms | 4.3x | 11.19 ms | 4.2x |
| 16 | 7.59 ms | 3.9x | 11.80 ms | 3.9x |

- Best throughput: 32 Whisper inputs in 6.8 ms with 8 threads, i.e. ~0.21 ms per 30 s
  input (~140 000x real time).
- Scaling flattens after 4–8 threads and 16 threads is slightly slower than 8. Each item
  takes < 1 ms, so the batch is short enough that thread-pool overhead, the 4 E-cores and
  hyper-threads (16 threads ≠ 16 full cores), and memory traffic (≈ 61 MB in + 31 MB out per
  whisper batch) all show; this was not profiled further. `n_threads` around the number of
  performance cores is a good default for this CPU.

## 5. Kernel variants

C-level timing (`bench/bench_frontend.c`, no Python), 30 s, median of 3 runs. The variant is
chosen at runtime; `ASRFRONT_KERNELS=generic|scalar` forces the slower ones. All three give
bit-identical output.

| Variant | Used when | whisper | sensevoice |
|---|---|---|---|
| `avx2` | x86-64 with AVX2 (default here) | 0.99 ms | 1.46 ms |
| `generic` | x86-64 without AVX2 (SSE2), aarch64 (NEON) | 1.68 ms | 2.04 ms |
| `scalar` | compilers without GCC/Clang vector extensions (MSVC) | 4.01 ms | 5.25 ms |

Even the plain-C `scalar` build is ~2x faster than the torch reference for Whisper and
~2.3x for SenseVoice. For reference, the original one-frame-at-a-time scalar FFT
(`bench/bench_fft.c`) needs ~0.98 µs per 400-point frame, i.e. 2.9 ms per 30 s for the FFT
alone, versus ~0.37 ms in the AVX2 kernels.

## 6. Audio loading

`asrfront.load` (section [5] of `bench/compare.py`, needs `--group load`): 30 s of speech
(jfk.wav looped) re-encoded by ffmpeg, decoded to 16 kHz mono float32, single thread, best of
N runs. "whisper.load_audio" is its ffmpeg command line run as a subprocess (what openai-whisper
does, including process start-up); `librosa.load(sr=16000)` uses soundfile + soxr_hq.

| File | asrfront | whisper.load_audio | librosa.load |
|---|---|---|---|
| WAV 16 kHz mono | **0.32 ms** | 10.4 ms (33x) | 2.5 ms (8.0x) |
| WAV 44.1 kHz stereo | **14.0 ms** | 35.7 ms (2.5x) | 31.4 ms (2.2x) |
| WAV 48 kHz mono | **3.9 ms** | 19.1 ms (4.9x) | 8.4 ms (2.2x) |
| WAV 8 kHz mono | 2.3 ms | 8.6 ms (3.8x) | **1.8 ms** (0.8x) |
| FLAC 44.1 kHz stereo | 15.9 ms | 15.1 ms (1.0x) | 35.4 ms (2.2x) |
| MP3 44.1 kHz stereo | **24.8 ms** | 38.3 ms (1.5x) | 33.5 ms (1.4x) |

- Resampling dominates. The float path must reproduce ffmpeg's arithmetic exactly (8 FMA
  accumulators per output, docs/load.md), so it cannot use a cheaper filter: per output it is a
  92-tap (44.1 kHz) / 100-tap (48 kHz) dot product, ~7 ns per output sample on the AVX2+FMA3
  kernel (4 outputs interleaved; the bank of 160 phases × 96 floats competes with the input
  stream for L1). soxr upsampling from 8 kHz is faster.
- The `generic` kernel calls libm `fmaf` per tap on x86-64 without FMA3 and is ~7x slower
  for resampled input (e.g. 106 ms for 44.1 kHz stereo); on aarch64 `fmaf` is a single
  instruction.
- Large output buffers are dominated by page faults on WSL2; the loader reuses one channel
  buffer for its output to keep the number of fresh allocations low.

## Where the time goes

Per-stage breakdown of the AVX2 kernels for 3000 frames (`bench/bench_frontend.c`):

| Stage | Whisper (400-pt FFT) | SenseVoice (512-pt FFT) |
|---|---|---|
| real FFT + power | ~0.37 ms | ~0.43 ms |
| mel filters | ~0.08 ms | ~0.10 ms |
| log | ~0.12 ms | ~0.12 ms |
| framing / preprocessing / output (remainder) | ~0.3–0.4 ms | ~0.6–0.8 ms |

The FFT is about 40 % (Whisper) / 30 % (SenseVoice) of the total; the remaining time is spread over framing (gathering and
transposing frames, Kaldi's DC removal / pre-emphasis), writing the output layout, and the
whole-spectrogram normalization pass for Whisper.

## History

| Milestone | whisper 30 s | sensevoice 30 s |
|---|---|---|
| M4/M5: scalar, one frame at a time, libm log | 5.8 ms | 5.2 ms |
| M7: 8-frame SIMD kernels, AVX2 dispatch, vectorized log | 0.90 ms | 1.37 ms |

| Milestone | load: WAV 44.1 kHz stereo, 33 s |
|---|---|
| M9.3: scalar resampler, libm `fmaf` per tap | 86 ms |
| M9.5: AVX2+FMA3 kernel, 4 outputs at a time, fewer allocations | 14 ms |

## Caveats

- WSL2 on a desktop CPU with turbo boost; absolute numbers vary by ±5–10 % between runs and
  will differ on other machines. Ratios are more stable than absolute times.
- This historical run measured x86-64 with AVX2 only. For current aarch64 `generic`
  (NEON) results, see the Apple M1 section above.
- The reference pipelines are re-implementations of the libraries' code paths (verified to
  produce the same features, see `tests/python/test_reference.py`), not the full
  `openai-whisper` / `funasr` packages, which add file decoding and model-specific glue.
