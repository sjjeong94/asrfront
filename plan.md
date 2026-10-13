# asrfront Implementation Plan

## 0. Goals and Scope

Implement input feature extraction for ASR models (Whisper, SenseVoiceSmall) in **dependency-free native C**, and expose it through a **nanobind**-based Python binding.

```python
import asrfront

mel = asrfront.whisper.log_mel(audio)  # (80, 3000) float32
feat = asrfront.sensevoice.fbank(audio)  # (T', 560)  float32
```

**Success criteria**
1. **Accuracy** — difference from the reference implementations:
   - Whisper: max abs err < 1e-4 vs. `openai-whisper`'s `log_mel_spectrogram(pad_or_trim(audio))`
   - SenseVoice: max abs err < 1e-3 (log domain) vs. FunASR `WavFrontend` (= `torchaudio.compliance.kaldi.fbank` + LFR + CMVN)
2. **Speed**: for 30 s of audio, single-threaded, at least 5x faster than the torch (CPU) implementation and at least 10x faster than librosa
3. **Dependencies**: the C core links against nothing beyond libc/libm (FFT implemented in-house)

**Out of scope (v1)**: resampling (input is assumed to be 16 kHz mono), audio file decoding, GPU, streaming (to be considered later).
Audio loading (decode + downmix + resample, matching ffmpeg) is planned as **M9**, see §7.

---

## 1. Reference Algorithm Specification

Pin down the reference behavior exactly before implementing. A single deviation here will make the numbers diverge.

### 1.1 Whisper `log_mel`

| Item | Value |
|---|---|
| sample rate | 16000 |
| input length | `pad_or_trim` → 480000 samples (30 s); zero-padded if shorter, truncated if longer |
| n_fft / win_length | 400 |
| hop_length | 160 |
| window | Hann, **periodic** (`torch.hann_window(400)`): `0.5 - 0.5*cos(2πn/400)` |
| center | True, **reflect** padding of 200 samples on each side |
| frame count | 3001 → drop last frame → **3000** |
| spectrum | `|STFT|²` (power), bins 0..200 (201 bins) |
| mel filter | librosa `filters.mel(sr=16000, n_fft=400, n_mels=80/128)` (slaney mel scale, slaney norm, fmin=0, fmax=8000) |
| log | `log10(max(mel, 1e-10))` |
| dynamic range | `x = max(x, global_max(x) - 8.0)` ← **max over the whole spectrogram** |
| normalization | `(x + 4.0) / 4.0` |
| output | `(n_mels, 3000)` float32, row-major |

- Also support `n_mels=128` (large-v3).
- Option: with `pad_or_trim=False`, accept arbitrary-length input → `(n_mels, N/160)` output (matching whisper's `padding` argument behavior).
- Generate the mel filters directly in C in float64, cast to float32, and compare bit-for-bit against whisper's shipped `mel_filters.npz`. (Fallback if they differ: embed the npz values as a C array.)

### 1.2 SenseVoiceSmall `fbank` (FunASR `WavFrontend`)

| Item | Value |
|---|---|
| input scale | float waveform × 32768 (FunASR does `waveform * (1 << 15)`) |
| frame_length / shift | 25 ms (400) / 10 ms (160) |
| snip_edges | True → `T = 1 + (N - 400) // 160` |
| dither | 0.0 (inference default; exposed as an option, with an in-house seedable RNG for reproducibility) |
| remove_dc_offset | True (subtract per-frame mean) |
| preemphasis | 0.97 (`x[i] -= 0.97*x[i-1]`, `x[0] -= 0.97*x[0]`) |
| window | Hamming (kaldi, **symmetric**): `0.54 - 0.46*cos(2πn/(N-1))` |
| FFT size | round_to_power_of_two → **512** (400 + zero padding) |
| spectrum | power, bins 0..256 (257 bins) |
| mel | Kaldi mel (`1127*ln(1+f/700)`), 80 bins, low_freq=20, high_freq=8000, triangular filters, **no normalization** |
| log | `log(max(mel, FLT_EPSILON))` (natural log) |
| use_energy | False |
| LFR | lfr_m=7, lfr_n=6. Prepend `(m-1)//2 = 3` copies of the first frame, `T' = ceil(T / 6)`, each output concatenates 7 frames → 560 dims. Pad the tail with the last frame if short |
| CMVN | `(x + shift) * scale` using the `<AddShift>` / `<Rescale>` vectors (560 dims) from `am.mvn` (Kaldi nnet text format) |
| output | `(T', 560)` float32 |

- CMVN vectors are either parsed from `am.mvn` on the Python side and passed in (`asrfront.sensevoice.load_cmvn(path)`), or CMVN can be skipped via an option.
- Expose per-stage outputs (raw fbank `(T, 80)`, before/after LFR) as separate functions for debugging and testing.

---

## 2. Architecture

```
asrfront/
├── CMakeLists.txt
├── pyproject.toml              # scikit-build-core + nanobind
├── README.md / plan.md
├── csrc/                       # pure C11, no external dependencies
│   ├── include/asrfront.h      # public C API
│   ├── fft.c / fft.h           # real FFT (n=400 mixed-radix, n=512 radix-2/4)
│   ├── window.c                # hann (periodic), hamming (symmetric)
│   ├── melbank.c               # slaney / kaldi mel filter generation (sparse form)
│   ├── whisper.c               # log_mel pipeline
│   ├── kaldi_fbank.c           # kaldi-compatible fbank
│   ├── lfr_cmvn.c              # LFR + CMVN
│   ├── threadpool.c            # (optional) pthread-based frame parallelism
│   └── simd.h                  # SSE/AVX2/NEON helpers + scalar fallback
├── src/asrfront/
│   ├── __init__.py             # exposes whisper, sensevoice submodules
│   ├── _ext.cpp                # nanobind binding (the only C++ file)
│   ├── whisper.py / sensevoice.py   # thin Python wrappers, cmvn parser
│   └── py.typed, *.pyi
├── tests/
│   ├── c/                      # C unit tests (FFT accuracy etc., plain asserts, no framework)
│   └── python/                 # pytest: accuracy vs. references
└── bench/                      # speed benchmark scripts
```

### 2.1 C API (`asrfront.h`)

Create an **opaque handle** holding state (plans/filterbanks/windows/scratch buffers) once and reuse it. No malloc on the hot path.

```c
typedef struct af_whisper af_whisper;
typedef struct af_whisper_config {
    int n_mels;          /* 80 | 128 */
    int pad_or_trim;     /* 1: fixed 30 s */
    int n_threads;       /* 0: auto, 1: single */
} af_whisper_config;

int  af_whisper_create(const af_whisper_config *cfg, af_whisper **out);
/* query output size, caller allocates the buffer */
size_t af_whisper_out_frames(const af_whisper *h, size_t n_samples);
int  af_whisper_log_mel(af_whisper *h, const float *audio, size_t n_samples, float *out);
void af_whisper_destroy(af_whisper *h);

typedef struct af_sensevoice af_sensevoice;
typedef struct af_sensevoice_config {
    int   n_mels;              /* 80 */
    int   lfr_m, lfr_n;        /* 7, 6 */
    float dither;              /* 0 */
    unsigned long long seed;
    const float *cmvn_shift;   /* NULL skips CMVN; length n_mels*lfr_m */
    const float *cmvn_scale;
    int   n_threads;
} af_sensevoice_config;

int  af_sensevoice_create(const af_sensevoice_config *cfg, af_sensevoice **out);
size_t af_sensevoice_raw_frames(const af_sensevoice *h, size_t n_samples); /* T  */
size_t af_sensevoice_out_frames(const af_sensevoice *h, size_t n_samples); /* T' */
size_t af_sensevoice_feat_dim(const af_sensevoice *h);                     /* 560 */
int  af_sensevoice_fbank(af_sensevoice *h, const float *audio, size_t n_samples, float *out);
int  af_sensevoice_fbank_raw(af_sensevoice *h, const float *audio, size_t n_samples, float *out); /* (T,80) */
void af_sensevoice_destroy(af_sensevoice *h);
```

- Return value: 0 on success / negative error code (`AF_EINVAL`, `AF_ENOMEM`, `AF_ENOTIMPL`). `*_create` returns the handle through an out-parameter so EINVAL and ENOMEM can be told apart.
- Handles are not thread-safe (they own scratch buffers). For multithreaded use, one handle per thread — handled on the Python side with a thread-local cache.

### 2.2 Python Binding (nanobind)

- Input: `nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>`. float64/int16/non-contiguous arrays are converted in the Python wrapper via `np.ascontiguousarray(..., dtype=np.float32)` (int16 divided by 32768).
- Output: numpy array that takes ownership of the C-allocated buffer via `nb::capsule` (no copy).
- Release the GIL during computation with `nb::gil_scoped_release` → enables parallel processing from Python threads.
- Default-config handles are lazily created and cached at module level (`functools.lru_cache` keyed on n_mels etc.).
- Batch API: `log_mel_batch(list_of_audio, n_threads=...)` → `(B, 80, 3000)`. Batch-level parallelism is more efficient than frame-level parallelism, so include it in v1.
- `torch.Tensor` input is accepted automatically by nanobind via DLPack (CPU only).

---

## 3. Performance Design

30 s of audio = 3000 frames × (FFT + mel product). Bottlenecks, in order:

1. **FFT**
   - Whisper n=400 = 2⁴·5² → implement a **mixed-radix (2, 4, 5) real FFT** in-house. Use the real-input trick (N/2 complex FFT + post-processing) for a 2x saving.
   - Kaldi n=512 → split-radix / radix-4 real FFT.
   - Twiddle tables precomputed at handle creation. Internal arithmetic in float32; twiddles generated in double and stored as float.
   - Alternative to evaluate: run FFTs on multiple frames simultaneously across SIMD lanes (structure-of-arrays, 8 frames per AVX2 vector) — lane utilization is high given the many frames. Decide by benchmark after the first implementation.
2. **Mel filter product**
   - Mel filters are very sparse (each spans a few to a few dozen bins). Store as sparse **(start, len, weights)** → ~10x fewer ops than a dense GEMM (80×201).
   - Or process in frame blocks (e.g. 16 frames) for cache friendliness.
3. **log**
   - `log10f`/`logf` called 3000×80 = 240k times → not negligible, so add an optional SIMD polynomial log approximation (~1 ulp accuracy). Default to libm for accuracy; decide after benchmarking.
4. **Memory**
   - Do not materialize the full STFT matrix; **stream per frame**: frame → FFT → power → mel → log, written straight to the output buffer. Whisper's global max is accumulated as a running max during mel computation, then clamp/normalize in one final pass.
   - Output layout: Whisper is `(n_mels, T)`, so compute in frame blocks and write transposed (cache-aware).
5. **SIMD**
   - Rely on compiler auto-vectorization first (`-O3`, but **no `-ffast-math`** — it breaks numerical reproducibility; use `restrict` and aligned buffers instead).
   - Hot loops get x86 SSE2/AVX2/AVX-512 and ARM NEON intrinsic paths plus a scalar fallback. Runtime dispatch (`__builtin_cpu_supports`) lets a single wheel support many CPUs.
6. **Threading**
   - Split frame ranges into n chunks processed by pthread workers (Win32 threads on Windows). For a single 30 s input, measure gain vs. overhead before choosing the default (expected: 1 thread by default, threads used in the batch API).

---

## 4. Implementation Milestones

### M1. Project skeleton (0.5 day) — done
- [x] `pyproject.toml` (scikit-build-core, nanobind), `CMakeLists.txt` (C11 static lib `asrfront_core` + nanobind module `_ext`)
- [x] Verify `import asrfront` works with stub functions; uv-based dev loop (`uv sync`, `uv run pytest`; native rebuild via `tool.uv.cache-keys`)
- [x] ruff/clang-format configuration

### M2. Pin references & test harness (1 day) — done
- [x] `tests/python/reference.py`: float64 numpy ports of whisper `log_mel_spectrogram` (+ librosa slaney filters), `torchaudio.compliance.kaldi.fbank`, FunASR `apply_lfr` / `apply_cmvn` / `load_cmvn`
- [x] Test audio (`tests/python/audio_signals.py`): silence, sine, multi-tone, chirp, white noise, near-silent, impulses, clipped, DC offset, real speech (`tests/data/jfk.wav`), 35 s speech, short inputs (201/399/400/1000 samples), odd length
- [x] Split the reference into stages too, so intermediate values (window, mel filter, power spectrum) can be compared
- [x] `test_reference.py` validates the reference against the originals (`uv sync --group ref` installs CPU torch/torchaudio; skipped otherwise):
  - slaney filters == whisper `mel_filters.npz` (80/128) to float32 rounding
  - whisper: ≤ 1e-4 vs. torch port of `log_mel_spectrogram` on all signals, 80/128 mels, pad_or_trim on/off
  - kaldi: ≤ 1e-6 vs. torchaudio when given torchaudio's filter bank and float64 input (exact port); ≤ 2e-3 end to end vs. FunASR's float32 call
  - C frame-count helpers == reference output shapes
- [x] `test_whisper.py` / `test_sensevoice.py`: C vs. reference, marked `xfail(strict, NotImplementedError)` until M4/M5

**Findings**
- torchaudio builds the kaldi mel banks in float32 (errors up to ~1.4e-5), and with its float32 FFT this gives up to ~1.3e-3 log-domain error on low-energy bins (worst case: chirp). Accuracy targets for asrfront are therefore measured **against the float64 reference** (Whisper 1e-4, SenseVoice 1e-3); agreement with torchaudio is only expected to ~2e-3.
- Whisper with `pad_or_trim=False` and ≤ 200 samples is an error in torch (reflect pad); asrfront's behavior for this case is decided in M4.

### M3. FFT (2 days) — done
- [x] Scalar mixed-radix complex FFT → real FFT wrapper (`csrc/fft.{h,c}`, internal API `af_rfft_*`)
  - iterative Stockham autosort (natural-order output, no bit reversal), radix-4/2/3/5 butterflies, any even n with n/2 = 2^a·3^b·5^c
  - real FFT = n/2 complex FFT of packed even/odd samples + split post-processing
  - one allocation per plan (twiddles + ping-pong buffers), no malloc on the hot path
- [x] Verify vs. naive DFT (double) (`tests/c/test_fft.c`): 27 sizes from 2 to 2048 incl. 400/512, random/impulse/DC/near-Nyquist inputs. Max error relative to max|X| ≤ 1.7e-7 (target 1e-6); power spectrum ≤ 2e-6
- [x] Benchmark (`bench/bench_fft.c`, `-DASRFRONT_BUILD_BENCH=ON`), rfft + power, i5-14400F, GCC 13 `-O3`, scalar:

  | n | asrfront | torch.fft.rfft (1 thread, batch 3000) | numpy.fft.rfft (batch 3000) |
  |---|---|---|---|
  | 400 | ~1.13 µs/frame (3.4 ms / 30 s) | ~0.97 µs | ~1.71 µs |
  | 512 | ~1.16–1.30 µs/frame | ~1.12 µs | ~2.20 µs |

  Already on par with pocketfft without SIMD; M7 candidates: twiddle-free first stage, radix-8, SIMD across frames, fused power.

### M4. Whisper log_mel (2 days) — done
- [x] Periodic Hann, reflect pad, frame loop, power, slaney mel (80/128) generation + npz comparison
  - `csrc/melbank.{h,c}`: slaney filters generated in double with librosa's float32 rounding order → **bit-identical** to whisper's `mel_filters.npz` (80 and 128); stored sparsely (non-zero range per filter)
  - reflect padding is done per frame (only edge frames take the slow path); `pad_or_trim` zero-padding/truncation is virtual — no padded copy of the input
- [x] log10 / clamp / global max / normalization: log10 written transposed into the output while tracking the running max, then one clamp + scale pass
- [x] pad_or_trim on/off; accuracy tests pass
  - C vs. float64 reference: worst 4.0e-5 (chirp), speech 1.6e-5 (target 1e-4); also ≤ 1e-4 vs. the torch port of whisper
  - `pad_or_trim=False` with ≤ 200 samples raises `ValueError` (C: `AF_EINVAL`), matching torch's reflect-pad error; empty input with `pad_or_trim=True` is valid (all-zero 30 s)
  - also tested: int16/float64/strided/list input, handle reuse, concurrent Python threads

**Speed (30 s, i5-14400F)**: asrfront 5.8 ms vs. torch 5.0 ms (1 thread) / 2.0 ms (16 threads) — currently *slower* than torch. FFT is ~3.4 ms of it; the rest is mostly the 240k `log10f` calls and per-frame overhead. Closing the gap to the 5x target is M7's job (SIMD FFT across frames, fused power+mel, vectorized log).

### M5. SenseVoice fbank (2 days) — done
- [x] Kaldi framing (snip_edges), DC removal, preemphasis, hamming, 512 FFT, kaldi mel, log (`csrc/sensevoice.c`; kaldi banks in `csrc/melbank.c`, generated in double)
- [x] Compare `af_sensevoice_fbank_raw` against the float64 reference: worst 7.7e-4 (chirp), 6.8e-4 (DC offset), speech 2.0e-4 (target 1e-3); ≤ 2e-3 vs. torchaudio's float32 kaldi.fbank. The largest errors sit in low mel bins that pre-emphasis attenuates by ~30 dB, where float32 rounding relative to the frame peak dominates
- [x] LFR, CMVN in C (any `lfr_m`/`lfr_n`; edge frames replicated as in FunASR), `load_cmvn` in Python (`asrfront.sensevoice.load_cmvn`, cached by path+mtime; `cmvn=` also accepts a path). Final features with SenseVoiceSmall's real `am.mvn`: worst 1.1e-4 vs. reference, 3.7e-5 vs. a FunASR-style torch pipeline (30 s speech)
- [x] Dither option (xorshift64* + Box–Muller, splitmix64 seeding), dither=0 by default. **The RNG is reseeded from `seed` on every call**, so identical inputs give identical outputs (unlike torch's global RNG)
- Real `am.mvn` test runs when `ASRFRONT_SENSEVOICE_MVN=/path/to/am.mvn` is set (file not vendored, license)

**Speed (30 s, i5-14400F, with CMVN)**: asrfront 5.2 ms vs. FunASR-style torch 9.2 ms (1 thread, 1.8x) / 4.8 ms (16 threads); torchaudio `kaldi.fbank` alone 7.8 ms. Target 5x → M7.

### M6. Polish Python API (1 day) — done
- [x] dtype/shape validation with helpful error messages; int16/float64 input handling
  - accepted: float16/32/64 (cast to float32), int16 PCM (÷32768), lists, CPU torch tensors (via `np.asarray`)
  - `TypeError` for other integer dtypes (ambiguous scale; message suggests the divisor), bool/complex/object
  - `ValueError` for non-1-D input (2-D hint: mix down channels or use `*_batch`), unsupported `n_mels`, bad `lfr_m`/`lfr_n`/`dither`/`seed`, short whisper input, cmvn length; batch errors name the offending item (`audios[3]`)
- [x] GIL release, zero-copy output, `.pyi` type stubs (overloads for `out=`), docstrings; `mypy --strict src/asrfront` clean (mypy in dev group)
- [x] Batch API (Python thread pool over GIL-releasing native calls, per-thread handles):
  - `whisper.log_mel_batch(audios, n_mels, pad_or_trim, n_threads)` → `(B, n_mels, 3000)` (list of arrays when `pad_or_trim=False`)
  - `sensevoice.fbank_batch(audios, cmvn, ..., n_threads)` → `(feats (B, max_len, 560) zero-padded, lengths int32)` like FunASR `WavFrontend`
  - low-level `out=` argument on `WhisperFrontend.log_mel` / `SenseVoiceFrontend.fbank` / `fbank_raw` writes straight into a caller buffer (exact shape, writable, C-contiguous float32; never converted), so batch results are not copied
  - scaling, 32 × 30 s on i5-14400F (6P+4E cores): whisper 178.8 ms (1 thread) → 31.6 ms (8) → 25.8 ms (16), **6.9x / 0.81 ms per item**; sensevoice 6.1x with 16 threads
  - Python wrapper overhead ~2.6 µs per call
- Since batch parallelism is already handled from Python, M7 threading inside the C core is only needed for single long inputs.

### M7. Optimization (3 days) — done
- [x] Benchmark scripts: `bench/compare.py` (asrfront vs. whisper/torch, librosa, torchaudio kaldi, kaldi-native-fbank, FunASR-style pipeline, batch scaling; `uv run --group ref --group bench python bench/compare.py`) and `bench/bench_frontend.c` (C-level totals + per-stage breakdown)
- [x] Sparse mel, SIMD FFT/power/mel, runtime dispatch
  - **8 frames per SIMD lane group** (SoA layout `buf[i * 8 + lane]`): FFT passes, fused split/power, mel, log, kaldi preprocessing all operate on 8 frames at once (`csrc/kernels.h`, `csrc/kernels_impl.c`)
  - kernels use GCC/Clang vector extensions at the native width (8 floats with AVX, 4 with SSE2/NEON; plain floats for other compilers); 8×8 register transpose to gather frames; branch-free vectorized Cephes `logf` instead of libm (`log10f` alone was 1.2 ms / 30 s)
  - `kernels_impl.c` is compiled twice on x86-64 (baseline SSE2 and `-mavx2`), selected at runtime via `__builtin_cpu_supports` / cpuid; `ASRFRONT_KERNELS=generic` forces the baseline; `_ext.kernel_isa()` reports it
  - no FMA, no reassociation → **generic and AVX2 outputs are bit-identical** (tested for every signal and both frontends), and the SoA FFT matches the scalar FFT bit for bit (C test)
- [x] Profiled with per-stage timers (no `perf` on this machine) and iterated, rerunning accuracy tests each time. Accuracy unchanged: whisper worst 4.0e-5, kaldi raw worst 7.7e-4. Tried and dropped: twiddle-free first stage (no measurable gain)
- [x] Threading and batch parallelism: batch parallelism is done in M6 (Python thread pool). C-level threading for a single input is **not implemented**: one 30 s input takes < 1 ms, so the thread start-up and synchronization would eat most of the gain; `n_threads` stays reserved in the C config

**Results** (30 s, i5-14400F, single thread, best of N; numbers vary ±10% between runs):

| | asrfront (AVX2) | asrfront (generic SSE2) | reference | speed-up |
|---|---|---|---|---|
| Whisper log-mel 80×3000 | 0.85–0.95 ms | 1.67 ms | openai-whisper (torch) 8.6–9.6 ms | **~10x** |
| | | | librosa STFT + mel (numpy) 6.6–7.0 ms | ~7.7x |
| SenseVoice fbank + LFR + CMVN | 1.3–1.4 ms | 2.1 ms | FunASR WavFrontend (torchaudio) 12–14 ms | **~9–10x** |
| kaldi fbank only | 1.3 ms | | torchaudio kaldi.fbank 11–12 ms / kaldi-native-fbank 18.7 ms* | ~9x / ~14x |

\* kaldi-native-fbank via its Python API, including list conversion and per-frame access.

Before M7 the same inputs took 5.8 ms (whisper) and 5.2 ms (sensevoice). Success criterion 2 is met against torch (≥ 5x); against librosa it is ~7.7x, short of the 10x goal.

### M8. Packaging & CI (1 day) — done (CI not yet run on GitHub)
- [x] GitHub Actions (`.github/workflows/ci.yml`, checked with actionlint):
  - `lint`: ruff check/format, `mypy --strict`, clang-format
  - `test`: Linux x86_64 / aarch64, Windows × Python 3.9 / 3.13 — `uv sync`, pytest, pytest again with `ASRFRONT_KERNELS=scalar`, C tests via CMake/CTest
  - `reference`: full suite with torch/torchaudio (`--group ref`)
  - `sanitizers`: C tests and pytest against an ASan/UBSan build (both verified locally: all pass, extension instrumented)
- [x] Wheels with cibuildwheel (`.github/workflows/wheels.yml`, config in `[tool.cibuildwheel]`): cp39–cp311 regular wheels + one **cp312-abi3** wheel per platform (nanobind `STABLE_ABI`, `wheel.py-api = "cp312"`; `abi3audit --strict` clean), manylinux x86_64/aarch64, Windows AMD64, macOS arm64/x86_64 (removed in M8 on request, re-added after M9 together with an `ffmpeg-macos` CI job that compares `asrfront.load` with ffmpeg on Apple silicon); sdist; pushing a `v*` tag publishes to PyPI via trusted publishing after checking the tag matches the version (needs the `pypi` environment registered on PyPI); manual runs can upload to TestPyPI
  - verified locally: `uv build` (sdist → wheel), fresh venvs: abi3 wheel on Python 3.13 and sdist build on Python 3.9 both pass the test suite
- [x] README: install, usage, performance and accuracy tables; MIT `LICENSE`
- Portability work done here:
  - `long` → `ptrdiff_t` / `long long` for sample indices (32-bit `long` on Windows)
  - third kernel variant **`scalar`** (no vector extensions = the code path MSVC compiles), built on every platform and selectable with `ASRFRONT_KERNELS=scalar`; bit-identical to the others (C + Python tests). Slower (whisper 3.8 ms / 30 s) but a correct fallback
  - cross-compiled the C core with clang (zig) for aarch64-linux, arm64/x86_64-macOS and x86_64-windows-gnu without warnings. **Not verified here**: MSVC (no Windows SDK; the `cpuid` dispatch and `/arch:AVX2` build only run in CI), and actually running on macOS/ARM/Windows

**Total estimate: ~12–13 working days** (all milestones implemented; first GitHub CI run pending)

### M9. Audio loading (~7.5 days) — done
- WAV / raw PCM / FLAC / MP3 (M9.6: + Wave64, AIFF / AIFF-C, CAF, NIST SPHERE, IMA / MS / IMA4 ADPCM) → 16 kHz mono float32 for Whisper / SenseVoiceSmall, ffmpeg-compatible; single-argument `asrfront.load(path_or_bytes)`. Details in §7.

---

## 5. Test Strategy

| Level | Contents |
|---|---|
| C unit | FFT vs. naive DFT, window values, mel filter sums/shapes, edge cases (n_samples=0, < frame_length) |
| Python accuracy | max abs / mean abs error vs. references across audio types and lengths (parametrized) |
| Regression | Output hashes/snapshots for fixed inputs (detects SIMD path changes) |
| Path consistency | Scalar vs. SIMD path output comparison (dispatch forced via environment variable) |
| Memory | Run C tests + pytest under an ASan/UBSan build (one CI job) |
| Threads | Identical results when called concurrently from multiple Python threads |

---

## 6. Risks & Open Decisions

1. **Whisper accuracy vs. speed**: torch uses float32 FFT (pocketfft/MKL), so differences around 1e-5 are unavoidable. Set tolerance to 1e-4 and run one end-to-end check that actual model output (decoded text) is identical.
2. **Global max dependency**: Whisper normalization depends on the max over the whole spectrogram → results change under streaming/chunked processing. v1 supports whole-input processing only.
3. **SenseVoice CMVN source**: `am.mvn` may differ per model, so instead of bundling it, accept a path/array as input. (Whether to bundle SenseVoiceSmall defaults for convenience is decided after checking the license.)
4. **Input scale**: SenseVoice expects int16 scale (×32768). To avoid confusion when users pass floats already in int16 scale, make the `input_scale` argument explicit or document dtype-based automatic handling.
5. **No `-ffast-math`**: avoided for reproducibility. Whether FMA is used (and resulting per-path differences) is also managed through tests.
6. **Follow-up candidates**: streaming kaldi fbank (easy for SenseVoice since frames are independent), resampler (→ M9, §7), presets for additional models (Paraformer, other Whisper-style models, Qwen-Audio, etc.).

---

## 7. M9: Audio Loading (`asrfront.load`)

A loader in C whose only job is to produce the input that **Whisper** and **SenseVoiceSmall**
expect: **decode → downmix → resample to 16 kHz → s16 quantization → float32**. One function, one
argument, no options.

```python
import asrfront

audio = asrfront.load("speech.mp3")  # (N,) float32 in [-1, 1), 16 kHz mono
audio = asrfront.load(request_body)  # bytes: a WAV/FLAC/MP3 file in memory, or raw s16le PCM
mel = asrfront.whisper.log_mel(audio)
feat = asrfront.sensevoice.fbank(audio, cmvn="am.mvn")
```

**Why no options**: both models take the same input (16 kHz mono float32 in [-1, 1]), so a single
fixed pipeline serves both. librosa-style knobs (`sr`, `mono`, `offset`, `duration`, `dtype`,
`res_type`) are deliberately left out; users who need them can slice the returned array or use
librosa. Options can be added later without breaking `load(source)`.

### 7.1 Goals

| Item | Decision |
|---|---|
| formats | **WAV**, **raw PCM** (headerless), **FLAC**, **MP3**; M9.6: **Wave64**, **AIFF / AIFF-C**, **CAF**, **NIST SPHERE**, IMA / Microsoft / IMA4 ADPCM |
| reference | **ffmpeg** as invoked by openai-whisper `load_audio`: `ffmpeg -nostdin -threads 0 -i FILE -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -` then `/ 32768` |
| output (fixed) | 1-D float32, 16 kHz, mono (all channels downmixed), always rounded through int16 like `-f s16le` → exactly `int16 / 32768` |
| input | a file path, or the file's bytes (`bytes` / `bytearray` / `memoryview`, e.g. an HTTP upload or a stream chunk) |
| raw PCM (fixed) | **s16le, 16 kHz, mono** — the usual ASR convention. Raw PCM has no magic bytes, so: **path** → chosen by `.pcm` / `.raw` extension; **bytes** → chosen when no container magic is found (bytes have no extension) |
| dependencies | still none at link time; FLAC/MP3 via vendored single-header decoders (public domain / MIT-0), compiled into `asrfront_core` |
| not supported | AAC/M4A, Opus, Vorbis, video containers, big-endian RIFX (ffmpeg decodes its samples as little-endian), GSM / G.722, shorten SPHERE (→ clear `AF_EUNSUPPORTED` error suggesting ffmpeg) |

**Accuracy targets** (vs. the ffmpeg pipeline above, compared as int16 samples):

| Case | Target |
|---|---|
| WAV/PCM/FLAC, already 16 kHz mono | **bit-exact** |
| downmix only (stereo → mono, 16 kHz) | bit-exact (f64 with ≥ 5 channels: ≤ 1 LSB) |
| resample, all common rates (exact-ratio mode) | **bit-exact samples** (vs. ffmpeg on AVX2+FMA3; ≤ 1 LSB vs. other CPUs). Length: exact for 8k/11.025k/12k/24k/32k/48k/96k, last sample may differ (±1) for 22.05k/44.1k/88.2k |
| resample, non-standard rates (> 1024 phases) | ≤ 1 LSB |
| MP3 | decoder-dependent (different float pipeline from ffmpeg's `mp3float`): max abs ≤ 1e-3 after resample, **same length/alignment** (encoder delay & padding trimmed like ffmpeg) |
| end-to-end Whisper | `whisper.log_mel(load(f))` vs. `whisper.log_mel_spectrogram(whisper.load_audio(f))` ≤ 1e-3; decoded text identical on jfk |
| end-to-end SenseVoice | `sensevoice.fbank(load(f))` vs. FunASR `WavFrontend` on FunASR's own loaded audio: identical for 16 kHz WAV; for other rates only the resampler differs (FunASR loads/resamples with torchaudio — **to verify**), so measure the fbank difference and check decoded text is identical |

### 7.2 Pin the ffmpeg behavior first (like §1)

Everything below is **to be verified against the pinned ffmpeg version, by experiment first** (see
§7.2.1 for how the swr source may be used) before writing the resampler. Current understanding:

1. **Pipeline**: decoder output (s16 / s32 / fltp depending on codec) → auto-inserted `aresample`
   (libswresample) doing rematrix + resample + format conversion to s16.
2. **Downmix (`-ac 1`)**: swr builds a matrix with FL→FC and FR→FC at `M_SQRT1_2`, then
   normalizes so the row sum is ≤ 1 for integer output → **0.5·L + 0.5·R** for stereo.
   Multichannel (5.1 etc.) uses center/surround mix levels — implement the matrix builder for
   the standard layouts (mono, stereo, 2.1, quad, 5.0, 5.1, 7.1) and fall back to equal-weight mean
   for unknown layouts (documented deviation).
3. **Resampler defaults** (`swr` engine): `filter_size=32`, `phase_shift=10` (1024 phases),
   `filter_type=kaiser`, `kaiser_beta=9`, `cutoff=0.97` (relative to the lower Nyquist),
   `exact_rational=1` (for rational ratios like 160/441 the phase count becomes the exact
   denominator, so no linear interpolation between phases), `dither_method=none`.
   Filter taps are generated in double, then stored in the internal sample format.
4. **Internal sample format**: swr picks s16p / s32p / fltp / dblp depending on input/output
   formats — this decides the rounding (s16 resampling uses 15-bit fixed-point taps and
   `(acc + 0x4000) >> 15` with clipping). Must reproduce this choice per decoder output format to
   reach bit-level agreement:
   - WAV/FLAC 16-bit → likely s16p path (integer taps)
   - MP3 (fltp) / 24-bit FLAC (s32) → float/s32 path
5. **Edge handling**: initial delay (filter half-length, primed with zeros), flushing at EOF, and
   exact output length `ceil(n_in * out_sr / in_sr)` vs. what ffmpeg actually emits — measure.
6. **Float → s16**: `lrintf(x * 32768)` with saturation (no dither).
7. **MP3 gapless**: ffmpeg reads the LAME/Xing header and skips `encoder_delay + 529` samples at the
   start and the padding at the end. Check what `dr_mp3` does and replicate.

Deliverable: `docs/load.md` — a **behavior spec** (parameters, rounding rules, delay/length rules,
internal-format selection conditions, downmix coefficients; prose, formulas and tables only, **no
code excerpts**) + a Python reference `ffmpeg_resample_ref` in `tests/python/reference.py`
**written from that spec** and validated against the real ffmpeg (bit-exact for the integer path).

#### 7.2.1 License policy (spec-first, clean-room style)

libswresample is **LGPL-2.1+**; asrfront is MIT and statically links its C core into the wheel, so
no LGPL-derived code may enter the repository (C *or* Python, tests included). Algorithms,
parameter values and numeric rules (Kaiser-windowed-sinc polyphase filtering, `filter_size=32`,
β=9, cutoff 0.97, 0.5/0.5 downmix, `(acc + 0x4000) >> 15` rounding) are not copyrightable;
the code's expression (function structure, variable layout, table-generation code, comments) is.

1. **Experiment first**: infer behavior from ffmpeg's output on probe signals — impulse
   (→ filter taps and delay), DC (→ tap normalization, int rounding), sines (→ cutoff, passband),
   lengths 0..N (→ output length / flush rule), stereo/5.1 test tones (→ downmix matrix).
2. **Source only to resolve what experiments can't explain**, and only recorded as *behavior* in
   `docs/load.md` (e.g. "taps are scaled by 2^15 and rounded to nearest"), never as code.
   Note in the doc which facts came from experiment vs. source reading, with the ffmpeg version.
3. **Implementation from the spec only**: M9.3 (`resample.c`, `remix.c`) and `ffmpeg_resample_ref`
   are written from `docs/load.md` + standard DSP literature (e.g. J. O. Smith, "Digital Audio
   Resampling"), without the swr source open. Own structure and naming; no copied comments,
   constants tables or helper code.
4. **Fallback**: if bit-exactness turns out to need code-level mimicry, drop the bit-exact goal
   for that path and keep the ≤ 2 LSB target (§7.1). Impact is limited to `resample.c` / `remix.c`.
5. **Other components**: dr_libs (Unlicense / MIT-0) is fine to vendor — ship `LICENSE.dr_libs`
   and make sure `csrc/third_party/` is in the sdist. ffmpeg (incl. the GPL `imageio-ffmpeg`
   build) and openai-whisper are **test-only tools** run as subprocess / imported in tests, never
   bundled or linked → no effect on the wheel's license. MP3 patents have expired; FLAC is
   royalty-free. Test fixtures are generated at test time, not vendored.
6. Not legal advice; revisit if the project is used commercially with stricter requirements.

#### 7.2.2 M9.1 results

Spec: [`docs/load.md`](docs/load.md); reference: `reference.py` "ffmpeg load pipeline"
(`ffmpeg_load_ref`); checks against the real ffmpeg: `tests/python/test_ffmpeg_ref.py`
(127 tests, `uv run --group load pytest tests/python/test_ffmpeg_ref.py`; also added to the
`reference` CI job). Reference: ffmpeg 7.0.2 from `imageio-ffmpeg==0.6.0` (new `load` group).

What the experiments found (several items differ from the "current understanding" above):
- **Order**: every channel is resampled first, then downmixed.
- **Three arithmetic paths** chosen by decoder sample format: **int16** (u8 always; s16 when not
  resampled — Q15 weights/taps, `(acc + 16384) >> 15`), **float32** (s16/s32/f32 when resampled
  or mixed), **float64** (f64). 16-bit WAV/FLAC at 44.1/48 kHz therefore takes the *float* path,
  not the integer one assumed earlier.
- **Filter**: Kaiser-windowed sinc, β = 9, `fc = min(1, 0.97·16000/sr)`, `N = ceil(32/fc)` rounded
  up to even, all phases normalized by **phase 0's** tap sum; exact rational phases when the
  reduced denominator ≤ 1024 (every common rate), else 1024 phases + linear interpolation.
  Edges: reflect at the start, mirror-including-last at the end. Matched to ~1e-16 on ffmpeg's
  double path for 15 rates.
- **ffmpeg's float path is CPU-dependent**: on AVX2+FMA3 it is an 8-accumulator FMA dot product with
  tree reduction (reproduced bit-exactly); plain float32 summation is off by 1 LSB in ~0.03 % of
  samples. Without FMA (older x86, ARM) ffmpeg itself gives slightly different results.
- **Quirks reproduced**: int16 downmix wraps instead of saturating (7/8 channels, Q15 weight sum >
  32768); Q15 tap 1.0 is stored as 32767 (u8 upsampling).
- **Output length** depends on ffmpeg's internal buffering at end of stream and could not be
  reduced to a formula; asrfront uses `ceil(n·16000/sr)` (sr ≤ 32 k) / `ceil((n−1)·16000/sr)`
  (sr > 32 k), exact for most rates, ±1 last sample for 22.05k/44.1k/88.2k in 3–6 % of lengths.
- **Not reproduced (documented ≤ 1 LSB)**: interpolated mode's exact float order (non-standard
  rates), f64 downmix with ≥ 5 channels.
- Source consulted (prose summaries only, §7.2.1) for: end-of-stream padding and the length
  mechanism (af_aresample / swresample / resample behavior). No code copied.

**Consequence for M9.3**: to be bit-exact with ffmpeg on the common path, the C float resampler
must use the same 8 accumulators, **fused multiply-add** (`fmaf`) and tree reduction on every ISA.
This replaces the project's "no FMA" rule *for this kernel*: `fmaf` is correctly rounded by the C
standard, so hardware FMA (AVX2+FMA3 variant) and the libm fallback (generic/scalar variants)
still give bit-identical results; the scalar fallback is slower on CPUs without FMA.

#### 7.2.3 M9.2 results

- C: `csrc/wav.c` (RIFF/RF64/BW64, PCM u8–s32, float, A-law/µ-law, extensible masks, damaged
  data sizes), `csrc/remix.c` (layouts, downmix, the three arithmetic paths, s16 conversion),
  `csrc/audio_load.c` (detection, `af_load_memory` / `af_load_file` / `af_audio_free`), internal
  header `csrc/audio.h`. New error codes `AF_EUNSUPPORTED`, `AF_EIO`, `AF_ECORRUPT`.
  `-ffp-contract=off` added for all C code (no implicit FMA; explicit `fmaf` stays allowed).
- Python: `asrfront.load(path | bytes)`, `asrfront.SAMPLE_RATE`; non-16 kHz files raise
  `ValueError` until M9.3, FLAC/MP3 until M9.4.
- Tests: `tests/c/test_load.c` (hand-made buffers, every truncation of a file);
  `tests/python/test_load.py` (132 tests: 8 codecs × 1–8 channels, 18 named layouts × 3 codecs,
  RF64, damaged/truncated files, mismatched masks, float specials, bytes == path, raw PCM, MP3
  false-positive patterns, threads) — **bit-exact against whisper's ffmpeg command**.
  Also 2·10⁶ random mutations of a WAV file under ASan/UBSan without findings.
- Extra ffmpeg behavior found and specified: out-of-range / non-finite float samples (double path
  reproduced; float path position-dependent in ffmpeg, asrfront uses the SIMD rule).

#### 7.2.4 M9.3 results

- `csrc/resample.c` / `resample.h` (internal), written from `docs/load.md` only: Kaiser-sinc
  bank (power-series I₀), exact / interpolated phases, mirrored edges, length rule; three paths:
  float32 (8 accumulators + `fmaf` + tree reduction), float64, int16 (Q15, accumulator
  interpolation for non-standard rates). Supported input rates 1 kHz – 384 kHz.
- `remix.c` now works on planar channels: extract in the path's type → resample each channel
  → downmix → s16. `af_audio_to_mono` allocates its output (length changes).
- Python reference gained the int16 interpolated mode (≤ 1 LSB vs. ffmpeg, measured; documented).
- Tests: `test_resample.py` (119: C vs. reference **bit-exact** on all paths, 18 rates incl.
  1 kHz / 384 kHz and interpolated mode, inputs shorter than the filter, bank vs. reference);
  `test_load.py` +130 resampled files vs. whisper's ffmpeg command (5 codecs × 1/2/6 channels ×
  8 rates, interpolated rates, 11 s speech at 44.1k/48k/22.05k/8k) — bit-exact except the
  documented deviations. ASan/UBSan: C tests, 3·10⁵ mutated WAVs, 400 random rates × short inputs.
- **Speed (33 s speech, i5-14400F)**: 16 kHz 0.5 ms; resampled 22–103 ms (44.1k stereo 86 ms)
  — slower than the ffmpeg subprocess (13–38 ms incl. process start) because the generic build
  calls libm `fmaf` per tap. Fix in M9.5: AVX2+FMA3 kernel variant (dispatch must also check
  FMA3), 8-tap vectorized dot products.

#### 7.2.5 M9.4 results

- Vendored dr_flac / dr_mp3 at dr_libs `dfe8377` (newer than the last releases for their
  Xing/metadata overflow fixes; `csrc/third_party/README.md`), compiled as a separate object
  library without warnings, with `DR_*_NO_STDIO`, `DR_FLAC_NO_OGG`, `DR_MP3_FLOAT_OUTPUT` and
  `-fwrapv`. One local patch in dr_flac (signed shifts, found by UBSan).
- `csrc/flac.c`: s16 for ≤ 16-bit, s32 otherwise; FLAC layouts + ffmpeg's
  `WAVEFORMATEXTENSIBLE_CHANNEL_MASK` Vorbis comment. `csrc/mp3.c`: unclipped float output,
  gapless trimming from the LAME header (done by dr_mp3).
- Findings: dr_mp3's default output is s16-based and **clipped** → thousands of LSB off after
  resampling; float output fixes it (decoder difference ≤ 5e-6 → ≤ 1 LSB). ffmpeg stores
  non-FLAC layouts in a Vorbis comment (3-/4-channel files failed until it was parsed).
- Tests (`test_load.py`, +191): FLAC s16/s24 × 1–8 channels × 6 rates and 5 layouts —
  **bit-exact**; MP3 CBR/VBR/no-Xing/ID3v1/no-ID3 × mono/stereo × 8 rates — **≤ 1 LSB, same
  length**; speech at 44.1 kHz stereo; bytes input and misnamed files (content detection);
  truncated files. ASan/UBSan: 2·10⁵ mutated FLAC/MP3 files.
- MP3 accuracy is much better than the planned 1e-3 target (≤ 1 LSB ≈ 3e-5).

#### 7.2.6 M9.5 results

- Kernel table gained `resample_dot` / `resample_dot4` (8 FMA accumulators + tree; 4 outputs
  interleaved). The `avx2` table is now built with `-mfma` and selected only with AVX2 **and**
  FMA3; it uses `_mm256_fmadd_ps`. generic/scalar use `fmaf()` (libm call on x86-64 without
  FMA3, one instruction on aarch64). All three variants pass the bit-exact tests.
- Other speed-ups: incremental output positions (no 64-bit divisions), branch-free per-format
  extraction, libm-free round-half-even for the s16 conversion, output written into channel 0's
  buffer (fewer large allocations — page faults dominated small inputs on WSL2). Tried and
  reverted: `mmap` for path input (16 kHz WAV 1.2 → 0.4 ms in Python, but a concurrently
  truncated file would crash with SIGBUS).
- Speed, 30 s speech (`bench/compare.py` [5], docs/benchmark.md §6): WAV 16 kHz 0.32 ms
  (33x faster than whisper's ffmpeg subprocess, 8x librosa), 44.1 kHz stereo 14 ms (2.5x / 2.2x;
  M9.3: 86 ms), MP3 44.1 kHz stereo 25 ms (1.5x / 1.4x); FLAC on par with ffmpeg; 8 kHz
  upsampling slower than librosa/soxr (0.8x).
- SenseVoice end-to-end (docs/load.md): identical to FunASR's loading at 16 kHz; for resampled
  input the fbank difference is below the torchaudio-vs-soxr difference. Model-level checks
  (decoded text) were not run: no model weights here.
- README: `asrfront.load` usage, accuracy and performance; benchmark doc §6.

#### 7.2.7 M9.6 results (more containers)

- Added Wave64, AIFF / AIFF-C, CAF and NIST SPHERE containers and the IMA ADPCM (WAV), Microsoft
  ADPCM and IMA4 (QuickTime) codecs. All behavior was established by experiment against ffmpeg
  7.0.2 (§7.2.1, no ffmpeg source consulted) and written down in `docs/load.md` ("Containers"):
  e.g. AIFF rates round halves up while CAF rates truncate, CAF 8-bit PCM is always signed, the
  CAF/AIFF sample size comes from the bit depth only, MS ADPCM rounds its prediction toward zero
  and ignores the `fmt ` coefficients, IMA WAV uses the product form of the difference and IMA4
  the shift-and-add form with state carried across packets.
- CoreAudio channel layouts (AIFF `CHAN`, CAF `chan`): the downmix weights of every predefined
  tag were measured by least squares; ffmpeg fails on layouts outside its native order
  (including MPEG_7_1_C, which it writes itself for 7.1) — asrfront falls back to the default
  layout there (documented deviation).
- Not added: big-endian RIFX (ffmpeg 7.0.2 decodes the samples little-endian, i.e. noise, so
  matching it is pointless and deviating from it is untestable), GSM 6.10 (no encoder in the
  reference build to test against), G.722 (rare).
- Accuracy: bit-exact against ffmpeg on ~600 new tests (`tests/python/test_load_formats.py`:
  ffmpeg-written files for every codec × 1/2/6 channels × 16/44.1 kHz, crafted header / damage /
  ADPCM corner cases, jfk speech), except the known f64 ≥ 5-channel near-tie (≤ 1 LSB).
  Multichannel (> 2) IMA WAV, which ffmpeg cannot decode, is checked per channel against
  ffmpeg's mono decoding.
- Robustness: C tests on hand-built buffers incl. every truncation; ~1.5 M mutated files
  (all new containers and codecs) under ASan + UBSan without findings.

### 7.3 C design

New files:

```
csrc/
├── audio_load.c                   # format sniffing, dispatch, downmix → resample → s16 → float
├── wav.c                          # RIFF/RF64 + Wave64 parser
├── aiff.c / caf.c / sphere.c      # M9.6: AIFF/AIFF-C, CAF (+ CoreAudio layouts), NIST SPHERE
├── adpcm.c                        # M9.6: IMA (WAV), Microsoft and IMA4 (QuickTime) ADPCM
├── pcm.c                          # PCM / float / G.711 conversion shared by the containers, raw PCM
├── remix.c                        # ffmpeg-compatible downmix matrix
├── resample.c / resample.h        # internal: swr-compatible polyphase resampler (s16 / float paths)
└── third_party/
    ├── dr_flac.h                  # mackron/dr_libs, pinned commit, public domain / MIT-0
    ├── dr_mp3.h
    └── LICENSE.dr_libs
```

**WAV** (in-house): `RIFF`/`RF64` (big-endian `RIFX` not supported), chunks `fmt `, `data`, skip `LIST`/`fact`/unknown
(odd-size padding). Formats: PCM u8/s16/s24/s32, IEEE float32/64, `WAVE_FORMAT_EXTENSIBLE`
(sub-format GUID + channel mask → layout for downmix). A-law/µ-law: cheap to add, optional.
Truncated `data` size (streamed WAVs with size 0/0xFFFFFFFF) → read to EOF like ffmpeg.

**Raw PCM**: fixed s16le / 16 kHz / mono, so it is just `int16 / 32768` (no resampling, no
downmix). An odd trailing byte is dropped (to be checked against ffmpeg `-f s16le -ar 16000 -ac 1 -i`).

**Format detection** by magic bytes (`RIFF..WAVE`, `RF64`, `fLaC`, `ID3`/MPEG frame sync), then:

| source | magic found | no magic |
|---|---|---|
| path `*.pcm` / `*.raw` | raw PCM (extension wins) | raw PCM |
| other path | that container | `AF_EUNSUPPORTED` |
| bytes | that container | **raw PCM** (fallback) |

- The bytes fallback means garbage bytes decode as noise instead of raising — accepted because raw
  PCM bytes (streaming / websocket audio) are the main reason to pass bytes without a container.
  Truncated or corrupt *containers* (magic present) still raise `AF_ECORRUPT`.
- MPEG sniffing must be **strict** so raw PCM is never mistaken for MP3 (int16 `-1` is `FF FF`,
  which looks like a frame sync): require a Layer III header with valid bitrate / sample-rate
  indices **and** a second valid header at the computed frame offset (ID3v2 tag skipped first).
  Tested with raw PCM of silence, `-1` runs, noise and speech.

**Public API** (`asrfront.h`) — just load and free:

```c
enum { AF_EUNSUPPORTED = -4, AF_EIO = -5, AF_ECORRUPT = -6 };

/* Load an audio file as 16 kHz mono float32 (ffmpeg/whisper-compatible).
 * *out is allocated by the library; release it with af_audio_free. */
int  af_load_file(const char *path_utf8, float **out, size_t *n_samples);
/* Same for an in-memory file (detection table above). */
typedef enum {
    AF_RAW_AUTO = 0,  /* container if magic found, else raw PCM (bytes input) */
    AF_RAW_NEVER = 1, /* container only, else AF_EUNSUPPORTED (path without .pcm/.raw) */
    AF_RAW_FORCE = 2  /* always raw s16le 16 kHz mono (.pcm/.raw path) */
} af_raw_mode;
int  af_load_memory(const void *buf, size_t size, af_raw_mode raw, float **out, size_t *n_samples);
void af_audio_free(float *p);
```

- Output size is unknown up front for compressed formats, so unlike the frontends the loader
  allocates. Bound checks on every header field (`n_frames * channels` overflow, absurd sample
  rates/channels → `AF_ECORRUPT`); fuzz-friendly.
- `af_load_file` maps the extension to `AF_RAW_FORCE` / `AF_RAW_NEVER`; on Windows it converts
  UTF-8 → UTF-16 and uses `_wfopen`. The Python binding reads the file itself and calls
  `af_load_memory` for both paths and bytes (one code path, avoids path encoding issues).
- The resampler and downmix are **internal** (`csrc/resample.h`, `csrc/remix.c`), not part of the
  public API; tests reach them through the `_ext` test hooks like the existing internal kernels.
  One-shot only (whole file in memory); a streaming resampler can be exposed later if needed.
- Resampler hot loop: polyphase FIR, taps contiguous per phase; reuse the `kernels_impl.c`
  multi-ISA scheme. Three paths per §7.2.2 (int16 / float32 / float64). The float32 path uses
  8 accumulators + `fmaf` + tree reduction (ffmpeg's AVX2/FMA3 kernel) on **every** ISA, so all
  variants stay bit-identical (AVX2 build adds `-mfma`; FMA3 must then be checked at dispatch).

### 7.4 Python API

```python
def load(source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> np.ndarray:
    """Load audio as Whisper / SenseVoiceSmall input.

    source: a file path, or the file's contents as bytes.
    WAV, FLAC, MP3, or raw PCM (s16le, 16 kHz, mono): a `.pcm` / `.raw` path, or bytes
    without a recognized container header.
    Returns a 1-D float32 array at 16 kHz mono, identical to whisper.load_audio(path).
    """
```

- `str` / `PathLike` → path (`str` is never interpreted as data). Any object supporting the buffer
  protocol (`bytes`, `bytearray`, `memoryview`, 1-D uint8 `np.ndarray`) → bytes; must be
  C-contiguous, read zero-copy (no copy of the input) while the GIL is released. The binding keeps a
  reference to the buffer for the duration of the call; a `bytearray` mutated from another thread
  during the call is undefined (documented). Other types → `TypeError`.
- File-like objects (`BinaryIO`) are not accepted directly: `load(f.read())` (keeps one input rule).

- Returns only the array (the rate is always 16 kHz; `asrfront.SAMPLE_RATE = 16000` exported for
  reference). No `load_batch` / `resample` in the public API.
- GIL released during decode/resample; zero-copy: the C buffer is wrapped in a capsule-owned ndarray.
- Errors: `FileNotFoundError` / `OSError` for I/O, `ValueError` for corrupt or unsupported files
  (message lists the supported formats and suggests converting with ffmpeg).

### 7.5 Tests

- **Reference tool**: ffmpeg is not installed here. Use the `imageio-ffmpeg` static binary in a new
  `load` dependency group (pinned version, same on CI) and `openai-whisper`'s `load_audio` command
  line; tests are skipped when ffmpeg is unavailable. Record the ffmpeg version in test output.
- **Fixtures generated at test time** (not vendored) with ffmpeg / soundfile from the existing
  `audio_signals.py` signals + jfk: every combination of
  - container: WAV (u8, s16, s24, s32, f32, f64, extensible), raw PCM, FLAC (16/24-bit), MP3 (CBR/VBR, with/without Xing header)
  - rate: 8000, 16000, 22050, 24000, 32000, 44100, 48000
  - channels: 1, 2, 6 (5.1)
- Compare `load()` vs. ffmpeg output: exact length, int16 max/mean diff per §7.1 targets.
- Bytes input: `load(open(f, "rb").read()) == load(f)` for every fixture (container files);
  raw PCM bytes == `.pcm` path; `bytearray` / `memoryview` / non-contiguous memoryview (→ error);
  `str` is never treated as data.
- Sniffing robustness: raw PCM bytes of silence, `-1` runs, noise, speech, and PCM that starts with
  `FF Fx` must **not** be detected as MP3.
- Edge cases: empty data chunk, truncated files, odd chunk sizes, odd-length `.pcm` / bytes,
  empty bytes, 1-sample input, huge header sizes (must not allocate), non-audio file / unknown
  extension → error.
- End-to-end per §7.1: Whisper (`whisper.load_audio` + `log_mel_spectrogram`) and SenseVoice
  (FunASR `WavFrontend` on FunASR-loaded audio).
- C tests: WAV parser on hand-crafted byte buffers; resampler impulse/DC/sine (passband ripple,
  stopband attenuation) through the internal header.
- ISA consistency (`ASRFRONT_KERNELS=scalar|generic|avx2`) and ASan/UBSan job as in M8;
  a small libFuzzer target for `af_load_memory` (run locally, optional in CI).

### 7.6 Steps

| Step | Contents | Est. |
|---|---|---|
| M9.1 ✅ | Pin ffmpeg behavior per §7.2.1: probe-signal experiments first, swr source only for unexplained gaps (behavior notes only); write the `docs/load.md` spec; then write `ffmpeg_resample_ref` (Python) **from the spec** and confirm downmix/resample/rounding/length rules bit-exact vs. ffmpeg | 1.5 d |
| M9.2 ✅ | WAV + raw PCM parser, downmix, s16 rounding, `af_load_memory/file` + format detection rules, Python `load(path | bytes)` (no resampling yet: non-16 kHz → `AF_EUNSUPPORTED` temporarily) | 1 d |
| M9.3 ✅ | Resampler **from `docs/load.md` + DSP literature only, swr source not consulted** (§7.2.1): s16 integer path + float path, exact length/delay handling; internal only (one-shot) | 1.5 d |
| M9.4 ✅ | FLAC (dr_flac) + MP3 (dr_mp3, gapless trim) integration, strict MP3 sniffing | 1 d |
| M9.5 ✅ | SIMD resampler kernels, benchmark vs. `whisper.load_audio` (ffmpeg subprocess) / `librosa.load`, end-to-end SenseVoice check, ASan + fuzz, README/docs, wheel size check | 1.5 d |
| M9.6 ✅ | More containers / codecs (§7.2.7): Wave64, AIFF / AIFF-C, CAF, NIST SPHERE, IMA / MS / IMA4 ADPCM; behavior pinned by experiment, docs, tests, fuzz | 1 d |

**Total ~7.5 days.**

### 7.7 Risks

1. **swr internals are version-dependent** (defaults changed historically, e.g. `exact_rational`).
   Pin one ffmpeg version as the reference and document it; accept ≤ 2 LSB elsewhere.
2. **MP3 is not bit-exact by nature**: ffmpeg's decoder and dr_mp3 (minimp3-derived) differ in
   float rounding. Alignment (delay/padding) matters more for ASR than LSB differences — test length
   and cross-correlation lag = 0 explicitly.
3. **Int path vs. float path**: if bit-exactness requires replicating swr's s16 fixed-point
   resampling, the same file decoded as s16 vs. s32 takes different code paths — mirror swr's
   internal format selection exactly.
4. **Security**: parsing untrusted files in C. Strict bounds checks, no size from headers trusted
   for allocation without validation against the buffer size, fuzzing (M9.5).
5. **Wheel size / build**: dr_flac + dr_mp3 add ~10k lines of C; check compile warnings under
   `-Wall -Wextra -Wpedantic` (compile them with relaxed warnings in a separate object library if needed).
6. **LGPL contamination**: bit-exact swr compatibility tempts code-level mimicry. Follow §7.2.1
   (spec-first, implementation without the source open); fall back to ≤ 2 LSB if needed.
