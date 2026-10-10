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
- [x] Wheels with cibuildwheel (`.github/workflows/wheels.yml`, config in `[tool.cibuildwheel]`): cp39–cp311 regular wheels + one **cp312-abi3** wheel per platform (nanobind `STABLE_ABI`, `wheel.py-api = "cp312"`; `abi3audit --strict` clean), manylinux x86_64/aarch64, Windows AMD64 (macOS removed from CI and wheels on request); sdist; PyPI trusted publishing on GitHub release (needs the `pypi` environment registered on PyPI)
  - verified locally: `uv build` (sdist → wheel), fresh venvs: abi3 wheel on Python 3.13 and sdist build on Python 3.9 both pass the test suite
- [x] README: install, usage, performance and accuracy tables; MIT `LICENSE`
- Portability work done here:
  - `long` → `ptrdiff_t` / `long long` for sample indices (32-bit `long` on Windows)
  - third kernel variant **`scalar`** (no vector extensions = the code path MSVC compiles), built on every platform and selectable with `ASRFRONT_KERNELS=scalar`; bit-identical to the others (C + Python tests). Slower (whisper 3.8 ms / 30 s) but a correct fallback
  - cross-compiled the C core with clang (zig) for aarch64-linux, arm64/x86_64-macOS and x86_64-windows-gnu without warnings. **Not verified here**: MSVC (no Windows SDK; the `cpuid` dispatch and `/arch:AVX2` build only run in CI), and actually running on macOS/ARM/Windows

**Total estimate: ~12–13 working days** (all milestones implemented; first GitHub CI run pending)

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
6. **Follow-up candidates**: streaming kaldi fbank (easy for SenseVoice since frames are independent), resampler, presets for additional models (Paraformer, other Whisper-style models, Qwen-Audio, etc.).
