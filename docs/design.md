# Design notes

Why asrfront is built the way it is: the principles, the decisions behind the public API, the
license policy for the audio loader, how it is tested, and what was tried and dropped. The
algorithms themselves are specified in [`whisper.md`](whisper.md),
[`sensevoice.md`](sensevoice.md) and [`load.md`](load.md); measurements are in
[`benchmark.md`](benchmark.md).

## Goals and principles

- **Scope**: produce the exact input of Whisper (`log_mel`), SenseVoiceSmall (`fbank` + LFR +
  CMVN) and Qwen3-ASR (Whisper's log-mel with 128 mels, no pad / trim), and load audio files into the 16 kHz mono waveform both expect. Nothing else: no
  general-purpose DSP, no GPU, no streaming (yet, see [Open items](#open-items)).
- **No dependencies**: the C core links only libc/libm. The FFT and SIMD kernels are written
  in-house; FLAC/MP3 decoding uses vendored single-header decoders (dr_flac / dr_mp3, public
  domain or MIT-0) compiled into the core.
- **Accuracy is measured against float64 references** (`tests/python/reference.py`), not against
  the original float32 libraries. torchaudio builds its kaldi mel banks in float32 (errors up to
  ~1.4e-5), which with its float32 FFT gives up to ~1.3e-3 log-domain error on low-energy bins, so
  "matches torchaudio" is not a precise target. Targets: Whisper 1e-4, kaldi fbank 1e-3 (log
  domain); agreement with the original libraries is expected only to ≤ 1e-4 (whisper/torch) and
  ≤ 2e-3 (torchaudio).
- **Reproducible arithmetic**:
  - no `-ffast-math`, and `-ffp-contract=off` everywhere (also for the vendored decoders), so the
    compiler never fuses or reorders floating-point operations;
  - every kernel variant (AVX2, baseline SSE2/NEON, plain C) uses the same operation order and gives
    **bit-identical** results; tests compare the variants exactly;
  - the one place that uses FMA is the resampler's float path, which must reproduce ffmpeg's
    AVX2/FMA3 dot product bit for bit ([`load.md`](load.md)). It calls `fmaf()` explicitly, which
    the C standard requires to be correctly rounded, so hardware FMA and the libm fallback agree.

## C core

- **Opaque handles** (`af_whisper`, `af_sensevoice`) hold everything that can be precomputed:
  FFT plans and twiddles, sparse mel banks, windows, scratch buffers. Create once, reuse; the hot
  path does not allocate. SenseVoice keeps the raw fbank frames LFR still needs in a ring of
  `lfr_m + 63` frames (rounded up to blocks of 8), filled 64 frames at a time, so its memory
  does not grow with the input; an earlier grow-only buffer for the whole raw fbank stayed at
  the size of the longest input in every cached handle (hundreds of MB after a batch of long
  inputs).
- **Caller-allocated output**: `*_out_frames()` gives the size, the caller passes the buffer. The
  loader is the exception (the decoded length is unknown up front), so it allocates and provides
  `af_audio_free`.
- **Handles are not thread-safe** (they own scratch buffers): use one per thread.
- **Qwen3-ASR reuses the Whisper handle** (128 mels, no pad / trim): its only C addition is
  `af_whisper_log_mel_padded`, the Whisper computation over virtual zeros up to any length
  (`pad_or_trim` is the case of 480 000), which reproduces transformers' batches padded to
  their longest item before the STFT.
- **Error codes** are negative ints; `*_create` returns the handle through an out-parameter so
  invalid arguments and allocation failures can be told apart.
- **No threading inside the core**: a 30 s input takes about 1 ms, so starting and synchronizing
  threads would eat most of the gain. Parallelism comes from processing many inputs at once (see
  below). The `n_threads` config fields are accepted but currently unused.

### SIMD kernels

- Frames are processed **8 at a time in a structure-of-arrays layout** (`buf[i * 8 + lane]`): FFT
  passes, the fused real-FFT split + power, mel, log and kaldi preprocessing all loop over lanes,
  so they vectorize at any SIMD width. Frames are gathered with register transposes.
- Kernels use GCC/Clang vector extensions at the native width (8 floats with AVX, 4 with SSE2 /
  NEON); other compilers (MSVC) get plain floats.
- `kernels_impl.c` is compiled once per ISA: `generic` (baseline), `avx2` (x86-64, with `-mfma`
  for the resampler only) and `scalar` (no vector extensions = the MSVC code path, built
  everywhere so it is tested on every platform). The table is selected at runtime with cpuid
  (AVX2 **and** FMA3, OS support for YMM state); `ASRFRONT_KERNELS=generic|scalar` forces a
  variant.
- `log`/`log10` use a branch-free vectorized Cephes polynomial (≈ 1 ulp) instead of libm, which
  alone cost 1.2 ms per 30 s.

## Python binding

- nanobind, one C++ file (`src/asrfront/_ext.cpp`) with thin RAII wrappers; argument
  normalization (dtype, contiguity, int16 scaling, validation with helpful messages) is done in
  Python.
- **Zero-copy output**: the C buffer is handed to NumPy through a capsule. The low-level `out=`
  argument writes straight into a caller buffer, which the batch functions use to avoid copies.
- **The GIL is released** during every native call, so Python threads run in parallel.
- **Free-threaded CPython** (3.14t): the module declares `FREE_THREADED`, so importing it
  keeps the GIL disabled. Nothing else had to change: the C core already runs concurrently
  (GIL released), its only shared state (the resampler cache) has a lock, and the Python
  side keeps its shared state (pool, handle caches) behind a lock or in thread-local storage.
  Handles stay one per thread.
- **Per-thread handle cache** (`threading.local`), keyed on the configuration, since handles are
  not thread-safe. It is a small LRU (8 handles per thread), so callers that vary the
  configuration (e.g. a dither seed per call) do not accumulate handles; the seed is left out
  of the key when dither is 0.
- **Batch parallelism in Python**: `*_batch` functions run items on a thread pool. This is simpler
  and scales better than splitting one input across threads in C. The pool is shared and
  persistent (created on first use with one thread per available core, reset after `fork`),
  so a batch call neither starts threads nor rebuilds the workers' cached handles; starting a
  pool per call cost more than the work for small batches. A call with `n_threads` above the
  core count gets a pool of its own, shut down afterwards, so the threads kept for the process
  stay bounded. A batch call from inside a worker runs inline instead of waiting on the pool.
  Only the cheap input checks (shape, dtype, length) run before the batch; the conversion to
  float32 and the NaN / inf check, a pass over every sample, run in the workers (moved there,
  an 8-thread batch got ~7 % faster with float32 input and ~2× with float64 input). A NaN /
  inf error is raised after the batch for the lowest bad index, as a serial check would.

## Audio loading (`asrfront.load`)

### One function, one argument

Both models take the same input (16 kHz mono float32 in [-1, 1]), so one fixed pipeline serves
both: decode → resample each channel → downmix → round through int16 → float32, matching
whisper's `load_audio` (ffmpeg). librosa-style options (`sr`, `mono`, `offset`, `duration`,
`dtype`, `res_type`) are deliberately left out; callers can slice the result or use librosa.
Options can be added later without breaking `load(source)`.

- `str` / `os.PathLike` is always a path, never data. Any C-contiguous buffer (`bytes`,
  `bytearray`, `memoryview`, 1-D uint8 array) is file contents, read without copying. File-like
  objects are not accepted (`load(f.read())` keeps one input rule).
- Only the array is returned (the rate is always 16 kHz, exported as `asrfront.SAMPLE_RATE`).
- The Python side reads files itself (`read()`, not `mmap`: a file truncated by another process
  while mapped would crash the interpreter with SIGBUS) and passes the bytes to
  `af_load_memory`, so paths and bytes share one code path and path encoding is not an issue.
- The resampler and downmix are internal (`csrc/resample.h`, `csrc/remix.c`), one-shot over the
  whole file. Tests reach them through `_ext` test hooks.
- Resamplers are read-only once built, so `load` shares them through a small process-wide
  cache (4 entries of ≤ 1 MiB, LRU among idle entries, reference-counted, mutex-protected):
  building the 44.1 kHz bank (160 phases of Kaiser-windowed sinc) cost ~0.5 ms, about 70 % of
  loading a 2 s clip. This is the only shared state in the C core.
- Untrusted input: malformed files must fail cleanly (bounds checks on every header field, no
  allocation sized from an unvalidated header, fuzzing in CI). The decoded size, however, can be
  far larger than the file, so `load(..., max_duration=)` rejects long audio from the frame
  count, before allocating the samples where the codec allows (every format but FLAC / MP3,
  which stop one frame past the limit) ([`load.md`, "Resource use"](load.md#resource-use)).
  It was first left to callers, but checking the duration before `load` means parsing the
  header twice, and for FLAC / MP3 there is no reliable header length to check.

### Format detection

The container is detected from magic bytes. Raw PCM (always s16le, 16 kHz, mono) has none:

| Source | Magic found | No magic |
|---|---|---|
| path `*.pcm` / `*.raw` | raw PCM (extension wins) | raw PCM |
| other path | that container | error (unsupported) |
| bytes | that container | **raw PCM** |

- The bytes fallback means garbage bytes decode as noise instead of raising. This is accepted
  because raw PCM bytes (streaming / websocket audio) are the main reason to pass bytes without
  a container. Truncated or corrupt containers (magic present) still raise.
- MPEG detection is strict so that raw PCM is never taken for MP3 (int16 `-1` is `FF FF`, which
  looks like a frame sync): a valid Layer III header **and** a consistent second header at the
  computed frame offset are required (an ID3v2 tag is skipped first).

### License policy (clean-room style)

ffmpeg's libswresample is LGPL-2.1+; asrfront is MIT and links its C core statically into the
wheel, so **no LGPL-derived code may enter the repository** (C or Python, tests included).
Algorithms, parameter values and numeric rules are not copyrightable; the code's expression is.

1. **Experiment first**: infer ffmpeg's behavior from its output on probe signals (impulses → taps
   and delay, DC → normalization and rounding, sines → cutoff, length sweeps → output length,
   multichannel tones → downmix weights).
2. **Source only for what experiments cannot explain**, recorded in [`load.md`](load.md) as
   behavior in prose (never code), with each rule marked as found by experiment or by reading the
   source, and the ffmpeg version.
3. **Implement from the spec only**: `resample.c`, `remix.c` and the Python reference are written
   from `load.md` and standard DSP literature, without the ffmpeg source open; own structure and
   naming, no copied comments, tables or helpers.
4. **Fallback**: if bit-exactness ever required code-level mimicry, drop it for that path and
   accept ≤ 2 LSB instead.
5. ffmpeg (including the GPL `imageio-ffmpeg` build) and openai-whisper are test-only tools, run
   as subprocesses or imported in tests, never bundled or linked. Test fixtures are generated at
   test time, not vendored.

This is not legal advice; revisit it if the project is used under stricter requirements.

## Testing

| Level | What |
|---|---|
| Reference ports | `tests/python/reference.py`: float64 NumPy ports of whisper's `log_mel_spectrogram` (+ librosa slaney filters), torchaudio's kaldi fbank, FunASR's LFR / CMVN, and ffmpeg's load pipeline (written from `load.md`). `test_reference.py` validates the ports against the originals. |
| Accuracy | C output vs. the references over 16 signals (silence, tones, chirp, noise, clipping, DC offset, speech, short and > 30 s inputs), 80/128 mels, `pad_or_trim` on/off. |
| Loading | `asrfront.load` vs. whisper's ffmpeg command on files written at test time (every codec × channel counts × rates, damaged and crafted headers). |
| Kernel variants | Exact comparison of `avx2` / `generic` / `scalar` (`ASRFRONT_KERNELS`); CI runs the whole suite again with `scalar`. |
| C unit | FFT vs. naive DFT (27 sizes), API contracts, loader on hand-built buffers including every truncation. |
| Memory and fuzzing | C tests and pytest under ASan/UBSan; libFuzzer on `af_load_memory`, with corpus replay in `ctest` ([`tests/fuzz/README.md`](../tests/fuzz/README.md)). |
| Threads | Identical results from concurrent Python threads. |

- Optional groups: `--group ref` (torch / torchaudio comparisons), `--group load` (ffmpeg 7.0.2
  via `imageio-ffmpeg`); the tests skip without them.
- The test with SenseVoiceSmall's real `am.mvn` runs when `ASRFRONT_SENSEVOICE_MVN` points to it
  (the file is not vendored for license reasons).

## Tried and dropped

- **Twiddle-free first FFT stage**: no measurable gain.
- **NEON transpose for SenseVoice's scaled gather**: slower than the compiler's scalar gather
  on M1 ([`benchmark.md`](benchmark.md)).
- **`mmap` for path input**: 16 kHz WAV 1.2 → 0.4 ms in Python, but a file truncated
  concurrently would crash the interpreter (SIGBUS).
- **Threads inside the C core** for a single input: see [C core](#c-core).

## Open items

- **Streaming**: Whisper's normalization uses the max over the whole spectrogram, so chunked
  processing changes the result; only whole-input processing is supported. A streaming kaldi
  fbank would be straightforward (frames are independent).
- **More models**: presets for other frontends (Paraformer, other Whisper-style models, ...).
- **Qwen3-ASR**: long-audio chunking (`qwen-asr` cuts at low-energy points) and a librosa /
  soxr-compatible loader are not provided ([`qwen3_asr.md`](qwen3_asr.md)).
- **Model-level checks**: end-to-end checks compare features, not decoded text (no model weights
  in the test environment).
- **Platforms**: MSVC and running on Windows / macOS are verified only in CI, not locally.
