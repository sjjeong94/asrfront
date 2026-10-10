# Audio loading: ffmpeg behavior spec

`asrfront.load` (M9) reproduces what openai-whisper's `load_audio` gets from ffmpeg:

```
ffmpeg -nostdin -threads 0 -i FILE -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -
```

followed by `int16 / 32768`. This document is the **behavior specification** the C
implementation and the Python reference (`tests/python/reference.py`, section "ffmpeg load
pipeline") are written from. It describes *what* ffmpeg outputs, not how ffmpeg's code is
structured; see the license policy in `plan.md` §7.2.1.

**Reference build**: ffmpeg **7.0.2** static (johnvansickle.com build shipped in
`imageio-ffmpeg==0.6.0`, `uv sync --group load`), measured on x86-64 with **AVX2 + FMA3**
(Intel i5-14400F). Where ffmpeg's output depends on the CPU this is noted; other builds/CPUs can
differ in the last bit of the float path.

**Provenance** of each rule: **E** = established by experiment on probe signals (impulses,
random noise, length sweeps; scripts summarized in [How this was measured](#how-this-was-measured)),
**S** = behavior description of the libswresample / libavfilter source consulted for a gap the
experiments could not explain (no code was copied; everything below was then confirmed or
quantified by experiment).

## Pipeline overview

```
decode ─► per-channel samples in the decoder's sample format
       ─► (sr ≠ 16 kHz) resample every channel           [internal format: see table]
       ─► (channels > 1) downmix to mono                  [same internal format]
       ─► convert to s16 (round to nearest even, saturate)
       ─► / 32768 → float32
```

Note the order: when both are needed, ffmpeg **resamples each channel first and downmixes
afterwards** (E).

### Internal sample format by decoder output

| Decoder output | Typical source | No resampling, mono | Resampling and/or downmix |
|---|---|---|---|
| u8 | 8-bit WAV | `(u − 128) << 8` | **int16 path** |
| s16 | 16-bit WAV / FLAC / raw `.pcm` | identity | int16 mix if 16 kHz; **float path** if resampled |
| s32 | 24/32-bit WAV, 24-bit FLAC | `x >> 16` (arithmetic shift, i.e. floor) | **float path** |
| f32 | float WAV, MP3 (`fltp`) | `lrint(x · 32768)` | **float path** |
| f64 | double WAV | `lrint(x · 32768)` | **double path** |

(E for every cell; 16-bit stereo without resampling uses the integer downmix below, with
resampling the float path.) 24-bit PCM is decoded as s32 with the sample in the top 24 bits
(`s24 << 8`), so `s32 >> 16 = s24 >> 8` (E).

Conversion to float for the float path: s16 → `x / 32768`, s32 → `float(x) · 2⁻³¹` (both in
float32) (E).

Final conversion (all paths): `s16 = clamp(lrint(v · 32768), −32768, 32767)` where `lrint`
rounds half to even (float/double paths), or the int16 path's own rounding (below) (E).

Out-of-range and non-finite values (E) — only reachable with float/double input beyond
±65536 (×32768 ≥ 2³¹), NaN or ±inf:

- **double path**: always ffmpeg's scalar conversion — a 64-bit `lrint` (NaN / beyond ±2⁶³ give
  INT64_MIN), truncated to its low 32 bits, then clamped. So NaN/inf → 0, −10⁵ → +32767.
  Reproduced exactly.
- **float path**: ffmpeg converts most samples with SIMD (NaN or beyond int32 → −32768) but the
  last few samples of each internal block with the scalar rule above, so the result depends on
  the sample's position. asrfront always uses the SIMD rule (−32768). Documented deviation.

## Downmix (`-ac 1`)

### Coefficients (E)

Each input channel gets a raw weight by its position:

| Position | Weight |
|---|---|
| FC | 1 |
| FL, FR, FLC, FRC, TFL, TFR | √½ |
| BL, BR, SL, SR, BC | ½ |
| LFE, and all other positions (TC, wide, surround-direct, …) | 0 |

A single-channel layout (mono, or any one position) is passed through unchanged (weight 1).
The weights are then **normalized by their sum** when the sum exceeds 1 (it always does for two
or more weighted channels), e.g. stereo → ½, ½; 5.1 (FL FR FC LFE BL BR) →
`[√½, √½, 1, 0, ½, ½] / (2 + √2)`. Some unusual layouts (e.g. `BL+BR`, `FL+FC`) are rejected by
ffmpeg altogether; asrfront applies the table to them instead (documented deviation).

### Default channel layouts (E)

When the container gives only a channel count (raw PCM, plain `WAVE_FORMAT_PCM`), ffmpeg uses:

| Channels | Layout |
|---|---|
| 1 | FC |
| 2 | FL FR |
| 3 | FL FR LFE (2.1) |
| 4 | FL FR FC BC (4.0) |
| 5 | FL FR FC BL BR (5.0) |
| 6 | FL FR FC LFE BL BR (5.1) |
| 7 | FL FR FC LFE BC SL SR (6.1) |
| 8 | FL FR FC LFE BL BR SL SR (7.1) |

FLAC defines its own layouts per channel count (3 channels = FL FR FC, 4 = FL FR BL BR, …) and
`WAVE_FORMAT_EXTENSIBLE` carries a channel mask; both take precedence (to be confirmed per
container in M9.2/M9.4 tests).

### Arithmetic (E)

- **int16 path** (s16 at 16 kHz, u8 always): coefficients are quantized to Q15,
  `q = lrint(c · 32768)`, and `out = (Σ xᵢ · qᵢ + 16384) >> 15` (arithmetic shift), stored as
  int16 **without saturation** (wraps around). Stereo becomes `(L + R + 1) >> 1`. The wrap is
  reachable: the rounded Q15 weights sum to 32770 for the 7-channel and 32769 for the 8-channel
  default layout, so full-scale input on all channels turns +32767 into −32767 / −32768.
  asrfront reproduces this (bit-exactness over sanity; it only affects clipped 6.1/7.1 audio).
- **float path**: coefficients as float32, `acc = 0; acc = acc + xᵢ · cᵢ` in channel order, all
  float32 (separate multiply and add, no FMA).
- **double path**: same as float, in float64. Exact for up to 4 channels; with 5–8 channels
  ffmpeg differs from this in ~1 of 10⁴ samples by 1 LSB (values within ~2·10⁻⁵ LSB of a rounding
  tie; neither float32 weights nor FMA explain it). Documented deviation — f64 multichannel input
  is practically nonexistent.

## Resampling

Input rates from 1 kHz to 384 kHz are supported (asrfront limit; filters grow with the rate,
N = 792 taps at 384 kHz).

### Filter design (E)

Output sample *k* sits at input time `t_k = k · sr_in / 16000` (exactly; `t_0 = 0`, no delay).
It is a dot product of the input around `t_k` with a Kaiser-windowed sinc:

- cutoff factor `fc = min(1, 0.97 · 16000 / sr_in)` (0.97 of the lower Nyquist; for upsampling the
  filter is a pure interpolator with `fc = 1`)
- tap count `N = ceil(32 / fc)`, rounded up to even (e.g. 48 k: 100, 44.1 k: 92, 22.05 k: 46,
  32 k: 66, 24 k: 50, 96 k: 198, upsampling: 32)
- with `b = floor(t_k)`, `f = t_k − b`, the taps cover inputs `x[b + r]`, `r = −N/2 + 1 … N/2`,
  at distance `d = f − r`:
  `h(d) = sinc(π · fc · d) · I₀(β · √(1 − (d / (N/2))²))`, `β = 9`, `sinc(0) = 1`
  (the window argument is relative to `N/2`; the window is not divided by `I₀(β)`, which cancels
  in the normalization)
- **normalization: every phase is divided by the tap sum of phase 0** (`f = 0`), not by its own
  sum, so non-zero phases have a DC gain slightly different from 1 (e.g. 24 k phase ½: 0.9999981).

### Phases (E)

Let `sr_in / 16000 = p / q` in lowest terms.

- **q ≤ 1024** (*exact* mode; every common rate: 8 k → q = 1, 11.025 k → 640, 22.05 k → 320,
  44.1 k → 160, 48 k → 1, …): `f` is one of `q` exact fractions `m / q`, and the filter for each
  is computed exactly as above.
- **q > 1024** (*interpolated* mode; non-standard rates such as 16494 or 44056 Hz): 1024 phases
  `m / 1024` plus one extra phase, `f` is located between phases `m = floor(f · 1024)` and `m + 1`,
  and the two filters are interpolated linearly with weight `a = f · 1024 − m`. The extra
  phase 1024 is **phase 0 shifted by one input sample** (its first tap is 0, the others are
  phase 0's taps moved one position to the right), not the formula evaluated at `f = 1`.

### Edges (E, S)

The input is extended on both sides before filtering:

- start: **reflection without repeating the first sample**, `x[−j] = x[j]`
- end: **mirror including the last sample**, `x[n − 1 + j] = x[n − j]` (j ≥ 1)

(S: the end-of-stream flush appends about half a filter length of the most recent input in
reverse order.)

### Arithmetic per path (E)

- **double path**: taps and accumulation in float64 (any summation order gives the same s16).
- **float path**: taps rounded to float32. The dot product uses **8 float32 accumulators**
  (`acc[i mod 8] = fma(h[i], x[i], acc[i mod 8])`, taps zero-padded to a multiple of 8, fused
  multiply-add with a single rounding), reduced as a tree:
  `(a0+a4, a1+a5, a2+a6, a3+a7)` → `(s0+s2, s1+s3)` → `s0+s1`. This is ffmpeg's AVX2/FMA3 kernel;
  plain sequential float32 summation differs in ~0.03 % of samples by 1 LSB. **CPU-dependent**:
  ffmpeg on a CPU without FMA3, or on ARM, uses other kernels and can differ by 1 LSB.
  In interpolated mode the two dot products `v1, v2` are interpolated as `v1 + (v2 − v1) · a`
  — this reproduces ffmpeg to ≤ 1 LSB in ~0.01 % of samples; the exact float order was not
  identified (documented deviation, non-standard rates only).
- **int16 path** (interpolated mode): the two integer accumulators are interpolated,
  `acc = acc1 + floor((acc2 − acc1) · fr / q)` before the shift — ≤ 1 LSB vs. ffmpeg in ~0.1 %
  of samples (non-standard rates with 8-bit input only; documented deviation).
- **int16 path**: taps quantized `q = clamp(lrint(h · 32768), −32768, 32767)` — so the unit tap
  of an upsampling filter's phase 0 is 32767, not 32768, and those outputs come out slightly
  attenuated (e.g. 21504 → 21503) — integer accumulation, `out = (acc + 16384) >> 15`, saturated.

### Output length (E, S)

Exact rule not reproduced. ffmpeg's count depends on how much input its resampler still holds
when the stream ends (S: the end padding is limited by the buffered amount, and the buffering
depends on internal block handling, including an x86-specific reserve), which cannot be
expressed as a function of `n` and the rate alone without mirroring its buffer management.
asrfront uses (E):

| Rate | `n_out` | Agreement with ffmpeg (random `n` in 150 … 200 000) |
|---|---|---|
| `sr_in ≤ 32000` | `ceil(n · 16000 / sr_in)` | exact for 8 k, 11.025 k, 12 k, 24 k, 32 k; 22.05 k: 3 % one sample longer |
| `sr_in > 32000` | `ceil((n − 1) · 16000 / sr_in)` | exact for 48 k, 96 k; 44.1 k: 6 % one sample shorter; 88.2 k: 3 % one sample longer |

When the lengths differ, all common samples are still identical; only the last sample is
missing/extra. Very short inputs behave differently in ffmpeg (e.g. at 48 kHz fewer than 67
input samples give **no output at all**); asrfront applies the formula (deviation, < 2 ms of audio).

## Containers

### WAV (E; checked against ffmpeg in `tests/python/test_load.py`)

- `RIFF`, `RF64` and `BW64` headers (RF64: 64-bit data size from the `ds64` chunk); the RIFF size
  field is ignored. Chunks are word aligned (odd sizes are followed by a pad byte); unknown chunks
  (`LIST`, `fact`, …) before or after `data` are skipped. `data` before `fmt ` is an error.
- Codecs: PCM u8 / s16 / s24 / s32, IEEE float 32 / 64, A-law and µ-law (G.711, decoded to s16),
  each also inside `WAVE_FORMAT_EXTENSIBLE` (sub-format GUID). Other codecs (e.g. MP3-in-WAV,
  ADPCM) and other bit depths are rejected.
- A `data` size of 0, 0xFFFFFFFF, or larger than the file means "until the end of the file"
  (streamed or truncated recordings); a partial last frame is dropped.
- Channel layout: the extensible channel mask when it names exactly as many positions as there
  are channels, otherwise (no mask, mask with too few / too many / unknown positions) the
  default layout for the channel count. Up to 18 channels with a mask, 8 without.

### FLAC (E)

Decoded with the vendored dr_flac (`csrc/third_party`). FLAC is lossless, so the decoded samples
equal ffmpeg's; what matters is the sample format and layout ffmpeg assigns:

- up to 16 bits per sample → **s16** (left-justified), 17–32 bits → **s32** (left-justified)
- channel layout: FLAC's channel assignment (1: FC, 2: FL FR, 3: FL FR FC, 4: FL FR BL BR,
  5: FL FR FC BL BR, 6: FL FR FC LFE BL BR, 7: FL FR FC LFE BC SL SR, 8: FL FR FC LFE BL BR SL SR),
  overridden by a Vorbis comment `WAVEFORMATEXTENSIBLE_CHANNEL_MASK=0x…` — ffmpeg writes it
  whenever the layout differs from FLAC's (e.g. 2.1, 4.0) and applies it when decoding.

### MP3 (E)

Decoded with the vendored dr_mp3 (minimp3-based) in float output mode, so samples are **not
clipped** before resampling — like ffmpeg's `mp3float` decoder, whose output exceeds ±1 on loud
material (dr_mp3's default s16-based output clips and then differs by thousands of LSB after
resampling). Encoder delay and padding from a LAME/Xing/Info header are removed (delay + 529
samples at the start, padding − 529 at the end), as ffmpeg does; without such a header neither
trims. The decoders differ by ≤ 5·10⁻⁶ before conversion, which gives **≤ 1 LSB** in ~1 % of the
output samples; length and alignment are identical. Only MPEG-1/2/2.5 Layer III is detected.

### Raw PCM

Headerless s16le, 16 kHz, mono (`int16 / 32768`); an odd trailing byte is dropped. Selected by
a `.pcm` / `.raw` file extension, or for bytes input without a recognized header (see
`plan.md` §7.3 for the detection rules; MP3 sync detection requires two consecutive valid
Layer III frame headers so that raw PCM is never taken for MP3).

## SenseVoice (FunASR) loading

FunASR does not use ffmpeg for local files: `load_audio_text_image_video` reads with
`torchaudio.load` (fallback soundfile), averages the channels in float32 and resamples with
`torchaudio.transforms.Resample` (defaults). Measured on jfk.wav converted to other rates
(`sensevoice.fbank_raw`, natural-log mel energies, |asrfront.load − FunASR-style|):

| Input | mean | p99 | for comparison: torchaudio vs. librosa/soxr resampling (mean / p99) |
|---|---|---|---|
| 16 kHz mono / stereo | 0 | 0 | 0 / 0 |
| 44.1 kHz stereo, 48 kHz, 22.05 kHz | 0.019–0.020 | 0.32–0.33 | 0.029 / 0.55 |
| 8 kHz (upsampled) | 0.35 | 4.6 | 0.55 / 6.4 |

So at 16 kHz the features are identical; for resampled input they differ less than two common
resamplers differ from each other (the largest differences sit in the top mel bins near the
cutoff, and for 8 kHz input in the empty band above 4 kHz). Not checked: model output (needs
the model weights).

## Other platforms (aarch64, Apple silicon)

asrfront's output does not depend on the CPU: the float resampler uses `fmaf()` (correctly
rounded), every other float operation has a fixed order and no contraction, so x86-64 (avx2,
generic, scalar kernels) and aarch64 give **bit-identical** `load()` results and features.
ffmpeg itself does not. Measured with the aarch64 build of the same ffmpeg 7.0.2
(`imageio-ffmpeg` manylinux2014_aarch64 wheel) under qemu-aarch64 8.2, against asrfront
cross-compiled for aarch64 (zig cc, musl; bit-identical to the x86-64 build):

- **float → s16 rounding**: ffmpeg on aarch64 rounds **half up** (`floor(v + 0.5)`), on x86-64
  half to even. Exact ties differ by 1 LSB — ~0.06 % of the samples of float32 input, rarer
  after resampling. This applies wherever ffmpeg converts through float32, including 16 kHz
  input that is f32, or s32 with more than one channel.
- **float resampling kernel**: NEON instead of AVX2/FMA3, ≤ 1 LSB.
- **int16 path** (s16 at 16 kHz, u8) and **double path** (f64): identical to x86-64.
- **NaN / ±inf / |x| ≥ 65536** float samples: handled differently again (see the out-of-range
  rules above); not reproduced for aarch64.

Results on 12 converted copies of jfk.wav (WAV 7 formats/rates/layouts, FLAC 2, MP3 3):

| | s16 / u8 at 16 kHz, f64 | float path (resampled, f32, s32 multichannel) | MP3 |
|---|---|---|---|
| ffmpeg aarch64 vs. ffmpeg x86-64 (AVX2+FMA3) | identical | ≤ 1 LSB, ~0.02 % of samples | ≤ 1 LSB, ~0.2 % |
| asrfront (any CPU) vs. ffmpeg aarch64 | identical | ≤ 1 LSB, ~0.02 % | ≤ 1 LSB, ~0.08 % |
| asrfront (any CPU) vs. ffmpeg x86-64 | identical | identical | ≤ 1 LSB |

and the full ffmpeg comparison suite (`test_ffmpeg_ref.py` + `test_load.py`, 576 tests) run
against the aarch64 ffmpeg passes with the float path allowed 1 LSB (one CPU-specific
NaN/inf test skipped); int16 and double paths stay exact. Lengths were identical everywhere.
So on ARM (including Apple silicon), asrfront matches whisper's ffmpeg-based loading to within
1 LSB — the amount by which ffmpeg differs between the two architectures.

The tests pick the tolerance with `helpers.ffmpeg_is_reference_build()` (x86-64 with AVX2 and
FMA3 → exact; anything else → 1 LSB on the float path); `ASRFRONT_FFMPEG_EXACT=0/1` overrides
it. On macOS, imageio-ffmpeg 0.6.0 ships ffmpeg **7.1** instead of 7.0.2 (not checked).

On aarch64 `fmaf()` compiles to NEON `fmla` (no libm call). Not measured: speed on real ARM
hardware (qemu only). The Linux aarch64 wheels are built with GCC in CI, where the aarch64 test
job runs the bit-exact C-vs-reference tests natively; macOS is not built or tested in CI.

## Summary of accuracy targets (M9)

| Case | Target |
|---|---|
| 16 kHz, any format, any channel count | bit-exact (f64 with ≥ 5 channels: ≤ 1 LSB) |
| exact-mode resampling (all common rates), float/double/int16 paths | bit-exact samples; length per table above |
| interpolated-mode resampling (non-standard rates) | ≤ 1 LSB |
| float path vs. ffmpeg on non-FMA / ARM CPUs | ≤ 1 LSB (ffmpeg itself differs; aarch64 measured, see above) |
| FLAC | bit-exact (same as WAV) |
| MP3 | ≤ 1 LSB (~1 % of samples), same length and alignment |

## How this was measured

All probes feed raw samples to the reference ffmpeg through a pipe
(`-f <fmt> -ar <sr> -ac <ch> [-ch_layout <layout>] -i -`) and read the whisper-style s16 output,
or f64 output (`-f f64le -acodec pcm_f64le`) to observe the double path without quantization.

- conversions/downmix: 10⁴–10⁵ random samples per format and channel count, compared with
  candidate rounding rules (round half up/even/down, Q15 variants)
- downmix weights: least-squares fit of the f64 output on random multichannel input, for named
  layouts
- filter: impulses at consecutive input positions give every tap of every phase; matched against
  the formula to ~1e-16 for 48 k, 24 k, 44.1 k, 22.05 k, 32 k, 96 k, 8 k, 11.025 k, 12 k, 15.8 k,
  16.2 k, 16.4 k; interpolated mode at 16494 and 44056 Hz
- arithmetic: 6·10⁴–4·10⁵ outputs per rate and input format, mismatch counts for candidate
  summation orders (1/2/4/8/16 lanes, with/without FMA, tree/sequential reduction)
- length: zero input of every length in 0–80 and 10000–10441, plus 300 random lengths up to
  200 000, for 10 rates; repeated with WAV and FLAC containers of different frame sizes
  (256/1152/4096) — the length does not depend on the container's block size

`tests/python/test_ffmpeg_ref.py` re-checks these rules against the real ffmpeg.
