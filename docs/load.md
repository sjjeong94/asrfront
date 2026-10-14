# Audio loading: ffmpeg behavior spec

`asrfront.load` (M9) reproduces what openai-whisper's `load_audio` gets from ffmpeg:

```
ffmpeg -nostdin -threads 0 -i FILE -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -
```

followed by `int16 / 32768`. This document is the **behavior specification** the C
implementation and the Python reference (`tests/python/reference.py`, section "ffmpeg load
pipeline") are written from. It describes *what* ffmpeg outputs, not how ffmpeg's code is
structured; see the license policy in [`design.md`](design.md#license-policy-clean-room-style).

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
| u8 | 8-bit PCM (WAV unsigned; AIFF / CAF / SPHERE signed, offset by 128) | `(u − 128) << 8` | **int16 path** |
| s16 | 16-bit PCM, FLAC, raw `.pcm`, A-law / µ-law, ADPCM | identity | int16 mix if 16 kHz; **float path** if resampled |
| s32 | 24/32-bit PCM, 24-bit FLAC | `x >> 16` (arithmetic shift, i.e. floor) | **float path** |
| f32 | float PCM, MP3 (`fltp`) | `lrint(x · 32768)` | **float path** |
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
  each also inside `WAVE_FORMAT_EXTENSIBLE` (sub-format GUID), and IMA / Microsoft ADPCM
  (see [ADPCM](#adpcm-e)). Other codecs (e.g. MP3-in-WAV, GSM) and other bit depths are
  rejected.
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

### Wave64 (E)

Sony Wave64 is WAV with 16-byte GUID chunk ids and 64-bit little-endian chunk sizes that
include the 24-byte chunk header; chunks are 8-byte aligned. The `fmt ` contents, codecs and
channel layout rules are those of WAV. A `data` size of 24 or less (or negative) is an error;
one beyond the file means "until the end of the file".

### AIFF / AIFF-C (E)

- `FORM` + `AIFF` / `AIFC`; big-endian, word-aligned chunks in any order (`COMM` may follow
  `SSND`). Missing `COMM` or `SSND` is an error.
- Sample rate: the 80-bit extended `COMM` value **rounded to the nearest integer, halves up**
  (22050.5 → 22051).
- The samples are the whole `SSND` chunk after its `offset` field; the `COMM` frame count is not
  used. An `SSND` size below 8 (including 0) is an error, 0xFFFFFFFF or a size beyond the file
  reads to the end of the file; a partial last frame is dropped.
- Codec from the AIFF-C compression type (bit depth from `COMM`):

  | Compression type | Codec |
  |---|---|
  | none (AIFF), `NONE`, `none`, `twos` | signed big-endian PCM, the bit depth rounded up to 8/16/24/32 (0–8 bits → 8-bit); above 32 unsupported |
  | `sowt` | s16le (whatever the bit depth) |
  | `raw ` | unsigned 8-bit |
  | `in24` / `in32` | s24be / s32be |
  | `fl32` `FL32` / `fl64` `FL64` | big-endian float32 / float64 |
  | `alaw` `ALAW` / `ulaw` `ULAW` | G.711 |
  | `ima4` | IMA4 ADPCM (below) |
  | anything else (`QDM2`, `MAC3`, …) | unsupported |

- Channel layout: the `CHAN` chunk (CoreAudio layout, below), else the default layout.

### CAF (E)

- `caff` version 1, and `desc` must be the first chunk (else an error). The float64 sample
  rate is **truncated** to an integer (16000.75 → 16000).
- `lpcm`: the sample size comes from the bits per channel alone, rounded up to 8/16/24/32
  (bytes per packet is not used, but 0 is an error, as is 0 frames per packet). Format flag
  bit 0 selects float (32 or 64 bits only), bit 1 little-endian. **8-bit `lpcm` is always
  signed.** `alaw` / `ulaw`: G.711. `ima4`: IMA4 ADPCM (34 bytes per channel per packet, 64
  frames). Other formats (`aac `, `alac`, …) are unsupported.
- `data`: a 4-byte edit count, then the samples. A size of −1 (streamed), one that leaves no
  samples (≤ 4) or one beyond the file reads to the end of the file — trailing chunks are then
  decoded as samples, as ffmpeg does. A later `data` chunk replaces an earlier one. (Sizes 1–3
  make ffmpeg misalign the data; asrfront reads to the end of the file. Damaged files only.)
- Channel layout: the `chan` chunk (CoreAudio layout, below), else the default layout.

### CoreAudio channel layouts (E)

AIFF `CHAN` and CAF `chan` hold an `AudioChannelLayout`: a tag, a channel bitmap and channel
descriptions. A chunk shorter than 12 bytes makes ffmpeg reject the file (so does asrfront).

- **bitmap** (tag `0x10000`): a WAVE channel mask, with WAV's rules.
- **descriptions** (tag 0): one label per channel — L, R, C, LFE → FL, FR, FC, LFE; Ls, Rs →
  SL, SR; Rls, Rrs → BL, BR; Lc, Rc → FLC, FRC; Cs → BC; the top labels → TC, TFL, TFC, TFR,
  TBL, TBC, TBR. Other labels (wide, LFE2, surround direct, Lt/Rt, ambisonic, discrete) get
  weight 0. Fewer descriptions than channels: default layout.
- **predefined tags**: the tag's low 16 bits must equal the channel count, else the default
  layout. ffmpeg downmixes only tags whose channels are in its native order; those whose
  weights differ from the default layout are Quadraphonic (FL FR SL SR), Cube (FL FR BL BR TFL
  TFR TBL TBR), MPEG_3_0_A (FL FR FC), SMPTE_DTV (5.1 + Lt Rt), ITU_2_1 (FL FR BC), ITU_2_2
  (FL FR SL SR), DVD_5 / 6 / 10 / 11 and AC3_1_0_1 (FC LFE). Unknown tags and the stereo
  variants (headphones, matrix, M/S, XY, binaural) use the default layout.
- ffmpeg **fails** on layouts outside its native channel order ("Input channel layout … is not
  supported"), e.g. MPEG_7_1_C — which its own muxers write for 7.1 — or descriptions such as
  C L R. asrfront uses the default layout for such tags and the label positions for such
  descriptions (documented deviation; ffmpeg gives no output to compare with).

### NIST SPHERE (E)

- `NIST_1A`, the header size on the second line, then `key -type value` lines up to
  `end_head`. Keys are matched by prefix and the last occurrence wins; the value is the third
  token, read as a leading integer (`-r 8000.5` → 8000). `channel_count` and `sample_rate`
  are required.
- The samples start at the header size and run to the end of the file (`sample_count` is not
  used). A header size beyond the file is an error (ffmpeg then starts after `end_head`).
- `sample_coding` (case-insensitive, default `pcm`): `pcm` with `sample_n_bytes` 1 (signed
  8-bit) or 2 (`sample_byte_format` `10`: big-endian; `01`, `1` or absent: little-endian);
  3 or 4 bytes are not implemented in ffmpeg and unsupported here. `ulaw` / `mu-law` and
  `alaw`: one byte per sample whatever `sample_n_bytes` says. Shorten-compressed files
  (`pcm,embedded-shorten-v2.00`) are unsupported, in ffmpeg too.
- Layout: the default layout for the channel count.

### ADPCM (E)

All three decode to **s16**: the int16 path at 16 kHz, the float path when resampled.

- **IMA ADPCM** (WAV / Wave64 tag `0x0011`, 4 bits): each block holds per channel a 4-byte
  header (s16 predictor = the first sample, step index ≤ 88, a zero byte — anything else makes
  ffmpeg fail, and asrfront report a malformed file), then 4-byte groups per channel of 8
  samples, low nibble first. Each nibble *n*: difference `((2·(n & 7) + 1) · step) >> 3`,
  subtracted when `n & 8`, predictor clipped to s16; step index += −1, −1, −1, −1, 2, 4, 6, 8
  for `n & 7`, clamped to 0…88. The block size must be a whole number of 4-byte groups per
  channel. A partial last block decodes the whole groups it holds; the `fact` sample count is
  not used (the padding of the last block is kept). 2/3/5-bit variants are unsupported. With
  more than two channels ffmpeg cannot decode the stream at all (it leaves the channel layout
  unset); asrfront decodes it with the default layout (documented deviation).
- **Microsoft ADPCM** (tag `0x0002`, one or two channels): the block header holds per channel
  the predictor index (≤ 6, else an error), delta, sample 1 and sample 2; the output starts with
  sample 2, sample 1, then one sample per nibble (high nibble first; stereo alternates
  channels). Prediction `(s1 · c1 + s2 · c2) / 256` **rounded toward zero**, plus `n · delta`
  (`n` signed 4-bit), clipped to s16; then `delta = adapt[n] · delta / 256`, at least 16 and at
  most `INT_MAX / 768`. The coefficients are always the seven standard pairs: ffmpeg ignores
  the ones in the `fmt ` extension. A partial last block decodes its whole bytes.
- **IMA4** (QuickTime; AIFF-C / CAF `ima4`): per channel 34-byte packets of 64 samples — a
  2-byte header (predictor = its top 9 bits, step index = its low 7 bits, ≤ 88), then 32 bytes,
  low nibble first. The difference is the reference shift-and-add (`step >> 3`, plus `step`,
  `step >> 1`, `step >> 2` for nibble bits 2, 1, 0) — not the IMA WAV product, which differs in
  rounding. The decoder state **carries over between packets** unless the header's step index
  differs or its predictor is more than 0x7F away from the running predictor. A partial packet
  is dropped.

Not supported, by design: big-endian `RIFX` WAV — ffmpeg 7.0.2 reads its header big-endian but
decodes the samples as little-endian (so a real RIFX file comes out as noise), and matching that
would be useless; GSM 6.10 / G.722 / other WAV codecs.

### Raw PCM

Headerless s16le, 16 kHz, mono (`int16 / 32768`); an odd trailing byte is dropped. Selected by
a `.pcm` / `.raw` file extension, or for bytes input without a recognized header (see
[`design.md`](design.md#format-detection) for the detection rules; MP3 sync detection
requires two consecutive valid Layer III frame headers so that raw PCM is never taken for
MP3).

## Resource use

`load` decodes the whole file before downmixing and resampling, so its peak memory is about
the decoded size (all channels, at the input rate, 2–8 bytes per sample) plus the 16 kHz mono
float32 output. That is proportional to the audio's duration, not to the file size:

| Input | Peak memory per byte of file (measured worst cases) |
|---|---|
| PCM / float / G.711 | ≤ ~70x (8-bit at 1 kHz upsampled to 16 kHz: 64 bytes of output per input byte) |
| IMA / MS / IMA4 ADPCM | ≤ ~130x (4 bits per sample, same worst case) |
| MP3 | ≤ ~100x (8 kbps at 8 kHz) |
| FLAC | unbounded in practice: silence or constant signals compress to almost nothing — 23 kB of 8-channel 384 kHz s32 silence (30 s) peaks at ~360 MB, ~16000x |

Header fields never size an allocation beyond what the file can hold (the FLAC total frame
count is only a hint, capped by the input size; tested by `tests/fuzz`), so the limit above is
the only amplification. ffmpeg has the same property. Callers that load untrusted uploads
should bound what they accept, e.g. the upload size (with FLAC in mind), or the duration read
from the header before calling `load`. There is deliberately no limit option in `load` itself.

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
it. On macOS, imageio-ffmpeg 0.6.0 ships ffmpeg **7.1** instead of 7.0.2; the CI job
`ffmpeg-macos` runs the comparison suite against it on Apple silicon.

On aarch64 `fmaf()` compiles to NEON `fmla` (no libm call). Not measured: speed on real ARM
hardware (qemu only). The Linux aarch64 wheels are built with GCC and the macOS wheels with
Apple clang in CI, where the aarch64 / Apple silicon test jobs run the bit-exact
C-vs-reference tests natively.

## Summary of accuracy targets (M9)

| Case | Target |
|---|---|
| 16 kHz, any format, any channel count | bit-exact (f64 with ≥ 5 channels: ≤ 1 LSB) |
| exact-mode resampling (all common rates), float/double/int16 paths | bit-exact samples; length per table above |
| interpolated-mode resampling (non-standard rates) | ≤ 1 LSB |
| float path vs. ffmpeg on non-FMA / ARM CPUs | ≤ 1 LSB (ffmpeg itself differs; aarch64 measured, see above) |
| FLAC | bit-exact (same as WAV) |
| Wave64, AIFF / AIFF-C, CAF, NIST SPHERE; IMA / MS / IMA4 ADPCM | bit-exact (same as WAV) |
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

- containers and ADPCM (AIFF, CAF, Wave64, SPHERE): hand-built files varying one header
  field at a time (compression types, bit depths, chunk sizes and order, rates, layout tags,
  channel labels, SPHERE keys), decoded by ffmpeg to s16 at the native rate; ADPCM blocks with
  chosen headers and nibbles, compared with candidate decoders (product vs. shift-and-add
  difference, floor vs. truncating prediction, state reset vs. carry-over between packets);
  downmix weights of layouts by least squares as above

`tests/python/test_ffmpeg_ref.py` re-checks these rules against the real ffmpeg, and
`tests/python/test_load.py` / `test_load_formats.py` the decoders.
