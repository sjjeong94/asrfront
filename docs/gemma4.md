# Gemma 4 frontend: audio log-mel of the E2B / E4B models

Audio input features of [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) and
[gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), and how `asrfront.gemma4`
reproduces them. The reference is transformers' `Gemma4AudioFeatureExtractor` (checked
against transformers 5.20.2).

```python
from asrfront import gemma4

mel = gemma4.log_mel(audio)  # (T, 128)
feats, lengths = gemma4.log_mel_batch([a1, a2])  # (2, T, 128), valid frames per item
mask = np.arange(feats.shape[1]) < lengths[:, None]  # input_features_mask
n_tokens = gemma4.num_audio_tokens(lengths)  # audio soft tokens per item
```

## What the model computes

Both checkpoints ship the same `feature_extractor` in `processor_config.json`:
`feature_size=128`, `frame_length=320` (20 ms), `hop_length=160` (10 ms), `fft_length=512`,
`min_frequency=0`, `max_frequency=8000`, `mel_floor=0.001`, `preemphasis=0`, `dither=0`,
`input_scale_factor=1`, no per-bin normalization. `Gemma4Processor` calls it with the
defaults: `padding="longest"`, `max_length=480000` with `truncation=True`,
`pad_to_multiple_of=128`.

1. **Batch padding.** Every waveform (cast to float32) is truncated to 480 000 samples
   (30 s), then all are zero-padded to the longest length rounded up to a multiple of 128
   samples, `L`. A sample mask marks the real samples.
2. **Semicausal framing.** 160 zeros are prepended. Windows of 321 samples every 160
   samples are cut from the result; the first 320 samples of each are the frame (with
   preemphasis 0 the 321st is unused). That is `T = (L + 160 − 321) // 160 + 1` frames, and
   frame `t` covers input samples `[160 t − 160, 160 t + 160)`.
3. **Spectrum.** Periodic Hann window of 320 (`np.hanning(321)[:-1]`), real FFT of size 512
   (the frame zero-padded), **magnitude** `|X|` of the 257 bins (not the power).
4. **Mel.** 128 triangular filters on the HTK mel scale `2595 log10(1 + f / 700)` from 0 to
   8 kHz, edges evenly spaced in mel, FFT bins at `linspace(0, 8000, 257)`, no normalization
   (transformers' `mel_filter_bank(norm=None, mel_scale="htk")`). The lowest filter lies
   between bins 0 and 1 and is all zero (transformers warns about it), so mel bin 0 is
   always `log(0.001)`.
5. **Log.** `log(mel + 0.001)` (natural log, additive floor).
6. **Mask.** Frame `t` is valid if sample `160 t + 320` of the padded signal (the unused
   321st of its window, input sample `160 t + 160`) is real: `(n − 1) // 160` valid frames for
   an input of `n` samples. The other frames are multiplied by zero. Valid frames read only
   real samples (and the leading zeros), so they do not depend on the batch.

Output: `input_features` `(B, T, 128)` (time first) and `input_features_mask` `(B, T)`.

### Audio tokens

The audio encoder subsamples the frames with two stride-2 convolutions (kernel 3, padding
1). `Gemma4Processor.replace_audio_token` takes the mask's every other entry twice and
inserts that many audio placeholders, `ceil(ceil(v / 2) / 2)` for `v` valid frames;
`_compute_audio_num_tokens` (used with vLLM) gets the same number from the sample count and
caps it at `audio_seq_length = 750`, which 30 s reaches exactly (2999 frames → 750 tokens).
`gemma4.num_audio_tokens` implements the former (no cap: with `truncation=False` longer
inputs give more).

## How asrfront computes it

`af_gemma4_log_mel` (C) computes only the valid frames, 8 at a time in SIMD lanes with the
kernels of the other frontends (packed real FFT of size 512, as for kaldi fbank), plus two
small kernels: `sqrt_f32` for the magnitude and `log_add` for `log(x + 0.001)`. The mask is
not computed per frame: the zero frames are written after the valid ones. Frame 0 is the only
one that reaches the leading zeros; it is assembled in a small buffer. The C API leaves the
truncation and the 128-sample padding to the caller (Python), which only decide `T`.

The mel filters are built in double and stored in float32 (transformers keeps them in
float64 and multiplies in float64). Everything else is float32.

## Accuracy

Against the float64 port of the extractor in `tests/python/reference.py`, over the 16 test
signals: ≤ 1.6e-4 in bins whose mel energy is ≥ 0.01, ≤ 6.4e-3 overall (speech: 1.1e-3).
Against transformers the same bounds hold, and the frame counts and masks are identical.

The larger differences are confined to bins far below the floor. A float32 FFT has an
absolute round-off of about 1e-6 of the frame's largest magnitude in **every** bin, so a bin
whose true mel energy is ~1e-6 (a pure tone's leakage, high bins of band-limited speech) can
come out as ~5e-6; `log(mel + 0.001)` turns that into a difference of a few 1e-3 around
`log(0.001) = −6.9`. transformers' numpy FFT is more precise in those bins (it differs from
float64 by ≤ 5e-4, from its float32 windowing). Making asrfront as precise there needs a
float64 FFT, which was not worth its cost for feature values that sit at the floor.

The tests (`tests/python/test_gemma4.py`) check both bounds against the reference, frame
counts and masks for lengths around the 128 / 160 / 161-sample boundaries, batches against
single items (bit-identical valid frames), and, when transformers ≥ 5.5 is installed, against
`Gemma4AudioFeatureExtractor` itself (that version conflicts with the `qwen` group's pin, so
it is not a dependency group).
