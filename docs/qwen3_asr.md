# Qwen3-ASR frontend: log-mel spectrogram without pad / trim

Input features of [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), and how
`asrfront.qwen3_asr` reproduces them. The reference is the
[`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) package, which pins `transformers==4.57.6`.

```python
from asrfront import qwen3_asr

mel = qwen3_asr.log_mel(audio)  # (128, len(audio) // 160)
feats, lengths = qwen3_asr.log_mel_batch([a1, a2])  # (2, 128, T), valid frames per item
feats, lengths = qwen3_asr.log_mel_batch([a1, a2], hf_padding=True)  # transformers batches
n_tokens = qwen3_asr.num_audio_tokens(lengths)  # audio placeholders in the prompt
```

## What the model computes

Both checkpoints ship the same `preprocessor_config.json`: transformers'
`WhisperFeatureExtractor` with `feature_size=128`, `n_fft=400`, `hop_length=160`,
`dither=0`, slaney mel filters from 0 to 8 kHz (the filters of Whisper large-v3).
`Qwen3ASRProcessor` calls it with `padding=True` (pad to the longest item of the batch),
`truncation=False` and `return_attention_mask=True`.

Per input, that is the computation of [`whisper.md`](whisper.md) **without** the pad / trim
to 30 s:

1. centered STFT: reflect padding of 200 samples on each side, 400-sample frames every 160
   samples, periodic Hann window, `|X|²` of the 201 bins; the last frame is dropped, which
   leaves `len // 160` frames;
2. 128 slaney mel filters, `log10(max(mel, 1e-10))`;
3. `max(x, max(x) − 8)` with the maximum over the whole input, then `(x + 4) / 4`.

With torch installed (as `qwen-asr` requires) transformers computes this with `torch.stft` in
float32 (`_torch_extract_fbank_features`).

### Batches

`padding=True` zero-pads every waveform to the longest length `L` of the batch **before** the
STFT, and every item gets `L // 160` frames:

- the last frames of a shorter item see zeros where the item alone would see its own
  reflection (up to 2 frames differ), and its maximum is taken over the padded spectrogram
  (padding frames are silence, so it does not change);
- the attention mask, `mask[::160]` trimmed to `L // 160` entries, marks
  `min(ceil(len / 160), L // 160)` frames as valid: one more than the item alone
  (`len // 160`) when `len % 160 != 0` and the item is not the longest.

So with the transformers backend an item's features depend on the batch it is in. vLLM
processes every request on its own, which is the single-item computation above; it also
accepts precomputed features (`input_audio_features` and `audio_feature_lengths`).

### Audio tokens

The audio encoder splits the features into windows of 100 frames; three stride-2
convolutions turn a full window into 13 tokens, and a last partial window of `r` frames into
`ceil(ceil(ceil(r / 2) / 2) / 2)`. The processor puts that many audio placeholder tokens in
the prompt (`_get_feat_extract_output_lengths`); `qwen3_asr.num_audio_tokens` computes the
same count.

## asrfront

| | `qwen3_asr` |
|---|---|
| `log_mel(audio)` | `(128, len // 160)`: the single-item features (= transformers with one input, = vLLM) |
| `log_mel_batch(audios)` | `(B, 128, max_i len_i // 160)` and `lengths = len_i // 160`; item `i` is `log_mel(audios[i])`, zeros after it |
| `log_mel_batch(audios, hf_padding=True)` | `(B, 128, L // 160)` and `lengths = min(ceil(len_i / 160), L // 160)`: transformers' batch, every frame filled |
| `log_mel_batch(audios, out=buf)` | either form, written into a caller's `(B, 128, T)` float32 buffer (e.g. reused across calls) and returned |
| `num_audio_tokens(n_frames)` | tokens per item, for ints or integer arrays |

`log_mel` runs the Whisper frontend (`n_mels=128, pad_or_trim=False`; the same native handle
as `whisper.log_mel(audio, n_mels=128, pad_or_trim=False)`, with identical output). Batches
with `hf_padding=True` use `af_whisper_log_mel_padded`, the same computation over the input
followed by virtual zeros up to `L` samples: blocks of 8 frames that read only padding are
filled with the known silence value, as for Whisper's `pad_or_trim`, and blocks whose frames
reflect back into the input are computed.

### Decisions

- **Single items by default.** `log_mel_batch` computes every item on its own unless
  `hf_padding=True`: features that do not depend on which other inputs share the batch, the
  same as `log_mel`, batch size 1 and the vLLM backend. The transformers backend's batches
  are available, exactly, for callers that must match it.
- **No new kernel.** The features are Whisper large-v3's (128 slaney mels), already
  implemented and tested; the frontend reuses the Whisper handle, and the C API only gains
  the virtual zero padding of `hf_padding` batches.
- **`num_audio_tokens`** is written from the rule above (three ceiling halvings per partial
  window); the tests compare it with `_get_feat_extract_output_lengths` as published.
- **Out of scope**: the long-audio chunking of `qwen-asr` (a pipeline step, not part of the
  features) and its librosa / soxr loader (see [Loading audio](#loading-audio)).

## Edge cases

| Input | transformers / qwen-asr | `asrfront.qwen3_asr` |
|---|---|---|
| ≤ 200 samples alone | torch reflect-pad error | `ValueError` |
| ≤ 200 samples in a batch | padded to the longest item | default: `ValueError`; `hf_padding=True`: computed (needs a longest item > 200) |
| empty item in a batch | 0 valid frames | same with `hf_padding=True` |
| any length | no limit (`qwen-asr` cuts inputs into ≤ 1200 s chunks at low-energy points; ≤ 180 s with timestamps) | no limit; chunking is left to the caller |
| NaN / inf samples | propagate | `ValueError` |
| int16 PCM | caller converts | divided by 32768 automatically |

## Loading audio

`qwen-asr` loads files with librosa / soundfile (`sr=None`), averages the channels,
resamples with `librosa.resample` (soxr) and divides by the peak when it exceeds 1.
`asrfront.load` follows whisper's ffmpeg command instead ([`load.md`](load.md)): identical
for 16 kHz mono PCM, and for other inputs the waveforms differ about as much as two good
resamplers do (and in the channel weights of multichannel downmixes). A soxr-compatible
loader is not planned: libsoxr is LGPL, so it would need the clean-room process of
[`design.md`](design.md#license-policy-clean-room-style).

## Accuracy

Against the float64 reference (`tests/python/reference.py`; the Whisper reference without
pad / trim, zero-padded for `hf_padding` batches) and against transformers 4.57.6
(`--group qwen`):

| | max abs error | target |
|---|---|---|
| single items, 16 signals vs. float64 | 4.0e-5 | 1e-4 |
| `hf_padding` batches vs. float64 (noise, 8 batches around the boundaries) | 3.6e-6 | 1e-4 |
| single items vs. transformers (float32 torch), 16 signals | 6.7e-5 | 1e-4 |
| `hf_padding` batches vs. transformers | 1.8e-5 | 1e-4 |
| `lengths` vs. transformers' attention mask | exact | |
| `num_audio_tokens` vs. `_get_feat_extract_output_lengths` (0 … 19999 frames) | exact | |

transformers' slaney filters (built in float64, used in float32) are identical to asrfront's
in 99.7 % of the values and within 1 ulp in the rest.

## Performance

Intel i5-14400F (AVX2), Python API, best of N runs:

| Input | `qwen3_asr.log_mel` | Whisper 80 mels, same length |
|---|---|---|
| 5 s | 0.22 ms | 0.16 ms |
| 30 s | 1.19 ms | 0.99 ms |
| 1200 s (one `qwen-asr` chunk) | 80 ms | 63 ms |

Batch of 32 × 30 s: 46 ms on 1 thread, 12 ms on 8 threads.

At 8 threads, a batch is limited less by the features than by memory: the same 48 × 30 s
written into reused buffers by the workers takes 14 ms, `log_mel_batch` 23 ms. Part of the
difference is the fresh 74 MB output array, touched for the first time (NumPy already maps it
with huge pages): with `out=` a reused buffer, `log_mel_batch` takes 20 ms. The float32
conversion and NaN / inf check run in the workers ([`design.md`](design.md#python-binding)):
done serially before the batch, the same batch took 25 ms, and 61 ms instead of 32 ms for
float64 input, which has to be converted.

A 1200 s input takes more than 40 × the 30 s time. Written into a reused `out=` buffer
(`_ext.WhisperFrontend.log_mel`) it takes 58 ms, so ~22 ms go to the fresh 61 MB output
array; the remaining ~15 % over 40 × 1.25 ms is likely the normalization pass reading the
whole spectrogram back from memory, as it no longer fits in the cache.

Whisper's normalization uses the maximum over the whole input, so the features cannot be
computed incrementally (streaming).
