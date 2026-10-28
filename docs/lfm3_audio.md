# LFM3-Audio frontend: audio log-mel of LFM3-Audio-2B

Audio input features of [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B), and
how `asrfront.lfm3_audio` reproduces them. The reference is the checkpoint's own
`Lfm3AudioFeatureExtractor` (`feature_extraction_lfm3_audio.py`, remote code, revision
`9d49701`).

```python
from asrfront import lfm3_audio

mel = lfm3_audio.log_mel(audio)  # (128, T)
feats, lengths = lfm3_audio.log_mel_batch([a1, a2])  # (2, 128, T), frames per item
mask = np.arange(feats.shape[2]) < lengths[:, None]  # feature_attention_mask
n_tokens = lfm3_audio.num_audio_tokens(lengths)  # <|audio|> placeholders per item

stream = lfm3_audio.LogMelStream()  # the extractor's stream()
frames = stream.push(chunk)  # (128, k)
frames = stream.finish()
```

## What the model computes

`preprocessor_config.json`: `feature_size=128`, `n_fft=400`, `win_length=400`,
`hop_length=160`, `log_floor=-12`, `normalization_offset=4`, `normalization_divisor=4`,
`encoder_stride=8`, `padding_value=0`. The extractor has no other options (unknown keyword
arguments raise).

1. **Right padding.** A clip of `n` samples is zero-padded to `pad_length(n)`: 400 if
   `n < 400`, else `n` rounded up so that `n − 400` is a multiple of 160. No sample is dropped;
   `T = 1 + (pad_length(n) − 400) / 160` frames, `1 + ceil((n − 400) / 160)` (1 for `n ≤ 400`,
   including an empty clip).
2. **Spectrum.** `torch.stft(n_fft=400, hop_length=160, window=torch.hann_window(400),
   center=False)`: frame `t` is samples `[160 t, 160 t + 400)`, no reflection; the power
   `|X|²` of the 201 bins.
3. **Mel.** transformers' `mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney",
   mel_scale="slaney")`: the same filters as Whisper's 128-bin bank (librosa's).
4. **Log.** `log10(mel).clamp(min=-12)`, then `(x + 4) / 4`. Unlike Whisper there is no
   `max − 8` dynamic range: the floor is fixed, values lie in `[−2, 2)` for input in [−1, 1].
5. **Batch.** Every clip is computed on its own; the features (not the audio) are padded with
   `padding_value = 0` to the most frames, and `feature_attention_mask` marks each clip's
   frames. An item's features do not depend on the batch.

Output: `input_features` `(B, 128, T)` (mel first) and `feature_attention_mask` `(B, T)`.

### Audio tokens

The encoder subsamples with three stride-2 convolutions over (mel, time): 8 frames become one
token, and `Lfm3AudioProcessor` inserts `num_encoder_tokens(frames) = ceil(frames / 8)`
`<|audio|>` placeholders per clip (`lfm3_audio.num_audio_tokens`).

### Streaming

`Lfm3AudioFeatureExtractor.stream(chunk, cache, is_final)` returns the frames whose window has
fully arrived and keeps the rest; with `is_final=True` it zero-pads the last window to
`num_frames(total)` frames. Frames are independent of each other, so the concatenated frames
are those of the whole clip. `lfm3_audio.LogMelStream` does the same (`push` / `finish`) and is
bit-identical to `log_mel` of the concatenated audio.

## How asrfront computes it

`af_lfm3_audio_log_mel` (C) uses the Whisper kernels: 8 frames per SIMD block, packed real FFT
of size 400, Slaney mel bank, `log10_floor` at `1e-12` and Whisper's normalization kernel with
a fixed max of −4 (`max(x, −4 − 8) = max(x, −12)`). The frames past the input read zeros from
a small per-lane buffer; the C function takes the frame count, so the same call gives the
padded output (`af_lfm3_audio_out_frames`), the streaming frames
(`af_lfm3_audio_complete_frames`), and batch rows written straight into the padded array
(`ld`).

## Accuracy

Against the float64 port (`tests/python/reference.py`) and a torch port of the extractor
(`lfm3_audio_torch`, float32 like the original) over the 16 test signals: ≤ 1.2e-4 in bins
whose mel energy is within 80 dB (1e-8) of the frame's largest (speech: 3.7e-5), ≤ 0.09
overall. Frame counts and masks are identical.

The larger differences are confined to bins far below the frame's peak. The fixed `−12` floor
(−120 dB) is far below Whisper's `max − 8`, so such bins are not floored: a float32 FFT's
round-off of about 1e-7 of the peak magnitude in every bin shows up in a bin whose energy is
1e-10 of the peak (a pure tone's or a chirp's leakage). torch's float32 STFT deviates from
float64 just as much there (≤ 0.063), so neither is "the" value; matching them needs a float64
FFT.

Speed (30 s of audio, x86-64 AVX2): 1.3 ms per clip, vs. 9.6 ms for the torch port on one
thread.
