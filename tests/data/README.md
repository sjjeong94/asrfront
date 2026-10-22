# Test data

| File | Source | License |
|---|---|---|
| `jfk.wav` | `tests/jfk.flac` from [openai/whisper](https://github.com/openai/whisper) (sha256 `63a4b1e4…`), downmixed to mono and resampled 44.1 kHz → 16 kHz (torchaudio), 16-bit PCM, 11.0 s | MIT (openai/whisper); the recording is a US public-domain speech |
| `whisper_mel_filters.npz` | `whisper/assets/mel_filters.npz` from openai/whisper (sha256 `7450ae70…`); `mel_80` / `mel_128`, float32 `(n_mels, 201)` | MIT |

openai/whisper's copyright notice and license: [`LICENSES/openai-whisper.txt`](../../LICENSES/openai-whisper.txt).
