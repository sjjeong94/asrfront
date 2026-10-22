# Third-party licenses

asrfront itself is MIT (`../LICENSE`). This directory holds the license texts of third-party
code that is compiled into the wheels or that parts of the repository are derived from. Every
file here is included in the wheels' `.dist-info/licenses/`.

## Compiled into the binary extension (`asrfront/_ext`)

| File | Component | License |
|---|---|---|
| `nanobind.txt` | [nanobind](https://github.com/wjakob/nanobind), linked statically (`NB_STATIC`) | BSD-3-Clause |
| `robin_map.txt` | [tsl::robin_map](https://github.com/Tessil/robin-map), compiled into nanobind | MIT |
| `../csrc/third_party/LICENSE.dr_libs` | dr_flac / dr_mp3 (see `../csrc/third_party/README.md`) | Unlicense or MIT-0 |

## Test reference code, test data and documentation excerpts

Not part of the installed package; in the repository and the sdist.

| File | Upstream | Used in |
|---|---|---|
| `openai-whisper.txt` | [openai/whisper](https://github.com/openai/whisper) | `tests/python/reference.py` (port of `whisper/audio.py`), `tests/data/` (`jfk.wav`, `whisper_mel_filters.npz`), excerpts in `docs/whisper.md` |
| `librosa.txt` | [librosa](https://github.com/librosa/librosa) | `tests/python/reference.py` (`librosa.filters.mel`) |
| `torchaudio.txt` | [pytorch/audio](https://github.com/pytorch/audio) | `tests/python/reference.py` (port of `torchaudio.compliance.kaldi.fbank`) |
| `FunASR.txt` | [modelscope/FunASR](https://github.com/modelscope/FunASR) | `tests/python/reference.py` (`apply_lfr` / `apply_cmvn` / `load_cmvn` of `WavFrontend`), excerpts in `docs/sensevoice.md` |
| `transformers.txt` | [huggingface/transformers](https://github.com/huggingface/transformers) | `tests/python/reference.py` (port of `Gemma4AudioFeatureExtractor` and `mel_filter_bank`, rewritten in float64 numpy) |
