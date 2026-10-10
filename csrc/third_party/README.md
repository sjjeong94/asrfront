# Third-party code

| File | Source | Version | License |
|---|---|---|---|
| `dr_flac.h` | [mackron/dr_libs](https://github.com/mackron/dr_libs) @ `dfe8377631000664666519fdb83da193fd8037f4` (2026-08-31) | v0.13.4 (unreleased: post-0.13.3 bounds-check and seeking fixes) | Public domain (Unlicense) or MIT-0, see `LICENSE.dr_libs` |
| `dr_mp3.h` | same commit | v0.7.4 (unreleased: post-0.7.3 Xing/Info tag overflow fixes) | same |

The commit is newer than the last tagged releases on purpose: it contains fixes for
heap-buffer-overflow / overflow bugs in metadata and Xing/Info tag parsing, which matter because
asrfront parses untrusted files.

`dr_mp3.h` is unmodified (sha256 `997b7ee1…a2f`). `dr_flac.h` (upstream sha256 `111144e7…945c`)
has one local patch: `drflac__be2host_32_ptr_unaligned` / `drflac__le2host_32_ptr_unaligned` cast
each byte to `drflac_uint32` before shifting (upstream shifts a promoted `int`, which is undefined
behavior for bytes ≥ 0x80 << 24; found by UBSan fuzzing of Vorbis comment lengths). The
library is compiled with `-fwrapv` (GCC/Clang) because corrupt FLAC residuals overflow its
int32 sample arithmetic; with `-fwrapv` this wraps (garbage audio) instead of being undefined.
It is also compiled with `-ffp-contract=off` like the rest of the C code, so the MP3 decoder's
float output does not depend on the compiler (Clang would otherwise fuse multiply-adds). They are compiled once in
`dr_libs.c` with `DR_FLAC_NO_STDIO`, `DR_FLAC_NO_OGG` and `DR_MP3_NO_STDIO` (memory decoding
only). To update: download both headers from a newer commit, rerun the test suite
(`uv run --group load pytest`) and update this table.
