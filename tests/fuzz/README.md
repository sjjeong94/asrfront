# Fuzzing `af_load_memory`

`fuzz_load.c` is a libFuzzer target for the audio loader (every container parser and
decoder). Besides memory errors (ASan) and undefined behavior (UBSan) it checks the output
contract: on success every sample is an int16 value / 32768, on failure the output is NULL and
the error code is one of `AF_ECORRUPT`, `AF_EUNSUPPORTED`, `AF_ENOMEM`, with an error
description (printable ASCII) exactly for the first two. A decoded input is
loaded again with `af_load_memory_ex` limited to its own length (same output) and to one
sample less (`AF_ETOOLONG`).

| File | Purpose |
|---|---|
| `fuzz_load.c` | the target (`LLVMFuzzerTestOneInput`) |
| `fuzz_main.c` | driver without libFuzzer: corpus replay, plus a simple mutator (`-mutate N`) |
| `corpus/` | seed corpus: one small file per container / codec variant |
| `make_corpus.py` | regenerates `corpus/` (`uv run --group load python tests/fuzz/make_corpus.py`) |

## Where it runs

- **every C test run** (`ctest`, incl. the ASan/UBSan CI job): `fuzz_corpus` replays `corpus/`
  through the target's checks, with any compiler
- **CI `Fuzz` workflow** (`.github/workflows/fuzz.yml`): libFuzzer for 5 minutes when the C
  code changes, 1 hour every Monday, 30 minutes by default when started by hand. The corpus it
  grows is cached between runs; crashing inputs are uploaded as the `fuzz-artifacts` artifact

## Running locally

With Clang (libFuzzer):

```bash
cmake -S . -B build/fuzz -G Ninja -DCMAKE_C_COMPILER=clang -DASRFRONT_BUILD_FUZZ=ON
cmake --build build/fuzz --target fuzz_load
mkdir -p /tmp/corpus
./build/fuzz/fuzz_load /tmp/corpus tests/fuzz/corpus -max_total_time=600 \
    -rss_limit_mb=2048 -malloc_limit_mb=1024
```

With GCC (no libFuzzer): build the tests with sanitizers and mutate the seeds:

```bash
cmake -S . -B build/asan -DASRFRONT_BUILD_TESTS=ON \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all"
cmake --build build/asan
ASAN_OPTIONS=max_allocation_size_mb=1024 ./build/asan/fuzz_replay -mutate 2000 tests/fuzz/corpus
```

## When it finds something

Reproduce with `fuzz_load CRASH_FILE` (or `fuzz_replay CRASH_FILE`), fix it, and add the input
to `corpus/` under a descriptive name (e.g. `regress_flac_huge_total.flac`) so that every
`ctest` run checks it from then on.
