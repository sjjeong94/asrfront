"""Repeat the frontend benchmarks without audio decoding or multi-threaded references.

.venv/bin/python bench/current_environment.py --json results.json
"""

import argparse
import json
from pathlib import Path

import compare as benchmark
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument(
        "--batch-only", action="store_true", help="only compare batch configurations"
    )
    args = parser.parse_args()
    signals = benchmark.signals()
    audio = np.ascontiguousarray(signals["speech_35s"][: 30 * benchmark.SR])
    runs = []
    for _ in range(3):
        benchmark.RESULTS = {}
        if not args.batch_only:
            benchmark.bench_single(audio)
            benchmark.bench_lengths(np.asarray(signals["speech"]))
        benchmark.bench_batch(audio, 32)
        runs.append(benchmark.RESULTS)
        args.json.write_text(json.dumps({"env": benchmark.environment(), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
