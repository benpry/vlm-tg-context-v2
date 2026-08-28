"""
Fingerprint check: is a newer copy of a results file a post-banana-fix rerun?

Compares the softmax-normalised choice probabilities of two versions of the
same results file, trial by trial. A rerun with the fixed text should shift the
banana-affected trials (message or in-context history mentions "banana") and
leave the others unchanged up to run-to-run noise. For interactive files the
comparison groups by session, because the model's own feedback history carries
a changed answer forward to later trials.

Typical use, with the stale copy downloaded from OSF node zk8gq and the fixed
copy from the project that holds the 16 Mar 2026 re-upload:

    python scripts/compare_result_versions.py \
        --old_path stale/full_feedback/yoked_Qwen3-VL-32B-Instruct_logprobs.csv \
        --new_path data/logprobs/full_feedback/yoked_Qwen3-VL-32B-Instruct_logprobs.csv

Exits non-zero if the two files are identical (i.e. the newer one is not a rerun).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from src.banana_rerun import compare_result_versions_files


@dataclass
class Args:
    old_path: Path
    """The older (corrupted-era) copy of the results file."""

    new_path: Path
    """The newer copy that should be a post-fix rerun."""


def main(args: Args) -> None:
    report = compare_result_versions_files(args.old_path, args.new_path)
    print(f"old: {args.old_path}\nnew: {args.new_path}")
    print(report.summary())
    if report.verdict.startswith("IDENTICAL"):
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
