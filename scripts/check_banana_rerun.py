"""
Check whether a banana rerun actually replaced the affected rows or sessions.

Compares a results file against its pre-rerun backup (``<results>.bak``) and
the context-prep CSV that defines which trials mention banana. Prints a summary
and exits non-zero if any affected row/session is unchanged (rerun incomplete).

Examples (run from the project root):

    python scripts/check_banana_rerun.py \
        --prep_path context_prep/human_history/limited_feedback_yoked.csv \
        --results_path data/logprobs/frontier/limited_feedback_yoked_gemini-3-flash-preview_logprobs.csv \
        --mode interactive

    python scripts/check_banana_rerun.py \
        --prep_path context_prep/full_feedback/no_context.csv \
        --results_path data/logprobs/frontier/no_context_gemini-3-flash-preview_logprobs.csv \
        --mode batch
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from src.banana_rerun import RerunMode, compare_rerun_files


@dataclass
class Args:
    prep_path: Path
    """Context-prep CSV defining the trials (e.g. context_prep/human_history/limited_feedback_yoked.csv)."""

    results_path: Path
    """Results CSV produced by the rerun."""

    mode: RerunMode
    """'batch' compares rows keyed by (gameId, trialNum, repNum); 'interactive' compares whole sessions keyed by workerid."""

    backup_path: Optional[Path] = None
    """Pre-rerun backup to compare against. Defaults to <results_path>.bak."""


def main(args: Args) -> None:
    backup_path = args.backup_path or Path(f"{args.results_path}.bak")
    report = compare_rerun_files(
        args.prep_path, args.results_path, backup_path, mode=args.mode
    )
    print(f"results: {args.results_path}\nbackup:  {backup_path}")
    print(report.summary())
    if not report.complete:
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
