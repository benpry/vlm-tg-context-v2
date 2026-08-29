"""
Validate the Llama 3.2 rerun (scripts/rerun_llama.sh) against the archive.

The archived Llama files hold nucleus-truncated logprobs (letters outside the
model's default top_p = 0.9 stored as -9998.3, see
REPORT_banana_verification.md §6.2). For every archived Llama file the rerun
must have a file with the same trial keys and no masked logprob (rows with a
letter missing from the top-1000 are noted, as they occur for other models too). Per file it prints how often the rerun's argmax agrees
with the archive's and the mean normalised P(target) before/after: P(target)
is expected to change, the argmax mostly not (interactive files less so, since
their feedback histories diverge after the first differing prediction). Exits
non-zero if any file is missing or fails a check.

    python scripts/check_llama_rerun.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

from src.banana_rerun import LETTERS, LOGPROBS_COL, normalised_probs, parse_logprobs, shared_key_cols
from src.utils import MASKED_LOGPROB


@dataclass
class Args:
    rerun_root: Path = Path("/juice2/scr2/benpry/vlm-tg-context-logprobs-llama-rerun")
    """Where rerun_llama.sh wrote the new result files (mirrors data/logprobs)."""

    archive_root: Path = Path(here("data/logprobs"))
    """The archived results."""

    model_short_name: str = "Llama-3.2-11B-Vision-Instruct"


def check_file(archive_path: Path, rerun_path: Path) -> list:
    """Return the problems with one rerun file (empty list = fine) and print its stats."""
    if not rerun_path.exists():
        return ["missing"]
    df_old, df_new = pd.read_csv(archive_path), pd.read_csv(rerun_path)
    problems = []
    key_cols = shared_key_cols(df_old, df_new)
    old_keys = set(df_old[key_cols].itertuples(index=False, name=None))
    new_keys = set(df_new[key_cols].itertuples(index=False, name=None))
    if old_keys != new_keys:
        problems.append(f"trial keys differ ({len(old_keys ^ new_keys)} not shared)")
    new_logprobs = [parse_logprobs(cell) for cell in df_new[LOGPROBS_COL]]
    if any(min(lp.values(), default=0) <= MASKED_LOGPROB for lp in new_logprobs):
        problems.append("masked logprobs (<= -9000) in the rerun")
    if problems:
        return problems
    # A letter outside the top-1000 logprobs is stored as absent (probability 0,
    # with a warning at run time); that happens to other models too and is a
    # note, not a failure.
    short = sum(len(lp) < len(LETTERS) for lp in new_logprobs)
    if short:
        print(f"  note: {short} rows with a letter missing from the top-1000 logprobs")

    merged = df_new[key_cols + [LOGPROBS_COL]].merge(
        df_old[key_cols + [LOGPROBS_COL]], on=key_cols, suffixes=("_new", "_old"), validate="one_to_one"
    )
    p_new = normalised_probs(merged, f"{LOGPROBS_COL}_new")
    p_old = normalised_probs(merged, f"{LOGPROBS_COL}_old")
    target_index = df_new.set_index(key_cols).loc[
        list(merged[key_cols].itertuples(index=False, name=None)), "target"
    ].map(LETTERS.index).to_numpy()
    rows = np.arange(len(merged))
    print(
        f"  argmax agreement {np.mean(p_new.argmax(1) == p_old.argmax(1)):.3f}   "
        f"mean P(target) archive {p_old[rows, target_index].mean():.4f} -> rerun {p_new[rows, target_index].mean():.4f}   "
        f"({len(merged)} trials)"
    )
    return []


def main(args: Args) -> None:
    archive_files = sorted(args.archive_root.glob(f"*/*{args.model_short_name}_logprobs*.csv"))
    if not archive_files:
        raise FileNotFoundError(f"No {args.model_short_name} files under {args.archive_root}")
    failures = 0
    for archive_path in archive_files:
        relative = archive_path.relative_to(args.archive_root)
        print(relative)
        problems = check_file(archive_path, args.rerun_root / relative)
        if problems:
            failures += 1
            print(f"  PROBLEM: {'; '.join(problems)}")
    print(f"\n{len(archive_files)} archived files, {failures} with problems")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
