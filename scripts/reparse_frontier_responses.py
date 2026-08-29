"""
Rebuild a frontier (sampled) results file from its stored raw responses.

Use after a change to src.lm._extract_choice, the rule that reads a sampled
text as a choice letter: the logprobs are just sample counts, so the file can
be rebuilt exactly instead of paying for a new run. Rows whose raw responses
do not reproduce the stored cell under the old rule are not this run's and
are left alone and listed. Interactive runs cascade (the fed-back prediction
is part of every later prompt of the session), so sessions whose fed-back
prediction changes are listed for scripts/rerun_sessions.py.

Dry run by default. With --write the file is backed up to
<results>.pre_reparse.bak (never overwritten), rewritten in place, and the
cascade sessions are saved to <results>.cascade_sessions.json.

Examples (from the project root):

    python scripts/reparse_frontier_responses.py --mode interactive \
        --model_name gemini-3-flash-preview \
        --raw_paths data/raw_responses/interactive/limited_feedback_yoked_gemini-3-flash-preview_logprobs.json

    python scripts/reparse_frontier_responses.py --mode batch --model_name gpt-5.2 \
        --raw_paths data/raw_responses/full_feedback/no_context_gpt-5.2_logprobs.json \
        --banana_rerun_raw_paths data/raw_responses/full_feedback/no_context_gpt-5.2_logprobs_banana_rerun.json
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import tyro
from pyprojroot import here

from src.banana_rerun import BATCH_KEY_COLS, SESSION_COL, affected_sessions, has_banana
from src.frontier_reparse import batch_raw_by_key, interactive_raw_by_key, reparse

INTERACTIVE_KEY_COLS = ["workerid", "matcher_trialNum"]
DEFAULTS = {
    "batch": ("context_prep/full_feedback/no_context.csv", "no_context"),
    "interactive": ("context_prep/human_history/limited_feedback_yoked.csv", "limited_feedback_yoked"),
}


@dataclass
class Args:
    mode: Literal["batch", "interactive"]
    """batch = no_context (independent trials); interactive = limited_feedback_yoked (cascading sessions)."""

    model_name: str
    """Frontier model short name, e.g. gemini-3-flash-preview."""

    raw_paths: list[Path]
    """Raw-response JSONs of runs over the whole prep file (data/raw_responses/...)."""

    banana_rerun_raw_paths: list[Path] = field(default_factory=list)
    """Raw-response JSONs written by rerun_banana_rows.py, which cover only the banana-affected rows/sessions."""

    results_path: Optional[Path] = None
    """Defaults to data/logprobs/frontier/<condition>_<model_name>_logprobs.csv."""

    prep_path: Optional[Path] = None
    """Defaults to the prep file of the mode."""

    write: bool = False
    """Back up and rewrite the results file; otherwise only report."""

    previous_rule: Literal["march", "v2"] = "march"
    """The extraction rule the results file was produced with: march (last capital letter,
    the March 2026 runs) or v2 (the 27 Aug 2026 reruns). Used only to check provenance."""


def load_raw(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def raw_sources(args: Args, df_prep: pd.DataFrame) -> list:
    """{key: samples} per raw file; banana reruns first because they are newer."""
    key_cols = BATCH_KEY_COLS if args.mode == "batch" else INTERACTIVE_KEY_COLS
    if args.mode == "batch":
        banana_subset = df_prep.loc[has_banana(df_prep)].reset_index(drop=True)
        to_keys = batch_raw_by_key
    else:
        sessions = affected_sessions(df_prep)
        banana_subset = df_prep.loc[df_prep[SESSION_COL].isin(sessions)].reset_index(drop=True)
        to_keys = interactive_raw_by_key
    sources = [to_keys(load_raw(p), banana_subset, key_cols) for p in args.banana_rerun_raw_paths]
    sources += [to_keys(load_raw(p), df_prep, key_cols) for p in args.raw_paths]
    return sources


def main(args: Args) -> None:
    default_prep, condition = DEFAULTS[args.mode]
    prep_path = args.prep_path or here(default_prep)
    results_path = args.results_path or here(
        f"data/logprobs/frontier/{condition}_{args.model_name}_logprobs.csv"
    )
    for path in [prep_path, results_path]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    df_prep = pd.read_csv(prep_path)
    df_results = pd.read_csv(results_path)
    key_cols = BATCH_KEY_COLS if args.mode == "batch" else INTERACTIVE_KEY_COLS
    session_col = SESSION_COL if args.mode == "interactive" else None

    df_new, report = reparse(df_results, raw_sources(args, df_prep), key_cols, session_col, previous_rule=args.previous_rule)
    print(f"results: {results_path}\n{report.summary()}")
    if report.unsourced:
        if args.mode == "interactive":
            sessions = sorted({str(key[0]) for key in report.unsourced})
            print(f"unsourced rows belong to sessions {sessions}: rerun them or supply their raw responses")
        else:
            print(f"unsourced rows: {sorted(report.unsourced)}")

    if not args.write:
        print("\nDry run; re-run with --write to rewrite the results file.")
        return
    backup_path = Path(f"{results_path}.pre_reparse.bak")
    if backup_path.exists():
        raise FileExistsError(f"{backup_path} exists; move it away before writing again")
    shutil.copy2(results_path, backup_path)
    df_new.to_csv(results_path, index=False)
    cascade_path = Path(f"{results_path}.cascade_sessions.json")
    with open(cascade_path, "w") as f:
        json.dump(sorted(report.cascade_sessions), f)
    print(f"\nbacked up to {backup_path}\nrewrote {results_path}\ncascade sessions -> {cascade_path}")


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
