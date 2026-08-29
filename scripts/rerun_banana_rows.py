"""
Rerun frontier models (Claude, GPT, Gemini) on rows affected by the banana -> ba''a bug.

The bug was in src/utils.py, where literal_eval(x.replace("nan", "''")) corrupted
"banana" to "ba''a" in the prompts. This script reruns only the affected trials
to save cost, then verifies against the pre-rerun backup that every affected
row / session actually changed.

- batch mode (no_context): reruns the individual affected rows (message contains "banana").
- interactive mode (limited_feedback_yoked): reruns every session containing an
  affected row, because model predictions cascade through the session's feedback history.

Results are read from and written to data/logprobs/frontier/, the layout used by
the analysis (01-data_processing.qmd) and by the OSF archive. A missing results
file is an error, never a skip. The first backup (<results>.bak) is kept as the
reference copy of the original run and is never overwritten.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import tyro
from PIL import Image
from pyprojroot import here

from src.banana_rerun import (
    BATCH_KEY_COLS,
    SESSION_COL,
    RerunReport,
    affected_rows,
    affected_sessions,
    compare_rerun,
    has_banana,
)
from src.clients import setup_client
from src.interactive import run_interactive_evaluation
from src.lm import get_logits

__all__ = ["has_banana", "rerun_batch", "rerun_interactive"]

RESULTS_DIR = here("data/logprobs/frontier")
RAW_RESPONSES_DIR = here("data/raw_responses/frontier")
BATCH_PREP_PATH = here("context_prep/full_feedback/no_context.csv")
INTERACTIVE_PREP_PATH = here("context_prep/human_history/limited_feedback_yoked.csv")

RerunMode = Literal["batch", "interactive", "both"]


def results_path_for(model_name: str, prep_path: Path) -> Path:
    """data/logprobs/frontier/<prep stem>_<model short name>_logprobs.csv"""
    short_name = model_name.split("/")[-1]
    path = Path(RESULTS_DIR) / f"{Path(prep_path).stem}_{short_name}_logprobs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existing results at {path}. Download the frontier results from OSF "
            "into data/logprobs/frontier/ before rerunning."
        )
    return path


def backup_results(results_path: Path) -> Path:
    """Back up the results file, keeping an existing .bak as the original reference."""
    backup_path = Path(f"{results_path}.bak")
    if backup_path.exists():
        print(f"  Keeping existing backup {backup_path} (original run) as the reference")
    else:
        shutil.copy2(results_path, backup_path)
        print(f"  Backed up results to {backup_path}")
    return backup_path


def raw_responses_path_for(results_path: Path, n_samples: Optional[int]) -> Optional[str]:
    if not n_samples:
        return None
    os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)
    return str(Path(RAW_RESPONSES_DIR) / f"{results_path.stem}_banana_rerun.json")


def verify_and_save(
    report: RerunReport, df_merged: pd.DataFrame, df_results: pd.DataFrame, results_path: Path
) -> None:
    """Fail loudly if the merge changed the row count or the rerun is incomplete."""
    if len(df_merged) != len(df_results):
        raise RuntimeError(
            f"Row count mismatch after merge: {len(df_merged)} vs {len(df_results)}"
        )
    print(report.summary())
    if not report.complete:
        raise RuntimeError(
            "Rerun incomplete: some affected rows/sessions did not change. "
            "The merged results were NOT saved."
        )
    df_merged.to_csv(results_path, index=False)
    print(f"  Saved updated results to {results_path}")


def rerun_batch(
    model_name: str,
    client,
    grid_image,
    n_samples: Optional[int],
    use_responses_api: bool,
    use_anthropic_api: bool,
    dry_run: bool,
) -> None:
    """Rerun affected rows in the batch (no_context) condition."""
    results_path = results_path_for(model_name, BATCH_PREP_PATH)
    df_input = pd.read_csv(BATCH_PREP_PATH)
    df_results = pd.read_csv(results_path)

    affected_mask = has_banana(df_input)
    print(f"  Batch (no_context): {affected_mask.sum()} affected rows out of {len(df_input)}")
    if affected_mask.sum() == 0:
        raise RuntimeError("No affected rows found; nothing to rerun.")
    if dry_run:
        print(df_input.loc[affected_mask, BATCH_KEY_COLS].to_string(index=False))
        return

    backup_path = backup_results(results_path)
    df_subset = df_input.loc[affected_mask].copy().reset_index(drop=True)
    df_new = get_logits(
        df_subset,
        model_name,
        client,
        grid_image,
        include_image=True,
        n_samples=n_samples,
        raw_responses_path=raw_responses_path_for(results_path, n_samples),
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
        checkpoint_path=f"{results_path}.banana_rerun.checkpoint",
    )

    # Replace the affected rows, matching on the trial key rather than position.
    key = lambda df: df[BATCH_KEY_COLS].astype(str).agg("_".join, axis=1)
    df_kept = df_results[~key(df_results).isin(set(key(df_new)))]
    df_merged = pd.concat([df_kept, df_new], ignore_index=True)

    report = compare_rerun(df_input, df_merged, pd.read_csv(backup_path), mode="batch")
    verify_and_save(report, df_merged, df_results, results_path)


def rerun_interactive(
    model_name: str,
    client,
    grid_image,
    n_samples: Optional[int],
    use_responses_api: bool,
    use_anthropic_api: bool,
    dry_run: bool,
) -> None:
    """Rerun affected sessions in the interactive (limited_feedback_yoked) condition."""
    results_path = results_path_for(model_name, INTERACTIVE_PREP_PATH)
    df_input = pd.read_csv(INTERACTIVE_PREP_PATH)
    df_results = pd.read_csv(results_path)

    sessions = affected_sessions(df_input)
    session_mask = df_input[SESSION_COL].isin(sessions)
    print(
        f"  Interactive (limited_feedback_yoked): {len(sessions)} affected sessions, "
        f"{session_mask.sum()} rows to rerun out of {len(df_input)}"
    )
    if not sessions:
        raise RuntimeError("No affected sessions found; nothing to rerun.")
    if dry_run:
        print(f"  Affected sessions: {sorted(sessions)}")
        return

    backup_path = backup_results(results_path)
    df_subset = df_input.loc[session_mask].copy().reset_index(drop=True)
    df_new = run_interactive_evaluation(
        df_subset,
        model_name,
        client,
        grid_image,
        include_image=True,
        n_samples=n_samples,
        raw_responses_path=raw_responses_path_for(results_path, n_samples),
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
        checkpoint_path=f"{results_path}.banana_rerun.checkpoint",
    )

    df_kept = df_results[~df_results[SESSION_COL].astype(str).isin({str(s) for s in sessions})]
    df_merged = pd.concat([df_kept, df_new], ignore_index=True)

    report = compare_rerun(
        df_input, df_merged, pd.read_csv(backup_path), mode="interactive"
    )
    verify_and_save(report, df_merged, df_results, results_path)


@dataclass
class Args:
    model_name: str
    """Frontier model name, e.g. gemini-3-flash-preview, gpt-5.2, claude-sonnet-4-6."""

    api_base: str
    """API base URL; selects the client (google / anthropic / openai)."""

    mode: RerunMode = "both"
    """Which condition(s) to rerun: batch (no_context), interactive (limited_feedback_yoked), or both."""

    grid_image_path: Path = Path("data/compiled_grid.png")

    n_samples: Optional[int] = None
    """Resample N times instead of using logprobs (frontier models use 10)."""

    dry_run: bool = False
    """Print the affected rows / sessions without calling any API."""


def main(args: Args) -> None:
    grid_image = Image.open(here(str(args.grid_image_path)))
    client, use_responses_api, use_anthropic_api = setup_client(args.api_base)
    print(f"Model: {args.model_name}\nMode: {args.mode}\nDry run: {args.dry_run}\n")

    common = (args.model_name, client, grid_image, args.n_samples,
              use_responses_api, use_anthropic_api, args.dry_run)
    if args.mode in ("batch", "both"):
        print("=== Batch mode (no_context) ===")
        rerun_batch(*common)
    if args.mode in ("interactive", "both"):
        print("\n=== Interactive mode (limited_feedback_yoked) ===")
        rerun_interactive(*common)


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
