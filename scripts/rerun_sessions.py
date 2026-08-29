"""
Rerun whole sessions of a frontier model's interactive-yoked evaluation.

Interactive runs cascade: the model's own prediction on trial t is part of
every later prompt of the session. After a change that alters what the model
was fed back (e.g. the choice-extraction fix, see
scripts/reparse_frontier_responses.py), the affected sessions have to be rerun
from scratch. This script reruns the given sessions (workerids), merges them
into the results file (backed up first to <results>.pre_session_rerun.bak,
never overwritten) and keeps the raw responses in
data/raw_responses/interactive/<results stem>_session_rerun.json.

    python scripts/rerun_sessions.py --model_name gemini-3-flash-preview \
        --api_base https://generativelanguage.googleapis.com/ --n_samples 10 \
        --sessions_file data/logprobs/frontier/limited_feedback_yoked_gemini-3-flash-preview_logprobs.csv.cascade_sessions.json
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro
from PIL import Image
from pyprojroot import here

from src.banana_rerun import SESSION_COL
from src.clients import setup_client
from src.interactive import run_interactive_evaluation
from src.session_rerun import merge_rerun_sessions

PREP_PATH = here("context_prep/human_history/limited_feedback_yoked.csv")
RAW_RESPONSES_DIR = here("data/raw_responses/interactive")


@dataclass
class Args:
    model_name: str
    """Frontier model name, e.g. gemini-3-flash-preview."""

    api_base: str
    """API base URL; selects the client (google / anthropic / openai)."""

    sessions: list[str] = field(default_factory=list)
    """Sessions (workerids) to rerun."""

    sessions_file: Optional[Path] = None
    """JSON list of sessions to rerun (e.g. the .cascade_sessions.json of a re-parse)."""

    n_samples: Optional[int] = 10
    """Resample N times instead of using logprobs (frontier models use 10)."""

    results_path: Optional[Path] = None
    """Defaults to data/logprobs/frontier/limited_feedback_yoked_<model_name>_logprobs.csv."""

    grid_image_path: Path = Path("data/compiled_grid.png")

    dry_run: bool = False
    """List the rows that would be rerun without calling any API."""


def main(args: Args) -> None:
    sessions = {str(s) for s in args.sessions}
    if args.sessions_file is not None:
        with open(args.sessions_file) as f:
            sessions |= {str(s) for s in json.load(f)}
    if not sessions:
        raise ValueError("No sessions given (use --sessions or --sessions_file)")

    results_path = args.results_path or here(
        f"data/logprobs/frontier/limited_feedback_yoked_{args.model_name}_logprobs.csv"
    )
    if not Path(results_path).exists():
        raise FileNotFoundError(results_path)
    df_prep = pd.read_csv(PREP_PATH)
    df_results = pd.read_csv(results_path)

    prep_sessions = set(df_prep[SESSION_COL].astype(str))
    unknown = sessions - prep_sessions
    if unknown:
        raise ValueError(f"Sessions {sorted(unknown)} are not in {PREP_PATH}")
    session_mask = df_prep[SESSION_COL].astype(str).isin(sessions)
    print(f"Rerunning {len(sessions)} sessions, {session_mask.sum()} rows of {len(df_prep)}: {sorted(sessions)}")
    if args.dry_run:
        return

    raw_responses_path = RAW_RESPONSES_DIR / f"{Path(results_path).stem}_session_rerun.json"
    backup_path = Path(f"{results_path}.pre_session_rerun.bak")
    for path in [raw_responses_path, backup_path]:
        if path.exists():
            raise FileExistsError(f"{path} exists; move it away before rerunning")
    RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_path, backup_path)
    print(f"Backed up results to {backup_path}")

    client, use_responses_api, use_anthropic_api = setup_client(args.api_base)
    grid_image = Image.open(here(str(args.grid_image_path)))
    df_new = run_interactive_evaluation(
        df_prep.loc[session_mask].copy().reset_index(drop=True),
        args.model_name,
        client,
        grid_image,
        include_image=True,
        n_samples=args.n_samples,
        raw_responses_path=str(raw_responses_path) if args.n_samples else None,
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
        checkpoint_path=f"{results_path}.session_rerun.checkpoint",
    )

    df_merged = merge_rerun_sessions(df_results, df_new, sessions, SESSION_COL)
    df_merged.to_csv(results_path, index=False)
    print(f"Saved {results_path} ({len(df_merged)} rows); raw responses in {raw_responses_path}")


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
