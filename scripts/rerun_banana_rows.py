"""
Rerun frontier models (Claude, GPT, Gemini) on rows affected by the banana→ba''a bug.

The bug was in src/utils.py where literal_eval(x.replace("nan", "''")) corrupted
"banana" to "ba''a". This script reruns only the affected rows to save cost.

- Batch mode (no_context): reruns individual affected rows (where message contains "banana")
- Interactive mode (limited_feedback_yoked): reruns full sessions that contain any affected row,
  since model predictions cascade through the session's feedback history.
"""

import os
import shutil
from argparse import ArgumentParser

import anthropic
import pandas as pd
from google import genai
from openai import OpenAI
from PIL import Image
from pyprojroot import here

from src.interactive import run_interactive_evaluation
from src.lm import get_logits

BATCH_KEY_COLS = ["gameId", "trialNum", "repNum"]
INTERACTIVE_SESSION_COL = "workerid"


def has_banana(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of rows where 'banana' appears in message or message_history."""
    in_message = df["message"].str.contains("banana", case=False, na=False)
    in_history = df["message_history"].str.contains("banana", case=False, na=False)
    return in_message | in_history


def setup_client(api_base: str):
    """Set up the API client based on the api_base URL. Returns (client, use_responses_api, use_anthropic_api)."""
    if "google" in api_base:
        client = genai.Client(api_key=os.getenv("COCOLAB_GEMINI_API_KEY"))
        return client, False, False
    elif "anthropic" in api_base:
        client = anthropic.Anthropic()
        return client, False, True
    else:
        client = OpenAI(base_url=api_base)
        use_responses_api = "openai.com" in api_base
        return client, use_responses_api, False


def rerun_batch(
    model_name: str,
    client,
    grid_image,
    n_samples: int | None,
    use_responses_api: bool,
    use_anthropic_api: bool,
    dry_run: bool,
):
    """Rerun affected rows in the batch (no_context) condition."""
    short_name = model_name.split("/")[-1]
    input_path = here("context_prep/full_feedback/no_context.csv")
    results_path = here(
        f"data/logprobs/full_feedback/no_context_{short_name}_logprobs.csv"
    )

    if not os.path.exists(results_path):
        print(f"  No existing results at {results_path}, skipping batch mode.")
        return

    df_input = pd.read_csv(input_path)
    df_results = pd.read_csv(results_path)

    # Find affected rows using the input CSV (which has uncorrupted text)
    affected_mask = has_banana(df_input)
    n_affected = affected_mask.sum()
    print(f"  Batch (no_context): {n_affected} affected rows out of {len(df_input)}")

    if n_affected == 0:
        print("  No affected rows, skipping.")
        return

    if dry_run:
        print("  Affected rows:")
        print(df_input.loc[affected_mask, BATCH_KEY_COLS].to_string(index=False))
        return

    # Back up existing results
    backup_path = str(results_path) + ".bak"
    shutil.copy2(results_path, backup_path)
    print(f"  Backed up results to {backup_path}")

    # Run inference on affected rows
    df_subset = df_input.loc[affected_mask].copy().reset_index(drop=True)

    checkpoint_path = str(results_path) + ".banana_rerun.checkpoint"

    raw_responses_path = None
    if n_samples:
        raw_responses_path = (
            str(results_path)
            .replace("data/logprobs", "data/raw_responses")
            .replace(".csv", "_banana_rerun.json")
        )
        os.makedirs(os.path.dirname(raw_responses_path), exist_ok=True)

    df_new = get_logits(
        df_subset,
        model_name,
        client,
        grid_image,
        include_image=True,
        n_samples=n_samples,
        raw_responses_path=raw_responses_path,
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
        checkpoint_path=checkpoint_path,
    )

    # Merge new results back into existing results
    # Create a key for matching
    df_results["_key"] = df_results[BATCH_KEY_COLS].astype(str).agg("_".join, axis=1)
    df_new["_key"] = df_new[BATCH_KEY_COLS].astype(str).agg("_".join, axis=1)

    # Replace affected rows
    affected_keys = set(df_new["_key"])
    df_kept = df_results[~df_results["_key"].isin(affected_keys)]
    df_merged = pd.concat([df_kept, df_new], ignore_index=True).drop(columns=["_key"])

    assert len(df_merged) == len(df_results), (
        f"Row count mismatch: {len(df_merged)} vs {len(df_results)}"
    )

    df_merged.to_csv(results_path, index=False)
    print(f"  Saved updated results to {results_path}")


def rerun_interactive(
    model_name: str,
    client,
    grid_image,
    n_samples: int | None,
    use_responses_api: bool,
    use_anthropic_api: bool,
    dry_run: bool,
):
    """Rerun affected sessions in the interactive (limited_feedback_yoked) condition."""
    short_name = model_name.split("/")[-1]
    input_path = here("context_prep/human_history/limited_feedback_yoked.csv")
    results_path = here(
        f"data/logprobs/interactive/limited_feedback_yoked_{short_name}_logprobs.csv"
    )

    if not os.path.exists(results_path):
        print(f"  No existing results at {results_path}, skipping interactive mode.")
        return

    df_input = pd.read_csv(input_path)
    df_results = pd.read_csv(results_path)

    # Find affected sessions: any session where any row has banana
    affected_mask = has_banana(df_input)
    affected_sessions = df_input.loc[affected_mask, INTERACTIVE_SESSION_COL].unique()
    n_sessions = len(affected_sessions)
    session_rows = df_input[
        df_input[INTERACTIVE_SESSION_COL].isin(affected_sessions)
    ].shape[0]
    print(
        f"  Interactive (limited_feedback_yoked): {n_sessions} affected sessions, "
        f"{session_rows} rows to rerun out of {len(df_input)}"
    )

    if n_sessions == 0:
        print("  No affected sessions, skipping.")
        return

    if dry_run:
        print(f"  Affected sessions: {list(affected_sessions)}")
        return

    # Back up existing results
    backup_path = str(results_path) + ".bak"
    shutil.copy2(results_path, backup_path)
    print(f"  Backed up results to {backup_path}")

    # Filter input to affected sessions only
    df_subset = (
        df_input[df_input[INTERACTIVE_SESSION_COL].isin(affected_sessions)]
        .copy()
        .reset_index(drop=True)
    )

    checkpoint_path = str(results_path) + ".banana_rerun.checkpoint"

    raw_responses_path = None
    if n_samples:
        raw_responses_path = (
            str(results_path)
            .replace("data/logprobs", "data/raw_responses")
            .replace(".csv", "_banana_rerun.json")
        )
        os.makedirs(os.path.dirname(raw_responses_path), exist_ok=True)

    df_new = run_interactive_evaluation(
        df_subset,
        model_name,
        client,
        grid_image,
        include_image=True,
        n_samples=n_samples,
        raw_responses_path=raw_responses_path,
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
        checkpoint_path=checkpoint_path,
    )

    # Merge new results back into existing results, replacing affected sessions
    df_kept = df_results[~df_results[INTERACTIVE_SESSION_COL].isin(affected_sessions)]
    df_merged = pd.concat([df_kept, df_new], ignore_index=True)

    assert len(df_merged) == len(df_results), (
        f"Row count mismatch: {len(df_merged)} vs {len(df_results)}"
    )

    df_merged.to_csv(results_path, index=False)
    print(f"  Saved updated results to {results_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rerun frontier models on banana-affected rows")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_base", type=str, required=True)
    parser.add_argument(
        "--grid_image_path",
        type=str,
        default="data/compiled_grid.png",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Resample N times instead of using logprobs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print affected rows/sessions without running inference",
    )
    args = parser.parse_args()

    grid_image = Image.open(here(args.grid_image_path))
    client, use_responses_api, use_anthropic_api = setup_client(args.api_base)

    print(f"Model: {args.model_name}")
    print(f"Dry run: {args.dry_run}")
    print()

    # print("=== Batch mode (no_context) ===")
    # rerun_batch(
    #     args.model_name,
    #     client,
    #     grid_image,
    #     args.n_samples,
    #     use_responses_api,
    #     use_anthropic_api,
    #     args.dry_run,
    # )

    print()
    print("=== Interactive mode (limited_feedback_yoked) ===")
    rerun_interactive(
        args.model_name,
        client,
        grid_image,
        args.n_samples,
        use_responses_api,
        use_anthropic_api,
        args.dry_run,
    )
