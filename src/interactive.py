"""
Code for the interactive version of the language model evaluation, where the model gets limited
feedback on its own choices rather than human responses.
"""

import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from src.lm import (
    CHOICES,
    SYSTEM_PROMPT,
    get_completion_with_backoff,
    get_samples_single_row,
)
from src.utils import (
    encode_image,
    get_logprobs_from_genai_response,
    get_logprobs_from_openai_choice,
    get_logprobs_from_responses_api,
    get_openai_messages,
    preprocess_messages,
)


def _add_session_id(df: pd.DataFrame) -> None:
    """Add a _session_id column that uniquely identifies each session within a trialNum."""
    if "workerid" in df.columns:
        df["_session_id"] = df["workerid"]
    elif "shuffle_rep" in df.columns:
        df["_session_id"] = (
            df["gameId"].astype(str) + "_" + df["shuffle_rep"].astype(str)
        )
    else:
        df["_session_id"] = df["gameId"]


def update_histories(df: pd.DataFrame, trial_num: int):
    if trial_num == df["trialNum"].max():
        return

    # Get the original indices of future rounds before filtering
    future_rounds_mask = df["trialNum"] > trial_num
    future_rounds_indices = df[future_rounds_mask].index

    df_future_rounds = df.loc[future_rounds_mask][
        [
            "_session_id",
            "trialNum",
            "selection_history",
            "correctness_history",
            "target_history",
        ]
    ].copy()

    # Use map instead of merge to preserve the original index and row count
    prediction_map = df[df["trialNum"] == trial_num].set_index("_session_id")[
        "model_prediction"
    ]
    df_future_rounds["model_prediction"] = df_future_rounds["_session_id"].map(
        prediction_map
    )

    # update the selection history
    df_future_rounds["selection_history"] = df_future_rounds.apply(
        lambda x: x["selection_history"] + [x["model_prediction"]],
        axis=1,
    )
    # update the correctness history
    df_future_rounds["correctness_history"] = df_future_rounds.apply(
        lambda x: x["correctness_history"]
        + [x["model_prediction"] == x["target_history"][trial_num]],
        axis=1,
    )

    # update the dataframe's selection and correctness histories using original indices
    df.loc[future_rounds_indices, "selection_history"] = df_future_rounds[
        "selection_history"
    ].values
    df.loc[future_rounds_indices, "correctness_history"] = df_future_rounds[
        "correctness_history"
    ].values


def process_interactive_row(
    client,
    model_name,
    messages,
    n_samples=None,
    use_responses_api=False,
    use_anthropic_api=False,
):
    if n_samples:
        choice_logprobs, raw_responses = get_samples_single_row(
            client,
            model_name,
            messages,
            n_samples,
            use_responses_api,
            use_anthropic_api,
        )
        prediction = (
            max(choice_logprobs, key=choice_logprobs.get) if choice_logprobs else ""
        )
        return choice_logprobs, prediction, raw_responses

    response = get_completion_with_backoff(
        client=client,
        model=model_name,
        messages=messages,
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
    )

    if "gemini" in model_name.lower():
        choice_logprobs = get_logprobs_from_genai_response(response, CHOICES)
    elif use_responses_api:
        choice_logprobs = get_logprobs_from_responses_api(response, CHOICES)
    else:
        choice_logprobs = get_logprobs_from_openai_choice(response.choices[0], CHOICES)

    if choice_logprobs:
        prediction = max(choice_logprobs, key=choice_logprobs.get)
    else:
        if use_responses_api:
            content = response.output_text
        elif use_anthropic_api:
            content = response.content[0].text
        else:
            content = response.choices[0].message.content
        prediction = content.strip() if content else ""

    return choice_logprobs, prediction, None


def _save_interactive_checkpoint(
    df: pd.DataFrame,
    checkpoint_path: str,
    trial_num: int,
    all_raw_responses: Optional[dict] = None,
):
    """Save checkpoint after completing a trial round."""
    df.to_csv(checkpoint_path, index=False)
    meta_path = checkpoint_path.replace(".checkpoint", ".checkpoint_meta.json")
    meta = {"last_completed_trial": trial_num}
    if all_raw_responses is not None:
        raw_path = checkpoint_path.replace(".checkpoint", ".checkpoint_raw.json")
        with open(raw_path, "w") as f:
            json.dump(all_raw_responses, f)
        meta["has_raw_responses"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Checkpoint saved after trial {trial_num}")


def _load_interactive_checkpoint(checkpoint_path: str):
    """Load checkpoint if it exists. Returns (df, last_completed_trial, raw_responses) or None."""
    meta_path = checkpoint_path.replace(".checkpoint", ".checkpoint_meta.json")
    if not os.path.exists(checkpoint_path) or not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.read_csv(checkpoint_path)
    # Restore list columns from their string representations
    for col in ["selection_history", "correctness_history", "target_history"]:
        if col in df.columns:

            def _parse_list(x):
                if not isinstance(x, str) or x in ("", "nan"):
                    return x
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return x

            df[col] = df[col].apply(_parse_list)
    # Restore model_logprobs dict column
    if "model_logprobs" in df.columns:

        def _parse_logprobs(x):
            if not isinstance(x, str) or x in ("", "nan"):
                return x
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x

        df["model_logprobs"] = df["model_logprobs"].apply(_parse_logprobs)
        df["model_logprobs"] = df["model_logprobs"].astype(object)
    raw_responses = None
    if meta.get("has_raw_responses"):
        raw_path = checkpoint_path.replace(".checkpoint", ".checkpoint_raw.json")
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                raw_responses = json.load(f)
            # Keys are stored as strings in JSON, convert back to int
            raw_responses = {int(k): v for k, v in raw_responses.items()}
    return df, meta["last_completed_trial"], raw_responses


def _cleanup_checkpoint(checkpoint_path: str):
    """Remove checkpoint files after successful completion."""
    for suffix in [".checkpoint", ".checkpoint_meta.json", ".checkpoint_raw.json"]:
        path = checkpoint_path.replace(".checkpoint", suffix)
        if os.path.exists(path):
            os.remove(path)


def run_interactive_evaluation(
    df: pd.DataFrame,
    model_name: str,
    client: OpenAI,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
    n_samples: Optional[int] = None,
    raw_responses_path: Optional[str] = None,
    use_responses_api: bool = False,
    use_anthropic_api: bool = False,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    if n_trials is not None:
        df = df.head(n_trials)

    if "trialNum" not in df.columns:
        df["trialNum"] = df["matcher_trialNum"]

    _add_session_id(df)

    # Try to resume from checkpoint
    resume_from_trial = -1
    all_raw_responses = {} if raw_responses_path else None

    if checkpoint_path:
        loaded = _load_interactive_checkpoint(checkpoint_path)
        if loaded is not None:
            df_loaded, resume_from_trial, loaded_raw = loaded
            # Restore the session id column for the loaded data
            _add_session_id(df_loaded)
            df = df_loaded
            if loaded_raw is not None and all_raw_responses is not None:
                all_raw_responses = loaded_raw
            print(
                f"Resuming from checkpoint after trial {resume_from_trial} "
                f"({resume_from_trial + 1}/{df['trialNum'].max() + 1} trials completed)"
            )

    if resume_from_trial < 0:
        df["selection_history"] = [[] for _ in range(len(df))]
        df["correctness_history"] = [[] for _ in range(len(df))]
        # Parse target_history from CSV string to list
        df["target_history"] = df["target_history"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        # Ensure columns exist
        df["model_logprobs"] = None
        df["model_prediction"] = None
        # We need to make sure we can assign lists to model_logprobs, so convert to object type if needed
        df["model_logprobs"] = df["model_logprobs"].astype(object)

    base64_image = encode_image(grid_image) if include_image else None

    for trial_num in range(df["trialNum"].max() + 1):
        if trial_num <= resume_from_trial:
            continue
        df_round = df[df["trialNum"] == trial_num].copy()

        if df_round.empty:
            continue

        df_round["chat_prompt"] = df_round.apply(preprocess_messages, axis=1)

        print(f"Processing round {trial_num}...")

        # Prepare messages outside threads
        row_messages = []
        for idx, row in df_round.iterrows():
            messages = get_openai_messages(
                SYSTEM_PROMPT,
                row["chat_prompt"],
                include_image,
                grid_image,
                model_name,
                use_responses_api=use_responses_api,
                base64_image=base64_image,
            )
            row_messages.append(messages)

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda msgs: process_interactive_row(
                            client,
                            model_name,
                            msgs,
                            n_samples,
                            use_responses_api,
                            use_anthropic_api,
                        ),
                        row_messages,
                    ),
                    total=len(row_messages),
                )
            )

        choice_logprobs_list = [r[0] for r in results]
        predictions_list = [r[1] for r in results]
        raw_responses_list = [r[2] for r in results]

        df.loc[df_round.index, "model_logprobs"] = choice_logprobs_list
        df.loc[df_round.index, "model_prediction"] = predictions_list
        if all_raw_responses is not None:
            for idx, raw in zip(df_round.index, raw_responses_list):
                if raw is not None:
                    all_raw_responses[int(idx)] = raw

        # update the selection and correctness histories
        update_histories(df, trial_num)

        # Save checkpoint after each completed trial
        if checkpoint_path:
            _save_interactive_checkpoint(
                df, checkpoint_path, trial_num, all_raw_responses
            )

    # Clean up checkpoint files after successful completion
    if checkpoint_path:
        _cleanup_checkpoint(checkpoint_path)

    if raw_responses_path and all_raw_responses:
        with open(raw_responses_path, "w") as f:
            json.dump(all_raw_responses, f, indent=2)
        print(f"Raw responses saved to {raw_responses_path}")

    return df.drop(columns=["_session_id"])
