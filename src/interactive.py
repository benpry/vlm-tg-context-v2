"""
Code for the interactive version of the language model evaluation, where the model gets limited
feedback on its own choices rather than human responses.
"""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from src.lm import (
    CHOICES,
    SYSTEM_PROMPT,
    _count_chat_prompt_tokens,
    _estimate_image_tokens,
    _get_encoding,
    get_completion_with_backoff,
)
from src.utils import (
    get_logprobs_from_genai_response,
    get_logprobs_from_openai_choice,
    get_openai_messages,
    preprocess_messages,
)


def update_histories(df: pd.DataFrame, trial_num: int):
    if trial_num == df["trialNum"].max():
        return

    # Get the original indices of future rounds before filtering
    future_rounds_mask = df["trialNum"] > trial_num
    future_rounds_indices = df[future_rounds_mask].index

    df_future_rounds = df.loc[future_rounds_mask][
        [
            "workerid",
            "trialNum",
            "selection_history",
            "correctness_history",
            "target_history",
        ]
    ].copy()

    # Use map instead of merge to preserve the original index and row count
    prediction_map = df[df["trialNum"] == trial_num].set_index("workerid")[
        "model_prediction"
    ]
    df_future_rounds["model_prediction"] = df_future_rounds["workerid"].map(
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
        + [x["model_prediction"] == x["target_history"][x["trialNum"] - 1]],
        axis=1,
    )

    # update the dataframe's selection and correctness histories using original indices
    df.loc[future_rounds_indices, "selection_history"] = df_future_rounds[
        "selection_history"
    ].values
    df.loc[future_rounds_indices, "correctness_history"] = df_future_rounds[
        "correctness_history"
    ].values


def process_interactive_row(client, model_name, messages):
    response = get_completion_with_backoff(
        client=client,
        model=model_name,
        messages=messages,
    )

    if "gemini" in model_name.lower():
        choice_logprobs = get_logprobs_from_genai_response(response, CHOICES)
    else:
        choice_logprobs = get_logprobs_from_openai_choice(response.choices[0], CHOICES)

    if choice_logprobs:
        prediction = max(choice_logprobs, key=choice_logprobs.get)
    else:
        # If no choice tokens found, take the generated text
        content = response.choices[0].message.content
        prediction = content.strip() if content else ""

    return choice_logprobs, prediction


def run_interactive_evaluation(
    df: pd.DataFrame,
    model_name: str,
    client: OpenAI,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> pd.DataFrame:
    if n_trials is not None:
        df = df.head(n_trials)

    if "trialNum" not in df.columns:
        df["trialNum"] = df["matcher_trialNum"]

    df["selection_history"] = [[] for _ in range(len(df))]
    df["correctness_history"] = [[] for _ in range(len(df))]

    # Ensure columns exist
    df["model_logprobs"] = None
    df["model_prediction"] = None
    # We need to make sure we can assign lists to model_logprobs, so convert to object type if needed
    df["model_logprobs"] = df["model_logprobs"].astype(object)

    for trial_num in range(df["trialNum"].max() + 1):
        df_round = df[df["trialNum"] == trial_num].copy()

        if df_round.empty:
            continue

        df_round["chat_prompt"] = df_round.apply(preprocess_messages, axis=1)

        print(f"Processing round {trial_num}...")

        # Prepare messages outside threads
        row_messages = []
        for idx, row in df_round.iterrows():
            messages = get_openai_messages(
                SYSTEM_PROMPT, row["chat_prompt"], include_image, grid_image, model_name
            )
            row_messages.append(messages)

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda msgs: process_interactive_row(client, model_name, msgs),
                        row_messages,
                    ),
                    total=len(row_messages),
                )
            )

        choice_logprobs_list = [r[0] for r in results]
        predictions_list = [r[1] for r in results]

        # save the logprobs to the dataframe
        df.loc[df_round.index, "model_logprobs"] = choice_logprobs_list
        df.loc[df_round.index, "model_prediction"] = predictions_list

        # update the selection and correctness histories
        update_histories(df, trial_num)

    return df


def count_tokens_interactive(
    df: pd.DataFrame,
    model_name: str,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> dict:
    """Count input/output tokens for interactive evaluation without calling the API.

    Uses random dummy responses (A-L) to simulate history updates so that
    later rounds have realistic context lengths.
    """
    if n_trials is not None:
        df = df.head(n_trials)

    if "trialNum" not in df.columns:
        df["trialNum"] = df["matcher_trialNum"]

    df["selection_history"] = [[] for _ in range(len(df))]
    df["correctness_history"] = [[] for _ in range(len(df))]
    df["model_prediction"] = None

    encoding = _get_encoding(model_name)
    system_prompt_tokens = len(encoding.encode(SYSTEM_PROMPT))
    image_tokens = _estimate_image_tokens(grid_image) if include_image else 0

    text_token_counts = []
    total_rows = 0

    for trial_num in range(df["trialNum"].max() + 1):
        df_round = df[df["trialNum"] == trial_num].copy()
        if df_round.empty:
            continue

        df_round["chat_prompt"] = df_round.apply(preprocess_messages, axis=1)

        for idx, row in df_round.iterrows():
            tokens = system_prompt_tokens + _count_chat_prompt_tokens(
                row["chat_prompt"], encoding
            )
            text_token_counts.append(tokens)
            total_rows += 1

        # Use random dummy predictions to update histories for future rounds
        predictions = [random.choice(CHOICES) for _ in range(len(df_round))]
        df.loc[df_round.index, "model_prediction"] = predictions

        update_histories(df, trial_num)

    total_input = sum(text_token_counts) + (image_tokens * total_rows)
    total_output = total_rows

    return {
        "n_rows": total_rows,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "image_tokens_per_row": image_tokens,
        "text_tokens_per_row": {
            "min": min(text_token_counts) if text_token_counts else 0,
            "max": max(text_token_counts) if text_token_counts else 0,
            "mean": (
                sum(text_token_counts) / len(text_token_counts)
                if text_token_counts
                else 0
            ),
        },
    }
