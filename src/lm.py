"""
Code for calling the language model to get choice logits
"""

import ast
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
from google.genai import types
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.utils import (
    convert_to_anthropic_format,
    convert_to_google_genai_style,
    encode_image,
    get_logprobs_from_genai_response,
    get_logprobs_from_openai_choice,
    get_logprobs_from_responses_api,
    get_openai_messages,
    preprocess_messages,
)

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def _extract_choice(text: str) -> str:
    """
    Extract a choice letter from potentially longer reasoning output.
    We take the last capital letter in the string that would be a valid choice.
    """
    text = text.strip()
    if text in CHOICES:
        return text
    for ch in reversed(text):
        if ch in CHOICES:
            return ch
    return text


SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose a shape from a set of shapes. Your goal is to guess which of the shapes the describer is trying to get the matcher to choose. The shapes, with their labels, are shown in the image.
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose, and no other text. You will receive feedback telling you whether your choice was correct or incorrect.
"""


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def get_completion_with_backoff(
    client,
    model,
    messages,
    use_logprobs=True,
    use_responses_api=False,
    use_anthropic_api=False,
):
    """Call a language model, using exponential backoff to get around rate limits."""
    if "gemini" in model.lower():
        # use the google genai client
        genai_messages, system_instruction = convert_to_google_genai_style(messages)
        return client.models.generate_content(
            model=model,
            contents=genai_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                tools=[],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                ),
                max_output_tokens=256,
            ),
        )
    elif use_anthropic_api:
        # use the native Anthropic Messages API
        system_prompt, anthropic_messages = convert_to_anthropic_format(messages)
        # Apply cache_control to the last content block
        last_msg = anthropic_messages[-1]
        if isinstance(last_msg["content"], list):
            last_msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
        else:
            # Convert string content to block format for cache_control
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": last_msg["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        max_tokens = 1 if use_logprobs else 256
        return client.messages.create(
            model=model,
            system=system_prompt,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            output_config={"effort": "low"},
        )
    elif use_responses_api:
        # use the OpenAI Responses API
        if use_logprobs:
            return client.responses.create(
                model=model,
                input=messages,
                reasoning={"effort": "none"},
                max_output_tokens=1,
                temperature=1,
                top_logprobs=20,
                include=["message.output_text.logprobs"],
            )
        else:
            return client.responses.create(
                model=model,
                input=messages,
                reasoning={"effort": "none"},
                max_output_tokens=256,
            )
    else:
        # use the OpenAI Chat Completions API (for local models, etc.)
        if use_logprobs:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1,
                temperature=1,
                logprobs=True,
                top_logprobs=1000,
            )
        else:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=256,
            )


def get_logits_single_row(
    client,
    model_name: str,
    messages: list,
    use_responses_api: bool = False,
    use_anthropic_api: bool = False,
) -> dict:
    """Get a completion from a model and return the logprobs"""
    response = get_completion_with_backoff(
        client=client,
        model=model_name,
        messages=messages,
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
    )
    if "gemini" in model_name.lower():
        return get_logprobs_from_genai_response(response, CHOICES)
    elif use_responses_api:
        return get_logprobs_from_responses_api(response, CHOICES)
    else:
        return get_logprobs_from_openai_choice(response.choices[0], CHOICES)


def get_single_sample(
    client, model_name, messages, use_responses_api=False, use_anthropic_api=False
):
    """Make a single API call and return (raw_text, extracted_choice)."""
    response = get_completion_with_backoff(
        client,
        model_name,
        messages,
        use_logprobs=False,
        use_responses_api=use_responses_api,
        use_anthropic_api=use_anthropic_api,
    )
    if "gemini" in model_name.lower():
        raw = (response.text or "").strip()
    elif use_responses_api:
        raw = response.output_text.strip()
    elif use_anthropic_api:
        raw = response.content[0].text.strip()
    else:
        raw = response.choices[0].message.content.strip()
    return raw, _extract_choice(raw)


def _counts_to_logprobs(counts, n_samples):
    """Convert a Counter of choice frequencies to log-probabilities."""
    return {choice: math.log(count / n_samples) for choice, count in counts.items()}


def get_samples_single_row(
    client,
    model_name,
    messages,
    n_samples,
    use_responses_api=False,
    use_anthropic_api=False,
):
    """Resample n_samples times and return (logprobs_dict, raw_responses)."""
    results = [
        get_single_sample(
            client, model_name, messages, use_responses_api, use_anthropic_api
        )
        for _ in range(n_samples)
    ]
    raw_responses = [raw for raw, _ in results]
    tokens = [choice for _, choice in results]
    counts = Counter(t for t in tokens if t in CHOICES)
    logprobs = _counts_to_logprobs(counts, n_samples) if counts else {}
    return logprobs, raw_responses


REQUIRED_COLUMNS = [
    "message_history",
    "selection_history",
    "correctness_history",
    "message",
]


def _save_batch_checkpoint(
    df: pd.DataFrame, checkpoint_path: str, completed_indices: list
):
    """Save checkpoint for batch mode with list of completed row indices."""
    df.to_csv(checkpoint_path, index=False)
    meta_path = checkpoint_path.replace(".checkpoint", ".checkpoint_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"completed_indices": completed_indices}, f)
    print(f"Checkpoint saved ({len(completed_indices)}/{len(df)} rows completed)")


def _load_batch_checkpoint(checkpoint_path: str):
    """Load batch checkpoint. Returns (df, completed_indices) or None."""
    meta_path = checkpoint_path.replace(".checkpoint", ".checkpoint_meta.json")
    if not os.path.exists(checkpoint_path) or not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.read_csv(checkpoint_path)
    # Restore list columns from string representations
    for col in [
        "selection_history",
        "correctness_history",
        "target_history",
        "message_history",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
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
    return df, meta["completed_indices"]


def _cleanup_batch_checkpoint(checkpoint_path: str):
    """Remove batch checkpoint files."""
    for suffix in [".checkpoint", ".checkpoint_meta.json"]:
        path = checkpoint_path.replace(".checkpoint", suffix)
        if os.path.exists(path):
            os.remove(path)


BATCH_CHECKPOINT_INTERVAL = 100


def get_logits(
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
    # Validate required columns exist
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "For yoked/limited feedback evaluation, ensure your data includes "
            "message_history, selection_history, correctness_history, and message columns."
        )

    if n_trials is not None:
        df = df.sample(n_trials)

    # Try to resume from checkpoint
    completed_indices = set()
    if checkpoint_path:
        loaded = _load_batch_checkpoint(checkpoint_path)
        if loaded is not None:
            df, completed_idx_list = loaded
            completed_indices = set(completed_idx_list)
            print(
                f"Resuming from checkpoint ({len(completed_indices)}/{len(df)} rows already completed)"
            )

    if "model_logprobs" not in df.columns:
        df["model_logprobs"] = None
        df["model_logprobs"] = df["model_logprobs"].astype(object)

    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

    # Determine which rows still need processing
    remaining_mask = ~df.index.isin(completed_indices)
    remaining_indices = df.index[remaining_mask].tolist()

    if not remaining_indices:
        print("All rows already completed from checkpoint.")
        return df.drop(columns=["chat_prompt"])

    print(f"Preparing messages for {len(remaining_indices)} remaining rows...")
    base64_image = encode_image(grid_image) if include_image else None
    remaining_messages = [
        get_openai_messages(
            SYSTEM_PROMPT,
            df.loc[idx, "chat_prompt"],
            include_image,
            grid_image,
            model_name,
            use_responses_api=use_responses_api,
            base64_image=base64_image,
        )
        for idx in remaining_indices
    ]

    print("Doing inference...")

    if n_samples:

        def row_fn(msgs):
            return get_samples_single_row(
                client,
                model_name,
                msgs,
                n_samples,
                use_responses_api,
                use_anthropic_api,
            )
    else:

        def row_fn(msgs):
            return get_logits_single_row(
                client, model_name, msgs, use_responses_api, use_anthropic_api
            )

    # Process in chunks for checkpointing
    all_raw_responses = []
    for chunk_start in range(0, len(remaining_indices), BATCH_CHECKPOINT_INTERVAL):
        chunk_end = min(chunk_start + BATCH_CHECKPOINT_INTERVAL, len(remaining_indices))
        chunk_indices = remaining_indices[chunk_start:chunk_end]
        chunk_messages = remaining_messages[chunk_start:chunk_end]

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(
                tqdm(
                    executor.map(row_fn, chunk_messages),
                    total=len(chunk_messages),
                    desc=f"Rows {chunk_start}-{chunk_end}",
                )
            )

        if n_samples:
            for idx, (logprobs, raw) in zip(chunk_indices, results):
                df.at[idx, "model_logprobs"] = logprobs
                all_raw_responses.append(raw)
        else:
            for idx, logprobs in zip(chunk_indices, results):
                df.at[idx, "model_logprobs"] = logprobs

        completed_indices.update(chunk_indices)

        if checkpoint_path:
            _save_batch_checkpoint(df, checkpoint_path, list(completed_indices))

    if n_samples and raw_responses_path:
        with open(raw_responses_path, "w") as f:
            json.dump(all_raw_responses, f, indent=2)
        print(f"Raw responses saved to {raw_responses_path}")

    # Clean up checkpoint after successful completion
    if checkpoint_path:
        _cleanup_batch_checkpoint(checkpoint_path)

    return df.drop(columns=["chat_prompt"])
