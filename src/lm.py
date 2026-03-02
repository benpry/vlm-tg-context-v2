"""
Code for calling the language model to get choice logits
"""

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
import tiktoken
from google.genai import types
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from src.utils import (
    convert_to_google_genai_style,
    encode_image,
    get_logprobs_from_genai_response,
    get_logprobs_from_openai_choice,
    get_openai_messages,
    preprocess_messages,
)

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose a shape from a set of shapes. Your goal is to guess which of the shapes the describer is trying to get the matcher to choose. The shapes, with their labels, are shown in the image.
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose, and no other text. You will receive feedback telling you whether your choice was correct or incorrect.
"""


# @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def get_completion_with_backoff(client, model, messages):
    if "gemini" in model.lower():
        # use the google genai client
        genai_messages, system_instruction = convert_to_google_genai_style(messages)
        return client.models.generate_content(
            model=model,
            contents=genai_messages,
            config=types.GenerateContentConfig(
                response_logprobs=True,
                logprobs=20,
                temperature=1,
                system_instruction=system_instruction,
            ),
        )
    else:
        # we're using an openai-style client
        if "claude" in model.lower():
            # add cache control to the last message
            messages[-1]["cache_control"] = {
                "type": "ephemeral",
            }
            n_logprobs = 20
        else:
            n_logprobs = 1000

        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=1,
            logprobs=True,
            top_logprobs=n_logprobs,
        )


def get_logits_single_row(
    client: OpenAI,
    model_name: str,
    messages: list,
) -> dict:
    response = get_completion_with_backoff(
        client=client,
        model=model_name,
        messages=messages,
    )
    if "gemini" in model_name.lower():
        return get_logprobs_from_genai_response(response, CHOICES)
    else:
        return get_logprobs_from_openai_choice(response.choices[0], CHOICES)


def _get_single_sample(client, model_name, messages):
    """Make a single API call and return the generated token."""
    response = get_completion_with_backoff(client, model_name, messages)
    if "gemini" in model_name.lower():
        return response.text.strip()
    else:
        return response.choices[0].message.content.strip()


def _counts_to_logprobs(counts, n_samples):
    """Convert a Counter of choice frequencies to log-probabilities."""
    return {choice: math.log(count / n_samples) for choice, count in counts.items()}


def get_samples_single_row(client, model_name, messages, n_samples):
    """Resample n_samples times and return frequency-based log-probabilities."""
    if "gemini" not in model_name.lower() and "claude" not in model_name.lower():
        # OpenAI models support n parameter — single call, multiple completions
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1,
            temperature=1,
            n=n_samples,
        )
        tokens = [choice.message.content.strip() for choice in response.choices]
    else:
        # Claude and Gemini: sequential calls per row (cross-row parallelism handles throughput)
        tokens = [
            _get_single_sample(client, model_name, messages) for _ in range(n_samples)
        ]

    counts = Counter(t for t in tokens if t in CHOICES)
    return _counts_to_logprobs(counts, n_samples) if counts else {}


REQUIRED_COLUMNS = [
    "message_history",
    "selection_history",
    "correctness_history",
    "message",
]


def get_logits(
    df: pd.DataFrame,
    model_name: str,
    client: OpenAI,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
    n_samples: Optional[int] = None,
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

    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

    print("Preparing messages...")
    all_messages = [
        get_openai_messages(
            SYSTEM_PROMPT, chat_prompt, include_image, grid_image, model_name
        )
        for chat_prompt in df["chat_prompt"]
    ]

    print("Doing inference...")

    def row_fn(msgs):
        if n_samples:
            return get_samples_single_row(client, model_name, msgs, n_samples)
        return get_logits_single_row(client, model_name, msgs)

    with ThreadPoolExecutor(max_workers=20) as executor:
        all_choice_logprobs = list(
            tqdm(
                executor.map(row_fn, all_messages),
                total=len(all_messages),
            )
        )

    df["model_logprobs"] = all_choice_logprobs

    return df.drop(columns=["chat_prompt"])


def _get_encoding(model_name: str):
    """Get a tiktoken encoding, falling back to cl100k_base for unknown models."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_message_tokens(messages: list, encoding) -> int:
    """Count text tokens in an OpenAI-style message list (skipping image data)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(encoding.encode(content))
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(encoding.encode(part["text"]))
                # image_url parts are counted separately
    return total


def _estimate_image_tokens(grid_image: Image.Image) -> int:
    """Estimate token count for a base64-encoded PNG image.

    Uses a rough heuristic: base64 bytes * 3/4 (to get raw bytes) / 768 tiles,
    ~170 tokens per tile. This is approximate and varies by provider.
    """
    base64_str = encode_image(grid_image)
    raw_bytes = len(base64_str) * 3 / 4
    n_tiles = max(1, raw_bytes / 768)
    return int(n_tiles * 170)


def _count_chat_prompt_tokens(chat_prompt: list, encoding) -> int:
    """Count text tokens in a preprocessed chat prompt (list of role/content dicts)."""
    total = 0
    for msg in chat_prompt:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(encoding.encode(content))
    return total


def count_tokens(
    df: pd.DataFrame,
    model_name: str,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> dict:
    """Count input/output tokens without calling the API.

    Counts tokens directly from preprocessed chat prompts plus the system prompt,
    avoiding expensive per-row image encoding.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}.")

    if n_trials is not None:
        df = df.head(n_trials)

    encoding = _get_encoding(model_name)
    system_prompt_tokens = len(encoding.encode(SYSTEM_PROMPT))
    image_tokens = _estimate_image_tokens(grid_image) if include_image else 0

    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

    text_token_counts = [
        system_prompt_tokens + _count_chat_prompt_tokens(chat_prompt, encoding)
        for chat_prompt in df["chat_prompt"]
    ]

    n_rows = len(text_token_counts)
    total_input = sum(text_token_counts) + (image_tokens * n_rows)
    total_output = n_rows  # max_tokens=1 per row

    return {
        "n_rows": n_rows,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "image_tokens_per_row": image_tokens,
        "text_tokens_per_row": {
            "min": min(text_token_counts),
            "max": max(text_token_counts),
            "mean": sum(text_token_counts) / len(text_token_counts),
        },
    }
