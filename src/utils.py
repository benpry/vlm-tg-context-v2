import base64
import json
import warnings
from ast import literal_eval
from io import BytesIO

import numpy as np


def get_user_message(messages):
    """
    Get the user message from a list of messages.
    """
    if not isinstance(messages, list):
        return ""

    user_message = ""
    for message in messages:
        if "text" not in message:
            warnings.warn(f"Message {message} is missing 'text' field.")
        else:
            user_message += f"{message['role']}: {message['text']}\n"

    return user_message


def add_user_message(chat_messages: list, user_message: str):
    if chat_messages and chat_messages[-1]["role"] == "user":
        chat_messages[-1]["content"] += "\n\n" + user_message
    else:
        chat_messages.append({"role": "user", "content": user_message})

    return chat_messages


def preprocess_messages(row):
    """
    Turn a row of the dataframe into a list of messages for the chat model.
    """
    chat_messages = []

    # parse the message history if it's a string, otherwise make sure it's a list
    message_history = row["message_history"]
    if isinstance(message_history, str):
        message_history = json.loads(message_history)
    elif not isinstance(message_history, list):
        raise ValueError(f"Message history is not a list: {message_history}")

    # parse the selection history if it's a string, otherwise make sure it's a list
    selection_history = row["selection_history"]
    if isinstance(selection_history, str):
        selection_history = json.loads(selection_history)
    elif not isinstance(selection_history, list):
        raise ValueError(f"Selection history is not a list: {selection_history}")

    # parse the correctness history if it's a string, otherwise make sure it's a list
    correctness_history = row["correctness_history"]
    if isinstance(correctness_history, str):
        correctness_history = json.loads(
            correctness_history.replace("True", "true").replace("False", "false")
        )
    elif not isinstance(correctness_history, list):
        raise ValueError(f"Correctness history is not a list: {correctness_history}")

    # make sure the lengths of the histories are the same
    if not (len(message_history) == len(selection_history) == len(correctness_history)):
        warnings.warn(
            f"Length of message_history, selection_history, and correctness_history must be the same. Got {len(message_history)}, {len(selection_history)}, and {len(correctness_history)}. game ID: {row['gameId']}"
        )

    # add the messages, selections, and correctness to the chat messages
    for messages, selection, correctness in zip(
        message_history, selection_history, correctness_history
    ):
        user_message = get_user_message(messages)
        chat_messages = add_user_message(chat_messages, user_message)
        chat_messages.append({"role": "assistant", "content": selection})
        chat_messages = add_user_message(
            chat_messages, "Correct." if correctness else "Incorrect."
        )

    this_trial_messages = row["message"]
    if not isinstance(this_trial_messages, str):
        chat_messages = add_user_message(chat_messages, "describer: \n")
    else:
        this_trial_messages = literal_eval(this_trial_messages)
        chat_messages = add_user_message(
            chat_messages, get_user_message(this_trial_messages)
        )

    return chat_messages


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_openai_messages(
    system_prompt,
    chat_prompt,
    include_image,
    grid_image,
    model_name="",
    use_responses_api=False,
    base64_image=None,
):
    # Copy chat_prompt to avoid mutating original
    chat_messages = [msg.copy() for msg in chat_prompt]
    text_block_type = "input_text" if use_responses_api else "text"

    # Molmo2 models don't support a system role, so we prepend the system
    # instruction to the first user message instead.
    if "Molmo2" in model_name:
        messages = []
        # Prepend system prompt to first user message
        for i, msg in enumerate(chat_messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    chat_messages[i]["content"] = f"{system_prompt}\n{msg['content']}"
                elif isinstance(msg["content"], list):
                    # If content is already a list, prepend system prompt as text
                    chat_messages[i]["content"] = [
                        {"type": text_block_type, "text": system_prompt},
                        *msg["content"],
                    ]
                break
    else:
        messages = [{"role": "system", "content": system_prompt}]

    if include_image and chat_messages:
        # Find first user message
        first_user_idx = -1
        for i, msg in enumerate(chat_messages):
            if msg["role"] == "user":
                first_user_idx = i
                break

        if first_user_idx != -1:
            content = chat_messages[first_user_idx]["content"]
            if base64_image is None:
                base64_image = encode_image(grid_image)
            image_data_url = f"data:image/png;base64,{base64_image}"

            if isinstance(content, list):
                # Content is already a list (e.g. system prompt was prepended);
                # insert the image at the beginning.
                if use_responses_api:
                    image_part = {"type": "input_image", "image_url": image_data_url}
                else:
                    image_part = {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                new_content = [image_part, *content]
            else:
                if use_responses_api:
                    image_part = {"type": "input_image", "image_url": image_data_url}
                else:
                    image_part = {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                new_content = [image_part, {"type": text_block_type, "text": content}]
            chat_messages[first_user_idx]["content"] = new_content

    messages.extend(chat_messages)
    return messages


def get_logprobs_from_openai_choice(choice, choice_tokens):
    if not choice.logprobs or not choice.logprobs.content:
        return {}

    first_token_logprobs = choice.logprobs.content[0]
    top_logprobs = first_token_logprobs.top_logprobs

    choice_logprobs = {}
    for top_lp in top_logprobs:
        token = top_lp.token.strip()
        if token in choice_tokens:
            if token in choice_logprobs:
                choice_logprobs[token] = float(
                    np.logaddexp(choice_logprobs[token], top_lp.logprob)
                )
            else:
                choice_logprobs[token] = top_lp.logprob

    # send a warning if not all the choice tokens are in the top logprobs
    if not all(token in choice_logprobs for token in choice_tokens):
        warnings.warn("Not all choice tokens found in top logprobs.")

    return choice_logprobs


def get_logprobs_from_responses_api(response, choice_tokens):
    """Parse logprobs from an OpenAI Responses API response."""
    if not response.output or not response.output[0].content:
        return {}

    content_item = response.output[0].content[0]
    if not hasattr(content_item, "logprobs") or not content_item.logprobs:
        return {}

    first_token_logprobs = content_item.logprobs[0]
    top_logprobs = first_token_logprobs.top_logprobs

    choice_logprobs = {}
    for top_lp in top_logprobs:
        token = top_lp.token.strip()
        if token in choice_tokens:
            if token in choice_logprobs:
                choice_logprobs[token] = float(
                    np.logaddexp(choice_logprobs[token], top_lp.logprob)
                )
            else:
                choice_logprobs[token] = top_lp.logprob

    if not all(token in choice_logprobs for token in choice_tokens):
        warnings.warn("Not all choice tokens found in top logprobs.")

    return choice_logprobs


def get_logprobs_from_genai_response(response, choice_tokens):
    candidates = response.candidates
    if not candidates or not candidates[0].logprobs_result:
        return {}
    top_candidates = candidates[0].logprobs_result.top_candidates
    if not top_candidates:
        return {}
    first_token_candidates = top_candidates[0].candidates
    choice_logprobs = {}
    for candidate in first_token_candidates:
        token = candidate.token.strip()
        if token in choice_tokens:
            if token in choice_logprobs:
                choice_logprobs[token] = float(
                    np.logaddexp(choice_logprobs[token], candidate.log_probability)
                )
            else:
                choice_logprobs[token] = candidate.log_probability
    if not all(token in choice_logprobs for token in choice_tokens):
        warnings.warn("Not all choice tokens found in top logprobs.")
    return choice_logprobs


from google.genai import types


def convert_to_google_genai_style(messages):
    """
    Convert OpenAI-style messages to Google GenAI-style contents.
    Returns a tuple (contents, system_instruction).
    """
    contents = []
    system_instruction = None

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            if isinstance(content, str):
                system_instruction = content
            elif isinstance(content, list):
                # Handle list content for system prompt if necessary (e.g. text only)
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part["text"])
                if parts:
                    system_instruction = "\n".join(parts)
            continue

        parts = []
        if isinstance(content, str):
            parts.append(types.Part(text=content))
        elif isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    parts.append(types.Part(text=part["text"]))
                elif part["type"] == "image_url":
                    image_url = part["image_url"]["url"]
                    if image_url.startswith("data:"):
                        # Extract mime_type and base64 data
                        header, data = image_url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type, data=base64.b64decode(data)
                                )
                            )
                        )
                    else:
                        # Handle regular URLs if needed, but for now we skip or just pass typical cases
                        pass

        if role == "assistant":
            role = "model"

        contents.append(types.Content(role=role, parts=parts))

    return contents, system_instruction


def convert_to_anthropic_format(messages):
    """Convert OpenAI-format messages to Anthropic API format.

    Returns (system_prompt, anthropic_messages) where system_prompt is extracted
    from the system role message and image content blocks are converted.
    """
    system_prompt = ""
    anthropic_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
            continue

        converted = {"role": msg["role"]}
        content = msg["content"]

        if isinstance(content, str):
            converted["content"] = content
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if block.get("type") in {"image_url", "input_image"}:
                    # Convert OpenAI image_url to Anthropic image format
                    image_url = block["image_url"]
                    url = image_url["url"] if isinstance(image_url, dict) else image_url
                    # Extract base64 data from data URL
                    base64_data = url.split("base64,", 1)[1]
                    new_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_data,
                            },
                        }
                    )
                elif block.get("type") in {"text", "input_text"}:
                    new_blocks.append({"type": "text", "text": block["text"]})
                else:
                    new_blocks.append(block)
            converted["content"] = new_blocks

        anthropic_messages.append(converted)

    return system_prompt, anthropic_messages
