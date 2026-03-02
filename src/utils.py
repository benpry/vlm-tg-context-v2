import base64
import warnings
from ast import literal_eval
from io import BytesIO

import numpy as np


def get_messages(system_prompt, chat_prompt, include_image, grid_image, model_name):
    messages = []
    if "gemma" in model_name.lower() or "molmo" in model_name.lower():
        # gemma and molmo models don't have a system role, so we add the instruction to the first user message
        messages = [*chat_prompt]
        if include_image:
            messages[0]["content"] = [
                {"type": "image", "image": grid_image},
                {"type": "text", "text": f"{system_prompt}\n{messages[0]['content']}"},
            ]
        else:
            messages[0]["content"] = f"{system_prompt}\n{messages[0]['content']}"
    elif "llama" in model_name.lower():
        # llama models don't take images in the system prompt, so we add the image to the first user message
        messages = [
            {"role": "system", "content": system_prompt},
            *chat_prompt,
        ]
        if include_image:
            messages[1]["content"] = [
                {"type": "image", "image": grid_image},
                {"type": "text", "text": messages[1]["content"]},
            ]
    else:
        # otherwise the image goes in the system prompt
        if include_image:
            system_content = [
                {"type": "image", "image": grid_image},
                {"type": "text", "text": system_prompt},
            ]
        else:
            system_content = system_prompt

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            *chat_prompt,
        ]

    return messages


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
    message_history = row["message_history"]
    if isinstance(message_history, str):
        message_history = literal_eval(message_history.replace("nan", "''"))
    elif not isinstance(message_history, list):
        warnings.warn(f"Message history is not a list: {message_history}")
        message_history = []

    selection_history = row["selection_history"]
    if isinstance(selection_history, str):
        selection_history = literal_eval(
            selection_history.replace("null", '"no response"')
        )
    elif not isinstance(selection_history, list):
        warnings.warn(f"Selection history is not a list: {selection_history}")
        selection_history = []

    correctness_history = row["correctness_history"]
    if isinstance(correctness_history, str):
        correctness_history = literal_eval(
            correctness_history.replace("true", "True").replace("false", "False")
        )
    elif not isinstance(correctness_history, list):
        warnings.warn(f"Correctness history is not a list: {correctness_history}")
        correctness_history = []

    if not (len(message_history) == len(selection_history) == len(correctness_history)):
        warnings.warn(
            f"Length of message_history, selection_history, and correctness_history must be the same. Got {len(message_history)}, {len(selection_history)}, and {len(correctness_history)}. game ID: {row['gameId']}"
        )

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
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages = add_user_message(
            chat_messages, get_user_message(this_trial_messages)
        )

    return chat_messages


def get_logprobs_from_outputs(outputs, choice_tokens):
    """
    Get the log probabilities of the choice tokens from the model outputs.
    """
    all_choice_logprobs = []
    for output in outputs:
        choice_logprobs = {}
        all_choice_logprobs.append(choice_logprobs)
        logprobs = output.outputs[0].logprobs[0].values()
        for logprob in logprobs:
            if logprob.decoded_token.strip() in choice_tokens:
                choice_logprobs[logprob.decoded_token.strip()] = logprob.logprob
            else:
                print(
                    f"Found high-probability non-letter token: {logprob.decoded_token.strip()} with logprob {logprob.logprob}"
                )
            if len(choice_logprobs) == len(choice_tokens):
                break

        if len(choice_logprobs) < len(choice_tokens):
            warnings.warn("Not all choice tokens found in top logprobs.")

    return all_choice_logprobs


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_openai_messages(
    system_prompt, chat_prompt, include_image, grid_image, model_name=""
):
    # Copy chat_prompt to avoid mutating original
    chat_messages = [msg.copy() for msg in chat_prompt]

    # Molmo2 models don't support a system role, so we prepend the system
    # instruction to the first user message instead.
    if "Molmo2" in model_name:
        messages = []
        # Prepend system prompt to first user message
        for i, msg in enumerate(chat_messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    chat_messages[i]["content"] = (
                        f"{system_prompt}\n{msg['content']}"
                    )
                elif isinstance(msg["content"], list):
                    # If content is already a list, prepend system prompt as text
                    chat_messages[i]["content"] = [
                        {"type": "text", "text": system_prompt},
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
            base64_image = encode_image(grid_image)

            if isinstance(content, list):
                # Content is already a list (e.g. system prompt was prepended);
                # insert the image at the beginning.
                new_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    *content,
                ]
            else:
                new_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": content},
                ]
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
