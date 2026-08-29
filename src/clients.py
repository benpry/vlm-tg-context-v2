"""
API client construction shared by the evaluation scripts.
"""

import os

import anthropic
from google import genai
from openai import OpenAI

GEMINI_API_KEY_ENV_VARS = ["LANGCOG_GEMINI_API_KEY", "COCOLAB_GEMINI_API_KEY"]


def setup_client(api_base: str):
    """Build the client for an API base URL.

    Returns (client, use_responses_api, use_anthropic_api). Local vLLM servers
    (localhost) speak the OpenAI Chat Completions API and need no real key.
    """
    if "google" in api_base:
        api_keys = [os.getenv(var) for var in GEMINI_API_KEY_ENV_VARS if os.getenv(var)]
        if not api_keys:
            raise EnvironmentError(
                f"Set one of {GEMINI_API_KEY_ENV_VARS} to call the Gemini API."
            )
        return genai.Client(api_key=api_keys[0]), False, False
    if "anthropic" in api_base:
        return anthropic.Anthropic(), False, True
    if "localhost" in api_base or "127.0.0.1" in api_base:
        return OpenAI(base_url=api_base, api_key=os.getenv("OPENAI_API_KEY", "EMPTY")), False, False
    return OpenAI(base_url=api_base), "openai.com" in api_base, False
