"""
Test getting logprobs from model responses
"""

import math
from collections import Counter

import numpy as np

from src.lm import _counts_to_logprobs, _extract_choice
from src.utils import get_logprobs_from_openai_choice


class MockTopLogprob:
    def __init__(self, token: str, logprob: float):
        self.token = token
        self.logprob = logprob


class MockTokenLogprobs:
    def __init__(self, top_logprobs: list):
        self.top_logprobs = top_logprobs


class MockLogprobs:
    def __init__(self, content: list):
        self.content = content


class MockChoice:
    def __init__(self, logprobs: list | None):
        if logprobs is None:
            self.logprobs = None
        else:
            content = [
                MockTokenLogprobs(
                    [
                        MockTopLogprob(lp["token"], lp["logprob"])
                        for lp in token_logprobs
                    ]
                )
                for token_logprobs in logprobs
            ]
            self.logprobs = MockLogprobs(content)


def test_logprobs_from_openai_choice():
    """
    Test getting logprobs from an OpenAI response.
    """

    # basic test
    choice = MockChoice(
        logprobs=[[{"token": "A", "logprob": -1}, {"token": "B", "logprob": -2}]]
    )

    logprobs = get_logprobs_from_openai_choice(choice, ["A", "B"])
    assert logprobs == {"A": -1, "B": -2}

    # test with multiple tokens per letter
    choice = MockChoice(
        logprobs=[
            [
                {"token": "A", "logprob": -1},
                {"token": "B", "logprob": -2},
                {"token": " A ", "logprob": -3},
            ],
        ]
    )
    logprobs = get_logprobs_from_openai_choice(choice, ["A", "B"])
    assert logprobs == {"A": np.logaddexp(-1, -3), "B": -2}

    # test with missing token
    logprobs = get_logprobs_from_openai_choice(choice, ["A", "B", "C"])
    assert logprobs == {"A": np.logaddexp(-1, -3), "B": -2}


def test_extract_choice():
    """
    Test extracting the choice from a model response.
    """

    # basic test
    choice = _extract_choice("A")
    assert choice == "A"

    # test with multiple tokens per letter
    choice = _extract_choice("The answer is A oh wait no it's B.")
    assert choice == "B"

    # test with no choice
    choice = _extract_choice("The answer is nothing.")
    assert choice == "The answer is nothing."


def test_counts_to_logprobs():
    """
    Test converting a Counter of choice frequencies to log-probabilities.
    """

    # basic test
    counts = Counter({"A": 1, "B": 2})
    logprobs = _counts_to_logprobs(counts, 3)
    assert logprobs == {"A": math.log(1 / 3), "B": math.log(2 / 3)}
