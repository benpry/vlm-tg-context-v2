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


def test_extract_choice_drops_hallucinated_next_turn():
    """
    Frontier models sometimes answer and then keep writing the conversation
    ("J\\nCorrect."). Everything from such a turn marker on is not the answer,
    so the last-letter rule must not pick "C" from "Correct.".
    """
    assert _extract_choice("J\nCorrect.") == "J"
    assert _extract_choice("Dbottom\nCorrect.") == "D"
    assert _extract_choice("J府\nuser\nIncorrect.\n\ndescriber: diamond head") == "J"
    assert _extract_choice("bottom_left_of_set_B\nIncorrect.\n\ndescriber: the first one") == "B"
    # a hallucinated turn with no answer in front of it is invalid, not "H" or "C"
    assert _extract_choice("Human: Correct.") not in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def test_extract_choice_answer_first():
    """
    A response that starts with a choice letter glued to punctuation, an
    uppercase letter or non-Latin text is the answer followed by junk.
    """
    assert _extract_choice("J.") == "J"
    assert _extract_choice("J探J") == "J"
    assert _extract_choice("JCheck values\n正") == "J"
    assert _extract_choice("D正面答案是 D") == "D"
    assert _extract_choice("F, the one on the left") == "F"
    assert _extract_choice("FI") == "F"
    assert _extract_choice("LL") == "L"
    assert _extract_choice("D保育士B") == "D"


def test_extract_choice_reasoning_first():
    """
    A leading letter that starts an English word ("Looking", "I'll", "Ithink")
    or a sentence ("I think", "A person ...") is not the answer; the last
    letter mentioned is.
    """
    assert _extract_choice("Looking at the remaining option, that would be F.") == "F"
    assert _extract_choice("I'll go with F.") == "F"
    assert _extract_choice("Ithink the answer is J.") == "J"
    assert _extract_choice("I think it is F") == "F"
    assert _extract_choice("A person sitting on the ground. Options F and G fit. G looks like a person") == "G"


def test_extract_choice_invalid_responses_unchanged():
    assert _extract_choice("f") == "f"
    assert _extract_choice("S") == "S"
    assert _extract_choice("") == ""


def test_logprobs_from_openai_choice_rejects_masked_logprobs():
    """
    vLLM clamps -inf to -9999.0; a -inf logprob means the server masked the
    token (top-k / top-p) before computing logprobs, so the distribution is
    not the model's. That must fail loudly instead of being stored.
    """
    import pytest

    choice = MockChoice(
        logprobs=[
            [
                {"token": "A", "logprob": -0.5},
                {"token": "B", "logprob": -9999.0},
            ]
        ]
    )
    with pytest.raises(ValueError, match="masked"):
        get_logprobs_from_openai_choice(choice, ["A", "B"])


def test_local_logprob_requests_disable_nucleus_sampling():
    """
    vLLM fills any sampling parameter the request leaves unset from the
    model's generation_config.json (Llama 3.2: top_p 0.9), and the V0 engine
    reports logprobs after that masking. Requests must pin top_p / top_k.
    """
    from src.lm import get_completion_with_backoff

    class RecordingCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return "response"

    class RecordingClient:
        def __init__(self):
            self.chat = type("Chat", (), {})()
            self.chat.completions = RecordingCompletions()

    client = RecordingClient()
    response = get_completion_with_backoff(
        client, "some/local-model", [{"role": "user", "content": "hi"}], use_logprobs=True
    )
    assert response == "response"
    kwargs = client.chat.completions.kwargs
    assert kwargs["top_p"] == 1.0
    assert kwargs["extra_body"]["top_k"] == -1
    assert kwargs["logprobs"] is True


def test_extract_choice_explicit_answer_phrases():
    """A stated answer beats any letter that happens to follow it."""
    assert _extract_choice("I think it's F, because the Head is tilted down.") == "F"
    assert _extract_choice("The answer is F. Both arms are out.") == "F"
    assert _extract_choice("Looking at the remaining option, that would be F. The Diamond head fits.") == "F"
    assert _extract_choice("I'll go with F") == "F"
    assert _extract_choice("I\n\nCorrect Answer: I") == "I"
    assert _extract_choice("Looking at the clues:\n- Rabbit lying down\n\n**K**") == "K"
    assert _extract_choice("I thought K was already used. Let me reconsider - C.") == "C"


def test_extract_choice_leading_letter_with_explanation():
    """A leading letter followed by a dash, bracket, colon or newline is the answer."""
    assert _extract_choice("F - the kneeling one with the Diamond head") == "F"
    assert _extract_choice("F (the one facing left). The Head is round.") == "F"
    assert _extract_choice("J\nB") == "J"
    assert _extract_choice("C was already correct. Let me reconsider the remaining shapes.\n\nD") == "D"


def test_extract_choice_pronoun_i_is_not_an_answer():
    """'I' used as a pronoun ('I think', "I'll") must not be read as the letter I."""
    assert _extract_choice("human, I'll continue to answer these") not in CHOICES_LIST
    assert _extract_choice("human, your messages don't include a question for me to answer. I think") not in CHOICES_LIST
    assert _extract_choice("human, I'll continue answering as requested. \n\nE") == "E"
    assert _extract_choice("F, I think") == "F"
    assert _extract_choice("I") == "I"


def test_extract_choice_standalone_letter_beats_letters_inside_words():
    assert _extract_choice("Based on the podium description, F. Head tilted, Both legs out.") == "F"
    assert _extract_choice("Looking at each option:\n\nA - Has angular pieces\nB - Kneeling figure\n\nF") == "F"


CHOICES_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def test_extract_choice_leading_letter_before_bracket_or_stray_apostrophe():
    assert _extract_choice("I [internal progress: \nA Correct\nB Incorrect\nL") == "I"
    assert _extract_choice("G'\nerror\nInvalid format. Please respond with only the letter.") == "G"
    assert _extract_choice("I'_") == "I"
    assert _extract_choice("I'll go with F") == "F"  # a real contraction is still a word
