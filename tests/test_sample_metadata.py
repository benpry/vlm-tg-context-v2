"""
Sampled frontier runs must record, next to each raw text, the finish reason
and the model version the API reported, so that later runs can be compared
(the endpoints behind undated model names change over time).
"""

import math
from types import SimpleNamespace

from src.frontier_reparse import logprobs_from_samples, sample_text
from src.lm import get_samples_single_row


class ChatClient:
    """A minimal OpenAI-style chat-completions client returning canned answers."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        text = self.answers.pop(0)
        return SimpleNamespace(
            model="some-model-2026-08-01",
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        )


def test_samples_carry_text_finish_reason_and_model():
    client = ChatClient(["A", "B", "A"])
    logprobs, raw = get_samples_single_row(client, "some-model", [{"role": "user", "content": "x"}], 3)
    assert logprobs == {"A": math.log(2 / 3), "B": math.log(1 / 3)}
    assert [r["text"] for r in raw] == ["A", "B", "A"]
    assert all(r["finish_reason"] == "stop" for r in raw)
    assert all(r["model"] == "some-model-2026-08-01" for r in raw)


def test_reparse_reads_both_old_and_new_raw_formats():
    assert sample_text("A") == "A"
    assert sample_text({"text": "B", "finish_reason": "stop", "model": "m"}) == "B"
    assert logprobs_from_samples([{"text": "A"}, "A", {"text": "zzz"}]) == {"A": math.log(2 / 3)}
