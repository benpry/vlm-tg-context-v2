"""
Tests for where call_lm.py writes result files.
"""

from pathlib import Path

from src.output_paths import raw_responses_path, results_path

PREP = Path("/repo/context_prep")
OUT = Path("/repo/data/logprobs")
MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def test_full_feedback_results_mirror_the_prep_layout():
    assert results_path(PREP / "full_feedback/yoked.csv", MODEL, PREP, OUT) == (
        OUT / "full_feedback/yoked_Llama-3.2-11B-Vision-Instruct_logprobs.csv"
    )
    assert results_path(PREP / "full_feedback/yoked.csv", MODEL, PREP, OUT, no_image=True) == (
        OUT / "full_feedback/yoked_Llama-3.2-11B-Vision-Instruct_logprobs_no_image.csv"
    )


def test_human_history_goes_to_human_yoked_or_interactive():
    prep = PREP / "human_history/limited_feedback_yoked.csv"
    assert results_path(prep, MODEL, PREP, OUT, yoked=True).parent == OUT / "human_yoked"
    assert results_path(prep, MODEL, PREP, OUT, interactive=True).parent == OUT / "interactive"
    practice = PREP / "practice/r6.csv"
    assert results_path(practice, MODEL, PREP, OUT, interactive=True).parent == OUT / "practice"


def test_a_custom_output_root_keeps_the_archive_untouched():
    other = Path("/juice2/llama-rerun")
    path = results_path(PREP / "full_feedback/yoked.csv", MODEL, PREP, other)
    assert path.parent == other / "full_feedback"


def test_raw_responses_sit_next_to_the_results_root():
    results = OUT / "frontier/no_context_gpt-5.2_logprobs.csv"
    assert raw_responses_path(results, OUT, OUT, Path("/repo/data/raw_responses")) == Path(
        "/repo/data/raw_responses/frontier/no_context_gpt-5.2_logprobs.json"
    )
    other = Path("/juice2/llama-rerun")
    assert raw_responses_path(other / "full_feedback/x_logprobs.csv", other, OUT, Path("/repo/data/raw_responses")) == Path(
        "/juice2/llama-rerun_raw_responses/full_feedback/x_logprobs.json"
    )
