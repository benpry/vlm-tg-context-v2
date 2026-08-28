"""
Tests for merging rerun sessions back into an interactive results file.
"""

import pandas as pd
import pytest

from src.session_rerun import merge_rerun_sessions


def make_results():
    return pd.DataFrame(
        {
            "workerid": ["w1", "w1", "w2", "w2"],
            "trialNum": [0, 1, 0, 1],
            "model_logprobs": ["{'A': 0.0}"] * 4,
        }
    )


def test_merge_replaces_exactly_the_rerun_sessions():
    rerun = pd.DataFrame(
        {"workerid": ["w2", "w2"], "trialNum": [0, 1], "model_logprobs": [{"B": 0.0}] * 2}
    )
    merged = merge_rerun_sessions(make_results(), rerun, {"w2"})
    assert len(merged) == 4
    assert merged.loc[merged.workerid == "w1", "model_logprobs"].tolist() == ["{'A': 0.0}"] * 2
    assert merged.loc[merged.workerid == "w2", "model_logprobs"].tolist() == [{"B": 0.0}] * 2


def test_merge_rejects_rerun_rows_from_other_sessions():
    rerun = pd.DataFrame({"workerid": ["w2", "w3"], "trialNum": [0, 0], "model_logprobs": [{}] * 2})
    with pytest.raises(ValueError, match="sessions"):
        merge_rerun_sessions(make_results(), rerun, {"w2"})


def test_merge_rejects_a_session_with_a_different_number_of_rows():
    rerun = pd.DataFrame({"workerid": ["w2"], "trialNum": [0], "model_logprobs": [{}]})
    with pytest.raises(ValueError, match="rows"):
        merge_rerun_sessions(make_results(), rerun, {"w2"})


def test_merge_rejects_sessions_missing_from_the_results():
    rerun = pd.DataFrame({"workerid": ["w9"], "trialNum": [0], "model_logprobs": [{}]})
    with pytest.raises(ValueError, match="not in the results"):
        merge_rerun_sessions(make_results(), rerun, {"w9"})
