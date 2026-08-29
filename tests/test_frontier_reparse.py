"""
Tests for rebuilding sampled (frontier) results from their stored raw responses.
"""

import math

import pandas as pd
import pytest

from src.frontier_reparse import (
    batch_raw_by_key,
    counts_from_logprobs,
    interactive_raw_by_key,
    legacy_extract_choice,
    logprobs_from_samples,
    prediction_from_logprobs,
    reparse,
)


def test_legacy_rule_is_the_last_capital_letter():
    assert legacy_extract_choice("J\nCorrect.") == "C"
    assert legacy_extract_choice("A") == "A"
    assert legacy_extract_choice("nothing") == "nothing"


def test_logprobs_from_samples_counts_valid_letters_over_all_samples():
    assert logprobs_from_samples(["A", "A", "B", "zzz"]) == {
        "A": math.log(2 / 4),
        "B": math.log(1 / 4),
    }
    assert logprobs_from_samples(["zzz", ""]) == {}


def test_counts_from_logprobs_inverts_the_stored_cell():
    assert counts_from_logprobs("{'A': 0.0}", 10) == {"A": 10}
    assert counts_from_logprobs({"A": math.log(0.3), "B": math.log(0.7)}, 10) == {"A": 3, "B": 7}
    assert counts_from_logprobs("{}", 10) == {}
    assert counts_from_logprobs(float("nan"), 10) == {}


def test_prediction_is_the_argmax_or_empty():
    assert prediction_from_logprobs({"A": math.log(0.3), "B": math.log(0.7)}) == "B"
    assert prediction_from_logprobs({}) == ""


def make_results():
    return pd.DataFrame(
        {
            "workerid": ["w1", "w1", "w2"],
            "matcher_trialNum": [0, 1, 0],
            "model_logprobs": ["{'C': 0.0}", "{'B': 0.0}", "{'A': 0.0}"],
            "model_prediction": ["C", "B", "A"],
        }
    )


def test_reparse_rebuilds_rows_whose_raw_responses_reproduce_the_stored_cell():
    # (w1, 0): the old rule read "J\nCorrect." as C; the new rule reads J.
    # (w1, 1): the raw texts say A but the file says B, so they are not this
    #          run's responses and the row must be left alone.
    # (w2, 0): reproduced and unchanged.
    raw = {
        ("w1", 0): ["J\nCorrect."] * 10,
        ("w1", 1): ["A"] * 10,
        ("w2", 0): ["A"] * 10,
    }
    df_new, report = reparse(
        make_results(), [raw], key_cols=["workerid", "matcher_trialNum"], session_col="workerid"
    )
    assert report.sourced == [("w1", 0), ("w2", 0)]
    assert report.unsourced == [("w1", 1)]
    assert report.changed == [("w1", 0)]
    assert report.prediction_changed == [("w1", 0)]
    assert report.cascade_sessions == {"w1"}
    assert df_new.loc[0, "model_logprobs"] == {"J": 0.0}
    assert df_new.loc[0, "model_prediction"] == "J"
    assert df_new.loc[1, "model_logprobs"] == "{'B': 0.0}"
    assert df_new.loc[2, "model_logprobs"] == {"A": 0.0}
    assert len(df_new) == 3
    assert "cascade" in report.summary()


def test_reparse_prefers_the_first_source_that_reproduces_a_row():
    stale = {("w2", 0): ["B"] * 10}  # an older run's responses for the same trial
    fresh = {("w2", 0): ["A"] * 10}
    _, report = reparse(make_results(), [stale, fresh], key_cols=["workerid", "matcher_trialNum"])
    assert ("w2", 0) in report.sourced
    assert report.cascade_sessions == set()


def test_reparse_requires_a_prediction_column_only_when_present():
    df = make_results().drop(columns=["model_prediction"])
    raw = {("w1", 0): ["J\nCorrect."] * 10}
    df_new, report = reparse(df, [raw], key_cols=["workerid", "matcher_trialNum"])
    assert "model_prediction" not in df_new.columns
    assert report.prediction_changed == [("w1", 0)]


def test_interactive_raw_by_key_maps_row_indices_to_prep_keys():
    df_prep = pd.DataFrame({"workerid": ["w1", "w1"], "matcher_trialNum": [0, 1]})
    raw = {"1": ["A"], "0": ["B"]}  # JSON keys are strings
    assert interactive_raw_by_key(raw, df_prep, ["workerid", "matcher_trialNum"]) == {
        ("w1", 0): ["B"],
        ("w1", 1): ["A"],
    }
    with pytest.raises(ValueError, match="index"):
        interactive_raw_by_key({"5": ["A"]}, df_prep, ["workerid", "matcher_trialNum"])


def test_batch_raw_by_key_requires_one_entry_per_prep_row():
    df_prep = pd.DataFrame({"gameId": ["g", "g"], "trialNum": [0, 1], "repNum": [0, 0]})
    assert batch_raw_by_key([["A"], ["B"]], df_prep, ["gameId", "trialNum", "repNum"]) == {
        ("g", 0, 0): ["A"],
        ("g", 1, 0): ["B"],
    }
    with pytest.raises(ValueError, match="rows"):
        batch_raw_by_key([["A"]], df_prep, ["gameId", "trialNum", "repNum"])


def test_reparse_can_verify_provenance_against_the_rule_the_file_was_made_with():
    """Files written on 27 Aug 2026 used the intermediate rule (extract_choice_v2), not the
    March last-letter rule; provenance must be checked with the rule that produced the cell."""
    from src.frontier_reparse import PREVIOUS_RULES, extract_choice_v2

    assert legacy_extract_choice("J\nCorrect.") == "C" and extract_choice_v2("J\nCorrect.") == "J"
    assert extract_choice_v2("I think it's F, because the Head is tilted") == "H"  # v2 still took the last capital
    df = pd.DataFrame({"workerid": ["w1"], "matcher_trialNum": [0], "model_logprobs": ["{'J': 0.0}"]})
    raw = {("w1", 0): ["J\nCorrect."] * 10}  # a cell produced by the v2 rule
    _, report_march = reparse(df, [raw], key_cols=["workerid", "matcher_trialNum"], previous_rule="march")
    _, report_v2 = reparse(df, [raw], key_cols=["workerid", "matcher_trialNum"], previous_rule="v2")
    assert report_march.unsourced == [("w1", 0)]
    assert report_v2.sourced == [("w1", 0)] and report_v2.changed == []
    assert set(PREVIOUS_RULES) == {"march", "v2"}
