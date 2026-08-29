"""
Tests for the banana-rerun helpers: which rows/sessions were affected by the
banana -> ba''a bug, and whether a rerun actually replaced them.
"""

import pandas as pd
import pytest

from src.banana_rerun import (
    affected_rows,
    affected_sessions,
    changed_rows,
    changed_sessions,
    compare_rerun,
    compare_rerun_files,
    has_banana,
)


def make_prep(sessions_with_banana):
    """Three sessions x two trials; the given sessions mention banana in trial 1."""
    rows = []
    for worker in ["w1", "w2", "w3"]:
        for trial in [0, 1]:
            text = "looks like a person"
            if worker in sessions_with_banana and trial == 1:
                text = "banana man again"
            rows.append(
                {
                    "workerid": worker,
                    "gameId": "g1",
                    "matcher_trialNum": trial,
                    "trialNum": trial,
                    "repNum": 0,
                    "message": f'[{{"role":"describer","text":"{text}"}}]',
                    "message_history": "[]",
                }
            )
    return pd.DataFrame(rows)


def make_results(df_prep, logprob_by_session):
    df = df_prep.copy()
    df["model_logprobs"] = df["workerid"].map(logprob_by_session)
    return df


def test_has_banana_checks_message_and_history():
    df = pd.DataFrame(
        {
            "message": ["a banana", "nothing", "nothing"],
            "message_history": ["[]", "[Banana man]", "[]"],
        }
    )
    assert has_banana(df).tolist() == [True, True, False]


def test_affected_rows_and_sessions():
    df_prep = make_prep({"w2"})
    assert affected_sessions(df_prep) == {"w2"}
    assert affected_rows(df_prep, ["gameId", "trialNum", "repNum"]) == {("g1", 1, 0)}


def test_changed_sessions_ignores_row_order_and_nan():
    df_prep = make_prep({"w2"})
    backup = make_results(df_prep, {"w1": "{'A': -0.1}", "w2": "{'A': -0.1}", "w3": None})
    results = make_results(df_prep, {"w1": "{'A': -0.1}", "w2": "{'B': -0.2}", "w3": None})
    results = results.iloc[::-1].reset_index(drop=True)  # merged files are reordered
    assert changed_sessions(results, backup) == {"w2"}
    assert changed_rows(results, backup, ["workerid", "trialNum"]) == {
        ("w2", 0),
        ("w2", 1),
    }


def test_changed_rows_fails_loudly_if_rows_were_dropped():
    df_prep = make_prep(set())
    backup = make_results(df_prep, {"w1": "x", "w2": "x", "w3": "x"})
    results = backup.iloc[:-1]
    with pytest.raises(ValueError, match="same set of rows"):
        changed_rows(results, backup, ["workerid", "trialNum"])


def test_compare_rerun_complete_interactive():
    df_prep = make_prep({"w2"})
    backup = make_results(df_prep, {"w1": "x", "w2": "x", "w3": "x"})
    results = make_results(df_prep, {"w1": "x", "w2": "y", "w3": "x"})
    report = compare_rerun(df_prep, results, backup, mode="interactive")
    assert report.affected == {"w2"}
    assert report.changed == {"w2"}
    assert report.complete


def test_compare_rerun_incomplete_batch():
    # batch files (no_context) have one row per (gameId, trialNum, repNum), no sessions
    df_prep = make_prep({"w2"}).query("workerid == 'w2'").drop(columns="workerid")
    backup = df_prep.assign(model_logprobs="x")
    report = compare_rerun(df_prep, backup, backup, mode="batch")
    assert report.affected == {("g1", 1, 0)}
    assert report.changed == set()
    assert report.affected_not_changed == {("g1", 1, 0)}
    assert not report.complete


def test_compare_rerun_files_roundtrip(tmp_path):
    df_prep = make_prep({"w3"})
    backup = make_results(df_prep, {"w1": "x", "w2": "x", "w3": "x"})
    results = make_results(df_prep, {"w1": "x", "w2": "x", "w3": "z"})
    df_prep.to_csv(tmp_path / "prep.csv", index=False)
    backup.to_csv(tmp_path / "results.csv.bak", index=False)
    results.to_csv(tmp_path / "results.csv", index=False)
    report = compare_rerun_files(
        tmp_path / "prep.csv",
        tmp_path / "results.csv",
        tmp_path / "results.csv.bak",
        mode="interactive",
    )
    assert report.complete
    assert report.changed == {"w3"}


def test_all_nans_in_yoked_prep_are_bananas():
    """Every 'nan' substring in the yoked prep file comes from the word banana,
    so has_banana identifies exactly the rows the bug corrupted."""
    df = pd.read_csv("context_prep/human_history/limited_feedback_yoked.csv")
    nan_mask = df["message"].str.contains("nan", case=False, na=False) | df[
        "message_history"
    ].str.contains("nan", case=False, na=False)
    assert (nan_mask == has_banana(df)).all()


# --- comparing two versions of the same results file -------------------------

import math

import numpy as np

from src.banana_rerun import (
    compare_result_versions,
    compare_result_versions_files,
    normalised_probs,
    probability_shift,
)


def lp(**kwargs):
    """Python-dict-literal logprob cell, as written by the pipeline."""
    return str({k: math.log(v) for k, v in kwargs.items()})


def test_normalised_probs_parses_normalises_and_handles_empty():
    df = pd.DataFrame(
        {"model_logprobs": [lp(A=0.5, B=0.5), "{'A': -9999, 'C': 0.0}", "{}", np.nan]}
    )
    probs = normalised_probs(df)
    assert probs.shape == (4, 12)
    np.testing.assert_allclose(probs[0, :2], [0.5, 0.5])
    np.testing.assert_allclose(probs[1, 2], 1.0)
    assert probs[2].sum() == 0 and probs[3].sum() == 0


def make_versions():
    """Two versions of a batch results file: trial 1 mentions banana and changed."""
    base = pd.DataFrame(
        {
            "gameId": ["g1"] * 3,
            "trialNum": [0, 1, 2],
            "repNum": [0, 0, 0],
            "message": ["a person", "banana man", "a bird"],
            "message_history": ["[]", "[]", "[]"],
        }
    )
    old = base.assign(model_logprobs=[lp(A=0.9, B=0.1), lp(A=0.9, B=0.1), lp(C=1.0)])
    new = base.assign(model_logprobs=[lp(A=0.9, B=0.1), lp(A=0.2, B=0.8), lp(C=1.0)])
    return old, new.iloc[::-1].reset_index(drop=True)  # different row order


def test_probability_shift_aligns_rows_by_key():
    old, new = make_versions()
    shift = probability_shift(old, new)
    by_trial = shift.set_index("trialNum")["max_abs_diff"]
    assert by_trial[0] == 0 and by_trial[2] == 0
    assert math.isclose(by_trial[1], 0.7)
    assert shift.set_index("trialNum")["affected"].to_dict() == {0: False, 1: True, 2: False}


def test_compare_result_versions_groups_affected_vs_unaffected():
    old, new = make_versions()
    report = compare_result_versions(old, new)
    stats = report.stats.set_index("group")
    assert stats.loc["affected row", "n"] == 1
    assert stats.loc["unaffected row", "n"] == 2
    assert stats.loc["unaffected row", "max"] == 0
    assert math.isclose(stats.loc["affected row", "mean"], 0.7)
    assert "affected" in report.verdict


def test_compare_result_versions_identical_files():
    old, _ = make_versions()
    assert compare_result_versions(old, old).verdict.startswith("IDENTICAL")


def test_compare_result_versions_uses_sessions_for_interactive_files():
    old, new = make_versions()
    # interactive files have a session column and model_prediction; a session with
    # a banana trial may legitimately change on its later, unaffected trials too
    old = old.assign(workerid=["w1", "w1", "w2"], model_prediction="A")
    new = new.assign(
        workerid=new["trialNum"].map({0: "w1", 1: "w1", 2: "w2"}), model_prediction="A"
    )
    report = compare_result_versions(old, new)
    groups = set(report.stats["group"])
    assert groups == {"affected row", "unaffected row in affected session", "unaffected session"}


def test_compare_result_versions_files_roundtrip(tmp_path):
    old, new = make_versions()
    old.to_csv(tmp_path / "old.csv", index=False)
    new.to_csv(tmp_path / "new.csv", index=False)
    report = compare_result_versions_files(tmp_path / "old.csv", tmp_path / "new.csv")
    assert report.stats.set_index("group").loc["affected row", "n"] == 1


# --- probe: replay archived trials with today's code ---------------------------

from src.banana_rerun import run_probe, select_probe_rows


def make_archive():
    """An archived batch file: 2 rows with banana in the message, 2 with banana
    only in the history, 4 unaffected."""
    rows = []
    for i in range(8):
        message = "banana man" if i < 2 else "a person"
        history = '[[{"text":"banana"}]]' if 2 <= i < 4 else "[]"
        rows.append(
            {"gameId": "g1", "trialNum": i, "repNum": 0,
             "message": message, "message_history": history,
             "model_logprobs": lp(A=0.9, B=0.1)}
        )
    return pd.DataFrame(rows)


def test_select_probe_rows_prefers_message_banana_then_history_then_unaffected():
    df = make_archive()
    chosen = select_probe_rows(df, n_unaffected=2, max_affected=3, seed=0)
    chosen_trials = sorted(df.loc[chosen, "trialNum"])
    assert {0, 1} <= set(chosen_trials)  # both message-banana rows
    assert len([t for t in chosen_trials if 2 <= t < 4]) == 1  # one history-only row
    assert len([t for t in chosen_trials if t >= 4]) == 2  # two unaffected rows
    assert select_probe_rows(df, 2, 3, seed=0).tolist() == chosen.tolist()  # deterministic


def test_select_probe_rows_fails_loudly_without_affected_rows():
    df = make_archive().query("trialNum >= 4")
    with pytest.raises(ValueError, match="no banana-affected"):
        select_probe_rows(df, n_unaffected=2, max_affected=2, seed=0)


def probe_runner(shift_affected):
    """Fake model call: returns the archived logprobs, shifted on affected rows."""
    def run_fn(df_subset):
        out = df_subset.copy()
        affected = df_subset["message"].str.contains("banana") | df_subset[
            "message_history"].str.contains("banana")
        out["model_logprobs"] = [
            lp(A=0.9 - shift_affected, B=0.1 + shift_affected) if a else lp(A=0.9, B=0.1)
            for a in affected
        ]
        return out
    return run_fn


def test_run_probe_detects_post_fix_archive():
    report, probe = run_probe(make_archive(), probe_runner(0.0), n_unaffected=2, max_affected=4, seed=0)
    assert report.verdict.startswith("ARCHIVE IS POST-FIX")
    assert len(probe) == 6


def test_run_probe_detects_corrupted_archive():
    report, _ = run_probe(make_archive(), probe_runner(0.6), n_unaffected=2, max_affected=4, seed=0)
    assert report.verdict.startswith("ARCHIVE IS CORRUPTED-ERA")


def test_run_probe_never_groups_by_session_even_for_interactive_files():
    df = make_archive().assign(workerid="w1", model_prediction="A")
    report, _ = run_probe(df, probe_runner(0.0), n_unaffected=2, max_affected=4, seed=0)
    assert not report.cascade


def test_normalise_history_columns_converts_python_reprs_to_json():
    from src.banana_rerun import normalise_history_columns

    df = pd.DataFrame(
        {
            "selection_history": ["['A', 'B']", '["A"]', np.nan],
            "correctness_history": ["[True, False]", "[true]", "[]"],
            "message_history": ["[[{'role': 'describer', 'text': \"it's a man\"}]]", "[]", "[]"],
        }
    )
    out = normalise_history_columns(df)
    assert out["selection_history"].tolist()[:2] == ['["A", "B"]', '["A"]']
    assert out["correctness_history"].tolist() == ["[true, false]", "[true]", "[]"]
    assert out["message_history"][0] == '[[{"role": "describer", "text": "it\'s a man"}]]'
    assert pd.isna(out["selection_history"][2])
