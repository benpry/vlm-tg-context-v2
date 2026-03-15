"""
Make sure that logprob files have the right number of rows for all trial numbers, etc.
"""

import json
import re
import warnings
from glob import glob

import numpy as np
import pandas as pd
import pytest
from pyprojroot import here

from src.interactive import _add_session_id


def logprob_to_prep_mapping(logprob_file: str, subfolder: str = "human_history"):
    """Turn a logprob filename into the corresponding prep filename.

    e.g. limited_feedback_random_Kimi-VL-A3B-Instruct_logprobs.csv -> limited_feedback_random.csv
    """
    filename = re.sub(r"_[^_]+_logprobs(_no_image)?\.csv$", ".csv", logprob_file)
    return re.sub(r"data/logprobs/[^/]+/", f"context_prep/{subfolder}/", filename)


@pytest.fixture
def interactive_logprob_files():
    return glob(str(here("data/logprobs/interactive/*.csv")))


@pytest.fixture
def yoked_logprob_files():
    return glob(str(here("data/logprobs/human_yoked/*.csv")))


@pytest.fixture
def full_feedback_logprob_files():
    return glob(str(here("data/logprobs/full_feedback/*.csv")))


def trial_shape_util(df_logprobs: pd.DataFrame, df_prep: pd.DataFrame):
    assert df_logprobs["trialNum"].max() == 71
    # make sure there's the same number of rows for each trial number
    prep_trialnum_col = (
        "matcher_trialNum" if "matcher_trialNum" in df_prep.columns else "trialNum"
    )
    assert np.all(
        df_logprobs.groupby("trialNum").count()["gameId"]
        == df_prep.groupby(prep_trialnum_col).count()["gameId"]
    )


def interactive_past_selections_util(df_logprobs: pd.DataFrame):
    # for each session id, make sure the last selection is at the end of the selection history for the next trial
    _add_session_id(df_logprobs)

    mismatches = 0
    for session_id in df_logprobs["_session_id"].unique():
        df_session = df_logprobs[df_logprobs["_session_id"] == session_id].sort_values(
            "trialNum"
        )

        for i in range(len(df_session) - 1):
            curr_row = df_session.iloc[i]
            next_row = df_session.iloc[i + 1]

            # Only check consecutive trials
            if next_row["trialNum"] != curr_row["trialNum"] + 1:
                continue

            sel_hist = next_row["selection_history"]

            if not isinstance(curr_row["model_prediction"], str):
                warnings.warn(
                    f"Model prediction is not a string: {curr_row['model_prediction']}"
                )
                continue
            if len(sel_hist) == 0:
                warnings.warn(f"Selection history is empty: {sel_hist}")
                continue

            if sel_hist[-1] != curr_row["model_prediction"]:
                mismatches += 1
                print(
                    f"Mismatch found for session {session_id}, trial {curr_row['trialNum']}"
                    f"next selection history: {sel_hist}"
                    f"model prediction: {curr_row['model_prediction']}"
                )

    assert mismatches == 0


def check_selection_history_correctness(row):
    for target, selection, correctness in zip(
        row["target_history"], row["selection_history"], row["correctness_history"]
    ):
        if (
            selection != target
            and correctness
            or selection == target
            and not correctness
        ):
            return False
    return True


def selection_history_correctness_util(df_logprobs: pd.DataFrame):
    # make sure the selection history is correct
    selection_history_is_correct = df_logprobs.apply(
        check_selection_history_correctness, axis=1
    )
    assert np.all(selection_history_is_correct)


def test_interactive_files(interactive_logprob_files):
    for f in interactive_logprob_files:
        print(f"Testing {f}...")
        # get each logprob file and its corresponding prep file

        df_logprobs = pd.read_csv(f)
        df_logprobs["target_history"] = df_logprobs["target_history"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )
        df_logprobs["selection_history"] = df_logprobs["selection_history"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )

        df_logprobs["correctness_history"] = df_logprobs["correctness_history"].apply(
            lambda x: json.loads(x.replace("T", "t").replace("F", "f"))
            if isinstance(x, str)
            else x
        )

        df_prep = pd.read_csv(logprob_to_prep_mapping(f, "human_history"))
        trial_shape_util(df_logprobs, df_prep)
        interactive_past_selections_util(df_logprobs)
        selection_history_correctness_util(df_logprobs)


def test_yoked_files(yoked_logprob_files):
    for f in yoked_logprob_files:
        print(f"Testing {f}...")
        df_logprobs = pd.read_csv(f)
        df_logprobs["target_history"] = df_logprobs["target_history"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )
        df_logprobs["selection_history"] = df_logprobs["selection_history"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )
        df_logprobs["correctness_history"] = df_logprobs["correctness_history"].apply(
            lambda x: json.loads(x.replace("T", "t").replace("F", "f"))
            if isinstance(x, str)
            else x
        )

        df_logprobs["trialNum"] = df_logprobs["matcher_trialNum"]

        df_prep = pd.read_csv(logprob_to_prep_mapping(f, "human_history"))
        trial_shape_util(df_logprobs, df_prep)
        selection_history_correctness_util(df_logprobs)


def test_full_feedback_files(full_feedback_logprob_files):
    for f in full_feedback_logprob_files:
        print(f"Testing {f}...")
        df_logprobs = pd.read_csv(f)
        df_logprobs["target_history"] = df_logprobs["target_history"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )
        df_logprobs["selection_history"] = df_logprobs["selection_history"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )
        df_logprobs["correctness_history"] = df_logprobs["correctness_history"].apply(
            lambda x: json.loads(x.replace("T", "t").replace("F", "f"))
            if isinstance(x, str)
            else x
        )

        if "matcher_trialNum" in df_logprobs and "trialNum" not in df_logprobs:
            df_logprobs["trialNum"] = df_logprobs["matcher_trialNum"]

        df_prep = pd.read_csv(logprob_to_prep_mapping(f, "full_feedback"))
        trial_shape_util(df_logprobs, df_prep)
        selection_history_correctness_util(df_logprobs)
        # make sure the correcness histories are all lists containing only True
        assert np.all(df_logprobs["correctness_history"].apply(lambda x: all(x)))
