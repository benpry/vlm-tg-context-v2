# make sure the banana rerun code is crerunning the correct rows
import sys

import pandas as pd

sys.path.append("scripts")

from rerun_banana_rows import has_banana


def has_nan(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask of rows where 'nan' appears in message or message_history.
    """
    in_message = df["message"].str.contains("nan", case=False, na=False)
    in_history = df["message_history"].str.contains("nan", case=False, na=False)
    return in_message | in_history


def test_all_nans_are_bananas():
    """
    Make sure that all instances of the string "nan" showing up in messages or message_history are because they contain the word "banana".
    """
    df = pd.read_csv("context_prep/human_history/limited_feedback_yoked.csv")
    nan_mask = has_nan(df)
    banana_mask = has_banana(df)
    assert (nan_mask == banana_mask).all()
