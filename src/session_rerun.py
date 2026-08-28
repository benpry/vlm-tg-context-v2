"""
Merge rerun sessions of an interactive results file back into that file.

Interactive runs cascade: the model's own prediction on trial t is part of the
prompt for every later trial of the same session. So when anything about the
prompts or the choice extraction changes for one trial, the whole session has
to be rerun and its rows replaced as a block. This module does the replacement
and fails loudly if the rerun does not cover exactly the sessions it was asked
to cover.
"""

import pandas as pd


def merge_rerun_sessions(
    df_results: pd.DataFrame,
    df_rerun: pd.DataFrame,
    sessions: set,
    session_col: str = "workerid",
) -> pd.DataFrame:
    """Replace the rows of `sessions` in df_results with the rows of df_rerun."""
    sessions = {str(session) for session in sessions}
    results_sessions = df_results[session_col].astype(str)
    rerun_sessions = df_rerun[session_col].astype(str)

    missing = sessions - set(results_sessions)
    if missing:
        raise ValueError(f"Sessions {sorted(missing)} are not in the results file")
    if set(rerun_sessions) != sessions:
        raise ValueError(
            f"The rerun covers sessions {sorted(set(rerun_sessions))}, "
            f"expected exactly {sorted(sessions)}"
        )
    if set(df_rerun.columns) != set(df_results.columns):
        raise ValueError(
            "The rerun's columns differ from the results file's: "
            f"only in rerun {sorted(set(df_rerun.columns) - set(df_results.columns))}, "
            f"only in results {sorted(set(df_results.columns) - set(df_rerun.columns))}"
        )
    old_rows = results_sessions[results_sessions.isin(sessions)].value_counts()
    new_rows = rerun_sessions.value_counts()
    for session in sorted(sessions):
        if old_rows[session] != new_rows[session]:
            raise ValueError(
                f"Session {session}: the rerun has {new_rows[session]} rows, "
                f"the results file has {old_rows[session]} rows"
            )

    kept = df_results[~results_sessions.isin(sessions)]
    return pd.concat([kept, df_rerun[df_results.columns]], ignore_index=True)
