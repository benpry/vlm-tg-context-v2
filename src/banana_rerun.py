"""
Helpers for auditing the banana -> ba''a rerun.

Before 15 Mar 2026, ``src/utils.py`` parsed message text with
``literal_eval(text.replace("nan", "''"))``, which corrupted every "banana" to
"ba''a" in the prompts shown to models. Results produced with that code must be
rerun for every trial whose message or in-context history mentions banana.

These helpers (1) identify the affected rows / sessions from a context-prep CSV
and (2) check, by comparing a results file against its pre-rerun ``.bak``
backup, that a rerun actually replaced them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

BATCH_KEY_COLS = ["gameId", "trialNum", "repNum"]
SESSION_COL = "workerid"
TRIAL_COL = "trialNum"
LOGPROBS_COL = "model_logprobs"

RerunMode = Literal["batch", "interactive"]


def has_banana(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows where 'banana' appears in message or message_history."""
    in_message = df["message"].astype(str).str.contains("banana", case=False, na=False)
    in_history = (
        df["message_history"].astype(str).str.contains("banana", case=False, na=False)
    )
    return in_message | in_history


def _row_keys(df: pd.DataFrame, key_cols: list[str]) -> list[tuple]:
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing key columns {missing}; have {list(df.columns)}")
    return list(df[key_cols].itertuples(index=False, name=None))


def affected_rows(df_prep: pd.DataFrame, key_cols: list[str]) -> set[tuple]:
    """Keys of the prep rows whose text was corrupted by the bug (batch mode)."""
    return set(_row_keys(df_prep.loc[has_banana(df_prep)], key_cols))


def affected_sessions(df_prep: pd.DataFrame) -> set:
    """Sessions (workerids) with at least one corrupted row (interactive mode).

    A whole session must be rerun because the model's own guesses cascade
    through the feedback history of every later trial in that session.
    """
    return set(df_prep.loc[has_banana(df_prep), SESSION_COL])


def changed_rows(
    df_results: pd.DataFrame, df_backup: pd.DataFrame, key_cols: list[str]
) -> set[tuple]:
    """Keys of rows whose model_logprobs differ between results and backup.

    Rows are matched by key, not position, because the rerun merge reorders
    rows. Both files must contain exactly the same set of rows.
    """
    results_keys = _row_keys(df_results, key_cols)
    backup_keys = _row_keys(df_backup, key_cols)
    for name, keys in [("results", results_keys), ("backup", backup_keys)]:
        if len(set(keys)) != len(keys):
            raise ValueError(f"Keys {key_cols} are not unique in the {name} file")
    if set(results_keys) != set(backup_keys):
        raise ValueError(
            "Results and backup must contain the same set of rows "
            f"(results: {len(results_keys)}, backup: {len(backup_keys)}, "
            f"only in results: {len(set(results_keys) - set(backup_keys))}, "
            f"only in backup: {len(set(backup_keys) - set(results_keys))})"
        )

    def logprobs_by_key(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            df[LOGPROBS_COL].fillna("").astype(str).to_numpy(),
            index=pd.MultiIndex.from_tuples(_row_keys(df, key_cols)),
        )

    results_lp = logprobs_by_key(df_results)
    backup_lp = logprobs_by_key(df_backup).reindex(results_lp.index)
    differs = results_lp != backup_lp
    return set(results_lp.index[differs.to_numpy()])


def changed_sessions(df_results: pd.DataFrame, df_backup: pd.DataFrame) -> set:
    """Sessions with at least one row whose model_logprobs changed."""
    changed = changed_rows(df_results, df_backup, [SESSION_COL, TRIAL_COL])
    return {session for session, _trial in changed}


@dataclass(frozen=True)
class RerunReport:
    mode: RerunMode
    affected: set
    changed: set

    @property
    def affected_not_changed(self) -> set:
        """Affected rows/sessions the rerun did not replace: the rerun is incomplete."""
        return self.affected - self.changed

    @property
    def changed_not_affected(self) -> set:
        """Rows/sessions that changed although the bug did not affect them."""
        return self.changed - self.affected

    @property
    def complete(self) -> bool:
        return len(self.affected_not_changed) == 0

    def summary(self) -> str:
        unit = "rows" if self.mode == "batch" else "sessions"
        lines = [
            f"mode: {self.mode}",
            f"affected {unit}: {len(self.affected)}",
            f"changed {unit}: {len(self.changed)}",
            f"affected but NOT changed: {len(self.affected_not_changed)}"
            + (f" -> {sorted(self.affected_not_changed)}" if self.affected_not_changed else ""),
            f"changed but not affected: {len(self.changed_not_affected)}"
            + (f" -> {sorted(self.changed_not_affected)}" if self.changed_not_affected else ""),
            "verdict: "
            + ("rerun COMPLETE" if self.complete else "rerun INCOMPLETE"),
        ]
        return "\n".join(lines)


def compare_rerun(
    df_prep: pd.DataFrame,
    df_results: pd.DataFrame,
    df_backup: pd.DataFrame,
    mode: RerunMode,
) -> RerunReport:
    """Compare a rerun's results against its backup for the affected rows/sessions."""
    if mode == "batch":
        affected = affected_rows(df_prep, BATCH_KEY_COLS)
        changed = changed_rows(df_results, df_backup, BATCH_KEY_COLS)
    elif mode == "interactive":
        affected = affected_sessions(df_prep)
        changed = changed_sessions(df_results, df_backup)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'batch' or 'interactive'")
    return RerunReport(mode=mode, affected=affected, changed=changed)


def compare_rerun_files(
    prep_path: Path, results_path: Path, backup_path: Path, mode: RerunMode
) -> RerunReport:
    """Load the prep, results and backup CSVs and compare them."""
    for path in [prep_path, results_path, backup_path]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    return compare_rerun(
        pd.read_csv(prep_path),
        pd.read_csv(results_path),
        pd.read_csv(backup_path),
        mode=mode,
    )


# --- comparing two versions of the same results file -------------------------
#
# Used to verify, when only archived copies survive, that a later upload of a
# results file really is a post-fix rerun: its normalised choice probabilities
# should differ from the corrupted-era copy on the banana-affected trials and
# (up to run-to-run noise) nowhere else. Results files carry their own
# ``message`` / ``message_history`` columns, so no prep file is needed.

import ast
import math

import numpy as np

LETTERS = list("ABCDEFGHIJKL")
KEY_CANDIDATES = [
    "workerid",
    "shuffle_rep",
    "gameId",
    "gameId.y",
    "matcher_trialNum",
    "trialNum",
    "repNum",
]
SESSION_CANDIDATES = ["workerid", "shuffle_rep"]
PREDICTION_COL = "model_prediction"
CHANGED_TOLERANCE = 0.01


def parse_logprobs(cell) -> dict[str, float]:
    """Parse one model_logprobs cell (a Python dict literal) into {letter: logprob}."""
    if cell is None or isinstance(cell, dict):
        return cell or {}
    if isinstance(cell, float) and math.isnan(cell):
        return {}
    text = str(cell).strip()
    if text in ("", "nan", "{}"):
        return {}
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"model_logprobs cell is not a dict: {text[:80]}")
    return {str(k): float(v) for k, v in parsed.items()}


def normalised_probs(df: pd.DataFrame, col: str = LOGPROBS_COL) -> np.ndarray:
    """Softmax-renormalised probabilities over A-L (n_rows x 12), as in the analysis.

    Letters missing from a cell get probability 0; a row with no letters at
    all (e.g. every frontier sample invalid) is all zeros.
    """
    probs = np.zeros((len(df), len(LETTERS)))
    for i, cell in enumerate(df[col].tolist()):
        logprobs = parse_logprobs(cell)
        for j, letter in enumerate(LETTERS):
            if letter in logprobs:
                probs[i, j] = math.exp(logprobs[letter])
    sums = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, sums, out=np.zeros_like(probs), where=sums > 0)


def shared_key_cols(df_old: pd.DataFrame, df_new: pd.DataFrame) -> list[str]:
    """Trial-identifying columns present in both files; must identify rows uniquely."""
    cols = [c for c in KEY_CANDIDATES if c in df_old.columns and c in df_new.columns]
    if not cols:
        raise KeyError(f"No shared key columns among {KEY_CANDIDATES}")
    for name, df in [("old", df_old), ("new", df_new)]:
        if df.duplicated(cols).any():
            raise ValueError(f"Key columns {cols} do not identify rows uniquely in the {name} file")
    return cols


def probability_shift(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Per-trial max |p_new - p_old| over the 12 letters, plus whether the trial is banana-affected.

    Rows are matched by key, not position. Both files must contain the same trials.
    """
    key_cols = shared_key_cols(df_old, df_new)
    merged = df_new[key_cols + [LOGPROBS_COL]].merge(
        df_old[key_cols + [LOGPROBS_COL]],
        on=key_cols,
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"Old and new files must contain the same trials: {unmatched.sum()} unmatched rows"
        )
    # merge with how="outer" sorts by key; restore new-file order so masks line up
    merged = df_new[key_cols].merge(merged, on=key_cols, how="left")
    diff = np.abs(
        normalised_probs(merged, f"{LOGPROBS_COL}_new")
        - normalised_probs(merged, f"{LOGPROBS_COL}_old")
    ).max(axis=1)
    shift = df_new[key_cols].copy()
    shift["affected"] = has_banana(df_new).to_numpy()
    shift["max_abs_diff"] = diff
    return shift


@dataclass(frozen=True)
class VersionReport:
    key_cols: list[str]
    cascade: bool
    stats: pd.DataFrame
    verdict: str

    def summary(self) -> str:
        mode = (
            "interactive file: grouping by session, because a banana trial changes the "
            "model's own feedback history for every later trial in that session"
            if self.cascade
            else "batch file: trials are independent"
        )
        return "\n".join(
            [
                f"key columns: {self.key_cols}",
                mode,
                self.stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
                f"verdict: {self.verdict}",
            ]
        )


def _group_labels(shift: pd.DataFrame, df_new: pd.DataFrame, cascade: bool) -> np.ndarray:
    if not cascade:
        return np.where(shift["affected"], "affected row", "unaffected row")
    session_col = next(c for c in SESSION_CANDIDATES if c in df_new.columns)
    affected_sessions = set(df_new.loc[shift["affected"].to_numpy(), session_col])
    in_affected_session = df_new[session_col].isin(affected_sessions).to_numpy()
    return np.where(
        shift["affected"],
        "affected row",
        np.where(in_affected_session, "unaffected row in affected session", "unaffected session"),
    )



def _probe_verdict(stats: pd.DataFrame) -> str:
    """Interpret probe statistics: does the archive match a fixed-text replay?"""
    by_group = stats.set_index("group")
    if "affected row" not in by_group.index or "unaffected row" not in by_group.index:
        raise ValueError("A probe needs both affected and unaffected rows")
    affected = by_group.loc["affected row"]
    clean = by_group.loc["unaffected row"]
    detail = (
        f"affected trials mean shift {affected['mean']:.3f} (max {affected['max']:.3f}) vs. "
        f"unaffected trials mean {clean['mean']:.3f} (max {clean['max']:.3f}, the run-to-run noise floor)."
    )
    if affected["mean"] <= max(3 * clean["mean"], CHANGED_TOLERANCE):
        return f"ARCHIVE IS POST-FIX: affected trials match the fixed-text replay as closely as unaffected ones. {detail}"
    if affected["mean"] > max(3 * clean["mean"], 5 * CHANGED_TOLERANCE):
        return f"ARCHIVE IS CORRUPTED-ERA: affected trials diverge from the fixed-text replay while unaffected ones match. {detail}"
    return f"INCONCLUSIVE: affected trials shifted somewhat more than the noise floor; increase the sample. {detail}"


def compare_result_versions(
    df_old: pd.DataFrame, df_new: pd.DataFrame, probe: bool = False
) -> VersionReport:
    """Compare an older and a newer version of the same results file.

    With ``probe=True`` the new frame is a fixed-text replay of the old
    (archived) rows, so rows are never grouped by session and the verdict is
    about the archive.
    """
    shift = probability_shift(df_old, df_new)
    cascade = (
        not probe
        and PREDICTION_COL in df_new.columns
        and any(c in df_new.columns for c in SESSION_CANDIDATES)
    )
    shift["group"] = _group_labels(shift, df_new, cascade)
    stats = (
        shift.groupby("group")["max_abs_diff"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p95=lambda x: x.quantile(0.95),
            max="max",
            frac_changed=lambda x: (x > CHANGED_TOLERANCE).mean(),
        )
        .reset_index()
    )

    if probe:
        verdict = _probe_verdict(stats)
    elif shift["max_abs_diff"].max() == 0:
        verdict = "IDENTICAL: no probability differs anywhere; the newer file is not a rerun."
    else:
        by_group = stats.set_index("group")
        affected = by_group.loc["affected row"] if "affected row" in by_group.index else None
        clean_group = "unaffected session" if cascade else "unaffected row"
        clean = by_group.loc[clean_group] if clean_group in by_group.index else None
        if affected is None:
            verdict = "No banana-affected trials in this file; differences reflect run-to-run noise only."
        elif clean is None:
            verdict = "Every trial is banana-affected; cannot separate the fix from run-to-run noise."
        else:
            detail = (
                f"affected rows mean shift {affected['mean']:.3f} (max {affected['max']:.3f}); "
                f"{clean_group}s mean shift {clean['mean']:.3f} (max {clean['max']:.3f})."
            )
            if clean["max"] <= CHANGED_TOLERANCE and affected["mean"] > 5 * CHANGED_TOLERANCE:
                verdict = f"FIXED-TEXT RERUN: only banana-affected trials changed. {detail}"
            elif affected["mean"] > 3 * max(clean["mean"], 1e-9):
                verdict = (
                    "CONSISTENT WITH A FIXED-TEXT RERUN: affected trials shifted much more "
                    f"than unaffected ones (the rest is run-to-run noise). {detail}"
                )
            else:
                verdict = (
                    "INCONCLUSIVE: affected and unaffected trials shifted similarly, so the "
                    f"difference cannot be attributed to the banana fix. {detail}"
                )
    return VersionReport(key_cols=shared_key_cols(df_old, df_new), cascade=cascade,
                         stats=stats, verdict=verdict)


def compare_result_versions_files(old_path: Path, new_path: Path) -> VersionReport:
    for path in [old_path, new_path]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    return compare_result_versions(pd.read_csv(old_path), pd.read_csv(new_path))


# --- probe: replay archived trials with today's code ---------------------------
#
# Archived results files store, for every trial, exactly the histories the model
# was given (message_history / selection_history / correctness_history) plus
# the message. Replaying a sample of those rows through the current, fixed code
# and comparing the new logprobs with the archived ones therefore tests one
# thing only: whether the archived run saw the corrupted "ba''a" text.
# Unaffected trials (no banana anywhere in the prompt) give the run-to-run
# noise floor; affected trials that match the probe as closely as those are
# post-fix, affected trials that diverge are corrupted-era.

import json

HISTORY_COLS = ["message_history", "selection_history", "correctness_history", "target_history"]


def _history_to_json(cell):
    """Return a JSON string for a history cell stored either as JSON or as a Python repr."""
    if not isinstance(cell, str):
        return cell
    try:
        json.loads(cell)
        return cell
    except json.JSONDecodeError:
        return json.dumps(ast.literal_eval(cell))


def normalise_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make history columns JSON again.

    Interactive runs write their histories back to CSV as Python reprs
    (single quotes, True/False), which ``preprocess_messages`` cannot parse.
    """
    out = df.copy()
    for col in HISTORY_COLS:
        if col in out.columns:
            out[col] = out[col].map(_history_to_json)
    return out


def select_probe_rows(
    df: pd.DataFrame, n_unaffected: int, max_affected: int, seed: int
) -> pd.Index:
    """Pick rows to replay: banana-in-message rows first, then banana-in-history
    rows, up to max_affected; plus n_unaffected rows with no banana at all."""
    rng = np.random.default_rng(seed)
    affected = has_banana(df)
    if not affected.any():
        raise ValueError("This file has no banana-affected trials; there is nothing to probe")
    in_message = df["message"].astype(str).str.contains("banana", case=False, na=False)
    unaffected_rows = df.index[~affected]
    if len(unaffected_rows) < n_unaffected:
        raise ValueError(
            f"Only {len(unaffected_rows)} unaffected rows available, need {n_unaffected}"
        )

    def sample(rows: pd.Index, k: int) -> list:
        k = min(k, len(rows))
        return sorted(rng.choice(rows.to_numpy(), size=k, replace=False).tolist()) if k > 0 else []

    chosen = sample(df.index[in_message], max_affected)
    chosen += sample(df.index[affected & ~in_message], max_affected - len(chosen))
    chosen += sample(unaffected_rows, n_unaffected)
    return pd.Index(chosen)


def run_probe(
    df_archived: pd.DataFrame,
    run_fn,
    n_unaffected: int,
    max_affected: int,
    seed: int,
) -> tuple[VersionReport, pd.DataFrame]:
    """Replay a sample of archived rows through ``run_fn`` and compare.

    ``run_fn`` takes the sampled rows (without model_logprobs) and returns them
    with a fresh ``model_logprobs`` column, e.g. via ``src.lm.get_logits``.
    """
    chosen = select_probe_rows(df_archived, n_unaffected, max_affected, seed)
    df_subset = df_archived.loc[chosen].copy().reset_index(drop=True)
    df_probe = run_fn(df_subset.drop(columns=[LOGPROBS_COL]))
    if LOGPROBS_COL not in df_probe.columns:
        raise RuntimeError("run_fn must return a frame with a model_logprobs column")
    if len(df_probe) != len(df_subset):
        raise RuntimeError(f"run_fn returned {len(df_probe)} rows for {len(df_subset)} inputs")
    report = compare_result_versions(df_subset, df_probe, probe=True)
    return report, df_probe
