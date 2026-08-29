"""
Rebuild sampled (frontier) results from their stored raw responses.

Frontier models are evaluated by sampling N answers per trial
(``src.lm.get_samples_single_row``): ``model_logprobs`` is log(count / N) of
the letter each sample was read as, and every raw text is kept under
data/raw_responses/. When the reading rule (``src.lm._extract_choice``)
changes, the results can therefore be rebuilt exactly from the raw texts. Two
things are checked while doing so:

* Provenance. A raw-responses file describes one run. A row is only rebuilt
  if the *old* reading rule applied to its raw texts reproduces the stored
  cell; otherwise the texts are not this row's, and it is reported as
  unsourced and left alone.
* Cascade. In interactive runs the argmax prediction is fed back into the
  prompts of the same session's later trials. If rebuilding changes a
  prediction, those later trials were conditioned on the wrong choice and the
  session has to be rerun (``scripts/rerun_sessions.py``).
"""

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from src.lm import CHOICES, HALLUCINATED_TURN, _counts_to_logprobs, _extract_choice

LOGPROBS_COL = "model_logprobs"
PREDICTION_COL = "model_prediction"


def legacy_extract_choice(text: str) -> str:
    """The reading rule the March 2026 runs used: the last valid capital letter."""
    text = text.strip()
    if text in CHOICES:
        return text
    for ch in reversed(text):
        if ch in CHOICES:
            return ch
    return text


def sample_text(sample) -> str:
    """The text of one raw sample: a bare string (runs before 28 Aug 2026) or
    {"text", "finish_reason", "model"} (later runs)."""
    return sample if isinstance(sample, str) else sample["text"]


_V2_ANSWER_FIRST = re.compile(r"^([A-L])(?=$|[^a-z'’\s])")


def extract_choice_v2(text: str) -> str:
    """The rule used for the 27 Aug 2026 reruns (Gemini yoked / no-context, the Claude
    session rerun): turn cut + glued answer-first, then the last capital letter."""
    text = text.strip()
    if text in CHOICES:
        return text
    turn = HALLUCINATED_TURN.search(text)
    answer = text[: turn.start()].strip() if turn else text
    if answer in CHOICES:
        return answer
    match = _V2_ANSWER_FIRST.match(answer)
    if match:
        return match.group(1)
    for ch in reversed(answer):
        if ch in CHOICES:
            return ch
    return text


# The rule a results file was produced with, for checking provenance.
PREVIOUS_RULES = {"march": legacy_extract_choice, "v2": extract_choice_v2}


def logprobs_from_samples(samples: list, extract=_extract_choice) -> dict:
    """log(count / N) per letter, exactly as get_samples_single_row computes it."""
    counts = Counter(extract(sample_text(s)) for s in samples)
    counts = Counter({choice: n for choice, n in counts.items() if choice in CHOICES})
    return _counts_to_logprobs(counts, len(samples)) if counts else {}


def parse_logprobs_cell(cell) -> dict:
    """A model_logprobs cell as read from CSV (str), in memory (dict) or empty (NaN)."""
    if isinstance(cell, dict):
        return cell
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return {}
    parsed = ast.literal_eval(str(cell))
    if not isinstance(parsed, dict):
        raise ValueError(f"model_logprobs cell is not a dict: {cell!r}")
    return parsed


def counts_from_logprobs(cell, n_samples: int) -> dict:
    """Invert log(count / N): the number of samples read as each letter."""
    return {
        letter: round(math.exp(logprob) * n_samples)
        for letter, logprob in parse_logprobs_cell(cell).items()
    }


def prediction_from_logprobs(logprobs: dict) -> str:
    """The choice fed back into an interactive session, as process_interactive_row picks it."""
    return max(logprobs, key=logprobs.get) if logprobs else ""


def interactive_raw_by_key(raw: dict, df_prep: pd.DataFrame, key_cols: list) -> dict:
    """Interactive raw responses are keyed by row index of the prep frame the run used."""
    keys = list(df_prep[key_cols].itertuples(index=False, name=None))
    by_key = {}
    for index, samples in raw.items():
        index = int(index)  # JSON keys are strings
        if not 0 <= index < len(keys):
            raise ValueError(f"Raw-response index {index} is outside the {len(keys)} prep rows")
        by_key[keys[index]] = samples
    return by_key


def batch_raw_by_key(raw: list, df_prep: pd.DataFrame, key_cols: list) -> dict:
    """Batch raw responses are a list in prep-row order."""
    if len(raw) != len(df_prep):
        raise ValueError(f"{len(raw)} raw-response entries for {len(df_prep)} prep rows")
    keys = list(df_prep[key_cols].itertuples(index=False, name=None))
    return dict(zip(keys, raw))


@dataclass(frozen=True)
class ReparseReport:
    n_rows: int
    sourced: list
    unsourced: list
    changed: list
    prediction_changed: list
    cascade_sessions: set

    def summary(self) -> str:
        lines = [
            f"rows: {self.n_rows}",
            f"rebuilt from raw responses: {len(self.sourced)}",
            f"no raw responses reproduce the stored cell (left alone): {len(self.unsourced)}",
            f"rows whose logprobs changed: {len(self.changed)}",
            f"rows whose fed-back prediction changed: {len(self.prediction_changed)}",
            f"sessions to rerun (cascade): {len(self.cascade_sessions)}"
            + (f" -> {sorted(self.cascade_sessions)}" if self.cascade_sessions else ""),
        ]
        return "\n".join(lines)


def reparse(
    df_results: pd.DataFrame,
    raw_sources: list,
    key_cols: list,
    session_col=None,
    previous_rule: str = "march",
) -> tuple:
    """Rebuild model_logprobs (and model_prediction, if present) from raw responses.

    ``raw_sources`` is a list of {key: samples} dicts, e.g. the original run's
    responses and a later partial rerun's; for each row the first source whose
    samples reproduce the stored cell under ``previous_rule`` (the rule the
    file was produced with, see PREVIOUS_RULES) is used.
    """
    if previous_rule not in PREVIOUS_RULES:
        raise ValueError(f"previous_rule must be one of {sorted(PREVIOUS_RULES)}, got {previous_rule!r}")
    previous_extract = PREVIOUS_RULES[previous_rule]
    df_new = df_results.copy()
    df_new[LOGPROBS_COL] = df_new[LOGPROBS_COL].astype(object)
    has_prediction = PREDICTION_COL in df_new.columns

    sourced, unsourced, changed, prediction_changed, cascade_sessions = [], [], [], [], set()
    keys = list(df_results[key_cols].itertuples(index=False, name=None))
    for row_index, key in zip(df_results.index, keys):
        stored = parse_logprobs_cell(df_results.at[row_index, LOGPROBS_COL])
        samples = None
        for source in raw_sources:
            if key in source and counts_from_logprobs(stored, len(source[key])) == dict(
                Counter(c for c in (previous_extract(sample_text(s)) for s in source[key]) if c in CHOICES)
            ):
                samples = source[key]
                break
        if samples is None:
            unsourced.append(key)
            continue

        sourced.append(key)
        new_logprobs = logprobs_from_samples(samples)
        df_new.at[row_index, LOGPROBS_COL] = new_logprobs
        if has_prediction:
            df_new.at[row_index, PREDICTION_COL] = prediction_from_logprobs(new_logprobs)
        if counts_from_logprobs(new_logprobs, len(samples)) != counts_from_logprobs(stored, len(samples)):
            changed.append(key)
        if prediction_from_logprobs(new_logprobs) != prediction_from_logprobs(stored):
            prediction_changed.append(key)
            if session_col is not None:
                cascade_sessions.add(str(df_results.at[row_index, session_col]))

    report = ReparseReport(
        n_rows=len(df_results),
        sourced=sourced,
        unsourced=unsourced,
        changed=changed,
        prediction_changed=prediction_changed,
        cascade_sessions=cascade_sessions,
    )
    return df_new, report
