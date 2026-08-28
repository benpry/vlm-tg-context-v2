"""
Where the evaluation scripts write result files.

Result files mirror the context_prep/ layout under an output root
(data/logprobs by default): context_prep/<folder>/<condition>.csv becomes
<output_root>/<folder>/<condition>_<model>_logprobs[_no_image].csv, except that
human_history results go to human_yoked/ (yoked runs) or interactive/
(interactive runs). A rerun can use a different output root so that the
archived results are never overwritten.
"""

from pathlib import Path


def results_path(
    prep_path,
    model_name: str,
    prep_root,
    output_root,
    no_image: bool = False,
    interactive: bool = False,
    yoked: bool = False,
) -> Path:
    """Result file for a context-prep CSV, a model and a run mode."""
    relative = Path(prep_path).relative_to(prep_root)  # raises if prep_path is elsewhere
    folder = relative.parent
    if folder.name == "human_history":
        if interactive:
            folder = folder.with_name("interactive")
        elif yoked:
            folder = folder.with_name("human_yoked")
    short_model_name = model_name.split("/")[-1]
    suffix = "_no_image" if no_image else ""
    return Path(output_root) / folder / f"{relative.stem}_{short_model_name}_logprobs{suffix}.csv"


def raw_responses_path(results_path, output_root, default_output_root, default_raw_root) -> Path:
    """Where the raw sampled texts behind a result file are stored.

    Results under the default root keep their raw texts in the default raw
    root (data/raw_responses); results under another root keep them in a
    sibling directory named <root>_raw_responses.
    """
    results_path, output_root = Path(results_path), Path(output_root)
    if output_root == Path(default_output_root):
        raw_root = Path(default_raw_root)
    else:
        raw_root = output_root.with_name(f"{output_root.name}_raw_responses")
    return raw_root / results_path.relative_to(output_root).with_suffix(".json")
