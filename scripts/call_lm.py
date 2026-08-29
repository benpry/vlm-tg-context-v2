import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd
from PIL import Image
from pyprojroot import here

from src.clients import setup_client
from src.interactive import run_interactive_evaluation
from src.lm import get_logits
from src.output_paths import raw_responses_path as raw_responses_path_for
from src.output_paths import results_path as results_path_for

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="the name of the model to evaluate",
        required=True,
    )
    parser.add_argument(
        "--grid_image_path",
        type=str,
        default="data/compiled_grid.png",
        help="the path to the image of the compiled tangrams",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="the number of trials to evaluate on (default: all)",
    )
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--yoked",
        action="store_true",
        help="Run batch evaluation with histories yoked to human selections (limited feedback)",
    )
    parser.add_argument("--api_base", type=str, default=None, help="API Base URL")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Resample N times instead of using logprobs (for frontier models without logprob support)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/logprobs",
        help="Directory (absolute, or relative to the project root) that receives the result files, "
        "mirroring context_prep/. Use another directory to rerun a model without touching the "
        "archived results in data/logprobs.",
    )

    args = parser.parse_args()

    # Read from the human history directory if we're using yoked or interactive evaluation
    if args.yoked or args.interactive:
        data_dir = "human_history"
    else:
        data_dir = "full_feedback"

    # Also run the interactive evaluation on the practice data
    if args.interactive:
        data_dirs = ["practice", data_dir]
    else:
        data_dirs = [data_dir]

    # compile a list of filepaths to read from
    data_filepaths = []
    for d in data_dirs:
        data_filepaths.extend(glob(str(here(f"context_prep/{d}/*.csv"))))
    print("data filepaths:", data_filepaths)

    dfs = []
    for filepath in data_filepaths:
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    prep_root = here("context_prep")
    default_output_root = here("data/logprobs")
    default_raw_root = here("data/raw_responses")
    output_root = (
        Path(args.output_root) if os.path.isabs(args.output_root) else here(args.output_root)
    )
    print("output root:", output_root)

    # set up the client (fails loudly if no Gemini key is set)
    client, use_responses_api, use_anthropic_api = setup_client(args.api_base)

    for filepath, df in zip(data_filepaths, dfs):
        # skip if we're not using a local model and we're not using limited feedback yoked or no context
        if (
            "localhost" not in args.api_base
            and "limited_feedback_yoked" not in filepath
            and "no_context" not in filepath
        ):
            continue

        # set up paths to write to
        output_path = str(
            results_path_for(
                filepath,
                args.model_name,
                prep_root,
                output_root,
                no_image=args.no_image,
                interactive=args.interactive,
                yoked=args.yoked,
            )
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        checkpoint_path = output_path + ".checkpoint"

        if args.overwrite:
            # Clean up any stale checkpoint files when overwriting
            for suffix in [
                ".checkpoint",
                ".checkpoint_meta.json",
                ".checkpoint_raw.json",
            ]:
                p = output_path + suffix
                if os.path.exists(p):
                    os.remove(p)

        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {filepath} as output file already exists.")
            continue

        print(f"Processing {filepath}...")

        # If we're sampling, set up a path to write raw responses to
        raw_responses_path = None
        if args.n_samples:
            raw_responses_path = str(
                raw_responses_path_for(
                    output_path, output_root, default_output_root, default_raw_root
                )
            )
            os.makedirs(os.path.dirname(raw_responses_path), exist_ok=True)

        if args.interactive:
            df_results = run_interactive_evaluation(
                df,
                args.model_name,
                client,
                grid_image,
                include_image=not args.no_image,
                n_trials=args.n_trials,
                n_samples=args.n_samples,
                raw_responses_path=raw_responses_path,
                use_responses_api=use_responses_api,
                use_anthropic_api=use_anthropic_api,
                checkpoint_path=checkpoint_path,
            )
            print(f"Saving {output_path}...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_results.to_csv(output_path, index=False)

        else:
            df_results = get_logits(
                df,
                args.model_name,
                client,
                grid_image,
                include_image=not args.no_image,
                n_trials=args.n_trials,
                n_samples=args.n_samples,
                raw_responses_path=raw_responses_path,
                use_responses_api=use_responses_api,
                use_anthropic_api=use_anthropic_api,
                checkpoint_path=checkpoint_path,
            )

        print(f"Saving {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
