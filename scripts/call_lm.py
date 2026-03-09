import os
from argparse import ArgumentParser
from glob import glob

import anthropic
import pandas as pd
from google import genai
from openai import OpenAI
from PIL import Image
from pyprojroot import here

from src.interactive import count_tokens_interactive, run_interactive_evaluation
from src.lm import count_tokens, get_logits

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
        "--dry_run",
        action="store_true",
        help="Count tokens without calling any language models (for cost estimation)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Resample N times instead of using logprobs (for frontier models without logprob support)",
    )

    args = parser.parse_args()

    # Handle --yoked flag as alias for human_history data
    if args.yoked or args.interactive:
        data_dir = "human_history"
    else:
        data_dir = "full_feedback"

    if args.interactive:
        data_dirs = ["practice", data_dir]
    else:
        data_dirs = [data_dir]

    data_filepaths = []
    for d in data_dirs:
        data_filepaths.extend(glob(str(here(f"context_prep/{d}/*.csv"))))
    print("data filepaths:", data_filepaths)

    dfs = []
    for filepath in data_filepaths:
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    use_responses_api = False
    use_anthropic_api = False

    if args.dry_run:
        client = None
    elif "google" in args.api_base:
        client = genai.Client(api_key=os.getenv("LANGCOG_GEMINI_API_KEY"))
    elif "anthropic" in args.api_base:
        client = anthropic.Anthropic()
        use_anthropic_api = True
    else:
        client = OpenAI(
            base_url=args.api_base,
        )
        use_responses_api = "openai.com" in args.api_base

    grand_totals = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    for filepath, df in zip(data_filepaths, dfs):
        # only run limited feedback yoked if we're not using a local model
        if (
            "localhost" not in args.api_base
            and "limited_feedback_yoked" not in filepath
            and "no_context" not in filepath
        ):
            continue

        output_path = filepath.replace(
            ".csv",
            f"_{args.model_name.split('/')[-1]}_logprobs{'_no_image' if args.no_image else ''}.csv",
        ).replace("context_prep", "data/logprobs")
        if args.interactive:
            output_path = output_path.replace("human_history", "interactive")
        elif args.yoked:
            output_path = output_path.replace("human_history", "human_yoked")
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

        if not args.dry_run and os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {filepath} as output file already exists.")
            continue

        print(f"Processing {filepath}...")

        if args.dry_run:
            if args.interactive:
                token_summary = count_tokens_interactive(
                    df,
                    args.model_name,
                    grid_image,
                    include_image=not args.no_image,
                    n_trials=args.n_trials,
                )
            else:
                token_summary = count_tokens(
                    df,
                    args.model_name,
                    grid_image,
                    include_image=not args.no_image,
                    n_trials=args.n_trials,
                )
            print(f"  Rows: {token_summary['n_rows']}")
            print(
                f"  Image tokens per row (estimate): {token_summary['image_tokens_per_row']}"
            )
            stats = token_summary["text_tokens_per_row"]
            print(
                f"  Text tokens per row: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.0f}"
            )
            print(f"  Total input tokens: {token_summary['total_input_tokens']}")
            print(f"  Total output tokens: {token_summary['total_output_tokens']}")
            grand_totals["total_input_tokens"] += token_summary["total_input_tokens"]
            grand_totals["total_output_tokens"] += token_summary["total_output_tokens"]
        elif args.interactive:
            raw_responses_path = None
            if args.n_samples:
                raw_responses_path = output_path.replace(
                    "data/logprobs", "data/raw_responses"
                ).replace(".csv", ".json")
                os.makedirs(os.path.dirname(raw_responses_path), exist_ok=True)

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
            df_results.to_csv(here(output_path), index=False)
        else:
            raw_responses_path = None
            if args.n_samples:
                raw_responses_path = output_path.replace(
                    "data/logprobs", "data/raw_responses"
                ).replace(".csv", ".json")
                os.makedirs(os.path.dirname(raw_responses_path), exist_ok=True)

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
            df_results.to_csv(here(output_path), index=False)

    if args.dry_run:
        print("\n=== Dry Run Summary ===")
        print(f"Model: {args.model_name}")
        print(f"Total input tokens:  {grand_totals['total_input_tokens']}")
        print(f"Total output tokens: {grand_totals['total_output_tokens']}")
        print(f"Total output tokens: {grand_totals['total_output_tokens']}")
