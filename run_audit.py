"""Run the item-validity detectors over Global-MMLU language subsets.

Examples
--------
Audit Hindi and German against the English source::

    python run_audit.py --languages en hi de

Audit every language listed in a file, writing flagged items to ./out::

    python run_audit.py --languages-file languages.txt --output-dir out

Print worked examples of what the translation introduced::

    python run_audit.py --languages en hi --examples 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from itemvalidity.detectors import (
    OPTION_COLUMNS,
    compare_to_source,
    flag_items,
    summarise,
)

DEFAULT_REPO = "CohereLabs/Global-MMLU"
DEFAULT_SPLIT = "test"
INDEX_COLUMN = "sample_id"


def load_language(repo: str, language: str, split: str) -> pd.DataFrame:
    """Load one language configuration, indexed by item identifier."""
    dataset = load_dataset(repo, language, split=split)
    frame = dataset.to_pandas()
    if INDEX_COLUMN not in frame.columns:
        raise ValueError(
            f"{repo}/{language} has no {INDEX_COLUMN!r} column; "
            f"cannot align items across languages"
        )
    return frame.set_index(INDEX_COLUMN)


def print_item(item_id: str, source: pd.DataFrame, target: pd.DataFrame, language: str) -> None:
    """Print one item side by side in the source and target languages."""
    print(f"\n  {item_id}  [{target.at[item_id, 'subject']}]  answer={target.at[item_id, 'answer']}")
    print(f"    EN  Q: {source.at[item_id, 'question']}")
    for column in OPTION_COLUMNS:
        print(f"      EN  {column[-1]}: {source.at[item_id, column]}")
    print(f"    {language.upper()}  Q: {target.at[item_id, 'question']}")
    for column in OPTION_COLUMNS:
        print(f"      {language.upper()}  {column[-1]}: {target.at[item_id, column]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    languages = parser.add_mutually_exclusive_group(required=True)
    languages.add_argument(
        "--languages",
        nargs="+",
        metavar="CODE",
        help="language configuration codes, e.g. en hi de",
    )
    languages.add_argument(
        "--languages-file",
        type=Path,
        help="file with one language code per line",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"dataset repository (default: {DEFAULT_REPO})")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"split to audit (default: {DEFAULT_SPLIT})")
    parser.add_argument("--source", default="en", help="source language to compare against (default: en)")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="where to write flagged items")
    parser.add_argument("--examples", type=int, default=0, help="worked examples to print per language and flag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.languages_file is not None:
        languages = [line.strip() for line in args.languages_file.read_text().splitlines() if line.strip()]
    else:
        languages = list(args.languages)

    if args.source not in languages:
        languages.insert(0, args.source)

    frames: dict[str, pd.DataFrame] = {}
    for language in languages:
        print(f"loading {language} ...", flush=True)
        frames[language] = flag_items(load_language(args.repo, language, args.split))

    source = frames[args.source]
    targets = [language for language in languages if language != args.source]

    print(f"\nitems per language: {len(source)}\n")
    print("FLAG COUNTS AND RATES")
    print(summarise(frames).to_string(float_format=lambda value: f"{value:.2%}"))

    print("\nINTRODUCED BY TRANSLATION")
    for language in targets:
        comparison = compare_to_source(source, frames[language])
        if comparison["compared"].iat[0] != len(source):
            print(
                f"  warning: {language} shares {comparison['compared'].iat[0]} "
                f"of {len(source)} item ids with {args.source}"
            )
        print(f"\n  {language}")
        print(comparison[["added", "removed"]].to_string())

    print("\nBY SUBJECT CATEGORY")
    for language, frame in frames.items():
        if "subject_category" not in frame.columns:
            continue
        print(f"\n  {language}")
        table = frame.groupby("subject_category")[
            ["informative_disclosure", "collapse", "key_corrupted"]
        ].agg(["sum", "count"])
        print(table.to_string())

    if args.examples:
        for language in targets:
            target = frames[language]
            shared = source.index.intersection(target.index)
            for flag in ("key_corrupted", "informative_disclosure"):
                introduced = (~source.loc[shared, flag]) & target.loc[shared, flag]
                item_ids = shared[introduced.to_numpy()][: args.examples]
                print(f"\n\n=== {language.upper()}: {flag} introduced by translation ({len(item_ids)} shown) ===")
                for item_id in item_ids:
                    print_item(item_id, source, target, language)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for language, frame in frames.items():
        path = args.output_dir / f"flagged_{language}.csv"
        frame.to_csv(path)
    print(f"\nwrote flagged items for {len(frames)} languages to {args.output_dir}/")


if __name__ == "__main__":
    main()
