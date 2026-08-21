from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .oracle import analyze
from .schema import validate_interactions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle-study")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)

    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--input", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--lambda-cost", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.input)

    if args.command == "validate":
        validate_interactions(frame)
        print(f"Valid interaction log: {len(frame)} rows")
        return

    summary, per_workflow, oracle_choices = analyze(frame, args.lambda_cost)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    per_workflow.to_csv(args.output_dir / "per_workflow.csv", index=False)
    oracle_choices.to_csv(args.output_dir / "oracle_choices.csv", index=False)
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()

