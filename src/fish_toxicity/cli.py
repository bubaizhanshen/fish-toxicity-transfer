"""Command-line entry points for the core analyses."""

import argparse
from pathlib import Path
import pandas as pd
from .models import FAMILIES


def main():
    parser = argparse.ArgumentParser(
        description="Fish toxicity prediction with limited target labels"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "import-inputs", help="Copy model inputs from a local numerical archive"
    )
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("destination", type=Path)
    train = commands.add_parser(
        "benchmark", help="Run one family and experimental condition"
    )
    train.add_argument("--inputs", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--family", choices=FAMILIES, required=True)
    train.add_argument("--split", type=int, choices=range(6), required=True)
    train.add_argument("--budget", type=int, choices=[5, 20, 100], required=True)
    train.add_argument("--species", action="store_true")
    train.add_argument("--embeddings", type=Path)
    train.add_argument("--checkpoint", type=Path)
    select = commands.add_parser(
        "select", help="Select a family using out-of-fold error"
    )
    select.add_argument("folders", nargs="+", type=Path)
    select.add_argument("--mode", choices=["low", "joint"], required=True)
    select.add_argument("--output", type=Path, required=True)
    calibrate = commands.add_parser(
        "calibrate", help="Run source-model residual calibration"
    )
    calibrate.add_argument("--inputs", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--split", type=int, choices=range(6), required=True)
    encode = commands.add_parser(
        "encode", help="Encode molecules using a frozen local checkpoint"
    )
    encode.add_argument("--molecules", type=Path, required=True)
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--matrix-rows", type=int, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Summarize predictions and optional paired intervals"
    )
    evaluate.add_argument("--predictions", nargs="+", type=Path, required=True)
    evaluate.add_argument("--baseline", nargs="+", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "import-inputs":
        from .data import import_inputs

        import_inputs(**args)
    elif command == "benchmark":
        from .benchmark import run

        run(**args)
    elif command == "select":
        from .benchmark import select_family

        select_family(**args)
    elif command == "calibrate":
        from .calibration import run

        run(**args)
    elif command == "encode":
        from .representations import encode

        args["molecules"] = pd.read_csv(args["molecules"])
        encode(**args)
    elif command == "evaluate":
        from .metrics import summarize, paired_bootstrap

        output = args["output"]
        output.mkdir(parents=True, exist_ok=False)
        predictions = pd.concat(
            [pd.read_csv(path) for path in args["predictions"]], ignore_index=True
        )
        summarize(predictions).rename("value").to_csv(output / "metrics.csv")
        if args["baseline"]:
            baseline = pd.concat(
                [pd.read_csv(path) for path in args["baseline"]], ignore_index=True
            )
            paired_bootstrap(predictions, baseline).to_csv(
                output / "paired_intervals.csv", index=False
            )
