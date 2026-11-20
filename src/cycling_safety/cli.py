"""Command line for source import, evaluation and portable feature inference."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .contract import FEATURE_NAMES
from .data import import_features, write_json
from .inference import load_model, predict_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-features")
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--output", type=Path, default=Path("data/derived"))
    importer.add_argument("--expected-manifest", type=Path, default=Path("provenance/source_manifest.json"))
    raw = commands.add_parser("prepare-raw")
    raw.add_argument("--archives", type=Path, nargs="+", required=True)
    raw.add_argument("--output", type=Path, default=Path("data/rebuilt"))
    cnn = commands.add_parser("evaluate-cnn")
    cnn.add_argument("--data", type=Path, default=Path("data/rebuilt"))
    cnn.add_argument("--output", type=Path, required=True)
    cnn.add_argument("--folds", type=int, default=5)
    cnn.add_argument("--seed", type=int, default=42)
    cnn.add_argument("--epochs", type=int, default=15)
    experiment = commands.add_parser("evaluate")
    experiment.add_argument("--data", type=Path, default=Path("data/derived"))
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--outer-folds", type=int, default=5)
    experiment.add_argument("--inner-folds", type=int, default=3)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--max-iter", type=int, default=200)
    predictor = commands.add_parser("predict")
    predictor.add_argument("--model", type=Path, default=Path("artifacts/model.json"))
    predictor.add_argument("--csv", type=Path, required=True)
    predictor.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "import-features":
        manifest = import_features(args.source, args.output, args.expected_manifest)
        print(json.dumps({"windows": manifest["windows"], "recordings": manifest["recordings"]}))
    elif args.command == "prepare-raw":
        from .preprocessing import prepare_archives
        manifest = prepare_archives(args.archives, args.output)
        print(json.dumps({"windows": manifest["windows"], "recordings": manifest["recordings"]}))
    elif args.command == "evaluate-cnn":
        from .cnn import evaluate_cnn
        results = evaluate_cnn(args.data, args.output, args.folds, args.seed, args.epochs)
        print(json.dumps({"macro_f1": results["macro_f1"], "accuracy": results["accuracy"]}))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        results = evaluate(args.data, args.output, args.outer_folds, args.inner_folds, args.seed, args.max_iter)
        print(json.dumps({k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in results.items()}))
    else:
        frame = pd.read_csv(args.csv)
        if set(frame.columns) != set(FEATURE_NAMES) or len(frame.columns) != len(FEATURE_NAMES):
            parser.error("CSV must contain exactly the 72 feature columns")
        predictions = predict_records(load_model(args.model), frame.to_dict(orient="records"))
        write_json(args.output, {"predictions": predictions, "notice": "Experimental, uncalibrated model outputs"})


if __name__ == "__main__":
    main()
