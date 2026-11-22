#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from module import util
from module.custom_tasks import build_model_kwargs, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Equiformer v3 for carbohydrate NMR shift prediction")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--test_path_mo", type=str, required=True, help="Path to monosaccharide test dataset")
    parser.add_argument("--test_path_di", type=str, required=True, help="Path to disaccharide test dataset")
    parser.add_argument("--test_path_tri", type=str, required=True, help="Path to trisaccharide test dataset")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--lr_reduce", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--checkpoint_interval", default=0, type=int)
    util.add_equiformer_model_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        args.epochs,
        args.batch_size,
        args.train_path,
        args.test_path_mo,
        args.test_path_di,
        args.test_path_tri,
        args.lr,
        args.seed,
        args.checkpoint_interval,
        build_model_kwargs(args),
        args.lr_reduce,
    )


if __name__ == "__main__":
    main()
