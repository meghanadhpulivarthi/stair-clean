import argparse
import sys


def add_prepare_data_subcommand(subparsers):
    parser = subparsers.add_parser(
        "prepare-data",
        help="Parse a raw corpus and build train/val/test splits plus a table of contents",
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Path to the raw corpus (a PDF file with bookmarks/table of contents)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for docs.jsonl, toc.json, train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional override YAML (see configs/base.yaml for all defaults)",
    )
    return parser


def add_train_subcommand(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="Fine-tune a model on prepared train/val data",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional override YAML (see configs/base.yaml for all defaults)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output run directory for checkpoints and logs",
    )
    return parser


def add_report_train_subcommand(subparsers):
    parser = subparsers.add_parser(
        "report-train",
        help="Print a terminal report of training curves and diagnostics for a run",
    )
    parser.add_argument("--run", required=True, help="Run directory produced by `stair train`")
    return parser


def add_eval_subcommand(subparsers):
    parser = subparsers.add_parser(
        "eval",
        help="Run the trained model on the test split and compute retrieval metrics",
    )
    parser.add_argument("--run", required=True, help="Run directory produced by `stair train`")
    parser.add_argument(
        "--config", default=None,
        help="Optional override YAML (see configs/base.yaml for all defaults)",
    )
    return parser


def add_report_eval_subcommand(subparsers):
    parser = subparsers.add_parser(
        "report-eval",
        help="Print a terminal report of retrieval metrics for a run",
    )
    parser.add_argument("--run", required=True, help="Run directory produced by `stair eval`")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stair",
        description="Parse a corpus, train a STAIR retrieval model, and report on it.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    add_prepare_data_subcommand(subparsers)
    add_train_subcommand(subparsers)
    add_report_train_subcommand(subparsers)
    add_eval_subcommand(subparsers)
    add_report_eval_subcommand(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_usage(sys.stderr)
        return 1

    print(f"stair {args.subcommand}: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
