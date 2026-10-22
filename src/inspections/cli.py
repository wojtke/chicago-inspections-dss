"""Command line: insp data | train | backtest."""
import argparse

from . import data, model, schedule


def main() -> None:
    parser = argparse.ArgumentParser(prog="insp", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data", help="download the city export and write data/inspections.parquet")
    p.add_argument("--force", action="store_true", help="download again even if the CSV is there")
    sub.add_parser("train", help="train the risk model on 2015-2023, report 2024 and 2025-26")
    p = sub.add_parser("backtest", help="replay a window of past inspections in a different order")
    p.add_argument("--start", default="2025-09-01")
    p.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    if args.cmd == "data":
        data.build(args.force)
    elif args.cmd == "train":
        model.train(data.load())
    elif args.cmd == "backtest":
        rows = schedule.score(model.load_model(), schedule.pool(data.load(), args.start, args.days))
        summary, _ = schedule.backtest(rows, args.days)
        print(f"window {args.start} + {args.days} days: {summary.attrs['inspections']:,} routine inspections, "
              f"{summary.attrs['fails']:,} failed")
        print(summary.to_string(formatters={"days_sooner": "{:+.1f}".format, "fails_in_first_half": "{:.0%}".format}))
