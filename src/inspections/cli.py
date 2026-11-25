"""Command line: insp data | train | evaluate | plan | users | demo  (city from --city or $INSP_CITY)."""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="insp", description="Inspection planning DSS")
    ap.add_argument("--city", default=None, help="city slug (default: $INSP_CITY or chicago)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data", help="download and clean the city's open data")
    d.add_argument("--force", action="store_true", help="download again")
    sub.add_parser("train", help="train, calibrate and audit the risk model")
    ev = sub.add_parser("evaluate", help="replay past windows for every policy and scenario")
    ev.add_argument("--quick", action="store_true", help="one window, capacity x1 only")
    pl = sub.add_parser("plan", help="print the routine plan for one day")
    pl.add_argument("--date", required=True)
    pl.add_argument("--out", default=None, help="write the plan to this CSV")
    sub.add_parser("users", help="create the demo team (a manager and 13 inspectors)")
    dm = sub.add_parser("demo", help="reset the app's database to a lived-in demo: team, plans and results")
    dm.add_argument("--days", type=int, default=8, help="working days of approved plans (default 8)")
    args = ap.parse_args()

    from .city import get_city
    city = get_city(args.city)
    if args.cmd == "data":
        from .data import build
        build(city, force_download=args.force)
    elif args.cmd == "train":
        from .data import load
        from .model import train
        train(load(city), city)
    elif args.cmd == "evaluate":
        from .evaluation import run
        run(city, quick=args.quick)
    elif args.cmd == "plan":
        from .planning import print_plan
        print_plan(city, args.date, args.out)
    elif args.cmd == "users":
        from .store import Store
        Store(city).seed_demo()
    elif args.cmd == "demo":
        from .demo import seed
        seed(city, days=args.days)


if __name__ == "__main__":
    main()
