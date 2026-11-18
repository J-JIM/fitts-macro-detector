"""Command line interface.

    fittsguard score session.json            judge a recorded session
    fittsguard predict --age 67 -d 400 -w 32  expected movement time / click rate
    fittsguard simulate --agent bezier -n 40 -o bot.json   make a demo session
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .age_model import AgeMotorModel
from .detector import load_detector
from .schema import Session


def cmd_score(args: argparse.Namespace) -> int:
    det = load_detector(args.model)
    session = Session.load(args.session)
    verdict = det.judge(session)
    out = verdict.to_dict()
    out["verdict"] = "BOT" if verdict.is_bot else "HUMAN-LIKE"
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        print(f"{out['verdict']}  ({verdict.n_trials} clicks; strongest test at {verdict.score:.2f}x its alarm line)")
        for t in verdict.tests:
            mark = "FIRED" if t["fired"] else "  ok "
            ev = ", ".join(f"{k.split('_', 1)[1]}={v:.3g}" for k, v in t["evidence"].items())
            print(f"  [{mark}] {t['test']:<24} {t['ratio']:>5.2f}x  {t['meaning']}")
            if t["fired"]:
                print(f"           evidence: {ev}")
    return 1 if verdict.is_bot else 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = AgeMotorModel.load(args.model)
    mt = model.movement_time(args.age, args.distance, args.width)
    hold = model.hold_time(args.age)
    cr = model.click_rate(args.age) if "tap" in model.curves else None
    if args.json:
        print(json.dumps({"movement": mt, "hold": hold, "click": cr}, indent=1))
        return 0
    if args.age < model.curves["a"].age_min or args.age > model.curves["a"].age_max:
        print(f"note: age {args.age:g} is outside the range the model was fitted on "
              f"({model.curves['a'].age_min:g}-{model.curves['a'].age_max:g}); the nearest supported age is used")
    print(f"age {args.age:g}, D={args.distance:g}px, W={args.width:g}px  (ID = {mt['id_bits']:.2f} bits)")
    print(f"  Fitts line        MT = {mt['a']*1000:.0f} ms + {mt['b']*1000:.0f} ms/bit x ID")
    print(f"  movement time     {mt['mt']*1000:.0f} ms   (95% of people: {mt['mt_lo']*1000:.0f} to {mt['mt_hi']*1000:.0f} ms)")
    print(f"  pointing speed    {mt['clicks_per_s']:.2f} clicks/s on this target,  throughput {mt['throughput']:.2f} bits/s")
    def caveat(key: str) -> str:
        c = model.curves[key]
        if c.age_min <= args.age <= c.age_max:
            return ""
        return f"   [no data at this age: value for age {min(max(args.age, c.age_min), c.age_max):g}]"

    print(f"  button hold time  {hold['hold']*1000:.0f} ms   (95% of people: {hold['hold_lo']*1000:.0f} to "
          f"{hold['hold_hi']*1000:.0f} ms){caveat('hold')}")
    if cr:
        print(f"  max click rate    {cr['cps']:.1f} clicks/s on one spot   (95% of people: {cr['cps_lo']:.1f} to "
              f"{cr['cps_hi']:.1f}){caveat('tap')}")
    if args.age < 18:
        print("  note: below 18 the model rests on three child groups from two published studies; treat as indicative")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .bots import default_zoo
    from .human_sim import HumanSim
    from .simulate import random_layout, run_layout

    rng = np.random.default_rng(args.seed)
    layout = random_layout(rng, args.n)
    if args.agent == "human":
        agent = HumanSim(AgeMotorModel.load().profile(args.age, rng))
    else:
        zoo = {b.family: b for b in default_zoo()}
        if args.agent not in zoo:
            print(f"unknown agent {args.agent!r}; choose from: human, {', '.join(zoo)}", file=sys.stderr)
            return 2
        agent = zoo[args.agent]
    session = run_layout(agent, layout, rng, subject=f"sim-{args.agent}", age=args.age if args.agent == "human" else None)
    session.save(args.out)
    print(f"wrote {args.out}: {len(session)} trials by {session.family}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fittsguard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="judge a recorded session (exit code 1 if flagged)")
    s.add_argument("session", type=Path)
    s.add_argument("--model", type=Path, help="human reference JSON (default: packaged)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_score)

    p = sub.add_parser("predict", help="expected movement time and click rate at an age")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("-d", "--distance", type=float, default=400.0, help="distance to target, px")
    p.add_argument("-w", "--width", type=float, default=32.0, help="target width, px")
    p.add_argument("--model", type=Path, help="age model JSON (default: packaged)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_predict)

    g = sub.add_parser("simulate", help="generate a demo session from a simulated person or bot")
    g.add_argument("--agent", default="human")
    g.add_argument("--age", type=float, default=25.0)
    g.add_argument("-n", type=int, default=40)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-o", "--out", type=Path, required=True)
    g.set_defaults(fn=cmd_simulate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
