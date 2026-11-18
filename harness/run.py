"""Harness entry point.

    python harness/run.py --name baseline

Runs every experiment in HARNESS.md on the real-human data, writes
``results/<name>/metrics.json`` and ``report.md``, checks the gates in
``harness/criteria.json`` and exits non-zero if a blocking gate fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import age_experiment  # noqa: E402
from datasets import load_input_accessibility  # noqa: E402
from evaluate import AgeConditioned, _folds, build_frame, cv_oneclass, cv_supervised  # noqa: E402
from fittsguard.bots import TIERS  # noqa: E402
from fittsguard.detector import EnvelopeDetector, LayeredDetector, MacroDetector  # noqa: E402

ROOT = HERE.parent
CACHE = ROOT / "results" / "cache"
KS = (10, 25, 50)
MAIN_K = 25
AGE_BANDS = ((18, 30), (30, 45), (45, 60), (60, 70), (70, 90))


def frame(humans, k: int, seed: int, refresh: bool) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"frame_k{k}_s{seed}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    t = time.time()
    df = build_frame(humans, k, seed=seed, aux_step=max(k // 5, 1))
    df.to_pickle(path)
    print(f"  built frame k={k}: {len(df)} rows in {time.time() - t:.0f}s", flush=True)
    return df


def detectors() -> dict:
    return {
        "layered_fitts": lambda: LayeredDetector(("fitts",)),
        "layered_fitts_kin": lambda: LayeredDetector(("fitts", "kin")),
        "generic_outlier": lambda: EnvelopeDetector(("fitts_", "kin_")),
    }


def human_alarms(df: pd.DataFrame, make_detector, target_fpr: float = 0.01) -> pd.DataFrame:
    """Held-out alarm for every primary human window."""
    hum = df[df.label == 0].reset_index(drop=True)
    alarms = np.zeros(len(hum), dtype=bool)
    for tr, te in _folds(hum, 5):
        det = make_detector().fit(hum.iloc[tr], target_fpr=target_fpr)
        alarms[te] = det.score(hum.iloc[te]) > det.threshold
    return hum.assign(alarm=alarms)[hum.primary]


def subgroup_fpr(df: pd.DataFrame, humans, make_detector) -> dict:
    """False-positive rate on held-out people, split by age band, motor impairment and device."""
    info = {s.subject: s for s in humans}
    hum = human_alarms(df, make_detector)
    alarms = hum.alarm.to_numpy()
    age = hum.subject.map(lambda s: info[s].age).to_numpy(dtype=float)
    imp = hum.subject.map(lambda s: bool(info[s].meta.get("motor_impairment"))).to_numpy()
    dev = hum.subject.map(lambda s: info[s].meta.get("device")).to_numpy()
    out = {"all": {"n": int(len(hum)), "fpr": float(alarms.mean())}}
    for lo, hi in AGE_BANDS:
        m = (age >= lo) & (age < hi)
        out[f"age_{lo}_{hi}"] = {"n": int(m.sum()), "fpr": float(alarms[m].mean())}
    for name, m in (("motor_impairment", imp), ("no_motor_impairment", ~imp),
                    ("mouse", np.isin(dev, ["optical mouse", "mechanical mouse"])), ("touchpad", dev == "touchpad")):
        out[name] = {"n": int(m.sum()), "fpr": float(alarms[m].mean())}
    return out


def tier_summary(res: dict) -> dict:
    tiers: dict[int, list[float]] = {}
    for fam, v in res["families"].items():
        tiers.setdefault(TIERS.get(fam, 9), []).append(v["tpr"])
    return {f"tier{t}": float(np.mean(v)) for t, v in sorted(tiers.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="rebuild cached feature frames")
    ap.add_argument("--export", action="store_true", help="write the packaged human reference and age model")
    args = ap.parse_args()

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    humans = load_input_accessibility()
    print(f"humans: {len(humans)} people, {sum(len(s) for s in humans)} trials", flush=True)
    M: dict = {"name": args.name, "seed": args.seed, "n_people": len(humans),
               "n_trials": int(sum(len(s) for s in humans)), "detection": {}, "supervised": {}, "detection_age": {}}

    # E1: one-class detection by window length
    for k in KS:
        df = frame(humans, k, args.seed, args.refresh)
        M["detection"][f"k{k}"] = {}
        for name, mk in detectors().items():
            res = cv_oneclass(df, mk)
            res["tiers"] = tier_summary(res)
            M["detection"][f"k{k}"][name] = res
            print(f"  k={k:<3} {name:<20} FPR={res['fpr']:.3f}  " +
                  "  ".join(f"{t}={v:.2f}" for t, v in res["tiers"].items()), flush=True)
        # E5: does conditioning the Fitts tests on age change anything?
        M["detection_age"][f"k{k}"] = {}
        for name, mk in (("age_blind", lambda: LayeredDetector(("fitts",))),
                         ("age_conditioned", lambda: AgeConditioned(LayeredDetector(("fitts",))))):
            res = cv_oneclass(df, mk)
            hum = human_alarms(df, mk)
            bands = [float(hum.alarm[(hum.age >= lo) & (hum.age < hi)].mean()) for lo, hi in AGE_BANDS]
            M["detection_age"][f"k{k}"][name] = {"fpr": res["fpr"], "tiers": tier_summary(res),
                                                 "worst_band_fpr": max(bands), "band_fpr": bands}

    df = frame(humans, MAIN_K, args.seed, False)
    # E2: who gets falsely accused
    M["subgroups"] = {name: subgroup_fpr(df, humans, mk) for name, mk in detectors().items()
                      if name.startswith("layered")}

    # E3: supervised reference and generalisation to unseen bot families
    for gname, groups in (("fitts", ("fitts_",)), ("fitts_kin", ("fitts_", "kin_"))):
        res = cv_supervised(df, groups)
        res["tiers"] = tier_summary(res)
        M["supervised"][gname] = res
        lofo = {}
        for fam in sorted(df[df.label == 1].family.unique()):
            r = cv_supervised(df, groups, hold_out_family=fam)
            lofo[fam] = {"tpr": r["families"][fam]["tpr"], "auc": r["families"][fam]["auc"], "fpr": r["fpr"]}
        M["supervised"][gname + "_leave_family_out"] = lofo
        print(f"  supervised {gname:<10} FPR={res['fpr']:.3f}  " +
              "  ".join(f"{t}={v:.2f}" for t, v in res["tiers"].items()), flush=True)

    # E4: age model
    M["age"] = age_experiment.run(humans, out, save_model=args.export)

    if args.export:   # packaged reference: fitted on every person, both layers, 25-click windows
        det = LayeredDetector(("fitts", "kin")).fit(df[df.label == 0])
        dest = ROOT / "src" / "fittsguard" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        MacroDetector(det, window=MAIN_K).save(dest / "human_reference.json")

    gates = check_gates(M)
    M["gates"] = gates
    (out / "metrics.json").write_text(json.dumps(M, indent=1))
    from report import write_report
    write_report(M, out / "report.md")
    blocking_fail = [g for g in gates if g["blocking"] and not g["passed"]]
    for g in gates:
        tag = "PASS" if g["passed"] else ("OPEN" if g["open"] else "FAIL")
        print(f"  [{tag}]{'' if g['blocking'] or g['open'] else ' (stretch)'} {g['id']}: "
              f"{g['desc']} -> {g['value']:.3f} (need {g['op']} {g['limit']})")
    return 1 if blocking_fail else 0


def _get(M: dict, path: str):
    cur = M
    for p in path.split("/"):
        cur = cur[p]
    return cur


def check_gates(M: dict) -> list[dict]:
    spec = json.loads((HERE / "criteria.json").read_text())
    out = []
    for g in spec["gates"]:
        if "path" in g:
            val = float(_get(M, g["path"]))
        else:   # aggregate over several paths
            vals = [float(_get(M, p)) for p in g["paths"]]
            val = {"min": min, "max": max}[g["agg"]](vals)
        ok = val <= g["limit"] if g["op"] == "<=" else val >= g["limit"]
        out.append({"id": g["id"], "desc": g["desc"], "value": val, "op": g["op"], "limit": g["limit"],
                    "blocking": g.get("blocking", True), "open": g.get("open", False), "passed": bool(ok)})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
