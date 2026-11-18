"""E6: how many clicks does the Fitts timing test need, and how much does task design help?

    python harness/power_analysis.py

The real data only contain targets between 1.9 and 4.1 bits and at most 55
trials per person, so this question cannot be answered by measurement alone.
This script answers it by *model-based extrapolation* and says so:

* each real person keeps their own fitted Fitts line and their own residuals
  (resampled), and is replayed on a new set of targets;
* the first scenario replays the dataset's own task, so the extrapolation can
  be checked against the measured detection rates before it is trusted
  elsewhere;
* the bots draw their click time at random, independent of the target, with
  the same delay distributions as ``bezier`` and ``ease_tween`` in the zoo.

Only the ``task_blind_timing`` test is evaluated. Assumption that matters:
Fitts' law keeps holding outside the 1.9-4.1 bit range for these people.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

from datasets import load_input_accessibility  # noqa: E402
from fittsguard.detector import TESTS, LayeredDetector  # noqa: E402
from fittsguard.features import extract_features  # noqa: E402
from fittsguard.schema import Session, Trial  # noqa: E402

ROOT = HERE.parent
TASK_BLIND = tuple(t for t in TESTS if t.name == "task_blind_timing")
SCENARIOS = {
    # name: (distances px, widths px) -> index-of-difficulty range
    "dataset task (1.9-4.1 bits)": ((250.0, 500.0), (32.0, 64.0, 96.0)),
    "typical web UI (1.0-5.2 bits)": ((80.0, 160.0, 320.0, 640.0, 900.0), (24.0, 48.0, 96.0)),
    "mixed sizes incl. small (0.7-6.2 bits)": ((80.0, 160.0, 320.0, 640.0, 900.0), (12.0, 24.0, 48.0, 96.0, 128.0)),
}
BOTS = {   # click time independent of the target: movement + wait + pause, as in bots.py
    "bezier-like timing": lambda rng, n: rng.uniform(0.3, 0.9, n) + rng.uniform(0.1, 0.4, n) + rng.uniform(0.03, 0.12, n),
    "tween-like timing": lambda rng, n: rng.uniform(0.2, 0.6, n) + rng.uniform(0.1, 0.4, n) + rng.uniform(0.02, 0.08, n),
}


def person_models(humans: list[Session]) -> list[dict]:
    out = []
    for s in humans:
        d = np.array([t.distance for t in s.trials])
        w = np.array([t.target_w for t in s.trials])
        mt = np.array([t.movement_time for t in s.trials])
        idb = np.log2(d / w + 1.0)
        b, a, _, _ = theilslopes(mt, idb)
        ratio = mt / np.maximum(a + b * idb, 0.05)
        ratio = ratio[(ratio > 0.2) & (ratio < 5.0)]
        if len(ratio) >= 20:
            out.append({"subject": s.subject, "a": float(a), "b": float(b), "ratio": ratio})
    return out


def _session(d: np.ndarray, w: np.ndarray, mt: np.ndarray, rng: np.random.Generator) -> Session:
    trials = []
    for di, wi, ti in zip(d, w, mt):
        ang = rng.uniform(0, 2 * np.pi)
        tx, ty = 1000 + di * np.cos(ang), 1000 + di * np.sin(ang)
        trials.append(Trial(t=[0.0, ti], x=[1000.0, tx], y=[1000.0, ty], target_x=tx, target_y=ty, target_w=wi,
                            t_down=float(ti), click_x=tx, click_y=ty))
    return Session(trials)


def run(seed: int = 0, ks=(10, 25, 50, 100, 200), reps: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    people = person_models(load_input_accessibility())
    res: dict = {"n_people": len(people), "scenarios": {}}
    cols = ["fitts_b_ucb", "fitts_bw_lcb"]
    for name, (ds, ws) in SCENARIOS.items():
        res["scenarios"][name] = {}
        for k in ks:
            rows = []
            for p in people:
                for _ in range(reps):
                    d, w = rng.choice(ds, k), rng.choice(ws, k)
                    idb = np.log2(d / w + 1.0)
                    mt = np.maximum(p["a"] + p["b"] * idb, 0.05) * rng.choice(p["ratio"], k)
                    f = extract_features(_session(d, w, mt, rng))
                    rows.append({"subject": p["subject"], "who": "human", **{c: f[c] for c in cols}})
                    for bname, draw in BOTS.items():
                        f = extract_features(_session(d, w, draw(rng, k), rng))
                        rows.append({"subject": p["subject"], "who": bname, **{c: f[c] for c in cols}})
            df = pd.DataFrame(rows)
            subj = df["subject"].unique()
            fold = dict(zip(subj, rng.permutation(len(subj)) % 5))
            f_id = df["subject"].map(fold).to_numpy()
            alarms = np.zeros(len(df), dtype=bool)
            for i in range(5):
                train = df[(f_id != i) & (df["who"] == "human")]
                det = LayeredDetector(("fitts",), tests=TASK_BLIND).fit(train, target_fpr=0.01)
                alarms[f_id == i] = det.score(df[f_id == i]) > det.threshold
            df["alarm"] = alarms
            res["scenarios"][name][f"k{k}"] = {who: float(g["alarm"].mean()) for who, g in df.groupby("who")}
            print(f"  {name:<40} k={k:<4} " + "  ".join(f"{w_}={v:.2f}" for w_, v in res["scenarios"][name][f"k{k}"].items()),
                  flush=True)
    return res


if __name__ == "__main__":
    # The 1 % threshold is set by a handful of simulated people, so single runs are
    # noisy. Average over seeds and keep every run so the spread stays visible.
    # Each seed is cached, so seeds can be run one at a time (or in parallel) and combined later.
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    cache = ROOT / "results" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    runs = []
    for s in seeds:
        path = cache / f"power_s{s}.json"
        if not path.exists():
            path.write_text(json.dumps(run(seed=s)))
        runs.append(json.loads(path.read_text()))
    mean: dict = {}
    for name in SCENARIOS:
        mean[name] = {}
        for k in runs[0]["scenarios"][name]:
            mean[name][k] = {who: {"mean": float(np.mean([r["scenarios"][name][k][who] for r in runs])),
                                   "min": float(np.min([r["scenarios"][name][k][who] for r in runs])),
                                   "max": float(np.max([r["scenarios"][name][k][who] for r in runs]))}
                             for who in runs[0]["scenarios"][name][k]}
    dest = ROOT / "results" / "power_analysis.json"
    dest.write_text(json.dumps({"seeds": seeds, "n_people": runs[0]["n_people"], "mean": mean, "runs": runs}, indent=1))
    print(f"wrote {dest}")
