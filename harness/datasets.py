"""Loaders that turn third-party recordings into fittsguard Sessions.

Raw data lives in ``data/raw/`` (fetched by ``harness/fetch_data.py``, never
committed). Parsed sessions are cached in ``data/processed/``.

Input Accessibility (Findlater & Zhang, ASSETS 2020, CC BY 4.0)
    414 people aged 18-83 doing an ISO-style 2-D pointing task with a mouse or
    touchpad in their own browser. Each trial: click a start area, then click a
    circular target of known centre and width. Every pointer event is logged.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np

from fittsguard.schema import Session, Trial

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

_FNAME = re.compile(r"UserId=(\d+)_.*DeviceType=DESKTOP_.*TaskName=Pointing_BlockIndex=(\d+)_")


def _parse_trial(tr: dict) -> Trial | None:
    """Start-area click -> target click. None if the log does not contain both."""
    ev = tr.get("mouseEvents") or []
    active = [e["t"] for e in tr.get("taskEvents", []) if e.get("e") == "startAreaActive"]
    if not active or "target" not in tr:
        return None
    t_active = active[-1]
    downs = [e for e in ev if e["e"] == "mousedown"]
    ups = [e for e in ev if e["e"] == "mouseup"]
    start = [e for e in downs if e["t"] <= t_active]
    after = [e for e in downs if e["t"] > t_active]
    if not start or not after:
        return None
    s, c = start[-1], after[0]          # the first click after leaving the start area, hit or miss
    t0 = s["t"]
    up = next((e["t"] for e in ups if e["t"] >= c["t"]), None)
    moves = [e for e in ev if e["e"] == "mousemove" and t0 < e["t"] <= c["t"]]
    t = np.array([0.0] + [(e["t"] - t0) / 1000.0 for e in moves])
    x = np.array([s["p"]["X"]] + [e["p"]["X"] for e in moves], dtype=float)
    y = np.array([s["p"]["Y"]] + [e["p"]["Y"] for e in moves], dtype=float)
    keep = np.concatenate([[True], np.diff(t) > 0])   # ms clock: drop same-timestamp duplicates
    tg = tr["target"]
    if not tg.get("width") or c["t"] <= t0:
        return None
    return Trial(t=t[keep], x=x[keep], y=y[keep],
                 target_x=float(tg["center"]["X"]), target_y=float(tg["center"]["Y"]), target_w=float(tg["width"]),
                 t_down=(c["t"] - t0) / 1000.0, t_up=None if up is None else (up - t0) / 1000.0,
                 click_x=float(c["p"]["X"]), click_y=float(c["p"]["Y"]))


def _participants(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(root / "ParticipantQuestionnaireData N=735.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["DeviceType"] != "DESKTOP":
                continue
            out[row["ParticipantId"]] = {
                "age": float(row["Age"]) if row["Age"] not in ("", "NULL") else None,
                "motor_impairment": row["SelfReportMotorImpairment"] == "1",
                "device": row["SelfReportPointingDevice"],
                "crowdsourced": row["IsCrowdsourced"] == "1",
                "gender": row["Gender"],
            }
    return out


def load_input_accessibility(refresh: bool = False) -> list[Session]:
    """One Session per participant: both pointing blocks, in order."""
    cache = PROCESSED / "input_accessibility.pkl"
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    root = RAW / "input_accessibility"
    if not root.exists():
        raise FileNotFoundError("run `python harness/fetch_data.py` first")
    people = _participants(root)
    blocks: dict[str, list[tuple[int, list[Trial], int]]] = {}
    for path in sorted(root.glob("alldata/*/*DeviceType=DESKTOP*TaskName=Pointing*.json")):
        m = _FNAME.search(path.name)
        if not m:
            continue
        block = json.loads(path.read_text())
        if block.get("practice") or block.get("skipped"):
            continue
        raw_trials = block.get("trials", [])
        trials = [t for t in (_parse_trial(tr) for tr in raw_trials) if t is not None]
        blocks.setdefault(m.group(1), []).append((int(m.group(2)), trials, len(raw_trials)))
    sessions = []
    for uid, bl in blocks.items():
        bl.sort(key=lambda b: b[0])
        trials = [t for _, ts, _ in bl for t in ts]
        if len(trials) < 10:
            continue
        info = people.get(uid, {})
        sessions.append(Session(trials, label="human", family="input_accessibility", subject=uid,
                                age=info.get("age"),
                                meta={**info, "n_logged": sum(n for _, _, n in bl), "n_blocks": len(bl)}))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(sessions))
    return sessions


DATASETS = {"input_accessibility": load_input_accessibility}
