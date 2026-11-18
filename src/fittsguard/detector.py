"""Detectors.

The detector is *one-class*: it is fitted on human sessions only and never
sees a bot. Its performance on any bot family is therefore performance on an
attack it was not trained for.

It is a short list of named, one-sided tests. Each asks one question that a
person's motor system answers one way and a program tends to answer another:

Fitts layer (needs click times, click positions, target geometry):
    superhuman_throughput   more bits per second than people produce
    task_blind_timing       time does not grow with difficulty / shrink with target size
    machine_regular_timing  residual scatter around the Fitts line too small
    machine_precise_clicks  endpoint scatter too small for the target sizes

Kinematic layer (needs the cursor trajectory):
    too_smooth, too_straight, stereotyped, no_verification_pause

Only deviations *toward the machine side* count. People are allowed to be
slow, erratic and sloppy: a pause, a wild misclick or a tremor never raises the
score. This is what keeps the false-positive rate flat across age and motor
ability, and it is also why a generic outlier detector does badly here: real
people have heavy tails in every direction, and those tails set its threshold.

Two kinds of test:

hard   the human and machine ranges do not overlap (a program that clicks dead
       centre every time). Threshold sits outside the observed human range,
       with a margin, and costs essentially no false positives.
stat   the ranges overlap and evidence accumulates with the number of clicks
       (is the Fitts slope distinguishable from zero?). These share the
       false-positive budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import extract_features
from .schema import Session

Z_CAP = 12.0


@dataclass(frozen=True)
class TestSpec:
    name: str
    layer: str                       # "fitts" | "kin"
    kind: str                        # "hard" | "stat"
    parts: tuple[tuple[str, int], ...]   # (feature, +1 suspicious when high / -1 suspicious when low)
    weight: float = 1.0              # share of the false-positive budget among stat tests
    what: str = ""


TESTS: tuple[TestSpec, ...] = (
    TestSpec("superhuman_throughput", "fitts", "hard", (("fitts_tpe", 1), ("fitts_tp_p90", 1)),
             what="more bits per second than people produce"),
    TestSpec("machine_regular_timing", "fitts", "hard", (("fitts_resid_cv", -1),),
             what="click times sit on a line with almost no scatter"),
    TestSpec("machine_precise_clicks", "fitts", "hard",
             (("fitts_end_sd_along", -1), ("fitts_end_sd_ortho", -1), ("fitts_end_exact", 1)),
             what="clicks land with less scatter than the target sizes allow a person"),
    # Confidence bounds, not point estimates or R^2: an erratic person has a noisy slope
    # (wide bounds, never "confidently zero") and a low R^2. R^2 would flag them; the bounds do not.
    TestSpec("task_blind_timing", "fitts", "stat", (("fitts_b_ucb", -1), ("fitts_bw_lcb", 1)),
             weight=3.0, what="time does not depend on distance and target size the way Fitts' law requires"),
    # Kinematic tests separate people from programs by a wide margin, so each needs only a
    # sliver of the false-positive budget; most of it goes to the Fitts timing test, whose
    # human and machine ranges genuinely overlap at small click counts.
    TestSpec("no_verification_pause", "kin", "stat", (("kin_settle_med", -1),), weight=0.25,
             what="button goes down the instant the cursor arrives"),
    TestSpec("too_smooth", "kin", "stat", (("kin_mjcorr_med", 1), ("kin_ldlj_med", 1)), weight=0.25,
             what="speed profiles are cleaner bells than a hand produces"),
    TestSpec("too_straight", "kin", "stat", (("kin_straight_med", 1), ("kin_maxdev_med", -1)), weight=0.25,
             what="paths are straighter than a hand produces"),
    TestSpec("stereotyped", "kin", "stat",
             (("kin_ttp_iqr", -1), ("kin_straight_iqr", -1), ("kin_ldlj_iqr", -1), ("kin_settle_iqr", -1),
              ("kin_latency_iqr", -1)), weight=0.25,
             what="every movement looks like every other movement"),
)
# Deliberately absent: "peak speed does not scale with distance". It catches constant-speed
# bots, but slow people who move in many small steps show the same pattern.


@dataclass
class _Marginal:
    """Human distribution of one feature: empirical CDF -> normal score, linear tails."""

    xs: list[float]
    scale: float
    present: float      # fraction of human sessions where the feature is defined

    @classmethod
    def fit(cls, v: np.ndarray) -> "_Marginal":
        ok = np.isfinite(v)
        x = np.sort(v[ok])
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        scale = max((q75 - q25) / 1.349, 1e-3 * max(abs(q50), 1e-3), 1e-9)
        # keep a compact quantile sketch so the reference stays small and holds no individual's data
        if len(x) > 257:
            x = np.quantile(x, np.linspace(0.0, 1.0, 257))
        return cls([float(a) for a in x], float(scale), float(ok.mean()))

    def z(self, v: np.ndarray, direction: int, nan_is_machine: bool = True) -> np.ndarray:
        """One-sided normal score: large positive = far toward the machine side."""
        x = np.asarray(self.xs)
        n = len(x)
        vv = np.where(np.isfinite(v), v, x[n // 2])
        r = (np.searchsorted(x, vv, side="left") + np.searchsorted(x, vv, side="right")) / 2.0
        z = norm.ppf(np.clip(r, 0.5, n - 0.5) / n)
        zmax = norm.ppf((n - 0.5) / n)
        z = np.where(vv > x[-1], zmax + (vv - x[-1]) / self.scale, z)
        z = np.where(vv < x[0], -zmax + (vv - x[0]) / self.scale, z)
        z = np.clip(z * direction, -Z_CAP, Z_CAP)
        # Fitts layer: a statistic people always have but this session lacks (zero variance,
        # so no correlation can be computed) is itself machine-like.
        # Kinematic layer: a missing trajectory is no evidence either way; some real
        # devices (touch, pen) deliver clicks with almost no pointer movement.
        return np.where(np.isfinite(v), z, Z_CAP if (nan_is_machine and self.present >= 0.99) else 0.0)


class LayeredDetector:
    """Named one-sided tests; alarm when any test passes its own threshold.

    ``score`` is max over tests of (statistic / threshold), so 1.0 is the alarm line.
    """

    def __init__(self, layers: tuple[str, ...] = ("fitts", "kin"), hard_margin: float = 1.0,
                 tests: tuple[TestSpec, ...] = TESTS, combine: str = "p4", hard_q: float = 1.0):
        self.layers = tuple(layers)
        self.hard_margin = hard_margin
        self.hard_q = hard_q
        self.combine = combine
        self.tests = [t for t in tests if t.layer in self.layers]
        self.marg: dict[str, _Marginal] = {}
        self.thr: dict[str, float] = {}
        self.threshold = 1.0
        self.target_fpr = 0.01

    # -- statistics ---------------------------------------------------------
    def _stat(self, t: TestSpec, F: pd.DataFrame) -> np.ndarray:
        zs = []
        for feat, direction in t.parts:
            if feat not in self.marg:
                continue
            v = F[feat].to_numpy(dtype=float) if feat in F.columns else np.full(len(F), np.nan)
            zs.append(self.marg[feat].z(v, direction, nan_is_machine=(t.layer == "fitts")))
        if not zs:
            return np.zeros(len(F))
        Z = np.vstack(zs)
        if t.kind == "hard":          # any one measure far outside the human range is enough
            return Z.max(axis=0)
        if self.combine == "mean":    # related measures averaged: agreement is what counts
            return Z.mean(axis=0)
        # power mean of the machine-ward parts: rewards agreement between measures, but one
        # strong measure is not diluted by neutral ones (a bot blind to width but not distance)
        p = 4.0 if self.combine == "p4" else 2.0
        return np.mean(np.maximum(Z, 0.0) ** p, axis=0) ** (1.0 / p)

    def fit(self, F: pd.DataFrame, target_fpr: float = 0.01) -> "LayeredDetector":
        self.target_fpr = target_fpr
        feats = {f for t in self.tests for f, _ in t.parts}
        self.marg = {f: _Marginal.fit(F[f].to_numpy(dtype=float)) for f in feats
                     if f in F.columns and np.isfinite(F[f]).sum() >= 20}
        stat = [t for t in self.tests if t.kind == "stat"]
        wsum = sum(t.weight for t in stat) or 1.0
        self.thr = {}
        for t in self.tests:
            s = self._stat(t, F)
            if t.kind == "hard":
                # just outside the human range; the quantile (not the maximum) keeps one
                # freak session, e.g. a burst of accidental double clicks, from setting the line
                self.thr[t.name] = float(np.quantile(s, self.hard_q) + self.hard_margin)
            else:
                alpha = target_fpr * t.weight / wsum
                self.thr[t.name] = float(max(np.quantile(s, 1.0 - alpha), 0.5))
        return self

    def stats(self, F: pd.DataFrame) -> pd.DataFrame:
        """Per-test statistic divided by its threshold (>= 1 means that test fires)."""
        return pd.DataFrame({t.name: self._stat(t, F) / self.thr[t.name] for t in self.tests}, index=F.index)

    def score(self, F: pd.DataFrame) -> np.ndarray:
        return self.stats(F).to_numpy().max(axis=1)

    def explain(self, feats: dict[str, float]) -> list[dict]:
        F = pd.DataFrame([feats])
        ratios = self.stats(F).iloc[0]
        out = []
        for t in self.tests:
            out.append({
                "test": t.name, "layer": t.layer, "fired": bool(ratios[t.name] > 1.0),
                "ratio": float(ratios[t.name]), "meaning": t.what,
                "evidence": {f: float(feats.get(f, float("nan"))) for f, _ in t.parts},
            })
        return sorted(out, key=lambda r: -r["ratio"])

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": "layered", "layers": list(self.layers), "hard_margin": self.hard_margin, "combine": self.combine,
                "target_fpr": self.target_fpr, "thr": self.thr,
                "marg": {k: {"xs": m.xs, "scale": m.scale, "present": m.present} for k, m in self.marg.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "LayeredDetector":
        det = cls(tuple(d["layers"]), d["hard_margin"], combine=d.get("combine", "p4"))
        det.target_fpr = d["target_fpr"]
        det.thr = d["thr"]
        det.marg = {k: _Marginal(**m) for k, m in d["marg"].items()}
        return det


@dataclass
class Verdict:
    is_bot: bool
    score: float            # >= 1.0 means at least one test fired
    n_trials: int
    fired: list[str]
    tests: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"is_bot": self.is_bot, "score": self.score, "n_trials": self.n_trials,
                "fired": self.fired, "tests": self.tests}


class MacroDetector:
    """End-to-end: Session -> Verdict, with a persisted human reference.

    The reference is specific to a window length: a statistic computed from 25
    clicks is compared with what people produce in 25 clicks. Sessions longer
    than the window are judged window by window and flagged when a majority of
    windows are.
    """

    def __init__(self, detector: LayeredDetector, window: int = 25):
        self.detector = detector
        self.window = window

    def judge(self, session: Session) -> Verdict:
        if len(session) < self.window:
            raise ValueError(f"need at least {self.window} clicks on known targets, got {len(session)}")
        wins = list(session.windows(self.window))
        verdicts = []
        for w in wins:
            feats = extract_features(w)
            tests = self.detector.explain(feats)
            verdicts.append((max(t["ratio"] for t in tests), tests))
        n_fired = sum(v[0] > 1.0 for v in verdicts)
        is_bot = n_fired * 2 > len(verdicts) if len(verdicts) > 1 else n_fired == 1
        score, tests = max(verdicts, key=lambda v: v[0])
        return Verdict(bool(is_bot), float(score), len(session), [t["test"] for t in tests if t["fired"]], tests)

    def to_dict(self) -> dict:
        return {"window": self.window, "detector": self.detector.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "MacroDetector":
        return cls(LayeredDetector.from_dict(d["detector"]), d["window"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "MacroDetector":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_detector(path: str | Path | None = None) -> MacroDetector:
    """The packaged detector (human reference from 414 real people), or one saved at ``path``."""
    if path is not None:
        return MacroDetector.load(path)
    from importlib import resources
    text = resources.files("fittsguard").joinpath("data/human_reference.json").read_text()
    return MacroDetector.from_dict(json.loads(text))


# --------------------------------------------------------------------------
# Generic outlier detector, kept as the baseline the layered design is measured against
# --------------------------------------------------------------------------
class EnvelopeDetector:
    """Two-sided robust z-scores on every feature; score = RMS of the largest few.

    This is the obvious first design and it fails on real people (see
    docs/RESEARCH_LOG.md, iteration 1): their heavy tails set the threshold.
    """

    def __init__(self, groups: tuple[str, ...] = ("fitts_", "kin_"), top: int = 3):
        self.groups, self.top = groups, top

    def fit(self, F: pd.DataFrame, target_fpr: float = 0.01) -> "EnvelopeDetector":
        self.cols = [c for c in F.columns if c.startswith(self.groups) and np.isfinite(F[c]).sum() >= 8]
        X = F[self.cols].to_numpy(dtype=float)
        self.med = np.nanmedian(X, axis=0)
        q05, q25, q75, q95 = np.nanpercentile(X, [5, 25, 75, 95], axis=0)
        floor = np.maximum(0.25 * (q75 - q25) / 1.349, 1e-6)
        self.lo = np.maximum((self.med - q05) / 1.645, floor)
        self.hi = np.maximum((q95 - self.med) / 1.645, floor)
        self.present = np.mean(np.isfinite(X), axis=0)
        self.threshold = float(np.quantile(self.score(F), 1.0 - target_fpr))
        return self

    def score(self, F: pd.DataFrame) -> np.ndarray:
        X = F.reindex(columns=self.cols).to_numpy(dtype=float)
        d = X - self.med
        z = np.where(d >= 0, d / self.hi, d / self.lo)
        z = np.where(np.isfinite(X), z, np.where(self.present >= 0.95, Z_CAP, 0.0))
        z = np.abs(np.clip(z, -Z_CAP, Z_CAP))
        k = min(self.top, z.shape[1])
        return np.sqrt(np.mean(np.sort(z, axis=1)[:, -k:] ** 2, axis=1))
