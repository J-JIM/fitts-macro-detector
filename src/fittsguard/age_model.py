"""Age-conditional human motor model.

Question it answers: *for a person of a given age, what movement time and what
click speed should we expect, and how wide is the normal range?*

    MT(age, D, W) = a(age) + b(age) * log2(D / W + 1)
    a(age) = a_ref * exp(g_a(age)),   b(age) = b_ref * exp(g_b(age)),   g(REF_AGE) = 0

Every quantity (Fitts intercept ``a``, slope ``b``, button hold time, maximum
repetitive click rate) has its own lifespan curve ``g``. Each curve is

    g(age) = dev_c * exp(-(age - 4) / dev_tau)  +  ageing * max(0, age - knot)     (shifted so g(25) = 0)

a childhood deficit that fades within a few years, a flat stretch for young
adults, and log-linear slowing after ``knot``.

Where the numbers come from (see ``harness/age_experiment.py``):

* the ageing side (18-83 y) is fitted to 414 real people doing a Fitts task
  with their own mouse (Findlater & Zhang 2020), with device and motor
  impairment as covariates;
* the development side (children) and the click-rate curve cannot come from
  that data, so they are fitted to group means from published studies in
  ``data/literature/age_groups.csv``. Published absolute values are not
  comparable across studies (device, task, what "movement time" includes), so
  only *within-study ratios between age groups* are used: each study gets its
  own intercept.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .human_sim import MotorProfile

REF_AGE = 25.0
CHILD_AGE0 = 4.0


def lifespan(age, dev_c: float, dev_tau: float, ageing: float, knot: float) -> np.ndarray:
    """g(age): childhood deficit that decays exponentially, plus log-linear ageing after ``knot``."""
    age = np.asarray(age, dtype=float)
    child = dev_c * (np.exp(-(age - CHILD_AGE0) / dev_tau) - np.exp(-(REF_AGE - CHILD_AGE0) / dev_tau)) if dev_c else 0.0
    return child + ageing * np.maximum(0.0, age - knot)


@dataclass
class AgeCurve:
    """log-multiplier relative to a young adult: exp(g(age))."""

    dev_c: float = 0.0        # childhood deficit at age 4 (log units); 0 = no child data
    dev_tau: float = 2.5      # years over which that deficit shrinks by 1/e
    ageing: float = 0.0       # per year above ``knot`` (log units)
    knot: float = 30.0
    age_min: float = 18.0     # youngest age supported by data; younger ages are clamped
    age_max: float = 83.0
    se_log: float = 0.0       # uncertainty of the curve itself (log units)
    child_se_log: float = 0.0  # extra uncertainty below 18, where only a few published groups exist
    source: str = ""

    def se(self, age: float) -> float:
        return float(np.hypot(self.se_log, self.child_se_log)) if age < 18.0 else self.se_log

    def g(self, age) -> np.ndarray:
        return lifespan(np.clip(age, self.age_min, self.age_max), self.dev_c, self.dev_tau, self.ageing, self.knot)

    def factor(self, age) -> np.ndarray:
        return np.exp(self.g(age))


def fit_ratio_curve(table: pd.DataFrame, metric: str, knot: float = 30.0) -> tuple[AgeCurve, dict]:
    """Fit ``dev`` and ``ageing`` to published group means (columns: study, metric, age, value, n).

    Each study has its own intercept, so only ratios between age groups of the
    same study matter. Returns the curve and a leave-one-study-out report.
    """
    d = table[table["metric"] == metric].copy()
    d = d[d.groupby("study")["age"].transform("nunique") >= 2]
    if d["study"].nunique() < 2:
        raise ValueError(f"metric {metric!r}: need two or more multi-group studies, got {d['study'].nunique()}")

    def fit(part: pd.DataFrame) -> np.ndarray:
        ly = np.log(part["value"].to_numpy(dtype=float))
        age = part["age"].to_numpy(dtype=float)
        wt = np.sqrt(part["n"].to_numpy(dtype=float))
        codes = pd.factorize(part["study"])[0]
        has_child, has_old = (age < 16.0).any(), (age > knot).any()

        def resid(th):
            r = ly - lifespan(age, th[0] if has_child else 0.0, th[1], th[2] if has_old else 0.0, knot)
            for c in np.unique(codes):
                m = codes == c
                r[m] -= np.average(r[m], weights=wt[m] ** 2)
            return r * wt

        th = least_squares(resid, np.array([1.0, 2.5, 0.005]), bounds=([-5.0, 1.0, -0.1], [5.0, 8.0, 0.1])).x
        return np.array([th[0] if has_child else 0.0, th[1], th[2] if has_old else 0.0])

    errs = []
    for held in d["study"].unique():
        tr, te = d[d["study"] != held], d[d["study"] == held]
        if tr["study"].nunique() < 1:
            continue
        th = fit(tr)
        r = np.log(te["value"].to_numpy(dtype=float)) - lifespan(te["age"].to_numpy(dtype=float), th[0], th[1], th[2], knot)
        errs.extend(r - np.average(r, weights=te["n"].to_numpy(dtype=float)))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    th = fit(d)
    curve = AgeCurve(float(th[0]), float(th[1]), float(th[2]), knot, float(d["age"].min()), float(d["age"].max()), rmse,
                     f"literature ratios, {d['study'].nunique()} studies / {len(d)} groups")
    return curve, {"loso_rmse_log": rmse, "n_studies": int(d["study"].nunique()), "n_groups": int(len(d))}


@dataclass
class Baseline:
    """Reference-age (25 y) values on the measured platform, and normal spread."""

    a: float = 0.28               # Fitts intercept, s (start click to target click)
    b: float = 0.15               # Fitts slope, s/bit
    hold: float = 0.095           # button hold time, s
    cps: float = 6.5              # maximum repetitive click rate, clicks/s
    cv_between: float = 0.20      # between-person s.d. of log movement time
    cv_hold: float = 0.25         # between-person s.d. of log hold time
    cv_cps: float = 0.12          # between-person s.d. of log click rate
    source: str = "default"


@dataclass
class AgeMotorModel:
    curves: dict[str, AgeCurve] = field(default_factory=dict)   # keys: a, b, hold, tap
    baseline: Baseline = field(default_factory=Baseline)

    def _f(self, key: str, age) -> float:
        c = self.curves.get(key)
        return float(c.factor(age)) if c is not None else 1.0

    def _se(self, key: str, age: float = 30.0) -> float:
        c = self.curves.get(key)
        return c.se(age) if c is not None else 0.0

    # -- predictions --------------------------------------------------------
    def fitts_params(self, age: float) -> tuple[float, float]:
        """Expected Fitts intercept and slope (s, s/bit) at this age."""
        return self.baseline.a * self._f("a", age), self.baseline.b * self._f("b", age)

    def movement_time(self, age: float, distance: float, width: float, z: float = 1.96) -> dict[str, float]:
        """Expected time from the previous click to the click on this target."""
        a, b = self.fitts_params(age)
        idb = float(np.log2(distance / width + 1.0))
        mean = a + b * idb
        s = float(np.hypot(self.baseline.cv_between, max(self._se("a", age), self._se("b", age))))
        return {"age": age, "id_bits": idb, "a": a, "b": b, "mt": mean,
                "mt_lo": mean * float(np.exp(-z * s)), "mt_hi": mean * float(np.exp(z * s)),
                "throughput": idb / mean, "clicks_per_s": 1.0 / mean}

    def hold_time(self, age: float, z: float = 1.96) -> dict[str, float]:
        mean = self.baseline.hold * self._f("hold", age)
        s = float(np.hypot(self.baseline.cv_hold, self._se("hold")))
        return {"age": age, "hold": mean, "hold_lo": mean * float(np.exp(-z * s)), "hold_hi": mean * float(np.exp(z * s))}

    def click_rate(self, age: float, z: float = 1.96) -> dict[str, float]:
        """Expected maximum repetitive click rate (same spot, as fast as possible)."""
        mean = self.baseline.cps * self._f("tap", age)
        s = float(np.hypot(self.baseline.cv_cps, self._se("tap")))
        return {"age": age, "cps": mean, "cps_lo": mean * float(np.exp(-z * s)), "cps_hi": mean * float(np.exp(z * s)),
                "interval_ms": 1000.0 / mean}

    def profile(self, age: float, rng: np.random.Generator | None = None) -> MotorProfile:
        """Motor profile for the human simulator; a random individual if ``rng`` is given."""
        a, b = self.fitts_params(age)
        ka = self._f("a", age)
        p = MotorProfile(a=a, b=b, latency=MotorProfile.latency * ka, settle=MotorProfile.settle * ka,
                         hold=self.baseline.hold * self._f("hold", age))
        if rng is None:
            return p
        from .human_sim import perturb
        return perturb(p, rng, cv=self.baseline.cv_between)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"baseline": asdict(self.baseline), "curves": {k: asdict(v) for k, v in self.curves.items()}}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> "AgeMotorModel":
        return cls({k: AgeCurve(**v) for k, v in d["curves"].items()}, Baseline(**d["baseline"]))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AgeMotorModel":
        if path is None:
            text = resources.files("fittsguard").joinpath("data/age_model.json").read_text()
        else:
            text = Path(path).read_text()
        return cls.from_dict(json.loads(text))
