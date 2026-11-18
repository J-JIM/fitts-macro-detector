"""Evaluation machinery shared by every harness experiment.

Protocol (fixed; see HARNESS.md):

* humans are real recorded sessions, cut into windows of ``k`` consecutive trials;
* every bot family clicks through *the same target sequence* as each human
  window, so task layout cannot separate the classes;
* folds are split by human subject: nobody appears in both train and test;
* one-class detectors are fitted on training humans only. Their alarm
  threshold is the (1 - target FPR) quantile of training-human scores, and the
  false-positive rate reported is the one *realised on held-out people*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from fittsguard.bots import default_zoo
from fittsguard.features import extract_features
from fittsguard.human_sim import HumanSim, MotorProfile, perturb
from fittsguard.schema import Session
from fittsguard.simulate import Layout, run_layout

META = ["label", "family", "subject", "window", "age", "primary"]


class SimBot(HumanSim):
    """The human simulator used as an attacker."""

    label = "bot"
    family = "sim_bot"

    def new_session(self, rng):
        self.profile = perturb(MotorProfile(), rng)


def build_frame(humans: list[Session], k: int, seed: int = 0, step: int | None = None,
                max_windows_per_subject: int | None = None, with_bots: bool = True,
                aux_step: int | None = None) -> pd.DataFrame:
    """One row per (window, agent). Humans first, then each bot on the same layouts."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    by_subject: dict[str, list[Session]] = {}
    for s in humans:
        by_subject.setdefault(s.subject, []).append(s)
    for subj, sessions in by_subject.items():
        windows = [w for s in sessions for w in s.windows(k, step)]
        if max_windows_per_subject and len(windows) > max_windows_per_subject:
            idx = rng.choice(len(windows), max_windows_per_subject, replace=False)
            windows = [windows[i] for i in sorted(idx)]
        pool = [t for o, ss in by_subject.items() if o != subj for s in ss for t in s.trials]
        if len(pool) > 4000:
            pool = [pool[i] for i in rng.choice(len(pool), 4000, replace=False)]
        zoo = (default_zoo(pool) + [SimBot()]) if with_bots else []
        for wi, win in enumerate(windows):
            f = extract_features(win)
            f.update(label=0, family=win.family, subject=subj, window=wi, age=win.age, primary=True)
            rows.append(f)
            if not with_bots:
                continue
            lay = Layout.from_session(win)
            for bot in zoo:
                f = extract_features(run_layout(bot, lay, rng))
                f.update(label=1, family=bot.family, subject=subj, window=wi, age=win.age, primary=True)
                rows.append(f)
        # Auxiliary human windows (overlapping the primary ones). They are used only to
        # *fit* human references, where tail quantiles need more rows; never for testing.
        if aux_step:
            for s in sessions:
                for start in range(0, len(s.trials) - k + 1, aux_step):
                    if start % (step or k) == 0:
                        continue
                    f = extract_features(s.slice(start, start + k))
                    f.update(label=0, family=s.family, subject=subj, window=-1, age=s.age, primary=False)
                    rows.append(f)
    return pd.DataFrame(rows)


class AgeConditioned:
    """Wraps a detector: time-scale features are divided by the expected slowing at the
    account's stated age before the tests run. The slowing curve (log-linear after 30)
    is estimated from the training people only, so nothing leaks from the test fold.

    A bot inherits the age of the account it is running on, as it would in practice.
    """

    TIME = ("fitts_b_ucb", "fitts_bw_lcb", "fitts_b", "fitts_a", "fitts_mt_mean", "fitts_beta_d", "fitts_beta_w")
    RATE = ("fitts_tpe", "fitts_tp_p90", "fitts_tp_mean")

    def __init__(self, base, knot: float = 30.0):
        self.base, self.knot = base, knot

    def _k(self, F: pd.DataFrame) -> np.ndarray:
        age = F["age"].to_numpy(dtype=float)
        return np.exp(self.slope * np.maximum(0.0, np.where(np.isfinite(age), age, self.knot) - self.knot))

    def _norm(self, F: pd.DataFrame) -> pd.DataFrame:
        k = self._k(F)
        G = F.copy()
        for c in self.TIME:
            G[c] = G[c] / k
        for c in self.RATE:
            G[c] = G[c] * k
        return G

    def fit(self, F: pd.DataFrame, target_fpr: float = 0.01):
        x = np.maximum(0.0, F["age"].to_numpy(dtype=float) - self.knot)
        y = np.log(F["fitts_mt_mean"].to_numpy(dtype=float))
        ok = np.isfinite(x) & np.isfinite(y)
        xc = x[ok] - x[ok].mean()
        self.slope = float(np.dot(xc, y[ok]) / np.dot(xc, xc))
        self.base.fit(self._norm(F), target_fpr=target_fpr)
        self.threshold = self.base.threshold
        return self

    def score(self, F: pd.DataFrame) -> np.ndarray:
        return self.base.score(self._norm(F))


def feature_cols(df: pd.DataFrame, groups: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c not in META and c.startswith(groups)]


def _folds(df: pd.DataFrame, n_splits: int):
    subjects = df["subject"].to_numpy()
    n = min(n_splits, len(np.unique(subjects)))
    yield from GroupKFold(n_splits=n).split(df, groups=subjects)


def _summarise(scores: pd.DataFrame) -> dict:
    """``scores`` has columns label, family, score, alarm, fold."""
    hum = scores[scores.label == 0]
    out = {"n_human": int(len(hum)), "fpr": float(hum.alarm.mean()),
           "fpr_by_fold": [float(v) for v in hum.groupby("fold").alarm.mean()], "families": {}}
    bots = scores[scores.label == 1]
    for fam, g in bots.groupby("family"):
        y = np.r_[np.zeros(len(hum)), np.ones(len(g))]
        s = np.r_[hum.score.to_numpy(), g.score.to_numpy()]
        # score threshold that gives exactly 1 % FPR on these held-out people
        thr = np.quantile(hum.score, 0.99)
        out["families"][fam] = {
            "n": int(len(g)), "auc": float(roc_auc_score(y, s)),
            "tpr": float(g.alarm.mean()),                       # at the trained threshold
            "tpr_at_1fpr": float((g.score > thr).mean()),       # at an oracle 1 % test FPR
        }
    if len(bots):
        out["tpr_all"] = float(bots.alarm.mean())
        out["auc_all"] = float(roc_auc_score(scores.label, scores.score))
    return out


def cv_oneclass(df: pd.DataFrame, make_detector, n_splits: int = 5, target_fpr: float = 0.01) -> dict:
    parts = []
    for fold, (tr, te) in enumerate(_folds(df, n_splits)):
        train, test = df.iloc[tr], df.iloc[te]
        test = test[test.primary]
        det = make_detector().fit(train[train.label == 0], target_fpr=target_fpr)
        s = det.score(test)
        parts.append(pd.DataFrame({"label": test.label.to_numpy(), "family": test.family.to_numpy(),
                                   "score": s, "alarm": s > det.threshold, "fold": fold}))
    return _summarise(pd.concat(parts, ignore_index=True))


def cv_supervised(df: pd.DataFrame, groups: tuple[str, ...], n_splits: int = 5, target_fpr: float = 0.01,
                  hold_out_family: str | None = None, train_families: list[str] | None = None) -> dict:
    """Gradient boosting on humans vs bots.

    ``hold_out_family`` removes one bot family from training, so its test
    performance measures generalisation to an attack never seen before.
    """
    cols = feature_cols(df, groups)
    df = df[df.primary]
    parts = []
    for fold, (tr, te) in enumerate(_folds(df, n_splits)):
        train, test = df.iloc[tr], df.iloc[te]
        if hold_out_family:
            train = train[train.family != hold_out_family]
            test = test[(test.label == 0) | (test.family == hold_out_family)]
        if train_families is not None:
            train = train[(train.label == 0) | train.family.isin(train_families)]
        # inner split by subject to set the threshold on people the model has not fitted
        subj = train.subject.unique()
        rng = np.random.default_rng(fold)
        cal_subj = set(rng.choice(subj, max(1, len(subj) // 4), replace=False))
        fit_part, cal_part = train[~train.subject.isin(cal_subj)], train[train.subject.isin(cal_subj)]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=fold)
        w = np.where(fit_part.label == 0, (fit_part.label == 1).sum() / max((fit_part.label == 0).sum(), 1), 1.0)
        clf.fit(fit_part[cols], fit_part.label, sample_weight=w)
        cal = clf.predict_proba(cal_part[cal_part.label == 0][cols])[:, 1]
        thr = float(np.quantile(cal, 1.0 - target_fpr))
        s = clf.predict_proba(test[cols])[:, 1]
        parts.append(pd.DataFrame({"label": test.label.to_numpy(), "family": test.family.to_numpy(),
                                   "score": s, "alarm": s > thr, "fold": fold}))
    return _summarise(pd.concat(parts, ignore_index=True))
