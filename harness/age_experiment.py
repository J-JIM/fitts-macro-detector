"""E4: the age model, fitted and validated on real people.

Model (person i, condition c with index of difficulty ID_c):

    MT_ic = m_device(i) * m_impairment(i) * ( a(age_i) + b(age_i) * ID_c )
    a(age) = a0 * exp(s_a * max(0, age - knot)),   b(age) = b0 * exp(s_b * max(0, age - knot))

fitted on log movement time. Variants differ in which of ``s_a``, ``s_b`` are
free; the variant is picked by person-wise cross-validation, and the same
cross-validation gives the errors reported in the harness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from fittsguard.age_model import AgeCurve, AgeMotorModel, Baseline, fit_ratio_curve
from fittsguard.schema import Session

ROOT = Path(__file__).resolve().parents[1]
KNOT = 30.0
BANDS = ((18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 90))
VARIANTS = {            # (s_a free, s_b free, shared)
    "age_blind": (False, False, False),
    "intercept_only": (True, False, False),
    "slope_only": (False, True, False),
    "both_separate": (True, True, False),
    "both_shared": (True, True, True),
}


def person_table(humans: list[Session]) -> pd.DataFrame:
    """Median movement time per person and task condition (robust to pauses)."""
    rows = []
    for s in humans:
        if s.age is None:
            continue
        d = np.array([t.distance for t in s.trials])
        w = np.array([t.target_w for t in s.trials])
        mt = np.array([t.movement_time for t in s.trials])
        cond = np.round(np.log2(np.round(d / 125.0) * 125.0 / w + 1.0), 2)     # nominal A in {250, 500}
        dev = s.meta.get("device", "")
        group = "mouse" if dev in ("optical mouse", "mechanical mouse") else ("touchpad" if dev == "touchpad" else "other")
        hold = np.array([t.t_up - t.t_down for t in s.trials if t.t_up is not None])
        for c in np.unique(cond):
            m = cond == c
            if m.sum() >= 3:
                rows.append({"subject": s.subject, "age": s.age, "device": group,
                             "imp": bool(s.meta.get("motor_impairment")),
                             "id": float(np.mean(np.log2(d[m] / w[m] + 1.0))), "mt": float(np.median(mt[m])),
                             "hold": float(np.median(hold)) if len(hold) else np.nan})
    return pd.DataFrame(rows)


def _predict(th: np.ndarray, tab: pd.DataFrame, variant: str) -> np.ndarray:
    fa, fb, shared = VARIANTS[variant]
    a0, b0, sa, sb, m_tp, m_ot, m_imp = th
    if shared:
        sb = sa
    over = np.maximum(0.0, tab["age"].to_numpy() - KNOT)
    a = a0 * np.exp((sa if fa else 0.0) * over)
    b = b0 * np.exp((sb if fb else 0.0) * over)
    dev = tab["device"].to_numpy()
    mult = np.where(dev == "touchpad", np.exp(m_tp), np.where(dev == "other", np.exp(m_ot), 1.0))
    mult = mult * np.where(tab["imp"].to_numpy(), np.exp(m_imp), 1.0)
    return mult * (a + b * tab["id"].to_numpy())


def _fit(tab: pd.DataFrame, variant: str) -> np.ndarray:
    y = np.log(tab["mt"].to_numpy())
    th0 = np.array([0.3, 0.15, 0.005, 0.0, 0.1, 0.1, 0.1])
    lo = np.array([0.01, 0.01, -0.05, -0.05, -1, -1, -1])
    hi = np.array([2.0, 1.0, 0.05, 0.05, 2, 2, 2])
    return least_squares(lambda th: np.log(_predict(th, tab, variant)) - y, th0, bounds=(lo, hi), loss="soft_l1",
                         f_scale=0.3).x


def _cv(tab: pd.DataFrame, variant: str, seed: int = 0, n_splits: int = 5) -> pd.DataFrame:
    subj = tab["subject"].unique()
    rng = np.random.default_rng(seed)
    fold = dict(zip(subj, rng.permutation(len(subj)) % n_splits))
    f = tab["subject"].map(fold).to_numpy()
    pred = np.zeros(len(tab))
    for k in range(n_splits):
        th = _fit(tab[f != k], variant)
        pred[f == k] = _predict(th, tab[f == k], variant)
    return tab.assign(pred=pred)


def _scores(cvtab: pd.DataFrame) -> dict:
    per = cvtab.groupby("subject").agg(age=("age", "first"), mt=("mt", "mean"), pred=("pred", "mean"))
    r = np.log(per["mt"]) - np.log(per["pred"])
    out = {"person_rmse_log": float(np.sqrt(np.mean(r ** 2))),
           "person_median_abs_pct": float(np.median(np.abs(per["pred"] / per["mt"] - 1.0)) * 100.0), "bands": {}}
    errs = []
    for lo, hi in BANDS:
        g = per[(per["age"] >= lo) & (per["age"] < hi)]
        # geometric means: the model is fitted on log time, and they are robust to a few very slow people
        obs, prd = float(np.exp(np.log(g["mt"]).mean())), float(np.exp(np.log(g["pred"]).mean()))
        out["bands"][f"{lo}-{hi}"] = {"n": int(len(g)), "mt_obs_ms": obs * 1000, "mt_pred_ms": prd * 1000,
                                      "err_pct": (prd / obs - 1.0) * 100.0}
        errs.append(abs(prd / obs - 1.0) * 100.0)
    out["band_mean_abs_pct"] = float(np.mean(errs))
    out["band_max_abs_pct"] = float(np.max(errs))
    return out


def run(humans: list[Session], out_dir: Path | None = None, save_model: bool = True) -> dict:
    tab = person_table(humans)
    res: dict = {"n_people": int(tab["subject"].nunique()), "knot": KNOT, "variants": {}}
    for v in VARIANTS:
        res["variants"][v] = _scores(_cv(tab, v))
    # Pick by cross-validated person-level error. If proportional slowing ("both_shared")
    # is within 0.5 % of the best, prefer it: when the data cannot tell intercept
    # from slope, scaling the whole line is the assumption that extrapolates most safely.
    rm = {v: res["variants"][v]["person_rmse_log"] for v in VARIANTS if v != "age_blind"}
    best = min(rm, key=rm.get)
    res["identifiable"] = bool(rm["both_shared"] > 1.005 * rm[best])
    if not res["identifiable"]:
        best = "both_shared"
    res["chosen"] = best
    res["gain_over_age_blind_pct"] = float(
        (1.0 - res["variants"][best]["person_rmse_log"] / res["variants"]["age_blind"]["person_rmse_log"]) * 100.0)
    res["chosen_scores"] = res["variants"][best]
    res["band_error_ratio"] = float(res["variants"][best]["band_mean_abs_pct"] /
                                    res["variants"]["age_blind"]["band_mean_abs_pct"])

    th = _fit(tab, best)
    fa, fb, shared = VARIANTS[best]
    a0, b0, sa, sb = float(th[0]), float(th[1]), float(th[2]) if fa else 0.0, float(th[3]) if fb else 0.0
    if shared:
        sb = sa
    # bootstrap over people: uncertainty of the ageing slopes
    rng = np.random.default_rng(0)
    subj = tab["subject"].unique()
    groups = {s: g for s, g in tab.groupby("subject")}
    boots = []
    for _ in range(100):
        pick = rng.choice(subj, len(subj), replace=True)
        bt = pd.concat([groups[s] for s in pick], ignore_index=True)
        tb = _fit(bt, best)
        boots.append([tb[2] if fa else 0.0, (tb[2] if shared else tb[3]) if fb else 0.0, tb[0], tb[1]])
    boots = np.array(boots)
    se_sa, se_sb = float(boots[:, 0].std()), float(boots[:, 1].std())

    # residual spread between people (log movement time), typical users on a mouse
    full = tab.assign(pred=_predict(th, tab, best))
    per = full.groupby("subject").agg(mt=("mt", "mean"), pred=("pred", "mean"), age=("age", "first"),
                                      hold=("hold", "first"), device=("device", "first"), imp=("imp", "first"))
    cv_between = float(np.std(np.log(per["mt"]) - np.log(per["pred"])))

    # button hold time: log-linear ageing with the same covariates
    h = per.dropna(subset=["hold"])
    h = h[h["hold"] > 0.01]
    X = np.column_stack([np.ones(len(h)), np.maximum(0, h["age"] - KNOT), h["device"] == "touchpad",
                         h["device"] == "other", h["imp"]]).astype(float)
    coef, *_ = np.linalg.lstsq(X, np.log(h["hold"].to_numpy()), rcond=None)
    hold_resid = np.log(h["hold"].to_numpy()) - X @ coef
    Xcov = np.linalg.inv(X.T @ X) * hold_resid.var()

    res["fit"] = {
        "a_ref_ms": a0 * 1000, "b_ref_ms_per_bit": b0 * 1000,
        "ageing_a_pct_per_decade": (np.exp(10 * sa) - 1) * 100, "ageing_b_pct_per_decade": (np.exp(10 * sb) - 1) * 100,
        "ageing_a_se": se_sa, "ageing_b_se": se_sb,
        "touchpad_mult": float(np.exp(th[4])), "other_device_mult": float(np.exp(th[5])),
        "motor_impairment_mult": float(np.exp(th[6])),
        "between_person_sd_log": cv_between,
        "hold_ref_ms": float(np.exp(coef[0]) * 1000), "hold_ageing_pct_per_decade": float((np.exp(10 * coef[1]) - 1) * 100),
        "hold_ageing_se": float(np.sqrt(Xcov[1, 1])), "hold_sd_log": float(hold_resid.std()),
    }
    # the table people actually ask for: expected values by age band
    src = "Findlater & Zhang 2020, 414 people"
    curves = {      # se_log: slope uncertainty evaluated 40 years past the knot (age 70)
        "a": AgeCurve(ageing=sa, knot=KNOT, age_min=18.0, age_max=83.0, se_log=se_sa * 40.0, source=src),
        "b": AgeCurve(ageing=sb, knot=KNOT, age_min=18.0, age_max=83.0, se_log=se_sb * 40.0, source=src),
        "hold": AgeCurve(ageing=float(coef[1]), knot=KNOT, age_min=18.0, age_max=83.0,
                         se_log=float(np.sqrt(Xcov[1, 1])) * 40.0, source=src),
    }
    base = Baseline(a=a0, b=b0, hold=float(np.exp(coef[0])), cv_between=cv_between,
                    cv_hold=float(hold_resid.std()), source="Findlater & Zhang 2020 (mouse, no motor impairment, age <= 30)")

    lit_path = ROOT / "data" / "literature" / "age_groups.csv"
    if lit_path.exists():
        lit = pd.read_csv(lit_path, comment="#")
        res["literature"] = {}
        for metric in ("mt", "tap"):
            try:
                c, rep = fit_ratio_curve(lit, metric, knot=KNOT)
            except ValueError as e:
                res["literature"][metric] = {"error": str(e)}
                continue
            res["literature"][metric] = {
                **rep, "dev_c": c.dev_c, "dev_tau": c.dev_tau, "ageing_per_year": c.ageing,
                "age_min": c.age_min, "age_max": c.age_max,
                "factor_at": {str(a): float(c.factor(a)) for a in (5, 8, 12, 25, 50, 70) if c.age_min <= a <= c.age_max},
            }
            if metric == "mt" and c.age_min < 18:
                # Children: published child/adult ratios scale the whole Fitts line. The
                # ageing side of the literature fit is reported but NOT used: the 414-person
                # data are larger, recent, and measured on people's own devices.
                n_child = int((lit[(lit["metric"] == "mt")]["age"] < 16).sum())
                res["literature"][metric]["n_child_groups"] = n_child
                for key in ("a", "b"):
                    curves[key].dev_c, curves[key].dev_tau, curves[key].age_min = c.dev_c, c.dev_tau, c.age_min
                    curves[key].child_se_log = rep["loso_rmse_log"]
                    curves[key].source += f"; ages below 18 from {n_child} child groups in published studies (weak)"
            if metric == "tap":
                c.dev_c = 0.0
                curves["tap"] = c
                ref = lit[(lit["metric"] == "tap_abs")]
                if len(ref):
                    base.cps = float(ref["value"].iloc[0])
    model = AgeMotorModel(curves, base)
    res["table"] = []
    for age in (5, 8, 12, 16, 20, 30, 40, 50, 60, 70, 80):
        if age < curves["a"].age_min:
            continue
        m = model.movement_time(age, 400, 32)
        row = {"age": age, "a_ms": m["a"] * 1000, "b_ms_per_bit": m["b"] * 1000, "mt_ms_D400_W32": m["mt"] * 1000,
               "mt_lo_ms": m["mt_lo"] * 1000, "mt_hi_ms": m["mt_hi"] * 1000, "throughput": m["throughput"],
               "hold_ms": model.hold_time(age)["hold"] * 1000}
        if "tap" in curves and curves["tap"].age_min <= age <= curves["tap"].age_max:
            row["max_cps"] = model.click_rate(age)["cps"]
        res["table"].append(row)
    if save_model:
        dest = ROOT / "src" / "fittsguard" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        model.save(dest / "age_model.json")
    return res


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import load_input_accessibility
    print(json.dumps(run(load_input_accessibility(), save_model=False), indent=1))
