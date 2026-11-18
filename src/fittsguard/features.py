"""Session-level features.

Three groups, selected by name prefix so the harness can ablate them:

``fitts_``  what Fitts' law says about the session as a whole. Needs only
            click times, click positions and target geometry: no trajectory.
``kin_``    kinematics of the individual movements (speed profile,
            submovements, path shape, settle time). Needs the cursor trajectory.
``art_``    recording artefacts (event timing regularity, button hold time).
            Powerful but device/pipeline dependent, so they are kept apart and
            never enter the headline numbers.

All trajectories are passed through :func:`resample` first, so that sampling
rate and pixel quantisation cannot leak the label.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, spearmanr, theilslopes

from .fitts import fit_fitts
from .schema import Session, Trial

GRID_HZ = 100.0          # common analysis rate
SMOOTH_S = 0.020         # gaussian sigma for position smoothing, seconds
SPEED_FRAC = 0.05        # onset / offset threshold as a fraction of peak speed


# --------------------------------------------------------------------------
# trajectory pre-processing
# --------------------------------------------------------------------------
def resample(tr: Trial, hz: float = GRID_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trajectory on a uniform grid from trial start to button-down.

    Pointer events only fire while the cursor moves, so a long gap between two
    events means "stayed put, then moved" and not "drifted slowly". Gaps longer
    than 2.5x the median interval are therefore held at the earlier position.
    """
    t, x, y = tr.t, tr.x, tr.y
    if t[-1] < tr.t_down:
        t = np.append(t, tr.t_down)
        x = np.append(x, tr.click_x)
        y = np.append(y, tr.click_y)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x, y = t[keep], x[keep], y[keep]
    if len(t) >= 3:
        dt = np.diff(t)
        med = float(np.median(dt))
        gaps = np.where(dt > 2.5 * med)[0]
        if len(gaps):
            t = np.insert(t, gaps + 1, t[gaps + 1] - med)
            x = np.insert(x, gaps + 1, x[gaps])
            y = np.insert(y, gaps + 1, y[gaps])
    n = max(int(round((tr.t_down - t[0]) * hz)) + 1, 2)
    grid = t[0] + np.arange(n) / hz
    return grid, np.interp(grid, t, x), np.interp(grid, t, y)


_KIN_KEYS = (
    "latency", "settle", "ttp", "nsub", "straight", "maxdev", "vpeak",
    "ldlj", "mjcorr", "move_dur",
)


def trial_kinematics(tr: Trial, hz: float = GRID_HZ) -> dict[str, float]:
    """Per-movement kinematic descriptors. NaN when the trial has no usable path."""
    out = {k: np.nan for k in _KIN_KEYS}
    if len(tr.t) < 3 or tr.movement_time <= 2.0 / hz:
        return out
    g, x, y = resample(tr, hz)
    sig = SMOOTH_S * hz
    xs, ys = gaussian_filter1d(x, sig, mode="nearest"), gaussian_filter1d(y, sig, mode="nearest")
    vx, vy = np.gradient(xs) * hz, np.gradient(ys) * hz
    speed = np.hypot(vx, vy)
    vpeak = float(speed.max())
    if vpeak <= 1e-9:
        return out
    moving = np.where(speed > SPEED_FRAC * vpeak)[0]
    i0, i1 = int(moving[0]), int(moving[-1])
    dur = (i1 - i0) / hz
    out["vpeak"] = vpeak
    out["latency"] = i0 / hz
    out["settle"] = (len(g) - 1 - i1) / hz
    out["move_dur"] = dur
    if i1 - i0 < 3:
        return out
    seg = speed[i0:i1 + 1]
    out["ttp"] = float(np.argmax(seg)) / (len(seg) - 1)
    peaks, _ = find_peaks(seg, prominence=0.10 * vpeak)
    out["nsub"] = float(max(len(peaks), 1))
    path = float(np.sum(np.hypot(np.diff(xs[i0:i1 + 1]), np.diff(ys[i0:i1 + 1]))))
    chord = float(np.hypot(xs[i1] - xs[i0], ys[i1] - ys[i0]))
    out["straight"] = chord / path if path > 0 else np.nan
    if chord > 0:
        ux, uy = (xs[i1] - xs[i0]) / chord, (ys[i1] - ys[i0]) / chord
        dev = np.abs((xs[i0:i1 + 1] - xs[i0]) * uy - (ys[i0:i1 + 1] - ys[i0]) * ux)
        out["maxdev"] = float(dev.max()) / chord
    # log dimensionless jerk (Balasubramanian et al. 2015): larger = smoother
    ax, ay = np.gradient(vx) * hz, np.gradient(vy) * hz
    jx, jy = np.gradient(ax) * hz, np.gradient(ay) * hz
    j2 = float(np.sum(jx[i0:i1 + 1] ** 2 + jy[i0:i1 + 1] ** 2)) / hz
    if j2 > 0 and dur > 0:
        out["ldlj"] = -float(np.log(dur ** 3 / vpeak ** 2 * j2))
    # similarity of the speed profile to the minimum-jerk bell tau^2 (1 - tau)^2
    tau = np.linspace(0.0, 1.0, 40)
    prof = np.interp(tau, np.linspace(0.0, 1.0, len(seg)), seg)
    bell = tau ** 2 * (1.0 - tau) ** 2
    if prof.std() > 0:
        out["mjcorr"] = float(np.corrcoef(prof, bell)[0, 1])
    return out


# --------------------------------------------------------------------------
# small statistics helpers
# --------------------------------------------------------------------------
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _med(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def _iqr(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.subtract(*np.percentile(a, [75, 25]))) if len(a) >= 4 else np.nan


def _cv(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if len(a) < 3 or abs(np.mean(a)) < 1e-12:
        return np.nan
    return float(np.std(a, ddof=1) / abs(np.mean(a)))


def _slope(xv: np.ndarray, yv: np.ndarray) -> float:
    ok = np.isfinite(xv) & np.isfinite(yv)
    if ok.sum() < 4 or np.ptp(xv[ok]) < 1e-9:
        return np.nan
    xc = xv[ok] - xv[ok].mean()
    return float(np.dot(xc, yv[ok]) / np.dot(xc, xc))


def _trim_mask(idb: np.ndarray, mt: np.ndarray, k: float = 3.5) -> np.ndarray:
    """True for trials to keep: not far *above* a robust Fitts line (pauses).

    One-sided on purpose. Suspiciously fast trials are never discarded: they are
    exactly what the detector is looking for.
    """
    n = len(mt)
    if n < 8:
        return np.ones(n, dtype=bool)
    if np.ptp(idb) > 1e-9:
        slope, icpt, _, _ = theilslopes(mt, idb)
        r = mt - (icpt + slope * idb)
    else:
        r = mt - np.median(mt)
    r = r - np.median(r)
    madn = 1.4826 * float(np.median(np.abs(r)))
    if madn < 1e-9:
        return np.ones(n, dtype=bool)
    keep = r <= k * madn
    return keep if keep.sum() >= max(6, n // 2) else np.ones(n, dtype=bool)


# --------------------------------------------------------------------------
# session features
# --------------------------------------------------------------------------
def extract_features(session: Session, hz: float = GRID_HZ) -> dict[str, float]:
    trials = session.trials
    n = len(trials)
    f: dict[str, float] = {}
    if n == 0:
        return f

    d = np.array([t.distance for t in trials])
    w = np.array([t.target_w for t in trials])
    idb = np.log2(d / w + 1.0)
    mt = np.array([t.movement_time for t in trials])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ---------------- Fitts layer ----------------
        # People pause: a trial where someone looked away is a real human trial but
        # tells nothing about their motor law. Trials whose time is far above the
        # session's own Fitts line are set aside for the regressions (and counted).
        keep = _trim_mask(idb, mt)
        f["fitts_trim_frac"] = float(1.0 - keep.mean())
        idk, mtk = idb[keep], mt[keep]
        nk = int(keep.sum())
        fit = fit_fitts(idk, mtk)
        f["fitts_a"] = fit.a
        f["fitts_b"] = fit.b
        f["fitts_r2"] = fit.r2
        mean_mt = float(np.mean(mtk))
        f["fitts_mt_mean"] = mean_mt
        f["fitts_mt_min"] = float(np.min(mt))
        resid = fit.resid if np.isfinite(fit.b) else mtk - mean_mt
        sd_res = float(np.std(resid, ddof=2)) if nk > 3 else np.nan
        f["fitts_resid_cv"] = sd_res / mean_mt if mean_mt > 0 else np.nan
        f["fitts_resid_skew"] = float(skew(resid)) if nk > 3 and np.std(resid) > 1e-12 else np.nan
        # multiplicative noise: |residual| grows with the predicted time in people
        f["fitts_resid_het"] = _corr(np.abs(resid), idk)
        f["fitts_resid_ac1"] = _corr(resid[:-1], resid[1:])
        # How sure are we that time depends on difficulty at all? A sloppy person
        # has a noisy slope estimate around a real slope; a program with random
        # delays has a slope that is *confidently* zero. The upper confidence bound
        # separates the two where the point estimate cannot.
        sxx = float(np.sum((idk - idk.mean()) ** 2)) if nk > 3 else 0.0
        if sxx > 1e-9 and np.isfinite(sd_res):
            se_b = sd_res / np.sqrt(sxx)
            f["fitts_b_ucb"] = fit.b + 2.0 * se_b
            f["fitts_b_t"] = fit.b / se_b if se_b > 1e-9 else float(np.sign(fit.b)) * 50.0
        else:
            f["fitts_b_ucb"] = f["fitts_b_t"] = np.nan
        f["fitts_b_t"] = float(np.clip(f["fitts_b_t"], -50.0, 50.0))
        f["fitts_rho"] = float(spearmanr(idb, mt).statistic) if n > 3 and np.ptp(idb) > 1e-9 and np.ptp(mt) > 1e-9 else np.nan
        # throughput: bits per second, per trial then summarised
        tp = np.where(mt > 0, idb / np.maximum(mt, 1e-6), np.inf)
        tp = np.minimum(tp, 1e3)
        f["fitts_tp_mean"] = float(np.mean(tp))
        f["fitts_tp_p90"] = float(np.percentile(tp, 90))
        # separate sensitivity to distance and to width. Fitts predicts
        # beta_d > 0 and beta_w = -beta_d; a program that ignores target size
        # has beta_w = 0 whatever its speed profile looks like.
        f["fitts_beta_d"] = f["fitts_beta_w"] = f["fitts_r2_dw"] = np.nan
        f["fitts_bw_lcb"] = f["fitts_bw_t"] = f["fitts_bd_t"] = np.nan
        ld, lw = np.log2(d[keep]), np.log2(w[keep])
        if nk >= 6 and np.ptp(ld) > 0.2 and np.ptp(lw) > 0.2:
            X = np.column_stack([np.ones(nk), ld, lw])
            xtx_inv = np.linalg.pinv(X.T @ X)
            coef = xtx_inv @ X.T @ mtk
            r = mtk - X @ coef
            s2 = float(np.sum(r ** 2)) / max(nk - 3, 1)
            ss_tot = float(np.sum((mtk - mtk.mean()) ** 2))
            se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0.0))
            f["fitts_beta_d"], f["fitts_beta_w"] = float(coef[1]), float(coef[2])
            f["fitts_r2_dw"] = 1.0 - float(np.sum(r ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan
            f["fitts_bw_lcb"] = float(coef[2] - 2.0 * se[2])      # most negative width effect the data allow
            f["fitts_bw_t"] = float(np.clip(coef[2] / se[2], -50, 50)) if se[2] > 1e-9 else float(np.sign(coef[2])) * 50.0
            f["fitts_bd_t"] = float(np.clip(coef[1] / se[1], -50, 50)) if se[1] > 1e-9 else float(np.sign(coef[1])) * 50.0

        # endpoints, in units of target width, in the approach-axis frame
        ux = np.array([(t.target_x - t.start_x) for t in trials]) / np.maximum(d, 1e-9)
        uy = np.array([(t.target_y - t.start_y) for t in trials]) / np.maximum(d, 1e-9)
        ex = np.array([t.click_x - t.target_x for t in trials])
        ey = np.array([t.click_y - t.target_y for t in trials])
        along = (ex * ux + ey * uy) / w
        ortho = (-ex * uy + ey * ux) / w
        # robust scatter (MAD): one wild misclick must not define a person's precision
        madn = lambda v: 1.4826 * float(np.median(np.abs(v - np.median(v))))  # noqa: E731
        f["fitts_end_sd_along"] = madn(along) if n > 3 else np.nan
        f["fitts_end_sd_ortho"] = madn(ortho) if n > 3 else np.nan
        f["fitts_end_bias_along"] = float(np.median(along))
        near = (np.abs(along) < 1.5) & (np.abs(ortho) < 1.5)
        f["fitts_end_kurt"] = float(kurtosis(along[near])) if near.sum() > 5 and np.std(along[near]) > 1e-9 else np.nan
        f["fitts_end_exact"] = float(np.mean(np.hypot(ex, ey) < 0.75))   # dead-centre clicks
        f["fitts_err_rate"] = float(np.mean([not t.hit for t in trials]))
        # effective throughput (ISO 9241-9): accuracy-adjusted
        we = 4.133 * madn(along * w) if n > 3 else np.nan
        if np.isfinite(we) and we > 1e-6:
            ide = np.log2(d / we + 1.0)
            f["fitts_tpe"] = float(np.mean(np.minimum(ide / np.maximum(mt, 1e-6), 1e3)))
        else:
            f["fitts_tpe"] = np.nan
        # speed-accuracy trade-off inside the session: faster-than-predicted
        # movements land further from the centre in people
        f["fitts_sat"] = _corr(resid, -np.abs(along[keep]))

        # ---------------- kinematic layer ----------------
        kin = [trial_kinematics(t, hz) for t in trials]
        K = {k: np.array([r[k] for r in kin]) for k in _KIN_KEYS}
        f["kin_valid_frac"] = float(np.mean(np.isfinite(K["ttp"])))
        for k in ("latency", "settle", "ttp", "nsub", "straight", "maxdev", "ldlj", "mjcorr"):
            f[f"kin_{k}_med"] = _med(K[k])
        for k in ("latency", "settle", "ttp", "straight", "ldlj"):
            f[f"kin_{k}_iqr"] = _iqr(K[k])
        f["kin_latency_cv"] = _cv(K["latency"])
        f["kin_settle_cv"] = _cv(K["settle"])
        # how the movement itself scales with the task
        f["kin_vpeak_d_slope"] = _slope(np.log(d), np.log(np.maximum(K["vpeak"], 1e-9)))
        f["kin_dur_id_slope"] = _slope(idb, K["move_dur"])
        f["kin_dur_id_corr"] = _corr(idb, K["move_dur"])
        f["kin_ttp_id_corr"] = _corr(idb, K["ttp"])
        f["kin_nsub_id_corr"] = _corr(idb, K["nsub"])
        f["kin_settle_w_corr"] = _corr(np.log2(w), K["settle"])

        # ---------------- artefact layer ----------------
        dts = np.concatenate([np.diff(t.t) for t in trials if len(t.t) > 2] or [np.array([])])
        dts = dts[dts > 0]
        f["art_dt_cv"] = _cv(dts) if len(dts) else np.nan
        f["art_samples_med"] = float(np.median([len(t.t) for t in trials]))
        hold = np.array([(t.t_up - t.t_down) if t.t_up is not None else np.nan for t in trials])
        f["art_hold_med"] = _med(hold)
        f["art_hold_cv"] = _cv(hold)
        frac = np.concatenate([np.abs(t.x - np.round(t.x)) for t in trials])
        f["art_subpixel"] = float(np.mean(frac > 1e-6))

    return f


FEATURE_GROUPS = ("fitts_", "kin_", "art_")


def select(names: list[str], groups: tuple[str, ...]) -> list[str]:
    return [n for n in names if n.startswith(groups)]
