import numpy as np
import pandas as pd
import pytest

from fittsguard.bots import ConstSpeed, ConstTime, Teleport, default_zoo
from fittsguard.detector import LayeredDetector, MacroDetector
from fittsguard.features import extract_features, resample
from fittsguard.fitts import fit_fitts, index_of_difficulty, throughput
from fittsguard.human_sim import HumanSim, MotorProfile, perturb
from fittsguard.schema import Session, Trial
from fittsguard.simulate import random_layout, run_layout


def test_fit_recovers_known_line():
    rng = np.random.default_rng(1)
    idb = rng.uniform(1, 6, 200)
    mt = 0.2 + 0.15 * idb + rng.normal(0, 0.01, 200)
    fit = fit_fitts(idb, mt)
    assert fit.a == pytest.approx(0.2, abs=0.01)
    assert fit.b == pytest.approx(0.15, abs=0.005)
    assert fit.r2 > 0.98


def test_fit_degenerate_is_nan_not_error():
    fit = fit_fitts(np.full(10, 3.0), np.linspace(0.3, 0.5, 10))
    assert np.isnan(fit.b)


def test_id_and_throughput():
    assert index_of_difficulty(300, 100) == pytest.approx(2.0)
    assert throughput(np.array([2.0, 4.0]), np.array([0.5, 1.0])) == pytest.approx(4.0)


def test_resample_holds_position_across_event_gaps():
    # still for 0.5 s, then a quick move: the pause must not turn into a slow drift
    t = np.array([0.0, 0.008, 0.508, 0.516, 0.524])
    x = np.array([0.0, 0.0, 0.0, 50.0, 100.0])
    tr = Trial(t=t, x=x, y=np.zeros(5), target_x=100, target_y=0, target_w=20, t_down=0.53, click_x=100, click_y=0)
    g, xs, _ = resample(tr)
    assert xs[np.searchsorted(g, 0.25)] == pytest.approx(0.0, abs=1.0)


def test_session_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    s = run_layout(HumanSim(), random_layout(rng, 12), rng, subject="s1", age=30)
    p = tmp_path / "s.json"
    s.save(p)
    s2 = Session.load(p)
    assert len(s2) == 12 and s2.age == 30
    assert s2.trials[3].movement_time == pytest.approx(s.trials[3].movement_time, abs=1e-4)


def test_simulated_person_obeys_fitts():
    rng = np.random.default_rng(3)
    s = run_layout(HumanSim(MotorProfile(a=0.25, b=0.16)), random_layout(rng, 300), rng)
    f = extract_features(s)
    assert f["fitts_b"] == pytest.approx(0.16, abs=0.03)
    assert f["fitts_beta_w"] < -0.08 < 0.08 < f["fitts_beta_d"]
    assert 0.02 <= f["fitts_err_rate"] <= 0.15


def test_machine_timing_ignores_width():
    rng = np.random.default_rng(4)
    for bot in (ConstSpeed(), ConstTime()):
        f = extract_features(run_layout(bot, random_layout(rng, 120), rng))
        assert abs(f["fitts_beta_w"]) < 0.01          # no sensitivity to target size at all
    f = extract_features(run_layout(Teleport(), random_layout(rng, 40), rng))
    assert f["fitts_tp_mean"] > 20                    # far beyond any human throughput


def test_fitts_layer_flags_non_adaptive_bots_and_passes_people():
    """Simulated people only (unit test). Real-people numbers come from the harness."""
    rng = np.random.default_rng(5)
    base = MotorProfile()
    humans = [run_layout(HumanSim(perturb(base, rng)), random_layout(rng, 32), rng) for _ in range(300)]
    F = pd.DataFrame([extract_features(s) for s in humans])
    det = LayeredDetector(("fitts",)).fit(F.iloc[:200], target_fpr=0.02)
    fpr = float(np.mean(det.score(F.iloc[200:]) > det.threshold))
    assert fpr <= 0.10
    pool = [t for s in humans[:20] for t in s.trials]
    for bot in default_zoo(pool):
        if bot.family in ("fitts_aware",):
            continue                                   # adaptive: not expected to fail the Fitts layer
        sess = [run_layout(bot, random_layout(rng, 32), rng) for _ in range(20)]
        hit = np.mean(det.score(pd.DataFrame([extract_features(s) for s in sess])) > det.threshold)
        assert hit >= 0.9, bot.family


def test_people_are_never_flagged_for_being_slow_or_sloppy():
    """One-sidedness: a pause, a wild miss or a slow session must not raise the score."""
    rng = np.random.default_rng(6)
    humans = [run_layout(HumanSim(perturb(MotorProfile(), rng)), random_layout(rng, 32), rng) for _ in range(200)]
    det = LayeredDetector(("fitts", "kin")).fit(pd.DataFrame([extract_features(s) for s in humans]))
    slow = MotorProfile(a=0.9, b=0.5, sigma_mt=0.45, kappa=2.5, settle=0.4, latency=0.5)   # far outside the fitted group
    sess = [run_layout(HumanSim(slow), random_layout(rng, 32), rng) for _ in range(30)]
    for s in sess[:10]:                                 # and some very long pauses
        s.trials[5].t_down += 8.0
    score = det.score(pd.DataFrame([extract_features(s) for s in sess]))
    assert np.mean(score > det.threshold) <= 0.1


def test_verdict_explains_itself():
    rng = np.random.default_rng(7)
    humans = [run_layout(HumanSim(perturb(MotorProfile(), rng)), random_layout(rng, 25), rng) for _ in range(200)]
    det = LayeredDetector(("fitts", "kin")).fit(pd.DataFrame([extract_features(s) for s in humans]))
    judge = MacroDetector(det, window=25)
    v = judge.judge(run_layout(Teleport(), random_layout(rng, 50), rng))
    assert v.is_bot and "superhuman_throughput" in v.fired
    again = MacroDetector.from_dict(judge.to_dict())
    assert again.judge(run_layout(ConstTime(), random_layout(rng, 25), rng)).is_bot
    with pytest.raises(ValueError):
        judge.judge(run_layout(HumanSim(), random_layout(rng, 10), rng))
