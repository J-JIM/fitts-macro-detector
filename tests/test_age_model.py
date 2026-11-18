import numpy as np
import pandas as pd
import pytest

from fittsguard.age_model import AgeCurve, AgeMotorModel, Baseline, fit_ratio_curve, lifespan


def _synthetic_table(rng, dev_c=1.6, dev_tau=2.5, ageing=0.008, noise=0.02):
    """Studies with wildly different absolute levels but one shared age shape."""
    rows = []
    for s in range(12):
        level = np.exp(rng.normal(0.0, 0.5))                # device / task / definition differences
        for age in rng.choice([4.5, 6, 9, 13, 22, 35, 50, 65, 78], size=3, replace=False):
            g = float(lifespan(age, dev_c, dev_tau, ageing, 30.0))
            rows.append({"study": f"s{s}", "metric": "mt", "age": float(age), "n": 20,
                         "value": level * np.exp(g + rng.normal(0, noise))})
    return pd.DataFrame(rows)


def test_ratio_curve_recovers_shape_despite_study_offsets():
    rng = np.random.default_rng(0)
    curve, rep = fit_ratio_curve(_synthetic_table(rng), "mt")
    truth = AgeCurve(1.6, 2.5, 0.008, age_min=4, age_max=80)
    for age in (5, 8, 13, 50, 75):
        assert float(curve.factor(age)) == pytest.approx(float(truth.factor(age)), rel=0.08)
    assert float(curve.factor(25)) == pytest.approx(1.0)
    assert rep["loso_rmse_log"] < 0.08


def test_predictions_are_ordered_clamped_and_serialisable():
    model = AgeMotorModel(
        {"a": AgeCurve(1.6, 2.5, 0.010, age_min=4, age_max=80),
         "b": AgeCurve(1.6, 2.5, 0.000, age_min=4, age_max=80),
         "hold": AgeCurve(ageing=0.006), "tap": AgeCurve(ageing=-0.004, age_min=20, age_max=80)},
        Baseline(a=0.28, b=0.15, cps=5.0))
    young, old, child = (model.movement_time(a, 400, 32) for a in (25, 75, 6))
    assert old["mt"] > young["mt"] and child["mt"] > young["mt"]
    assert young["mt_lo"] < young["mt"] < young["mt_hi"]
    # ageing here moves the intercept only: the slope stays put
    assert old["b"] == pytest.approx(young["b"], rel=1e-3) and old["a"] > young["a"]
    assert model.click_rate(75)["cps"] < model.click_rate(25)["cps"]
    assert model.hold_time(75)["hold"] > model.hold_time(25)["hold"]
    # no extrapolation beyond the ages the data covers
    assert model.movement_time(120, 400, 32)["mt"] == pytest.approx(model.movement_time(80, 400, 32)["mt"])
    assert model.movement_time(1, 400, 32)["mt"] == pytest.approx(model.movement_time(4, 400, 32)["mt"])
    again = AgeMotorModel.from_dict(model.to_dict())
    assert again.movement_time(60, 300, 20)["mt"] == pytest.approx(model.movement_time(60, 300, 20)["mt"])
