"""The files shipped inside the package load and behave."""
import numpy as np
import pytest

from fittsguard import AgeMotorModel, load_detector
from fittsguard.bots import ConstSpeed, EaseTween, Teleport
from fittsguard.cli import main
from fittsguard.simulate import random_layout, run_layout


def test_packaged_detector_flags_machine_timing():
    det = load_detector()
    rng = np.random.default_rng(0)
    for bot in (Teleport(), ConstSpeed(), EaseTween()):
        v = det.judge(run_layout(bot, random_layout(rng, 50), rng))
        assert v.is_bot, bot.family
        assert v.fired and all(t["meaning"] for t in v.tests)


def test_packaged_age_model_matches_the_data_it_was_fitted_on():
    m = AgeMotorModel.load()
    young, old = m.movement_time(25, 400, 32), m.movement_time(70, 400, 32)
    assert 0.6 < young["mt"] < 1.0                      # about 0.8 s for a 3.75-bit target
    assert 1.2 < old["mt"] / young["mt"] < 1.5          # about +7 % per decade after 30
    assert m.hold_time(70)["hold"] > m.hold_time(25)["hold"]
    assert 4.0 < m.click_rate(25)["cps"] < 6.0
    assert m.movement_time(6, 400, 32)["mt"] > young["mt"]


def test_cli_roundtrip(tmp_path, capsys):
    out = tmp_path / "s.json"
    assert main(["simulate", "--agent", "const_time", "-n", "25", "-o", str(out)]) == 0
    assert main(["score", str(out)]) == 1               # flagged -> exit code 1
    assert "FIRED" in capsys.readouterr().out
    assert main(["predict", "--age", "40"]) == 0
    assert "movement time" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["predict"])                               # age is required
