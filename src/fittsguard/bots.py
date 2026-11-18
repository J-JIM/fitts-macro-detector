"""The bot zoo: programs that click targets, from crude to adaptive.

These exist so the detector can be measured, not so anything can be evaded.
They are deliberately ordered by how much they know about people:

tier 0  no movement at all (``teleport``, ``teleport_jitter``)
tier 1  movement with machine timing (``const_speed``, ``const_time``, ``ease_tween``)
tier 2  "human-like" paths with timing unrelated to the task
        (``bezier``, ``wind``): the typical open-source mouse humaniser
tier 3  adaptive: timing follows Fitts' law (``fitts_aware``), recorded human
        movements replayed onto new targets (``replay``), and a full
        submovement simulator (``sim_human`` in :mod:`human_sim`)

Tier 3 is where a detector built on Fitts' law is *expected* to weaken; the
harness reports that honestly instead of hiding it.
"""
from __future__ import annotations

import numpy as np

from .schema import Trial
from .simulate import DENSE_HZ, Move, Target


def _dense(dur: float) -> np.ndarray:
    n = max(int(round(dur * DENSE_HZ)), 1)
    return np.arange(n + 1) / DENSE_HZ


def _assemble(wait: float, tm: np.ndarray, xm: np.ndarray, ym: np.ndarray, after: float, hold: float,
              sx: float, sy: float) -> Move:
    """Wait at the start, follow the path, pause ``after`` seconds, press."""
    # the cursor stays put during the wait; a path that starts elsewhere is a jump, not a glide
    t = np.concatenate([[0.0, wait], wait + 1e-4 + tm])
    x = np.concatenate([[sx, sx], xm])
    y = np.concatenate([[sy, sy], ym])
    return Move(t=t, x=x, y=y, t_down=float(t[-1] + after), hold=float(hold))


def _in_disk(rng: np.random.Generator, tg: Target, frac: float) -> tuple[float, float]:
    r = frac * tg.w / 2.0 * np.sqrt(rng.uniform())
    a = rng.uniform(0.0, 2.0 * np.pi)
    return tg.x + r * np.cos(a), tg.y + r * np.sin(a)


class _Bot:
    label = "bot"
    family = "bot"
    event_hz = 125.0
    event_jitter = 0.0

    def new_session(self, rng: np.random.Generator) -> None:
        return None


class Teleport(_Bot):
    """Sets the cursor position and clicks. Fixed delay, dead centre."""

    family = "teleport"

    def move(self, sx, sy, tg, rng):
        return _assemble(0.05, np.array([0.0]), np.array([tg.x]), np.array([tg.y]), 0.0, 0.0, sx, sy)


class TeleportJitter(_Bot):
    """Teleport with a random delay and a random point inside the target."""

    family = "teleport_jitter"

    def move(self, sx, sy, tg, rng):
        ex, ey = _in_disk(rng, tg, 0.8)
        return _assemble(rng.uniform(0.15, 0.6), np.array([0.0]), np.array([ex]), np.array([ey]),
                         rng.uniform(0.01, 0.05), rng.uniform(0.03, 0.10), sx, sy)


class ConstSpeed(_Bot):
    """Straight line at a fixed pixels-per-second speed: time grows linearly with distance."""

    family = "const_speed"

    def new_session(self, rng):
        self.v = rng.uniform(800.0, 2500.0)

    def move(self, sx, sy, tg, rng):
        dist = np.hypot(tg.x - sx, tg.y - sy)
        tm = _dense(max(dist / self.v, 0.004))
        s = tm / tm[-1]
        return _assemble(rng.uniform(0.05, 0.15), tm, sx + (tg.x - sx) * s, sy + (tg.y - sy) * s,
                         0.0, 0.02, sx, sy)


class ConstTime(_Bot):
    """``moveTo(x, y, duration=T)`` with a linear tween and a fixed pause afterwards."""

    family = "const_time"

    def new_session(self, rng):
        self.T = rng.uniform(0.15, 0.5)

    def move(self, sx, sy, tg, rng):
        tm = _dense(self.T)
        s = tm / tm[-1]
        return _assemble(0.1, tm, sx + (tg.x - sx) * s, sy + (tg.y - sy) * s, 0.1, 0.0, sx, sy)


class EaseTween(_Bot):
    """Ease-in-out tween, random duration, gaussian offset inside the target."""

    family = "ease_tween"

    def move(self, sx, sy, tg, rng):
        ex = tg.x + rng.normal(0.0, tg.w / 6.0)
        ey = tg.y + rng.normal(0.0, tg.w / 6.0)
        tm = _dense(rng.uniform(0.2, 0.6))
        u = tm / tm[-1]
        s = np.where(u < 0.5, 2 * u ** 2, 1 - 2 * (1 - u) ** 2)   # easeInOutQuad
        return _assemble(rng.uniform(0.1, 0.4), tm, sx + (ex - sx) * s, sy + (ey - sy) * s,
                         rng.uniform(0.02, 0.08), rng.uniform(0.04, 0.12), sx, sy)


def _bezier(sx, sy, ex, ey, rng, spread=0.3):
    dist = np.hypot(ex - sx, ey - sy) + 1e-9
    ux, uy = (ex - sx) / dist, (ey - sy) / dist
    c = []
    for f in (rng.uniform(0.15, 0.45), rng.uniform(0.55, 0.85)):
        off = rng.normal(0.0, spread / 2.0) * dist
        c.append((sx + f * (ex - sx) - uy * off, sy + f * (ey - sy) + ux * off))

    def at(s):
        b0, b1, b2, b3 = (1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * s ** 2 * (1 - s), s ** 3
        return (b0 * sx + b1 * c[0][0] + b2 * c[1][0] + b3 * ex,
                b0 * sy + b1 * c[0][1] + b2 * c[1][1] + b3 * ey)
    return at


class Bezier(_Bot):
    """Curved path with ease-out timing; duration is random and unrelated to the task."""

    family = "bezier"
    event_jitter = 0.001

    def move(self, sx, sy, tg, rng):
        ex, ey = _in_disk(rng, tg, 0.7)
        tm = _dense(rng.uniform(0.3, 0.9))
        u = tm / tm[-1]
        s = 1 - (1 - u) ** 3                                         # easeOutCubic
        x, y = _bezier(sx, sy, ex, ey, rng)(s)
        return _assemble(rng.uniform(0.1, 0.4), tm, x, y, rng.uniform(0.03, 0.12),
                         rng.uniform(0.05, 0.12), sx, sy)


class Wind(_Bot):
    """Gravity-plus-random-wind physics: the cursor is pulled toward the target
    while a slowly varying random force pushes it around. Time is whatever the
    physics takes, so it scales with distance and ignores target size."""

    family = "wind"
    event_jitter = 0.001

    def new_session(self, rng):
        self.gravity = rng.uniform(7.0, 11.0)
        self.wind = rng.uniform(2.0, 5.0)
        self.vmax = rng.uniform(12.0, 20.0)
        self.tick = rng.uniform(0.006, 0.010)

    def move(self, sx, sy, tg, rng):
        ex, ey = _in_disk(rng, tg, 0.5)
        x, y, vx, vy, wx, wy = sx, sy, 0.0, 0.0, 0.0, 0.0
        xs, ys = [x], [y]
        near = 12.0
        rho = 0.6                                   # wind memory (AR(1) coefficient)
        for _ in range(5000):
            dist = np.hypot(ex - x, ey - y)
            if dist < 1.0:
                break
            far = dist >= near
            # wind: an AR(1) random force that dies out close to the target
            wx = rho * wx + (rng.normal(0.0, self.wind * 0.45) if far else 0.0)
            wy = rho * wy + (rng.normal(0.0, self.wind * 0.45) if far else 0.0)
            vx = 0.7 * vx + wx + self.gravity * (ex - x) / dist     # 0.7: drag, prevents orbiting
            vy = 0.7 * vy + wy + self.gravity * (ey - y) / dist
            vcap = self.vmax if far else max(self.vmax * dist / near, 1.5)
            v = np.hypot(vx, vy)
            if v > vcap:                            # speed limit with a random haircut
                lim = vcap * rng.uniform(0.5, 1.0)
                vx, vy = vx / v * lim, vy / v * lim
            x, y = x + vx, y + vy
            xs.append(x)
            ys.append(y)
        xs.append(ex)
        ys.append(ey)
        tk = np.arange(len(xs)) * self.tick
        tm = _dense(tk[-1])
        return _assemble(rng.uniform(0.08, 0.3), tm, np.interp(tm, tk, xs), np.interp(tm, tk, ys),
                         rng.uniform(0.03, 0.10), rng.uniform(0.05, 0.11), sx, sy)


class FittsAware(_Bot):
    """Adaptive adversary: movement time drawn from a human-range Fitts line with
    log-normal noise, gaussian endpoints scaled to the target, one smooth stroke."""

    family = "fitts_aware"
    event_jitter = 0.0012

    def new_session(self, rng):
        self.a = rng.uniform(0.18, 0.32)
        self.b = rng.uniform(0.12, 0.20)

    def move(self, sx, sy, tg, rng):
        dist = np.hypot(tg.x - sx, tg.y - sy)
        mt = (self.a + self.b * np.log2(dist / tg.w + 1.0)) * np.exp(rng.normal(0.0, 0.17))
        wait = min(rng.uniform(0.12, 0.25), 0.4 * mt)
        after = min(rng.uniform(0.03, 0.10), 0.2 * mt)
        ex = tg.x + rng.normal(0.0, tg.w / 4.133)
        ey = tg.y + rng.normal(0.0, tg.w / 4.133)
        tm = _dense(max(mt - wait - after, 0.06))
        u = tm / tm[-1]
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        x, y = _bezier(sx, sy, ex, ey, rng, spread=0.08)(s)
        return _assemble(wait, tm, x, y, after, 0.095 * np.exp(rng.normal(0, 0.25)), sx, sy)


class Replay(_Bot):
    """Replays recorded human movements, rotated and scaled onto the new target.

    Timing is copied from the recording, so each replayed movement was a valid
    human movement for *its* original target, not for the one being clicked.
    """

    family = "replay"
    event_jitter = 0.0012

    def __init__(self, pool: list[Trial]):
        self.pool = [t for t in pool if len(t.t) >= 5 and t.distance > 5]
        if not self.pool:
            raise ValueError("replay needs recorded trials with trajectories")

    def move(self, sx, sy, tg, rng):
        src = self.pool[int(rng.integers(len(self.pool)))]
        ex, ey = _in_disk(rng, tg, 0.5)
        # similarity transform taking (src start -> src click) onto (start -> chosen endpoint)
        a = complex(src.click_x - src.start_x, src.click_y - src.start_y)
        b = complex(ex - sx, ey - sy)
        if abs(a) < 1e-6:
            a = complex(1e-6, 0.0)
        z = (src.x - src.start_x + 1j * (src.y - src.start_y)) * (b / a)
        t = src.t - src.t[0]
        hold = (src.t_up - src.t_down) if src.t_up is not None else 0.09
        mv = Move(t=np.append(t, src.movement_time), x=np.append(sx + z.real, ex),
                  y=np.append(sy + z.imag, ey), t_down=float(src.movement_time), hold=float(hold))
        return mv


def default_zoo(replay_pool: list[Trial] | None = None) -> list[_Bot]:
    zoo: list[_Bot] = [Teleport(), TeleportJitter(), ConstSpeed(), ConstTime(), EaseTween(),
                       Bezier(), Wind(), FittsAware()]
    if replay_pool:
        zoo.append(Replay(replay_pool))
    return zoo


TIERS = {
    "teleport": 0, "teleport_jitter": 0,
    "const_speed": 1, "const_time": 1, "ease_tween": 1,
    "bezier": 2, "wind": 2,
    "fitts_aware": 3, "replay": 3, "sim_bot": 3,
}
