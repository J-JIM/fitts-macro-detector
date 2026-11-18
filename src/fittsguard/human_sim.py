"""A generative model of human point-and-click movements.

Used for three things: unit tests, building age cohorts (public pointing data
with ages attached is scarce), and as the strongest adversary in the harness
("a bot that implements everything this file knows about people").

The model is a stochastic submovement model in the spirit of Meyer et al.
(1988): a ballistic primary submovement that lands near the target with
signal-dependent error, followed by zero or more corrective submovements, then
a short verification pause before the button goes down. Each submovement has an
asymmetric minimum-jerk speed profile. Total time follows the subject's own
Fitts line with multiplicative (log-normal) noise, so the simulated person can
be calibrated directly to published ``a``, ``b`` values.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .simulate import DENSE_HZ, Move, Target


@dataclass
class MotorProfile:
    """One person's motor parameters. Times in seconds."""

    a: float = 0.25            # Fitts intercept (click-to-click)
    b: float = 0.16            # Fitts slope, s/bit
    sigma_mt: float = 0.17     # s.d. of log movement-time noise
    kappa: float = 1.0         # endpoint spread multiplier; 1 -> We == W (about 4 % misses in 1-D)
    latency: float = 0.17      # median time before the cursor starts moving
    settle: float = 0.07       # median verification pause before button-down
    hold: float = 0.095        # median button hold time
    corr_rate: float = 0.35    # corrective submovements per bit above 1.5 bits
    skew: float = 0.85         # time-warp exponent; < 1 puts the speed peak early
    bow: float = 0.04          # s.d. of lateral path curvature as a fraction of D
    tremor: float = 0.35       # px, band-limited positional noise while moving


def _minjerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


class HumanSim:
    """Simulated person. ``profile`` fixes the individual; noise is per trial."""

    label = "human"
    family = "sim_human"
    event_hz = 125.0
    event_jitter = 0.0012

    def __init__(self, profile: MotorProfile | None = None):
        self.profile = profile or MotorProfile()

    def new_session(self, rng: np.random.Generator) -> None:  # nothing carries over between sessions
        return None

    def move(self, sx: float, sy: float, tg: Target, rng: np.random.Generator) -> Move:
        p = self.profile
        dx, dy = tg.x - sx, tg.y - sy
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            dist, dx, dy = 1e-6, 1e-6, 0.0
        ux, uy = dx / dist, dy / dist
        idb = float(np.log2(dist / tg.w + 1.0))

        eps = rng.normal(0.0, p.sigma_mt)
        mt = max((p.a + p.b * idb) * np.exp(eps), 0.16)
        lat = min(p.latency * np.exp(rng.normal(0.0, 0.25)), 0.45 * mt)
        settle = min((p.settle + 0.012 * idb) * np.exp(rng.normal(0.0, 0.45)), 0.30 * mt)
        dur = max(mt - lat - settle, 0.08)
        mt = lat + dur + settle

        # endpoint: normal around a point slightly short of centre; hurried trials scatter more
        sd_along = tg.w / 4.133 * p.kappa * np.exp(-0.6 * eps)
        e_al = rng.normal(-0.04 * tg.w, sd_along)
        e_or = rng.normal(0.0, 0.75 * sd_along)
        ex = tg.x + e_al * ux - e_or * uy
        ey = tg.y + e_al * uy + e_or * ux

        n_corr = int(min(rng.poisson(max(0.0, p.corr_rate * (idb - 1.5))), 3))
        # way-points: primary lands short/long with direction error, corrections close the gap
        pts = [(sx, sy)]
        if n_corr > 0:
            gain = rng.normal(0.94, 0.06)
            aim = rng.normal(0.0, 0.035) * dist
            px, py = sx + gain * (ex - sx) - aim * uy, sy + gain * (ey - sy) + aim * ux
            pts.append((px, py))
            for k in range(n_corr - 1):
                rem = rng.uniform(0.15, 0.45)
                px, py = px + (1 - rem) * (ex - px), py + (1 - rem) * (ey - py)
                pts.append((px, py))
        pts.append((ex, ey))
        nsub = len(pts) - 1

        # time allotted to each submovement: primary takes most of it
        if nsub == 1:
            shares = np.array([1.0])
        else:
            first = rng.uniform(0.55, 0.78)
            rest = rng.dirichlet(np.full(nsub - 1, 6.0)) * (1.0 - first)
            shares = np.concatenate([[first], rest])
        durs = shares * dur
        overlap = rng.uniform(0.05, 0.25)
        starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
        starts[1:] -= overlap * durs[:-1]
        lens = durs.copy()
        lens[1:] += overlap * durs[:-1]   # later submovements start early and still end on time

        t = np.arange(0.0, mt + 1.0 / DENSE_HZ, 1.0 / DENSE_HZ)
        tm = t - lat
        x = np.full_like(t, sx)
        y = np.full_like(t, sy)
        for i in range(nsub):
            g = rng.normal(p.skew, 0.07)
            s = _minjerk(np.clip((tm - starts[i]) / lens[i], 0.0, 1.0) ** max(g, 0.5))
            x += (pts[i + 1][0] - pts[i][0]) * s
            y += (pts[i + 1][1] - pts[i][1]) * s
            if i == 0:
                bow = rng.normal(0.0, p.bow) * dist
                x += -uy * bow * np.sin(np.pi * s)
                y += ux * bow * np.sin(np.pi * s)

        # physiological tremor: band-limited noise, only while the hand is travelling
        if p.tremor > 0:
            env = np.clip(tm / 0.03, 0.0, 1.0) * np.clip((dur - tm) / 0.03, 0.0, 1.0)
            k = np.exp(-0.5 * (np.arange(-60, 61) / 18.0) ** 2)
            k /= np.sqrt(np.sum(k ** 2))
            x += env * p.tremor * np.convolve(rng.normal(size=len(t)), k, mode="same")
            y += env * p.tremor * np.convolve(rng.normal(size=len(t)), k, mode="same")

        hold = p.hold * np.exp(rng.normal(0.0, 0.25))
        return Move(t=t, x=x, y=y, t_down=float(mt), hold=float(hold))


def perturb(profile: MotorProfile, rng: np.random.Generator, cv: float = 0.15) -> MotorProfile:
    """A different individual drawn around ``profile`` (between-subject variation)."""
    j = lambda v, s=cv: float(v * np.exp(rng.normal(0.0, s)))  # noqa: E731
    return replace(
        profile,
        a=j(profile.a), b=j(profile.b), sigma_mt=j(profile.sigma_mt, 0.2), kappa=j(profile.kappa, 0.12),
        latency=j(profile.latency), settle=j(profile.settle, 0.25), hold=j(profile.hold, 0.2),
        corr_rate=j(profile.corr_rate, 0.25), skew=float(np.clip(profile.skew + rng.normal(0, 0.05), 0.6, 1.1)),
        bow=j(profile.bow, 0.3), tremor=j(profile.tremor, 0.3),
    )
