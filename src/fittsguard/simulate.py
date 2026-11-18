"""Running an agent (simulated person or bot) through a sequence of targets.

Every agent implements :class:`Agent`: given where the cursor is and where the
target is, return a densely sampled path plus the click timing. The *recorder*
then turns that path into what a server would actually receive: pointer events
at a finite rate, integer pixels, no event while the cursor is still.

Humans and bots go through the same recorder and the same layouts, so the only
thing that differs between the classes is the movement itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .schema import Session, Trial

DENSE_HZ = 1000.0


@dataclass
class Target:
    x: float
    y: float
    w: float


@dataclass
class Layout:
    """Initial cursor position and the targets to click, in order.

    ``starts`` is None for serial pointing (each trial begins where the last
    click landed). For discrete pointing (the task puts the cursor on a start
    area before each target) it holds one start position per trial.
    """

    start: tuple[float, float]
    targets: list[Target]
    starts: list[tuple[float, float]] | None = None

    @classmethod
    def from_session(cls, s: Session) -> "Layout":
        """Reproduce the exact start/target geometry of a recorded session."""
        starts = [(t.start_x, t.start_y) for t in s.trials]
        return cls(starts[0], [Target(t.target_x, t.target_y, t.target_w) for t in s.trials], starts)


@dataclass
class Move:
    """One aimed movement, on the agent's own dense clock starting at 0."""

    t: np.ndarray        # (n,) seconds, starts at 0
    x: np.ndarray
    y: np.ndarray
    t_down: float        # >= t[-1] is allowed: cursor rests, then clicks
    hold: float          # button hold time, seconds


class Agent(Protocol):
    family: str
    label: str
    event_hz: float      # rate at which this agent's pointer events reach the page
    event_jitter: float  # s.d. of event timing jitter, seconds

    def new_session(self, rng: np.random.Generator) -> None: ...
    def move(self, sx: float, sy: float, tg: Target, rng: np.random.Generator) -> Move: ...


def record(mv: Move, hz: float, jitter: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense path -> pointer events (finite rate, integer px, only on change)."""
    n = int(np.floor(mv.t_down * hz)) + 1
    te = np.arange(n) / hz
    if jitter > 0 and n > 2:
        te[1:] = np.sort(np.clip(te[1:] + rng.normal(0.0, jitter, n - 1), 1e-4, mv.t_down))
    xe = np.round(np.interp(te, mv.t, mv.x))
    ye = np.round(np.interp(te, mv.t, mv.y))
    keep = np.concatenate([[True], (np.diff(xe) != 0) | (np.diff(ye) != 0)])
    return te[keep], xe[keep], ye[keep]


def run_layout(agent: Agent, layout: Layout, rng: np.random.Generator, subject: str = "",
               age: float | None = None) -> Session:
    """Each trial starts at the previous click (serial) or at the layout's own start point."""
    agent.new_session(rng)
    sx, sy = layout.start
    trials: list[Trial] = []
    for i, tg in enumerate(layout.targets):
        if layout.starts is not None:
            sx, sy = layout.starts[i]
        mv = agent.move(sx, sy, tg, rng)
        t, x, y = record(mv, agent.event_hz, agent.event_jitter, rng)
        cx = float(np.round(np.interp(mv.t_down, mv.t, mv.x)))
        cy = float(np.round(np.interp(mv.t_down, mv.t, mv.y)))
        trials.append(Trial(t=t, x=x, y=y, target_x=tg.x, target_y=tg.y, target_w=tg.w,
                            t_down=float(mv.t_down), t_up=float(mv.t_down + mv.hold),
                            click_x=cx, click_y=cy))
        sx, sy = cx, cy
    return Session(trials, label=agent.label, family=agent.family, subject=subject, age=age)


def random_layout(rng: np.random.Generator, n: int, screen: tuple[int, int] = (1600, 900),
                  widths: Sequence[float] = (16, 24, 32, 48, 64, 96),
                  d_range: tuple[float, float] = (80.0, 900.0)) -> Layout:
    """Targets scattered over a screen with log-uniform distances and mixed sizes.

    Mixed sizes at comparable distances are what make the width term of Fitts'
    law identifiable inside a single session.
    """
    sx, sy = screen[0] / 2.0, screen[1] / 2.0
    start = (sx, sy)
    out: list[Target] = []
    for _ in range(n):
        w = float(rng.choice(widths))
        for _try in range(200):
            dist = float(np.exp(rng.uniform(np.log(d_range[0]), np.log(d_range[1]))))
            ang = rng.uniform(0.0, 2.0 * np.pi)
            x, y = sx + dist * np.cos(ang), sy + dist * np.sin(ang)
            if w <= x <= screen[0] - w and w <= y <= screen[1] - w:
                break
        else:
            x, y = screen[0] / 2.0, screen[1] / 2.0
        out.append(Target(float(x), float(y), w))
        sx, sy = x, y
    return Layout(start, out)
