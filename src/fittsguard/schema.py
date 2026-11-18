"""Data model: a *trial* is one aimed movement that ends in a click on a known target.

Everything the detector needs is in here, and nothing else. A server that knows
where its own buttons are (centre + size) and receives pointer events can build
these objects; no client-side instrumentation beyond ordinary mouse events is
assumed.

Units: seconds and pixels. ``t`` is relative to the start of the trial, which is
the moment the *previous* click was released (serial pointing) or the moment the
target appeared (discrete pointing).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class Trial:
    t: np.ndarray            # (n,) seconds, monotonically non-decreasing, t[0] == trial start
    x: np.ndarray            # (n,) px
    y: np.ndarray            # (n,) px
    target_x: float
    target_y: float
    target_w: float          # target size along the approach axis (diameter for circles), px
    t_down: float            # button-down time, seconds (same clock as ``t``)
    click_x: float
    click_y: float
    t_up: float | None = None  # button-up time; None if the log has no release events

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if not (len(self.t) == len(self.x) == len(self.y)):
            raise ValueError("t, x, y must have the same length")
        if len(self.t) < 1:
            raise ValueError("a trial needs at least the start sample")
        if self.target_w <= 0:
            raise ValueError("target_w must be positive")

    # -- geometry ---------------------------------------------------------
    @property
    def start_x(self) -> float:
        return float(self.x[0])

    @property
    def start_y(self) -> float:
        return float(self.y[0])

    @property
    def distance(self) -> float:
        """Centre-to-start distance D (px)."""
        return float(np.hypot(self.target_x - self.start_x, self.target_y - self.start_y))

    @property
    def index_of_difficulty(self) -> float:
        """Shannon formulation, ID = log2(D / W + 1) in bits."""
        return float(np.log2(self.distance / self.target_w + 1.0))

    @property
    def movement_time(self) -> float:
        """Trial start to button-down (s). Includes reaction and verification time."""
        return float(self.t_down - self.t[0])

    @property
    def hit(self) -> bool:
        r = np.hypot(self.click_x - self.target_x, self.click_y - self.target_y)
        return bool(r <= self.target_w / 2.0)

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "t": [round(float(v), 5) for v in self.t],
            "x": [round(float(v), 2) for v in self.x],
            "y": [round(float(v), 2) for v in self.y],
            "target": {"x": self.target_x, "y": self.target_y, "w": self.target_w},
            "click": {"t_down": self.t_down, "t_up": self.t_up, "x": self.click_x, "y": self.click_y},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        return cls(
            t=d["t"], x=d["x"], y=d["y"],
            target_x=d["target"]["x"], target_y=d["target"]["y"], target_w=d["target"]["w"],
            t_down=d["click"]["t_down"], t_up=d["click"].get("t_up"),
            click_x=d["click"]["x"], click_y=d["click"]["y"],
        )


@dataclass
class Session:
    """A run of trials produced by one agent (a person or a program)."""

    trials: list[Trial]
    label: str = "unknown"          # "human" | "bot" | "unknown"
    family: str = "unknown"         # data source or bot family
    subject: str = ""               # stable id of the agent; used for group-wise CV
    age: float | None = None        # years, if known
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trials)

    def slice(self, start: int, stop: int) -> "Session":
        return Session(self.trials[start:stop], self.label, self.family, self.subject, self.age, dict(self.meta))

    def windows(self, k: int, step: int | None = None) -> Iterable["Session"]:
        """Consecutive windows of ``k`` trials (non-overlapping by default)."""
        step = step or k
        for s in range(0, len(self.trials) - k + 1, step):
            yield self.slice(s, s + k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "family": self.family, "subject": self.subject,
            "age": self.age, "meta": self.meta,
            "trials": [t.to_dict() for t in self.trials],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        return cls(
            trials=[Trial.from_dict(t) for t in d["trials"]],
            label=d.get("label", "unknown"), family=d.get("family", "unknown"),
            subject=str(d.get("subject", "")), age=d.get("age"), meta=d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        return cls.from_dict(json.loads(Path(path).read_text()))
