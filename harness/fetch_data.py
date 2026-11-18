"""Fetch the third-party data the harness runs on. Nothing here is redistributed.

    python harness/fetch_data.py

Input Accessibility: Findlater, L. and Zhang, L. (2020). Input Accessibility: A
Large Dataset and Summary Analysis of Age, Motor Ability and Input Performance.
ACM ASSETS 2020. Licence CC BY 4.0. https://github.com/leahkf/mouseandtouchinput

Only the mouse/touchpad pointing blocks and the questionnaire are downloaded
(about 250 MB of the 3.3 GB repository) using a partial, sparse git clone.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
REPO = "https://github.com/leahkf/mouseandtouchinput.git"
COMMIT = "376fbd0640d99f1ae89d6384a42d51966e4a04f5"     # pinned: the harness numbers refer to this snapshot
PATTERNS = ["/*.csv", "/*.md", "/*.xlsx", "alldata/*/*DeviceType=DESKTOP*TaskName=Pointing*"]


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch_input_accessibility() -> Path:
    dest = RAW / "input_accessibility"
    if (dest / "alldata").exists():
        print(f"already present: {dest}")
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    git("clone", "--filter=blob:none", "--no-checkout", REPO, str(dest))
    git("sparse-checkout", "set", "--no-cone", *PATTERNS, cwd=dest)
    git("checkout", COMMIT, cwd=dest)
    n = len(list(dest.glob("alldata/*/*.json")))
    print(f"fetched {n} pointing blocks into {dest}")
    return dest


if __name__ == "__main__":
    fetch_input_accessibility()
    sys.exit(0)
