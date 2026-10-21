"""Pass/fail limits for a page, and the checks against them. Density targets come from calibration on known-good
pages, because they depend on the press, ink and paper; the other limits are engineering choices kept in YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .analyze import PageResult

DEFAULT_LIMITS = {
    "density_tolerance": 0.10,  # |measured - target| optical density
    "max_streak_contrast": 0.03,  # optical density
    "max_banding_amplitude": 0.015,  # optical density
    "max_dead_nozzles": 0,
    "max_registration_um": 60.0,
    "tvi50_range": [0.0, 0.15],
}


@dataclass
class Check:
    name: str
    value: float
    limit: str
    passed: bool


def calibrate(results: list[PageResult]) -> dict:
    targets = {sep: round(float(np.median([r.density[sep] for r in results])), 4) for sep in results[0].density}
    return {"density_targets": targets, **DEFAULT_LIMITS}


def load(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    missing = {"density_targets", *DEFAULT_LIMITS} - set(spec)
    if missing:
        raise ValueError(f"spec is missing {sorted(missing)}")
    return spec


def check(r: PageResult, spec: dict) -> list[Check]:
    out = []
    for sep, target in spec["density_targets"].items():
        dev = r.density[sep] - target
        out.append(
            Check(
                f"density {sep}", round(dev, 4), f"±{spec['density_tolerance']}", abs(dev) <= spec["density_tolerance"]
            )
        )
    worst = max((abs(s["contrast"]) for s in r.streaks), default=0.0)
    out.append(
        Check("streaks", round(worst, 4), f"≤{spec['max_streak_contrast']}", worst <= spec["max_streak_contrast"])
    )
    band = max(b["amplitude"] for b in r.banding.values())
    out.append(
        Check("banding", round(band, 4), f"≤{spec['max_banding_amplitude']}", band <= spec["max_banding_amplitude"])
    )
    out.append(
        Check(
            "dead nozzles",
            len(r.dead_nozzles),
            f"≤{spec['max_dead_nozzles']}",
            len(r.dead_nozzles) <= spec["max_dead_nozzles"],
        )
    )
    for sep, (dx, dy) in r.registration_um.items():
        mag = float(np.hypot(dx, dy))
        out.append(
            Check(
                f"registration {sep}",
                round(mag, 1),
                f"≤{spec['max_registration_um']} µm",
                mag <= spec["max_registration_um"],
            )
        )
    lo, hi = spec["tvi50_range"]
    out.append(Check("tone value increase 50%", r.tvi50, f"{lo}–{hi}", lo <= r.tvi50 <= hi))
    return out
