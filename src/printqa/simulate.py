"""A print-and-scan simulator with injectable defects, so the analyzer can be tested against known truth.

Each separation is a coverage map (0 = no ink, 1 = solid). The press model adds dot gain, per-separation
registration shifts and the defects; ink absorbs light per RGB channel; the scanner adds blur, rotation, scale,
offset, noise and 8-bit quantization."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import chart as c

# fraction of red, green and blue light each ink absorbs at full coverage
ABSORB = {"C": (0.85, 0.22, 0.06), "M": (0.08, 0.84, 0.28), "Y": (0.02, 0.06, 0.82), "K": (0.88, 0.88, 0.88)}
PAPER = 0.93
DOT_GAIN = 0.15
MOTTLE = 0.012  # low-frequency ink-density unevenness, as a fraction of coverage


@dataclass
class Defects:
    """Ground truth for one page. Positions and sizes in chart pixels (300 dpi) unless noted."""

    streaks: list[dict] = field(default_factory=list)  # {sep, x, width, depth}: depth > 0 lighter, < 0 darker
    banding: list[dict] = field(default_factory=list)  # {sep, period, amplitude}: coverage x (1 + a sin)
    dead_nozzles: list[int] = field(default_factory=list)
    registration: dict[str, tuple[float, float]] = field(default_factory=dict)  # sep -> (dx, dy) in pixels
    density: dict[str, float] = field(default_factory=dict)  # sep -> coverage multiplier (ink starvation < 1)


def coverage(defects: Defects) -> dict[str, np.ndarray]:
    cov = {s: np.zeros((c.H, c.W), np.float32) for s in c.SEPARATIONS}
    for sep, r in c.PATCHES.items():
        cov[sep][r.y0 : r.y1, r.x0 : r.x1] = 1.0
    for i in range(c.NOZZLES):
        if i not in defects.dead_nozzles:
            b = c.nozzle_box(i)
            cov["K"][b.y0 : b.y1, b.x0 : b.x1] = 1.0
    t, a = c.CROSS_THICK, c.CROSS_ARM
    for (sep, _), (x, y) in c.CROSSES.items():
        cov[sep][y - t // 2 : y + t // 2 + 1, x - a : x + a + 1] = 1.0
        cov[sep][y - a : y + a + 1, x - t // 2 : x + t // 2 + 1] = 1.0
    for level, r in zip(c.TONE_LEVELS, c.TONES, strict=True):
        cov["K"][r.y0 : r.y1, r.x0 : r.x1] = level
    for x, y in c.FIDUCIALS:
        cv2.circle(cov["K"], (x, y), c.FIDUCIAL_RADIUS, 1.0, -1, lineType=cv2.LINE_AA)
    return cov


def press(defects: Defects, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    cov = coverage(defects)
    rng = rng or np.random.default_rng(0)
    yy = np.arange(c.H, dtype=np.float32)[:, None]
    for sep in c.SEPARATIONS:
        m = cov[sep]
        m = m + DOT_GAIN * m * (1 - m)
        # mottle: smooth random unevenness, so a real streak has to stand out from normal variation
        field = cv2.GaussianBlur(rng.normal(0, 1, (c.H // 4, c.W // 4)).astype(np.float32), (0, 0), 3)
        field = cv2.resize(field / (field.std() + 1e-9), (c.W, c.H), interpolation=cv2.INTER_CUBIC)
        m = m * (1 + MOTTLE * field)
        m *= defects.density.get(sep, 1.0)
        for b in defects.banding:
            if b["sep"] == sep:
                m *= 1 + b["amplitude"] * np.sin(2 * np.pi * yy / b["period"])
        for s in defects.streaks:
            if s["sep"] == sep:
                x0, x1 = int(s["x"]), int(s["x"] + s["width"])
                m[:, x0:x1] *= 1 - s["depth"]
        dx, dy = defects.registration.get(sep, (0.0, 0.0))
        if dx or dy:
            m = cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (c.W, c.H), flags=cv2.INTER_LINEAR)
        cov[sep] = np.clip(m, 0, 1)
    return cov


def render(defects: Defects, rng: np.random.Generator | None = None) -> np.ndarray:
    """Reflectance image (H x W x 3, RGB, 0..1) of the printed page."""
    cov = press(defects, rng)
    rgb = np.full((c.H, c.W, 3), PAPER, np.float32)
    for sep in c.SEPARATIONS:
        for ch in range(3):
            rgb[..., ch] *= 1 - ABSORB[sep][ch] * cov[sep]
    return rgb


@dataclass
class ScanParams:
    rotation_deg: float = 0.0
    scale: float = 1.0
    shift: tuple[float, float] = (0.0, 0.0)
    noise: float = 0.006
    blur: float = 0.7
    margin: int = 60


def random_scan(rng: np.random.Generator) -> ScanParams:
    return ScanParams(
        rotation_deg=rng.uniform(-0.6, 0.6),
        scale=rng.uniform(0.995, 1.005),
        shift=(rng.uniform(-15, 15), rng.uniform(-15, 15)),
    )


def scan(page: np.ndarray, p: ScanParams, rng: np.random.Generator) -> np.ndarray:
    """What a flatbed scanner returns: a slightly rotated, scaled and shifted 8-bit RGB image on a white bed."""
    h, w = page.shape[:2]
    out_w, out_h = w + 2 * p.margin, h + 2 * p.margin
    m = cv2.getRotationMatrix2D((w / 2, h / 2), p.rotation_deg, p.scale)
    m[:, 2] += (p.margin + p.shift[0], p.margin + p.shift[1])
    img = cv2.warpAffine(page, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0.97, 0.97, 0.97))
    if p.blur:
        img = cv2.GaussianBlur(img, (0, 0), p.blur)
    img = img + rng.normal(0, p.noise, img.shape).astype(np.float32)
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


def print_and_scan(defects: Defects, rng: np.random.Generator, params: ScanParams | None = None) -> np.ndarray:
    return scan(render(defects, rng), params or random_scan(rng), rng)
