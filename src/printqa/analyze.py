"""Measure a scanned test chart.

1. Find the four fiducial dots and warp the scan back onto the chart's own coordinates, undoing the scanner's
   rotation, scale and offset, so every later measurement knows exactly where to look.
2. Convert to optical density per channel (D = -log10 reflectance relative to the paper), the unit print quality is
   specified in. Cyan is read in the red channel, magenta in green, yellow in blue, black in all three.
3. Measure each check:
   density     mean density of each solid patch
   streaks     columns of a solid whose density departs from a running median by more than the spec and by more than
               6 robust standard deviations of normal unevenness
   banding     the strongest periodic component of the row profile (FFT), with its period and amplitude
   nozzles     which nozzle-check lines are missing
   registration  where each color's crosses sit relative to the black ones, in micrometers
   tone        density of each tone step and the tone value increase (dot gain) at 50%
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import chart as c

UM_PER_PX = 25400 / c.DPI
CHANNEL = {"C": 0, "M": 1, "Y": 2}


class AlignmentError(RuntimeError):
    pass


@dataclass
class PageResult:
    density: dict[str, float] = field(default_factory=dict)
    streaks: list[dict] = field(default_factory=list)  # sep, x_mm, width_mm, contrast
    banding: dict[str, dict] = field(default_factory=dict)  # sep -> period_mm, amplitude
    dead_nozzles: list[int] = field(default_factory=list)
    registration_um: dict[str, tuple[float, float]] = field(default_factory=dict)  # sep -> (dx, dy) relative to K
    tone: list[float] = field(default_factory=list)
    tvi50: float = 0.0
    alignment_residual_px: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def find_fiducials(rgb: np.ndarray) -> np.ndarray:
    """Centers of the four corner dots (x, y), ordered like chart.FIDUCIALS, with sub-pixel precision."""
    gray = rgb.mean(axis=2)
    mask = (gray < 0.45).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    expected = np.pi * c.FIDUCIAL_RADIUS**2
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        fill = area / max(w * h, 1)
        if 0.6 * expected < area < 1.6 * expected and 0.8 < w / max(h, 1) < 1.25 and 0.65 < fill < 0.9:
            ys, xs = np.nonzero(labels[y : y + h, x : x + w] == i)
            wts = (1 - gray[y : y + h, x : x + w])[ys, xs]
            blobs.append(((xs * wts).sum() / wts.sum() + x, (ys * wts).sum() / wts.sum() + y))
    if len(blobs) < 4:
        raise AlignmentError(f"found {len(blobs)} fiducials, need 4")
    h, w = gray.shape
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    pts = np.array(blobs)
    return np.array([pts[np.argmin(((pts - corner) ** 2).sum(axis=1))] for corner in corners], np.float32)


def align(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    src = find_fiducials(rgb)
    dst = np.array(c.FIDUCIALS, np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rgb, m, (c.W, c.H), flags=cv2.INTER_LINEAR, borderValue=(1.0, 1.0, 1.0))
    # how well a similarity transform alone explains the dots: large values mean a bad scan or a wrong detection
    sim, _ = cv2.estimateAffinePartial2D(src, dst)
    resid = float(np.abs(cv2.transform(src[None], sim)[0] - dst).max())
    return warped, resid


def density(rgb: np.ndarray) -> np.ndarray:
    p = c.PAPER_PROBE
    white = np.median(rgb[p.y0 : p.y1, p.x0 : p.x1].reshape(-1, 3), axis=0)
    return -np.log10(np.clip(rgb / white, 1e-3, 1.0))


def channel(d: np.ndarray, sep: str) -> np.ndarray:
    return d[..., CHANNEL[sep]] if sep in CHANNEL else d.mean(axis=2)


def rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def find_streaks(region: np.ndarray, min_contrast: float, k_sigma: float = 6.0) -> list[dict]:
    prof = region.mean(axis=0)
    resid = prof - rolling_median(prof, 41)
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    hot = np.abs(resid) > max(k_sigma * sigma, min_contrast)
    out, x = [], 0
    while x < len(hot):
        if hot[x]:
            x0 = x
            while x < len(hot) and hot[x]:
                x += 1
            seg = resid[x0:x]
            out.append({"x": x0, "width": x - x0, "contrast": float(seg[np.argmax(np.abs(seg))])})
        else:
            x += 1
    return out


def find_banding(region: np.ndarray, min_period: int = 8, max_period: int = 300) -> dict:
    prof = region.mean(axis=1)
    t = np.arange(len(prof))
    prof = prof - np.polyval(np.polyfit(t, prof, 1), t)
    win = np.hanning(len(prof))
    spec = np.abs(np.fft.rfft(prof * win)) * 2 / win.sum()
    freqs = np.fft.rfftfreq(len(prof))
    ok = (freqs > 1 / max_period) & (freqs < 1 / min_period)
    i = np.argmax(np.where(ok, spec, 0))
    return {"period_px": float(1 / freqs[i]), "amplitude": float(spec[i])}


def _profile_center(profile: np.ndarray) -> float:
    p = np.clip(profile - np.median(profile), 0, None)
    idx = np.arange(len(p)) - len(p) // 2
    return float((idx * p).sum() / p.sum())


def cross_center(d: np.ndarray, sep: str, x: int, y: int) -> tuple[float, float]:
    """Center of a registration cross, from the cross-section profiles of its arms.

    The vertical arm, averaged down its length (skipping the middle), gives a column profile whose centroid is the
    x position; the horizontal arm gives y. Averaging along an arm cancels ink mottle, and using the profile rather
    than a thresholded shape avoids the bias a slightly tilted line causes at a threshold edge."""
    a, w, gap = c.CROSS_ARM, 10, 6
    img = channel(d, sep)
    vert = np.vstack([img[y - a : y - gap, x - w : x + w + 1], img[y + gap + 1 : y + a + 1, x - w : x + w + 1]])
    horiz = np.hstack([img[y - w : y + w + 1, x - a : x - gap], img[y - w : y + w + 1, x + gap + 1 : x + a + 1]])
    return _profile_center(vert.mean(axis=0)), _profile_center(horiz.mean(axis=1))


def analyze(scan_rgb: np.ndarray, min_streak: float = 0.02) -> PageResult:
    rgb = scan_rgb.astype(np.float32) / 255.0 if scan_rgb.dtype == np.uint8 else scan_rgb
    page, resid = align(rgb)
    d = density(page)
    res = PageResult(alignment_residual_px=round(resid, 3))
    for sep, r in c.PATCHES.items():
        region = channel(d, sep)[r.inset(20).y0 : r.inset(20).y1, r.inset(20).x0 : r.inset(20).x1]
        res.density[sep] = round(float(region.mean()), 4)
        for s in find_streaks(region, min_streak):
            res.streaks.append(
                {
                    "sep": sep,
                    "x_mm": round((r.inset(20).x0 + s["x"]) * 25.4 / c.DPI, 2),
                    "width_mm": round(s["width"] * 25.4 / c.DPI, 2),
                    "contrast": round(s["contrast"], 4),
                }
            )
        b = find_banding(region)
        res.banding[sep] = {"period_mm": round(b["period_px"] * 25.4 / c.DPI, 2), "amplitude": round(b["amplitude"], 4)}
    k = d.mean(axis=2)
    peaks = []
    for i in range(c.NOZZLES):
        b = c.nozzle_box(i)
        peaks.append(float(k[b.y0 + 4 : b.y1 - 4, b.x0 - 3 : b.x1 + 3].mean(axis=0).max()))
    typical = float(np.median(peaks))
    res.dead_nozzles = [i for i, p in enumerate(peaks) if p < 0.4 * typical]
    centers = {}
    for sep in c.SEPARATIONS:
        offs = [cross_center(d, sep, *c.CROSSES[(sep, side)]) for side in ("left", "right")]
        centers[sep] = np.mean(offs, axis=0)
    for sep in ("C", "M", "Y"):
        dx, dy = (centers[sep] - centers["K"]) * UM_PER_PX
        res.registration_um[sep] = (round(float(dx), 1), round(float(dy), 1))
    res.tone = [
        round(float(k[r.inset(15).y0 : r.inset(15).y1, r.inset(15).x0 : r.inset(15).x1].mean()), 4) for r in c.TONES
    ]
    solid = res.tone[-1]
    apparent = (1 - 10 ** -res.tone[5]) / (1 - 10**-solid)  # Murray-Davies coverage of the 50% step
    res.tvi50 = round(float(apparent - 0.5), 4)
    return res
