"""The test chart: where everything is, in inches, so the renderer and the analyzer share one layout.

fiducials       four black dots near the corners, used to undo the scanner's rotation, scale and shift
solid patches   cyan, magenta, yellow and black solids, where streaks, banding and density are measured
nozzle check    one short black line per nozzle in four staggered rows; a missing line is a dead nozzle
registration    a cross in each color at two places; how far C, M and Y sit from where they should is misregistration
tone steps      eleven black patches from 0% to 100% coverage, for the tone curve
"""

from __future__ import annotations

from dataclasses import dataclass

DPI = 300
WIDTH_IN, HEIGHT_IN = 6.0, 8.0
SEPARATIONS = ("C", "M", "Y", "K")


def px(inches: float) -> int:
    return int(round(inches * DPI))


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    def inset(self, n: int) -> Rect:
        return Rect(self.x0 + n, self.y0 + n, self.x1 - n, self.y1 - n)


W, H = px(WIDTH_IN), px(HEIGHT_IN)
FIDUCIAL_RADIUS = px(0.06)
FIDUCIALS = [(px(0.25), px(0.25)), (W - px(0.25), px(0.25)), (px(0.25), H - px(0.25)), (W - px(0.25), H - px(0.25))]

PATCHES = {sep: Rect(px(0.5 + 1.3 * i), px(0.6), px(0.5 + 1.3 * i + 1.1), px(2.6)) for i, sep in enumerate(SEPARATIONS)}

NOZZLES = 96
NOZZLE_PITCH = px(5.0) / NOZZLES  # nozzles spread over 5 inches
NOZZLE_ROWS = 4
NOZZLE_LINE = (2, px(0.12))  # width, height in pixels


def nozzle_box(i: int) -> Rect:
    x = int(round(px(0.5) + i * NOZZLE_PITCH))
    y = px(2.9) + (i % NOZZLE_ROWS) * px(0.2)
    return Rect(x, y, x + NOZZLE_LINE[0], y + NOZZLE_LINE[1])


CROSS_ARM = px(0.12)
CROSS_THICK = 3
CROSSES = {
    (sep, side): (px(0.8 + 0.5 * i) + (px(2.6) if side == "right" else 0), px(4.0))
    for i, sep in enumerate(SEPARATIONS)
    for side in ("left", "right")
}

TONE_LEVELS = [i / 10 for i in range(11)]
TONES = [Rect(px(0.5 + 0.45 * i), px(4.6), px(0.5 + 0.45 * i + 0.4), px(5.1)) for i in range(11)]

PAPER_PROBE = Rect(px(0.5), px(5.5), px(5.5), px(7.5))  # blank paper, for the white reference
