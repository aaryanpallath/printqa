"""printqa simulate-run | calibrate | inspect"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import report
from . import spec as specs
from .analyze import AlignmentError, analyze
from .simulate import Defects, print_and_scan


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _analyze_file(path: str):
    try:
        return path, analyze(_read(path)), None
    except AlignmentError as e:
        return path, None, str(e)


def press_run(page: int, pages: int) -> Defects:
    """A run that degrades: cyan registration drifts, a nozzle dies, magenta ink starves, a yellow streak appears."""
    f = page / max(pages - 1, 1)
    d = Defects(registration={"C": (1.2 * f, 0.6 * f)}, density={"M": 1 - 0.12 * f})
    if page >= int(0.6 * pages):
        d.dead_nozzles = [41]
    if page >= int(0.8 * pages):
        d.streaks = [{"sep": "Y", "x": 1080, "width": 3, "depth": 0.08}]  # inside the yellow patch
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="printqa", description="Automated print-quality inspection of scanned test charts")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate-run", help="write scans of a simulated press run (with ground truth)")
    s.add_argument("--pages", type=int, default=60)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("calibrate", help="build a spec from scans of known-good pages")
    k.add_argument("scans", nargs="+")
    k.add_argument("--out", required=True)
    i = sub.add_parser("inspect", help="measure scans, check them against a spec, write CSV, JSON and an HTML report")
    i.add_argument("scans", nargs="+")
    i.add_argument("--spec", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    if a.cmd == "simulate-run":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        truth = []
        for n in range(a.pages):
            d = press_run(n, a.pages)
            img = print_and_scan(d, rng)
            cv2.imwrite(str(out / f"page_{n:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            truth.append({"page": n, **d.__dict__})
        (out / "truth.json").write_text(json.dumps(truth, indent=1))
        print(f"wrote {a.pages} scans to {out}")
        return 0

    if a.cmd == "calibrate":
        results = [analyze(_read(f)) for f in a.scans]
        Path(a.out).write_text(yaml.safe_dump(specs.calibrate(results), sort_keys=False))
        print(f"wrote {a.out} from {len(results)} known-good pages")
        return 0

    spec = specs.load(a.spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.scans)
    with ProcessPoolExecutor(a.workers) as ex:
        analyzed = list(ex.map(_analyze_file, files))
    rows, raw = [], []
    for n, (path, r, err) in enumerate(analyzed):
        if r is None:
            rows.append(
                {
                    "page": n,
                    "file": Path(path).name,
                    "failures": [f"unreadable: {err}"],
                    "registration": {s: float("nan") for s in "CMY"},
                    "density_dev": {s: float("nan") for s in "CMYK"},
                    "dead_nozzles": 0,
                    "streak": 0,
                    "banding": 0,
                }
            )
            continue
        checks = specs.check(r, spec)
        raw.append({"file": Path(path).name, **r.to_dict(), "checks": [c.__dict__ for c in checks]})
        rows.append(
            {
                "page": n,
                "file": Path(path).name,
                "failures": [c.name for c in checks if not c.passed],
                "registration": {s: float(np.hypot(*r.registration_um[s])) for s in "CMY"},
                "density_dev": {s: r.density[s] - spec["density_targets"][s] for s in "CMYK"},
                "dead_nozzles": len(r.dead_nozzles),
                "streak": max((abs(x["contrast"]) for x in r.streaks), default=0.0),
                "banding": max(b["amplitude"] for b in r.banding.values()),
            }
        )
    (out / "results.json").write_text(json.dumps(raw, indent=1))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "file", "status", "failed_checks"])
        for r in rows:
            w.writerow([r["page"], r["file"], "FAIL" if r["failures"] else "pass", "; ".join(r["failures"])])
    report.write(rows, spec, out / "report.html")
    failed = sum(bool(r["failures"]) for r in rows)
    print(f"{len(rows)} pages, {failed} failed; report at {out / 'report.html'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
