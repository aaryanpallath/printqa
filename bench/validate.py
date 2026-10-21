"""How good is the inspector? Thousands of simulated scans with known defects, measured against the truth.

  uv run python bench/validate.py        # writes results/validation.json (about 2 minutes on 8 cores)

false alarms   clean pages: how often does each check fail when nothing is wrong?
streaks        detection rate by streak depth, and position error
banding        detection rate by amplitude, and period error
nozzles        exact identification of 1-4 dead nozzles among 96
registration   measured offset vs true offset (micrometers), and pass/fail agreement at the 60 um limit
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from printqa import chart as c
from printqa import spec as specs
from printqa.analyze import UM_PER_PX, analyze
from printqa.simulate import Defects, print_and_scan

ROOT = Path(__file__).resolve().parent.parent
MM = 25.4 / c.DPI


def case(args: tuple) -> dict:
    kind, level, seed = args
    rng = np.random.default_rng(seed)
    d, truth = Defects(), {}
    if kind == "streak":
        sep = str(rng.choice(c.SEPARATIONS))
        r = c.PATCHES[sep].inset(45)
        x, width = int(rng.integers(r.x0, r.x1 - 8)), int(rng.integers(1, 7))
        d.streaks = [{"sep": sep, "x": x, "width": width, "depth": level}]
        truth = {"sep": sep, "x_mm": (x + width / 2) * MM}
    elif kind == "banding":
        sep = str(rng.choice(c.SEPARATIONS))
        period = float(rng.uniform(15, 250))
        d.banding = [{"sep": sep, "period": period, "amplitude": level}]
        truth = {"sep": sep, "period_mm": period * MM}
    elif kind == "nozzles":
        d.dead_nozzles = sorted(rng.choice(c.NOZZLES, int(level), replace=False).tolist())
        truth = {"dead": d.dead_nozzles}
    elif kind == "registration":
        d.registration = {s: tuple(rng.uniform(-1.2, 1.2, 2)) for s in "CMY"}
        truth = {s: [v * UM_PER_PX for v in d.registration[s]] for s in "CMY"}
    t0 = time.perf_counter()
    img = print_and_scan(d, rng)
    t1 = time.perf_counter()
    r = analyze(img)
    return {
        "kind": kind,
        "level": level,
        "truth": truth,
        "result": r.to_dict(),
        "analyze_s": time.perf_counter() - t1,
        "simulate_s": t1 - t0,
    }


def main() -> None:
    jobs = [("clean", 0, 1000 + i) for i in range(120)]
    jobs += [
        ("streak", depth, 2000 + 100 * j + i)
        for j, depth in enumerate([0.005, 0.01, 0.02, 0.04, 0.08])
        for i in range(30)
    ]
    jobs += [("banding", amp, 3000 + 100 * j + i) for j, amp in enumerate([0.005, 0.01, 0.02, 0.04]) for i in range(30)]
    jobs += [("nozzles", k, 4000 + 100 * k + i) for k in (1, 2, 3, 4) for i in range(15)]
    jobs += [("registration", 0, 5000 + i) for i in range(80)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(case, jobs, chunksize=4))
    wall = time.perf_counter() - t0

    clean = [r for r in rows if r["kind"] == "clean"]
    spec = specs.calibrate([type("R", (), {"density": r["result"]["density"]}) for r in clean[:10]])
    out: dict = {
        "pages": len(rows),
        "wall_seconds": round(wall, 1),
        "analyze_seconds_per_page": round(float(np.median([r["analyze_s"] for r in rows])), 3),
    }

    def checks(r):
        from printqa.analyze import PageResult

        return specs.check(PageResult(**r["result"]), spec)

    fa = defaultdict(int)
    for r in clean[10:]:  # the first 10 calibrate the density targets
        for ch in checks(r):
            if not ch.passed:
                fa[ch.name.split(" ")[0]] += 1
    out["false_alarms"] = {
        "clean_pages": len(clean) - 10,
        "failed_checks": dict(fa),
        "pages_failing_any": sum(any(not ch.passed for ch in checks(r)) for r in clean[10:]),
    }

    st = defaultdict(lambda: {"n": 0, "reported": 0, "fails_spec": 0, "pos_err_mm": [], "contrast": []})
    for r in rows:
        if r["kind"] != "streak":
            continue
        s, t = st[r["level"]], r["truth"]
        s["n"] += 1
        hits = [x for x in r["result"]["streaks"] if x["sep"] == t["sep"] and abs(x["x_mm"] - t["x_mm"]) < 1.5]
        if hits:
            s["reported"] += 1
            s["pos_err_mm"].append(abs(hits[0]["x_mm"] - t["x_mm"]))
            s["contrast"].append(abs(hits[0]["contrast"]))
            s["fails_spec"] += abs(hits[0]["contrast"]) > spec["max_streak_contrast"]
    out["streaks"] = {
        str(k): {
            "pages": v["n"],
            "reported": v["reported"],
            "fails_spec": v["fails_spec"],
            "median_density_contrast": round(float(np.median(v["contrast"])), 3) if v["contrast"] else None,
            "median_position_error_mm": round(float(np.median(v["pos_err_mm"])), 2) if v["pos_err_mm"] else None,
        }
        for k, v in sorted(st.items())
    }

    bd = defaultdict(lambda: {"n": 0, "flagged": 0, "period_err": []})
    for r in rows:
        if r["kind"] != "banding":
            continue
        b, t = bd[r["level"]], r["truth"]
        b["n"] += 1
        m = r["result"]["banding"][t["sep"]]
        if m["amplitude"] > spec["max_banding_amplitude"]:
            b["flagged"] += 1
            b["period_err"].append(abs(m["period_mm"] - t["period_mm"]) / t["period_mm"])
    out["banding"] = {
        str(k): {
            "pages": v["n"],
            "flagged": v["flagged"],
            "median_period_error_pct": round(100 * float(np.median(v["period_err"])), 1) if v["period_err"] else None,
        }
        for k, v in sorted(bd.items())
    }

    nz = [r for r in rows if r["kind"] == "nozzles"]
    exact = sum(r["result"]["dead_nozzles"] == r["truth"]["dead"] for r in nz)
    tp = sum(len(set(r["result"]["dead_nozzles"]) & set(r["truth"]["dead"])) for r in nz)
    fp = sum(len(set(r["result"]["dead_nozzles"]) - set(r["truth"]["dead"])) for r in nz)
    fn = sum(len(set(r["truth"]["dead"]) - set(r["result"]["dead_nozzles"])) for r in nz)
    out["nozzles"] = {
        "pages": len(nz),
        "exact_pages": exact,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }

    errs, agree, total, disagree_near = [], 0, 0, []
    for r in rows:
        if r["kind"] != "registration":
            continue
        for s in "CMY":
            m, t = np.array(r["result"]["registration_um"][s]), np.array(r["truth"][s])
            errs.append(m - t)
            tm, mm = float(np.hypot(*t)), float(np.hypot(*m))
            total += 1
            if (tm > spec["max_registration_um"]) == (mm > spec["max_registration_um"]):
                agree += 1
            else:
                disagree_near.append(round(abs(tm - spec["max_registration_um"]), 1))
    e = np.array(errs)
    out["registration"] = {
        "measurements": total,
        "rmse_um": [round(float(v), 2) for v in np.sqrt((e**2).mean(axis=0))],
        "max_abs_error_um": round(float(np.abs(e).max()), 2),
        "um_per_scan_pixel": round(UM_PER_PX, 1),
        "pass_fail_agreement": f"{agree}/{total}",
        "disagreements_distance_from_limit_um": disagree_near,
    }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "validation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
