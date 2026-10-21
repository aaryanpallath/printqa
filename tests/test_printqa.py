import json

import cv2
import numpy as np
import pytest

from printqa import chart as c
from printqa import spec as specs
from printqa.analyze import UM_PER_PX, AlignmentError, analyze, find_banding, find_fiducials, find_streaks
from printqa.cli import main
from printqa.simulate import Defects, ScanParams, print_and_scan, render, scan


@pytest.fixture(scope="module")
def clean():
    return analyze(print_and_scan(Defects(), np.random.default_rng(7)))


def test_alignment_recovers_rotation_scale_and_shift():
    rng = np.random.default_rng(1)
    params = ScanParams(rotation_deg=0.5, scale=1.004, shift=(12.0, -9.0))
    img = scan(render(Defects(), rng), params, rng).astype(np.float32) / 255
    pts = find_fiducials(img)
    m = cv2.getRotationMatrix2D((c.W / 2, c.H / 2), params.rotation_deg, params.scale)
    m[:, 2] += (params.margin + params.shift[0], params.margin + params.shift[1])
    expected = cv2.transform(np.array([c.FIDUCIALS], np.float32), m)[0]
    assert np.abs(pts - expected).max() < 0.3  # sub-pixel


def test_alignment_fails_loudly_on_a_blank_scan():
    with pytest.raises(AlignmentError):
        analyze(np.full((500, 400, 3), 240, np.uint8))


def test_clean_page_passes_every_check(clean):
    spec = specs.calibrate([clean])
    assert all(ch.passed for ch in specs.check(clean, spec))
    assert clean.streaks == [] and clean.dead_nozzles == []


def test_finds_a_streak_where_it_is():
    d = Defects(streaks=[{"sep": "M", "x": 700, "width": 3, "depth": 0.08}])
    r = analyze(print_and_scan(d, np.random.default_rng(2)))
    assert len(r.streaks) == 1 and r.streaks[0]["sep"] == "M"
    assert abs(r.streaks[0]["x_mm"] - 701.5 * 25.4 / c.DPI) < 0.5


def test_streak_detector_on_a_synthetic_profile():
    region = np.full((50, 300), 0.8) + np.random.default_rng(0).normal(0, 0.002, (50, 300))
    region[:, 140:143] -= 0.05
    found = find_streaks(region, min_contrast=0.02)
    assert [(s["x"], s["width"]) for s in found] == [(140, 3)]


def test_banding_period_and_amplitude():
    y = np.arange(600)[:, None]
    region = 0.8 + 0.03 * np.sin(2 * np.pi * y / 50) + np.zeros((600, 100))
    b = find_banding(region)
    assert abs(b["period_px"] - 50) < 2 and abs(b["amplitude"] - 0.03) < 0.004


def test_identifies_dead_nozzles_exactly():
    r = analyze(print_and_scan(Defects(dead_nozzles=[0, 37, 95]), np.random.default_rng(3)))
    assert r.dead_nozzles == [0, 37, 95]


def test_measures_registration_to_a_few_micrometers():
    truth = {"C": (1.5, -0.7), "M": (-0.4, 1.1), "Y": (0.0, 0.3)}
    r = analyze(print_and_scan(Defects(registration=truth), np.random.default_rng(4)))
    for sep, (dx, dy) in truth.items():
        assert np.allclose(r.registration_um[sep], (dx * UM_PER_PX, dy * UM_PER_PX), atol=5)


def test_spec_flags_each_problem(clean):
    spec = specs.calibrate([clean])
    d = Defects(density={"Y": 0.8}, registration={"C": (1.2, 0)}, dead_nozzles=[5])
    failed = {
        ch.name for ch in specs.check(analyze(print_and_scan(d, np.random.default_rng(5))), spec) if not ch.passed
    }
    assert {"density Y", "registration C", "dead nozzles"} <= failed


def test_spec_file_must_be_complete(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("density_targets: {C: 0.8}\n")
    with pytest.raises(ValueError, match="missing"):
        specs.load(p)


def test_cli_end_to_end(tmp_path):
    run = tmp_path / "run"
    assert main(["simulate-run", "--pages", "5", "--out", str(run)]) == 0
    assert (
        main(["calibrate", *[str(run / f"page_000{i}.png") for i in range(2)], "--out", str(tmp_path / "s.yaml")]) == 0
    )
    code = main(
        [
            "inspect",
            *sorted(str(p) for p in run.glob("page_*.png")),
            "--spec",
            str(tmp_path / "s.yaml"),
            "--out",
            str(tmp_path / "rep"),
            "--workers",
            "2",
        ]
    )
    assert code == 1  # the simulated run degrades, so later pages fail
    assert (tmp_path / "rep" / "report.html").read_text().startswith("<!doctype html>")
    rows = json.loads((tmp_path / "rep" / "results.json").read_text())
    assert len(rows) == 5 and 41 in rows[-1]["dead_nozzles"]
