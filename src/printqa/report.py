"""An HTML run report: status per page, trend charts against the spec, the first page each check failed, and the
worst page's measurements. Charts are Matplotlib PNGs embedded in the page, so the report is a single file."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"C": "#00a0c6", "M": "#d4148c", "Y": "#c9b200", "K": "#222222"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def trends(rows: list[dict], spec: dict) -> str:
    pages = [r["page"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for sep in ("C", "M", "Y"):
        ax[0, 0].plot(pages, [r["registration"][sep] for r in rows], color=COLORS[sep], label=sep)
    ax[0, 0].axhline(spec["max_registration_um"], ls="--", color="red", lw=1)
    ax[0, 0].set(title="Registration error vs black (µm)", xlabel="page")
    for sep in ("C", "M", "Y", "K"):
        ax[0, 1].plot(pages, [r["density_dev"][sep] for r in rows], color=COLORS[sep], label=sep)
    for s in (-1, 1):
        ax[0, 1].axhline(s * spec["density_tolerance"], ls="--", color="red", lw=1)
    ax[0, 1].set(title="Solid density deviation from target", xlabel="page")
    ax[1, 0].step(pages, [r["dead_nozzles"] for r in rows], where="post", color="#333")
    ax[1, 0].set(title="Dead nozzles", xlabel="page")
    ax[1, 1].plot(pages, [r["streak"] for r in rows], label="worst streak contrast", color="#7b3fa0")
    ax[1, 1].plot(pages, [r["banding"] for r in rows], label="banding amplitude", color="#e07b00")
    ax[1, 1].axhline(spec["max_streak_contrast"], ls="--", color="#7b3fa0", lw=1)
    ax[1, 1].axhline(spec["max_banding_amplitude"], ls="--", color="#e07b00", lw=1)
    ax[1, 1].set(title="Streaks and banding (optical density)", xlabel="page")
    for a in ax.flat:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.tight_layout()
    return _png(fig)


def build(rows: list[dict], spec: dict, title: str = "Print quality run") -> str:
    failed = [r for r in rows if r["failures"]]
    first: dict[str, int] = {}
    for r in rows:
        for f in r["failures"]:
            first.setdefault(f, r["page"])
    table = "".join(
        f"<tr class='{'fail' if r['failures'] else ''}'><td>{r['page']}</td><td>{html.escape(r['file'])}</td>"
        f"<td>{'FAIL' if r['failures'] else 'pass'}</td><td>{html.escape(', '.join(r['failures']))}</td></tr>"
        for r in rows
    )
    firsts = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(first.items(), key=lambda x: x[1])
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#fafafa;color:#1b1b1b}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid #e5e5e5;padding:4px 8px;text-align:left}} tr.fail td{{background:#fff1f0}}
.kpi{{display:inline-block;margin:0 24px 12px 0}} .kpi b{{font-size:24px;display:block}} img{{max-width:100%}}
.scroll{{max-height:420px;overflow:auto;border:1px solid #e5e5e5}}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div class="kpi">Pages<b>{len(rows)}</b></div><div class="kpi">Passed<b>{len(rows) - len(failed)}</b></div>
<div class="kpi">Failed<b>{len(failed)}</b></div>
<h2>Trends against the spec</h2><img alt="trend charts" src="data:image/png;base64,{trends(rows, spec)}">
<h2>First page each check failed</h2><table><tr><th>Check</th><th>Page</th></tr>{firsts or "<tr><td colspan=2>none</td></tr>"}</table>
<h2>Pages</h2><div class="scroll"><table><tr><th>Page</th><th>File</th><th>Status</th><th>Failed checks</th></tr>{table}</table></div>
</main></body></html>"""


def write(rows: list[dict], spec: dict, out: str | Path, title: str = "Print quality run") -> Path:
    path = Path(out)
    path.write_text(build(rows, spec, title))
    return path
