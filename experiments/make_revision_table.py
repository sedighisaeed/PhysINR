#!/usr/bin/env python3
"""Render paper/04_tables/revision_v4.md (Table 6) from the v4 revision eval.

Reads experiments/v4_revision_results.json (produced by
experiments/v4_revision_eval.py) and emits the prior-design study table the
results section prints. Every cell comes from the JSON; nothing is typed by
hand, so the table cannot drift from the run it reports.

Rows are the four candidate transmission priors plus the raw input, on the
same 20 UIEB and 12 UCCS frames, 480 px, 12 adaptation steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "experiments" / "v4_revision_results.json"
OUT = REPO / "paper" / "04_tables" / "revision_v4.md"

# Column order in the table, with the header for each.
SIDES = [
    ("input", "raw input"),
    ("published", "A: water-type lookup"),
    ("typehead", "intermediate: content head"),
    ("v2", "B: measured only"),
    ("v3", "C: + level, uncalibrated"),
    ("v4", "D: + level, calibrated"),
]
FIELDS = [("clip_all", "clip (all)"), ("clip_G", "clip (G)"), ("cast", "cast"),
          ("psnr", "PSNR"), ("ssim", "SSIM"), ("uciqe", "UCIQE")]


def _mean(rows, side, field):
    vals = [r[side][field] for r in rows if r[side].get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v, nd=3):
    if v is None:
        return "---"
    return f"{v:.{nd}f}"


def main() -> int:
    if not SRC.is_file():
        print(f"missing {SRC}; run experiments/v4_revision_eval.py first",
              file=sys.stderr)
        return 1
    blob = json.loads(SRC.read_text(encoding="utf-8"))
    records = blob["records"]

    header = "| Variant | " + " | ".join(f for _, f in FIELDS) + " |"
    rule = "| --- |" + "---|" * len(FIELDS)
    lines = [
        "<!-- Prior design study: four candidate transmission priors on the",
        "     same 20 UIEB test frames and 12 UCCS colour-cast frames, 480 px",
        "     max side, 12 adaptation steps, one protocol for every column.",
        "     Clipping is the fraction of pixels within 1/255 of either rail,",
        "     averaged over the three channels; clip (G) is the green channel",
        "     alone, the one the water-type-lookup prior saturates. Cast is",
        "     the CIELAB chroma magnitude. The published checkpoint runs under",
        "     its stored configuration (legacy); the intermediate row is the",
        "     content-dependent prior that preceded the measured-level fix;",
        "     B/C/D are the three priors of results §4.5. The adaptation loop",
        "     is chaotic at the metric margin (±0.02 clipping, ±0.35 dB PSNR",
        "     across repeat runs), so only larger differences are",
        "     interpretable. -->",
        "",
        header,
        rule,
    ]

    for setname, label in (("uieb", f"**UIEB, n = {0}**"),
                           ("uccs", f"**UCCS, n = {0}**")):
        rows = [r for r in records if r.get("set") == setname]
        if not rows:
            continue
        lines.append(f"| **{setname.upper()}, n = {len(rows)}** |"
                     + " |" * len(FIELDS))
        for key, name in SIDES:
            cells = []
            for field, _ in FIELDS:
                v = _mean(rows, key, field)
                nd = 4 if field == "ssim" else 3
                cells.append(_fmt(v, nd))
            bold = key in ("input", "v4")
            row = f"| {'**' + name + '**' if bold else name} | " \
                  + " | ".join(cells) + " |"
            lines.append(row)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
