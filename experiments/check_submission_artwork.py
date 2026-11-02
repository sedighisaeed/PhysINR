#!/usr/bin/env python3
"""Check every submission artwork against the publisher's artwork contract.

Requirements R12/R13 (paper/00_journal/requirements.md):

* accepted formats are TIFF, JPEG, EPS and PDF -- PNG is *not* accepted, so the
  repository PNGs are never shipped;
* 300 dpi halftone, 500 dpi combination art, 1000 dpi line art;
* widths must be 90 mm, 140 mm or 190 mm.

Every figure this project ships is a chart (axes, lines, fills, text), i.e.
combination art, so the 500 dpi / standard-width checks are the ones that
apply. The check fails loudly on any figure that would be rejected, rather
than silently passing a mixed bag.

Usage:  python3 experiments/check_submission_artwork.py [dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ACCEPTED_FORMATS = {".tiff", ".tif", ".jpg", ".jpeg", ".eps", ".pdf"}
STANDARD_WIDTHS_MM = (190.0, 140.0, 90.0)
COMBINATION_DPI = 500


def check_tiff(path: Path, failures: list) -> None:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        dpi = float(im.info.get("dpi", (0, 0))[0])
    if dpi < COMBINATION_DPI:
        failures.append(f"{path.name}: {dpi:.0f} dpi < {COMBINATION_DPI} required")
        return
    width_mm = w / dpi * 25.4
    if not any(abs(width_mm - t) < 1.0 for t in STANDARD_WIDTHS_MM):
        failures.append(
            f"{path.name}: {width_mm:.1f} mm wide, must be one of "
            f"{STANDARD_WIDTHS_MM} mm")


def check_pdf(path: Path, failures: list) -> None:
    from pypdf import PdfReader

    box = PdfReader(str(path)).pages[0].mediabox
    width_mm = float(box.width) / 72 * 25.4
    if not any(abs(width_mm - t) < 1.0 for t in STANDARD_WIDTHS_MM):
        failures.append(
            f"{path.name}: {width_mm:.1f} mm wide, must be one of "
            f"{STANDARD_WIDTHS_MM} mm")


def main() -> int:
    fig_dir = Path(sys.argv[1] if len(sys.argv) > 1
                   else REPO / "paper" / "06_submission" / "figures")
    if not fig_dir.is_dir():
        print(f"no such directory: {fig_dir}", file=sys.stderr)
        return 2

    failures: list = []
    found = 0
    for path in sorted(fig_dir.iterdir()):
        if path.suffix.lower() not in ACCEPTED_FORMATS:
            failures.append(f"{path.name}: '{path.suffix}' not in the publisher's "
                            f"accepted formats {sorted(ACCEPTED_FORMATS)}")
            continue
        found += 1
        if path.suffix.lower() in (".tiff", ".tif"):
            check_tiff(path, failures)
        elif path.suffix.lower() == ".pdf":
            check_pdf(path, failures)

    if not found:
        print(f"no artwork files found in {fig_dir}", file=sys.stderr)
        return 2

    print(f"checked {found} artwork file(s) in {fig_dir}")
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all artwork passes the width and resolution contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
