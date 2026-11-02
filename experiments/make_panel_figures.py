#!/usr/bin/env python3
"""Build the qualitative comparison figure and the method diagram.

Both figures are composed from real artifacts of the study:

* The comparison panel is a grid of the actual enhanced images the benchmark
  produced -- inputs and references from the UIEB set, and each method's real
  output PNG for one test image. Nothing is synthesised or retouched.
* The method diagram is drawn from the model's own code: the field structure,
  the Jerlov prior, the forward model and the reference-free losses are the
  ones the checkpoint actually uses. It is drawn with matplotlib so it stays
  a vector figure at submission resolution.

Usage:  python3 experiments/make_panel_figures.py [--submission-dir DIR]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RES = REPO / "hydro_bench_results" / "UIEB__test" / "images"
PHYS = REPO / "experiments" / "physinr_results" / "UIEB__test" / "images" / "PhysINR-Zero"
RAW = REPO / "datasets" / "UIEB" / "raw_input_imgs"
REF = REPO / "datasets" / "UIEB" / "reference_imgs"

TEST_ID = "100_img_"

# Order the panel by leaderboard position on UIEB PSNR, so a reader scanning
# left-to-right and top-to-bottom sees methods best-first.
ORDER = [
    ("Input", None),
    ("Reference", None),
    ("UIEC2-Net", None),
    ("Five-A+", None),
    ("Water-Net", None),
    ("WaterMamba", None),
    ("PhysINR-Zero", None),
    ("Retinex", None),
    ("UDCP", None),
]


def panel_paths() -> list[tuple[str, Path]]:
    out = []
    for name, _ in ORDER:
        if name == "Input":
            p = RAW / f"{TEST_ID}.png"
        elif name == "Reference":
            p = REF / f"{TEST_ID}.png"
        elif name == "PhysINR-Zero":
            p = PHYS / f"{TEST_ID}.png"
        else:
            p = RES / name / f"{TEST_ID}.png"
        if not p.is_file():
            raise FileNotFoundError(f"panel image missing: {p}")
        out.append((name, p))
    return out


def make_comparison(submission_dir: Path | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    cells = panel_paths()
    cols = 3
    rows = (len(cells) + cols - 1) // cols
    # 190 mm at 500 dpi is the target; fix the figure size so the saved PDF
    # lands on the standard submission width rather than near it.
    fig, axes = plt.subplots(rows, cols, figsize=(7.48, 2.35 * rows),
                             dpi=200)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (name, p) in zip(axes.ravel(), cells):
        ax.imshow(imread(str(p)))
        ax.set_title(name, fontsize=8, pad=2)
    # any trailing empty cells
    for ax in axes.ravel()[len(cells):]:
        ax.set_visible(False)
    fig.tight_layout(pad=0.4)

    out = Path("paper/03_figures/qualitative_UIEB.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if submission_dir:
        submission_dir.mkdir(parents=True, exist_ok=True)
        pdf_out = submission_dir / "qualitative_UIEB.pdf"
        fig.savefig(pdf_out, bbox_inches="tight")
        # 500 dpi TIFF at 190 mm, matching the other submission artwork
        tiff_out = submission_dir / "qualitative_UIEB.tiff"
        fig.savefig(tiff_out, dpi=1000, bbox_inches="tight")
        _to_500dpi(tiff_out, target_mm=190.0)
        # The PDF is a vector page; snap its box to exactly 190 mm so the
        # artwork contract check sees a standard width.
        _snap_pdf_width(pdf_out, target_mm=190.0)
        print(f"submission artwork: {pdf_out}, {tiff_out}")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def _snap_pdf_width(path: Path, target_mm: float) -> None:
    """Scale a vector PDF page so its width is exactly ``target_mm``.

    bbox_inches="tight" leaves the page a hair off a standard width; scaling
    the page box is lossless for vector content.
    """
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    target_pt = target_mm / 25.4 * 72.0
    r = PdfReader(str(path))
    w = PdfWriter()
    for page in r.pages:
        cur = float(page.mediabox.width)
        scale = target_pt / cur
        page.scale_by(scale)
        page.mediabox = RectangleObject(
            (0, 0, target_pt, float(page.mediabox.height) * scale))
        w.add_page(page)
    with open(path, "wb") as fh:
        w.write(fh)


def _to_500dpi(path: Path, target_mm: float) -> None:
    """Resample a 1000-dpi raster down to the exact 500-dpi pixel count.

    matplotlib's dpi scales pixels and metadata together, leaving the physical
    width unchanged; downsampling after the fact is what actually sets the
    width to the target.
    """
    from PIL import Image

    target_px = int(round(target_mm / 25.4 * 500))
    im = Image.open(path)
    im = im.resize((target_px, int(round(im.height * target_px / im.width))),
                   Image.LANCZOS)
    im.save(path, dpi=(500, 500), compression="tiff_deflate")


# --------------------------------------------------------------- method diagram
def make_diagram(submission_dir: Path | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, label, fc="#eef3f8", ec="#33506b", fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color="#12222f",
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2, label=None, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=9,
                                     lw=1.0, color="#33506b",
                                     linestyle="--" if dashed else "-"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.9, label, ha="center",
                    va="bottom", fontsize=6.2, color="#33506b", style="italic")

    # --- input
    box(1.5, 24, 14, 14, "Input image\n$I(x)$\nsRGB, any size", fc="#f4f1e8",
        ec="#8a6d3b", bold=True)

    # --- coordinate field (the proposed representation)
    box(24, 20, 24, 22,
        "Coordinate field $\\Phi(x)$\n"
        "$\\tau_c$: optical thickness\n"
        "$\\alpha_c$: nonlinear gain\n"
        "$A_c$: ambient light",
        fc="#e8f0ea", ec="#2e6b46", bold=True)
    ax.text(36, 47.5, "9,813 parameters", ha="center", fontsize=6.4,
            color="#2e6b46", style="italic")

    # --- physics prior
    box(24, 1.5, 24, 12,
        "Jerlov water-type prior\n"
        "10 types, $\\beta_c \\cdot d$\n"
        "deviation $\\leq \\delta_{\\max}$",
        fc="#eef0f6", ec="#4a4a7a")
    arrow(36, 13.5, 36, 20, "anchors $\\tau_c$")

    # --- forward model
    box(55, 24, 20, 14,
        "Forward model\n"
        "$I'_c = J_c t_c + A_c (1 - t_c)$\n"
        "$J_c = g(v;\\alpha_c)$",
        fc="#f6ecec", ec="#7a3a3a")
    arrow(48, 31, 55, 31)

    # --- losses (reference-free)
    box(55, 3.5, 20, 13,
        "Reference-free losses\n"
        "colour balance, contrast,\n"
        "cycle consistency, TV",
        fc="#f6ecec", ec="#7a3a3a")
    arrow(65, 24, 65, 16.5, label="$\\nabla$")

    # --- output
    box(83, 24, 15.5, 14, "Enhanced\nimage $\\hat{J}(x)$", fc="#f4f1e8",
        ec="#8a6d3b", bold=True)
    arrow(75, 31, 83, 31)

    # --- meta-training provenance
    box(55, 43.5, 28, 15,
        "Meta-training (FOMAML)\n"
        "4,000 tasks from 882 terrestrial\n"
        "photographs, synthetic degradation",
        fc="#f0f0f0", ec="#555555")
    arrow(69, 43.5, 45, 42, dashed=True)
    ax.text(58, 44.5, "initialisation only", ha="center", fontsize=6.0,
            color="#555555", style="italic")

    # --- test-time adaptation
    ax.annotate("", xy=(36, 20), xytext=(36, 13.5),
                arrowprops=dict(arrowstyle="-|>", color="#2e6b46", lw=1.0))
    ax.text(2, 47,
            "Test-time adaptation: the field is fit per image against the\n"
            "reference-free losses; no underwater image is seen in training.",
            fontsize=6.6, color="#12222f", va="top", linespacing=1.3)

    fig.tight_layout(pad=0.3)
    out = Path("paper/03_figures/method_diagram.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if submission_dir:
        pdf_out = submission_dir / "method_diagram.pdf"
        fig.savefig(pdf_out, bbox_inches="tight")
        tiff_out = submission_dir / "method_diagram.tiff"
        fig.savefig(tiff_out, dpi=1000, bbox_inches="tight")
        _to_500dpi(tiff_out, target_mm=190.0)
        print(f"submission artwork: {pdf_out}, {tiff_out}")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", default="paper/06_submission/figures")
    args = ap.parse_args()
    sd = Path(args.submission_dir) if args.submission_dir else None
    make_comparison(sd)
    make_diagram(sd)
