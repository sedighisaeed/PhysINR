"""Qualitative comparison figures: side-by-side panels with zoomed patches.

Produces the standard paper figure -- one row per sample, columns for
Raw / GT / Method A / Method B ..., with a zoomed detail patch and optional
bounding-box overlay so a reader can verify the metric story visually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ZOOM = 2.0
DEFAULT_PATCH_FRACTION = 0.28


@dataclass
class PanelSpec:
    """Where and how to draw one comparison panel."""

    sample_id: str
    raw: Optional[np.ndarray] = None
    gt: Optional[np.ndarray] = None
    methods: Dict[str, np.ndarray] = field(default_factory=dict)
    # Region to magnify, as (y0, y1, x0, x1) in pixels. Auto-chosen if None.
    patch: Optional[Tuple[int, int, int, int]] = None
    # Optional detection boxes per image: {(method_or_"raw"): [(label, xyxy)]}
    boxes: Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]] = field(default_factory=dict)


@dataclass
class ComparisonFigure:
    """Renders a grid of qualitative comparisons."""

    specs: List[PanelSpec]
    column_order: Optional[List[str]] = None      # method display order
    zoom: float = DEFAULT_ZOOM
    patch_fraction: float = DEFAULT_PATCH_FRACTION
    dpi: int = 150
    show_zoom: bool = True
    font_size: int = 9
    # Square-ish patch so the magnified inset stays readable.
    box_color: Tuple[int, int, int] = (255, 80, 80)
    zoom_position: str = "bottom_right"          # corner of the inset overlay

    # ------------------------------------------------------------------ io
    def render(self, out_path: Path, title: str = "") -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        out_path = Path(out_path)
        columns = self._columns()
        n_rows, n_cols = len(self.specs), len(columns)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 1.9, n_rows * 1.9),
                                 dpi=self.dpi)
        axes = np.atleast_2d(axes)
        plt.rcParams.update({"font.size": self.font_size})

        for r, spec in enumerate(self.specs):
            images = self._row_images(spec, columns)
            for c, (name, image) in enumerate(images):
                ax = axes[r, c]
                if image is None:
                    ax.axis("off")
                    continue
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(name, fontsize=self.font_size)
                # Detection overlay for this column, if provided.
                key = "raw" if name == "Raw" else name
                if key in spec.boxes:
                    self._draw_boxes(ax, image.shape, spec.boxes[key])
                # Magnified inset drawn over the first image of the row.
                if self.show_zoom and c == 0:
                    patch = spec.patch or self._auto_patch(image)
                    self._draw_zoom_inset(ax, image, patch, Rectangle)
        if title:
            fig.suptitle(title, fontsize=self.font_size + 1)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------- helpers
    def _columns(self) -> List[str]:
        if self.column_order:
            return list(self.column_order)
        columns: List[str] = []
        for spec in self.specs:
            if spec.raw is not None and "Raw" not in columns:
                columns.append("Raw")
            if spec.gt is not None and "GT" not in columns:
                columns.append("GT")
            for name in spec.methods:
                if name not in columns:
                    columns.append(name)
        return columns

    def _row_images(self, spec: PanelSpec, columns: List[str]
                    ) -> List[Tuple[str, Optional[np.ndarray]]]:
        out = []
        for name in columns:
            if name == "Raw":
                out.append((name, spec.raw))
            elif name == "GT":
                out.append((name, spec.gt))
            else:
                out.append((name, spec.methods.get(name)))
        return out

    def _auto_patch(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = image.shape[:2]
        size_h = int(h * self.patch_fraction)
        size_w = int(w * self.patch_fraction)
        # Centre by default; callers override for a specific detail.
        y0 = (h - size_h) // 2
        x0 = (w - size_w) // 2
        return (y0, y0 + size_h, x0, x0 + size_w)

    def _draw_zoom_inset(self, ax, image, patch, Rectangle) -> None:
        y0, y1, x0, x1 = patch
        if not (0 <= y0 < y1 <= image.shape[0] and 0 <= x0 < x1 <= image.shape[1]):
            return
        crop = image[y0:y1, x0:x1]
        h, w = crop.shape[:2]
        # Outline the patch on the base image.
        ax.add_patch(Rectangle((x0, y0), w, h, linewidth=1.2,
                               edgecolor="#ff5050", facecolor="none"))
        # Inset placement.
        if self.zoom_position == "bottom_right":
            corner = (image.shape[1] - w * self.zoom, 0)
        else:
            corner = (0, 0)
        inset = ax.inset_axes([corner[0] / image.shape[1],
                               corner[1] / image.shape[0],
                               (w * self.zoom) / image.shape[1],
                               (h * self.zoom) / image.shape[0]])
        inset.imshow(crop, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor("#ff5050")
            spine.set_linewidth(1.2)

    def _draw_boxes(self, ax, shape, boxes) -> None:
        for _label, (x0, y0, x1, y1) in boxes:
            ax.add_patch(_Rectangle_xyxy(x0, y0, x1, y1, shape))


def _Rectangle_xyxy(x0, y0, x1, y1, shape):
    from matplotlib.patches import Rectangle

    return Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0,
                     edgecolor="#39e600", facecolor="none")


# --------------------------------------------------------------- builders
def build_panel(raw: np.ndarray, gt: Optional[np.ndarray],
                methods: Dict[str, np.ndarray], sample_id: str = "",
                patch: Optional[Tuple[int, int, int, int]] = None) -> PanelSpec:
    return PanelSpec(sample_id=sample_id, raw=raw, gt=gt,
                     methods=dict(methods), patch=patch)


def error_heatmap(reference: np.ndarray, distorted: np.ndarray,
                  out_path: Path, cmap: str = "inferno",
                  dpi: int = 150) -> Path:
    """An SSIM-error heatmap, useful as a per-sample failure diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..metrics.full_reference import ssim

    _, ssim_map = ssim(reference, distorted, return_map=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    im = ax.imshow(1.0 - ssim_map, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("1 - SSIM map")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
