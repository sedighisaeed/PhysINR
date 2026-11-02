#!/usr/bin/env python3
"""Render the manuscript to a single review PDF from the markdown sources.

The repository has no LaTeX, no pandoc and no network access, so this uses the
tools that *are* present: matplotlib's vector PDF backend for text and tables,
and pypdf to splice in the submission-quality vector figure pages that
``make_figures.py`` already produced.

What it is for: letting a person read the whole paper as one document. It is a
faithful rendering of the markdown sources -- the same files are the source of
record, and the submission itself is the elsarticle conversion described in
``paper/06_submission/submission_checklist.md``.

Layout: A4, 72 pt margins, DejaVu Serif body with DejaVu Sans headings, page
numbers, hanging-indent references, ruled tables, and one page per figure with
a caption set above it.

Usage:  python3 experiments/build_pdf.py [output.pdf]
"""

from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
MS = REPO / "paper" / "02_manuscript"
TAB = REPO / "paper" / "04_tables"
FIG = REPO / "paper" / "06_submission" / "figures"

# ---------------------------------------------------------------- page geometry
PAGE_W, PAGE_H = 595.27, 841.89      # A4, points
MARGIN = 72.0
TEXT_W = PAGE_W - 2 * MARGIN
TOP = PAGE_H - MARGIN
BOTTOM = MARGIN
PAGE_NUM_Y = 40.0

BODY, BODY_LEAD = 9.5, 1.34
TABLE_FS = 6.9
CODE_FS = 8.5
CAPTION_FS = 8.5
H1_FS, H2_FS, TITLE_FS = 12.5, 10.5, 17.0

SERIF, SANS = "DejaVu Serif", "DejaVu Sans"

# Order in which the assembled document is read.
SECTIONS = [
    ("00_front_matter.md", None),
    ("01_introduction.md", None),
    ("02_related_work.md", None),
    ("03_methods.md", None),
    ("results.md", None),
    ("05_discussion.md", None),
    ("07_declarations.md", None),
    ("08_references.md", None),
]

HIGHLIGHTS_FILE = REPO / "paper" / "06_submission" / "highlights.txt"

# Back-matter tables, with the captions the manuscript refers to them by.
TABLES = [
    ("comparison_UIEB.md", "Table 1",
     "UIEB main comparison, n = 134 images per method. Eleven baselines and "
     "PhysINR-Zero, full-reference and no-reference metrics (mean ± sd). "
     "Superscripts rank each column: * best, + second. WF-Diff ran an analytic "
     "fallback, not its published model; its row is marked and indicative only."),
    ("comparison_LSUI.md", "Table 2",
     "LSUI main comparison, n = 120 images per method. Conventions as Table 1."),
    ("comparison_EUVP.md", "Table 3",
     "EUVP main comparison, n = 120 images per method. Conventions as Table 1."),
    ("comparison_UCCS.md", "Table 4",
     "UCCS comparison, n = 45 images per method, no-reference metrics only: "
     "the set ships no reference images."),
    ("comparison_RUIE.md", "Table 5",
     "RUIE comparison, n = 100 images per method, no-reference metrics only."),
    ("ablation_UIEB.md", "Table 6",
     "UIEB ablation, n = 40 images per method (the first 40 of the split, the "
     "same subset every variant was scored on). The full model's column is "
     "truncated to that subset so the paired tests see matched samples; its "
     "mean therefore differs from Table 1."),
    ("ablation_UIEB_significance.tex", "Table 7",
     "UIEB ablation significance: paired Wilcoxon signed-rank against the full "
     "model, Holm-corrected across the 15 variants. ✓ marks p < 0.05; r is the "
     "rank-biserial correlation."),
]

# Back-matter figures. Each entry is (caption, [files]) and renders one page
# per file, the caption set once above the first file of the group.
FIGURES = [
    ("Fig. 1. Meta-training convergence: the outer-loop meta loss across the "
     "4,000 synthetic tasks drawn from terrestrial photographs. The curve is "
     "read from the checkpoint's own provenance file, not re-fit for display.",
     ["meta_train_curve.pdf"]),
    ("Fig. 2. Radar comparison across metrics on the five benchmark sets, one "
     "polygon per method. Metrics are min-max normalised across methods with "
     "direction corrected so outward is better on every axis; the radial scale "
     "is omitted because it differs per axis.",
     ["radar_UIEB.pdf", "radar_LSUI.pdf", "radar_EUVP.pdf",
      "radar_RUIE.pdf", "radar_UCCS.pdf"]),
    ("Fig. 3. UIEB score distributions per method, n = 134: PSNR, SSIM, "
     "MS-SSIM and LPIPS with the AlexNet backbone. Boxes are interquartile, "
     "points are per-image scores; methods are ordered best-first.",
     ["box_UIEB_psnr.pdf", "box_UIEB_ssim.pdf",
      "box_UIEB_ms_ssim.pdf", "box_UIEB_lpips_alex.pdf"]),
    ("Fig. 4. LSUI score distributions per method, n = 120. Conventions as "
     "Fig. 3.",
     ["box_LSUI_psnr.pdf", "box_LSUI_ssim.pdf",
      "box_LSUI_ms_ssim.pdf", "box_LSUI_lpips_alex.pdf"]),
    ("Fig. 5. EUVP score distributions per method, n = 120. Conventions as "
     "Fig. 3.",
     ["box_EUVP_psnr.pdf", "box_EUVP_ssim.pdf",
      "box_EUVP_ms_ssim.pdf", "box_EUVP_lpips_alex.pdf"]),
    ("Fig. 6. UIEB ablation: mean PSNR change per variant relative to the full "
     "model, n = 40. Red is a deficit, green a gain.",
     ["ablation_UIEB_psnr.pdf"]),
]


# ------------------------------------------------------------- inline markup
SEGMENT = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")


@dataclass
class Chunk:
    text: str
    bold: bool = False
    italic: bool = False
    mono: bool = False


def parse_inline(text: str) -> List[Chunk]:
    """Split text into styled chunks. Table cells pass through unstyled."""
    out: List[Chunk] = []
    for piece in SEGMENT.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            out.append(Chunk(piece[2:-2], bold=True))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(Chunk(piece[1:-1], italic=True))
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            out.append(Chunk(piece[1:-1], mono=True))
        else:
            out.append(Chunk(piece))
    return out


def words_of(chunks: Sequence[Chunk]) -> List[Tuple[str, Chunk]]:
    """Flatten styled chunks into (word, style) pairs, keeping spacing."""
    words: List[Tuple[str, Chunk]] = []
    for ch in chunks:
        parts = ch.text.split(" ")
        for i, p in enumerate(parts):
            if p:
                words.append((p, ch))
            if i < len(parts) - 1:
                words.append((" ", ch))
    return words


# ------------------------------------------------------------------- the writer
class PDFWriter:
    """A simple flowing-layout typesetter on matplotlib pages."""

    def __init__(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.pages: List[io.BytesIO] = []
        self.fig = None
        self.ax = None
        self.y = 0.0
        self.page_no = 0
        self._new_page()

    # -- page plumbing ---------------------------------------------------
    def _new_page(self) -> None:
        if self.fig is not None:
            self._stamp_page_number()
            buf = io.BytesIO()
            self.fig.savefig(buf, format="pdf")
            self.pages.append(buf)
            self.plt.close(self.fig)
        self.page_no += 1
        self.fig = self.plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=72)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(0, PAGE_H)
        self.ax.axis("off")
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        self.y = TOP

    def _stamp_page_number(self) -> None:
        self.ax.text(PAGE_W / 2, PAGE_NUM_Y, str(self.page_no),
                     fontsize=8, fontfamily=SANS, ha="center", va="center",
                     color="0.35")

    def space_left(self, need: float) -> bool:
        return self.y - need >= BOTTOM

    def ensure(self, need: float, break_header: Optional[str] = None) -> None:
        """Page-break if `need` points will not fit, optionally repeating a
        header line at the top of the new page."""
        if not self.space_left(need):
            self._new_page()
            if break_header:
                self.paragraph(break_header, size=BODY, bold=True,
                               indent=0.0, space_after=4)

    # -- measuring -------------------------------------------------------
    def width_of(self, text: str, size: float, bold: bool = False,
                 italic: bool = False, mono: bool = False) -> float:
        t = self.ax.text(0, 0, text, fontsize=size,
                         fontfamily=font_for(text, SERIF),
                         fontweight="bold" if bold else "normal",
                         fontstyle="italic" if italic else "normal")
        if mono:
            t.set_fontfamily("DejaVu Sans Mono")
        w = t.get_window_extent(self.renderer).width
        t.remove()
        return float(w)

    # -- primitives ------------------------------------------------------
    def paragraph(self, text: str, size: float = BODY, bold: bool = False,
                  italic: bool = False, indent: float = 0.0,
                  hang: float = 0.0, space_after: float = 6.0,
                  align: str = "left") -> None:
        chunks = [Chunk(text, bold=bold, italic=italic)] if (bold or italic) \
            else parse_inline(text)
        words = words_of(chunks)
        lead = size * BODY_LEAD
        maxw = TEXT_W - indent

        line: List[Tuple[str, Chunk]] = []
        lw = 0.0
        space_w = self.width_of(" ", size)

        def flush(first: bool) -> None:
            nonlocal line, lw
            if not line:
                return
            x = MARGIN + (indent if first else hang)
            for w_, ch in line:
                self.ax.text(x, self.y - size, w_, fontsize=size,
                             fontfamily=font_for(w_, SERIF),
                             fontweight="bold" if ch.bold else "normal",
                             fontstyle="italic" if ch.italic else "normal",
                             va="baseline", ha="left")
                if ch.mono:
                    # re-fetch the drawn artist to switch family
                    self.ax.texts[-1].set_fontfamily("DejaVu Sans Mono")
                x += self.width_of(w_, size, ch.bold, ch.italic, ch.mono)
            self.y -= lead
            line = []
            lw = 0.0

        first = True
        for w_, ch in words:
            ww = self.width_of(w_, size, ch.bold, ch.italic, ch.mono)
            if w_ == " ":
                if line and lw + space_w <= maxw:
                    line.append((w_, ch))
                    lw += space_w
                continue
            if lw + ww > maxw and line:
                flush(first)
                first = False
                # drop a leading space after a wrap
                line.append((w_, ch))
                lw = ww
            else:
                line.append((w_, ch))
                lw += ww
        flush(first)
        self.y -= space_after

    def heading(self, text: str, level: int) -> None:
        size, before, after = {0: (TITLE_FS, 0, 10), 1: (H1_FS, 16, 7),
                               2: (H2_FS, 12, 5)}[level]
        self.ensure(size * 2 + after, break_header=None)
        self.y -= before
        self.paragraph(text, size=size, bold=True, space_after=after,
                       indent=0.0)
        if level <= 1:
            self.ax.plot([MARGIN, MARGIN + TEXT_W], [self.y + 3, self.y + 3],
                         color="0.0", lw=0.8 if level == 1 else 1.4)
            self.y -= 4

    def rule(self) -> None:
        self.ensure(8)
        self.y -= 4
        self.ax.plot([MARGIN, MARGIN + TEXT_W], [self.y, self.y],
                     color="0.55", lw=0.5)
        self.y -= 8

    def code(self, lines: Sequence[str]) -> None:
        lead = CODE_FS * 1.25
        for ln in lines:
            self.ensure(lead)
            self.ax.text(MARGIN + 14, self.y - CODE_FS, ln, fontsize=CODE_FS,
                         fontfamily="DejaVu Sans Mono", va="baseline")
            self.y -= lead
        self.y -= 5

    def bullet(self, text: str) -> None:
        size = BODY
        lead = size * BODY_LEAD
        indent = 14.0
        # draw the marker and flow the text inside the hanging indent
        self.ensure(lead)
        self.ax.text(MARGIN + 2, self.y - size, "•", fontsize=size,
                     fontfamily=SERIF, va="baseline")
        chunks = parse_inline(text)
        words = words_of(chunks)
        space_w = self.width_of(" ", size)
        maxw = TEXT_W - indent - 12
        line: List[Tuple[str, Chunk]] = []
        lw = 0.0
        x0 = MARGIN + indent + 4

        def flush(first: bool) -> None:
            nonlocal line, lw
            if not line:
                return
            x = x0
            for w_, ch in line:
                self.ax.text(x, self.y - size, w_, fontsize=size,
                             fontfamily=font_for(w_, SERIF),
                             fontweight="bold" if ch.bold else "normal",
                             fontstyle="italic" if ch.italic else "normal",
                             va="baseline")
                if ch.mono:
                    self.ax.texts[-1].set_fontfamily("DejaVu Sans Mono")
                x += self.width_of(w_, size, ch.bold, ch.italic, ch.mono)
            self.y -= lead
            line = []
            lw = 0.0

        first = True
        for w_, ch in words:
            ww = self.width_of(w_, size, ch.bold, ch.italic, ch.mono)
            if w_ == " ":
                if line and lw + space_w <= maxw:
                    line.append((w_, ch))
                    lw += space_w
                continue
            if lw + ww > maxw and line:
                flush(first)
                first = False
                line.append((w_, ch))
                lw = ww
            else:
                line.append((w_, ch))
                lw += ww
        flush(first)
        self.y -= 2.5

    # -- tables ----------------------------------------------------------
    def table(self, rows: Sequence[Sequence[str]],
              caption: Optional[str] = None) -> None:
        self._new_page()
        if caption:
            self.paragraph(caption, size=CAPTION_FS, space_after=8)
            self.y -= 4
        if not rows:
            return

        fs = TABLE_FS
        while fs > 4.6:
            widths = [max(self.width_of(str(c), fs) for c in col)
                      for col in zip(*rows)]
            if sum(widths) <= TEXT_W:
                break
            fs -= 0.2
        widths = [max(self.width_of(str(c), fs) for c in col)
                  for col in zip(*rows)]
        scale = min(1.0, TEXT_W / max(sum(widths), 1e-6))
        widths = [w * scale for w in widths]
        lead = fs * 1.25
        pad = 2.6

        header = rows[0]
        self._draw_row(header, widths, fs, lead, pad, bold=True)
        self.y -= 1.5
        self.ax.plot([MARGIN, MARGIN + TEXT_W], [self.y + lead, self.y + lead],
                     color="0.0", lw=0.7)
        self.y -= lead + 1.5

        for r in rows[1:]:
            if not self.space_left(lead + 3 * pad):
                self._new_page()
                self._draw_row(header, widths, fs, lead, pad, bold=True)
                self.y -= 1.5
                self.ax.plot([MARGIN, MARGIN + TEXT_W],
                             [self.y + lead, self.y + lead], color="0.0",
                             lw=0.7)
                self.y -= lead + 1.5
            self._draw_row(list(r), widths, fs, lead, pad)

    def _draw_row(self, cells: Sequence[str], widths: Sequence[float],
                  fs: float, lead: float, pad: float, bold: bool = False) \
            -> None:
        x = MARGIN
        row_lines: List[Tuple[float, List[str]]] = []
        max_lines = 1
        for cell, w in zip(cells, widths):
            avail = w - 2 * pad
            words, line, lw = [], [], 0.0
            space_w = self.width_of(" ", fs)
            toks = str(cell).split(" ")
            for tok in toks:
                tw = self.width_of(tok, fs, bold=bold)
                if line and lw + space_w + tw > avail:
                    words.append(" ".join(line))
                    line, lw = [tok], tw
                else:
                    if line:
                        lw += space_w
                    line.append(tok)
                    lw += tw
            words.append(" ".join(line))
            row_lines.append((x, words))
            max_lines = max(max_lines, len(words))
            x += w
        for i in range(max_lines):
            for x, lines in row_lines:
                txt = lines[i] if i < len(lines) else ""
                t = self.ax.text(x + pad, self.y - fs, txt, fontsize=fs,
                                 fontfamily=font_for(txt, SERIF),
                                 fontweight="bold" if bold else "normal",
                                 va="baseline", ha="left")
                if txt in MISSING_GLYPH_FONTS:
                    t.set_fontfamily(MISSING_GLYPH_FONTS[txt])
            self.y -= lead
        self.y -= pad


# ------------------------------------------------------------- markdown parsing
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def clean(src: str) -> str:
    return COMMENT.sub("", src)


def parse_markdown(src: str):
    """Yield (kind, payload) tokens for a manuscript section.

    kinds: h1, h2, p, bullet, quote, code, table
    """
    src = clean(src)
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            yield ("h1", ln[2:].strip())
            i += 1
        elif ln.startswith("## "):
            yield ("h2", ln[3:].strip())
            i += 1
        elif ln.startswith("    "):
            block = []
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            yield ("code", block)
        elif ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = []
            for j, bl in enumerate(block):
                if re.match(r"^\|[\s:|-]+\|?$", bl.strip()):
                    continue                # the alignment rule
                cells = [c.strip() for c in bl.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                yield ("table", rows)
        elif ln.startswith("- "):
            while i < len(lines) and lines[i].startswith("- "):
                yield ("bullet", lines[i][2:].strip())
                i += 1
        elif ln.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:])
                i += 1
            yield ("quote", " ".join(block))
        else:
            block = [ln]
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and not lines[i].startswith(("#", "    ", "|", "- ", "> "))):
                block.append(lines[i])
                i += 1
            yield ("p", " ".join(b.strip() for b in block))


LATEX_ESCAPES = {
    r"\_": "_", r"\&": "&", r"\%": "%", r"\#": "#", r"\$": "$",
    r"\checkmark": "✓", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\rightarrow": "→", r"\Delta": "Δ", r"\delta": "δ",
    r"\rm": "", r"--": "—", r"\\": "",
}

# DejaVu Serif lacks U+2713 (✓); DejaVu Sans has it. The significance table
# uses the mark per row, so the font switch is applied cell by cell.
MISSING_GLYPH_FONTS = {"✓": "DejaVu Sans"}


def font_for(text: str, family: str) -> str:
    for ch, alt in MISSING_GLYPH_FONTS.items():
        if ch in text:
            return alt
    return family


def parse_latex_table(src: str):
    """Parse the booktabs significance table into rows of plain strings."""
    rows = []
    for raw in src.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("%"):
            continue
        if any(t in ln for t in (r"\begin", r"\end", r"\toprule", r"\midrule",
                                 r"\bottomrule", r"\hline", r"\caption",
                                 r"\label", r"\centering")):
            continue
        if "&" not in ln:
            continue
        ln = ln.rstrip(r"\\").strip()
        # Mathtext like ``$\uparrow$`` renders as arrows in the PDF, but
        # ``$\rm vgg}$`` is a font switch with no glyph: keep the text, drop
        # the math delimiters and the switch itself.
        ln = re.sub(r"\$\\rm\s+([^$]*)\$", r"\1", ln)
        ln = re.sub(r"\$([^$]*)\$", r"\1", ln)
        # ``$_{\rm vgg}$`` is a subscripted font switch; render it as text.
        ln = re.sub(r"_\{\\rm\s+([^}]*)\}", r"(\1)", ln)
        for a, b in LATEX_ESCAPES.items():
            ln = ln.replace(a, b)
        cells = [c.strip() for c in ln.split("&")]
        cells = [re.sub(r"[\{\}]", "", c) for c in cells]
        rows.append(cells)
    return rows


# ------------------------------------------------------------------- assembly
def build(output: Path) -> Path:
    w = PDFWriter()

    # Title block: the front matter is special-cased so the title is large and
    # the abstract is set narrower than body text.
    fm = clean((MS / "00_front_matter.md").read_text(encoding="utf-8"))
    parts = {}
    cur = None
    for ln in fm.split("\n"):
        if ln.startswith("## "):
            cur = ln[3:].strip()
            parts[cur] = []
        elif cur is not None:
            parts[cur].append(ln)
    join = lambda k: " ".join(l.strip() for l in parts.get(k, []) if l.strip())

    w.heading(join("Title"), 0)
    w.y -= 6
    w.paragraph(join("Authors"), size=10.5, space_after=2)
    w.paragraph(join("Abstract"), size=9.5, italic=True, indent=0,
                space_after=8)
    kw = join("Keywords")
    w.paragraph(kw, size=9.5, space_after=10)
    w.rule()

    # Highlights, as a short page right after the front matter.
    if HIGHLIGHTS_FILE.is_file():
        w.heading("Highlights", 1)
        for line in HIGHLIGHTS_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("- "):
                w.bullet(line[2:].strip())
        w.y -= 4
        w.rule()

    # Body sections in document order.
    for fname, _ in SECTIONS:
        path = MS / fname
        if not path.is_file():
            print(f"  skip {fname}: not found")
            continue
        src = path.read_text(encoding="utf-8")
        if fname.endswith(".tex"):
            rows = parse_latex_table(src)
            w.table(rows)
            continue
        for kind, payload in parse_markdown(src):
            if kind == "h1":
                # the front matter's own H1 is not repeated
                if payload.lower() == "front matter":
                    continue
                w.heading(payload, 1)
            elif kind == "h2":
                w.heading(payload, 2)
            elif kind == "p":
                w.paragraph(payload)
            elif kind == "bullet":
                w.bullet(payload)
            elif kind == "quote":
                w.paragraph(payload, italic=True, indent=18, space_after=8)
            elif kind == "code":
                w.code(payload)
            elif kind == "table":
                w.table(payload)

    # Back matter: tables then figures.
    for fname, label, caption in TABLES:
        path = TAB / fname
        if not path.is_file():
            print(f"  skip {fname}: not found")
            continue
        if path.suffix == ".tex":
            rows = parse_latex_table(path.read_text(encoding="utf-8"))
        else:
            rows = [r for r in parse_markdown(path.read_text(encoding="utf-8"))
                    if True]
            rows = [p[1] for p in rows if p[0] == "table"][0]
        w.table(rows, caption=f"{label}. {caption}")
        print(f"  table {label} ({fname}): {len(rows)} rows")

    # Flush the text pages, then splice in the vector figures with pypdf.
    text_pages = [p.getvalue() for p in w.pages]

    from pypdf import PdfReader, PdfWriter, Transformation

    writer = PdfWriter()
    for tp in text_pages:
        writer.add_page(PdfReader(io.BytesIO(tp)).pages[0])

    n_figs = 0
    for caption, files in FIGURES:
        for idx, fname in enumerate(files):
            fpath = FIG / fname
            if not fpath.is_file():
                print(f"  skip figure {fname}: not found")
                continue
            n_figs += 1
            cap_page = _caption_page(caption if idx == 0 else None)
            reader = PdfReader(str(fpath))
            page = reader.pages[0]
            mb = page.mediabox
            fw, fh = float(mb.width), float(mb.height)
            cap_h = 46.0 if idx == 0 else 14.0
            avail_w = TEXT_W
            avail_h = PAGE_H - MARGIN - (TOP if False else 0) - cap_h - MARGIN
            avail_h = PAGE_H - 2 * MARGIN - cap_h
            s = min(avail_w / fw, avail_h / fh, 1.0)
            tx = MARGIN + (TEXT_W - fw * s) / 2.0
            ty = PAGE_H - MARGIN - cap_h - fh * s
            cap_page.merge_transformed_page(
                page, Transformation().scale(s).translate(tx, ty))
            writer.add_page(cap_page)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as fh:
        writer.write(fh)
    print(f"\nwrote {output} ({len(writer.pages)} pages, {n_figs} figures)")
    return output


def _caption_page(caption: Optional[str]):
    """A blank A4 page with the caption set at the top, for merging a figure
    onto."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")
    if caption:
        ax.text(MARGIN, PAGE_H - MARGIN, caption, fontsize=CAPTION_FS,
                fontfamily=SERIF, va="top", ha="left", wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    from pypdf import PdfReader

    return PdfReader(io.BytesIO(buf.getvalue())).pages[0]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else REPO / "paper" / "06_submission" / "manuscript.pdf"
    raise SystemExit(0 if build(out) else 1)
