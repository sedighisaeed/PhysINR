#!/usr/bin/env python3
"""Build the Elsevier submission PDF from the markdown sources with elsarticle.

The manuscript is authored in markdown; the journal requires the elsarticle
class (requirement R16). This converts each section to LaTeX, embeds the
back-matter tables and the submission-quality vector figures, and runs
pdflatex.

Everything is derived from the same files ``build_pdf.py`` reads, so the two
renderings cannot drift apart in content -- only in typography.

Usage:  python3 experiments/build_latex.py [output.tex]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MS = REPO / "paper" / "02_manuscript"
TAB = REPO / "paper" / "04_tables"
FIG = REPO / "paper" / "06_submission" / "figures"
OUT_DIR = REPO / "paper" / "06_submission" / "latex"

TEXLIVE = "/usr/local/texlive/2026/bin/universal-darwin"

SECTIONS = [
    ("01_introduction.md", "Introduction"),
    ("02_related_work.md", "Related work"),
    ("03_methods.md", "Materials and methods"),
    ("results.md", "Results"),
    ("05_discussion.md", "Discussion"),
]

TABLES = [
    ("comparison_UIEB.md", "tab:uieb",
     "UIEB main comparison, n = 134 images per method. Eleven baselines and "
     "PhysINR-Zero, full-reference and no-reference metrics (mean $\\pm$ sd). "
     "Superscripts rank each column: * best, + second. WF-Diff ran an "
     "analytic fallback, not its published model; its row is marked and "
     "indicative only."),
    ("comparison_LSUI.md", "tab:lsui",
     "LSUI main comparison, n = 120 images per method. Conventions as "
     "Table~\\ref{tab:uieb}."),
    ("comparison_EUVP.md", "tab:euvp",
     "EUVP main comparison, n = 120 images per method. Conventions as "
     "Table~\\ref{tab:uieb}."),
    # Sentinel: the two no-reference sets share method list and metric columns,
    # so they are emitted side by side as one table (nr_merged_tex below).
    # The per-set .md files stay the source of truth and what the number
    # checker reads.
    ("MERGED_NR", "tab:nr",
     "The two sets without reference photographs, n = 45 UCCS and n = 100 "
     "RUIE images per method, no-reference metrics only: both sets ship no "
     "ground-truth restoration. Conventions as Table~\\ref{tab:uieb}."),
    ("revision_v4.md", "tab:revision",
     "Prior design study: four candidate transmission priors on the same "
     "frames and protocol. Clipping is the fraction of pixels within 1/255 of "
     "either rail; the green channel is the one the water-type-lookup prior "
     "saturates. Chroma and L* are CIELAB measured against the reference "
     "photograph. The adaptation loop is chaotic at the metric margin "
     "($\\pm$0.02 clipping, $\\pm$0.35 dB PSNR across repeat runs), so only "
     "larger differences are interpretable."),
    ("ablation_UIEB.md", "tab:ablation",
     "UIEB ablation, n = 40 images per method (the first 40 of the split, the "
     "same subset every variant was scored on). The full model's column is "
     "truncated to that subset so the paired tests see matched samples; its "
     "mean therefore differs from Table~\\ref{tab:uieb}."),
]

# The significance table is authored in LaTeX already; pass it through.
TEX_TABLES = [
    ("ablation_UIEB_significance.tex", "tab:ablation-sig",
     "UIEB ablation significance: paired Wilcoxon signed-rank against the full "
     "model, Holm-corrected across the 15 variants. \\checkmark marks "
     "$p < 0.05$; $r$ is the rank-biserial correlation."),
]

# (caption, label, [files]) -- one figure environment per group.
# Figures that visualise data already tabulated in full (per-dataset radar
# charts, the LSUI/EUVP distribution boxes, the meta-training convergence
# curve) live in the supplementary PDF; see SUPPLEMENTARY_FIGURES below.
FIGURES = [
    ("Overview of PhysINR-Zero. A coordinate field $\\Phi(x)$ of physical "
     "quantities -- per-channel optical thickness $\\tau_c$, a bounded "
     "nonlinear gain $\\alpha_c$ and ambient light $A_c$ -- is fitted to each "
     "image against reference-free losses, inverting a Jerlov-anchored "
     "forward model. Meta-training supplies only the initialisation; no "
     "underwater image is used in training.",
     "fig:method", ["method_diagram.pdf"]),
    ("Qualitative comparison on one UIEB test image (id 100): the input, the "
     "reference, and each method's actual output under the benchmark. Methods "
     "are ordered by their UIEB PSNR rank. The figure shows the same split "
     "between perceptual colour and reference reconstruction that the tables "
     "quantify -- PhysINR-Zero and Retinex carry the colour, the supervised "
     "networks carry the structure.",
     "fig:qualitative", ["qualitative_UIEB.pdf"]),
    ("Prior design study: raw input and the five candidate transmission "
     "priors of Table~\\ref{tab:revision} -- water-type lookup, the "
     "content-dependent head that replaced it, measured only, learned level "
     "on an uncalibrated depth range, and the calibrated model this paper "
     "ships -- on UIEB frames (left) and UCCS colour-cast frames (right). "
     "The lookup prior's posterised neon washes -- rail-pinned pixels read "
     "as flat colour regions -- are absent from the calibrated column; the "
     "content head removes them too but under-restores instead. On UCCS the "
     "lookup prior amplifies the cast it is supposed to remove, while the "
     "content head and the calibrated model reduce it below the input's own "
     "cast level without saturating any channel.",
     "fig:revision", ["revision_v4_uieb.pdf", "revision_v4_uccs.pdf"],
     "0.49\\textwidth"),
    ("UIEB score distributions per method, n = 134: PSNR, SSIM, MS-SSIM and "
     "LPIPS with the AlexNet backbone. Boxes are interquartile, points are "
     "per-image scores; methods are ordered best-first. The corresponding "
     "LSUI and EUVP distributions are Supplementary Figs.~S2 and~S3.",
     "fig:box-uieb",
     ["box_UIEB_psnr.pdf", "box_UIEB_ssim.pdf",
      "box_UIEB_ms_ssim.pdf", "box_UIEB_lpips_alex.pdf"]),
    ("UIEB ablation: mean PSNR change per variant relative to the full model, "
     "n = 40. Red is a deficit, green a gain.",
     "fig:ablation", ["ablation_UIEB_psnr.pdf"], "0.78\\textwidth"),
]

# Supplementary material: one PDF, figures referenced from the main text as
# S1..S4 in this order. Each visualises data the main text reports in tables;
# none is load-bearing for a claim.
SUPPLEMENTARY_FIGURES = [
    ("S1", "Meta-training convergence: the outer-loop meta loss across the "
     "4,000 synthetic tasks drawn from terrestrial photographs. The curve is "
     "read from the checkpoint's own provenance file, not re-fit for display.",
     "meta_train_curve.pdf"),
    ("S2", "LSUI score distributions per method, n = 120: PSNR, SSIM, MS-SSIM "
     "and LPIPS with the AlexNet backbone. Conventions as the UIEB "
     "distributions figure of the main text.",
     ["box_LSUI_psnr.pdf", "box_LSUI_ssim.pdf",
      "box_LSUI_ms_ssim.pdf", "box_LSUI_lpips_alex.pdf"]),
    ("S3", "EUVP score distributions per method, n = 120: PSNR, SSIM, MS-SSIM "
     "and LPIPS with the AlexNet backbone. Conventions as the UIEB "
     "distributions figure of the main text.",
     ["box_EUVP_psnr.pdf", "box_EUVP_ssim.pdf",
      "box_EUVP_ms_ssim.pdf", "box_EUVP_lpips_alex.pdf"]),
    ("S4", "Radar comparison across metrics on the five benchmark sets, one "
     "polygon per method. Metrics are min--max normalised across methods with "
     "direction corrected so outward is better on every axis; the radial "
     "scale is omitted because it differs per axis.",
     ["radar_UIEB.pdf", "radar_LSUI.pdf", "radar_EUVP.pdf",
      "radar_RUIE.pdf", "radar_UCCS.pdf"]),
]

COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# The sources use Unicode mathematics (Greek, minus signs, dashes, operators).
# pdflatex + T1 cannot address these directly, so each maps to a macro. The
# combining circumflex in the methods equations decomposes first: t̂ -> t + U+0302.
UNICODE_MACROS = {
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ε": r"$\epsilon$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "Δ": r"$\Delta$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "·": r"$\cdot$",
    "≈": r"$\approx$",
    "∈": r"$\in$",
    "−": r"$-$",
    "²": r"$^{2}$",
    "–": r"--",
    "—": r"---",
    "§": r"\S{}",
    "̂": "",          # combining circumflex: handled by the char before it
    "Ĵ": r"$\hat{J}$",
    "Â": r"$\hat{A}$",
}


def fold_unicode(text: str) -> str:
    """Replace Unicode mathematics with the LaTeX macro for each."""
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "̂" and out:
            # combining circumflex: convert the previous char to \hat{.}
            prev = out.pop()
            out.append(r"$\hat{" + prev + "}$")
            continue
        out.append(UNICODE_MACROS.get(ch, ch))
    return "".join(out)

# ----------------------------------------------------------------- inline text
# Physical quantities appear in the prose as ``tau_max`` or ``β_c``. Folded
# into one math run they typeset properly; left in text the underscore
# renders literally and a Greek letter switches fonts mid-word. This is a
# closed whitelist rather than a regex over ``\w_\w`` because the sources
# also contain code identifiers (mamba_ssm, ms_ssim, ablation_UIEB) that are
# structurally identical and must stay typewriter text.
_PROSE_MATH = {
    "δ_max": r"$\delta_{\max}$",
    "τ_max": r"$\tau_{\max}$",
    "β_c":   r"$\beta_c$",
    "τ_c":   r"$\tau_c$",
    "α_c":   r"$\alpha_c$",
    "t_c":   r"$t_c$",
    "J_c":   r"$J_c$",
    "I_c":   r"$I_c$",
    "r_c":   r"$r_c$",
    "A_c":   r"$A_c$",
    "K_d":   r"$K_d$",
}
_PROSE_MATH_RE = re.compile(
    r"(?<![\w$])(" + "|".join(sorted(_PROSE_MATH, key=len, reverse=True))
    + r")(?![\w])")


def _fold_prose_math(text: str) -> str:
    """Typeset the physical quantities the prose names with subscript syntax."""
    return _PROSE_MATH_RE.sub(lambda m: _PROSE_MATH[m.group(1)], text)


_MATH_RUN = re.compile(r"(\$[^$]*\$)")


_ESCAPED_BY_MARKDOWN = set("&%#_~\"")


def _escape_chars(text: str) -> str:
    """Escape LaTeX-special characters the markdown never uses as markup.

    A markdown backslash-escape (``\\_``, ``\\%``) already supplies the LaTeX
    escape the char needs, so it is passed through as-is; a backslash before a
    non-special char is markdown's "literal" escape and drops the backslash.
    Without this, ``\\_`` would render as a line break followed by a math
    subscript.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in _ESCAPED_BY_MARKDOWN:
                out.append("\\" + nxt)
            else:
                out.append(nxt)
            i += 2
            continue
        if ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == '"':
            out.append("''")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def esc(text: str, fold_math: bool = True) -> str:
    """Escape text for LaTeX, folding prose maths into math mode first."""
    if fold_math:
        text = _fold_prose_math(text)
    text = fold_unicode(text)
    # Markdown's escaped asterisk (``L\*``) is a literal star in print; \* in
    # text mode is a math-only discretionary and would break the compile.
    text = text.replace("\\*", "*")
    # ``UIEC^2-Net`` is a superscript in prose, not an escaped caret.
    text = re.sub(r"\^(\d+)", r"$^{\1}$", text)
    # Escape specials outside math runs only -- inside, LaTeX already handles
    # them and re-escaping would undo the subscripts just built.
    out = []
    pos = 0
    for m in _MATH_RUN.finditer(text):
        if m.start() > pos:
            out.append(_escape_chars(text[pos:m.start()]))
        out.append(m.group(1))
        pos = m.end()
    if pos < len(text):
        out.append(_escape_chars(text[pos:]))
    return "".join(out)


# Bare URLs are matched too: LaTeX cannot hyphenate a long literal, so an
# unconverted <https://...> overruns the right margin (55pt on the declarations
# page). \url{} is allowed to break anywhere, which is what keeps it inside the
# column. The angle brackets of a markdown autolink are consumed here so they
# do not reach the output as literal < >.
URL_BODY = r"(?:https?|ftp)://[^\s<>(){}\[\]\"']+"
# A cross-reference written as ``[Fig.](fig:box-uieb)`` resolves to
# ``Fig.~\ref{fig:box-uieb}``; the link text names the kind of float only --
# ``\ref`` supplies the number, which is what keeps the prose right when
# floats are reordered. Up to three labels may be piped:
# ``[Tables](tab:lsui|tab:euvp)`` -> ``Tables~\ref{a} and~\ref{b}``.
XREF = re.compile(
    r"\[([^\]]+)\]\((fig:[\w-]+(?:\|(?:fig|tab):[\w-]+){0,2})\)")
# A backslash-escaped asterisk (``L\*``) is markdown for a literal star, not an
# emphasis delimiter, so the lookbehind keeps it out of the italic spans.
SEGMENT = re.compile(
    r"(<" + URL_BODY + r">|" + URL_BODY
    + r"|(?<!\\)\*\*.+?(?<!\\)\*\*|(?<!\\)\*[^*]+?(?<!\\)\*|`[^`]+?`)")


def inline(text: str) -> str:
    """Convert **bold**, *italic*, `code`, bare URLs and xrefs to LaTeX."""
    # Cross-references are resolved to \ref{} here, but the LaTeX they emit
    # must survive the escaping pass below, which would mangle it (\ref ->
    # \textasciitilde{}ref). Each one is stashed behind a plain-ASCII token
    # and restored once the prose around it is escaped. A paired label
    # ("4 and 5") keeps the connective word between the two \ref calls.
    stashed: list[str] = []

    def _xref(m: re.Match) -> str:
        text_, labels = m.group(1), m.group(2).split("|")
        # The link text names the kind of float ("Fig.", "Figures", "Table")
        # and never the number: \ref supplies it, which is what keeps the
        # prose right when floats are reordered.
        if len(labels) == 1:
            resolved = text_ + r"~\ref{" + labels[0] + r"}"
        elif len(labels) == 2:
            resolved = text_ + r"~\ref{" + labels[0] + r"} and~\ref{" + labels[1] + r"}"
        else:
            resolved = (text_ + r"~\ref{" + labels[0] + "}, "
                        + r"\ref{" + labels[1] + "} and~\ref{" + labels[2] + "}")
        stashed.append(resolved)
        return "XXREFTOKEN%dXXREF" % (len(stashed) - 1)

    text = XREF.sub(_xref, text)
    out = []
    for piece in SEGMENT.split(text):
        if not piece:
            continue
        if piece.startswith(("http://", "https://", "ftp://", "<http")):
            # \url takes its argument verbatim, so it must not be escaped.
            # Strip the markdown autolink's angle brackets, if present.
            body = piece.strip("<>")
            out.append(r"\url{" + body + "}")
        elif piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            out.append(r"\textbf{" + inline_inner(piece[2:-2]) + "}")
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append(r"\textit{" + inline_inner(piece[1:-1]) + "}")
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            # Code spans keep their identifiers verbatim: no math folding, so
            # mamba_ssim typesets rather than becoming $mamba_{ssm}$.
            out.append(r"\texttt{" + inline_inner(piece[1:-1], fold_math=False) + "}")
        else:
            out.append(inline_inner(piece))
    body = "".join(out)
    for i, resolved in enumerate(stashed):
        body = body.replace("XXREFTOKEN%dXXREF" % i, resolved)
    return body


def inline_inner(text: str, fold_math: bool = True) -> str:
    """Escape text inside an already-emitted command argument."""
    return esc(text, fold_math)


# ---------------------------------------------------------------- markdown -> tex
def clean(src: str) -> str:
    # Drop HTML comments, but keep the <!-- V --> verification markers out too.
    return COMMENT.sub("", src)


# The markdown sources hardcode section numbers in the heading text
# ("## 2.1 Supervised..."). LaTeX numbers its own headings, so the prefix is
# stripped here; otherwise every subsection renders as "0.1 2.1 Supervised...".
HEADING_NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def strip_heading_num(text: str) -> str:
    return HEADING_NUM.sub("", text.strip())


def md_to_tex(src: str) -> str:
    """Convert one manuscript section to LaTeX body text."""
    src = clean(src)
    lines = src.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            # The H1 is the section title: emit \section here (numbered by
            # LaTeX) rather than skipping it, so subsections number as 2.1
            # and not 0.1.
            out.append(r"\section{" + inline(strip_heading_num(ln[2:])) + "}")
            out.append("")
            i += 1
            continue
        if ln.startswith("## "):
            out.append(r"\subsection{" + inline(strip_heading_num(ln[3:])) + "}")
            out.append("")
            i += 1
            continue
        if ln.startswith("    "):
            block = []
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            out.append(eq_block_to_tex(block))
            out.append("")
            continue
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(md_table_to_tex(block))
            out.append("")
            continue
        if ln.startswith("- "):
            out.append(r"\begin{itemize}")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(r"\item " + inline(lines[i][2:].strip()))
                i += 1
            out.append(r"\end{itemize}")
            out.append("")
            continue
        if ln.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:])
                i += 1
            quote = " ".join(b.strip() for b in block)
            out.append(r"\begin{quote}" + inline(quote) + r"\end{quote}")
            out.append("")
            continue
        # paragraph: gather until a blank or a structural line
        block = [ln]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(("#", "    ", "|", "- ", "> "))):
            block.append(lines[i])
            i += 1
        out.append(inline(" ".join(b.strip() for b in block)))
        out.append("")
    return "\n".join(out)


RULE = re.compile(r"^\|[\s:|-]+\|?$")


def _table_rows(block: list[str]):
    """Split a pipe table into (data rows, note rows).

    Note rows carry prose in the first cell and nothing else -- the fallback
    annotations the reporter appends. Typeset inside the tabular they squeeze
    into one column and overrun it, so they are returned separately and set
    below the table instead.
    """
    rows: list[list[str]] = []
    notes: list[str] = []
    seen_header = False
    for bl in block:
        if RULE.match(bl.strip()):
            continue
        cells = [c.strip() for c in bl.strip().strip("|").split("|")]
        if not any(cells):
            continue                      # padding row between body and notes
        if not seen_header:
            seen_header = True
            rows.append(cells)
        elif cells[0] and not any(cells[1:]):
            notes.append(cells[0])
        else:
            rows.append(cells)
    ncols = max(len(r) for r in rows) if rows else 0
    for r in rows:
        while len(r) < ncols:
            r.append("")
    return rows, notes


def md_table_to_tex(block: list[str]) -> str:
    """Convert a pipe table to a booktabs table with per-column alignment."""
    rows, _notes = _table_rows(block)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    body = [" & ".join(inline(c) for c in r) + r" \\" for r in rows]
    spec = "l" + "c" * (ncols - 1)
    # No \begin{center}: the table float centers with \centering, and a center
    # environment inside \resizebox's restricted-horizontal-mode box throws
    # "Not allowed in LR mode".
    return (r"\begin{tabular}{" + spec + "}"
            + "\n" + r"\toprule" + "\n"
            + "\n".join([body[0], r"\midrule"])
            + "\n" + "\n".join(body[1:]) + "\n" + r"\bottomrule" + "\n"
            + r"\end{tabular}")


def md_table_notes(block: list[str]) -> str:
    """Prose annotations that belong below the table, not inside it."""
    _rows, notes = _table_rows(block)
    if not notes:
        return ""
    out = []
    for n in notes:
        # A leading "- " reads as a bullet in markdown; keep it as a dash so
        # the note stays one line of footnotesize text.
        text = n[2:] if n.startswith("- ") else n
        out.append(inline(text))
    return "\n".join(out)


def nr_merged_tex() -> tuple[str, str]:
    """Emit the UCCS and RUIE tables side by side as one booktabs tabular.

    Both sets are unpaired (no reference photographs) with identical method
    lists and metric columns, so a two-tier header with \\multicolumn group
    titles carries both. Returns (tabular, notes); the per-set .md files in
    paper/04_tables remain the source of record and what the number checker
    validates against.
    """
    blocks = []
    notes: list[str] = []
    for fname, set_name in (("comparison_UCCS.md", "UCCS (n = 45)"),
                            ("comparison_RUIE.md", "RUIE (n = 100)")):
        rows, set_notes = _table_rows((TAB / fname).read_text(
            encoding="utf-8").split("\n"))
        blocks.append((set_name, rows))
        for n in set_notes:
            if n not in notes:
                notes.append(n)
    (name_u, rows_u), (name_r, rows_r) = blocks
    metrics = [c.strip() for c in rows_u[0][1:]]
    # The two headers must agree column-for-column or the side-by-side merge
    # would silently mislabel metric columns.
    assert [c.strip() for c in rows_r[0][1:]] == metrics, "metric columns differ"
    k = len(metrics)
    out = [r"\begin{tabular}{l" + "c" * (2 * k) + "}", r"\toprule",
           " & ".join(["Method",
                       r"\multicolumn{" + str(k) + "}{c}{" + name_u + "}",
                       r"\multicolumn{" + str(k) + "}{c}{" + name_r + "}"])
           + r" \\",
           r"\cmidrule(lr){2-" + str(1 + k) + "}"
           r"\cmidrule(lr){" + str(2 + k) + "-" + str(1 + 2 * k) + "}",
           " & ".join([""] + metrics + metrics) + r" \\", r"\midrule"]
    for ru, rr in zip(rows_u[1:], rows_r[1:]):
        assert ru[0] == rr[0], f"method rows misaligned: {ru[0]!r} vs {rr[0]!r}"
        out.append(" & ".join([ru[0]] + ru[1:] + rr[1:]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out), "\n".join(notes)


# ------------------------------------------------------------------ equations
# The manuscript's indented blocks are display equations of the form
#   lhs = rhs                (n)
# rendered as a numbered align environment. Markdown is the source of record,
# so the equation text is kept readable there and converted here.
EQNUM = re.compile(r"\((\d+)\)\s*$")


def eq_block_to_tex(block: list[str]) -> str:
    rows = []
    for raw in block:
        ln = raw.rstrip()
        m = EQNUM.search(ln)
        num = m.group(1) if m else None
        body = ln[: m.start()].strip() if m else ln
        # Fall back to a verbatim block for anything that is not an equation,
        # so prose or code is never silently typeset as math.
        if "=" not in body:
            return (r"\begin{verbatim}" + "\n"
                    + "\n".join(fold_unicode(b) for b in block) + "\n"
                    + r"\end{verbatim}")
        rows.append((body, num))
    out = [r"\begin{align}"]
    for body, num in rows:
        tag = (r"\tag{" + num + "}") if num else r"\notag"
        out.append(r"  " + fold_eq(body) + r" \\ " + tag)
    out.append(r"\end{align}")
    return "\n".join(out)


def fold_eq(text: str) -> str:
    """Turn the readable equation text into LaTeX math, leaving it in math."""
    repl = {
        "·": r"\cdot", "−": "-", "×": r"\times",
        "α": r"\alpha", "β": r"\beta", "δ": r"\delta", "ε": r"\epsilon",
        "σ": r"\sigma", "τ": r"\tau", "Δ": r"\Delta",
        "Â": r"\hat{A}", "Ĵ": r"\hat{J}",
        "exp": r"\exp", "bounded_nonlinear": r"\mathrm{bounded\_nonlinear}",
        "t̂": r"\hat{t}", "δ_max": r"\delta_{\max}",
    }
    # combining circumflex on the preceding char
    text = text.replace("̂", "")
    for a, b in repl.items():
        text = text.replace(a, b)
    # subscripts: J_c -> J_c (already valid LaTeX); Greek is handled above.
    return text


def tex_table_to_env(src: str, label: str, caption: str) -> str:
    """Wrap an already-LaTeX booktabs table body in a float."""
    body = []
    in_tabular = False
    for raw in src.splitlines():
        s = raw.strip()
        if s.startswith(r"\begin{table}") or s.startswith(r"\begin{center}"):
            continue
        if s.startswith(r"\end{table}") or s.startswith(r"\end{center}"):
            continue
        if s.startswith(r"\begin{tabular}"):
            in_tabular = True
            body.append(s)
            continue
        if s.startswith(r"\end{tabular}"):
            body.append(s)
            in_tabular = False
            continue
        if in_tabular:
            body.append(s)
        elif s.startswith(r"\caption") or s.startswith(r"\label"):
            continue
    return (r"\begin{table}[htbp]" + "\n" + r"\centering" + "\n"
            + r"\caption{" + caption + "}" + "\n"
            + r"\label{" + label + "}" + "\n"
            + r"\resizebox{\textwidth}{!}{" + "\n"
            + "\n".join(body) + "\n"
            + "}" + "\n" + r"\end{table}")


# ------------------------------------------------------------------- references
# A reference entry begins at a line that starts with a capital letter and
# runs until the next such line. The markdown source is one entry per block
# separated by blank lines, which is the reliable signal.
def _split_entries(lines: list[str]) -> list[str]:
    entries: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if ln.strip() and ln[0].isalpha() and ln[0].isupper() and cur:
            entries.append(" ".join(c.strip() for c in cur))
            cur = []
        if ln.strip():
            cur.append(ln)
    if cur:
        entries.append(" ".join(c.strip() for c in cur))
    return entries


def references_tex() -> str:
    """The reference list is author-year Harvard, set as an unnumbered list.

    elsarticle's thebibliography numbers entries, which conflicts with the
    author-year style the guide requires; the bibliography is therefore set
    directly, each entry as its own paragraph with a hanging indent.
    """
    src = clean((MS / "08_references.md").read_text(encoding="utf-8"))
    lines = [ln for ln in src.split("\n") if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("# ")]

    out = [r"\section*{References}", "",
           # The reference list is set compactly (small, near-single spaced):
           # the published elsarticle format does the same, and it keeps the
           # references section from inflating the R4 page count.
           r"\begingroup", r"\small", r"\linespread{1}\selectfont",
           r"\setlength{\parindent}{0pt}",
           r"\setlength{\parskip}{0.4ex plus 0.2ex}"]
    for e in _split_entries(lines):
        e = " ".join(e.split())
        if not e:
            continue
        out.append(r"\hangindent=1.2em \hangafter=1 " + esc(e))
    out.append(r"\endgroup")
    return "\n".join(out)


# ---------------------------------------------------------------------- figures
def figures_tex() -> str:
    out = []
    for entry in FIGURES:
        caption, label, files = entry[0], entry[1], entry[2]
        # An optional fourth element fixes the placement width, used for raster
        # figures that must not be scaled up past their natural size.
        fixed_width = entry[3] if len(entry) > 3 else None
        out.append(r"\begin{figure}[htbp]")
        out.append(r"\centering")
        n = len(files)
        for f in files:
            path = FIG / f
            if not path.is_file():
                out.append("%% MISSING FIGURE: " + f)
                continue
            width = (fixed_width if fixed_width is not None
                     else (r"\textwidth" if n == 1
                           else (r"0.48\textwidth" if n > 1 else r"\textwidth")))
            out.append(r"\includegraphics[width=" + width + "]{" + f + "}")
            if n > 1:
                out.append(r"\hfill")
        out.append(r"\caption{" + caption + "}")
        out.append(r"\label{" + label + "}")
        out.append(r"\end{figure}")
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------------ front
def frontmatter_tex() -> tuple[str, str, str, str]:
    fm = clean((MS / "00_front_matter.md").read_text(encoding="utf-8"))
    parts: dict[str, list[str]] = {}
    cur = None
    for ln in fm.split("\n"):
        if ln.startswith("## "):
            cur = ln[3:].strip()
            parts[cur] = []
        elif cur is not None:
            parts[cur].append(ln)

    def join(k: str) -> str:
        return " ".join(l.strip() for l in parts.get(k, []) if l.strip())

    title = join("Title")
    # Authors/affiliation come from the block the front matter now carries.
    authors_block = "\n".join(parts.get("Authors", []))
    abstract = join("Abstract")
    keywords = join("Keywords")
    return title, authors_block, abstract, keywords


def author_tex(block: str) -> str:
    """Turn the front-matter author block into \\author/\\address lines."""
    text = clean(block)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    author_line = ""
    aff_lines: list[str] = []
    corr = ""
    for l in lines:
        if l.startswith("**Corresponding author:**"):
            corr = l.replace("**Corresponding author:**", "").strip()
            continue
        if l.startswith("(a)"):
            aff_lines.append(l[3:].strip().lstrip(") ").strip())
            continue
        # A wrapped affiliation continues the one above it: the affiliation has
        # already started, and this line is neither a new address nor the
        # author line.
        if aff_lines and not l.startswith("("):
            aff_lines.append(l)
            continue
        if l and not author_line:
            author_line = l
    aff_line = " ".join(aff_lines)
    # Split "Name (a,*), Name (a)" into elsauthor entries.
    people = [p.strip() for p in re.split(r",(?![^(]*\))", author_line)]
    out = []
    for p in people:
        m = re.match(r"(.+?)\s*\(([^)]*)\)\s*$", p)
        name = (m.group(1).strip() if m else p)
        marks = (m.group(2) if m else "")
        corr_mark = "corr" if "*" in marks else ""
        a = "a" if "a" in marks else ""
        frag = [x for x in [a, corr_mark] if x]
        out.append(r"\author[" + ",".join(frag) + "]{" + esc(name) + "}")
    out.append(r"\address[a]{" + esc(aff_line) + "}")
    out.append(r"\cortext[corr]{Corresponding author: " + esc(corr) + "}")
    return "\n".join(out)


# ------------------------------------------------------------------------ build
PREAMBLE = r"""\documentclass[review]{elsarticle}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lineno}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{url}

\journal{Ocean Engineering}

\begin{document}

\begin{frontmatter}
"""

EPILOGUE = r"""
\end{document}
"""


def build(output: Path) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUT_DIR / "manuscript.tex"

    title, authors_block, abstract, keywords = frontmatter_tex()
    lines: list[str] = []
    lines.append(PREAMBLE)
    lines.append(r"\title{" + inline(title) + "}")
    lines.append("")
    lines.append(author_tex(authors_block))
    lines.append("")
    lines.append(r"\begin{abstract}")
    lines.append(inline(abstract))
    lines.append(r"\end{abstract}")
    lines.append("")
    lines.append(r"\begin{keyword}")
    lines.append(inline(keywords.replace(";", r" \sep ")))
    lines.append(r"\end{keyword}")
    lines.append(r"\end{frontmatter}")
    lines.append("")
    lines.append(r"\linenumbers")
    lines.append("")

    for fname, _title in SECTIONS:
        path = MS / fname
        if not path.is_file():
            print(f"  skip {fname}: not found")
            continue
        lines.append(md_to_tex(path.read_text(encoding="utf-8")))
        lines.append("")

    # Back matter: declarations, then tables, then figures, then references.
    decl = md_to_tex((MS / "07_declarations.md").read_text(encoding="utf-8"))
    lines.append(r"\section*{Declarations}")
    lines.append(decl)
    lines.append("")

    for fname, label, caption in TABLES:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + caption + "}")
        lines.append(r"\label{" + label + "}")
        if fname == "MERGED_NR":
            tabular, notes = nr_merged_tex()
            lines.append(r"\resizebox{\textwidth}{!}{")
            lines.append(tabular)
            lines.append("}")
        else:
            src = clean((TAB / fname).read_text(encoding="utf-8"))
            block = src.split("\n")
            notes = md_table_notes(block)
            lines.append(r"\resizebox{\textwidth}{!}{")
            lines.append(md_table_to_tex(block))
            lines.append("}")
        if notes:
            # Set under the table at footnote size: inside the tabular the
            # notes would be squeezed into the first column.
            lines.append(r"\vspace{2pt}")
            lines.append(r"\begin{footnotesize}")
            lines.append(notes)
            lines.append(r"\end{footnotesize}")
        lines.append(r"\end{table}")
        lines.append("")

    for fname, label, caption in TEX_TABLES:
        src = (TAB / fname).read_text(encoding="utf-8")
        lines.append(tex_table_to_env(src, label, caption))
        lines.append("")

    lines.append(figures_tex())
    lines.append(references_tex())
    lines.append(EPILOGUE)

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {tex_path}")
    return tex_path


def compile_pdf(tex_path: Path) -> Path:
    import shutil

    env_path = shutil.which("pdflatex")
    if env_path is None and Path(TEXLIVE).is_dir():
        env_path = str(Path(TEXLIVE) / "pdflatex")
    if env_path is None:
        print("pdflatex not found; wrote .tex only")
        return tex_path

    # Figures are referenced by basename; copy them next to the .tex. Both
    # the main manuscript and the supplementary PDF draw from FIG.
    for entry in FIGURES + SUPPLEMENTARY_FIGURES:
        for f in entry[2]:
            src = FIG / f
            if src.is_file():
                (tex_path.parent / f).write_bytes(src.read_bytes())

    pdf = tex_path.with_suffix(".pdf")
    # Cross-references need at least two passes; run three unconditionally so
    # \ref and page numbers settle, then report.
    last_log = ""
    for _ in range(3):
        r = subprocess.run(
            [env_path, "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True)
        log = tex_path.with_suffix(".log")
        last_log = log.read_text(errors="replace") if log.is_file() else ""
        if r.returncode != 0:
            errs = [l for l in last_log.split("\n")
                    if l.startswith("!") or "Error" in l]
            print("\n".join(errs[:25]) if errs else r.stdout[-2000:])
            break
    if pdf.is_file():
        unresolved = [l for l in last_log.split("\n") if "undefined" in l]
        if unresolved:
            print("unresolved references remain:")
            print("\n".join(unresolved[:10]))
        print(f"wrote {pdf}")
        return pdf
    print("compile failed")
    return tex_path


SUPPLEMENTARY_PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\begin{document}

\section*{Supplementary material for\\
PhysINR-Zero: zero-shot underwater image enhancement with
physics-constrained coordinate fields, evaluated under a
leakage-guarded re-benchmarking protocol}

This document contains figures that visualise data reported in full in the
main text's tables: the meta-training convergence curve, the LSUI and EUVP
score distributions, and the per-set radar charts. They are referenced from
the main text as Figs.~S1--S4.
"""


def build_supplementary(output: Path) -> Path:
    """Emit the supplementary PDF: figures S1--S4, one float each."""
    lines = [SUPPLEMENTARY_PREAMBLE]
    for label, caption, files in SUPPLEMENTARY_FIGURES:
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        n = len(files)
        for f in files:
            path = FIG / f
            if not path.is_file():
                lines.append("%% MISSING FIGURE: " + f)
                continue
            width = (r"\textwidth" if n == 1 else r"0.48\textwidth")
            lines.append(r"\includegraphics[width=" + width + "]{" + f + "}")
            if n > 1:
                lines.append(r"\hfill")
        lines.append(r"\caption*{\textbf{" + label + "} " + caption + "}")
        lines.append(r"\end{figure}")
        lines.append("")
    lines.append(r"\end{document}")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    tex = build(out) if out else build(OUT_DIR / "manuscript.tex")
    compile_pdf(tex)
    supp_tex = build_supplementary(OUT_DIR / "supplementary.tex")
    compile_pdf(supp_tex)
