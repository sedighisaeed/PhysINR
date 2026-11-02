#!/usr/bin/env python3
"""Build the IEEE JOE submission manuscript.tex from paper/02_manuscript/*.md.

Mirrors experiments/build_latex.py (the Ocean Engineering build) but targets
the IEEE Journal of Oceanic Engineering class:
  * documentclass[journal]{IEEEtran}
  * numeric citations through references.bib (IEEEtran.bst, \cite)
  * author-year prose citations in the Markdown are converted to \cite keys
    by AUTHOR2KEY (single-source-of-truth mapping).
  * abstract (no citations), \IEEEkeywords, AI disclosure in the
    Acknowledgment section.
  * equations keep their (N) tags from the Markdown; cross-references
    "(N)" in prose resolve via \eqref.

Usage:
    python3 experiments/build_latex_joe.py          # build + compile
    python3 experiments/build_latex_joe.py --no-compile
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MS = REPO / "paper" / "02_manuscript"
TAB = REPO / "paper" / "04_tables"
FIG = REPO / "paper" / "06_submission" / "figures"
OUT_DIR = REPO / "paper" / "06_submission" / "latex"
TEXLIVE = "/usr/local/texlive/2026/bin/universal-darwin"

# Reuse the table machinery of the proven Ocean Engineering build verbatim:
# byte-identical tabular markup keeps audit_numbers.py's table parser (and
# its rank checks) consistent with the manuscript that audited clean.
sys.path.insert(0, str(REPO / "experiments"))
import build_latex as _oe  # noqa: E402

_table_rows = _oe._table_rows
_md_table_to_tex_oe = _oe.md_table_to_tex
_md_table_notes = _oe.md_table_notes
_nr_merged_tex_oe = _oe.nr_merged_tex
_tex_table_to_env = _oe.tex_table_to_env
_inline_oe = _oe.inline
_esc_oe = _oe.esc

SECTIONS = [
    ("01_introduction.md", 1),
    ("02_related_work.md", 2),
    ("03_methods.md", 3),
    ("results.md", 4),
    ("05_discussion.md", 5),
]

# --------------------------------------------------------------------------
# Author-year -> \cite mapping (verified entries, see references.bib)
# --------------------------------------------------------------------------
AUTHOR2KEY = [
    ("Agustsson and Timofte, 2017", "agustsson2017ntire"),
    ("Akkaynak and Treibitz, 2019", "akkaynak2019seathru"),
    ("Austin and Petzold, 1986", "austin1986kd"),
    ("Cheng et al., 2026", "cheng2026wweuie"),
    ("Cong et al. (2024)", "cong2024survey"),
    ("Cong et al., 2024", "cong2024survey"),
    ("Drews Jr. et al., 2013", "drews2013udcp"),
    ("Finn et al., 2017", "finn2017maml"),
    ("Guan et al., 2024", "guan2024watermamba"),
    ("Guo et al., 2025", "guo2025guidedhyb"),
    ("Islam et al., 2020", "islam2020funie"),
    ("Jerlov, 1976", "jerlov1976"),
    ("Jiang et al., 2023", "jiang2023fivea"),
    ("Jiang et al., 2022", "jiang2022topal"),
    ("Jobson et al., 1997", "jobson1997msr"),
    ("Khan et al., 2025", "khan2025phaseformer"),
    ("Li and Cavallaro (2022)", "li2022limits"),
    ("Li and Cavallaro, 2022", "li2022limits"),
    ("Li et al., 2020", "li2020uieb"),
    ("McGlamery, 1980", "mcglamery1980"),
    ("Panetta et al. (2016)", "panetta2016uiqm"),
    ("Panetta et al., 2016", "panetta2016uiqm"),
    ("Panetta et al. (2022)", "panetta2022uwtbench"),
    ("Peng et al., 2023", "peng2023ushape"),
    ("Wang et al., 2004", "wang2004ssim"),
    ("Wang et al., 2003", "wang2003msssim"),
    ("Yang and Sowmya, 2015", "yang2015uciqe"),
    ("Zhang et al., 2025", "zhang2025uniuir"),
    ("Zhang et al. (2022)", "zhang2022attenuated"),
    ("Zhang et al., 2022", "zhang2022attenuated"),
    ("Zhao et al., 2024", "zhao2024wfdiff"),
    ("Zhou et al. (2025)", "zhou2025sfiformer"),
    ("Zhou et al., 2025", "zhou2025sfiformer"),
    ("Wang et al., 2021", "wang2021uiec2"),
    ("Zhang et al., 2018", "zhang2018lpips"),
    ("Zhang et al., TIP 2025", "zhang2025uniuir"),
    ("Jerlov, 1976, Table XXVII", "jerlov1976"),
    ("Li et al., 2020, §IV", "li2020uieb"),
    ("Austin and Petzold, 1986", "austin1986kd"),
]

UNICODE_MACROS = {
    "≈": r"$\approx$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "−": r"$-$",
    "–": "--",
    "—": "---",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "α": r"$\alpha$",
    "τ": r"$\tau$",
    "β": r"$\beta$",
    "δ": r"$\delta$",
    "ε": r"$\varepsilon$",
    "σ": r"$\sigma$",
    "Δ": r"$\Delta$",
    "·": r"$\cdot$",
    "²": r"$^{2}$",
    "³": r"$^{3}$",
    "⁴": r"$^{4}$",
    "§": r"\S{}",
    "Ĵ": r"$\hat{J}$",
    "Â": r"$\hat{A}$",
    "t̂": r"$\hat{t}$",
    "∈": r"$\in$",
    "→": r"$\rightarrow$",
    "←": r"$\leftarrow$",
}

_PROSE_MATH = [
    r"\bδ_max\b", r"\bτ_max\b", r"\bτ_data\b", r"\bτ_prior\b", r"\bτ_level\b",
    r"\bτ_Jerlov\b", r"\bτ_G\b", r"\bτ\b", r"\bβ_c\b", r"\bβ\b", r"\bα_c\b",
    r"\bα\b", r"\bt_data,c\b", r"\bt_data\b", r"\bt_best\b", r"\bt_min\b",
    r"\bt̂_c\b", r"\bJ_c\b", r"\bI_c\b", r"\bI'_c\b", r"\bĴ_c\b",
    r"\bA_c\b", r"\br_c\b", r"\bt_c\b", r"\bK_d\b",
    r"\bτ_true\b", r"\bτ_G\b", r"\bw_τ\b", r"\bδ\b", r"\bβ_best\b",
    r"\bexp\(−β_best · d\) = 0\.59\b", r"\bβ_best · d\b",
]

def fold_subscripts(text: str) -> str:
    """Residual symbolic tokens with subscripts appearing in prose -> math."""
    text = re.sub(r"\bt_([RGBd])\b", lambda m: "$t_{" + m.group(1) + "}$", text)
    text = re.sub(r"\bmax_c\b", r"$\\max_c$", text)
    return text


_PROSE_MATH_TEX = {
    "δ_max": r"$\delta_{\max}$", "τ_max": r"$\tau_{\max}$",
    "τ_data": r"$\tau_{\mathrm{data}}$", "τ_prior": r"$\tau_{\mathrm{prior}}$",
    "τ_level": r"$\tau_{\mathrm{level}}$", "τ_Jerlov": r"$\tau_{\mathrm{Jerlov}}$",
    "τ_G": r"$\tau_G$", "τ": r"$\tau$", "β_c": r"$\beta_c$", "β": r"$\beta$",
    "α_c": r"$\alpha_c$", "α": r"$\alpha$", "t_data,c": r"$t_{\mathrm{data},c}$",
    "t_data": r"$t_{\mathrm{data}}$", "t_best": r"$t_{\mathrm{best}}$",
    "t_min": r"$t_{\min}$", "t̂_c": r"$\hat{t}_c$", "J_c": r"$J_c$",
    "I_c": r"$I_c$", "I'_c": r"$I'_c$", "Ĵ_c": r"$\hat{J}_c$", "A_c": r"$A_c$",
    "A": r"$A$", "r_c": r"$r_c$", "t_c": r"$t_c$", "K_d": r"$K_d$",
    "τ_true": r"$\tau_{\mathrm{true}}$", "τ_G": r"$\tau_G$", "w_τ": r"$w_\tau$",
    "δ": r"$\delta$", "β_best": r"$\beta_{\mathrm{best}}$",
    "exp(−β_best · d) = 0.59": r"$\exp(-\beta_{\mathrm{best}}\cdot d)=0.59$",
    "β_best · d": r"$\beta_{\mathrm{best}}\cdot d$",
}

# narrative citations "Author (Year)" / "Author et al. (Year)" -> \cite
_NARRATIVE = [
    ("Cong et al. (2024)", "cong2024survey"),
    ("Guo et al. (2025)", "guo2025guidedhyb"),
    ("Li and Cavallaro (2022)", "li2022limits"),
    ("Yang and Sowmya (2015)", "yang2015uciqe"),
    ("Panetta et al. (2016)", "panetta2016uiqm"),
    ("Panetta et al. (2022)", "panetta2022uwtbench"),
    ("Zhang et al. (2022)", "zhang2022attenuated"),
    ("Zhou et al. (2025)", "zhou2025sfiformer"),
    ("McGlamery (1980)", "mcglamery1980"),
    ("Austin and Petzold (1986)", "austin1986kd"),
    ("Li et al. (2020)", "li2020uieb"),
]

# (compound "(Author, year; Fig.~\ref{...})" forms are handled inline in
#  md_to_tex before convert_citations runs.)


def narrative_citations(text: str) -> str:
    """Convert 'Cong et al. (2024)' style mentions to 'Cong et al.~\\cite{...}'."""
    for pat, key in _NARRATIVE:
        names = pat.split(" (")[0]
        text = text.replace(pat, names + "~\\cite{" + key + "}")
    return text

XREF_RE = re.compile(r"\[([^\]]+)\]\((tab:[^)]+|fig:[^)]+)\)")


def fold_unicode(text: str) -> str:
    for src, dst in UNICODE_MACROS.items():
        if src:
            text = text.replace(src, dst)
    # stray combining circumflex (e.g. "t\u0302" already folded above)
    text = re.sub(r"u\u0302", r"$\\hat{t}$", text)
    text = text.replace("\u0302", "")
    return text


def esc(text: str) -> str:
    """Escape TeX specials outside math runs."""
    # ``UIEC^2-Net`` is a superscript in prose, not an escaped caret.
    text = re.sub(r"\^(\d+)", r"$^{\1}$", text)
    # A backslash-escaped asterisk (``L\*``) is markdown for a literal star.
    text = text.replace("\\*", "*")
    parts = re.split(r"(\$[^$]*\$)", text)
    out = []
    for i, p in enumerate(parts):
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            out.append(p)
        else:
            # protect \texttt{...} bodies; escape _ and ^ inside them
            tt = []
            def _stash_tt(m):
                body = m.group(1).replace("_", r"\_").replace("^", r"\^{}")
                tt.append(r"\texttt{" + body + "}")
                return f"\x01{len(tt) - 1}\x01"
            p = re.sub(r"\\texttt\{([^{}]*)\}", _stash_tt, p)
            p = p.replace("\\", r"\textbackslash{}")
            for ch, rep in (
                ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                ("~", r"\textasciitilde{}"), ('"', "''"), ("<", r"$<$"),
                (">", r"$>$"), ("|", r"$|$"),
            ):
                p = p.replace(ch, rep)
            for j, t in enumerate(tt):
                p = p.replace(f"\x01{j}\x01", t)
            out.append(p)
    return "".join(out)


def inline(text: str) -> str:
    """Inline formatting: bold, italics, code, xrefs, citations.

    Bold/italic markers may span the wrapped lines of a joined paragraph,
    so asterisk runs are first normalised against the flattened text.
    """
    tokens = []

    def stash(latex: str) -> str:
        tokens.append(latex)
        return f"\x00{len(tokens) - 1}\x00"

    def xref_sub(m):
        label, refs = m.group(1), m.group(2)
        keys = [r.strip() for r in refs.split("|")]
        if all(k.startswith("tab:") for k in keys):
            macro = "Tables" if len(keys) > 1 else "Table"
        else:
            macro = "Figs." if len(keys) > 1 else "Fig."
        refs_tex = ",~".join(r"\\ref{" + k + "}" for k in keys)
        return stash(f"{label}~{refs_tex}")

    def _safe(s: str) -> str:
        # escape % only; $ runs are real math produced by prose_math and
        # must reach the output verbatim
        return s.replace("%", r"\%")

    text = XREF_RE.sub(xref_sub, text)
    # fold Unicode FIRST so −, Δ, τ … become LaTeX macros inside the stashed
    # bold/italic bodies as well
    text = fold_unicode(text)
    text = re.sub(r"\*\*([^*]+)\*\*",
                  lambda m: stash(r"\textbf{" + _safe(m.group(1)) + "}"), text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)",
                  lambda m: stash(r"\emph{" + _safe(m.group(1)) + "}"), text)
    text = re.sub(r"`([^`]+)`",
                  lambda m: stash(r"\texttt{"
                                  + m.group(1).replace("_", r"\_")
                                  + "}"), text)
    # stash every LaTeX command produced earlier in the pipeline so esc()
    # cannot mangle it (\cite, \eqref, \S{}, narrative-citation ties)
    text = re.sub(r"\\(?:cite|eqref|ref)\{[^}]*\}",
                  lambda m: stash(m.group(0)), text)
    text = re.sub(r"\\S\{\}", lambda m: stash(r"\S{}"), text)
    # esc() also turns the ~ of a narrative citation (\x00-tie) into
    # \textasciitilde{}; restore it as a non-breaking space afterwards.
    text = esc(text)
    text = text.replace(r"\textasciitilde{}", "~")
    # restore stashed tokens
    for i, tok in enumerate(tokens):
        text = text.replace(f"\x00{i}\x00", tok)
    return text


# author-year citation conversion -------------------------------------------
CITE_RE = re.compile(r"\(([^()]{3,90}?\d{4}[^()]*?)\)")


def convert_citations(text: str) -> str:
    """Replace (Author, year) patterns with \\cite commands.

    Only parentheses whose content starts with a known author-year prefix
    are converted; everything else (statistics, eqrefs) is left alone.
    """
    def repl(m):
        inner = m.group(1)
        keys = []
        rest_parts = []
        for chunk in inner.split(";"):
            chunk = chunk.strip()
            matched = False
            for pat, key in AUTHOR2KEY:
                if chunk.startswith(pat):
                    keys.append(key)
                    tail = chunk[len(pat):].strip()
                    if tail:
                        rest_parts.append(tail)
                    matched = True
                    break
            if not matched:
                return m.group(0)
        cite = "\\cite{" + ",".join(keys) + "}"
        if rest_parts:
            return cite + " " + " ".join(rest_parts)
        return cite

    return CITE_RE.sub(repl, text)


# prose tokens that may legally precede "(n)" without it being an equation
# reference: function applications such as sigmoid(0), counts, ranges
_EQREF_NONREF = re.compile(
    r"(?:sigmoid|exp|log|ln|max|min|range|tan|cos|sin)\s*$"
    r"|[.,;:] (?:the|a|an|of|in|to|and|with|at|for|from|on)\s+\S+\s*$"
    r"|\d\s*$"
)


def eqref_prose(text: str) -> str:
    """Convert references like 'in (2)' or 'identity (5)' to \\eqref.

    Guards: never convert inside math ($...$), never convert a parenthesised
    number that directly follows a function name (sigmoid(0)) or another
    digit (a range endpoint), and never convert (0) at all -- equations are
    numbered from 1.
    """
    out = []
    pos = 0
    for m in re.finditer(r"\((\d)\)", text):
        if m.start() < pos:
            continue
        if m.group(1) == "0":
            continue
        pre = text[max(0, m.start() - 24):m.start()]
        if _EQREF_NONREF.search(pre):
            continue
        # count $ since text start: odd means we are inside math
        if text.count("$", 0, m.start()) % 2 == 1:
            continue
        out.append((m.start(), m.end()))
        pos = m.end()
    if not out:
        return text
    res = text
    for s, e in reversed(out):
        res = res[:s] + "\\eqref{eq:" + res[s + 1:e - 1] + "}" + res[e:]
    return res


def prose_math(text: str) -> str:
    for pat in _PROSE_MATH:
        token = pat.replace(r"\b", "")
        text = re.sub(pat, lambda m, t=token: _PROSE_MATH_TEX.get(t, t), text)
    return text


def md_table_to_tex(rows):
    header, sep, *body = rows
    cols = len(header)
    colspec = "l" + "c" * (cols - 1)
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for r in body:
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines), cols


# --------------------------------------------------------------------------
# Tables and figures (same sources as the OE build)
# --------------------------------------------------------------------------
TABLES = [
    ("tab:uieb", "Comparison on the UIEB test split (134 images, seeded 70/15/15).",
     "comparison_UIEB.md"),
    ("tab:lsui", "Comparison on the LSUI test split (120 images).",
     "comparison_LSUI.md"),
    ("tab:euvp", "Comparison on the EUVP test split (120 images).",
     "comparison_EUVP.md"),
    ("tab:nr", "No-reference comparison on UCCS (45 images) and RUIE (100 images); these sets ship no reference images.",
     "comparison_UCCS.md", "comparison_RUIE.md"),
    ("tab:revision", "Prior design study on 20 UIEB and 12 UCCS frames at 480 px (Section 4.4).",
     "revision_v4.md"),
    ("tab:ablation", "Ablation on the first 40 UIEB test images (15 variants).",
     "ablation_UIEB.md"),
]

TEX_TABLES = [
    ("tab:ablation-sig", "Paired Wilcoxon significance panel over the 15 ablation variants (40 UIEB images; Holm--Bonferroni correction within each metric family).",
     "ablation_UIEB_significance.tex"),
]

FIGURES = [
    ("fig:method", "The PhysINR-Zero pipeline: forward model, coordinate field, meta-initialisation and per-image adaptation.",
     "method_diagram.pdf"),
    ("fig:qualitative", "Representative UIEB test frame: input, PhysINR-Zero, and the supervised networks that lead the paired metrics.",
     "qualitative_UIEB.pdf"),
    ("fig:revision", "Prior design study frames (Section 4.4): the legacy lookup (A), the measured-only prior (B), the uncalibrated (C) and calibrated (D) level heads against input and reference.",
     "revision_v4_uieb.pdf", "revision_v4_uccs.pdf"),
    ("fig:box-uieb", "UIEB score distributions (PSNR, SSIM, MS-SSIM, LPIPS-AlexNet) for all eleven methods.",
     "box_UIEB_psnr.pdf", "box_UIEB_ssim.pdf", "box_UIEB_ms_ssim.pdf", "box_UIEB_lpips_alex.pdf"),
    ("fig:ablation", "Ablation PSNR distributions on the 40-image UIEB subset.",
     "ablation_UIEB_psnr.pdf"),
]

SUPPLEMENTARY_FIGURES = [
    ("S1", "Meta-training curve: outer loss over the 4,000-task run (held-out synthetic tasks).",
     "meta_train_curve.pdf"),
    ("S2", "LSUI score distributions for all eleven methods (PSNR, SSIM, MS-SSIM, LPIPS-AlexNet).",
     "box_LSUI_psnr.pdf", "box_LSUI_ssim.pdf", "box_LSUI_ms_ssim.pdf", "box_LSUI_lpips_alex.pdf"),
    ("S3", "EUVP score distributions for all eleven methods (PSNR, SSIM, MS-SSIM, LPIPS-AlexNet).",
     "box_EUVP_psnr.pdf", "box_EUVP_ssim.pdf", "box_EUVP_ms_ssim.pdf", "box_EUVP_lpips_alex.pdf"),
    ("S4", "No-reference profiles (UCIQE, UIQM, UICM, UISM, UIConM) on UCCS and RUIE.",
     "radar_UCCS.pdf", "radar_RUIE.pdf"),
]

EQNUM_RE = re.compile(r"\s*\((\d)\)\s*$")


def eq_block_to_tex(block: str) -> str:
    """Convert an indented equation block to align environments.

    Equations that carry a manual tag '(N)' become a numbered align with
    \\label{eq:N}; untagged lines inside the same block are rendered with
    \\notag.
    """
    lines = [l.rstrip() for l in block.splitlines() if l.strip()]
    out = [r"\begin{align}"]
    for l in lines:
        m = EQNUM_RE.search(l)
        if m:
            body = EQNUM_RE.sub("", l).strip()
            out.append(fold_eq(body) + rf"\label{{eq:{m.group(1)}}} \\")
        else:
            out.append(fold_eq(l) + r" \notag \\")
    if out[-1].endswith(r" \\"):
        out[-1] = out[-1][:-2]
    out.append(r"\end{align}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Front matter, preamble, assembly
# --------------------------------------------------------------------------
def frontmatter_tex() -> str:
    fm = (MS / "00_front_matter.md").read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", fm, re.M).group(1).strip()
    authors_block = re.search(r"## Authors.*?(?=\n## |\Z)", fm, re.S).group(0)
    abstract = re.search(r"## Abstract\s*(.+?)(?=\n## |\Z)", fm, re.S).group(1).strip()
    abstract = " ".join(l.strip() for l in abstract.splitlines())
    abstract = convert_citations(prose_math(abstract))
    abstract = inline(abstract)
    kw_m = re.search(r"## Keywords\s*(.+?)(?=\n## |\Z)", fm, re.S)
    kw = kw_m.group(1).strip() if kw_m else ""
    kw = " ".join(l.strip() for l in kw.splitlines())
    kw_tex = "; ".join(inline(k.strip()) for k in kw.split(";") if k.strip())

    out = [r"\title{" + inline(title) + "}"]
    out.append(r"""
\author{Ali~Hosseini$^{*}$, and Saeed~Sedighi%
\thanks{Manuscript submitted to the IEEE Journal of Oceanic Engineering.%
The authors are with the Department of Computer Engineering, Shiraz
Branch (Shi.C.), Islamic Azad University, Shiraz, Iran.%
$^{*}$Corresponding author: Ali Hosseini
(e-mail: ali.hosseini7148@iau.ac.ir; saeed.sedighi@iau.ac.ir).}}

\markboth{IEEE Journal of Oceanic Engineering}%
{Hosseini and Sedighi: PhysINR-Zero}
""")
    out.append(r"\IEEEtitleabstractindextext{%")
    out.append(r"\begin{abstract}")
    out.append(abstract)
    out.append(r"\end{abstract}")
    out.append(r"\begin{IEEEkeywords}")
    out.append(kw_tex)
    out.append(r"\end{IEEEkeywords}}")
    return "\n".join(out)


def declarations_tex() -> str:
    """Acknowledgment section: AI disclosure + data availability."""
    return r"""
\section*{Acknowledgment}

\paragraph*{Use of artificial intelligence}
The authors disclose the use of an AI assistant (ZCode, Atria model) during
the preparation of this manuscript. The assistant was used for drafting and
editing prose, consolidating bibliographic records, and developing the
benchmark code under the direction of the authors. All scientific content,
experimental design, results and conclusions were produced and verified by
the authors, who take full responsibility for the work.

\paragraph*{Data availability}
All five benchmark datasets are publicly available: UIEB~\cite{li2020uieb},
LSUI~\cite{peng2023ushape}, EUVP~\cite{islam2020funie},
RUIE/UCCS~\cite{ruiedataset}, and the DIV2K photographic
pool~\cite{agustsson2017ntire} used for meta-training. Code, per-image
results, provenance records and the trained checkpoint are available at
\url{https://github.com/bornaseirco/PhysINR}.
"""


PREAMBLE = r"""\documentclass[journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\hyphenation{op-tical net-works semi-conduc-tor}
\begin{document}
"""


def table_envs() -> list:
    """Table floats: single-method tables via the OE machinery; the UCCS+RUIE
    pair via the proven nr_merged_tex (two-tier multicolumn header)."""
    out = []
    for spec in TABLES:
        label, caption = spec[0], spec[1]
        files = spec[2:]
        if len(files) == 2:
            inner, notes = _nr_merged_tex_oe()
            wide = True
        else:
            raw = (TAB / files[0]).read_text(encoding="utf-8")
            block = [l for l in raw.split("\n")
                     if not l.lstrip().startswith("<!--")]
            inner = _md_table_to_tex_oe(block)
            notes = _md_table_notes(block)
            ncols = max(len(r) for r in _table_rows(block)[0]) if _table_rows(block)[0] else 0
            wide = ncols >= 6
        begin = r"\begin{table*}" if wide else r"\begin{table}"
        end = r"\end{table*}" if wide else r"\end{table}"
        body = [
            begin,
            r"\caption{" + fold_unicode(_inline_oe(caption))
            + r"\label{" + label + "}}",
            r"\centering",
        ]
        if wide:
            body.append(r"\resizebox{\textwidth}{!}{" + "\n"
                        + fold_unicode(inner) + "\n}")
        else:
            body.append(fold_unicode(inner))
        if notes:
            body.append(r"\par\smallskip\footnotesize "
                        + fold_unicode(_inline_oe(notes)))
        body.append(end)
        out.append("\n".join(body))
    for label, caption, f in TEX_TABLES:
        env = _tex_table_to_env(
            (TAB / f).read_text(encoding="utf-8"), label,
            fold_unicode(_inline_oe(caption)))
        env = fold_unicode(env)
        # the significance panel is wider than one column: float it full-width
        env = env.replace(r"\begin{table}[htbp]", r"\begin{table*}[!t]")
        env = env.replace(r"\resizebox{\textwidth}{!}",
                          r"\resizebox{\textwidth}{!}")
        env = env.replace(r"\end{table}", r"\end{table*}")
        out.append(env)
    return out


def figure_envs() -> list:
    out = []
    for spec in FIGURES:
        label, caption = spec[0], spec[1]
        files = spec[2:]
        if len(files) == 1:
            out.append("\n".join([
                r"\begin{figure}[!t]",
                r"\centering",
                r"\includegraphics[width=\columnwidth]{figures/" + files[0] + "}",
                r"\caption{" + inline(caption) + r"\label{" + label + "}}",
                r"\end{figure}",
            ]))
        elif len(files) == 2:
            out.append("\n".join([
                r"\begin{figure}[!t]",
                r"\centering",
                rf"\includegraphics[width=0.49\textwidth]{{figures/{files[0]}}}",
                r"\vspace{2pt}",
                rf"\includegraphics[width=0.49\textwidth]{{figures/{files[1]}}}",
                r"\caption{" + inline(caption) + r"\label{" + label + "}}",
                r"\end{figure}",
            ]))
        else:  # 3-4 panels: 2x2 grid
            rows = []
            for j in range(0, len(files), 2):
                pair = files[j:j + 2]
                cells = [r"\includegraphics[width=0.49\columnwidth]{figures/" + f + "}" for f in pair]
                rows.append(" & ".join(cells) + r" \\")
                if len(pair) == 2 and j + 2 < len(files):
                    rows.append(r"\vspace{2pt} & \vspace{2pt} \\")
            grid = "\n".join([r"\begin{tabular}{" + "c" * min(2, len(files)) + "}",
                             "\n".join(rows), r"\end{tabular}"])
            begin = r"\begin{figure*}[!t]" if len(files) > 2 else r"\begin{figure}[!t]"
            end = r"\end{figure*}" if len(files) > 2 else r"\end{figure}"
            out.append("\n".join([
                begin,
                r"\centering",
                grid,
                r"\caption{" + inline(caption) + r"\label{" + label + "}}",
                end,
            ]))
    return out


def build() -> str:
    parts = [PREAMBLE, frontmatter_tex(), ""]
    for fname, _num in SECTIONS:
        md = (MS / fname).read_text(encoding="utf-8")
        md = clean(md)
        parts.append(md_to_tex(md))
        parts.append("")
    parts.append(declarations_tex())
    parts.append(r"\bibliographystyle{IEEEtran}")
    parts.append(r"\bibliography{references}")
    parts.append(r"\end{document}")
    # floats after the first mention pass: tables then figures at the end is
    # not IEEE style; instead emit floats near their sections.
    parts.append("")
    body = "\n".join(parts)
    table_blocks = table_envs()
    figure_blocks = figure_envs()
    return body, table_blocks, figure_blocks


def clean(md: str) -> str:
    """Strip HTML comments and the TABLE/FIGURE MAP block."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"^\|.*TABLE/FIGURE MAP.*$", "", md, flags=re.M)
    return md


def compile_pdf(name: str = "manuscript_joe.tex",
                bib: bool = True) -> None:
    pdflatex = TEXLIVE + "/pdflatex"
    bibtex = TEXLIVE + "/bibtex"
    figdir = REPO / "paper" / "06_submission" / "figures"
    figout = OUT_DIR / "figures"
    figout.mkdir(exist_ok=True)
    for f in figdir.glob("*.pdf"):
        shutil.copy2(f, figout / f.name)
    for cls in ("IEEEtran.cls", "IEEEtran.bst"):
        src = Path("/usr/local/texlive/2026/texmf-dist/tex/latex/IEEEtran")
        hits = list(src.rglob(cls))
        if hits:
            shutil.copy2(hits[0], OUT_DIR / cls)
    for _ in range(2):
        subprocess.run([pdflatex, "-interaction=nonstopmode", name],
                       cwd=OUT_DIR, capture_output=True)
    if bib:
        subprocess.run([bibtex, name[:-4]], cwd=OUT_DIR, capture_output=True)
    for _ in range(2):
        subprocess.run([pdflatex, "-interaction=nonstopmode", name],
                       cwd=OUT_DIR, capture_output=True)


def main():
    body, tables, figures = build()
    # insert tables+figures just before the Acknowledgment (IEEE allows floats
    # after references; for submission place them after section 5 text).
    anchor = r"\section*{Acknowledgment}"
    floats = "\n\n".join(tables + figures)
    if anchor in body:
        body = body.replace(anchor, floats + "\n\n" + anchor)
    else:
        body = body.replace(r"\end{document}", floats + "\n\\end{document}")
    (OUT_DIR / "manuscript_joe.tex").write_text(body, encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'manuscript_joe.tex'} "
          f"({len(body.splitlines())} lines, {len(tables)} tables, {len(figures)} figures)")
    if "--no-compile" not in sys.argv:
        compile_pdf()
        pdf = OUT_DIR / "manuscript_joe.pdf"
        if pdf.exists():
            print(f"Compiled {pdf} ({pdf.stat().st_size} bytes)")
        else:
            print("Compilation FAILED — inspect manuscript_joe.log")


def fold_eq(line: str) -> str:
    repl = {
        "α": r"\alpha ", "τ": r"\tau ", "β": r"\beta ", "δ": r"\delta ",
        "ε": r"\varepsilon ", "σ": r"\sigma ", "Δ": r"\Delta ",
        "·": r"\cdot ", "×": r"\times ", "−": "-", "≈": r"\approx ",
        "≤": r"\le ", "≥": r"\ge ",
        "bounded_nonlinear": r"\mathrm{bounded\_nonlinear}",
        "range": r"\mathrm{range}",
        "exp": r"\exp",
        "ln": r"\ln",
        "max": r"\max",
        "t̂": r"\hat{t}", "Ĵ": r"\hat{J}", "Â": r"\hat{A}",
        "_max": r"_{\max}",
        "τ_data": r"\tau_{\mathrm{data}}", "τ_prior": r"\tau_{\mathrm{prior}}",
        "τ_level": r"\tau_{\mathrm{level}}", "τ_Jerlov": r"\tau_{\mathrm{Jerlov}}",
        "t_data,c": r"t_{\mathrm{data},c}", "t_data": r"t_{\mathrm{data}}",
        "t_best": r"t_{\mathrm{best}}",
    }
    for src, dst in repl.items():
        line = line.replace(src, dst)
    line = re.sub(r"_(\w+)", r"_{\1}", line)
    line = re.sub(r"\|", r"\|", line)
    return line.strip()


def md_to_tex(md: str) -> str:
    out = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("<!--"):
            i += 1
            continue
        # heading
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title)
            title = inline(title)
            if level == 1:
                out.append(r"\section{" + title + "}")
            elif level == 2:
                out.append(r"\subsection{" + title + "}")
            else:
                out.append(r"\subsubsection{" + title + "}")
            i += 1
            continue
        # indented equation block
        if line.startswith("    ") and not line.startswith("     http"):
            block = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                if lines[i].startswith("    "):
                    block.append(lines[i][4:])
                i += 1
                if block and not lines[i - 1].strip():
                    break
            out.append(eq_block_to_tex("\n".join(b for b in block if b.strip())))
            continue
        # table
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body_rows = [r for r in rows
                         if not re.match(r"^:?-+:?$", r[0].replace(" ", ""))
                         and not r[0].startswith("<!--")]
            ncols = len(body_rows[0]) if body_rows else 0
            if ncols == 1:
                # a one-column "table" is prose the reporter left pipe-framed;
                # typeset as a paragraph, not a tabular
                for r in body_rows:
                    out.append(inline(" ".join(r)))
                continue
            tex, ncols = md_table_to_tex(body_rows)
            if ncols >= 6:
                out.append(r"\begin{table*}[!t]" + "\n" + r"\centering"
                           + "\n" + r"\resizebox{\textwidth}{!}{" + "\n"
                           + tex + "\n}" + "\n" + r"\end{table*}")
            else:
                out.append(tex)
            continue
        # itemize
        if re.match(r"^- ", line):
            out.append(r"\begin{itemize}")
            while i < len(lines) and re.match(r"^- ", lines[i]):
                item = lines[i][2:].strip()
                cont = []
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^- ", lines[i]):
                    cont.append(lines[i].strip())
                    i += 1
                text = " ".join([item] + cont)
                text = prose_math(text)
                text = fold_subscripts(text)
                text = narrative_citations(text)
                text = convert_citations(text)
                text = eqref_prose(text)
                out.append(r"\item " + inline(text))
            out.append(r"\end{itemize}")
            continue
        # blockquote
        if line.startswith("> "):
            out.append(r"\begin{quote}" + inline(line[2:].strip()) + r"\end{quote}")
            i += 1
            continue
        # paragraph
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\s{4}|\||- |> )", lines[i]) and not lines[i].lstrip().startswith("<!--"):
            para.append(lines[i])
            i += 1
        text = " ".join(para)
        text = prose_math(text)
        text = fold_subscripts(text)
        # compound "(Author, year; Fig.~\ref{...})" — cite + keep the ref part
        text = text.replace(
            "(McGlamery, 1980; ",
            "(\\cite{mcglamery1980}; ",
        )
        text = narrative_citations(text)
        text = convert_citations(text)
        text = eqref_prose(text)
        text = inline(text)
        out.append(text)
        out.append("")
    return "\n".join(out)


SUPPLEMENTARY_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\graphicspath{{../figures/}}
\begin{document}

\section*{Supplementary material for\\
PhysINR-Zero: zero-shot underwater image enhancement with
physics-constrained coordinate fields, evaluated under a
leakage-guarded re-benchmarking protocol}

This document contains figures that visualise data reported in full in the
main text's tables: the meta-training convergence curve, the LSUI and EUVP
score distributions, and the per-set radar charts. They are referenced from
the main text as Figs.~S1--S4. All plots are vector PDF; raster TIFF
siblings ship in the source package.
"""


def build_supplementary(output: Path) -> Path:
    """Emit the supplementary PDF: figures S1--S4, one float each.

    Graphics resolve through \\graphicspath{{../figures/}} so the file
    compiles in place and ships as a standalone PDF.
    """
    lines = [SUPPLEMENTARY_PREAMBLE]
    for entry in SUPPLEMENTARY_FIGURES:
        label, caption, files = entry[0], entry[1], entry[2:]
        missing = [f for f in files if not (FIG / f).is_file()]
        if missing:
            print("  supplementary: MISSING " + ", ".join(missing))
            continue
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        n = len(files)
        for f in files:
            width = (r"\textwidth" if n == 1 else r"0.48\textwidth")
            lines.append(r"\includegraphics[width=" + width + "]{" + f + "}")
            if n > 1:
                lines.append(r"\hfill")
        lines.append(r"\caption*{\textbf{" + label + "} "
                     + fold_unicode(_inline_oe(caption)) + "}")
        lines.append(r"\end{figure}")
        lines.append("")
    lines.append(r"\end{document}")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    main()
    supp = build_supplementary(OUT_DIR / "supplementary_joe.tex")
    compile_pdf(name=supp.name, bib=False)
